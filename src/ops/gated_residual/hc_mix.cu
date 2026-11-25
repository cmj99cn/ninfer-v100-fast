#include "ops/gated_residual/gated_residual.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include "ops/gated_residual/act.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;
constexpr int kGemvWarpsPerBlock = 8;

template <class A>
__global__ void hc_mix_silu_kernel(A* values, std::int32_t streams, std::int64_t count) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (index >= count) { return; }
    const float value = HcAct<A>::to_float(values[index]);
    values[index]     = HcAct<A>::from_float(silu(value / static_cast<float>(streams)));
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& value, float* out) {
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&value);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 pair = __bfloat1622float2(pairs[i]);
        out[2 * i]       = pair.x;
        out[2 * i + 1]   = pair.y;
    }
}

template <class A>
__device__ __forceinline__ void unpack_act_x8(const uint4& value, float* out) {
    if constexpr (std::is_same_v<A, __half>) {
        const __half2* pairs = reinterpret_cast<const __half2*>(&value);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 pair = __half22float2(pairs[i]);
            out[2 * i]        = pair.x;
            out[2 * i + 1]    = pair.y;
        }
    } else {
        unpack_bf16x8(value, out);
    }
}

// Warp-per-row GEMV for small depths (e.g. the up projection, depth <= 2048): one warp per
// row, lane-strided uint4 loads, fully unrolled inner loop.
template <class A>
__global__ __launch_bounds__(kGemvWarpsPerBlock * kWarpSize) void hc_project_gemv_warp_row_kernel(
    const A* weight, const A* activation, A* output, std::int32_t rows, std::int32_t depth) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row  = static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + warp;
    if (row >= rows) { return; }
    const int vectors    = depth / 8;
    const auto* weights  = reinterpret_cast<const uint4*>(
        __builtin_assume_aligned(weight + static_cast<std::int64_t>(row) * depth, 16));
    const auto* values   = reinterpret_cast<const uint4*>(
        __builtin_assume_aligned(activation, 16));
    float sum            = 0.0F;
#pragma unroll 2
    for (int idx = lane; idx < vectors; idx += kWarpSize) {
        const uint4 wv = weights[idx];
        const uint4 av = values[idx];
        float w[8];
        float a[8];
        unpack_act_x8<A>(wv, w);
        unpack_act_x8<A>(av, a);
#pragma unroll
        for (int i = 0; i < 8; ++i) { sum += w[i] * a[i]; }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) { output[row] = HcAct<A>::from_float(sum); }
}

// Fused up GEMV + sigmoid gating for decode (columns == 1, streams == 4): one CTA covers two
// d positions; warps 0-3 compute the four stream rows for d0 and warps 4-7 for d0+1, each with
// the warp-per-row lane-strided association (bit-identical to hc_project_gemv_warp_row_kernel),
// then a per-half sigmoid combine produces block_input[d]. No cross-CTA communication.
template <class A>
__global__ __launch_bounds__(kGemvWarpsPerBlock * kWarpSize) void hc_mix_up_gate_kernel(
    const A* __restrict__ weight, const A* __restrict__ low, const A* __restrict__ normalized,
    A* __restrict__ block_input, std::int32_t hidden, std::int32_t rank, std::int32_t streams,
    std::int32_t depth) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int half = warp >> 2;
    const int s    = warp & 3;
    const int d    = static_cast<int>(blockIdx.x) * 2 + half;
    __shared__ float partials[2][4];
    __shared__ A gate[2][4];
    const int vectors = depth / 8;
    const A* weight_row = weight + static_cast<std::int64_t>(s) * hidden * depth +
                          static_cast<std::int64_t>(d) * depth;
    const auto* weights = reinterpret_cast<const uint4*>(
        __builtin_assume_aligned(weight_row, 16));
    const auto* values  = reinterpret_cast<const uint4*>(
        __builtin_assume_aligned(low, 16));
    float sum           = 0.0F;
#pragma unroll 2
    for (int idx = lane; idx < vectors; idx += kWarpSize) {
        const uint4 wv = weights[idx];
        const uint4 av = values[idx];
        float w[8];
        float a[8];
        unpack_act_x8<A>(wv, w);
        unpack_act_x8<A>(av, a);
#pragma unroll
        for (int i = 0; i < 8; ++i) { sum += w[i] * a[i]; }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) { partials[half][s] = sum; }
    __syncthreads();
    if (s == 0 && lane == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { gate[half][i] = HcAct<A>::from_float(partials[half][i]); }
    }
    __syncthreads();
    if (threadIdx.x == 0 || threadIdx.x == 128) {
        const int dd = d;
        float total = 0.0F;
#pragma unroll
        for (std::int32_t stream = 0; stream < streams; ++stream) {
            const std::int64_t index = static_cast<std::int64_t>(stream) * hidden + dd;
            total += sigmoid(HcAct<A>::to_float(gate[half][stream])) *
                     HcAct<A>::to_float(normalized[index]);
        }
        block_input[dd] = HcAct<A>::from_float(total / static_cast<float>(streams));
    }
}

// Multi-column counterpart of hc_mix_up_gate_kernel. Each warp still owns one (stream, d) weight
// row and keeps the same lane-strided dot product, warp reduction and gate rounding as the T=1
// path; the weight row is fetched once from global memory and reused for every token column.
template <class A, int Columns>
__global__ __launch_bounds__(kGemvWarpsPerBlock * kWarpSize) void hc_mix_up_gate_multi_kernel(
    const A* __restrict__ weight, const A* __restrict__ low, const A* __restrict__ normalized,
    A* __restrict__ block_input, std::int32_t hidden, std::int32_t rank, std::int32_t streams,
    std::int32_t depth) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int half = warp >> 2;
    const int s    = warp & 3;
    const int d    = static_cast<int>(blockIdx.x) * 2 + half;
    __shared__ float partials[2][4][Columns];
    __shared__ A gate[2][4][Columns];
    const int vectors = depth / 8;
    const A* weight_row = weight + static_cast<std::int64_t>(s) * hidden * depth +
                          static_cast<std::int64_t>(d) * depth;
    const auto* weights = reinterpret_cast<const uint4*>(
        __builtin_assume_aligned(weight_row, 16));
    float sums[Columns];
#pragma unroll
    for (int c = 0; c < Columns; ++c) { sums[c] = 0.0F; }
    for (int idx = lane; idx < vectors; idx += kWarpSize) {
        const uint4 wv = weights[idx];
        float w[8];
        unpack_act_x8<A>(wv, w);
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            const auto* column_values = reinterpret_cast<const uint4*>(
                __builtin_assume_aligned(low + static_cast<std::int64_t>(c) * depth, 16));
            const uint4 av = column_values[idx];
            float a[8];
            unpack_act_x8<A>(av, a);
#pragma unroll
            for (int i = 0; i < 8; ++i) { sums[c] += w[i] * a[i]; }
        }
    }
#pragma unroll
    for (int c = 0; c < Columns; ++c) {
        sums[c] = warp_reduce_sum(sums[c]);
        if (lane == 0) { partials[half][s][c] = sums[c]; }
    }
    __syncthreads();
    if (s == 0 && lane == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int c = 0; c < Columns; ++c) {
                gate[half][i][c] = HcAct<A>::from_float(partials[half][i][c]);
            }
        }
    }
    __syncthreads();
    if ((warp == 0 || warp == 4) && lane < Columns) {
        const int dd = d;
        float total = 0.0F;
#pragma unroll
        for (std::int32_t stream = 0; stream < streams; ++stream) {
            const std::int64_t index = static_cast<std::int64_t>(lane) * hidden * streams +
                                       static_cast<std::int64_t>(stream) * hidden + dd;
            total += sigmoid(HcAct<A>::to_float(gate[half][stream][lane])) *
                     HcAct<A>::to_float(normalized[index]);
        }
        block_input[static_cast<std::int64_t>(lane) * hidden + dd] =
            HcAct<A>::from_float(total / static_cast<float>(streams));
    }
}

template <class A, bool Vectorized, int Rows>
__global__ __launch_bounds__(kGemvWarpsPerBlock * kWarpSize) void hc_project_gemv_kernel(
    const A* weight, const A* activation, A* output, std::int32_t rows, std::int32_t depth) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    __shared__ float partials[kGemvWarpsPerBlock];
    for (int local = 0; local < Rows; ++local) {
        const int row = static_cast<int>(blockIdx.x) * Rows + local;
        if (row >= rows) { break; }
        const A* weight_row = weight + static_cast<std::int64_t>(row) * depth;
        float sum                       = 0.0F;
        if constexpr (Vectorized) {
            constexpr int kVector = 8;
            const int vectors     = depth / kVector;
            const auto* weights   = reinterpret_cast<const uint4*>(
                __builtin_assume_aligned(weight_row, 16));
            const auto* values    = reinterpret_cast<const uint4*>(
                __builtin_assume_aligned(activation, 16));
            for (int base = warp * kWarpSize; base < vectors;
                 base += kGemvWarpsPerBlock * kWarpSize) {
                const int vector_index = base + lane;
                if (vector_index >= vectors) { break; }
                const uint4 wv = weights[vector_index];
                const uint4 av = values[vector_index];
                float w[kVector];
                float a[kVector];
                unpack_act_x8<A>(wv, w);
                unpack_act_x8<A>(av, a);
#pragma unroll
                for (int i = 0; i < kVector; ++i) { sum += w[i] * a[i]; }
            }
        } else {
            for (std::int32_t index = warp * kWarpSize + lane; index < depth;
                 index += kGemvWarpsPerBlock * kWarpSize) {
                sum += HcAct<A>::to_float(weight_row[index]) *
                       HcAct<A>::to_float(activation[index]);
            }
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) { partials[warp] = sum; }
        __syncthreads();
        if (warp == 0) {
            float total = 0.0F;
#pragma unroll
            for (int i = 0; i < kGemvWarpsPerBlock; ++i) { total += partials[i]; }
            if (lane == 0) { output[row] = HcAct<A>::from_float(total); }
        }
        __syncthreads();
    }
}

template <class A, bool Vectorized>
__global__ __launch_bounds__(kGemvWarpsPerBlock * kWarpSize) void hc_mix_down_kernel(
    const A* __restrict__ weight, const A* __restrict__ activation,
    A* __restrict__ output, const A* __restrict__ injection_rows,
    float* __restrict__ injection, std::int32_t rows, std::int32_t depth,
    std::int32_t streams) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (blockIdx.x < static_cast<unsigned>(rows)) {
        const int row = static_cast<int>(blockIdx.x);
        const A* weight_row = weight + static_cast<std::int64_t>(row) * depth;
        float sum           = 0.0F;
        if constexpr (Vectorized) {
            constexpr int kVector = 8;
            const int vectors     = depth / kVector;
            const auto* weights   = reinterpret_cast<const uint4*>(
                __builtin_assume_aligned(weight_row, 16));
            const auto* values    = reinterpret_cast<const uint4*>(
                __builtin_assume_aligned(activation, 16));
            for (int base = warp * kWarpSize; base < vectors;
                 base += kGemvWarpsPerBlock * kWarpSize) {
                const int vector_index = base + lane;
                if (vector_index >= vectors) { break; }
                const uint4 wv = weights[vector_index];
                const uint4 av = values[vector_index];
                float w[kVector];
                float a[kVector];
                unpack_act_x8<A>(wv, w);
                unpack_act_x8<A>(av, a);
#pragma unroll
                for (int i = 0; i < kVector; ++i) { sum += w[i] * a[i]; }
            }
        } else {
            for (std::int32_t index = warp * kWarpSize + lane; index < depth;
                 index += kGemvWarpsPerBlock * kWarpSize) {
                sum += HcAct<A>::to_float(weight_row[index]) *
                       HcAct<A>::to_float(activation[index]);
            }
        }
        sum = warp_reduce_sum(sum);
        __shared__ float partials[kGemvWarpsPerBlock];
        if (lane == 0) { partials[warp] = sum; }
        __syncthreads();
        if (warp == 0) {
            float total = 0.0F;
#pragma unroll
            for (int i = 0; i < kGemvWarpsPerBlock; ++i) { total += partials[i]; }
            if (lane == 0) {
                const A rounded = HcAct<A>::from_float(total);
                output[row] =
                    HcAct<A>::from_float(silu(HcAct<A>::to_float(rounded) /
                                              static_cast<float>(streams)));
            }
        }
        return;
    }
    if (injection_rows == nullptr) { return; }
    const std::int32_t stream_index = static_cast<std::int32_t>(blockIdx.x) - rows;
    if (stream_index >= streams) { return; }
    const A* row = injection_rows + static_cast<std::int64_t>(stream_index) * depth;
    float sum = 0.0F;
    for (std::int32_t c = static_cast<std::int32_t>(threadIdx.x); c < depth; c += kBlock) {
        sum += HcAct<A>::to_float(row[c]) * HcAct<A>::to_float(activation[c]);
    }
    __shared__ float scratch[kBlock / kWarpSize];
    const float total = block_reduce_sum<kBlock>(sum, scratch);
    if (threadIdx.x == 0) {
        injection[stream_index] = 2.0F * sigmoid(total / static_cast<float>(streams));
    }
}

// Multi-column counterpart of hc_mix_down_kernel. A CTA covering a rank row fetches each weight
// vector exactly once and applies it to all token columns, preserving the T=1 lane-strided
// multiply-add order, warp reduction, warp-partial sum order and explicit from_float rounding.
template <class A, bool Vectorized, int Columns>
__global__ __launch_bounds__(kGemvWarpsPerBlock * kWarpSize) void hc_mix_down_silu_multi_kernel(
    const A* __restrict__ weight, const A* __restrict__ activation,
    A* __restrict__ output, const A* __restrict__ injection_rows,
    float* __restrict__ injection, std::int32_t rows, std::int32_t depth,
    std::int32_t streams) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    __shared__ float shared[kGemvWarpsPerBlock][Columns];

    if (blockIdx.x < static_cast<unsigned>(rows)) {
        const int row = static_cast<int>(blockIdx.x);
        const A* weight_row = weight + static_cast<std::int64_t>(row) * depth;
        float sums[Columns];
#pragma unroll
        for (int c = 0; c < Columns; ++c) { sums[c] = 0.0F; }
        if constexpr (Vectorized) {
            constexpr int kVector = 8;
            const int vectors     = depth / kVector;
            const auto* weights   = reinterpret_cast<const uint4*>(
                __builtin_assume_aligned(weight_row, 16));
            for (int base = warp * kWarpSize; base < vectors;
                 base += kGemvWarpsPerBlock * kWarpSize) {
                const int vector_index = base + lane;
                if (vector_index >= vectors) { break; }
                const uint4 wv = weights[vector_index];
                float w[kVector];
                unpack_act_x8<A>(wv, w);
#pragma unroll
                for (int c = 0; c < Columns; ++c) {
                    const auto* column_values = reinterpret_cast<const uint4*>(
                        __builtin_assume_aligned(activation + static_cast<std::int64_t>(c) * depth,
                                                 16));
                    const uint4 av = column_values[vector_index];
                    float a[kVector];
                    unpack_act_x8<A>(av, a);
#pragma unroll
                    for (int i = 0; i < kVector; ++i) { sums[c] += w[i] * a[i]; }
                }
            }
        } else {
            for (std::int32_t index = warp * kWarpSize + lane; index < depth;
                 index += kGemvWarpsPerBlock * kWarpSize) {
                const float w = HcAct<A>::to_float(weight_row[index]);
#pragma unroll
                for (int c = 0; c < Columns; ++c) {
                    sums[c] += w * HcAct<A>::to_float(
                                   activation[static_cast<std::int64_t>(c) * depth + index]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            sums[c] = warp_reduce_sum(sums[c]);
            if (lane == 0) { shared[warp][c] = sums[c]; }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int c = 0; c < Columns; ++c) {
                float total = 0.0F;
#pragma unroll
                for (int i = 0; i < kGemvWarpsPerBlock; ++i) { total += shared[i][c]; }
                if (lane == 0) {
                    const A rounded = HcAct<A>::from_float(total);
                    output[static_cast<std::int64_t>(c) * rows + row] =
                        HcAct<A>::from_float(silu(HcAct<A>::to_float(rounded) /
                                                  static_cast<float>(streams)));
                }
            }
        }
        return;
    }
    if (injection_rows == nullptr) { return; }
    const std::int32_t stream_index = static_cast<std::int32_t>(blockIdx.x) - rows;
    if (stream_index >= streams) { return; }
    const A* row = injection_rows + static_cast<std::int64_t>(stream_index) * depth;
    float sums[Columns];
#pragma unroll
    for (int c = 0; c < Columns; ++c) { sums[c] = 0.0F; }
    for (std::int32_t c = static_cast<std::int32_t>(threadIdx.x); c < depth; c += kBlock) {
        const float w = HcAct<A>::to_float(row[c]);
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            sums[col] += w * HcAct<A>::to_float(
                             activation[static_cast<std::int64_t>(col) * depth + c]);
        }
    }
#pragma unroll
    for (int col = 0; col < Columns; ++col) {
        sums[col] = warp_reduce_sum(sums[col]);
        if (lane == 0) { shared[warp][col] = sums[col]; }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            const float x = lane < kGemvWarpsPerBlock ? shared[lane][col] : 0.0F;
            const float total = warp_reduce_sum<kGemvWarpsPerBlock>(x);
            if (lane == 0) {
                injection[static_cast<std::int64_t>(col) * streams + stream_index] =
                    2.0F * sigmoid(total / static_cast<float>(streams));
            }
        }
    }
}

template <class A>
__global__ __launch_bounds__(kBlock) void hc_mix_gate_kernel(
    const A* gates, const A* normalized, A* block_input, std::int32_t hidden,
    std::int32_t streams) {
    const std::int32_t m = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t d = static_cast<std::int32_t>(blockIdx.y) * kBlock +
                           static_cast<std::int32_t>(threadIdx.x);
    if (d >= hidden) { return; }

    float total = 0.0F;
    for (std::int32_t s = 0; s < streams; ++s) {
        const std::int64_t index = static_cast<std::int64_t>(s) * hidden + d +
                                   static_cast<std::int64_t>(m) * hidden * streams;
        total += sigmoid(HcAct<A>::to_float(gates[index])) * HcAct<A>::to_float(normalized[index]);
    }
    block_input[static_cast<std::int64_t>(m) * hidden + d] =
        HcAct<A>::from_float(total / static_cast<float>(streams));
}

template <class W, class A>
__global__ __launch_bounds__(kBlock) void hc_mix_injection_kernel(
    const W* injection_rows, const A* flattened, float* injection, std::int32_t channels,
    std::int32_t streams) {
    const std::int32_t m        = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t s        = static_cast<std::int32_t>(blockIdx.y);
    const W* row     = injection_rows + static_cast<std::int64_t>(s) * channels;
    const A* column  = flattened + static_cast<std::int64_t>(m) * channels;

    float sum = 0.0F;
    for (std::int32_t c = static_cast<std::int32_t>(threadIdx.x); c < channels; c += kBlock) {
        sum += HcAct<W>::to_float(row[c]) * HcAct<A>::to_float(column[c]);
    }
    __shared__ float scratch[kBlock / kWarpSize];
    const float total = block_reduce_sum<kBlock>(sum, scratch);
    if (threadIdx.x == 0) {
        injection[static_cast<std::int64_t>(m) * streams + s] =
            2.0F * sigmoid(total / static_cast<float>(streams));
    }
}

template <class A, int Columns>
void launch_up_gate_multi(const void* weight, const void* low, const void* normalized,
                          void* block_input, std::int32_t hidden, std::int32_t rank,
                          std::int32_t streams, cudaStream_t stream) {
    const unsigned grid = static_cast<unsigned>(hidden / 2);
    hc_mix_up_gate_multi_kernel<A, Columns>
        <<<grid, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
            static_cast<const A*>(weight), static_cast<const A*>(low),
            static_cast<const A*>(normalized), static_cast<A*>(block_input), hidden, rank, streams,
            rank);
}

template <int Columns>
void dispatch_up_gate_multi(const void* weight, const void* low, const void* normalized,
                            void* block_input, std::int32_t hidden, std::int32_t rank,
                            std::int32_t streams, DType dtype, cudaStream_t stream) {
    if (dtype == DType::FP16) {
        launch_up_gate_multi<__half, Columns>(weight, low, normalized, block_input, hidden, rank,
                                              streams, stream);
    } else {
        launch_up_gate_multi<__nv_bfloat16, Columns>(weight, low, normalized, block_input, hidden,
                                                     rank, streams, stream);
    }
}

template <class A, bool Vectorized, int Columns>
void launch_down_multi(const void* weight, const void* activation, void* output,
                       const void* injection_rows, float* injection, std::int32_t rows,
                       std::int32_t depth, std::int32_t streams, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(rows) + (injection_rows != nullptr ? streams : 0);
    hc_mix_down_silu_multi_kernel<A, Vectorized, Columns>
        <<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
            static_cast<const A*>(weight), static_cast<const A*>(activation),
            static_cast<A*>(output), static_cast<const A*>(injection_rows), injection, rows, depth,
            streams);
}

template <int Columns>
void dispatch_down_multi(const void* weight, const void* activation, void* output,
                         const void* injection_rows, float* injection, std::int32_t rows,
                         std::int32_t depth, std::int32_t streams, DType weight_dtype,
                         bool vectorizable, cudaStream_t stream) {
    if (weight_dtype == DType::FP16) {
        if (vectorizable) {
            launch_down_multi<__half, true, Columns>(weight, activation, output, injection_rows,
                                                     injection, rows, depth, streams, stream);
        } else {
            launch_down_multi<__half, false, Columns>(weight, activation, output, injection_rows,
                                                      injection, rows, depth, streams, stream);
        }
    } else {
        if (vectorizable) {
            launch_down_multi<__nv_bfloat16, true, Columns>(
                weight, activation, output, injection_rows, injection, rows, depth, streams, stream);
        } else {
            launch_down_multi<__nv_bfloat16, false, Columns>(
                weight, activation, output, injection_rows, injection, rows, depth, streams, stream);
        }
    }
}

} // namespace

void hc_project_gemv_launch(const void* weight, std::int32_t rows, std::int32_t depth,
                            const void* activation, Tensor& output, cudaStream_t stream) {
    const bool vectorizable = (depth % 8) == 0;
    const bool fp16        = output.dtype == DType::FP16;
    const auto blocks =
        static_cast<unsigned int>((rows + ((rows >= 1024 ? 3 : 0))) / (rows >= 1024 ? 4 : 1));
    if (vectorizable && depth <= 2048) {
        const auto warp_blocks =
            static_cast<unsigned int>((rows + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock);
        if (fp16) {
            hc_project_gemv_warp_row_kernel<__half><<<warp_blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __half*>(weight), static_cast<const __half*>(activation),
                static_cast<__half*>(output.data), rows, depth);
        } else {
            hc_project_gemv_warp_row_kernel<__nv_bfloat16><<<warp_blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(activation),
                static_cast<__nv_bfloat16*>(output.data), rows, depth);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool tiled = rows >= 1024;
    if (fp16) {
        if (vectorizable) {
            if (tiled) {
                hc_project_gemv_kernel<__half, true, 4><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                    static_cast<const __half*>(weight), static_cast<const __half*>(activation),
                    static_cast<__half*>(output.data), rows, depth);
            } else {
                hc_project_gemv_kernel<__half, true, 1><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                    static_cast<const __half*>(weight), static_cast<const __half*>(activation),
                    static_cast<__half*>(output.data), rows, depth);
            }
        } else {
            hc_project_gemv_kernel<__half, false, 1><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __half*>(weight), static_cast<const __half*>(activation),
                static_cast<__half*>(output.data), rows, depth);
        }
    } else {
        if (vectorizable) {
            if (tiled) {
                hc_project_gemv_kernel<__nv_bfloat16, true, 4><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(weight),
                    static_cast<const __nv_bfloat16*>(activation),
                    static_cast<__nv_bfloat16*>(output.data), rows, depth);
            } else {
                hc_project_gemv_kernel<__nv_bfloat16, true, 1><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(weight),
                    static_cast<const __nv_bfloat16*>(activation),
                    static_cast<__nv_bfloat16*>(output.data), rows, depth);
            }
        } else {
            hc_project_gemv_kernel<__nv_bfloat16, false, 1><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(activation),
                static_cast<__nv_bfloat16*>(output.data), rows, depth);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_up_gate_launch(const void* weight, std::int32_t hidden, std::int32_t rank,
                           std::int32_t streams, const void* low, const Tensor& normalized,
                           Tensor& block_input, DType dtype, cudaStream_t stream) {
    const unsigned grid = static_cast<unsigned>(hidden / 2);
    const auto* low_p   = static_cast<const uint8_t*>(low);
    if (dtype == DType::FP16) {
        hc_mix_up_gate_kernel<__half><<<grid, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
            static_cast<const __half*>(weight),
            reinterpret_cast<const __half*>(low_p),
            static_cast<const __half*>(normalized.data),
            static_cast<__half*>(block_input.data), hidden, rank, streams, rank);
    } else {
        hc_mix_up_gate_kernel<__nv_bfloat16><<<grid, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<const __nv_bfloat16*>(low_p),
            static_cast<const __nv_bfloat16*>(normalized.data),
            static_cast<__nv_bfloat16*>(block_input.data), hidden, rank, streams, rank);
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_up_gate_multi_launch(const void* weight, std::int32_t hidden, std::int32_t rank,
                                  std::int32_t streams, const void* low, const Tensor& normalized,
                                  Tensor& block_input, DType dtype, std::int32_t columns,
                                  cudaStream_t stream) {
    const auto* low_p = static_cast<const uint8_t*>(low);
    switch (columns) {
        case 2:
            dispatch_up_gate_multi<2>(weight, low_p, normalized.data, block_input.data, hidden,
                                      rank, streams, dtype, stream);
            break;
        case 3:
            dispatch_up_gate_multi<3>(weight, low_p, normalized.data, block_input.data, hidden,
                                      rank, streams, dtype, stream);
            break;
        case 4:
            dispatch_up_gate_multi<4>(weight, low_p, normalized.data, block_input.data, hidden,
                                      rank, streams, dtype, stream);
            break;
        case 5:
            dispatch_up_gate_multi<5>(weight, low_p, normalized.data, block_input.data, hidden,
                                      rank, streams, dtype, stream);
            break;
        case 6:
            dispatch_up_gate_multi<6>(weight, low_p, normalized.data, block_input.data, hidden,
                                      rank, streams, dtype, stream);
            break;
        case 7:
            dispatch_up_gate_multi<7>(weight, low_p, normalized.data, block_input.data, hidden,
                                      rank, streams, dtype, stream);
            break;
        case 8:
            dispatch_up_gate_multi<8>(weight, low_p, normalized.data, block_input.data, hidden,
                                      rank, streams, dtype, stream);
            break;
        default:
            throw std::invalid_argument("hc_mix_up_gate_multi_launch: unsupported column count");
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_silu_launch(Tensor& values, std::int32_t streams, cudaStream_t stream) {
    const std::int64_t count = values.numel();
    const auto blocks = static_cast<unsigned int>((count + kBlock - 1) / kBlock);
    if (values.dtype == DType::FP16) {
        hc_mix_silu_kernel<__half><<<blocks, kBlock, 0, stream>>>(
            static_cast<__half*>(values.data), streams, count);
    } else {
        hc_mix_silu_kernel<__nv_bfloat16><<<blocks, kBlock, 0, stream>>>(
            static_cast<__nv_bfloat16*>(values.data), streams, count);
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_gate_launch(const Tensor& gates, const Tensor& normalized, Tensor& block_input,
                        std::int32_t hidden, std::int32_t streams, cudaStream_t stream) {
    const std::int32_t columns = normalized.ne[2];
    const dim3 blocks(static_cast<unsigned int>(columns),
                      static_cast<unsigned int>((hidden + kBlock - 1) / kBlock));
    if (gates.dtype == DType::FP16) {
        hc_mix_gate_kernel<__half><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __half*>(gates.data), static_cast<const __half*>(normalized.data),
            static_cast<__half*>(block_input.data), hidden, streams);
    } else {
        hc_mix_gate_kernel<__nv_bfloat16><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gates.data),
            static_cast<const __nv_bfloat16*>(normalized.data),
            static_cast<__nv_bfloat16*>(block_input.data), hidden, streams);
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_down_silu_launch(const void* weight, std::int32_t rows, std::int32_t depth,
                             const void* activation, Tensor& output, const void* injection_rows,
                             DType weight_dtype, Tensor* injection, std::int32_t streams,
                             cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(rows) + (injection_rows != nullptr ? streams : 0);
    const bool vectorizable = (depth % 8) == 0;
    float* injection_data = injection != nullptr ? static_cast<float*>(injection->data) : nullptr;
    if (weight_dtype == DType::FP16) {
        if (vectorizable) {
            hc_mix_down_kernel<__half, true><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __half*>(weight), static_cast<const __half*>(activation),
                static_cast<__half*>(output.data), static_cast<const __half*>(injection_rows),
                injection_data, rows, depth, streams);
        } else {
            hc_mix_down_kernel<__half, false><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __half*>(weight), static_cast<const __half*>(activation),
                static_cast<__half*>(output.data), static_cast<const __half*>(injection_rows),
                injection_data, rows, depth, streams);
        }
    } else if (vectorizable) {
        hc_mix_down_kernel<__nv_bfloat16, true>
            <<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(activation),
                static_cast<__nv_bfloat16*>(output.data),
                static_cast<const __nv_bfloat16*>(injection_rows), injection_data, rows, depth,
                streams);
    } else {
        hc_mix_down_kernel<__nv_bfloat16, false>
            <<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(activation),
                static_cast<__nv_bfloat16*>(output.data),
                static_cast<const __nv_bfloat16*>(injection_rows), injection_data, rows, depth,
                streams);
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_down_silu_multi_launch(const void* weight, std::int32_t rows, std::int32_t depth,
                                    const void* activation, Tensor& output,
                                    const void* injection_rows, DType weight_dtype,
                                    Tensor* injection, std::int32_t streams, std::int32_t columns,
                                    cudaStream_t stream) {
    const bool vectorizable = (depth % 8) == 0;
    float* injection_data = injection != nullptr ? static_cast<float*>(injection->data) : nullptr;
    switch (columns) {
        case 2:
            dispatch_down_multi<2>(weight, activation, output.data, injection_rows, injection_data,
                                   rows, depth, streams, weight_dtype, vectorizable, stream);
            break;
        case 3:
            dispatch_down_multi<3>(weight, activation, output.data, injection_rows, injection_data,
                                   rows, depth, streams, weight_dtype, vectorizable, stream);
            break;
        case 4:
            dispatch_down_multi<4>(weight, activation, output.data, injection_rows, injection_data,
                                   rows, depth, streams, weight_dtype, vectorizable, stream);
            break;
        case 5:
            dispatch_down_multi<5>(weight, activation, output.data, injection_rows, injection_data,
                                   rows, depth, streams, weight_dtype, vectorizable, stream);
            break;
        case 6:
            dispatch_down_multi<6>(weight, activation, output.data, injection_rows, injection_data,
                                   rows, depth, streams, weight_dtype, vectorizable, stream);
            break;
        case 7:
            dispatch_down_multi<7>(weight, activation, output.data, injection_rows, injection_data,
                                   rows, depth, streams, weight_dtype, vectorizable, stream);
            break;
        case 8:
            dispatch_down_multi<8>(weight, activation, output.data, injection_rows, injection_data,
                                   rows, depth, streams, weight_dtype, vectorizable, stream);
            break;
        default:
            throw std::invalid_argument("hc_mix_down_silu_multi_launch: unsupported column count");
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_injection_launch(const void* injection_rows, DType weight_dtype,
                             const Tensor& flattened, std::int32_t channels, std::int32_t streams,
                             Tensor& injection, cudaStream_t stream) {
    const std::int32_t columns = flattened.ne[1];
    const dim3 blocks(static_cast<unsigned int>(columns), static_cast<unsigned int>(streams));
    if (flattened.dtype == DType::FP16 && weight_dtype == DType::FP16) {
        hc_mix_injection_kernel<__half, __half><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __half*>(injection_rows),
            static_cast<const __half*>(flattened.data), static_cast<float*>(injection.data),
            channels, streams);
    } else {
        hc_mix_injection_kernel<__nv_bfloat16, __nv_bfloat16><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(injection_rows),
            static_cast<const __nv_bfloat16*>(flattened.data),
            static_cast<float*>(injection.data), channels, streams);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
