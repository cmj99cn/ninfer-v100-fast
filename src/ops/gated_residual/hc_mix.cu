#include "ops/gated_residual/gated_residual.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include "ops/gated_residual/act.h"

#include <cuda_bf16.h>

#include <cstdint>
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

template <bool Vectorized, int Rows>
__global__ __launch_bounds__(kGemvWarpsPerBlock * kWarpSize) void hc_project_gemv_kernel(
    const __nv_bfloat16* weight, const __nv_bfloat16* activation, __nv_bfloat16* output,
    std::int32_t rows, std::int32_t depth) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    __shared__ float partials[kGemvWarpsPerBlock];
    for (int local = 0; local < Rows; ++local) {
        const int row = static_cast<int>(blockIdx.x) * Rows + local;
        if (row >= rows) { break; }
        const __nv_bfloat16* weight_row = weight + static_cast<std::int64_t>(row) * depth;
        float sum                       = 0.0F;
        if constexpr (Vectorized) {
            constexpr int kVector = 8;
            const int vectors     = depth / kVector;
            const auto* weights   = reinterpret_cast<const uint4*>(weight_row);
            const auto* values    = reinterpret_cast<const uint4*>(activation);
            for (int base = warp * kWarpSize; base < vectors;
                 base += kGemvWarpsPerBlock * kWarpSize) {
                const int vector_index = base + lane;
                if (vector_index >= vectors) { break; }
                float w[kVector];
                float a[kVector];
                unpack_bf16x8(weights[vector_index], w);
                unpack_bf16x8(values[vector_index], a);
#pragma unroll
                for (int i = 0; i < kVector; ++i) { sum += w[i] * a[i]; }
            }
        } else {
            for (std::int32_t index = warp * kWarpSize + lane; index < depth;
                 index += kGemvWarpsPerBlock * kWarpSize) {
                sum += __bfloat162float(weight_row[index]) * __bfloat162float(activation[index]);
            }
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) { partials[warp] = sum; }
        __syncthreads();
        if (warp == 0) {
            float total = 0.0F;
#pragma unroll
            for (int i = 0; i < kGemvWarpsPerBlock; ++i) { total += partials[i]; }
            if (lane == 0) { output[row] = __float2bfloat16_rn(total); }
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
            const auto* weights   = reinterpret_cast<const uint4*>(weight_row);
            const auto* values    = reinterpret_cast<const uint4*>(activation);
            for (int base = warp * kWarpSize; base < vectors;
                 base += kGemvWarpsPerBlock * kWarpSize) {
                const int vector_index = base + lane;
                if (vector_index >= vectors) { break; }
                float w[kVector];
                float a[kVector];
                unpack_act_x8<A>(weights[vector_index], w);
                unpack_act_x8<A>(values[vector_index], a);
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

template <class A>
__global__ __launch_bounds__(kBlock) void hc_mix_up_gate_kernel(
    const A* __restrict__ weight, const A* __restrict__ activation,
    const A* __restrict__ normalized, A* __restrict__ block_input, std::int32_t hidden,
    std::int32_t streams, std::int32_t depth) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int d    = static_cast<int>(blockIdx.x);
    const int s    = warp >> 1;
    const int chunk = warp & 1;
    __shared__ float partials[4][kGemvWarpsPerBlock];
    __shared__ A gate[4];
    if (threadIdx.x < 4 * kGemvWarpsPerBlock) {
        reinterpret_cast<float*>(partials)[threadIdx.x] = 0.0F;
    }
    __syncthreads();

    const int row = s * hidden + d;
    const A* weight_row = weight + static_cast<std::int64_t>(row) * depth;
    const int vectors = depth / 8;
    const auto* weights = reinterpret_cast<const uint4*>(weight_row);
    const auto* values  = reinterpret_cast<const uint4*>(activation);
    float sum           = 0.0F;
    for (int base = chunk * kWarpSize; base < vectors;
         base += kGemvWarpsPerBlock * kWarpSize) {
        const int vector_index = base + lane;
        if (vector_index >= vectors) { break; }
        float w[8];
        float a[8];
        unpack_act_x8<A>(weights[vector_index], w);
        unpack_act_x8<A>(values[vector_index], a);
#pragma unroll
        for (int i = 0; i < 8; ++i) { sum += w[i] * a[i]; }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) { partials[s][chunk] = sum; }
    __syncthreads();
    if (lane == 0 && chunk == 0) {
        float total = 0.0F;
#pragma unroll
        for (int i = 0; i < kGemvWarpsPerBlock; ++i) { total += partials[s][i]; }
        gate[s] = HcAct<A>::from_float(total);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0F;
        for (std::int32_t stream = 0; stream < streams; ++stream) {
            const std::int64_t index = static_cast<std::int64_t>(stream) * hidden + d;
            total += sigmoid(HcAct<A>::to_float(gate[stream])) *
                     HcAct<A>::to_float(normalized[index]);
        }
        block_input[d] = HcAct<A>::from_float(total / static_cast<float>(streams));
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

} // namespace

void hc_project_gemv_launch(const void* weight, std::int32_t rows, std::int32_t depth,
                            const void* activation, Tensor& output, cudaStream_t stream) {
    const bool vectorizable = (depth % 8) == 0;
    const bool tiled        = rows >= 1024;
    const auto blocks = static_cast<unsigned int>((rows + (tiled ? 3 : 0)) / (tiled ? 4 : 1));
    if (vectorizable) {
        if (tiled) {
            hc_project_gemv_kernel<true, 4><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(activation),
                static_cast<__nv_bfloat16*>(output.data), rows, depth);
        } else {
            hc_project_gemv_kernel<true, 1><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(activation),
                static_cast<__nv_bfloat16*>(output.data), rows, depth);
        }
    } else {
        hc_project_gemv_kernel<false, 1><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(weight), static_cast<const __nv_bfloat16*>(activation),
            static_cast<__nv_bfloat16*>(output.data), rows, depth);
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

void hc_mix_up_gate_launch(const void* weight, std::int32_t rank, const void* activation,
                            const Tensor& normalized, Tensor& block_input, std::int32_t hidden,
                            std::int32_t streams, cudaStream_t stream) {
    if (normalized.dtype == DType::FP16) {
        hc_mix_up_gate_kernel<__half><<<static_cast<unsigned>(hidden), kBlock, 0, stream>>>(
            static_cast<const __half*>(weight), static_cast<const __half*>(activation),
            static_cast<const __half*>(normalized.data), static_cast<__half*>(block_input.data),
            hidden, streams, rank);
    } else {
        hc_mix_up_gate_kernel<__nv_bfloat16><<<static_cast<unsigned>(hidden), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(weight),
            static_cast<const __nv_bfloat16*>(activation),
            static_cast<const __nv_bfloat16*>(normalized.data),
            static_cast<__nv_bfloat16*>(block_input.data), hidden, streams, rank);
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
