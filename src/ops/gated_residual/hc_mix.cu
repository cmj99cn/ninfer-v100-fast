#include "ops/gated_residual/gated_residual.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;
constexpr int kGemvWarpsPerBlock = 8;

__global__ void hc_mix_silu_kernel(__nv_bfloat16* values, std::int32_t streams,
                                   std::int64_t count) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (index >= count) { return; }
    const float value = __bfloat162float(values[index]);
    values[index]     = __float2bfloat16_rn(silu(value / static_cast<float>(streams)));
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

template <bool Vectorized>
__global__ __launch_bounds__(kGemvWarpsPerBlock * kWarpSize) void hc_project_gemv_kernel(
    const __nv_bfloat16* weight, const __nv_bfloat16* activation, __nv_bfloat16* output,
    std::int32_t rows, std::int32_t depth) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row  = static_cast<int>(blockIdx.x);
    if (row >= rows) { return; }

    const __nv_bfloat16* weight_row = weight + static_cast<std::int64_t>(row) * depth;
    float sum                        = 0.0F;
    if constexpr (Vectorized) {
        constexpr int kVector = 8;
        const int vectors     = depth / kVector;
        const auto* weights   = reinterpret_cast<const uint4*>(weight_row);
        const auto* values    = reinterpret_cast<const uint4*>(activation);
        for (int base = warp * kWarpSize; base < vectors; base += kGemvWarpsPerBlock * kWarpSize) {
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
    __shared__ float partials[kGemvWarpsPerBlock];
    if (lane == 0) { partials[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float total = 0.0F;
#pragma unroll
        for (int i = 0; i < kGemvWarpsPerBlock; ++i) { total += partials[i]; }
        if (lane == 0) { output[row] = __float2bfloat16_rn(total); }
    }
}

__global__ __launch_bounds__(kBlock) void hc_mix_gate_kernel(    const __nv_bfloat16* gates, const __nv_bfloat16* normalized, __nv_bfloat16* block_input,
    std::int32_t hidden, std::int32_t streams) {
    const std::int32_t m = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t d = static_cast<std::int32_t>(blockIdx.y) * kBlock +
                           static_cast<std::int32_t>(threadIdx.x);
    if (d >= hidden) { return; }

    float total = 0.0F;
    for (std::int32_t s = 0; s < streams; ++s) {
        const std::int64_t index = static_cast<std::int64_t>(s) * hidden + d +
                                   static_cast<std::int64_t>(m) * hidden * streams;
        total += sigmoid(__bfloat162float(gates[index])) * __bfloat162float(normalized[index]);
    }
    block_input[static_cast<std::int64_t>(m) * hidden + d] =
        __float2bfloat16_rn(total / static_cast<float>(streams));
}

__global__ __launch_bounds__(kBlock) void hc_mix_injection_kernel(
    const __nv_bfloat16* injection_rows, const __nv_bfloat16* flattened, float* injection,
    std::int32_t channels, std::int32_t streams) {
    const std::int32_t m        = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t s        = static_cast<std::int32_t>(blockIdx.y);
    const __nv_bfloat16* row     = injection_rows + static_cast<std::int64_t>(s) * channels;
    const __nv_bfloat16* column  = flattened + static_cast<std::int64_t>(m) * channels;

    float sum = 0.0F;
    for (std::int32_t c = static_cast<std::int32_t>(threadIdx.x); c < channels; c += kBlock) {
        sum += __bfloat162float(row[c]) * __bfloat162float(column[c]);
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
    const auto blocks = static_cast<unsigned int>(rows);
    const bool vectorizable = (depth % 8) == 0;
    if (vectorizable) {
        hc_project_gemv_kernel<true><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(weight), static_cast<const __nv_bfloat16*>(activation),
            static_cast<__nv_bfloat16*>(output.data), rows, depth);
    } else {
        hc_project_gemv_kernel<false><<<blocks, kGemvWarpsPerBlock * kWarpSize, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(weight), static_cast<const __nv_bfloat16*>(activation),
            static_cast<__nv_bfloat16*>(output.data), rows, depth);
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_silu_launch(Tensor& values, std::int32_t streams, cudaStream_t stream) {
    const std::int64_t count = values.numel();
    const auto blocks = static_cast<unsigned int>((count + kBlock - 1) / kBlock);
    hc_mix_silu_kernel<<<blocks, kBlock, 0, stream>>>(static_cast<__nv_bfloat16*>(values.data),
                                                      streams, count);
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_gate_launch(const Tensor& gates, const Tensor& normalized, Tensor& block_input,
                        std::int32_t hidden, std::int32_t streams, cudaStream_t stream) {
    const std::int32_t columns = normalized.ne[2];
    const dim3 blocks(static_cast<unsigned int>(columns),
                      static_cast<unsigned int>((hidden + kBlock - 1) / kBlock));
    hc_mix_gate_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gates.data),
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<__nv_bfloat16*>(block_input.data), hidden, streams);
    CUDA_CHECK(cudaGetLastError());
}

void hc_mix_injection_launch(const void* injection_rows, const Tensor& flattened,
                             std::int32_t channels, std::int32_t streams, Tensor& injection,
                             cudaStream_t stream) {
    const std::int32_t columns = flattened.ne[1];
    const dim3 blocks(static_cast<unsigned int>(columns), static_cast<unsigned int>(streams));
    hc_mix_injection_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(injection_rows),
        static_cast<const __nv_bfloat16*>(flattened.data), static_cast<float*>(injection.data),
        channels, streams);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
