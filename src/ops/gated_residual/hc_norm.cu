#include "ops/gated_residual/gated_residual.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include "ops/gated_residual/act.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <class A, int Block>
__global__ __launch_bounds__(Block) void hc_expand_norm_kernel(
    const A* embedding, const __nv_bfloat16* norm_weight, A* residual, A* normalized,
    std::int32_t hidden, std::int32_t streams, float epsilon) {
    const std::int32_t m           = static_cast<std::int32_t>(blockIdx.x);
    const A* column                = embedding + static_cast<std::int64_t>(m) * hidden;
    const std::int64_t column_base = static_cast<std::int64_t>(m) * hidden * streams;

    float sum = 0.0F;
    for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < hidden; d += Block) {
        const float value = HcAct<A>::to_float(column[d]);
        sum += value * value;
    }
    __shared__ float scratch[Block / kWarpSize];
    __shared__ float inverse_shared;
    const float total = block_reduce_sum<Block>(sum, scratch);
    if (threadIdx.x == 0) {
        inverse_shared = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();
    const float inverse = inverse_shared;

    for (std::int32_t s = 0; s < streams; ++s) {
        A* out_residual = residual + column_base + static_cast<std::int64_t>(s) * hidden;
        A* out_normalized = normalized + column_base + static_cast<std::int64_t>(s) * hidden;
        const __nv_bfloat16* weight = norm_weight + static_cast<std::int64_t>(s) * hidden;
        for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < hidden; d += Block) {
            const float value = HcAct<A>::to_float(column[d]);
            const float gain  = 1.0F + __bfloat162float(weight[d]);
            out_residual[d]   = column[d];
            out_normalized[d] = HcAct<A>::from_float(value * gain * inverse);
        }
    }
}

template <class A, int Block>
__global__ __launch_bounds__(Block) void hc_grouped_rmsnorm_kernel(
    const A* residual, const __nv_bfloat16* norm_weight, A* normalized, std::int32_t hidden,
    std::int32_t streams, float epsilon) {
    const std::int32_t m        = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t s        = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base     = static_cast<std::int64_t>(m) * hidden * streams +
                               static_cast<std::int64_t>(s) * hidden;
    const A* source    = residual + base;
    A* destination     = normalized + base;

    float sum = 0.0F;
    for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < hidden; d += Block) {
        const float value = HcAct<A>::to_float(source[d]);
        sum += value * value;
    }
    __shared__ float scratch[Block / kWarpSize];
    __shared__ float inverse_shared;
    const float total = block_reduce_sum<Block>(sum, scratch);
    if (threadIdx.x == 0) {
        inverse_shared = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();
    const float inverse = inverse_shared;

    const __nv_bfloat16* weight = norm_weight + static_cast<std::int64_t>(s) * hidden;
    for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < hidden; d += Block) {
        const float value = HcAct<A>::to_float(source[d]);
        const float gain  = 1.0F + __bfloat162float(weight[d]);
        destination[d]    = HcAct<A>::from_float(value * gain * inverse);
    }
}

} // namespace

void hc_expand_norm_launch(const Tensor& embedding, const Tensor& norm_weight, float epsilon,
                           Tensor& residual, Tensor& normalized, cudaStream_t stream) {
    const std::int32_t hidden  = embedding.ne[0];
    const std::int32_t streams = norm_weight.ne[1];
    const std::int32_t columns = embedding.ne[1];
    if (embedding.dtype == DType::FP16) {
        hc_expand_norm_kernel<__half, kBlock>
            <<<static_cast<unsigned int>(columns), kBlock, 0, stream>>>(
                static_cast<const __half*>(embedding.data),
                static_cast<const __nv_bfloat16*>(norm_weight.data),
                static_cast<__half*>(residual.data), static_cast<__half*>(normalized.data), hidden,
                streams, epsilon);
    } else {
        hc_expand_norm_kernel<__nv_bfloat16, kBlock>
            <<<static_cast<unsigned int>(columns), kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(embedding.data),
                static_cast<const __nv_bfloat16*>(norm_weight.data),
                static_cast<__nv_bfloat16*>(residual.data),
                static_cast<__nv_bfloat16*>(normalized.data), hidden, streams, epsilon);
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_grouped_rmsnorm_launch(const Tensor& residual, const Tensor& norm_weight, float epsilon,
                                Tensor& normalized, cudaStream_t stream) {
    const std::int32_t hidden  = residual.ne[0];
    const std::int32_t streams = residual.ne[1];
    const std::int32_t columns = residual.ne[2];
    const dim3 blocks(static_cast<unsigned int>(columns), static_cast<unsigned int>(streams));
    if (residual.dtype == DType::FP16) {
        hc_grouped_rmsnorm_kernel<__half, kBlock><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __half*>(residual.data),
            static_cast<const __nv_bfloat16*>(norm_weight.data),
            static_cast<__half*>(normalized.data), hidden, streams, epsilon);
    } else {
        hc_grouped_rmsnorm_kernel<__nv_bfloat16, kBlock><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<const __nv_bfloat16*>(norm_weight.data),
            static_cast<__nv_bfloat16*>(normalized.data), hidden, streams, epsilon);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
