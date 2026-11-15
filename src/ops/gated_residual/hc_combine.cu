#include "ops/gated_residual/gated_residual.h"

#include "core/device.h"

#include "ops/gated_residual/act.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;
constexpr int kRowTile = 256;

template <class A>
__global__ __launch_bounds__(kBlock) void hc_combine_kernel(
    const A* residual, const A* block_output, const float* injection, A* updated,
    std::int32_t hidden, std::int32_t streams) {
    const std::int32_t m    = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t s    = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t tile = static_cast<std::int32_t>(blockIdx.z);
    const std::int64_t base = static_cast<std::int64_t>(m) * hidden * streams +
                           static_cast<std::int64_t>(s) * hidden;
    const float coefficient =
        injection[static_cast<std::int64_t>(m) * streams + s];
    const A* output = block_output + static_cast<std::int64_t>(m) * hidden;
    const std::int32_t begin = tile * kRowTile;
    const std::int32_t end   = min(begin + kRowTile, hidden);

    for (std::int32_t d = begin + static_cast<std::int32_t>(threadIdx.x); d < end; d += kBlock) {
        const float value = HcAct<A>::to_float(residual[base + d]) +
                            coefficient * HcAct<A>::to_float(output[d]);
        updated[base + d] = HcAct<A>::from_float(value);
    }
}

template <class A>
__global__ __launch_bounds__(kBlock) void hc_combine_norm_kernel(
    const A* __restrict__ residual, const A* __restrict__ block_output,
    const float* __restrict__ injection, const __nv_bfloat16* __restrict__ norm_weight,
    A* __restrict__ updated, A* __restrict__ normalized, std::int32_t hidden,
    std::int32_t streams, float epsilon) {
    const std::int32_t m    = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t s    = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(m) * hidden * streams +
                              static_cast<std::int64_t>(s) * hidden;
    const float coefficient = injection[static_cast<std::int64_t>(m) * streams + s];
    const A* output         = block_output + static_cast<std::int64_t>(m) * hidden;
    const __nv_bfloat16* weight = norm_weight + static_cast<std::int64_t>(s) * hidden;

    float sum = 0.0F;
    for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < hidden; d += kBlock) {
        const float value =
            HcAct<A>::to_float(residual[base + d]) +
            coefficient * HcAct<A>::to_float(output[d]);
        const A rounded   = HcAct<A>::from_float(value);
        updated[base + d] = rounded;
        const float stored = HcAct<A>::to_float(rounded);
        sum += stored * stored;
    }
    __shared__ float scratch[kBlock / kWarpSize];
    __shared__ float inverse_shared;
    const float total = block_reduce_sum<kBlock>(sum, scratch);
    if (threadIdx.x == 0) {
        inverse_shared = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();
    const float inverse = inverse_shared;
    for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < hidden; d += kBlock) {
        const float value = HcAct<A>::to_float(updated[base + d]);
        const float gain  = 1.0F + __bfloat162float(weight[d]);
        normalized[base + d] = HcAct<A>::from_float(value * gain * inverse);
    }
}

} // namespace

void hc_combine_norm_launch(const Tensor& residual, const Tensor& block_output,
                            const Tensor& injection, const Tensor& next_norm_weight, float epsilon,
                            Tensor& updated_residual, Tensor& normalized, cudaStream_t stream) {
    const std::int32_t hidden  = residual.ne[0];
    const std::int32_t streams = residual.ne[1];
    const std::int32_t columns = residual.ne[2];
    const dim3 blocks(static_cast<unsigned int>(columns), static_cast<unsigned int>(streams));
    if (residual.dtype == DType::FP16) {
        hc_combine_norm_kernel<__half><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __half*>(residual.data),
            static_cast<const __half*>(block_output.data),
            static_cast<const float*>(injection.data),
            static_cast<const __nv_bfloat16*>(next_norm_weight.data),
            static_cast<__half*>(updated_residual.data), static_cast<__half*>(normalized.data),
            hidden, streams, epsilon);
    } else {
        hc_combine_norm_kernel<__nv_bfloat16><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<const __nv_bfloat16*>(block_output.data),
            static_cast<const float*>(injection.data),
            static_cast<const __nv_bfloat16*>(next_norm_weight.data),
            static_cast<__nv_bfloat16*>(updated_residual.data),
            static_cast<__nv_bfloat16*>(normalized.data), hidden, streams, epsilon);
    }
    CUDA_CHECK(cudaGetLastError());
}

void hc_combine_launch(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                       Tensor& updated_residual, cudaStream_t stream) {
    const std::int32_t hidden  = residual.ne[0];
    const std::int32_t streams = residual.ne[1];
    const std::int32_t columns = residual.ne[2];
    const dim3 blocks(static_cast<unsigned int>(columns),
                      static_cast<unsigned int>(streams),
                      static_cast<unsigned int>((hidden + kRowTile - 1) / kRowTile));
    if (residual.dtype == DType::FP16) {
        hc_combine_kernel<__half><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __half*>(residual.data),
            static_cast<const __half*>(block_output.data),
            static_cast<const float*>(injection.data), static_cast<__half*>(updated_residual.data),
            hidden, streams);
    } else {
        hc_combine_kernel<__nv_bfloat16><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<const __nv_bfloat16*>(block_output.data),
            static_cast<const float*>(injection.data),
            static_cast<__nv_bfloat16*>(updated_residual.data), hidden, streams);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
