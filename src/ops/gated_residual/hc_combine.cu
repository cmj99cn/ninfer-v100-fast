#include "ops/gated_residual/gated_residual.h"

#include "core/device.h"

#include "ops/gated_residual/act.h"

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
    const std::int32_t m    = static_cast<std::int32_t>(blockIdx.z);
    const std::int32_t s    = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(m) * hidden * streams +
                           static_cast<std::int64_t>(s) * hidden;
    const float coefficient =
        injection[static_cast<std::int64_t>(m) * streams + s];
    const A* output = block_output + static_cast<std::int64_t>(m) * hidden;
    const std::int32_t begin = static_cast<std::int32_t>(blockIdx.x) * kRowTile;
    const std::int32_t end   = min(begin + kRowTile, hidden);

    for (std::int32_t d = begin + static_cast<std::int32_t>(threadIdx.x); d < end; d += kBlock) {
        const float value = HcAct<A>::to_float(residual[base + d]) +
                            coefficient * HcAct<A>::to_float(output[d]);
        updated[base + d] = HcAct<A>::from_float(value);
    }
}

} // namespace

void hc_combine_launch(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                       Tensor& updated_residual, cudaStream_t stream) {
    const std::int32_t hidden  = residual.ne[0];
    const std::int32_t streams = residual.ne[1];
    const std::int32_t columns = residual.ne[2];
    const dim3 blocks(static_cast<unsigned int>((hidden + kRowTile - 1) / kRowTile),
                      static_cast<unsigned int>(streams), static_cast<unsigned int>(columns));
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
