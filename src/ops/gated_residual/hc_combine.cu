#include "ops/gated_residual/gated_residual.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

__global__ __launch_bounds__(kBlock) void hc_combine_kernel(
    const __nv_bfloat16* residual, const __nv_bfloat16* block_output, const float* injection,
    __nv_bfloat16* updated, std::int32_t hidden, std::int32_t streams) {
    const std::int32_t m    = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t s    = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(m) * hidden * streams +
                           static_cast<std::int64_t>(s) * hidden;
    const float coefficient =
        injection[static_cast<std::int64_t>(m) * streams + s];
    const __nv_bfloat16* output = block_output + static_cast<std::int64_t>(m) * hidden;

    for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < hidden; d += kBlock) {
        const float value = static_cast<float>(__bfloat162float(residual[base + d])) +
                            coefficient * static_cast<float>(__bfloat162float(output[d]));
        updated[base + d] = __float2bfloat16_rn(value);
    }
}

} // namespace

void hc_combine_launch(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                       Tensor& updated_residual, cudaStream_t stream) {
    const std::int32_t hidden  = residual.ne[0];
    const std::int32_t streams = residual.ne[1];
    const std::int32_t columns = residual.ne[2];
    const dim3 blocks(static_cast<unsigned int>(columns), static_cast<unsigned int>(streams));
    hc_combine_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(block_output.data), static_cast<const float*>(injection.data),
        static_cast<__nv_bfloat16*>(updated_residual.data), hidden, streams);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
