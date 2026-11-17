#include "ops/launcher/scatter.h"

#include "ops/kernel/scatter.cuh"
#include "core/device.h"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__global__ void stage_gdn_decode_kernel(const __half* q, const __half* k, const __half* v,
                                        const float* g, const float* beta, __half* conv_in,
                                        float* g_rank, float* beta_rank, int key_width,
                                        int conv_channels, int value_heads, int base) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < key_width) {
        conv_in[index] = q[index];
    } else if (index < 2 * key_width) {
        conv_in[index] = k[index - key_width];
    } else if (index < conv_channels) {
        conv_in[index] = v[index - 2 * key_width];
    } else if (index < conv_channels + value_heads) {
        g_rank[index - conv_channels] = g[base + index - conv_channels];
    } else if (index < conv_channels + 2 * value_heads) {
        beta_rank[index - conv_channels - value_heads] =
            beta[base + index - conv_channels - value_heads];
    }
}

} // namespace

void stage_gdn_decode_launch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, std::int32_t rank, Tensor& conv_in,
                             Tensor& g_rank, Tensor& beta_rank, cudaStream_t stream) {
    const int key_width    = q.ne[0];
    const int conv_channels = conv_in.ne[0];
    const int value_heads  = g_rank.ne[0];
    const int total = conv_channels + 2 * value_heads;
    constexpr int block = 256;
    const int grid = (total + block - 1) / block;
    stage_gdn_decode_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __half*>(q.data), static_cast<const __half*>(k.data),
        static_cast<const __half*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<__half*>(conv_in.data),
        static_cast<float*>(g_rank.data), static_cast<float*>(beta_rank.data), key_width,
        conv_channels, value_heads, rank * value_heads);
    CUDA_CHECK(cudaGetLastError());
}

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream) {
    constexpr int block        = 256;
    constexpr int vector_block = 128;
    const int d                = src.ne[0];
    const int vision           = src.ne[1];
    const auto src_addr        = reinterpret_cast<std::uintptr_t>(src.data);
    const auto dst_addr        = reinterpret_cast<std::uintptr_t>(dst.data);
    if ((d % 8) == 0 && ((src_addr | dst_addr) & 0xfu) == 0) {
        scatter_bf16x8_kernel<<<vision, vector_block, 0, stream>>>(
            static_cast<const uint4*>(src.data), static_cast<const std::int32_t*>(indices.data),
            static_cast<uint4*>(dst.data), d / 8);
    } else if ((d & 1) == 0 && ((src_addr | dst_addr) & 0x3u) == 0) {
        scatter_bf16x2_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(src.data),
            static_cast<const std::int32_t*>(indices.data), static_cast<__nv_bfloat162*>(dst.data),
            d / 2);
    } else {
        scatter_scalar_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(src.data),
            static_cast<const std::int32_t*>(indices.data), static_cast<__nv_bfloat16*>(dst.data),
            d);
    }
    CUDA_CHECK(cudaGetLastError());
}

void scatter_bf16_batch_launch(const Tensor& source, const Tensor& lanes,
                               const Tensor& valid_columns, Tensor& destination,
                               cudaStream_t stream) {
    constexpr int block = 128;
    const dim3 grid(source.ne[1], source.ne[2], 1);
    scatter_bf16_batch_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(source.data), static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(valid_columns.data), static_cast<uint4*>(destination.data),
        source.ne[0] / 8, source.ne[1],
        destination.nb[1] / static_cast<std::int64_t>(sizeof(uint4)),
        destination.nb[2] / static_cast<std::int64_t>(sizeof(uint4)));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
