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

__global__ void stage_gdn_span_kernel(const __half* q, const __half* k, const __half* v,
                                      const float* g, const float* beta, __half* conv_in,
                                      float* g_rank, float* beta_rank, int key_width,
                                      int conv_channels, int value_heads, int g_heads,
                                      int head_offset) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token  = blockIdx.y;
    const int total  = conv_channels + 2 * value_heads;
    if (column >= total) { return; }
    if (column < key_width) {
        conv_in[token * conv_channels + column] = q[token * key_width + column];
    } else if (column < 2 * key_width) {
        conv_in[token * conv_channels + column] =
            k[token * key_width + column - key_width];
    } else if (column < conv_channels) {
        const int value_width = conv_channels - 2 * key_width;
        conv_in[token * conv_channels + column] =
            v[token * value_width + column - 2 * key_width];
    } else if (column < conv_channels + value_heads) {
        const int head = column - conv_channels;
        g_rank[token * value_heads + head] = g[token * g_heads + head_offset + head];
    } else {
        const int head = column - conv_channels - value_heads;
        beta_rank[token * value_heads + head] = beta[token * g_heads + head_offset + head];
    }
}

__global__ void extract_gdn_columns_kernel(const __half* source, __half* q, __half* k, __half* v,
                                           int key_width, int conv_channels, int tokens) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token  = blockIdx.y;
    if (column >= conv_channels) { return; }
    const int value_width = conv_channels - 2 * key_width;
    const auto value = source[token * conv_channels + column];
    if (column < key_width) {
        q[token * key_width + column] = value;
    } else if (column < 2 * key_width) {
        k[token * key_width + column - key_width] = value;
    } else {
        v[token * value_width + column - 2 * key_width] = value;
    }
}

void stage_gdn_span_launch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                           const Tensor& beta, std::int32_t rank, Tensor& conv_in,
                           Tensor& g_rank, Tensor& beta_rank, cudaStream_t stream) {
    const int key_width    = q.ne[0];
    const int conv_channels = conv_in.ne[0];
    const int value_heads  = g_rank.ne[0];
    const int g_heads      = g.ne[0];
    const int tokens       = q.ne[1];
    const int total        = conv_channels + 2 * value_heads;
    constexpr int block = 256;
    const dim3 grid((total + block - 1) / block, tokens);
    stage_gdn_span_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __half*>(q.data), static_cast<const __half*>(k.data),
        static_cast<const __half*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<__half*>(conv_in.data),
        static_cast<float*>(g_rank.data), static_cast<float*>(beta_rank.data), key_width,
        conv_channels, value_heads, g_heads, rank * value_heads);
    CUDA_CHECK(cudaGetLastError());
}

void extract_gdn_columns_launch(const Tensor& source, Tensor& q, Tensor& k, Tensor& v,
                                cudaStream_t stream) {
    const int key_width    = q.ne[0];
    const int conv_channels = source.ne[0];
    const int tokens       = source.ne[1];
    constexpr int block = 256;
    const dim3 grid((conv_channels + block - 1) / block, tokens);
    extract_gdn_columns_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __half*>(source.data), static_cast<__half*>(q.data),
        static_cast<__half*>(k.data), static_cast<__half*>(v.data), key_width, conv_channels,
        tokens);
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
