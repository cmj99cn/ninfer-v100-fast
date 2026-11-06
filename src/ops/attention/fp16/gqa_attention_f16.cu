#include "ninfer/ops/gqa_attention_f16.h"

#include "1cat_flash_attention/fp16_attention_paged.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("gqa_attention_f16: ") + message); }
}

__global__ void transpose_q_f16(const __half* src, __half* dst, int h, int t, int d) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = static_cast<std::int64_t>(h) * t * d;
    if (index >= count) { return; }
    const int dim = static_cast<int>(index % d);
    const int token = static_cast<int>((index / d) % t);
    const int head = static_cast<int>(index / (static_cast<std::int64_t>(d) * t));
    dst[index] = src[static_cast<std::int64_t>(dim) + static_cast<std::int64_t>(head) * d +
                     static_cast<std::int64_t>(token) * d * h];
}

__global__ void transpose_out_f16(const __half* src, __half* dst, int h, int t, int d) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = static_cast<std::int64_t>(h) * t * d;
    if (index >= count) { return; }
    const int dim = static_cast<int>(index % d);
    const int token = static_cast<int>((index / d) % t);
    const int head = static_cast<int>(index / (static_cast<std::int64_t>(d) * t));
    dst[static_cast<std::int64_t>(dim) + static_cast<std::int64_t>(head) * d +
        static_cast<std::int64_t>(token) * d * h] = src[index];
}

__global__ void set_seq_len(const int* positions, int count, int* seq_len) {
    seq_len[0] = positions[count - 1] + 1;
}

__global__ void append_f16_kernel(const __half* k, const __half* v, const int* positions,
                                  __half* k_pages, __half* v_pages, const int* block_table,
                                  int heads, int tokens, int page, int max_blocks, int d) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = static_cast<std::int64_t>(tokens) * heads * d;
    if (index >= count) { return; }
    const int dim = static_cast<int>(index % d);
    const int token = static_cast<int>((index / d) % tokens);
    const int head = static_cast<int>(index / (static_cast<std::int64_t>(d) * tokens));
    const int position = positions[token];
    const int page_index = position / page;
    if (page_index < 0 || page_index >= max_blocks) { return; }
    const int slot = position % page;
    const std::int64_t dst = static_cast<std::int64_t>(dim) +
                             static_cast<std::int64_t>(slot) * d +
                             static_cast<std::int64_t>(head) * d * page +
                             static_cast<std::int64_t>(block_table[page_index]) * d * page * heads;
    const std::int64_t src = static_cast<std::int64_t>(dim) +
                             static_cast<std::int64_t>(head) * d +
                             static_cast<std::int64_t>(token) * d * heads;
    k_pages[dst] = k[src];
    v_pages[dst] = v[src];
}

void validate(const Tensor& q, const Tensor& positions, const Tensor& out,
              const PagedKVLayerView& cache, float scale, GqaExecutionEnvelope envelope) {
    require(q.dtype == DType::FP16 && out.dtype == DType::FP16, "q/out must be FP16");
    require(positions.dtype == DType::I32, "positions must be I32");
    require(std::isfinite(scale) && std::fabs(scale - 0.0625f) <= 1.0e-6f,
            "scale must be 1/sqrt(256)");
    require(q.ne[0] == kHeadDim && out.ne[0] == kHeadDim, "head_dim must be 256");
    require(q.ne[2] > 0 && q.ne[3] == 1 && out.ne[3] == 1, "q/out must be [D,H,T,1]");
    require(positions.ne[0] == q.ne[2] && positions.ne[1] == 1, "positions must be [T]");
    require(out.ne[1] == q.ne[1] && out.ne[2] == q.ne[2], "out shape must match q");
    require(cache.dtype == DType::FP16 && cache.head_dim == kHeadDim,
            "cache must be FP16 with head_dim 256");
    require(cache.k_pages.dtype == DType::FP16 && cache.v_pages.dtype == DType::FP16,
            "cache pages must be FP16");
    require(cache.block_table.dtype == DType::I32, "block table must be I32");
    require(cache.quant_group == 0 && cache.k_scale_pages.data == nullptr &&
                cache.v_scale_pages.data == nullptr,
            "FP16 cache must not carry scales");
    require(cache.num_kv_heads == 4 || cache.num_kv_heads == 2,
            "unsupported KV head count");
    require(envelope.min_visible_keys > 0 &&
                envelope.min_visible_keys <= envelope.max_visible_keys &&
                envelope.max_visible_keys >= static_cast<std::uint32_t>(q.ne[2]),
            "invalid execution envelope");
}

} // namespace

std::size_t gqa_attention_f16_workspace_capacity_bytes(std::int32_t q_heads,
                                                       GqaExecutionEnvelope envelope,
                                                       std::int32_t min_width,
                                                       std::int32_t max_width) {
    (void)envelope;
    require(q_heads == 24 || q_heads == 12 || q_heads == 16 || q_heads == 6,
            "unsupported query head count");
    require(min_width > 0 && max_width >= min_width, "invalid width interval");
    return static_cast<std::size_t>(q_heads) * max_width * kHeadDim * sizeof(__half) * 2 +
           static_cast<std::size_t>(q_heads) * max_width * sizeof(float) + 4096;
}

void gqa_attention_f16_cached(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    validate(q, positions, out, cache, scale, envelope);
    const int h = q.ne[1];
    const int t = q.ne[2];
    const int page = cache.k_pages.ne[1];
    const int kv_heads = cache.num_kv_heads;
    const int max_blocks = cache.block_table.ne[0];
    require(page > 0 && max_blocks > 0, "invalid paged cache");
    auto scope = workspace.scope();
    Tensor q_t = workspace.alloc(DType::FP16, {h, t, kHeadDim});
    Tensor out_t = workspace.alloc(DType::FP16, {h, t, kHeadDim});
    Tensor lse = workspace.alloc(DType::FP32, {h, t});
    Tensor seq_len = workspace.alloc(DType::I32, {1});
    const std::int64_t transposed = static_cast<std::int64_t>(h) * t * kHeadDim;
    const int transpose_blocks = static_cast<int>((transposed + 255) / 256);
    transpose_q_f16<<<transpose_blocks, 256, 0, stream>>>(
        static_cast<const __half*>(q.data), static_cast<__half*>(q_t.data), h, t, kHeadDim);
    set_seq_len<<<1, 1, 0, stream>>>(static_cast<const int*>(positions.data), t,
                                     static_cast<int*>(seq_len.data));
    CUDA_CHECK(cudaGetLastError());

    const auto k_block_stride = cache.k_pages.nb[3] / static_cast<std::int64_t>(sizeof(__half));
    const auto k_token_stride = cache.k_pages.nb[1] / static_cast<std::int64_t>(sizeof(__half));
    const auto k_head_stride = cache.k_pages.nb[2] / static_cast<std::int64_t>(sizeof(__half));
    const auto v_block_stride = cache.v_pages.nb[3] / static_cast<std::int64_t>(sizeof(__half));
    const auto v_token_stride = cache.v_pages.nb[1] / static_cast<std::int64_t>(sizeof(__half));
    const auto v_head_stride = cache.v_pages.nb[2] / static_cast<std::int64_t>(sizeof(__half));

    third_party::onecat::flash_attention_forward_f16_paged_launch(
        q_t.data, cache.k_pages.data, cache.v_pages.data, out_t.data,
        static_cast<float*>(lse.data), static_cast<const int*>(cache.block_table.data),
        static_cast<const int*>(seq_len.data), 1, h, t, page, kv_heads, max_blocks,
        k_block_stride, k_token_stride, k_head_stride, v_block_stride, v_token_stride,
        v_head_stride, scale, true, 1.0f, 1.0f, -1, -1, stream);
    CUDA_CHECK(cudaGetLastError());
    transpose_out_f16<<<transpose_blocks, 256, 0, stream>>>(
        static_cast<const __half*>(out_t.data), static_cast<__half*>(out.data), h, t, kHeadDim);
    CUDA_CHECK(cudaGetLastError());
}

void gqa_kv_append_f16(const Tensor& k, const Tensor& v, const Tensor& positions,
                       PagedKVLayerView cache, cudaStream_t stream) {
    require(k.dtype == DType::FP16 && v.dtype == DType::FP16, "k/v must be FP16");
    require(positions.dtype == DType::I32, "positions must be I32");
    require(k.ne[0] == kHeadDim && v.ne[0] == kHeadDim, "head_dim must be 256");
    require(k.ne[1] == v.ne[1] && k.ne[2] == v.ne[2] && k.ne[3] == 1 && v.ne[3] == 1,
            "k/v shape mismatch");
    require(positions.ne[0] == k.ne[2] && positions.ne[1] == 1, "positions must be [T]");
    require(cache.dtype == DType::FP16 && cache.head_dim == kHeadDim, "cache must be FP16");
    require(cache.k_pages.dtype == DType::FP16 && cache.v_pages.dtype == DType::FP16,
            "cache pages must be FP16");
    require(cache.block_table.dtype == DType::I32, "block table must be I32");
    require(cache.num_kv_heads == k.ne[1], "cache head count mismatch");
    const int heads = k.ne[1];
    const int tokens = k.ne[2];
    const int page = cache.k_pages.ne[1];
    const int max_blocks = cache.block_table.ne[0];
    require(page > 0 && max_blocks > 0, "invalid paged cache");
    const std::int64_t count = static_cast<std::int64_t>(tokens) * heads * kHeadDim;
    const int blocks = static_cast<int>((count + 255) / 256);
    append_f16_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __half*>(k.data), static_cast<const __half*>(v.data),
        static_cast<const int*>(positions.data), static_cast<__half*>(cache.k_pages.data),
        static_cast<__half*>(cache.v_pages.data), static_cast<const int*>(cache.block_table.data),
        heads, tokens, page, max_blocks, kHeadDim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
