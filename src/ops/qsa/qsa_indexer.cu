#include "ninfer/ops/qsa.h"

#include "core/device.h"
#include "ops/qsa/qsa_common.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using detail::qsa_row_position;

__device__ __forceinline__ bool candidate_better(float score_a, int block_a, float score_b,
                                                 int block_b) {
    return score_a > score_b || (score_a == score_b && block_a < block_b);
}

__device__ __forceinline__ void heap_sift_up(float* scores, int* blocks, int index) {
    while (index > 0) {
        const int parent = (index - 1) / 2;
        if (!candidate_better(scores[parent], blocks[parent], scores[index], blocks[index])) {
            break;
        }
        const float score = scores[index];
        const int block   = blocks[index];
        scores[index]     = scores[parent];
        blocks[index]     = blocks[parent];
        scores[parent]    = score;
        blocks[parent]    = block;
        index             = parent;
    }
}

__device__ __forceinline__ void heap_sift_down(float* scores, int* blocks, int size, int index) {
    while (true) {
        const int left  = index * 2 + 1;
        const int right = left + 1;
        if (left >= size) { break; }
        int worse = left;
        if (right < size && candidate_better(scores[left], blocks[left], scores[right], blocks[right])) {
            worse = right;
        }
        if (!candidate_better(scores[worse], blocks[worse], scores[index], blocks[index])) { break; }
        const float score = scores[index];
        const int block   = blocks[index];
        scores[index]     = scores[worse];
        blocks[index]     = blocks[worse];
        scores[worse]     = score;
        blocks[worse]     = block;
        index             = worse;
    }
}

__global__ void indexer_kernel(const __nv_bfloat16* index_query, const int* positions,
                               const int* valid_columns, int tokens, int batch, int axes,
                               const int* table_rows, int index_dim, int index_heads,
                               const int* page_table, int logical_pages,
                               const __nv_bfloat16* index_pages, int* selection, int* counts,
                               int capacity) {
    __shared__ float heap_score[512];
    __shared__ int heap_block[512];
    __shared__ float chunk_score[128];
    __shared__ int chunk_block[128];

    const int t = blockIdx.x % tokens;
    const int b = blockIdx.x / tokens;
    const int columns = valid_columns == nullptr ? tokens : valid_columns[b];
    const int output_base = 512 * (t + tokens * b);
    if (t >= columns) {
        if (threadIdx.x == 0) { counts[t + tokens * b] = 0; }
        return;
    }

    const int position = qsa_row_position(positions, t, b, batch, axes);
    const int complete = (position + 1) / kQsaIndexCompress;
    if (threadIdx.x == 0) { counts[t + tokens * b] = min(complete, capacity); }
    if (complete <= capacity) {
        for (int g = threadIdx.x; g < complete; g += blockDim.x) {
            selection[output_base + g] = g;
        }
        return;
    }

    int size = 0;
    for (int base = 0; base < complete; base += blockDim.x) {
        const int g = base + threadIdx.x;
        float score = 0.0F;
        if (g < complete) {
            const int logical_page = g / kQsaIndexBlockPageSize;
            const int row          = g % kQsaIndexBlockPageSize;
            const int table_row    = table_rows[b];
            const int physical =
                page_table[static_cast<std::int64_t>(logical_page) +
                           static_cast<std::int64_t>(logical_pages) * table_row];
            {
                const std::int64_t key_base =
                    index_dim * (static_cast<std::int64_t>(row) +
                                 kQsaIndexBlockPageSize * static_cast<std::int64_t>(physical));
                for (int h = 0; h < index_heads; ++h) {
                    const std::int64_t query_base =
                        index_dim * (h + static_cast<std::int64_t>(index_heads) *
                                             (t + static_cast<std::int64_t>(tokens) * b));
                    float dot = 0.0F;
                    for (int d = 0; d < index_dim; ++d) {
                        dot += __bfloat162float(index_query[query_base + d]) *
                               __bfloat162float(index_pages[key_base + d]);
                    }
                    score += fmaxf(0.0F, dot);
                }
            }
        }
        chunk_score[threadIdx.x] = score;
        chunk_block[threadIdx.x] = g;
        __syncthreads();
        if (threadIdx.x == 0) {
            const int chunk = min(static_cast<int>(blockDim.x), complete - base);
            for (int i = 0; i < chunk; ++i) {
                if (size < capacity) {
                    heap_score[size] = chunk_score[i];
                    heap_block[size] = chunk_block[i];
                    heap_sift_up(heap_score, heap_block, size);
                    ++size;
                } else if (candidate_better(chunk_score[i], chunk_block[i], heap_score[0],
                                            heap_block[0])) {
                    heap_score[0] = chunk_score[i];
                    heap_block[0] = chunk_block[i];
                    heap_sift_down(heap_score, heap_block, size, 0);
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int i = 1; i < size; ++i) {
            const float score = heap_score[i];
            const int block   = heap_block[i];
            int j             = i - 1;
            while (j >= 0 && heap_block[j] > block) {
                heap_score[j + 1] = heap_score[j];
                heap_block[j + 1] = heap_block[j];
                --j;
            }
            heap_score[j + 1] = score;
            heap_block[j + 1] = block;
        }
        for (int i = 0; i < size; ++i) { selection[output_base + i] = heap_block[i]; }
    }
}

} // namespace

void qsa_indexer(const Tensor& index_query, const ConstQsaIndexCacheView& cache,
                 const QsaBatchView& batch, std::uint32_t max_visible_tokens,
                 WorkspaceArena& workspace, QsaBlockSelection& selected, cudaStream_t stream) {
    (void)workspace;
    (void)max_visible_tokens;
    using detail::qsa_require;
    using detail::qsa_require_contiguous;
    using detail::qsa_require_dtype;

    qsa_require_dtype(index_query, DType::BF16, "index_query must be BF16");
    qsa_require_contiguous(index_query, "index_query must be contiguous");
    qsa_require_dtype(cache.index_k_pages, DType::BF16, "index key cache must be BF16");
    qsa_require_contiguous(cache.index_k_pages, "index key cache must be contiguous");
    qsa_require_dtype(cache.page_table, DType::I32, "page table must be I32");
    qsa_require_contiguous(cache.page_table, "page table must be contiguous");
    qsa_require_dtype(batch.positions, DType::I32, "positions must be I32");
    qsa_require_contiguous(batch.positions, "positions must be contiguous");
    qsa_require_dtype(batch.table_rows, DType::I32, "table_rows must be I32");
    qsa_require_contiguous(batch.table_rows, "table_rows must be contiguous");
    qsa_require_dtype(selected.selection, DType::I32, "selection must be I32");
    qsa_require_dtype(selected.counts, DType::I32, "counts must be I32");
    qsa_require_contiguous(selected.selection, "selection must be contiguous");
    qsa_require_contiguous(selected.counts, "counts must be contiguous");

    const std::int32_t index_dim = index_query.ne[0];
    const std::int32_t heads     = index_query.ne[1];
    const std::int32_t tokens    = index_query.ne[2];
    const std::int32_t batch_sz  = index_query.ne[3];
    qsa_require(index_dim == 128 && heads == 4, "index_query must be [128,4,T,B]");
    qsa_require(tokens > 0 && batch_sz > 0, "index_query token/batch extents must be positive");
    qsa_require(selected.selection.ne[0] == kQsaSelectionCapacity &&
                    selected.selection.ne[1] == tokens && selected.selection.ne[2] == batch_sz,
                "selection must be [512,T,B]");
    qsa_require(selected.counts.ne[0] == tokens && selected.counts.ne[1] == batch_sz,
                "counts must be [T,B]");
    qsa_require(batch.positions.ne[0] == tokens && batch.positions.ne[2] == batch_sz,
                "positions shape mismatch");
    qsa_require(batch.positions.ne[1] == 1 || batch.positions.ne[1] == 3,
                "positions must carry one or three axes");
    qsa_require(!detail::qsa_has_valid_columns(batch.valid_columns) ||
                    (batch.valid_columns.dtype == DType::I32 &&
                     batch.valid_columns.is_contiguous() &&
                     batch.valid_columns.ne[0] == batch_sz),
                "valid_columns must be I32 [B] or empty");
    qsa_require(batch.table_rows.ne[0] == batch_sz, "table_rows shape mismatch");
    qsa_require(cache.index_k_pages.ne[0] == index_dim &&
                    cache.index_k_pages.ne[1] == kQsaIndexBlockPageSize &&
                    cache.index_k_pages.ne[2] == 1,
                "index key cache plane shape mismatch");
    qsa_require(cache.page_table.ne[0] > 0 && cache.page_table.ne[1] > 0,
                "page table must have positive extents");

    const std::int64_t grid = static_cast<std::int64_t>(tokens) * batch_sz;
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("qsa_indexer: grid exceeds int range");
    }

    indexer_kernel<<<static_cast<unsigned>(grid), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(index_query.data),
        static_cast<const int*>(batch.positions.data),
        detail::qsa_has_valid_columns(batch.valid_columns)
            ? static_cast<const int*>(batch.valid_columns.data)
            : nullptr,
        tokens, batch_sz, batch.positions.ne[1], static_cast<const int*>(batch.table_rows.data),
        index_dim, heads, static_cast<const int*>(cache.page_table.data), cache.page_table.ne[0],
        static_cast<const __nv_bfloat16*>(cache.index_k_pages.data),
        static_cast<int*>(selected.selection.data), static_cast<int*>(selected.counts.data),
        kQsaSelectionCapacity);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
