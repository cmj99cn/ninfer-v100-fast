#include "ninfer/ops/qsa.h"

#include "core/device.h"
#include "ops/qsa/qsa_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using detail::QsaAct;
using detail::qsa_is_activation;
using detail::qsa_row_position;

constexpr int kScoreThreads          = 256;
constexpr int kSelectThreads         = 1024;
constexpr int kSelectPerThread       = 16;
constexpr int kSelectElementsPerPass = kSelectThreads * kSelectPerThread;
constexpr int kSelectRounds          = 7;
constexpr int kSelectHistogramBins   = 256;
constexpr int kSelectHistogramWarps  = kSelectHistogramBins / 32;
constexpr std::uint32_t kBlockKeyIdBits = 20;
constexpr std::uint32_t kBlockKeyIdMask = (1U << kBlockKeyIdBits) - 1U;

__device__ __forceinline__ unsigned long long block_selection_key(float score, int block) {
    const unsigned int bits = __float_as_uint(score);
    return (static_cast<unsigned long long>(bits) << kBlockKeyIdBits) |
           static_cast<unsigned long long>(kBlockKeyIdMask - static_cast<unsigned int>(block));
}

template <class A>
__global__ void indexer_score_kernel(const A* index_query, const int* positions,
                                     const int* valid_columns, int tokens, int batch, int axes,
                                     const int* table_rows, int index_dim, int index_heads,
                                     const int* page_table, int logical_pages,
                                     const __nv_bfloat16* index_pages, int* selection, int* counts,
                                     int capacity, float* scores, int score_stride,
                                     int token_base) {
    using Pair = typename QsaAct<A>::pair;

    const int chunk = static_cast<int>(blockIdx.x);
    const int tb    = token_base + static_cast<int>(blockIdx.y);
    const int t     = tb % tokens;
    const int b     = tb / tokens;
    const int columns = valid_columns == nullptr ? tokens : valid_columns[b];
    const int output_base = kQsaSelectionCapacity * (t + tokens * b);
    if (t >= columns) {
        if (threadIdx.x == 0) { counts[t + tokens * b] = 0; }
        return;
    }

    const int position = qsa_row_position(positions, t, b, batch, axes);
    const int complete = (position + 1) / kQsaIndexCompress;
    if (threadIdx.x == 0) { counts[t + tokens * b] = min(complete, capacity); }
    if (complete <= capacity) {
        for (int g = static_cast<int>(threadIdx.x); g < complete; g += blockDim.x) {
            selection[output_base + g] = g;
        }
        return;
    }

    const int g = chunk * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (g >= complete) { return; }
    const int logical_page = g / kQsaIndexBlockPageSize;
    const int row          = g % kQsaIndexBlockPageSize;
    const int table_row    = table_rows[b];
    const int physical =
        page_table[static_cast<std::int64_t>(logical_page) +
                   static_cast<std::int64_t>(logical_pages) * table_row];
    const std::int64_t key_base =
        index_dim * (static_cast<std::int64_t>(row) +
                     kQsaIndexBlockPageSize * static_cast<std::int64_t>(physical));
    const __nv_bfloat16* key_ptr = index_pages + key_base;

    float score = 0.0F;
    for (int h = 0; h < index_heads; ++h) {
        const A* query_ptr =
            index_query + index_dim * (h + static_cast<std::int64_t>(index_heads) *
                                              (t + static_cast<std::int64_t>(tokens) * b));
        float dot = 0.0F;
        for (int d = 0; d < index_dim; d += 8) {
            const uint4 raw_key    = *reinterpret_cast<const uint4*>(key_ptr + d);
            const uint4 raw_query  = *reinterpret_cast<const uint4*>(query_ptr + d);
            const __nv_bfloat162* key_pairs = reinterpret_cast<const __nv_bfloat162*>(&raw_key);
            const Pair* query_pairs          = reinterpret_cast<const Pair*>(&raw_query);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 key_values   = QsaAct<__nv_bfloat16>::to_float2(key_pairs[j]);
                const float2 query_values = QsaAct<A>::to_float2(query_pairs[j]);
                dot += query_values.x * key_values.x;
                dot += query_values.y * key_values.y;
            }
        }
        score += fmaxf(0.0F, dot);
    }
    scores[static_cast<std::size_t>(tb - token_base) * score_stride + g] = score;
}

__global__ void __launch_bounds__(kSelectThreads)
indexer_select_kernel(const int* positions, const int* valid_columns, int tokens, int batch,
                      int axes, const float* scores, int* selection, int* counts, int capacity,
                      int score_stride, int token_base) {
    __shared__ unsigned int histogram[kSelectHistogramBins];
    __shared__ unsigned int suffix[kSelectHistogramBins];
    __shared__ unsigned int warp_totals[kSelectHistogramWarps];
    __shared__ int picked[kQsaSelectionCapacity];
    __shared__ int picked_count;

    const int tb = token_base + static_cast<int>(blockIdx.x);
    const int t  = tb % tokens;
    const int b  = tb / tokens;
    const int columns = valid_columns == nullptr ? tokens : valid_columns[b];
    const int output_base = kQsaSelectionCapacity * (t + tokens * b);
    if (t >= columns) {
        if (threadIdx.x == 0) { counts[t + tokens * b] = 0; }
        return;
    }

    const int position = qsa_row_position(positions, t, b, batch, axes);
    const int complete = (position + 1) / kQsaIndexCompress;
    if (complete <= capacity) { return; }

    const float* block_scores = scores + static_cast<std::size_t>(tb - token_base) * score_stride;
    const unsigned int lane = threadIdx.x & 31U;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;

    unsigned long long prefix = 0;
    int unresolved           = 56;
    int remaining            = capacity;

    for (int round = 0; round < kSelectRounds; ++round) {
        for (int i = static_cast<int>(threadIdx.x); i < kSelectHistogramBins; i += kSelectThreads) {
            histogram[i] = 0;
        }
        __syncthreads();
        const int digit_shift = unresolved - 8;
        for (int base = 0; base < complete; base += kSelectElementsPerPass) {
            const int start = base + static_cast<int>(threadIdx.x);
#pragma unroll
            for (int k = 0; k < kSelectPerThread; ++k) {
                const int g = start + k * kSelectThreads;
                if (g < complete) {
                    const unsigned long long key = block_selection_key(block_scores[g], g);
                    if ((key >> unresolved) == prefix) {
                        const int digit = static_cast<int>((key >> digit_shift) & 0xFFULL);
                        const unsigned int peers = __match_any_sync(__activemask(), digit);
                        if (lane == static_cast<unsigned int>(__ffs(static_cast<int>(peers)) - 1)) {
                            atomicAdd(&histogram[digit], static_cast<unsigned int>(__popc(peers)));
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < kSelectHistogramBins) {
            unsigned int value = histogram[threadIdx.x];
            for (int offset = 1; offset < 32; offset <<= 1) {
                const unsigned int other = __shfl_down_sync(0xffffffffU, value, offset);
                if (lane + static_cast<unsigned int>(offset) < 32U) { value += other; }
            }
            suffix[threadIdx.x] = value;
            if (lane == 0U) { warp_totals[warp] = value; }
        }
        __syncthreads();
        if (threadIdx.x < kSelectHistogramBins) {
            unsigned int above = 0;
            for (int w = warp + 1; w < kSelectHistogramWarps; ++w) { above += warp_totals[w]; }
            suffix[threadIdx.x] += above;
        }
        __syncthreads();
        int chosen = kSelectHistogramBins - 1;
        {
            int low  = 0;
            int high = kSelectHistogramBins - 1;
            while (low < high) {
                const int mid = (low + high + 1) >> 1;
                if (suffix[mid] >= static_cast<unsigned int>(remaining)) {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            chosen = low;
        }
        const unsigned int above =
            (chosen == kSelectHistogramBins - 1) ? 0U : suffix[chosen + 1];
        const unsigned int at_chosen = suffix[chosen] - above;
        remaining  -= static_cast<int>(above);
        prefix      = (prefix << 8) | static_cast<unsigned long long>(chosen);
        unresolved  = digit_shift;
        if (at_chosen == static_cast<unsigned int>(remaining)) { break; }
    }

    if (threadIdx.x == 0) { picked_count = 0; }
    __syncthreads();
    const unsigned long long threshold = prefix;
    for (int base = 0; base < complete; base += kSelectElementsPerPass) {
        const int start = base + static_cast<int>(threadIdx.x);
#pragma unroll
        for (int k = 0; k < kSelectPerThread; ++k) {
            const int g = start + k * kSelectThreads;
            if (g < complete) {
                const unsigned long long key = block_selection_key(block_scores[g], g);
                if ((key >> unresolved) >= threshold) {
                    const int slot = atomicAdd(&picked_count, 1);
                    if (slot < kQsaSelectionCapacity) { picked[slot] = g; }
                }
            }
        }
    }
    __syncthreads();
    const int total = picked_count;
    for (int index = static_cast<int>(threadIdx.x); index < total; index += kSelectThreads) {
        const int block = picked[index];
        int rank        = 0;
        for (int j = 0; j < total; ++j) { rank += (picked[j] < block) ? 1 : 0; }
        selection[output_base + rank] = block;
    }
}

} // namespace

void qsa_indexer(const Tensor& index_query, const ConstQsaIndexCacheView& cache,
                 const QsaBatchView& batch, std::uint32_t max_visible_tokens,
                 WorkspaceArena& workspace, QsaBlockSelection& selected, cudaStream_t stream) {
    (void)max_visible_tokens;
    using detail::qsa_require;
    using detail::qsa_require_contiguous;
    using detail::qsa_require_dtype;

    qsa_require(qsa_is_activation(index_query.dtype),
                "index_query must be BF16 or FP16");
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
    qsa_require(cache.index_k_pages.ne[3] > 0, "index key cache must have a page extent");
    qsa_require(cache.page_table.ne[0] > 0 && cache.page_table.ne[1] > 0,
                "page table must have positive extents");

    const std::int64_t grid = static_cast<std::int64_t>(tokens) * batch_sz;
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("qsa_indexer: grid exceeds int range");
    }

    const std::int64_t max_blocks =
        static_cast<std::int64_t>(cache.index_k_pages.ne[3]) * kQsaIndexBlockPageSize;
    qsa_require(max_blocks <= static_cast<std::int64_t>(kBlockKeyIdMask) + 1,
                "index key cache holds more blocks than the selection key can order");

    const auto* positions = static_cast<const int*>(batch.positions.data);
    const auto* valid_columns = detail::qsa_has_valid_columns(batch.valid_columns)
                                    ? static_cast<const int*>(batch.valid_columns.data)
                                    : nullptr;
    const auto* table_rows = static_cast<const int*>(batch.table_rows.data);
    const auto* page_table = static_cast<const int*>(cache.page_table.data);
    const auto* index_pages = static_cast<const __nv_bfloat16*>(cache.index_k_pages.data);
    auto* selection = static_cast<int*>(selected.selection.data);
    auto* counts = static_cast<int*>(selected.counts.data);

    const std::size_t score_stride = static_cast<std::size_t>(max_blocks);
    const std::size_t score_bytes  = score_stride * sizeof(float);
    const std::size_t reserved     = workspace.used();
    const std::size_t capacity     = workspace.capacity();
    const std::size_t available    = capacity > reserved ? capacity - reserved : 0;
    std::size_t wave = std::min(kQsaIndexerScoreWaveBytes, available) / score_bytes;
    if (wave == 0) { wave = 1; }
    wave = std::min(wave, static_cast<std::size_t>(grid));

    auto scope = workspace.scope();
    const DeviceSpan score_span = workspace.alloc_bytes(score_bytes * wave, 256);
    auto* scores = static_cast<float*>(score_span.data);

    const std::int32_t chunks = static_cast<std::int32_t>(
        (score_stride + kScoreThreads - 1) / kScoreThreads);
    for (std::size_t base = 0; base < static_cast<std::size_t>(grid); base += wave) {
        const std::int32_t width = static_cast<std::int32_t>(
            std::min(wave, static_cast<std::size_t>(grid) - base));
        const int token_base = static_cast<int>(base);
        if (index_query.dtype == DType::FP16) {
            indexer_score_kernel<__half><<<dim3(chunks, width), kScoreThreads, 0, stream>>>(
                static_cast<const __half*>(index_query.data), positions, valid_columns, tokens,
                batch_sz, batch.positions.ne[1], table_rows, index_dim, heads, page_table,
                cache.page_table.ne[0], index_pages, selection, counts, kQsaSelectionCapacity,
                scores, static_cast<int>(score_stride), token_base);
        } else {
            indexer_score_kernel<__nv_bfloat16>
                <<<dim3(chunks, width), kScoreThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(index_query.data), positions, valid_columns,
                tokens, batch_sz, batch.positions.ne[1], table_rows, index_dim, heads, page_table,
                cache.page_table.ne[0], index_pages, selection, counts, kQsaSelectionCapacity,
                scores, static_cast<int>(score_stride), token_base);
        }
        indexer_select_kernel<<<width, kSelectThreads, 0, stream>>>(
            positions, valid_columns, tokens, batch_sz, batch.positions.ne[1], scores, selection,
            counts, kQsaSelectionCapacity, static_cast<int>(score_stride), token_base);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
