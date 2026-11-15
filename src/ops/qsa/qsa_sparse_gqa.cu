#include "ninfer/ops/qsa.h"

#include "core/device.h"
#include "ops/qsa/qsa_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

using detail::QsaAct;
using detail::qsa_is_activation;
using detail::qsa_row_position;

inline constexpr int kScoreWarpLanes = 8;
inline constexpr int kScoreWarpWidth = kScoreWarpLanes * 32;
static_assert(kScoreWarpWidth == 256, "qsa score warp reduction requires head_dim 256");

__device__ float block_reduce_sum(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }
    return scratch[0];
}

__device__ float block_reduce_max(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

template <class A>
__global__ void sparse_gqa_kernel(const A* query, const A* output_gate,
                                  const int* positions, const int* valid_columns,
                                  const int* table_rows, int tokens, int batch, int axes,
                                  int head_dim, int q_heads, int kv_heads, const int* page_table,
                                  int logical_pages,
                                  const __nv_bfloat16* k_pages, const __half* v_pages,
                                  const int* selection, const int* counts, A* out) {
    __shared__ float reduce[256];
    __shared__ float probabilities[4 * kQsaSelectionCapacity + 3];

    const int index = blockIdx.x;
    const int h     = index % q_heads;
    const int t     = (index / q_heads) % tokens;
    const int b     = index / (q_heads * tokens);
    const int columns = valid_columns == nullptr ? tokens : valid_columns[b];
    const std::int64_t output_base =
        head_dim * (h + static_cast<std::int64_t>(q_heads) *
                            (t + static_cast<std::int64_t>(tokens) * b));

    if (t >= columns) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out[output_base + d] = QsaAct<A>::from_float(0.0F);
        }
        return;
    }

    const int position = qsa_row_position(positions, t, b, batch, axes);
    const int complete = (position + 1) / kQsaIndexCompress;
    const int count    = counts[t + tokens * b];
    const int tail     = position + 1 - 4 * complete;
    const int count_j  = 4 * count + tail;
    const int selection_base = 512 * (t + tokens * b);
    const int group          = q_heads / kv_heads;
    const int kv_head        = h / group;
    const float scale        = 1.0F / sqrtf(static_cast<float>(head_dim));
    const int table_row      = table_rows[b];

    if (count_j == 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out[output_base + d] = QsaAct<A>::from_float(0.0F);
        }
        return;
    }

    const auto token_of = [&](int j) {
        if (j < 4 * count) {
            return 4 * selection[selection_base + j / 4] + j % 4;
        }
        return 4 * complete + (j - 4 * count);
    };
    const auto key_offset = [&](int token) {
        const int page     = token / kQsaTokenPageSize;
        const int row      = token % kQsaTokenPageSize;
        const int physical = page_table[static_cast<std::int64_t>(page) +
                                        static_cast<std::int64_t>(logical_pages) * table_row];
        return head_dim * (static_cast<std::int64_t>(row) +
                           kQsaTokenPageSize * (static_cast<std::int64_t>(kv_head) +
                                                static_cast<std::int64_t>(kv_heads) * physical));
    };

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    for (int j = warp; j < count_j; j += warps) {
        const int token = token_of(j);
        const std::int64_t cache_base = key_offset(token);
        float partial[kScoreWarpLanes];
#pragma unroll
        for (int k = 0; k < kScoreWarpLanes; ++k) {
            const int d = lane + 32 * k;
            partial[k] = QsaAct<A>::to_float(query[output_base + d]) *
                         __bfloat162float(k_pages[cache_base + d]);
        }
        partial[0] += partial[4];
        partial[1] += partial[5];
        partial[2] += partial[6];
        partial[3] += partial[7];
        partial[0] += partial[2];
        partial[1] += partial[3];
        float total = partial[0] + partial[1];
        total += __shfl_down_sync(0xffffffffU, total, 16);
        total += __shfl_down_sync(0xffffffffU, total, 8);
        total += __shfl_down_sync(0xffffffffU, total, 4);
        total += __shfl_down_sync(0xffffffffU, total, 2);
        total += __shfl_down_sync(0xffffffffU, total, 1);
        if (lane == 0) { probabilities[j] = total * scale; }
    }
    __syncthreads();

    float maximum = -INFINITY;
    for (int j = threadIdx.x; j < count_j; j += blockDim.x) {
        maximum = fmaxf(maximum, probabilities[j]);
    }
    maximum = block_reduce_max(reduce, maximum);

    float sum = 0.0F;
    for (int j = threadIdx.x; j < count_j; j += blockDim.x) {
        sum += expf(probabilities[j] - maximum);
    }
    const float denominator = block_reduce_sum(reduce, sum);
    for (int j = threadIdx.x; j < count_j; j += blockDim.x) {
        probabilities[j] = expf(probabilities[j] - maximum) / denominator;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float attention = 0.0F;
        for (int j = 0; j < count_j; ++j) {
            const int token = token_of(j);
            const std::int64_t cache_base = key_offset(token);
            attention += probabilities[j] * __half2float(v_pages[cache_base + d]);
        }
        const float gate = QsaAct<A>::to_float(output_gate[output_base + d]);
        const float sigmoid = 1.0F / (1.0F + expf(-gate));
        out[output_base + d] = QsaAct<A>::from_float(attention * sigmoid);
    }
}

} // namespace

void qsa_sparse_gqa(const Tensor& query, const Tensor& output_gate, const QsaCacheView& cache,
                    const QsaBatchView& batch, const QsaBlockSelection& selected,
                    std::uint32_t max_visible_tokens, WorkspaceArena& workspace, Tensor& out,
                    cudaStream_t stream) {
    (void)workspace;
    (void)max_visible_tokens;
    using detail::qsa_require;
    using detail::qsa_require_contiguous;
    using detail::qsa_require_dtype;

    qsa_require(qsa_is_activation(query.dtype), "query must be BF16 or FP16");
    qsa_require(output_gate.dtype == query.dtype, "output_gate dtype must match query");
    qsa_require(out.dtype == query.dtype, "out dtype must match query");
    qsa_require_dtype(cache.k_pages, DType::BF16, "key cache must be BF16");
    qsa_require_dtype(cache.v_pages, DType::FP16, "value cache must be FP16");
    qsa_require_dtype(cache.page_table, DType::I32, "page table must be I32");
    qsa_require_dtype(batch.positions, DType::I32, "positions must be I32");
    qsa_require_dtype(batch.table_rows, DType::I32, "table_rows must be I32");
    qsa_require_dtype(selected.selection, DType::I32, "selection must be I32");
    qsa_require_dtype(selected.counts, DType::I32, "counts must be I32");

    const std::int32_t head_dim = query.ne[0];
    const std::int32_t q_heads  = query.ne[1];
    const std::int32_t tokens   = query.ne[2];
    const std::int32_t batch_sz = query.ne[3];
    const std::int32_t kv_heads = cache.k_pages.ne[2];
    qsa_require(head_dim == kScoreWarpWidth && q_heads > 0 && kv_heads > 0 && tokens > 0 &&
                    batch_sz > 0,
                "invalid query geometry");
    qsa_require(q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads");
    qsa_require(output_gate.ne[0] == head_dim && output_gate.ne[1] == q_heads &&
                    output_gate.ne[2] == tokens && output_gate.ne[3] == batch_sz,
                "output_gate shape mismatch");
    qsa_require(out.ne[0] == head_dim && out.ne[1] == q_heads && out.ne[2] == tokens &&
                    out.ne[3] == batch_sz,
                "out shape mismatch");
    qsa_require(cache.k_pages.ne[0] == head_dim &&
                    cache.k_pages.ne[1] == kQsaTokenPageSize && cache.k_pages.ne[2] == kv_heads,
                "key cache plane shape mismatch");
    qsa_require(cache.v_pages.ne[0] == head_dim &&
                    cache.v_pages.ne[1] == kQsaTokenPageSize && cache.v_pages.ne[2] == kv_heads,
                "value cache plane shape mismatch");
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
    qsa_require(cache.page_table.ne[0] > 0 && cache.page_table.ne[1] > 0,
                "page table must have positive extents");
    qsa_require_contiguous(query, "query must be contiguous");
    qsa_require_contiguous(output_gate, "output_gate must be contiguous");
    qsa_require_contiguous(out, "out must be contiguous");
    qsa_require_contiguous(cache.k_pages, "key cache must be contiguous");
    qsa_require_contiguous(cache.v_pages, "value cache must be contiguous");
    qsa_require_contiguous(cache.page_table, "page table must be contiguous");
    qsa_require_contiguous(batch.positions, "positions must be contiguous");
    qsa_require_contiguous(batch.table_rows, "table_rows must be contiguous");
    qsa_require_contiguous(selected.selection, "selection must be contiguous");
    qsa_require_contiguous(selected.counts, "counts must be contiguous");

    const std::int64_t grid = static_cast<std::int64_t>(tokens) * batch_sz * q_heads;
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("qsa_sparse_gqa: grid exceeds int range");
    }

    const auto* positions = static_cast<const int*>(batch.positions.data);
    const auto* valid_columns = detail::qsa_has_valid_columns(batch.valid_columns)
                                    ? static_cast<const int*>(batch.valid_columns.data)
                                    : nullptr;
    const auto* table_rows = static_cast<const int*>(batch.table_rows.data);
    const auto* page_table = static_cast<const int*>(cache.page_table.data);
    const auto* k_pages = static_cast<const __nv_bfloat16*>(cache.k_pages.data);
    const auto* v_pages = static_cast<const __half*>(cache.v_pages.data);
    const auto* selection = static_cast<const int*>(selected.selection.data);
    const auto* count_values = static_cast<const int*>(selected.counts.data);
    if (query.dtype == DType::FP16) {
        sparse_gqa_kernel<__half><<<static_cast<unsigned>(grid), 256, 0, stream>>>(
            static_cast<const __half*>(query.data),
            static_cast<const __half*>(output_gate.data), positions, valid_columns, table_rows,
            tokens, batch_sz, batch.positions.ne[1], head_dim, q_heads, kv_heads, page_table,
            cache.page_table.ne[0], k_pages, v_pages, selection, count_values,
            static_cast<__half*>(out.data));
    } else {
        sparse_gqa_kernel<__nv_bfloat16><<<static_cast<unsigned>(grid), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(query.data),
            static_cast<const __nv_bfloat16*>(output_gate.data), positions, valid_columns,
            table_rows, tokens, batch_sz, batch.positions.ne[1], head_dim, q_heads, kv_heads,
            page_table, cache.page_table.ne[0], k_pages, v_pages, selection, count_values,
            static_cast<__nv_bfloat16*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
