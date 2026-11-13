#include "ninfer/ops/qsa.h"

#include "core/device.h"
#include "ops/qsa/qsa_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using detail::qsa_position_axis;
using detail::qsa_rope_dim;
using detail::qsa_row_position;

__device__ float block_sum(float* scratch, float value, int threads) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }
    return scratch[0];
}

__global__ void prepare_query_kernel(
    const __nv_bfloat16* projected, const __nv_bfloat16* weight, const int* positions,
    const int* valid_columns, int tokens, int batch, int axes, int head_dim, int heads,
    float epsilon, int rotary_dim, float theta, __nv_bfloat16* out) {
    extern __shared__ float shared[];
    float* values = shared;
    float* sums   = shared + head_dim;

    const int index   = blockIdx.x;
    const int h       = index % heads;
    const int t       = (index / heads) % tokens;
    const int b       = index / (heads * tokens);
    const int columns = valid_columns == nullptr ? tokens : valid_columns[b];
    if (t >= columns) { return; }

    const std::int64_t input_base =
        static_cast<std::int64_t>(head_dim) * (h + static_cast<std::int64_t>(heads) *
                                                      (t + static_cast<std::int64_t>(tokens) * b));
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        values[d] = __bfloat162float(projected[input_base + d]);
    }
    __syncthreads();

    float square = 0.0F;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) { square += values[d] * values[d]; }
    const float total = block_sum(sums, square, blockDim.x);
    const float inverse = rsqrtf(total / static_cast<float>(head_dim) + epsilon);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        values[d] = values[d] * (1.0F + __bfloat162float(weight[d])) * inverse;
    }
    __syncthreads();

    const int rho0 = qsa_position_axis(positions, t, b, batch, axes, 0);
    const int rho1 = qsa_position_axis(positions, t, b, batch, axes, 1);
    const int rho2 = qsa_position_axis(positions, t, b, batch, axes, 2);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[input_base + d] =
            __float2bfloat16_rn(qsa_rope_dim(values, d, rotary_dim, theta, rho0, rho1, rho2));
    }
}

__global__ void prepare_key_value_kernel(
    const __nv_bfloat16* projected_key, const __nv_bfloat16* projected_value,
    const __nv_bfloat16* weight, const int* positions, const int* valid_columns,
    const int* table_rows, int tokens, int batch, int axes, int head_dim, int kv_heads,
    float epsilon, int rotary_dim, float theta, const int* page_table,
    int logical_pages, __nv_bfloat16* k_pages, __half* v_pages) {
    extern __shared__ float shared[];
    float* values = shared;
    float* sums   = shared + head_dim;

    const int index   = blockIdx.x;
    const int kv_head = index % kv_heads;
    const int t       = (index / kv_heads) % tokens;
    const int b       = index / (kv_heads * tokens);
    const int columns = valid_columns == nullptr ? tokens : valid_columns[b];
    if (t >= columns) { return; }

    const std::int64_t input_base =
        static_cast<std::int64_t>(head_dim) *
        (kv_head + static_cast<std::int64_t>(kv_heads) *
                       (t + static_cast<std::int64_t>(tokens) * b));
    const int position = qsa_row_position(positions, t, b, batch, axes);
    const int table_row = table_rows[b];
    const int page      = position / kQsaTokenPageSize;
    const int row  = position % kQsaTokenPageSize;
    const int physical = page_table[static_cast<std::int64_t>(page) +
                                    static_cast<std::int64_t>(logical_pages) * table_row];
    const std::int64_t cache_base =
        head_dim * (static_cast<std::int64_t>(row) +
                    kQsaTokenPageSize * (static_cast<std::int64_t>(kv_head) +
                                         static_cast<std::int64_t>(kv_heads) * physical));

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        values[d] = __bfloat162float(projected_key[input_base + d]);
        v_pages[cache_base + d] = __float2half_rn(__bfloat162float(projected_value[input_base + d]));
    }
    __syncthreads();

    float square = 0.0F;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) { square += values[d] * values[d]; }
    const float total = block_sum(sums, square, blockDim.x);
    const float inverse = rsqrtf(total / static_cast<float>(head_dim) + epsilon);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        values[d] = values[d] * (1.0F + __bfloat162float(weight[d])) * inverse;
    }
    __syncthreads();

    const int rho0 = qsa_position_axis(positions, t, b, batch, axes, 0);
    const int rho1 = qsa_position_axis(positions, t, b, batch, axes, 1);
    const int rho2 = qsa_position_axis(positions, t, b, batch, axes, 2);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        k_pages[cache_base + d] =
            __float2bfloat16_rn(qsa_rope_dim(values, d, rotary_dim, theta, rho0, rho1, rho2));
    }
}

__global__ void prepare_records_kernel(const __nv_bfloat16* index_key, const int* positions,
                                       const int* valid_columns, int tokens, int batch, int axes,
                                       int index_dim, int record_width,
                                       __nv_bfloat16* record_keys, int* record_positions,
                                       int* base_positions) {
    const int t       = blockIdx.x;
    const int b       = blockIdx.y;
    const int columns = valid_columns == nullptr ? tokens : valid_columns[b];
    if (t >= columns) { return; }

    for (int d = threadIdx.x; d < index_dim; d += blockDim.x) {
        record_keys[d + static_cast<std::int64_t>(index_dim) *
                            (t + static_cast<std::int64_t>(record_width) * b)] =
            index_key[d + static_cast<std::int64_t>(index_dim) *
                              (t + static_cast<std::int64_t>(tokens) * b)];
    }
    for (int axis = threadIdx.x; axis < 3; axis += blockDim.x) {
        record_positions[axis + 3 * (t + static_cast<std::int64_t>(record_width) * b)] =
            qsa_position_axis(positions, t, b, batch, axes, axis);
    }
    if (threadIdx.x == 0 && t == 0) { base_positions[b] = qsa_row_position(positions, 0, b, batch, axes); }
}

__global__ void prepare_index_blocks_kernel(
    const __nv_bfloat16* index_key, const __nv_bfloat16* index_key_norm,
    const int* positions, const int* valid_columns, const int* table_rows,
    const __nv_bfloat16* tail_keys, const int* tail_rope, int tokens, int batch, int axes,
    int index_dim, int rotary_dim, float theta, float epsilon,
    const int* page_table, int logical_pages, int compress, __nv_bfloat16* index_pages) {
    extern __shared__ float shared[];
    float* values = shared;
    float* sums   = shared + index_dim;

    const int b       = blockIdx.y;
    const int columns = valid_columns == nullptr ? tokens : valid_columns[b];
    const int first   = qsa_row_position(positions, 0, b, batch, axes);
    const int first_block = first / compress;
    const int end_block   = (first + columns) / compress;
    const int g           = first_block + blockIdx.x;
    if (g >= end_block) { return; }

    const int block_start = g * compress;
    float raw_sum         = 0.0F;
    for (int d = threadIdx.x; d < index_dim; d += blockDim.x) {
        float sum = 0.0F;
        for (int c = 0; c < compress; ++c) {
            const int token = block_start + c;
            float value = 0.0F;
            if (token < first) {
                const int column = token - (first - first % compress);
                value = __bfloat162float(tail_keys[d + static_cast<std::int64_t>(index_dim) *
                                                         (column + 4 * b)]);
            } else {
                const int local = token - first;
                value = __bfloat162float(index_key[d + static_cast<std::int64_t>(index_dim) *
                                                           (local + static_cast<std::int64_t>(tokens) * b)]);
            }
            sum += value;
        }
        values[d] = __bfloat162float(__float2bfloat16_rn(sum / static_cast<float>(compress)));
        raw_sum += values[d] * values[d];
    }
    const float total = block_sum(sums, raw_sum, blockDim.x);
    const float inverse = rsqrtf(total / static_cast<float>(index_dim) + epsilon);
    for (int d = threadIdx.x; d < index_dim; d += blockDim.x) {
        values[d] = values[d] * (1.0F + __bfloat162float(index_key_norm[d])) * inverse;
    }
    __syncthreads();

    int rho0 = 0;
    int rho1 = 0;
    int rho2 = 0;
    if (block_start < first) {
        rho0 = tail_rope[0 + 3 * b];
        rho1 = tail_rope[1 + 3 * b];
        rho2 = tail_rope[2 + 3 * b];
    } else {
        const int local = block_start - first;
        rho0 = qsa_position_axis(positions, local, b, batch, axes, 0);
        rho1 = qsa_position_axis(positions, local, b, batch, axes, 1);
        rho2 = qsa_position_axis(positions, local, b, batch, axes, 2);
    }

    const int logical_page = g / kQsaIndexBlockPageSize;
    const int table_row = table_rows[b];
    const int physical =
        page_table[static_cast<std::int64_t>(logical_page) +
                   static_cast<std::int64_t>(logical_pages) * table_row];
    const int row = g % kQsaIndexBlockPageSize;
    const std::int64_t cache_base =
        index_dim * (static_cast<std::int64_t>(row) +
                     kQsaIndexBlockPageSize * static_cast<std::int64_t>(physical));
    for (int d = threadIdx.x; d < index_dim; d += blockDim.x) {
        index_pages[cache_base + d] =
            __float2bfloat16_rn(qsa_rope_dim(values, d, rotary_dim, theta, rho0, rho1, rho2));
    }
}

std::int64_t positive_grid(std::int64_t value, const char* label) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("qsa_prepare: ") + label);
    }
    return value;
}

} // namespace

void qsa_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const QsaTailStateView& before, QsaCacheView cache,
                 WorkspaceArena& workspace, QsaRawRecordWriteView& records, Tensor& query,
                 Tensor& index_query, cudaStream_t stream) {
    (void)workspace;
    using detail::qsa_require;
    using detail::qsa_require_contiguous;
    using detail::qsa_require_dtype;

    qsa_require_dtype(projected.query, DType::BF16, "query projection must be BF16");
    qsa_require_dtype(projected.key, DType::BF16, "key projection must be BF16");
    qsa_require_dtype(projected.value, DType::BF16, "value projection must be BF16");
    qsa_require_dtype(projected.index_query, DType::BF16, "index query projection must be BF16");
    qsa_require_dtype(projected.index_key, DType::BF16, "index key projection must be BF16");
    qsa_require_dtype(query, DType::BF16, "query output must be BF16");
    qsa_require_dtype(index_query, DType::BF16, "index query output must be BF16");
    qsa_require_dtype(cache.k_pages, DType::BF16, "key cache must be BF16");
    qsa_require_dtype(cache.v_pages, DType::FP16, "value cache must be FP16");
    qsa_require_dtype(cache.index_k_pages, DType::BF16, "index key cache must be BF16");
    qsa_require_dtype(cache.page_table, DType::I32, "page table must be I32");

    qsa_require_dtype(parameters.query_norm, DType::BF16, "query norm must be BF16");
    qsa_require_dtype(parameters.key_norm, DType::BF16, "key norm must be BF16");
    qsa_require_dtype(parameters.index_query_norm, DType::BF16, "index query norm must be BF16");
    qsa_require_dtype(parameters.index_key_norm, DType::BF16, "index key norm must be BF16");
    qsa_require(parameters.rotary_dim == 64, "rotary_dim must be 64");
    qsa_require(parameters.rms_epsilon > 0.0F && std::isfinite(parameters.rms_epsilon),
                "rms_epsilon must be positive and finite");
    qsa_require(parameters.theta > 0.0F && std::isfinite(parameters.theta),
                "theta must be positive and finite");

    const std::int32_t head_dim = projected.query.ne[0];
    const std::int32_t q_heads  = projected.query.ne[1];
    const std::int32_t tokens   = projected.query.ne[2];
    const std::int32_t batch_sz = projected.query.ne[3];
    const std::int32_t kv_heads = projected.key.ne[1];
    const std::int32_t index_dim = projected.index_query.ne[0];
    const std::int32_t index_heads = projected.index_query.ne[1];

    qsa_require(head_dim > 0 && q_heads > 0 && kv_heads > 0 && tokens > 0 && batch_sz > 0,
                "invalid projection geometry");
    qsa_require(q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads");
    qsa_require(index_heads == 4 && index_dim == 128 && projected.index_key.ne[1] == 1,
                "index geometry must be [128,4] with one index kv head");
    qsa_require(parameters.rotary_dim <= head_dim && parameters.rotary_dim <= index_dim,
                "rotary_dim exceeds a head dimension");
    qsa_require(projected.key.ne[0] == head_dim && projected.key.ne[2] == tokens &&
                    projected.key.ne[3] == batch_sz,
                "key projection shape mismatch");
    qsa_require(projected.value.ne[0] == head_dim && projected.value.ne[1] == kv_heads &&
                    projected.value.ne[2] == tokens && projected.value.ne[3] == batch_sz,
                "value projection shape mismatch");
    qsa_require(projected.index_key.ne[0] == index_dim &&
                    projected.index_key.ne[2] == tokens &&
                    projected.index_key.ne[3] == batch_sz,
                "index key projection shape mismatch");
    qsa_require(query.ne[0] == head_dim && query.ne[1] == q_heads && query.ne[2] == tokens &&
                    query.ne[3] == batch_sz,
                "query output shape mismatch");
    qsa_require(index_query.ne[0] == index_dim && index_query.ne[1] == index_heads &&
                    index_query.ne[2] == tokens && index_query.ne[3] == batch_sz,
                "index query output shape mismatch");

    qsa_require(parameters.query_norm.numel() == head_dim &&
                    parameters.key_norm.numel() == head_dim,
                "head norm width mismatch");
    qsa_require(parameters.index_query_norm.numel() == index_dim &&
                    parameters.index_key_norm.numel() == index_dim,
                "index norm width mismatch");

    qsa_require_dtype(batch.positions, DType::I32, "positions must be I32");
    qsa_require(batch.positions.data != nullptr && batch.positions.is_contiguous(),
                "positions must be contiguous");
    qsa_require(batch.positions.ne[0] == tokens && batch.positions.ne[2] == batch_sz,
                "positions shape mismatch");
    const std::int32_t axes = batch.positions.ne[1];
    qsa_require(axes == 1 || axes == 3, "positions must carry one or three axes");
    qsa_require(detail::qsa_has_valid_columns(batch.valid_columns) == false ||
                    (batch.valid_columns.dtype == DType::I32 &&
                     batch.valid_columns.is_contiguous() && batch.valid_columns.ne[0] == batch_sz),
                "valid_columns must be I32 [B] or empty");
    qsa_require_dtype(batch.table_rows, DType::I32, "table_rows must be I32");
    qsa_require(batch.table_rows.is_contiguous() && batch.table_rows.ne[0] == batch_sz,
                "table_rows must be contiguous I32 [B]");

    qsa_require(cache.k_pages.ne[0] == head_dim &&
                    cache.k_pages.ne[1] == kQsaTokenPageSize &&
                    cache.k_pages.ne[2] == kv_heads,
                "key cache plane shape mismatch");
    qsa_require(cache.v_pages.ne[0] == head_dim &&
                    cache.v_pages.ne[1] == kQsaTokenPageSize &&
                    cache.v_pages.ne[2] == kv_heads,
                "value cache plane shape mismatch");
    qsa_require(cache.index_k_pages.ne[0] == index_dim &&
                    cache.index_k_pages.ne[1] == kQsaIndexBlockPageSize &&
                    cache.index_k_pages.ne[2] == 1,
                "index key cache plane shape mismatch");
    qsa_require(cache.page_table.ne[0] > 0 && cache.page_table.ne[1] > 0,
                "page table must have positive extents");

    qsa_require_dtype(before.raw_keys, DType::BF16, "tail raw keys must be BF16");
    qsa_require_dtype(before.first_rope, DType::I32, "tail first_rope must be I32");
    qsa_require(before.raw_keys.ne[0] == index_dim && before.raw_keys.ne[1] == kQsaIndexCompress &&
                    before.raw_keys.ne[2] == batch_sz,
                "tail raw key shape mismatch");
    qsa_require(before.first_rope.ne[0] == 3 && before.first_rope.ne[1] == batch_sz,
                "tail first_rope shape mismatch");

    qsa_require_dtype(records.raw_keys, DType::BF16, "record raw keys must be BF16");
    qsa_require_dtype(records.rope_positions, DType::I32, "record rope positions must be I32");
    qsa_require_dtype(records.base_positions, DType::I32, "record base positions must be I32");
    qsa_require(records.raw_keys.ne[0] == index_dim && records.raw_keys.ne[2] == batch_sz &&
                    records.raw_keys.ne[1] >= tokens,
                "record raw key shape mismatch");
    qsa_require(records.rope_positions.ne[0] == 3 && records.rope_positions.ne[1] >= tokens &&
                    records.rope_positions.ne[2] == batch_sz,
                "record rope position shape mismatch");
    qsa_require(records.base_positions.ne[0] == batch_sz,
                "record base position shape mismatch");
    const std::int32_t record_width = records.raw_keys.ne[1];

    qsa_require_contiguous(projected.query, "query projection must be contiguous");
    qsa_require_contiguous(projected.key, "key projection must be contiguous");
    qsa_require_contiguous(projected.value, "value projection must be contiguous");
    qsa_require_contiguous(projected.index_query, "index query projection must be contiguous");
    qsa_require_contiguous(projected.index_key, "index key projection must be contiguous");
    qsa_require_contiguous(query, "query output must be contiguous");
    qsa_require_contiguous(index_query, "index query output must be contiguous");
    qsa_require_contiguous(cache.k_pages, "key cache must be contiguous");
    qsa_require_contiguous(cache.v_pages, "value cache must be contiguous");
    qsa_require_contiguous(cache.index_k_pages, "index key cache must be contiguous");
    qsa_require_contiguous(cache.page_table, "page table must be contiguous");
    qsa_require_contiguous(records.raw_keys, "record raw keys must be contiguous");
    qsa_require_contiguous(records.rope_positions, "record rope positions must be contiguous");
    qsa_require_contiguous(records.base_positions, "record base positions must be contiguous");

    const int logical_pages = cache.page_table.ne[0];

    const auto* projected_query = static_cast<const __nv_bfloat16*>(projected.query.data);
    const auto* projected_key = static_cast<const __nv_bfloat16*>(projected.key.data);
    const auto* projected_value = static_cast<const __nv_bfloat16*>(projected.value.data);
    const auto* projected_index_query =
        static_cast<const __nv_bfloat16*>(projected.index_query.data);
    const auto* projected_index_key = static_cast<const __nv_bfloat16*>(projected.index_key.data);
    const auto* query_norm = static_cast<const __nv_bfloat16*>(parameters.query_norm.data);
    const auto* key_norm = static_cast<const __nv_bfloat16*>(parameters.key_norm.data);
    const auto* index_query_norm = static_cast<const __nv_bfloat16*>(parameters.index_query_norm.data);
    const auto* index_key_norm = static_cast<const __nv_bfloat16*>(parameters.index_key_norm.data);
    const auto* tail_keys = static_cast<const __nv_bfloat16*>(before.raw_keys.data);
    const auto* positions = static_cast<const int*>(batch.positions.data);
    const auto* valid_columns =
        detail::qsa_has_valid_columns(batch.valid_columns)
            ? static_cast<const int*>(batch.valid_columns.data)
            : nullptr;
    const auto* table_rows = static_cast<const int*>(batch.table_rows.data);
    const auto* tail_rope = static_cast<const int*>(before.first_rope.data);
    const auto* page_table = static_cast<const int*>(cache.page_table.data);

    const std::int64_t query_blocks =
        positive_grid(static_cast<std::int64_t>(tokens) * batch_sz * q_heads, "query grid");
    const std::int64_t index_query_blocks =
        positive_grid(static_cast<std::int64_t>(tokens) * batch_sz * index_heads,
                      "index query grid");
    const std::int64_t kv_blocks =
        positive_grid(static_cast<std::int64_t>(tokens) * batch_sz * kv_heads, "key/value grid");
    const std::int64_t block_columns = positive_grid((tokens + 3) / 4 + 1, "index block grid");

    prepare_query_kernel<<<static_cast<unsigned>(query_blocks), 256,
                           2 * head_dim * sizeof(float), stream>>>(
        projected_query, query_norm, positions, valid_columns, tokens, batch_sz, axes, head_dim,
        q_heads, parameters.rms_epsilon, parameters.rotary_dim, parameters.theta,
        static_cast<__nv_bfloat16*>(query.data));

    prepare_query_kernel<<<static_cast<unsigned>(index_query_blocks), 128,
                           2 * index_dim * sizeof(float), stream>>>(
        projected_index_query, index_query_norm, positions, valid_columns, tokens, batch_sz, axes,
        index_dim, index_heads, parameters.rms_epsilon, parameters.rotary_dim, parameters.theta,
        static_cast<__nv_bfloat16*>(index_query.data));

    prepare_key_value_kernel<<<static_cast<unsigned>(kv_blocks), 256,
                               2 * head_dim * sizeof(float), stream>>>(
        projected_key, projected_value, key_norm, positions, valid_columns, table_rows, tokens,
        batch_sz, axes, head_dim, kv_heads, parameters.rms_epsilon, parameters.rotary_dim,
        parameters.theta, page_table, logical_pages,
        static_cast<__nv_bfloat16*>(cache.k_pages.data),
        static_cast<__half*>(cache.v_pages.data));

    prepare_records_kernel<<<dim3(static_cast<unsigned>(tokens), static_cast<unsigned>(batch_sz)),
                             128, 0, stream>>>(
        projected_index_key, positions, valid_columns, tokens, batch_sz, axes, index_dim,
        record_width, static_cast<__nv_bfloat16*>(records.raw_keys.data),
        static_cast<int*>(records.rope_positions.data),
        static_cast<int*>(records.base_positions.data));

    prepare_index_blocks_kernel<<<dim3(static_cast<unsigned>(block_columns),
                                       static_cast<unsigned>(batch_sz)),
                                  128, 2 * index_dim * sizeof(float), stream>>>(
        projected_index_key, index_key_norm, positions, valid_columns, table_rows, tail_keys,
        tail_rope, tokens, batch_sz, axes, index_dim, parameters.rotary_dim,
        parameters.theta, parameters.rms_epsilon, page_table, logical_pages,
        kQsaIndexCompress, static_cast<__nv_bfloat16*>(cache.index_k_pages.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
