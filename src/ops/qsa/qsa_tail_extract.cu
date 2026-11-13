#include "ninfer/ops/qsa.h"

#include "core/device.h"
#include "ops/qsa/qsa_common.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

__global__ void tail_extract_kernel(const __nv_bfloat16* before_keys, const int* before_rope,
                                    const __nv_bfloat16* record_keys, const int* record_rope,
                                    const int* base_positions, const int* valid_columns, int width,
                                    int batch, int index_dim, __nv_bfloat16* after_keys,
                                    int* after_rope) {
    const int b    = blockIdx.x;
    const int base = base_positions[b];
    const int span = valid_columns[b];
    const int end  = base + span;
    const int remainder = end % kQsaIndexCompress;
    const int tail_start = end - remainder;

    for (int c = threadIdx.x; c < index_dim; c += blockDim.x) {
        for (int j = 0; j < kQsaIndexCompress; ++j) {
            __nv_bfloat16 value = __float2bfloat16_rn(0.0F);
            if (j < remainder) {
                const int source = tail_start + j;
                const std::int64_t destination =
                    c + static_cast<std::int64_t>(index_dim) *
                            (j + static_cast<std::int64_t>(kQsaIndexCompress) * b);
                if (source < base) {
                    const int column = source - (base - base % kQsaIndexCompress);
                    value = before_keys[c + static_cast<std::int64_t>(index_dim) *
                                                 (column + static_cast<std::int64_t>(kQsaIndexCompress) * b)];
                } else {
                    const int column = source - base;
                    value = record_keys[c + static_cast<std::int64_t>(index_dim) *
                                                 (column + static_cast<std::int64_t>(width) * b)];
                }
                after_keys[destination] = value;
            } else {
                const std::int64_t destination =
                    c + static_cast<std::int64_t>(index_dim) *
                            (j + static_cast<std::int64_t>(kQsaIndexCompress) * b);
                after_keys[destination] = value;
            }
        }
    }

    if (threadIdx.x < 3) {
        const int axis = threadIdx.x;
        int value      = 0;
        if (remainder != 0) {
            if (tail_start < base) {
                value = before_rope[axis + 3 * b];
            } else {
                value = record_rope[axis + 3 * (tail_start - base + width * b)];
            }
        }
        after_rope[axis + 3 * b] = value;
    }
}

} // namespace

void qsa_tail_extract(const QsaTailStateView& before, const QsaRawRecordReadView& records,
                      const Tensor& base_positions, const Tensor& valid_columns,
                      QsaTailStateView& after, cudaStream_t stream) {
    using detail::qsa_require;
    using detail::qsa_require_contiguous;
    using detail::qsa_require_dtype;

    qsa_require_dtype(before.raw_keys, DType::BF16, "before.raw_keys must be BF16");
    qsa_require_dtype(before.first_rope, DType::I32, "before.first_rope must be I32");
    qsa_require_dtype(records.raw_keys, DType::BF16, "records.raw_keys must be BF16");
    qsa_require_dtype(records.rope_positions, DType::I32, "records.rope_positions must be I32");
    qsa_require_dtype(base_positions, DType::I32, "base_positions must be I32");
    qsa_require_dtype(valid_columns, DType::I32, "valid_columns must be I32");
    qsa_require_dtype(after.raw_keys, DType::BF16, "after.raw_keys must be BF16");
    qsa_require_dtype(after.first_rope, DType::I32, "after.first_rope must be I32");
    qsa_require_contiguous(before.raw_keys, "before.raw_keys must be contiguous");
    qsa_require_contiguous(before.first_rope, "before.first_rope must be contiguous");
    qsa_require_contiguous(records.raw_keys, "records.raw_keys must be contiguous");
    qsa_require_contiguous(records.rope_positions, "records.rope_positions must be contiguous");
    qsa_require_contiguous(base_positions, "base_positions must be contiguous");
    qsa_require_contiguous(valid_columns, "valid_columns must be contiguous");
    qsa_require_contiguous(after.raw_keys, "after.raw_keys must be contiguous");
    qsa_require_contiguous(after.first_rope, "after.first_rope must be contiguous");

    const std::int32_t index_dim = before.raw_keys.ne[0];
    const std::int32_t batch_sz  = before.raw_keys.ne[2];
    const std::int32_t width     = records.raw_keys.ne[1];
    qsa_require(index_dim == 128, "raw_keys must carry 128 channels");
    qsa_require(before.raw_keys.ne[1] == kQsaIndexCompress, "before.raw_keys must have 4 columns");
    qsa_require(batch_sz > 0, "batch must be positive");
    qsa_require(width > 0, "records must have positive width");
    qsa_require(before.first_rope.ne[0] == 3 && before.first_rope.ne[1] == batch_sz,
                "before.first_rope shape mismatch");
    qsa_require(records.raw_keys.ne[0] == index_dim && records.raw_keys.ne[2] == batch_sz,
                "records.raw_keys shape mismatch");
    qsa_require(records.rope_positions.ne[0] == 3 && records.rope_positions.ne[1] == width &&
                    records.rope_positions.ne[2] == batch_sz,
                "records.rope_positions shape mismatch");
    qsa_require(base_positions.ne[0] == batch_sz, "base_positions shape mismatch");
    qsa_require(valid_columns.ne[0] == batch_sz, "valid_columns shape mismatch");
    qsa_require(after.raw_keys.ne[0] == index_dim &&
                    after.raw_keys.ne[1] == kQsaIndexCompress && after.raw_keys.ne[2] == batch_sz,
                "after.raw_keys shape mismatch");
    qsa_require(after.first_rope.ne[0] == 3 && after.first_rope.ne[1] == batch_sz,
                "after.first_rope shape mismatch");

    tail_extract_kernel<<<static_cast<unsigned>(batch_sz), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(before.raw_keys.data),
        static_cast<const int*>(before.first_rope.data),
        static_cast<const __nv_bfloat16*>(records.raw_keys.data),
        static_cast<const int*>(records.rope_positions.data),
        static_cast<const int*>(base_positions.data), static_cast<const int*>(valid_columns.data),
        width, batch_sz, index_dim, static_cast<__nv_bfloat16*>(after.raw_keys.data),
        static_cast<int*>(after.first_rope.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
