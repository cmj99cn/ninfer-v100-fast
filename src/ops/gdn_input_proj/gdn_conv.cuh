#pragma once

#include "ops/common/activation.cuh"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Act = __nv_bfloat16>
struct SnapshotHistoryPublishOf {
    Act* state_write;
    const std::int32_t* snapshot_base_slots;
    std::int32_t channels;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float s1, float s2, float p) const {
        const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t base =
            static_cast<std::int64_t>(snapshot_base_slots[batch] + token) * slot_stride;
        ActivationTraits<Act>::store(state_write + base + row, s1);
        ActivationTraits<Act>::store(state_write + base + channels + row, s2);
        ActivationTraits<Act>::store(state_write + base + 2LL * channels + row, p);
    }
};

using SnapshotHistoryPublish = SnapshotHistoryPublishOf<__nv_bfloat16>;

template <class Act = __nv_bfloat16>
struct RecordColumnPublishOf {
    Act* record;
    std::int32_t channels;
    std::int32_t width;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float, float, float p) const {
        const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
        ActivationTraits<Act>::store(record + column * channels + row, p);
    }
};

using RecordColumnPublish = RecordColumnPublishOf<__nv_bfloat16>;

struct NoHistoryPublish {
    __device__ __forceinline__ void publish(std::int32_t, std::int32_t, std::int32_t, float, float,
                                            float) const {}
};

// Device-side implementation detail shared by exact packed projection kernels. Projection
// accumulators stay in the route's existing private precision; Publish changes only the side
// effect after the convolution has consumed that accumulator.
template <class Publish, class Act = __nv_bfloat16>
struct GdnConvEpilogue {
    const __nv_bfloat16* conv_weight;
    const Act* state_read;
    const std::int32_t* initial_slots;
    const std::int32_t* valid_columns;
    Act* query;
    Act* key;
    Act* value;
    std::int32_t channels;
    std::int32_t query_rows;
    std::int32_t key_rows;
    std::int32_t value_rows;
    std::int32_t global_row_offset;
    std::int32_t width;
    std::int32_t batch_row;
    Publish publish;

    template <int Tokens>
    __device__ __forceinline__ void store(std::int32_t local_row,
                                          const float (&projected)[Tokens]) const {
        static_assert(Tokens >= 1);
        const std::int32_t row         = global_row_offset + local_row;
        const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t initial_base =
            static_cast<std::int64_t>(initial_slots[batch_row]) * slot_stride;
        std::int32_t valid = valid_columns == nullptr ? Tokens : valid_columns[batch_row];
        valid              = valid < 0 ? 0 : (valid > Tokens ? Tokens : valid);

        float s0       = ActivationTraits<Act>::load(state_read + initial_base + row);
        float s1 = ActivationTraits<Act>::load(state_read + initial_base + channels + row);
        float s2 = ActivationTraits<Act>::load(state_read + initial_base + 2LL * channels + row);
        const float w0 = __bfloat162float(conv_weight[row]);
        const float w1 = __bfloat162float(conv_weight[channels + row]);
        const float w2 = __bfloat162float(conv_weight[2LL * channels + row]);
        const float w3 = __bfloat162float(conv_weight[3LL * channels + row]);

#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(batch_row) * width + token;
            if (token >= valid) {
                if (row < query_rows) {
                    ActivationTraits<Act>::store(query + column * query_rows + row, 0.0F);
                } else if (row < query_rows + key_rows) {
                    ActivationTraits<Act>::store(key + column * key_rows + row - query_rows, 0.0F);
                } else {
                    ActivationTraits<Act>::store(
                        value + column * value_rows + row - query_rows - key_rows, 0.0F);
                }
                continue;
            }

            const float p              = projected[token];
            float conv                 = fmaf(w0, s0, 0.0F);
            conv                       = fmaf(w1, s1, conv);
            conv                       = fmaf(w2, s2, conv);
            conv                       = fmaf(w3, p, conv);
            const float output = silu(conv);
            if (row < query_rows) {
                ActivationTraits<Act>::store(query + column * query_rows + row, output);
            } else if (row < query_rows + key_rows) {
                ActivationTraits<Act>::store(key + column * key_rows + row - query_rows, output);
            } else {
                ActivationTraits<Act>::store(
                    value + column * value_rows + row - query_rows - key_rows, output);
            }

            publish.publish(token, batch_row, row, s1, s2, p);
            s0 = s1;
            s1 = s2;
            s2 = p;
        }
    }
};

} // namespace ninfer::ops::detail
