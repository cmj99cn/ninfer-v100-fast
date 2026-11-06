#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>

#include "ops/common/activation.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Channels, int QueryRows, int KeyRows, int ValueRows, int StaticWidth, class Publish,
          class Act = __nv_bfloat16>
__global__ void gdn_projected_conv_kernel(
    const Act* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const Act* __restrict__ state_read, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ initial_state_slots, Act* __restrict__ query,
    Act* __restrict__ key, Act* __restrict__ value, std::int32_t width, Publish publish) {
    static_assert(Channels == QueryRows + KeyRows + ValueRows);
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= Channels) { return; }
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    if constexpr (StaticWidth != 0) { width = StaticWidth; }

    std::int32_t valid                 = valid_columns == nullptr ? width : valid_columns[batch];
    valid                              = valid < 0 ? 0 : (valid > width ? width : valid);
    constexpr std::int64_t slot_stride = static_cast<std::int64_t>(Channels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_state_slots[batch]) * slot_stride;
    float s0       = ActivationTraits<Act>::load(state_read + initial_base + row);
    float s1       = ActivationTraits<Act>::load(state_read + initial_base + Channels + row);
    float s2       = ActivationTraits<Act>::load(state_read + initial_base + 2LL * Channels + row);
    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[Channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * Channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * Channels + row]);

    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
        if (token >= valid) {
            if (row < QueryRows) {
                ActivationTraits<Act>::store(query + column * QueryRows + row, 0.0F);
            } else if (row < QueryRows + KeyRows) {
                ActivationTraits<Act>::store(key + column * KeyRows + row - QueryRows, 0.0F);
            } else {
                ActivationTraits<Act>::store(
                    value + column * ValueRows + row - QueryRows - KeyRows, 0.0F);
            }
            continue;
        }

        const float p              = ActivationTraits<Act>::load(projected + column * Channels + row);
        float conv                 = fmaf(w0, s0, 0.0F);
        conv                       = fmaf(w1, s1, conv);
        conv                       = fmaf(w2, s2, conv);
        conv                       = fmaf(w3, p, conv);
        const float output = silu(conv);
        if (row < QueryRows) {
            ActivationTraits<Act>::store(query + column * QueryRows + row, output);
        } else if (row < QueryRows + KeyRows) {
            ActivationTraits<Act>::store(key + column * KeyRows + row - QueryRows, output);
        } else {
            ActivationTraits<Act>::store(
                value + column * ValueRows + row - QueryRows - KeyRows, output);
        }
        publish.publish(token, batch, row, s1, s2, p);
        s0 = s1;
        s1 = s2;
        s2 = p;
    }
}

template <int Channels, int QueryRows, int KeyRows, int ValueRows, class Publish,
          class Act = __nv_bfloat16>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
            const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
            Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    constexpr int kDefaultThreads = 256;
    const std::int32_t width      = projected.ne[1];
    const std::int32_t batch      = projected.ne[2];
    // The narrow-CTA specialization for the single-row W=4 decode shape. It applies to the tp2
    // shard for the same reason it applies to the tp1 geometry -- the kernel is channel-parallel, so
    // the choice is purely "how many channels per CTA" -- and withholding it would have made the
    // shard pay a 256-wide block for 5120 channels on the latency-critical verify path.
    if constexpr (Channels == 10240 || Channels == 5120 || Channels == 2560) {
        if (width == 4 && batch == 1) {
            constexpr int kT4Threads = 64;
            gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 4, Publish, Act>
                <<<(Channels + kT4Threads - 1) / kT4Threads, kT4Threads, 0, stream>>>(
                    static_cast<const Act*>(projected.data),
                    static_cast<const __nv_bfloat16*>(conv_weight.data),
                    static_cast<const Act*>(state_read.data),
                    valid_columns.data == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(initial_state_slots.data),
                    static_cast<Act*>(query.data), static_cast<Act*>(key.data),
                    static_cast<Act*>(value.data), width, publish);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const dim3 grid((Channels + kDefaultThreads - 1) / kDefaultThreads,
                    static_cast<unsigned>(batch));
    gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 0, Publish, Act>
        <<<grid, kDefaultThreads, 0, stream>>>(
            static_cast<const Act*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const Act*>(state_read.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<Act*>(query.data), static_cast<Act*>(key.data),
            static_cast<Act*>(value.data), width, publish);
    CUDA_CHECK(cudaGetLastError());
}

template <class Publish, class Act = __nv_bfloat16>
void dispatch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
              const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
              Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    if (projected.ne[0] == 10240 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 6144) {
        launch<10240, 2048, 2048, 6144, Publish, Act>(projected, conv_weight, state_read,
                                                       valid_columns, initial_state_slots, query,
                                                       key, value, publish, stream);
        return;
    }
    if (projected.ne[0] == 8192 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 4096) {
        launch<8192, 2048, 2048, 4096, Publish, Act>(projected, conv_weight, state_read,
                                                      valid_columns, initial_state_slots, query, key,
                                                      value, publish, stream);
        return;
    }
    // tp == 2 shard of the 27B <10240, 2048, 2048, 6144> geometry. The GDN conv is
    // depthwise, so channel c never mixes with any other channel; device r owns 8 of 16 key heads
    // (Q and K, 1024 channels each) and 24 of 48 value heads (3072 channels), concatenated in the
    // parent's own Q|K|V order -- byte-for-byte the packing `gdn_input_proj_column_parallel`
    // writes and the channel set `gdn/convolution`'s three-block ShardPlan hands this device
    // (bindings.cpp: channels [1024r,+1024) u [2048+1024r,+1024) u [4096+3072r,+3072)).
    if (projected.ne[0] == 5120 && query.ne[0] == 1024 && key.ne[0] == 1024 &&
        value.ne[0] == 3072) {
        launch<5120, 1024, 1024, 3072, Publish, Act>(projected, conv_weight, state_read,
                                                      valid_columns, initial_state_slots, query, key,
                                                      value, publish, stream);
        return;
    }
    // tp == 4 shard of the same geometry: each rank owns 4 of 16 key heads (Q and K, 512 channels
    // each) and 12 of 48 value heads (1536 channels), packed in the parent Q|K|V order.
    if (projected.ne[0] == 2560 && query.ne[0] == 512 && key.ne[0] == 512 && value.ne[0] == 1536) {
        launch<2560, 512, 512, 1536, Publish, Act>(projected, conv_weight, state_read,
                                                    valid_columns, initial_state_slots, query, key,
                                                    value, publish, stream);
        return;
    }
    throw std::invalid_argument("GDN projected-conv received an unregistered geometry");
}

} // namespace

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream) {
    if (projected.dtype == DType::FP16) {
        dispatch<SnapshotHistoryPublishOf<half>, half>(
            projected, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
            value,
            SnapshotHistoryPublishOf<half>{static_cast<half*>(conv_states.data),
                                           static_cast<const std::int32_t*>(
                                               snapshot_base_slots.data),
                                           projected.ne[0]},
            stream);
        return;
    }
    dispatch(projected, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value,
             SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                    static_cast<const std::int32_t*>(snapshot_base_slots.data),
                                    projected.ne[0]},
             stream);
}

void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream) {
    if (conv_record.dtype == DType::FP16) {
        dispatch<NoHistoryPublish, half>(conv_record, conv_weight, conv_states, valid_columns,
                                         initial_state_slots, query, key, value, NoHistoryPublish{},
                                         stream);
        return;
    }
    dispatch(conv_record, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value, NoHistoryPublish{}, stream);
}

} // namespace ninfer::ops::detail
