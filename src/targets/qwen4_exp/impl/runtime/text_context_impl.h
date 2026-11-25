#pragma once

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sigmoid_gated_rmsnorm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe_nvfp4.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/scatter.h"
#include "targets/qwen4_exp/impl/runtime/activation_dtype.h"
#include "targets/qwen4_exp/impl/runtime/text_context.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef NINFER_QWEN4EXP_LAST_STAGE
#define NINFER_QWEN4EXP_LAST_STAGE(sample, logits) \
    capture_last_stage((logits), stream_for(rank))
#endif

#ifndef NINFER_QWEN4EXP_PLE_PREFETCH_LAUNCH
#define NINFER_QWEN4EXP_PLE_PREFETCH_LAUNCH()
#endif

#ifndef NINFER_QWEN4EXP_PLE_FETCH_BEGIN
#define NINFER_QWEN4EXP_PLE_FETCH_BEGIN(count)
#endif

#ifndef NINFER_QWEN4EXP_PLE_FETCH_END
#define NINFER_QWEN4EXP_PLE_FETCH_END()
#endif

#ifndef NINFER_QWEN4EXP_PLE_PREFETCH_AWAIT
#define NINFER_QWEN4EXP_PLE_PREFETCH_AWAIT()
#endif

#ifndef NINFER_QWEN4EXP_EXPERT_READ_BEGIN
#define NINFER_QWEN4EXP_EXPERT_READ_BEGIN(layer)
#endif

#ifndef NINFER_QWEN4EXP_EXPERT_READ_END
#define NINFER_QWEN4EXP_EXPERT_READ_END(layer)
#endif

#ifndef NINFER_QWEN4EXP_DIAG_MARK
#define NINFER_QWEN4EXP_DIAG_MARK(label)
#endif

#ifndef NINFER_QWEN4EXP_DIAG_OP_BEGIN
#define NINFER_QWEN4EXP_DIAG_OP_BEGIN(name)
#endif

#ifndef NINFER_QWEN4EXP_DIAG_OP_END
#define NINFER_QWEN4EXP_DIAG_OP_END(name)
#endif

#ifndef NINFER_QWEN4EXP_MOE_ENABLED
#define NINFER_QWEN4EXP_MOE_ENABLED() true
#endif

#ifndef NINFER_QWEN4EXP_HAS_PLE
#define NINFER_QWEN4EXP_HAS_PLE() true
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_IDS_SRC
#define NINFER_QWEN4EXP_GRAPH_IDS_SRC(rank, tokens) (tokens.data())
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_POSITIONS_SRC
#define NINFER_QWEN4EXP_GRAPH_POSITIONS_SRC(rank, host_positions) (host_positions.data())
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_SCALARS_SRC
#define NINFER_QWEN4EXP_GRAPH_SCALARS_SRC(rank, value, label) (&(value))
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_WINNER_PTR
#define NINFER_QWEN4EXP_GRAPH_WINNER_PTR() (&token)
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_WINNER_COPY
#define NINFER_QWEN4EXP_GRAPH_WINNER_COPY(dst, src, bytes, stream) \
    CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost))
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_PREPARE_PLE
#define NINFER_QWEN4EXP_GRAPH_PREPARE_PLE(rank, state, tokens) (state).prepare_ple_rows(tokens)
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_PLE_AWAIT
#define NINFER_QWEN4EXP_GRAPH_PLE_AWAIT(rank, state) (state).await_ple_rows()
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_ROUTED_FINISH
#define NINFER_QWEN4EXP_GRAPH_ROUTED_FINISH(weights, stream) \
    Variant::post_mixer_routed_finish((weights), (stream))
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_PREFETCH
#define NINFER_QWEN4EXP_GRAPH_PREFETCH(weights, stream) \
    Variant::post_mixer_prefetch((weights), (stream))
#endif

#ifndef NINFER_QWEN4EXP_GRAPH_FINISH_SYNC
#define NINFER_QWEN4EXP_GRAPH_FINISH_SYNC() \
    CUDA_CHECK(cudaStreamSynchronize(states_[0]->stream()))
#endif

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

inline constexpr std::int32_t kHidden   = TextConfig::hidden;
inline constexpr std::int32_t kStreams  = TextConfig::hc_streams;
inline constexpr std::int32_t kChannels = TextConfig::residual_width;
inline constexpr std::int32_t kHeadDim  = TextConfig::head_dim;
inline constexpr std::int32_t kBatch    = 1;

inline TextState::DeviceScope::DeviceScope(int device) {
    CUDA_CHECK(cudaGetDevice(&previous_));
    if (previous_ != device) { CUDA_CHECK(cudaSetDevice(device)); }
}

inline TextState::DeviceScope::~DeviceScope() {
    if (previous_ != -1) { (void)cudaSetDevice(previous_); }
}

inline TextState::TextState(DeviceContext& device, const ModelView& model,
                            std::uint32_t kv_capacity_tokens, std::uint32_t forward_tokens)
    : TextState(device, model, kv_capacity_tokens, forward_tokens, 1, 0) {}

inline std::int32_t count_qsa_layers(std::size_t layers) {
    std::int32_t count = 0;
    for (std::size_t layer = 0; layer < layers; ++layer) {
        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) { ++count; }
    }
    return count;
}

// One rank's replay records: the same persistent width and row capacity on every rank, with the
// per-rank head/channel shard supplied by TextState. The width is the speculative verify span
// (1 + draft window), independent of the forward/prefill width. The backing is a single allocation
// whose plane offsets come from the shared planner, so no storage is shared between ranks.
inline TextGdnReplayPlan plan_text_gdn_replay_records(std::int32_t conv_channels,
                                                      std::int32_t qk_heads,
                                                      std::int32_t value_heads,
                                                      std::uint32_t replay_width) {
    LayoutBuilder builder;
    const GdnReplayRecordLayout layout = plan_gdn_replay_records(
        builder, GdnReplayRecordSpec{
                     .activation_dtype = kTextActivationDtype,
                     .layers           = TextConfig::gdn_layers(),
                     .record_capacity  = 8,
                     .width = static_cast<std::int32_t>(replay_width == 0 ? 1U : replay_width),
                     .conv_channels    = conv_channels,
                     .qk_heads         = qk_heads,
                     .value_heads      = value_heads,
                     .key_dim          = TextConfig::gdn_head_dim,
                     .value_dim        = TextConfig::gdn_head_dim,
                 });
    return {layout, builder.finish(256)};
}

inline TextState::TextState(DeviceContext& device, const ModelView& model,
                            std::uint32_t kv_capacity_tokens, std::uint32_t forward_tokens,
                            int tp, int rank)
    : TextState(device, model, kv_capacity_tokens, forward_tokens, tp, rank, forward_tokens) {}

inline TextState::TextState(DeviceContext& device, const ModelView& model,
                            std::uint32_t kv_capacity_tokens, std::uint32_t forward_tokens,
                            int tp, int rank, std::uint32_t replay_width)
    : device_scope_(device.device), ple_source_(model.ple.rows), device_(device.device),
      rank_(rank), stream_(device.stream),
      layers_(model.loaded_layers), forward_tokens_(forward_tokens == 0 ? 1 : forward_tokens),
      pages_(kv_capacity_tokens / 64 + 1), kv_heads_(TextConfig::kv_heads / tp),
      gdn_key_heads_(TextConfig::gdn_key_heads / tp),
      gdn_value_heads_(TextConfig::gdn_value_heads / tp),
      conv_channels_(2 * gdn_key_heads_ * TextConfig::gdn_head_dim +
                     gdn_value_heads_ * TextConfig::gdn_head_dim),
      gdn_layers_(layers_), gdn_index_(layers_),
      qsa_layers_(std::max<std::int32_t>(1, count_qsa_layers(layers_))),
      qsa_slot_of_(layers_, -1),
      qsa_page_bytes_(static_cast<std::size_t>(kHeadDim) * 64 * kv_heads_ * pages_ * 2),
      qsa_index_page_bytes_(static_cast<std::size_t>(TextConfig::index_head_dim) * 16 *
                            TextConfig::index_kv_heads * pages_ * 2),
      qsa_tail_bytes_(static_cast<std::size_t>(TextConfig::index_head_dim) * 4 * kBatch * 2),
      qsa_tail_rope_bytes_(3 * kBatch * 4),
      conv_kernels_(layers_ * static_cast<std::size_t>(conv_channels_) * 4 * 2),
      conv_states_(static_cast<std::size_t>(TextConfig::gdn_layers()) *
                   static_cast<std::size_t>(conv_channels_) * 3 * 2),
      recurrent_states_(static_cast<std::size_t>(TextConfig::gdn_layers()) * 128 * 128 *
                        static_cast<std::size_t>(gdn_value_heads_) * 4),
      gdn_replay_plan_(plan_text_gdn_replay_records(conv_channels_, gdn_key_heads_,
                                                   gdn_value_heads_, replay_width)),
      gdn_replay_backing_(gdn_replay_plan_.backing_bytes),
      k_pages_(qsa_page_bytes_ * static_cast<std::size_t>(qsa_layers_)),
      v_pages_(qsa_page_bytes_ * static_cast<std::size_t>(qsa_layers_)),
      index_k_pages_(qsa_index_page_bytes_ * static_cast<std::size_t>(qsa_layers_)),
      page_table_(pages_ * 4),
      tail_raw_keys_(qsa_tail_bytes_ * static_cast<std::size_t>(qsa_layers_)),
      tail_first_rope_(qsa_tail_rope_bytes_ * static_cast<std::size_t>(qsa_layers_)),
      after_raw_keys_(qsa_tail_bytes_),
      after_first_rope_(qsa_tail_rope_bytes_),
      record_raw_keys_(static_cast<std::size_t>(TextConfig::index_head_dim) * forward_tokens_ *
                       kBatch * 2),
      record_rope_positions_(3 * forward_tokens_ * kBatch * 4),
      record_base_positions_(kBatch * 4),
      ple_history_(static_cast<std::size_t>(kChannels) * ops::kPleConvWindow * kBatch * 2),
      ple_rows_(ops::kPleHeads * forward_tokens_ * kBatch * 4),
      ple_staged_rows_(ops::kPleHeads * forward_tokens_ * kBatch * 128),
      ple_divisors_(ops::kPleShards * 4), route_counter_(4) {
    route_counter_.fill(0);
    ple_rows_host_ = std::make_unique<PinnedHostBuffer>(
        static_cast<std::size_t>(ops::kPleHeads) * forward_tokens_ * kBatch * 4);
    ple_staged_host_ = std::make_unique<PinnedHostBuffer>(
        static_cast<std::size_t>(ops::kPleHeads) * forward_tokens_ * kBatch * 128);
    if (tp < 1 || rank < 0 || rank >= tp) {
        throw std::invalid_argument("the text state rank is outside the tensor-parallel group");
    }
    if (TextConfig::kv_heads % tp != 0 || TextConfig::gdn_key_heads % tp != 0 ||
        TextConfig::gdn_value_heads % tp != 0) {
        throw std::invalid_argument("the text state head counts are not divisible by the width");
    }
    if (kv_heads_ < 1 || gdn_key_heads_ < 1 || gdn_value_heads_ % gdn_key_heads_ != 0) {
        throw std::invalid_argument("the text state GDN head split is not a legal group");
    }
    gdn_replay_records_ = GdnReplayRecords(
        DeviceSpan{gdn_replay_backing_.p, gdn_replay_backing_.bytes}, gdn_replay_plan_.layout);
    verification_.width = gdn_replay_records_.spec.width;
    verification_.qsa_raw_keys = DeviceBuffer(
        static_cast<std::size_t>(qsa_layers_) * TextConfig::index_head_dim *
        static_cast<std::size_t>(verification_.width) * kBatch * sizeof(std::uint16_t));
    verification_.qsa_rope = DeviceBuffer(static_cast<std::size_t>(qsa_layers_) * 3 *
                                          static_cast<std::size_t>(verification_.width) * kBatch *
                                          sizeof(std::int32_t));
    verification_.qsa_base = DeviceBuffer(static_cast<std::size_t>(qsa_layers_) * kBatch *
                                          sizeof(std::int32_t));
    verification_.qsa_before_keys = DeviceBuffer(
        static_cast<std::size_t>(qsa_layers_) * TextConfig::index_head_dim * 4 * kBatch *
        sizeof(std::uint16_t));
    verification_.qsa_before_rope = DeviceBuffer(static_cast<std::size_t>(qsa_layers_) * 3 *
                                                  kBatch * sizeof(std::int32_t));
    verification_.ple_values = DeviceBuffer(static_cast<std::size_t>(kChannels) *
                                            static_cast<std::size_t>(verification_.width) * kBatch *
                                            sizeof(std::uint16_t));
    verification_.ple_before = DeviceBuffer(static_cast<std::size_t>(kChannels) *
                                            ops::kPleConvWindow * kBatch *
                                            sizeof(std::uint16_t));
    verification_.residual = DeviceBuffer(static_cast<std::size_t>(verification_.width) *
                                          TextConfig::residual_width * sizeof(std::uint16_t));
    verification_.columns = DeviceBuffer(sizeof(std::int32_t));
    if (ple_source_.fetch == nullptr || ple_source_.divisors == nullptr) {
        throw std::runtime_error("the text state requires a PLE row source");
    }
    ple_controls_ = model.ple.controls;
    const std::array<std::uint64_t, 2> table_shape{TextConfig::ple_shard_rows,
                                                   TextConfig::ple_head_width};
    ple_table_.geometry =
        artifact::row_paged_nf4_geometry(artifact::NumericFormat::NF4_G16_E4M3S, table_shape);
    ple_table_.shard_count = ops::kPleShards;
    ple_table_.divisors    = static_cast<const float*>(ple_divisors_.p);
    ple_divisors_.copy_from_host(ple_source_.divisors, ops::kPleShards * 4);
    const std::int32_t key_width   = gdn_key_heads_ * TextConfig::gdn_head_dim;
    const std::int32_t value_width = gdn_value_heads_ * TextConfig::gdn_head_dim;
    auto* conv_kernel_base = static_cast<std::byte*>(conv_kernels_.p);
    std::size_t gdn = 0;
    std::int32_t qsa = 0;
    for (std::size_t layer_index = 0; layer_index < layers_; ++layer_index) {
        if (TextConfig::is_qsa_layer(static_cast<int>(layer_index))) {
            gdn_index_[layer_index] = std::nullopt;
            qsa_slot_of_[layer_index] = qsa++;
            continue;
        }
        gdn_index_[layer_index] = gdn;
        gdn_layers_[gdn++] = layer_index;
        const Tensor& source = model.layers[layer_index].gdn_projection.convolution;
        if (source.data == nullptr) {
            CUDA_CHECK(cudaMemset(
                conv_kernel_base +
                    layer_index * static_cast<std::size_t>(conv_channels_) * 4 * 2,
                0, static_cast<std::size_t>(conv_channels_) * 4 * 2));
            continue;
        }
        const auto copy_segment = [&](std::int32_t source_column, std::int32_t width,
                                      std::int32_t destination_column) {
            for (std::int32_t tap = 0; tap < TextConfig::gdn_conv_kernel; ++tap) {
                CUDA_CHECK(cudaMemcpy(
                    conv_kernel_base +
                        (layer_index * static_cast<std::size_t>(conv_channels_) * 4 +
                         static_cast<std::size_t>(tap) * conv_channels_ + destination_column) *
                            sizeof(std::uint16_t),
                    static_cast<const std::byte*>(source.data) +
                        (static_cast<std::size_t>(tap) * TextConfig::gdn_conv_channels +
                         source_column) *
                            sizeof(std::uint16_t),
                    static_cast<std::size_t>(width) * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToDevice));
            }
        };
        copy_segment(rank_ * key_width, key_width, 0);
        copy_segment(TextConfig::gdn_key_width + rank_ * key_width, key_width, key_width);
        copy_segment(2 * TextConfig::gdn_key_width + rank_ * value_width, value_width,
                     2 * key_width);
    }
    gdn_layers_.resize(gdn);
    std::vector<std::int32_t> table(pages_);
    for (std::size_t page = 0; page < pages_; ++page) {
        table[page] = static_cast<std::int32_t>(page);
    }
    page_table_.copy_from_host(table.data(), pages_ * 4);
    CUDA_CHECK(cudaStreamCreateWithFlags(&ple_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ple_ready_, cudaEventDisableTiming));
    ple_workers_ = std::make_unique<HostWorkerPool>(1, 2);
    reset();
}

inline TextState::~TextState() {
    DeviceScope scope(device_);
    if (ple_prefetch_.valid()) { ple_prefetch_.wait(); }
    if (ple_ready_ != nullptr) { (void)cudaEventDestroy(ple_ready_); }
    if (ple_stream_ != nullptr) { (void)cudaStreamDestroy(ple_stream_); }
}

inline void TextState::reset() {
    DeviceScope scope(device_);
    if (ple_prefetch_.valid()) { ple_prefetch_.wait(); }
    if (ple_ready_ != nullptr) { (void)cudaEventSynchronize(ple_ready_); }
    conv_states_.fill(0);
    recurrent_states_.fill(0);
    gdn_replay_backing_.fill(0);
    verification_.qsa_raw_keys.fill(0);
    verification_.qsa_rope.fill(0);
    verification_.qsa_base.fill(0);
    verification_.qsa_before_keys.fill(0);
    verification_.qsa_before_rope.fill(0);
    verification_.ple_values.fill(0);
    verification_.ple_before.fill(0);
    verification_.residual.fill(0);
    verification_.columns.fill(0);
    k_pages_.fill(0);
    v_pages_.fill(0);
    index_k_pages_.fill(0);
    tail_raw_keys_.fill(0);
    tail_first_rope_.fill(0);
    after_raw_keys_.fill(0);
    after_first_rope_.fill(0);
    record_raw_keys_.fill(0);
    record_rope_positions_.fill(0);
    record_base_positions_.fill(0);
    ple_history_.fill(0);
    ple_rows_.fill(0);
    ple_staged_rows_.fill(0);
    {
        std::vector<std::int32_t> table(pages_);
        for (std::size_t page = 0; page < pages_; ++page) {
            table[page] = static_cast<std::int32_t>(page);
        }
        page_table_.copy_from_host(table.data(), pages_ * 4);
    }
    next_position_ = 0;
    ple_tail_[0]   = TextConfig::eos_token_id;
    ple_tail_[1]   = TextConfig::eos_token_id;
    std::memset(ple_rows_host_->data(), 0, ple_rows_host_->size());
    std::memset(ple_staged_host_->data(), 0, ple_staged_host_->size());
}

inline void TextState::prepare_ple_rows(std::span<const TokenId> tokens) {
    DeviceScope scope(device_);
    const std::size_t count = tokens.size();
    const std::size_t row_bytes = static_cast<std::size_t>(ple_table_.geometry.row_bytes);
    if (count == 0) { return; }
    if (count > forward_tokens_) { throw std::runtime_error("the PLE row round exceeds the forward width"); }
    std::vector<std::uint64_t> ids(count);
    for (std::size_t index = 0; index < count; ++index) {
        ids[index] = static_cast<std::uint64_t>(tokens[index]);
    }
    std::vector<std::uint32_t> rows(ops::kPleHeads * count);
    ops::ple_hash_tokens(ids, {ple_tail_[0], ple_tail_[1]}, TextConfig::eos_token_id,
                         ple_controls_, rows);
    std::vector<ops::PleRowAddress> addresses(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        addresses[index] = ops::ple_row_address(ple_table_, rows[index]);
    }
    const std::size_t staged_bytes = rows.size() * row_bytes;
    if (staged_bytes > ple_staged_host_->size() || rows.size() * 4 > ple_rows_host_->size()) {
        throw std::runtime_error("the PLE row round exceeds its pinned staging capacity");
    }
    for (const TokenId token : tokens) { advance_ple_tail(token); }
    if (ple_prefetch_.valid()) { ple_prefetch_.get(); }
    NINFER_QWEN4EXP_PLE_PREFETCH_LAUNCH();
    ple_prefetch_ = ple_workers_->submit(
        [this, rows = std::move(rows), addresses = std::move(addresses), staged_bytes] {
            CUDA_CHECK(cudaSetDevice(device_));
            CUDA_CHECK(cudaEventSynchronize(ple_ready_));
            NINFER_QWEN4EXP_PLE_FETCH_BEGIN(addresses.size());
            ple_source_.fetch(ple_source_.context, addresses.data(), addresses.size(),
                              static_cast<std::byte*>(ple_staged_host_->data()));
            NINFER_QWEN4EXP_PLE_FETCH_END();
            std::memcpy(ple_rows_host_->data(), rows.data(), rows.size() * 4);
            CUDA_CHECK(cudaMemcpyAsync(ple_rows_.p, ple_rows_host_->data(), rows.size() * 4,
                                       cudaMemcpyHostToDevice, ple_stream_));
            CUDA_CHECK(cudaMemcpyAsync(ple_staged_rows_.p, ple_staged_host_->data(), staged_bytes,
                                       cudaMemcpyHostToDevice, ple_stream_));
            CUDA_CHECK(cudaEventRecord(ple_ready_, ple_stream_));
        });
}

inline void TextState::await_ple_rows() {
    DeviceScope scope(device_);
    if (ple_prefetch_.valid()) { ple_prefetch_.get(); }
    enqueue_ple_wait();
    NINFER_QWEN4EXP_PLE_PREFETCH_AWAIT();
}

inline void TextState::enqueue_ple_wait() {
    DeviceScope scope(device_);
    CUDA_CHECK(cudaStreamWaitEvent(stream_, ple_ready_, 0));
}

inline std::int32_t TextState::base_position() const {
    DeviceScope scope(device_);
    std::int32_t base = 0;
    record_base_positions_.copy_to_host(&base, 4);
    return base;
}

inline Tensor TextState::gdn_conv_state(std::size_t gdn_index) const {
    return Tensor(conv_states_.p + gdn_index * static_cast<std::size_t>(conv_channels_) * 3 * 2,
                  kTextActivationDtype, {3, conv_channels_});
}

inline Tensor TextState::gdn_conv_kernel(std::size_t layer) const {
    return Tensor(conv_kernels_.p +
                      layer * static_cast<std::size_t>(conv_channels_) * 4 * 2,
                  DType::BF16, {conv_channels_, 4});
}

inline Tensor TextState::gdn_recurrent_state(std::size_t gdn_index) const {
    return Tensor(recurrent_states_.p +
                      gdn_index * 128 * 128 * static_cast<std::size_t>(gdn_value_heads_) * 4,
                  DType::FP32, {128, 128, gdn_value_heads_});
}

inline const GdnReplayRecords& TextState::gdn_replay_records() const noexcept {
    return gdn_replay_records_;
}

inline LinearAttentionStateAllLayersView TextState::gdn_all_layers_view() const {
    constexpr std::int32_t kStateDim = 128;
    LinearAttentionStateAllLayersView view;
    view.conv_layer0 = Tensor(conv_states_.p, kTextActivationDtype, {3, conv_channels_, 1});
    view.recurrent_layer0 =
        Tensor(recurrent_states_.p, DType::FP32, {kStateDim, kStateDim, gdn_value_heads_, 1});
    view.conv_layer_stride_bytes = static_cast<std::int64_t>(conv_channels_) * 3 *
                                   static_cast<std::int64_t>(sizeof(std::uint16_t));
    view.recurrent_layer_stride_bytes = static_cast<std::int64_t>(kStateDim) * kStateDim *
                                        gdn_value_heads_ * static_cast<std::int64_t>(sizeof(float));
    view.spec = LinearAttentionStatePoolSpec{
        .layers         = static_cast<std::uint32_t>(TextConfig::gdn_layers()),
        .conv_channels  = conv_channels_,
        .conv_width     = 3,
        .value_heads    = gdn_value_heads_,
        .value_head_dim = kStateDim,
        .key_head_dim   = kStateDim,
        .slot_count     = 1,
        .conv_dtype     = kTextActivationDtype,
    };
    return view;
}

inline void TextState::capture_verification_state() {
    DeviceScope scope(device_);
    for (int slot = 0; slot < qsa_layers_; ++slot) {
        const ops::ConstQsaTailStateView tail = const_tail(slot);
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::byte*>(verification_.qsa_before_keys.p) +
                static_cast<std::size_t>(slot) * qsa_tail_bytes_,
            tail.raw_keys.data, tail.raw_keys.bytes(), cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::byte*>(verification_.qsa_before_rope.p) +
                static_cast<std::size_t>(slot) * qsa_tail_rope_bytes_,
            tail.first_rope.data, tail.first_rope.bytes(), cudaMemcpyDeviceToDevice, stream_));
    }
    const Tensor history = ple_history();
    CUDA_CHECK(cudaMemcpyAsync(verification_.ple_before.p, history.data, history.bytes(),
                               cudaMemcpyDeviceToDevice, stream_));
}

inline void TextState::set_verification_columns(std::int32_t columns, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(verification_.columns.p, &columns, sizeof(columns),
                               cudaMemcpyHostToDevice, stream));
}

inline ops::QsaRawRecordWriteView TextState::verification_records(std::int32_t slot,
                                                                   std::size_t entries) const {
    const std::size_t keys = static_cast<std::size_t>(slot) * TextConfig::index_head_dim *
                             static_cast<std::size_t>(verification_.width) * kBatch * sizeof(std::uint16_t);
    const std::size_t rope = static_cast<std::size_t>(slot) * 3 *
                             static_cast<std::size_t>(verification_.width) * kBatch * sizeof(std::int32_t);
    const std::size_t base = static_cast<std::size_t>(slot) * kBatch * sizeof(std::int32_t);
    return ops::QsaRawRecordWriteView{
        .raw_keys = Tensor(static_cast<std::byte*>(verification_.qsa_raw_keys.p) + keys,
                           DType::BF16,
                           {TextConfig::index_head_dim, static_cast<std::int32_t>(entries), kBatch}),
        .rope_positions = Tensor(static_cast<std::byte*>(verification_.qsa_rope.p) + rope,
                                 DType::I32,
                                 {3, static_cast<std::int32_t>(entries), kBatch}),
        .base_positions = Tensor(static_cast<std::byte*>(verification_.qsa_base.p) + base,
                                 DType::I32, {kBatch}),
    };
}

inline ops::QsaRawRecordReadView TextState::verification_read_records(std::int32_t slot,
                                                                      std::size_t entries) const {
    const auto write = verification_records(slot, entries);
    return ops::QsaRawRecordReadView{write.raw_keys, write.rope_positions};
}

inline Tensor TextState::verification_base_positions(std::int32_t slot) const {
    return Tensor(static_cast<std::byte*>(verification_.qsa_base.p) +
                      static_cast<std::size_t>(slot) * kBatch * sizeof(std::int32_t),
                  DType::I32, {kBatch});
}

inline ops::ConstQsaTailStateView TextState::verification_before(std::int32_t slot) const {
    return ops::ConstQsaTailStateView{
        .raw_keys = Tensor(static_cast<std::byte*>(verification_.qsa_before_keys.p) +
                               static_cast<std::size_t>(slot) * qsa_tail_bytes_,
                           DType::BF16,
                           {TextConfig::index_head_dim, 4, kBatch}),
        .first_rope = Tensor(static_cast<std::byte*>(verification_.qsa_before_rope.p) +
                                 static_cast<std::size_t>(slot) * qsa_tail_rope_bytes_,
                             DType::I32, {3, kBatch}),
    };
}

inline Tensor TextState::verification_ple_values(std::size_t entries) const {
    return Tensor(verification_.ple_values.p, kTextActivationDtype,
                  {kChannels, static_cast<std::int32_t>(entries), kBatch});
}

inline Tensor TextState::verification_ple_before() const {
    return Tensor(verification_.ple_before.p, kTextActivationDtype,
                  {kChannels, static_cast<std::int32_t>(ops::kPleConvWindow), kBatch});
}

inline Tensor TextState::verification_residual(std::size_t token) const {
    return Tensor(static_cast<std::byte*>(verification_.residual.p) +
                      token * static_cast<std::size_t>(TextConfig::residual_width) *
                          sizeof(std::uint16_t),
                  kTextActivationDtype, {TextConfig::residual_width, 1});
}

inline Tensor TextState::verification_residual_view(int entries) const {
    return Tensor(verification_.residual.p, kTextActivationDtype,
                  {TextConfig::residual_width, entries, kBatch});
}

inline Tensor TextState::verification_columns() const {
    return Tensor(verification_.columns.p, DType::I32, {kBatch});
}

inline std::optional<std::size_t> TextState::gdn_state_index(std::size_t layer) const {
    return gdn_index_.at(layer);
}

inline std::size_t TextState::gdn_state_entries() const noexcept { return gdn_layers_.size(); }

inline std::uint64_t TextState::kv_payload_bytes() const noexcept {
    return static_cast<std::uint64_t>(k_pages_.bytes) + v_pages_.bytes + index_k_pages_.bytes +
           page_table_.bytes + tail_raw_keys_.bytes + tail_first_rope_.bytes +
           after_raw_keys_.bytes + after_first_rope_.bytes + record_raw_keys_.bytes +
           record_rope_positions_.bytes + record_base_positions_.bytes;
}

inline std::uint64_t TextState::gdn_state_bytes() const noexcept {
    return static_cast<std::uint64_t>(conv_kernels_.bytes) + conv_states_.bytes +
           recurrent_states_.bytes;
}

inline std::uint64_t TextState::reserved_bytes() const noexcept {
    return kv_payload_bytes() + gdn_state_bytes() + gdn_replay_backing_.bytes +
           verification_.qsa_raw_keys.bytes + verification_.qsa_rope.bytes +
           verification_.qsa_base.bytes + verification_.qsa_before_keys.bytes +
           verification_.qsa_before_rope.bytes + verification_.ple_values.bytes +
           verification_.ple_before.bytes + verification_.residual.bytes +
           verification_.columns.bytes + ple_history_.bytes + ple_rows_.bytes +
           ple_staged_rows_.bytes + ple_divisors_.bytes;
}

inline std::int32_t TextState::qsa_slot(std::size_t layer) const {
    if (layer >= qsa_slot_of_.size() || qsa_slot_of_[layer] < 0) {
        throw std::invalid_argument("the requested layer is not a QSA layer");
    }
    return qsa_slot_of_[layer];
}

inline ops::QsaCacheView TextState::cache(std::int32_t slot) const {
    const std::int32_t pages = static_cast<std::int32_t>(pages_);
    const std::size_t base = static_cast<std::size_t>(slot) * qsa_page_bytes_;
    const std::size_t index_base = static_cast<std::size_t>(slot) * qsa_index_page_bytes_;
    return ops::QsaCacheView{
        .k_pages       = Tensor(static_cast<std::byte*>(k_pages_.p) + base, DType::BF16,
                                {kHeadDim, 64, kv_heads_, pages}),
        .v_pages       = Tensor(static_cast<std::byte*>(v_pages_.p) + base, DType::FP16,
                                {kHeadDim, 64, kv_heads_, pages}),
        .index_k_pages = Tensor(static_cast<std::byte*>(index_k_pages_.p) + index_base,
                                DType::BF16,
                                {TextConfig::index_head_dim, 16, TextConfig::index_kv_heads, pages}),
        .page_table    = Tensor(page_table_.p, DType::I32, {pages, kBatch}),
    };
}

inline ops::QsaTailStateView TextState::tail(std::int32_t) const {
    return ops::QsaTailStateView{
        .raw_keys   = Tensor(after_raw_keys_.p, DType::BF16, {TextConfig::index_head_dim, 4, kBatch}),
        .first_rope = Tensor(after_first_rope_.p, DType::I32, {3, kBatch}),
    };
}

inline ops::ConstQsaTailStateView TextState::const_tail(std::int32_t slot) const {
    const std::size_t base = static_cast<std::size_t>(slot) * qsa_tail_bytes_;
    const std::size_t rope_base = static_cast<std::size_t>(slot) * qsa_tail_rope_bytes_;
    return ops::ConstQsaTailStateView{
        .raw_keys   = Tensor(static_cast<std::byte*>(tail_raw_keys_.p) + base, DType::BF16,
                             {TextConfig::index_head_dim, 4, kBatch}),
        .first_rope = Tensor(static_cast<std::byte*>(tail_first_rope_.p) + rope_base, DType::I32,
                             {3, kBatch}),
    };
}

inline ops::QsaTailStateView TextState::before_tail(std::int32_t slot) const {
    const std::size_t base = static_cast<std::size_t>(slot) * qsa_tail_bytes_;
    const std::size_t rope_base = static_cast<std::size_t>(slot) * qsa_tail_rope_bytes_;
    return ops::QsaTailStateView{
        .raw_keys   = Tensor(static_cast<std::byte*>(tail_raw_keys_.p) + base, DType::BF16,
                             {TextConfig::index_head_dim, 4, kBatch}),
        .first_rope = Tensor(static_cast<std::byte*>(tail_first_rope_.p) + rope_base, DType::I32,
                             {3, kBatch}),
    };
}

inline ops::QsaRawRecordWriteView TextState::records(std::size_t entries) const {
    const std::int32_t width = static_cast<std::int32_t>(entries);
    return ops::QsaRawRecordWriteView{
        .raw_keys       = Tensor(record_raw_keys_.p, DType::BF16,
                                 {TextConfig::index_head_dim, width, kBatch}),
        .rope_positions = Tensor(record_rope_positions_.p, DType::I32, {3, width, kBatch}),
        .base_positions = Tensor(record_base_positions_.p, DType::I32, {kBatch}),
    };
}

inline Tensor TextState::ple_history() const {
    return Tensor(ple_history_.p, kTextActivationDtype,
                  {kChannels, static_cast<std::int32_t>(ops::kPleConvWindow), kBatch});
}

inline const std::uint32_t* TextState::ple_rows() const noexcept {
    return static_cast<const std::uint32_t*>(ple_rows_.p);
}

inline const std::uint8_t* TextState::ple_staged_rows() const noexcept {
    return static_cast<const std::uint8_t*>(ple_staged_rows_.p);
}

inline const ops::PleNf4Table& TextState::ple_table() const noexcept { return ple_table_; }

inline void TextState::publish_tail(std::int32_t slot) {
    DeviceScope scope(device_);
    const std::size_t base = static_cast<std::size_t>(slot) * qsa_tail_bytes_;
    const std::size_t rope_base = static_cast<std::size_t>(slot) * qsa_tail_rope_bytes_;
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(tail_raw_keys_.p) + base,
                               after_raw_keys_.p, qsa_tail_bytes_, cudaMemcpyDeviceToDevice,
                               stream_));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(tail_first_rope_.p) + rope_base,
                               after_first_rope_.p, qsa_tail_rope_bytes_, cudaMemcpyDeviceToDevice,
                               stream_));
}

inline TextTrunk::TextTrunk(const ModelView& model, TextState& state, WorkspaceArena& activations,
                            WorkspaceArena& workspace, cudaStream_t stream)
    : execution_(nullptr), comms_(nullptr), tp_(1), stream_(stream) {
    models_[0]      = &model;
    states_[0]      = &state;
    activations_[0] = &activations;
    workspaces_[0]  = &workspace;
}

inline TextTrunk::TextTrunk(const std::array<const ModelView*, kMaximumDevices>& models,
                            const std::array<TextState*, kMaximumDevices>& states,
                            const std::array<WorkspaceArena*, kMaximumDevices>& activations,
                            const std::array<WorkspaceArena*, kMaximumDevices>& workspaces,
                            ExecutionContext& execution, const ops::CollectiveComms& comms)
    : models_(models), states_(states), activations_(activations), workspaces_(workspaces),
      execution_(&execution), comms_(&comms), tp_(execution.tp),
      stream_(execution.primary().stream) {
    if (tp_ < 1 || tp_ > static_cast<int>(kMaximumDevices)) {
        throw std::invalid_argument("the text trunk tensor-parallel width is invalid");
    }
    for (int rank = 0; rank < tp_; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (models_[slot] == nullptr || states_[slot] == nullptr || activations_[slot] == nullptr ||
            workspaces_[slot] == nullptr) {
            throw std::invalid_argument("the text trunk is missing a per-rank context");
        }
    }
}

inline void TextTrunk::row_parallel_linear(
    const std::array<Tensor, kMaximumDevices>& x, const std::array<Weight, kMaximumDevices>& w,
    const std::array<Tensor, kMaximumDevices>& out,
    const std::array<Tensor, kMaximumDevices>& staging,
    const std::array<WorkspaceArena*, kMaximumDevices>& workspaces) const {
    if (tp_ <= 1) {
        Tensor destination = out[0];
        ops::linear(x[0], w[0], destination, ops::LinearPolicy::A16Only, *workspaces[0],
                    stream_for(0));
        return;
    }
    ops::linear_row_parallel(x, w, out, staging, ops::LinearPolicy::A16Only, workspaces,
                             *execution_, *comms_);
}

inline void TextTrunk::reduce_rows(const std::array<Tensor, kMaximumDevices>& buffer,
                                   const std::array<Tensor, kMaximumDevices>& staging) const {
    if (tp_ <= 1) { return; }
    ops::allreduce_sum(buffer, staging, *execution_, *comms_);
}

inline void TextTrunk::set_residual_sink(int rank, void* data, std::size_t bytes) noexcept {
    if (rank < 0 || rank >= static_cast<int>(kMaximumDevices)) { return; }
    residual_sinks_[static_cast<std::size_t>(rank)]       = data;
    residual_sink_bytes_[static_cast<std::size_t>(rank)] = bytes;
}

inline void TextTrunk::set_gdn_state_action(GdnStateAction action) noexcept {
    gdn_state_action_ = action;
}

inline TextTrunk::VerificationBaseline TextTrunk::capture_verification_baseline() {
    VerificationBaseline baseline;
    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        baseline.positions[rank] = states_[slot]->next_position();
        baseline.ple_tails[rank] = {states_[slot]->ple_tail(0), states_[slot]->ple_tail(1)};
        states_[slot]->capture_verification_state();
    });
    return baseline;
}

inline void TextTrunk::commit_verified_span(std::span<const TokenId> verify_tokens,
                                             std::uint32_t committed_count,
                                             const VerificationBaseline& baseline) {

    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        TextState& state   = *states_[slot];
        const ops::GdnReplayFoldRow row{
            .linear_state_slot = 0,
            .commit_columns    = static_cast<std::int32_t>(committed_count),
        };
        ops::gdn_replay_fold(state.gdn_replay_records(), state.gdn_all_layers_view(),
                             std::span<const ops::GdnReplayFoldRow>(&row, 1), state.stream());
    });

    if (committed_count < verify_tokens.size()) {
        for_each_rank([&](int rank) {
            const auto slot = static_cast<std::size_t>(rank);
            TextState& state = *states_[slot];
            state.set_verification_columns(static_cast<std::int32_t>(committed_count),
                                           state.stream());
            for (int qsa = 0; qsa < TextConfig::qsa_layers(); ++qsa) {
                const ops::QsaRawRecordReadView records =
                    state.verification_read_records(qsa, verify_tokens.size());
                ops::QsaTailStateView after = state.tail(0);
                ops::qsa_tail_extract(state.verification_before(qsa), records,
                                      state.verification_base_positions(qsa),
                                      state.verification_columns(), after, state.stream());
                state.publish_tail(qsa);
            }
            Tensor after_history = state.ple_history();
            ops::ple_state_extract(
                state.verification_ple_before(),
                state.verification_ple_values(verify_tokens.size()),
                state.verification_columns(), after_history, state.stream());
            const std::uint64_t previous =
                committed_count == 1 ? baseline.ple_tails[rank][1]
                                     : verify_tokens[committed_count - 2];
            const std::uint64_t current =
                committed_count == 1 ? verify_tokens[0] : verify_tokens[committed_count - 1];
            state.rewind_scalars(baseline.positions[rank] + committed_count, previous, current);
        });
    }

    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (residual_sinks_[slot] == nullptr) { return; }
        TextState& state = *states_[slot];
        const Tensor record = state.verification_residual(committed_count - 1);
        CUDA_CHECK(cudaMemcpyAsync(residual_sinks_[slot], record.data, record.bytes(),
                                   cudaMemcpyDeviceToDevice, state.stream()));
    });
}

inline TokenId TextTrunk::prefill(std::span<const TokenId> tokens) { return forward(tokens); }

inline TokenId TextTrunk::decode(TokenId token) {
    return forward(std::span<const TokenId>{&token, 1});
}

inline std::vector<TokenId> TextTrunk::decode_span(std::span<const TokenId> tokens) {
    std::vector<std::int32_t> winners(tokens.size());
    forward(tokens, std::span<std::int32_t>{winners});
    std::vector<TokenId> result(tokens.size());
    for (std::size_t index = 0; index < winners.size(); ++index) {
        result[index] = static_cast<TokenId>(winners[index]);
    }
    return result;
}

inline void TextTrunk::run_gdn(std::size_t layer, std::int32_t T,
                               const std::array<Tensor, kMaximumDevices>& block_input,
                               std::array<Tensor, kMaximumDevices>& mixer_out,
                               const std::array<Tensor, kMaximumDevices>& staging) {
    std::array<Tensor, kMaximumDevices> projected{};
    std::array<Weight, kMaximumDevices> output_weight{};
    std::array<WorkspaceArena*, kMaximumDevices> arena_ptrs{};
    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        WorkspaceArena& workspace_ = *workspaces_[slot];
        const auto& weights = models_[slot]->layers[layer].gdn_projection;
        TextState& state    = *states_[slot];
        const std::int32_t key_width   = state.gdn_key_heads() * TextConfig::gdn_head_dim;
        const std::int32_t value_width = state.gdn_value_heads() * TextConfig::gdn_head_dim;
        const std::int32_t conv_channels = state.conv_channels();
        const bool record_for_replay = gdn_state_action_ == GdnStateAction::RecordForReplay;
        const std::optional<std::size_t> state_slot = state.gdn_state_index(layer);
        if (!state_slot.has_value()) { throw std::logic_error("a QSA layer has no GDN state slot"); }
        const std::int32_t gdn_index = static_cast<std::int32_t>(*state_slot);
        // The replay Op's execution domain is T in [2,16]; a one-token forward cannot be a
        // speculative verify span and must stay on the in-place commit path.
        if (record_for_replay && (T < 2 || T > state.gdn_replay_records().spec.width)) {
            throw std::logic_error("GDN replay records require a multi-token span within the record width");
        }
        std::optional<GdnReplayRecordLayer> record_layer;
        if (record_for_replay) {
            // Each rank records only its own head/channel shard into its own backing.
            record_layer = state.gdn_replay_records().layer(gdn_index, kBatch);
        }

        Tensor q    = activations_[slot]->alloc(kTextActivationDtype, {key_width, T});
        Tensor k    = activations_[slot]->alloc(kTextActivationDtype, {key_width, T});
        Tensor v    = activations_[slot]->alloc(kTextActivationDtype, {value_width, T});
        Tensor z    = activations_[slot]->alloc(kTextActivationDtype, {value_width, T});
        Tensor g    = activations_[slot]->alloc(DType::FP32, {TextConfig::gdn_value_heads, T});
        Tensor beta = activations_[slot]->alloc(DType::FP32, {TextConfig::gdn_value_heads, T});
        Tensor g_rank    = activations_[slot]->alloc(DType::FP32, {state.gdn_value_heads(), T});
        Tensor beta_rank = activations_[slot]->alloc(DType::FP32, {state.gdn_value_heads(), T});
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("gdn_proj");
        Variant::gdn_projection_control(block_input[slot], weights, q, k, v, z, g, beta, workspace_,
                                        stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("gdn_proj");
        NINFER_QWEN4EXP_DIAG_MARK("gdn_proj");

        Tensor conv_in  = activations_[slot]->alloc(kTextActivationDtype, {conv_channels, T});
        Tensor conv_out = activations_[slot]->alloc(kTextActivationDtype, {conv_channels, T});
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("gdn_stage");
        if (T == 1) {
            ops::stage_gdn_decode(q, k, v, g, beta, rank, conv_in, g_rank, beta_rank,
                                  stream_for(rank));
        } else {
            ops::stage_gdn_span(q, k, v, g, beta, rank, conv_in, g_rank, beta_rank,
                                stream_for(rank));
        }
        NINFER_QWEN4EXP_DIAG_OP_END("gdn_stage");

        if (record_layer.has_value()) {
            // conv_in is [channels,T] and the record plane is [channels,width,rows]; copy the
            // valid T columns bit-preserving into this rank's row-0 slice.
            CUDA_CHECK(cudaMemcpyAsync(record_layer->conv.data, conv_in.data, conv_in.bytes(),
                                       cudaMemcpyDeviceToDevice, stream_for(rank)));
        }

        Tensor conv_state = state.gdn_conv_state(gdn_index);
        Tensor conv_state_update = conv_state;
        if (record_layer.has_value()) {
            // The convolution writes its state; stage a scratch copy of the persistent history so
            // the verify round computes the same output without committing the update.
            conv_state_update = activations_[slot]->alloc(kTextActivationDtype, {3, conv_channels});
            CUDA_CHECK(cudaMemcpyAsync(conv_state_update.data, conv_state.data, conv_state.bytes(),
                                       cudaMemcpyDeviceToDevice, stream_for(rank)));
        }
        const Tensor conv_kernel = state.gdn_conv_kernel(layer);
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("gdn_conv");
        ops::causal_conv1d_silu(conv_in, conv_kernel, conv_state_update, conv_state_update,
                                conv_out, stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("gdn_conv");
        NINFER_QWEN4EXP_DIAG_MARK("gdn_conv");

        NINFER_QWEN4EXP_DIAG_OP_BEGIN("gdn_extract");
        ops::extract_gdn_columns(conv_out, q, k, v, stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("gdn_extract");

        Tensor o  = activations_[slot]->alloc(kTextActivationDtype, {value_width, T});
        Tensor on = activations_[slot]->alloc(kTextActivationDtype, {value_width, T});
        Tensor q_recurrent = q.view({128, state.gdn_key_heads(), T});
        Tensor k_recurrent = k.view({128, state.gdn_key_heads(), T});
        Tensor v_recurrent = v.view({128, state.gdn_value_heads(), T});
        Tensor o_recurrent = o.view({128, state.gdn_value_heads(), T});
        Tensor o_normalized = on.view({128, state.gdn_value_heads(), T});
        Tensor recurrent_state = state.gdn_recurrent_state(gdn_index);
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("gdn_recurrent");
        if (record_layer.has_value()) {
            Tensor valid_columns = activations_[slot]->alloc(DType::I32, {kBatch});
            Tensor initial_slots = activations_[slot]->alloc(DType::I32, {kBatch});
            CUDA_CHECK(cudaMemcpyAsync(
                valid_columns.data,
                NINFER_QWEN4EXP_GRAPH_SCALARS_SRC(rank, T, "gdn_record_columns"), 4,
                cudaMemcpyHostToDevice, stream_for(rank)));
            CUDA_CHECK(cudaMemsetAsync(initial_slots.data, 0, 4, stream_for(rank)));
            Tensor q_record = q.view({128, state.gdn_key_heads(), T, kBatch});
            Tensor k_record = k.view({128, state.gdn_key_heads(), T, kBatch});
            Tensor v_record = v.view({128, state.gdn_value_heads(), T, kBatch});
            Tensor g_record = g_rank.view({state.gdn_value_heads(), T, kBatch});
            Tensor beta_record = beta_rank.view({state.gdn_value_heads(), T, kBatch});
            Tensor key_record   = record_layer->key.slice(2, 0, T);
            Tensor value_record = record_layer->value.slice(2, 0, T);
            Tensor gate_record  = record_layer->gate.slice(2, 0, T);
            Tensor record_out   = o.view({128, state.gdn_value_heads(), T, kBatch});
            // The replay Op reads the persistent state without writing it, records this rank's
            // raw k/v/{g,beta} for a later fold, and still produces o_recurrent for the norm.
            ops::gated_delta_net_replay_record(
                q_record, k_record, v_record, g_record, beta_record, kGdnScale,
                recurrent_state.view({128, 128, state.gdn_value_heads(), 1}), valid_columns,
                initial_slots, key_record, value_record, gate_record, record_out, stream_for(rank));
        } else if (T == 1) {
            // The one-token decode must share the record/fold recurrence body so a folded
            // continuation reproduces the sequential state bit-for-bit. The slot tensor is a
            // transient arena allocation: its address is replay-stable under graph capture and it
            // introduces no persistent state.
            Tensor slots = activations_[slot]->alloc(DType::I32, {kBatch});
            CUDA_CHECK(cudaMemsetAsync(slots.data, 0, slots.bytes(), stream_for(rank)));
            Tensor q_batch    = q.view({128, state.gdn_key_heads(), T, kBatch});
            Tensor k_batch    = k.view({128, state.gdn_key_heads(), T, kBatch});
            Tensor v_batch    = v.view({128, state.gdn_value_heads(), T, kBatch});
            Tensor g_batch    = g_rank.view({state.gdn_value_heads(), T, kBatch});
            Tensor beta_batch = beta_rank.view({state.gdn_value_heads(), T, kBatch});
            Tensor out_batch  = o.view({128, state.gdn_value_heads(), T, kBatch});
            Tensor state_pool = recurrent_state.view({128, 128, state.gdn_value_heads(), 1});
            ops::gated_delta_net_snapshot(q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                                          /*normalize_qk=*/true, state_pool,
                                          /*valid_columns=*/Tensor{}, slots, slots, out_batch,
                                          stream_for(rank));
        } else {
            ops::gated_delta_net(q_recurrent, k_recurrent, v_recurrent, g_rank, beta_rank, kGdnScale,
                                 true, workspace_, recurrent_state, o_recurrent, stream_for(rank));
        }
        NINFER_QWEN4EXP_DIAG_OP_END("gdn_recurrent");
        NINFER_QWEN4EXP_DIAG_MARK("gdn_recurrent");
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("gdn_norm");
        ops::sigmoid_gated_rmsnorm(o_recurrent, weights.norm,
                                   z.view({128, state.gdn_value_heads(), T}),
                                   TextConfig::rms_epsilon, o_normalized, stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("gdn_norm");
        NINFER_QWEN4EXP_DIAG_MARK("gdn_norm");
        projected[slot] = on.view({value_width, T});
        output_weight[slot] = weights.output;
        arena_ptrs[slot] = workspaces_[slot];
    });
    for (int rank = 0; rank < tp_; ++rank) {
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("gdn_out");
    }
    row_parallel_linear(projected, output_weight, mixer_out, staging, arena_ptrs);
    for (int rank = 0; rank < tp_; ++rank) {
        NINFER_QWEN4EXP_DIAG_OP_END("gdn_out");
    }
}

inline void publish_text_qsa_tail(void* context) {
    auto* pair = static_cast<std::pair<TextState*, std::int32_t>*>(context);
    pair->first->publish_tail(pair->second);
}

template <class AttentionWeights>
inline void TextTrunk::run_qsa_layer(
    std::int32_t T, const std::array<const AttentionWeights*, kMaximumDevices>& weights,
    const std::array<QsaLayerStateView, kMaximumDevices>& state,
    const std::array<Tensor, kMaximumDevices>& block_input,
    const std::array<Tensor, kMaximumDevices>& positions,
    const std::array<Tensor, kMaximumDevices>& valid_columns,
    const std::array<Tensor, kMaximumDevices>& table_rows,
    std::array<Tensor, kMaximumDevices>& mixer_out,
    const std::array<Tensor, kMaximumDevices>& staging) {
    const std::int32_t index_dim = TextConfig::index_head_dim;
    std::array<Tensor, kMaximumDevices> projected{};
    std::array<Weight, kMaximumDevices> output_weight{};
    std::array<WorkspaceArena*, kMaximumDevices> arena_ptrs{};

    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        WorkspaceArena& workspace_ = *workspaces_[slot];
        const auto& layer_weights = *weights[slot];
        const QsaLayerStateView& qsa_state = state[slot];
        const std::int32_t query_heads = TextConfig::query_heads / tp_;
        const std::int32_t kv_heads    = TextConfig::kv_heads / tp_;
        const std::int32_t query_width = query_heads * TextConfig::head_dim;
        const std::int32_t kv_width    = kv_heads * TextConfig::head_dim;

        Tensor query_flat       = activations_[slot]->alloc(kTextActivationDtype, {query_width, T});
        Tensor output_gate_flat = activations_[slot]->alloc(kTextActivationDtype, {query_width, T});
        Tensor key_flat         = activations_[slot]->alloc(kTextActivationDtype, {kv_width, T});
        Tensor value_flat       = activations_[slot]->alloc(kTextActivationDtype, {kv_width, T});
        Tensor index_query_flat =
            activations_[slot]->alloc(kTextActivationDtype, {index_dim * TextConfig::index_heads, T});
        Tensor index_key_flat =
            activations_[slot]->alloc(kTextActivationDtype, {index_dim * TextConfig::index_kv_heads, T});
        Tensor query       = query_flat.view({kHeadDim, query_heads, T, kBatch});
        Tensor output_gate = output_gate_flat.view({kHeadDim, query_heads, T, kBatch});
        Tensor key         = key_flat.view({kHeadDim, kv_heads, T, kBatch});
        Tensor value       = value_flat.view({kHeadDim, kv_heads, T, kBatch});
        Tensor index_query = index_query_flat.view({index_dim, TextConfig::index_heads, T, kBatch});
        Tensor index_key   = index_key_flat.view({index_dim, TextConfig::index_kv_heads, T, kBatch});
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("qsa_proj");
        Variant::attention_projection(block_input[slot], layer_weights, query_flat, output_gate_flat,
                                       key_flat, value_flat, index_query_flat, index_key_flat,
                                       workspace_, stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("qsa_proj");
        NINFER_QWEN4EXP_DIAG_MARK("qsa_proj");

        Tensor rotated_query =
            activations_[slot]->alloc(kTextActivationDtype, {kHeadDim, query_heads, T, kBatch});
        Tensor rotated_index_query =
            activations_[slot]->alloc(kTextActivationDtype, {index_dim, TextConfig::index_heads, T, kBatch});

        const ops::QsaPrepareParameters parameters{
            .query_norm        = layer_weights.query_norm,
            .key_norm          = layer_weights.key_norm,
            .index_query_norm  = layer_weights.index_query_norm,
            .index_key_norm    = layer_weights.index_key_norm,
            .rms_epsilon       = TextConfig::rms_epsilon,
            .rotary_dim        = TextConfig::rotary_dim,
            .theta             = static_cast<float>(TextConfig::rope_theta),
        };
        const ops::QsaBatchView batch{.positions = positions[slot],
                                       .valid_columns = valid_columns[slot],
                                       .table_rows = table_rows[slot]};
        ops::QsaRawRecordWriteView records = qsa_state.records;
        const ops::QsaCacheView& cache = qsa_state.cache;

        NINFER_QWEN4EXP_DIAG_OP_BEGIN("qsa_prep");
        ops::qsa_prepare(ops::QsaProjectionView{query, key, value, index_query, index_key},
                         parameters, batch, qsa_state.before, cache, workspace_, records,
                         rotated_query, rotated_index_query, stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("qsa_prep");
        NINFER_QWEN4EXP_DIAG_MARK("qsa_prep");

        ops::QsaBlockSelection selected{
            .selection = activations_[slot]->alloc(DType::I32,
                                                    {TextConfig::index_block_budget, T, kBatch}),
            .counts    = activations_[slot]->alloc(DType::I32, {T, kBatch}),
        };
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("qsa_index");
        ops::qsa_indexer(rotated_index_query,
                         ops::ConstQsaIndexCacheView{cache.index_k_pages, cache.page_table}, batch,
                         static_cast<std::uint32_t>(T), workspace_, selected, stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("qsa_index");
        NINFER_QWEN4EXP_DIAG_MARK("qsa_index");

        Tensor attention =
            activations_[slot]->alloc(kTextActivationDtype, {kHeadDim, query_heads, T, kBatch});
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("qsa_attn");
        ops::qsa_sparse_gqa(rotated_query, output_gate, cache, batch, selected,
                            static_cast<std::uint32_t>(T), workspace_, attention, stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("qsa_attn");
        NINFER_QWEN4EXP_DIAG_MARK("qsa_attn");

        const ops::QsaRawRecordReadView read_records{records.raw_keys, records.rope_positions};
        ops::QsaTailStateView after = qsa_state.after;
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("qsa_tail");
        ops::qsa_tail_extract(qsa_state.const_tail, read_records, records.base_positions,
                              valid_columns[slot], after, stream_for(rank));
        if (qsa_state.publish != nullptr) { qsa_state.publish(qsa_state.publish_context); }
        NINFER_QWEN4EXP_DIAG_OP_END("qsa_tail");
        NINFER_QWEN4EXP_DIAG_MARK("qsa_tail");

        projected[slot] = attention.view({query_width, T});
        output_weight[slot] = layer_weights.output;
        arena_ptrs[slot] = workspaces_[slot];
    });
    for (int rank = 0; rank < tp_; ++rank) {
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("qsa_out");
    }
    row_parallel_linear(projected, output_weight, mixer_out, staging, arena_ptrs);
    for (int rank = 0; rank < tp_; ++rank) {
        NINFER_QWEN4EXP_DIAG_OP_END("qsa_out");
    }
}

inline void TextTrunk::run_qsa(std::size_t layer, std::int32_t T,
                               const std::array<Tensor, kMaximumDevices>& block_input,
                               const std::array<Tensor, kMaximumDevices>& positions,
                               const std::array<Tensor, kMaximumDevices>& valid_columns,
                               const std::array<Tensor, kMaximumDevices>& table_rows,
                               std::array<Tensor, kMaximumDevices>& mixer_out,
                               const std::array<Tensor, kMaximumDevices>& staging) {
    using Weights = std::remove_cvref_t<decltype(models_[0]->layers[layer].attention_projection)>;
    std::array<const Weights*, kMaximumDevices> weights{};
    std::array<QsaLayerStateView, kMaximumDevices> views{};
    std::array<std::pair<TextState*, std::int32_t>, kMaximumDevices> contexts{};
    for (int rank = 0; rank < tp_; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        TextState& state = *states_[slot];
        const std::int32_t qsa_slot = state.qsa_slot(layer);
        weights[slot] = &models_[slot]->layers[layer].attention_projection;
        contexts[slot] = {&state, qsa_slot};
        views[slot] = QsaLayerStateView{
            .cache       = state.cache(qsa_slot),
            .before      = state.before_tail(qsa_slot),
            .const_tail  = state.const_tail(qsa_slot),
            .after       = state.tail(qsa_slot),
            .records     = verifying() ? state.verification_records(qsa_slot, static_cast<std::size_t>(T))
                                       : state.records(static_cast<std::size_t>(T)),
            .publish     = publish_text_qsa_tail,
            .publish_context = &contexts[slot],
        };
    }
    run_qsa_layer(T, weights, views, block_input, positions, valid_columns, table_rows, mixer_out,
                  staging);
}

inline void TextTrunk::run_ple(std::int32_t T,
                               std::array<Tensor, kMaximumDevices>& residual,
                               std::array<Tensor, kMaximumDevices>& updated,
                               std::array<Tensor, kMaximumDevices>& normalized) {
    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        WorkspaceArena& workspace_ = *workspaces_[slot];
        const auto& ple = models_[slot]->ple;
        TextState& state = *states_[slot];
        const std::int32_t entries = static_cast<std::int32_t>(ops::kPleHeads) * T * kBatch;

        Tensor embedding = activations_[slot]->alloc(kTextActivationDtype,
                                                     {TextConfig::ple_embed_dim, T, kBatch});
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("ple_lookup");
        ops::ple_lookup_rows(state.ple_table(), state.ple_rows(),
                             static_cast<std::size_t>(entries), state.ple_staged_rows(), embedding,
                             stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("ple_lookup");
        NINFER_QWEN4EXP_DIAG_MARK("ple_lookup");

        Tensor projected_key   = activations_[slot]->alloc(kTextActivationDtype, {kChannels, T, kBatch});
        Tensor projected_value = activations_[slot]->alloc(kTextActivationDtype, {kHidden, T, kBatch});
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("ple_proj");
        ops::linear(embedding, ple.key, projected_key, ops::LinearPolicy::A16Only, workspace_,
                    stream_for(rank));
        ops::linear(embedding, ple.value, projected_value, ops::LinearPolicy::A16Only, workspace_,
                    stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("ple_proj");
        NINFER_QWEN4EXP_DIAG_MARK("ple_proj");

        Tensor gated_value     = activations_[slot]->alloc(kTextActivationDtype, {kChannels, T, kBatch});
        Tensor normalized_conv = activations_[slot]->alloc(kTextActivationDtype, {kChannels, T, kBatch});
        Tensor offsets         = activations_[slot]->alloc(DType::I32, {kBatch});
        Tensor counts          = activations_[slot]->alloc(DType::I32, {kBatch});
        const std::int32_t live         = T;
        const std::int32_t column_start = 0;
        CUDA_CHECK(cudaMemcpyAsync(offsets.data,
                                   NINFER_QWEN4EXP_GRAPH_SCALARS_SRC(rank, column_start, "ple_offset"), 4,
                                   cudaMemcpyHostToDevice, stream_for(rank)));
        CUDA_CHECK(cudaMemcpyAsync(counts.data,
                                   NINFER_QWEN4EXP_GRAPH_SCALARS_SRC(rank, live, "ple_count"), 4,
                                   cudaMemcpyHostToDevice, stream_for(rank)));

        NINFER_QWEN4EXP_DIAG_OP_BEGIN("ple_gate");
        ops::ple_gate(projected_key, projected_value, residual[slot], ple.norm_key, ple.norm_query,
                      ple.norm_conv, TextConfig::rms_epsilon, gated_value, normalized_conv,
                      stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("ple_gate");
        NINFER_QWEN4EXP_DIAG_MARK("ple_gate");
        if (verifying()) {
            const Tensor record = state.verification_ple_values(static_cast<std::size_t>(T));
            CUDA_CHECK(cudaMemcpyAsync(record.data, normalized_conv.data, record.bytes(),
                                       cudaMemcpyDeviceToDevice, stream_for(rank)));
        }

        const Tensor next_norm =
            models_[slot]->layers[TextConfig::ple_layer].attention_hc.norm.view({kChannels});
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("ple_conv");
        ops::ple_conv_add_norm(state.ple_history(), normalized_conv, residual[slot], gated_value,
                               ple.kernel, offsets, counts, next_norm, TextConfig::rms_epsilon,
                               updated[slot], normalized[slot], stream_for(rank));

        Tensor after_history = activations_[slot]->alloc(
            kTextActivationDtype, {kChannels, static_cast<std::int32_t>(ops::kPleConvWindow), kBatch});
        ops::ple_state_extract(state.ple_history(), normalized_conv, counts, after_history,
                               stream_for(rank));
        CUDA_CHECK(cudaMemcpyAsync(
            state.ple_history().data, after_history.data,
            static_cast<std::size_t>(kChannels) * ops::kPleConvWindow * kBatch * 2,
            cudaMemcpyDeviceToDevice, stream_for(rank)));
        NINFER_QWEN4EXP_DIAG_OP_END("ple_conv");
        NINFER_QWEN4EXP_DIAG_MARK("ple_conv");
    });
}

inline TokenId TextTrunk::forward(std::span<const TokenId> tokens,
                                  std::span<std::int32_t> winner_out) {
    const std::int32_t T = static_cast<std::int32_t>(tokens.size());
    for_each_rank([&](int rank) {
        NINFER_QWEN4EXP_GRAPH_PREPARE_PLE(rank, *states_[static_cast<std::size_t>(rank)], tokens);
    });

    std::array<std::optional<DeviceArena::Scope>, kMaximumDevices> activation_scopes;
    std::array<std::optional<DeviceArena::Scope>, kMaximumDevices> workspace_scopes;
    for (int rank = 0; rank < tp_; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        activation_scopes[slot].emplace(activations_[slot]->scope());
        workspace_scopes[slot].emplace(workspaces_[slot]->scope());
    }

    const std::uint32_t base_position = states_[0]->next_position();
    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(T));
    for (std::int32_t t = 0; t < T; ++t) {
        host_positions[static_cast<std::size_t>(t)] =
            static_cast<std::int32_t>(base_position + static_cast<std::uint32_t>(t));
    }
    const std::int32_t live = T;
    const std::int32_t row  = 0;

    std::array<Tensor, kMaximumDevices> ids{};
    std::array<Tensor, kMaximumDevices> embedding_out{};
    std::array<Tensor, kMaximumDevices> residual{};
    std::array<Tensor, kMaximumDevices> normalized{};
    std::array<Tensor, kMaximumDevices> stage{};
    std::array<Tensor, kMaximumDevices> block_input{};
    std::array<Tensor, kMaximumDevices> injection{};
    std::array<Tensor, kMaximumDevices> mixer_out{};
    std::array<Tensor, kMaximumDevices> moe_out{};
    std::array<Tensor, kMaximumDevices> sample{};
    std::array<Tensor, kMaximumDevices> logits{};
    std::array<Tensor, kMaximumDevices> logits_part{};
    std::array<Tensor, kMaximumDevices> positions{};
    std::array<Tensor, kMaximumDevices> valid_columns{};
    std::array<Tensor, kMaximumDevices> table_rows{};
    std::array<Tensor, kMaximumDevices> staging{};

    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        cudaStream_t stream = stream_for(rank);
        NINFER_QWEN4EXP_DIAG_MARK("round_begin");
        ids[slot] = activations_[slot]->alloc(DType::I32, {T});
        CUDA_CHECK(cudaMemcpyAsync(ids[slot].data,
                                    NINFER_QWEN4EXP_GRAPH_IDS_SRC(rank, tokens),
                                    static_cast<std::size_t>(T) * 4,
                                    cudaMemcpyHostToDevice, stream));
        embedding_out[slot] = activations_[slot]->alloc(kTextActivationDtype, {kHidden, T});
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("embedding");
        ops::embedding(ids[slot], models_[slot]->embedding, embedding_out[slot], stream);
        NINFER_QWEN4EXP_DIAG_OP_END("embedding");

        residual[slot]    = activations_[slot]->alloc(kTextActivationDtype, {kHidden, kStreams, T, kBatch});
        normalized[slot]  = activations_[slot]->alloc(kTextActivationDtype, {kHidden, kStreams, T, kBatch});
        stage[slot]       = activations_[slot]->alloc(kTextActivationDtype, {kHidden, kStreams, T, kBatch});
        block_input[slot] = activations_[slot]->alloc(kTextActivationDtype, {kHidden, T});
        injection[slot]   = activations_[slot]->alloc(DType::FP32, {kStreams, T});
        mixer_out[slot]   = activations_[slot]->alloc(kTextActivationDtype, {kHidden, T});
        moe_out[slot]     = activations_[slot]->alloc(kTextActivationDtype, {kHidden, T});
        sample[slot]      = activations_[slot]->alloc(kTextActivationDtype, {kHidden, T});
        logits_part[slot] = activations_[slot]->alloc(kTextActivationDtype, {TextConfig::vocab / tp_, T});
        logits[slot]      = tp_ > 1
                                ? activations_[slot]->alloc(kTextActivationDtype, {TextConfig::vocab, T})
                                : logits_part[slot];
        positions[slot]     = activations_[slot]->alloc(DType::I32, {T, 1, kBatch});
        valid_columns[slot] = activations_[slot]->alloc(DType::I32, {kBatch});
        table_rows[slot]    = activations_[slot]->alloc(DType::I32, {kBatch});
        staging[slot]       = workspaces_[slot]->alloc(kTextActivationDtype, {kHidden, T});
        CUDA_CHECK(cudaMemcpyAsync(positions[slot].data,
                                    NINFER_QWEN4EXP_GRAPH_POSITIONS_SRC(rank, host_positions),
                                    static_cast<std::size_t>(T) * 4, cudaMemcpyHostToDevice,
                                    stream));
        CUDA_CHECK(cudaMemcpyAsync(valid_columns[slot].data,
                                    NINFER_QWEN4EXP_GRAPH_SCALARS_SRC(rank, live, "live"), 4,
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(table_rows[slot].data,
                                    NINFER_QWEN4EXP_GRAPH_SCALARS_SRC(rank, row, "row"), 4,
                                    cudaMemcpyHostToDevice, stream));
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("stem");
        ops::hc_expand_norm(embedding_out[slot],
                            models_[slot]->layers[0].attention_hc.norm, TextConfig::rms_epsilon,
                            residual[slot], normalized[slot], stream);
        NINFER_QWEN4EXP_DIAG_OP_END("stem");
        NINFER_QWEN4EXP_DIAG_MARK("stem");
    });

    std::array<WorkspaceArena*, kMaximumDevices> workspace_ptrs{};
    for (int rank = 0; rank < tp_; ++rank) {
        workspace_ptrs[static_cast<std::size_t>(rank)] = workspaces_[static_cast<std::size_t>(rank)];
    }

    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        NINFER_QWEN4EXP_GRAPH_PREFETCH(models_[slot]->layers[0].post_mixer, stream_for(rank));
    });

    for (std::uint32_t layer = 0;
         layer < static_cast<std::uint32_t>(models_[0]->loaded_layers); ++layer) {
        std::array<std::optional<DeviceArena::Scope>, kMaximumDevices> layer_activation_scopes;
        std::array<std::optional<DeviceArena::Scope>, kMaximumDevices> layer_workspace_scopes;
        for (int rank = 0; rank < tp_; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            layer_activation_scopes[slot].emplace(activations_[slot]->scope());
            layer_workspace_scopes[slot].emplace(workspaces_[slot]->scope());
        }
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("residual_in", layer, residual[0]);
        NINFER_QWEN4EXP_LAYER_DUMP("normalized_in", layer, normalized[0]);
#endif
        for_each_rank([&](int rank) {
            const auto slot = static_cast<std::size_t>(rank);
            const auto& attention_hc = models_[slot]->layers[layer].attention_hc;
            NINFER_QWEN4EXP_DIAG_OP_BEGIN("attn_hc");
            ops::hc_mix(normalized[slot],
                        ops::HcMixWeights{attention_hc.down_injection, attention_hc.up},
                        *workspaces_[slot], block_input[slot], injection[slot], stream_for(rank));
            NINFER_QWEN4EXP_DIAG_OP_END("attn_hc");
            NINFER_QWEN4EXP_DIAG_MARK("attn_hc");
            });
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("attention_block", layer, block_input[0]);
        NINFER_QWEN4EXP_LAYER_DUMP("attention_injection", layer, injection[0]);
#endif
        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            run_qsa(layer, T, block_input, positions, valid_columns, table_rows, mixer_out, staging);
        } else {
            run_gdn(layer, T, block_input, mixer_out, staging);
        }
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("mixer", layer, mixer_out[0]);
#endif
        for_each_rank([&](int rank) {
            const auto slot = static_cast<std::size_t>(rank);
            NINFER_QWEN4EXP_DIAG_MARK("mixer");
            const auto& moe_hc = models_[slot]->layers[layer].moe_hc;
            NINFER_QWEN4EXP_DIAG_OP_BEGIN("moe_hc_combine");
            ops::hc_combine_norm(residual[slot], mixer_out[slot], injection[slot], moe_hc.norm,
                                 TextConfig::rms_epsilon, stage[slot], normalized[slot],
                                 stream_for(rank));
            NINFER_QWEN4EXP_DIAG_OP_END("moe_hc_combine");
            NINFER_QWEN4EXP_DIAG_OP_BEGIN("moe_hc_mix");
            ops::hc_mix(normalized[slot],
                        ops::HcMixWeights{moe_hc.down_injection, moe_hc.up}, *workspaces_[slot],
                        block_input[slot], injection[slot], stream_for(rank));
            NINFER_QWEN4EXP_DIAG_OP_END("moe_hc_mix");
            CUDA_CHECK(cudaMemsetAsync(moe_out[slot].data, 0, moe_out[slot].bytes(),
                                       stream_for(rank)));
            NINFER_QWEN4EXP_DIAG_MARK("moe_hc");
        });
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("stage", layer, stage[0]);
        NINFER_QWEN4EXP_LAYER_DUMP("normalized_moe", layer, normalized[0]);
        NINFER_QWEN4EXP_LAYER_DUMP("moe_block", layer, block_input[0]);
        NINFER_QWEN4EXP_LAYER_DUMP("moe_injection", layer, injection[0]);
#endif

        if (NINFER_QWEN4EXP_MOE_ENABLED()) {
            std::array<ops::MoeNvfp4RouteView, kMaximumDevices> routes{};
            for_each_rank([&](int rank) {
                const auto slot = static_cast<std::size_t>(rank);
                const std::int32_t top_k = TextConfig::experts_per_token;
                Tensor route_ids = activations_[slot]->alloc(DType::I32, {top_k, T});
                Tensor route_coefficients = activations_[slot]->alloc(DType::FP32, {top_k, T});
                Tensor route_gate = activations_[slot]->alloc(DType::FP32, {T});
                const bool fused_route =
                    T >= 1 && T <= ops::kMoeRouteFusedMaxTokens;
                Tensor route_logits =
                    fused_route
                        ? activations_[slot]->alloc(DType::FP32, {TextConfig::experts + 1, T})
                        : Tensor{};
                NINFER_QWEN4EXP_DIAG_OP_BEGIN("route");
                ops::moe_nvfp4_route(block_input[slot], models_[slot]->layers[layer].post_mixer,
                                      route_ids, route_coefficients, route_gate, stream_for(rank),
                                      fused_route ? static_cast<float*>(route_logits.data) : nullptr,
                                      fused_route ? states_[slot]->route_counter() : nullptr);
                NINFER_QWEN4EXP_DIAG_OP_END("route");
                NINFER_QWEN4EXP_DIAG_MARK("route");
                routes[slot] = ops::MoeNvfp4RouteView{route_ids, route_coefficients, route_gate};
            });
            for_each_rank([&](int rank) {
                const auto slot = static_cast<std::size_t>(rank);
                NINFER_QWEN4EXP_EXPERT_READ_BEGIN(layer);
                NINFER_QWEN4EXP_DIAG_OP_BEGIN("routed");
                Variant::post_mixer_routed(block_input[slot],
                                            models_[slot]->layers[layer].post_mixer, routes[slot],
                                            moe_out[slot], stream_for(rank));
            });
            for_each_rank([&](int rank) {
                const auto slot = static_cast<std::size_t>(rank);
                NINFER_QWEN4EXP_GRAPH_ROUTED_FINISH(models_[slot]->layers[layer].post_mixer,
                                                    stream_for(rank));
                NINFER_QWEN4EXP_EXPERT_READ_END(layer);
                NINFER_QWEN4EXP_DIAG_OP_END("routed");
                NINFER_QWEN4EXP_DIAG_MARK("routed");
            });
            for (int rank = 0; rank < tp_; ++rank) {
                NINFER_QWEN4EXP_DIAG_OP_BEGIN("moe_reduce");
            }
            reduce_rows(moe_out, staging);
            for (int rank = 0; rank < tp_; ++rank) {
                NINFER_QWEN4EXP_DIAG_OP_END("moe_reduce");
            }
            for_each_rank([&](int rank) {
                const auto slot = static_cast<std::size_t>(rank);
                NINFER_QWEN4EXP_DIAG_MARK("moe_reduce");
                NINFER_QWEN4EXP_DIAG_OP_BEGIN("moe_shared");
                Variant::post_mixer_shared(block_input[slot],
                                            models_[slot]->layers[layer].post_mixer, routes[slot],
                                            moe_out[slot], stream_for(rank));
                NINFER_QWEN4EXP_DIAG_OP_END("moe_shared");
                NINFER_QWEN4EXP_DIAG_MARK("moe_shared");
            });
        }
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("moe", layer, moe_out[0]);
#endif

        if (NINFER_QWEN4EXP_HAS_PLE() &&
            layer + 1 == static_cast<std::size_t>(TextConfig::ple_layer)) {
            for_each_rank([&](int rank) {
                NINFER_QWEN4EXP_GRAPH_PLE_AWAIT(rank, *states_[static_cast<std::size_t>(rank)]);
                NINFER_QWEN4EXP_DIAG_MARK("ple_await");
            });
            for_each_rank([&](int rank) {
                const auto slot = static_cast<std::size_t>(rank);
                NINFER_QWEN4EXP_DIAG_OP_BEGIN("pre_ple");
                ops::hc_combine(stage[slot], moe_out[slot], injection[slot], residual[slot],
                                stream_for(rank));
                NINFER_QWEN4EXP_DIAG_OP_END("pre_ple");
                NINFER_QWEN4EXP_DIAG_MARK("pre_ple");
            });
            run_ple(T, residual, stage, normalized);
            for (int rank = 0; rank < tp_; ++rank) {
                std::swap(residual[static_cast<std::size_t>(rank)],
                          stage[static_cast<std::size_t>(rank)]);
            }
        } else {
            for_each_rank([&](int rank) {
                const auto slot = static_cast<std::size_t>(rank);
                const Tensor& next_norm =
                    layer + 1 < models_[slot]->loaded_layers
                        ? models_[slot]->layers[layer + 1].attention_hc.norm
                        : models_[slot]->tail_norm;
                NINFER_QWEN4EXP_DIAG_OP_BEGIN("layer_end_combine");
                ops::hc_combine_norm(stage[slot], moe_out[slot], injection[slot], next_norm,
                                     TextConfig::rms_epsilon, residual[slot], normalized[slot],
                                     stream_for(rank));
                NINFER_QWEN4EXP_DIAG_OP_END("layer_end_combine");
                NINFER_QWEN4EXP_DIAG_MARK("layer_end");
            });
        }
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("residual_out", layer, residual[0]);
        NINFER_QWEN4EXP_LAYER_DUMP("normalized_out", layer, normalized[0]);
#endif
    }

    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (verifying()) {
            TextState& state = *states_[slot];
            const std::size_t token_bytes =
                static_cast<std::size_t>(TextConfig::residual_width) * sizeof(std::uint16_t);
            for (std::int32_t token = 0; token < T; ++token) {
                const Tensor record = state.verification_residual(static_cast<std::size_t>(token));
                CUDA_CHECK(cudaMemcpyAsync(
                    record.data,
                    static_cast<const std::byte*>(residual[slot].data) +
                        static_cast<std::size_t>(token) * token_bytes,
                    record.bytes(), cudaMemcpyDeviceToDevice, stream_for(rank)));
            }
        }
        if (residual_sinks_[slot] != nullptr) {
            const std::size_t bytes = std::min(residual[slot].bytes(), residual_sink_bytes_[slot]);
            CUDA_CHECK(cudaMemcpyAsync(residual_sinks_[slot], residual[slot].data, bytes,
                                       cudaMemcpyDeviceToDevice, stream_for(rank)));
        }
    });

    std::array<Weight, kMaximumDevices> head{};
    std::array<Tensor, kMaximumDevices> winner{};
    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("head_mix");
        ops::hc_final_mix(normalized[slot],
                          ops::HcFinalMixWeights{models_[slot]->final_hc.down_injection,
                                                 models_[slot]->final_hc.up},
                          *workspaces_[slot], sample[slot], stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("head_mix");
        NINFER_QWEN4EXP_DIAG_MARK("head_mix");
        head[slot] = models_[slot]->output_head;
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("head_linear");
        ops::linear(sample[slot], head[slot], logits_part[slot], ops::LinearPolicy::A16Only,
                    *workspaces_[slot], stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("head_linear");
        NINFER_QWEN4EXP_DIAG_MARK("head_linear");
    });
    if (tp_ > 1) {
        const std::int32_t vocab = TextConfig::vocab;
        for (int rank = 0; rank < tp_; ++rank) {
            NINFER_QWEN4EXP_DIAG_OP_BEGIN("head_gather");
        }
        for (std::int32_t token = 0; token < T; ++token) {
            std::array<Tensor, kMaximumDevices> destination_token{};
            std::array<Tensor, kMaximumDevices> part_token{};
            for (int rank = 0; rank < tp_; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                destination_token[slot] = Tensor(
                    static_cast<std::byte*>(logits[slot].data) +
                        static_cast<std::size_t>(token) * vocab * 2,
                    kTextActivationDtype, {1, vocab});
                part_token[slot] = Tensor(
                    static_cast<std::byte*>(logits_part[slot].data) +
                        static_cast<std::size_t>(token) * (vocab / tp_) * 2,
                    kTextActivationDtype, {1, vocab / tp_});
            }
            ops::allgather_rows(destination_token, part_token, *execution_, *comms_);
        }
        for (int rank = 0; rank < tp_; ++rank) {
            NINFER_QWEN4EXP_DIAG_OP_END("head_gather");
        }
    }
    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        NINFER_QWEN4EXP_DIAG_MARK("head_gather");
        NINFER_QWEN4EXP_LAST_STAGE(sample[slot], logits[slot]);
        winner[slot] = activations_[slot]->alloc(DType::I32, {T});
        NINFER_QWEN4EXP_DIAG_OP_BEGIN("argmax");
        ops::argmax(logits[slot], winner[slot], TextConfig::vocab, stream_for(rank));
        NINFER_QWEN4EXP_DIAG_OP_END("argmax");
        NINFER_QWEN4EXP_DIAG_MARK("argmax");
    });

    NINFER_QWEN4EXP_GRAPH_FINISH_SYNC();
    std::int32_t token = 0;
    if (winner_out.size() >= static_cast<std::size_t>(T)) {
        NINFER_QWEN4EXP_GRAPH_WINNER_COPY(winner_out.data(), winner[0].data,
                                          static_cast<std::size_t>(T) * 4, stream_for(0));
        token = winner_out[static_cast<std::size_t>(T - 1)];
    } else {
        NINFER_QWEN4EXP_GRAPH_WINNER_COPY(NINFER_QWEN4EXP_GRAPH_WINNER_PTR(),
                                          static_cast<const std::byte*>(winner[0].data) +
                                              static_cast<std::size_t>(T - 1) * 4,
                                          4, stream_for(0));
    }
    for (int rank = 0; rank < tp_; ++rank) {
        states_[static_cast<std::size_t>(rank)]->advance_position(static_cast<std::uint32_t>(T));
    }
    return static_cast<TokenId>(token);
}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
