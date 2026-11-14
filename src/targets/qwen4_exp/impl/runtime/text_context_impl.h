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
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe_nvfp4.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/scatter.h"
#include "targets/qwen4_exp/impl/runtime/text_context.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#ifndef NINFER_QWEN4EXP_LAST_STAGE
#define NINFER_QWEN4EXP_LAST_STAGE(sample, logits)
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

inline TextState::TextState(DeviceContext& device, const ModelView& model,
                            std::uint32_t kv_capacity_tokens, std::uint32_t forward_tokens,
                            int tp, int rank)
    : device_scope_(device.device), ple_source_(model.ple.rows), device_(device.device),
      stream_(device.stream),
      layers_(model.loaded_layers), forward_tokens_(forward_tokens == 0 ? 1 : forward_tokens),
      pages_(kv_capacity_tokens / 64 + 1), kv_heads_(TextConfig::kv_heads / tp),
      gdn_key_heads_(TextConfig::gdn_key_heads / tp),
      gdn_value_heads_(TextConfig::gdn_value_heads / tp),
      conv_channels_(2 * gdn_key_heads_ * TextConfig::gdn_head_dim +
                     gdn_value_heads_ * TextConfig::gdn_head_dim),
      gdn_layers_(layers_), gdn_index_(layers_),
      conv_states_(layers_ * static_cast<std::size_t>(conv_channels_) * 3 * 2),
      recurrent_states_(layers_ * 128 * 128 * static_cast<std::size_t>(gdn_value_heads_) * 4),
      k_pages_(static_cast<std::size_t>(kHeadDim) * 64 * kv_heads_ * pages_ * 2),
      v_pages_(static_cast<std::size_t>(kHeadDim) * 64 * kv_heads_ * pages_ * 2),
      index_k_pages_(static_cast<std::size_t>(TextConfig::index_head_dim) * 16 *
                     TextConfig::index_kv_heads * pages_ * 2),
      page_table_(pages_ * 4),
      tail_raw_keys_(static_cast<std::size_t>(TextConfig::index_head_dim) * 4 * kBatch * 2),
      tail_first_rope_(3 * kBatch * 4),
      after_raw_keys_(static_cast<std::size_t>(TextConfig::index_head_dim) * 4 * kBatch * 2),
      after_first_rope_(3 * kBatch * 4),
      record_raw_keys_(static_cast<std::size_t>(TextConfig::index_head_dim) * forward_tokens_ *
                       kBatch * 2),
      record_rope_positions_(3 * forward_tokens_ * kBatch * 4),
      record_base_positions_(kBatch * 4),
      ple_history_(static_cast<std::size_t>(kChannels) * ops::kPleConvWindow * kBatch * 2),
      ple_rows_(ops::kPleHeads * forward_tokens_ * kBatch * 4),
      ple_staged_rows_(ops::kPleHeads * forward_tokens_ * kBatch * 128),
      ple_divisors_(ops::kPleShards * 4) {
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
    std::size_t gdn = 0;
    for (std::size_t layer = 0; layer < layers_; ++layer) {
        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            gdn_index_[layer] = std::nullopt;
        } else {
            gdn_index_[layer] = gdn;
            gdn_layers_[gdn++] = layer;
        }
    }
    gdn_layers_.resize(gdn);
    std::vector<std::int32_t> table(pages_);
    for (std::size_t page = 0; page < pages_; ++page) {
        table[page] = static_cast<std::int32_t>(page);
    }
    page_table_.copy_from_host(table.data(), pages_ * 4);
    reset();
}

inline void TextState::reset() {
    DeviceScope scope(device_);
    conv_states_.fill(0);
    recurrent_states_.fill(0);
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
    next_position_ = 0;
    ple_tail_[0]   = TextConfig::eos_token_id;
    ple_tail_[1]   = TextConfig::eos_token_id;
    ple_staged_host_.assign(ple_staged_rows_.bytes, std::byte{0});
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
    ple_staged_host_.resize(rows.size() * row_bytes);
    ple_source_.fetch(ple_source_.context, addresses.data(), addresses.size(),
                      ple_staged_host_.data());
    CUDA_CHECK(cudaMemcpyAsync(ple_rows_.p, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice,
                               stream_));
    CUDA_CHECK(cudaMemcpyAsync(ple_staged_rows_.p, ple_staged_host_.data(),
                               ple_staged_host_.size(), cudaMemcpyHostToDevice, stream_));
    for (const TokenId token : tokens) { advance_ple_tail(token); }
}

inline std::int32_t TextState::base_position() const {
    DeviceScope scope(device_);
    std::int32_t base = 0;
    record_base_positions_.copy_to_host(&base, 4);
    return base;
}

inline Tensor TextState::gdn_conv_state(std::size_t gdn_index) const {
    const std::size_t layer = gdn_layers_.at(gdn_index);
    return Tensor(conv_states_.p + layer * static_cast<std::size_t>(conv_channels_) * 3 * 2,
                  DType::BF16, {3, conv_channels_});
}

inline Tensor TextState::gdn_recurrent_state(std::size_t gdn_index) const {
    const std::size_t layer = gdn_layers_.at(gdn_index);
    return Tensor(recurrent_states_.p +
                      layer * 128 * 128 * static_cast<std::size_t>(gdn_value_heads_) * 4,
                  DType::FP32, {128, 128, gdn_value_heads_});
}

inline std::optional<std::size_t> TextState::gdn_state_index(std::size_t layer) const {
    return gdn_index_.at(layer);
}

inline std::size_t TextState::gdn_state_entries() const noexcept { return gdn_layers_.size(); }

inline ops::QsaCacheView TextState::cache() const {
    const std::int32_t pages = static_cast<std::int32_t>(pages_);
    return ops::QsaCacheView{
        .k_pages       = Tensor(k_pages_.p, DType::BF16, {kHeadDim, 64, kv_heads_, pages}),
        .v_pages       = Tensor(v_pages_.p, DType::FP16, {kHeadDim, 64, kv_heads_, pages}),
        .index_k_pages = Tensor(index_k_pages_.p, DType::BF16,
                                {TextConfig::index_head_dim, 16, TextConfig::index_kv_heads, pages}),
        .page_table    = Tensor(page_table_.p, DType::I32, {pages, kBatch}),
    };
}

inline ops::QsaTailStateView TextState::tail() const {
    return ops::QsaTailStateView{
        .raw_keys   = Tensor(after_raw_keys_.p, DType::BF16, {TextConfig::index_head_dim, 4, kBatch}),
        .first_rope = Tensor(after_first_rope_.p, DType::I32, {3, kBatch}),
    };
}

inline ops::ConstQsaTailStateView TextState::const_tail() const {
    return ops::ConstQsaTailStateView{
        .raw_keys   = Tensor(tail_raw_keys_.p, DType::BF16, {TextConfig::index_head_dim, 4, kBatch}),
        .first_rope = Tensor(tail_first_rope_.p, DType::I32, {3, kBatch}),
    };
}

inline ops::QsaTailStateView TextState::before_tail() const {
    return ops::QsaTailStateView{
        .raw_keys   = Tensor(tail_raw_keys_.p, DType::BF16, {TextConfig::index_head_dim, 4, kBatch}),
        .first_rope = Tensor(tail_first_rope_.p, DType::I32, {3, kBatch}),
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
    return Tensor(ple_history_.p, DType::BF16,
                  {kChannels, static_cast<std::int32_t>(ops::kPleConvWindow), kBatch});
}

inline const std::uint32_t* TextState::ple_rows() const noexcept {
    return static_cast<const std::uint32_t*>(ple_rows_.p);
}

inline const std::uint8_t* TextState::ple_staged_rows() const noexcept {
    return static_cast<const std::uint8_t*>(ple_staged_rows_.p);
}

inline const ops::PleNf4Table& TextState::ple_table() const noexcept { return ple_table_; }

inline void TextState::publish_tail() {
    DeviceScope scope(device_);
    CUDA_CHECK(cudaMemcpyAsync(tail_raw_keys_.p, after_raw_keys_.p,
                               static_cast<std::size_t>(TextConfig::index_head_dim) * 4 * kBatch * 2,
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(tail_first_rope_.p, after_first_rope_.p, 3 * kBatch * 4,
                               cudaMemcpyDeviceToDevice, stream_));
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
      execution_(&execution), comms_(&comms), tp_(execution.tp), stream_(execution.primary().stream) {
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

inline TokenId TextTrunk::prefill(std::span<const TokenId> tokens) { return forward(tokens); }

inline TokenId TextTrunk::decode(TokenId token) {
    return forward(std::span<const TokenId>{&token, 1});
}

inline TokenId TextTrunk::forward(std::span<const TokenId> tokens) {
    return tp_ <= 1 ? forward_single(tokens) : forward_parallel(tokens);
}

inline void TextTrunk::run_gdn(std::size_t layer, std::int32_t T, const Tensor& block_input,
                               Tensor& mixer_out) {
    const auto& weights = models_[0]->layers[layer].gdn_projection;
    TextState& state    = *states_[0];
    WorkspaceArena& workspace_ = *workspaces_[0];
    const std::int32_t key_width   = TextConfig::gdn_key_width;
    const std::int32_t value_width = TextConfig::gdn_value_width;
    const std::int32_t conv_channels = state.conv_channels();

    Tensor q    = activations_[0]->alloc(DType::BF16, {key_width, T});
    Tensor k    = activations_[0]->alloc(DType::BF16, {key_width, T});
    Tensor v    = activations_[0]->alloc(DType::BF16, {value_width, T});
    Tensor z    = activations_[0]->alloc(DType::BF16, {value_width, T});
    Tensor g    = activations_[0]->alloc(DType::FP32, {TextConfig::gdn_value_heads, T});
    Tensor beta = activations_[0]->alloc(DType::FP32, {TextConfig::gdn_value_heads, T});
    Variant::gdn_projection_control(block_input, weights, q, k, v, z, g, beta, workspace_, stream_);
    Tensor conv_in  = activations_[0]->alloc(DType::BF16, {conv_channels, T});
    Tensor conv_out = activations_[0]->alloc(DType::BF16, {conv_channels, T});
    const auto copy_segment = [&](const Tensor& source, std::int32_t column) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            static_cast<std::byte*>(conv_in.data) +
                static_cast<std::size_t>(column) * sizeof(std::uint16_t),
            static_cast<std::size_t>(conv_channels) * sizeof(std::uint16_t),
            source.data, static_cast<std::size_t>(source.ne[0]) * sizeof(std::uint16_t),
            static_cast<std::size_t>(source.ne[0]) * sizeof(std::uint16_t), static_cast<std::size_t>(T),
            cudaMemcpyDeviceToDevice, stream_));
    };
    copy_segment(q, 0);
    copy_segment(k, key_width);
    copy_segment(v, 2 * key_width);

    const std::optional<std::size_t> slot = state.gdn_state_index(layer);
    if (!slot.has_value()) {
        throw std::logic_error("a QSA layer has no GDN state slot");
    }
    const std::size_t gdn_index = *slot;
    Tensor conv_state = state.gdn_conv_state(gdn_index);
    Tensor conv_kernel = weights.convolution.view({conv_channels, TextConfig::gdn_conv_kernel});
    ops::causal_conv1d_silu(conv_in, conv_kernel, conv_state, conv_state, conv_out, stream_);

    ops::extract_bf16_columns(conv_out, 0, q, stream_);
    ops::extract_bf16_columns(conv_out, key_width, k, stream_);
    ops::extract_bf16_columns(conv_out, 2 * key_width, v, stream_);

    Tensor o = activations_[0]->alloc(DType::BF16, {value_width, T});
    Tensor on = activations_[0]->alloc(DType::BF16, {value_width, T});
    Tensor q_recurrent = q.view({128, TextConfig::gdn_key_heads, T});
    Tensor k_recurrent = k.view({128, TextConfig::gdn_key_heads, T});
    Tensor v_recurrent = v.view({128, TextConfig::gdn_value_heads, T});
    Tensor o_recurrent = o.view({128, TextConfig::gdn_value_heads, T});
    Tensor o_normalized = on.view({128, TextConfig::gdn_value_heads, T});
    Tensor recurrent_state = state.gdn_recurrent_state(gdn_index);
    ops::gated_delta_net(q_recurrent, k_recurrent, v_recurrent, g, beta, kGdnScale, true, workspace_,
                         recurrent_state, o_recurrent, stream_);
    ops::gated_rmsnorm(o_recurrent, weights.norm, z.view({128, TextConfig::gdn_value_heads, T}),
                       TextConfig::rms_epsilon, o_normalized, stream_);
    ops::linear(on, weights.output, mixer_out, ops::LinearPolicy::A16Only, workspace_, stream_);
}

inline void TextTrunk::run_qsa(std::size_t layer, std::int32_t T, const Tensor& block_input,
                               const Tensor& positions, const Tensor& valid_columns,
                               const Tensor& table_rows, Tensor& mixer_out) {
    const auto& weights = models_[0]->layers[layer].attention_projection;
    TextState& state    = *states_[0];
    WorkspaceArena& workspace_ = *workspaces_[0];
    const std::int32_t index_dim = TextConfig::index_head_dim;

    const std::int32_t query_rows = kHeadDim * TextConfig::query_heads;
    Tensor query_flat       = activations_[0]->alloc(DType::BF16, {query_rows, T});
    Tensor output_gate_flat = activations_[0]->alloc(DType::BF16, {query_rows, T});
    Tensor key_flat         = activations_[0]->alloc(DType::BF16, {kHeadDim * TextConfig::kv_heads, T});
    Tensor value_flat       = activations_[0]->alloc(DType::BF16, {kHeadDim * TextConfig::kv_heads, T});
    Tensor index_query_flat =
        activations_[0]->alloc(DType::BF16, {index_dim * TextConfig::index_heads, T});
    Tensor index_key_flat =
        activations_[0]->alloc(DType::BF16, {index_dim * TextConfig::index_kv_heads, T});
    Tensor query       = query_flat.view({kHeadDim, TextConfig::query_heads, T, kBatch});
    Tensor output_gate = output_gate_flat.view({kHeadDim, TextConfig::query_heads, T, kBatch});
    Tensor key         = key_flat.view({kHeadDim, TextConfig::kv_heads, T, kBatch});
    Tensor value       = value_flat.view({kHeadDim, TextConfig::kv_heads, T, kBatch});
    Tensor index_query = index_query_flat.view({index_dim, TextConfig::index_heads, T, kBatch});
    Tensor index_key   = index_key_flat.view({index_dim, TextConfig::index_kv_heads, T, kBatch});
    Variant::attention_projection(block_input, weights, query_flat, output_gate_flat, key_flat,
                                  value_flat, index_query_flat, index_key_flat, workspace_, stream_);

    Tensor rotated_query = activations_[0]->alloc(DType::BF16, {kHeadDim, TextConfig::query_heads, T, kBatch});
    Tensor rotated_index_query =
        activations_[0]->alloc(DType::BF16, {index_dim, TextConfig::index_heads, T, kBatch});

    const ops::QsaPrepareParameters parameters{
        .query_norm        = weights.query_norm,
        .key_norm          = weights.key_norm,
        .index_query_norm  = weights.index_query_norm,
        .index_key_norm    = weights.index_key_norm,
        .rms_epsilon       = TextConfig::rms_epsilon,
        .rotary_dim        = TextConfig::rotary_dim,
        .theta             = static_cast<float>(TextConfig::rope_theta),
    };
    const ops::QsaBatchView batch{.positions = positions, .valid_columns = valid_columns,
                                  .table_rows = table_rows};
    ops::QsaRawRecordWriteView records = state.records(static_cast<std::size_t>(T));
    const ops::QsaCacheView cache = state.cache();

    ops::qsa_prepare(ops::QsaProjectionView{query, key, value, index_query, index_key}, parameters,
                     batch, state.before_tail(), cache, workspace_, records, rotated_query,
                     rotated_index_query, stream_);

    ops::QsaBlockSelection selected{
        .selection = activations_[0]->alloc(DType::I32, {TextConfig::index_block_budget, T, kBatch}),
        .counts    = activations_[0]->alloc(DType::I32, {T, kBatch}),
    };
    ops::qsa_indexer(rotated_index_query, ops::ConstQsaIndexCacheView{cache.index_k_pages,
                                                                      cache.page_table},
                     batch, static_cast<std::uint32_t>(T), workspace_, selected, stream_);
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
    NINFER_QWEN4EXP_LAYER_DUMP("index_query", layer, rotated_index_query);
    NINFER_QWEN4EXP_LAYER_DUMP("index_keys", layer, cache.index_k_pages);
#endif

    Tensor attention = activations_[0]->alloc(DType::BF16, {kHeadDim, TextConfig::query_heads, T, kBatch});
    ops::qsa_sparse_gqa(rotated_query, output_gate, cache, batch, selected,
                        static_cast<std::uint32_t>(T), workspace_, attention, stream_);
    ops::linear(attention.view({TextConfig::query_width, T}), weights.output, mixer_out,
                ops::LinearPolicy::A16Only, workspace_, stream_);

    const ops::QsaRawRecordReadView read_records{records.raw_keys, records.rope_positions};
    ops::QsaTailStateView after = state.tail();
    ops::qsa_tail_extract(state.const_tail(), read_records, records.base_positions, valid_columns,
                          after, stream_);
    state.publish_tail();
}

inline void TextTrunk::run_ple(std::int32_t T, const Tensor& residual, Tensor& updated,
                               Tensor& normalized) {
    const auto& ple = models_[0]->ple;
    TextState& state = *states_[0];
    WorkspaceArena& workspace_ = *workspaces_[0];
    const std::int32_t entries = static_cast<std::int32_t>(ops::kPleHeads) * T * kBatch;

    Tensor embedding = activations_[0]->alloc(DType::BF16, {TextConfig::ple_embed_dim, T, kBatch});
    ops::ple_lookup_rows(state.ple_table(), state.ple_rows(), static_cast<std::size_t>(entries),
                         state.ple_staged_rows(), embedding, stream_);

    Tensor projected_key   = activations_[0]->alloc(DType::BF16, {kChannels, T, kBatch});
    Tensor projected_value = activations_[0]->alloc(DType::BF16, {kHidden, T, kBatch});
    ops::linear(embedding, ple.key, projected_key, ops::LinearPolicy::A16Only, workspace_, stream_);
    ops::linear(embedding, ple.value, projected_value, ops::LinearPolicy::A16Only, workspace_, stream_);

    Tensor gated_value     = activations_[0]->alloc(DType::BF16, {kChannels, T, kBatch});
    Tensor normalized_conv = activations_[0]->alloc(DType::BF16, {kChannels, T, kBatch});
    Tensor offsets         = activations_[0]->alloc(DType::I32, {kBatch});
    Tensor counts          = activations_[0]->alloc(DType::I32, {kBatch});
    const std::int32_t live         = T;
    const std::int32_t column_start = 0;
    CUDA_CHECK(cudaMemcpyAsync(offsets.data, &column_start, 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(counts.data, &live, 4, cudaMemcpyHostToDevice, stream_));

    ops::ple_gate(projected_key, projected_value, residual, ple.norm_key, ple.norm_query,
                  ple.norm_conv, TextConfig::rms_epsilon, gated_value, normalized_conv, stream_);

    const Tensor next_norm =
        models_[0]->layers[TextConfig::ple_layer].attention_hc.norm.view({kChannels});
    ops::ple_conv_add_norm(state.ple_history(), normalized_conv, residual, gated_value, ple.kernel,
                           offsets, counts, next_norm, TextConfig::rms_epsilon, updated, normalized,
                           stream_);

    Tensor after_history = activations_[0]->alloc(DType::BF16,
                                            {kChannels, static_cast<std::int32_t>(ops::kPleConvWindow),
                                             kBatch});
    ops::ple_state_extract(state.ple_history(), normalized_conv, counts, after_history, stream_);
    CUDA_CHECK(cudaMemcpyAsync(state.ple_history().data, after_history.data,
                               static_cast<std::size_t>(kChannels) * ops::kPleConvWindow * kBatch * 2,
                               cudaMemcpyDeviceToDevice, stream_));
}

inline TokenId TextTrunk::forward_single(std::span<const TokenId> tokens) {
    const ModelView& model = *models_[0];
    TextState& state       = *states_[0];
    WorkspaceArena& workspace_ = *workspaces_[0];
    const std::int32_t T = static_cast<std::int32_t>(tokens.size());
    auto scope = workspace_.scope();
    state.prepare_ple_rows(tokens);

    Tensor ids = activations_[0]->alloc(DType::I32, {T});
    CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data(), static_cast<std::size_t>(T) * 4,
                               cudaMemcpyHostToDevice, stream_));

    const auto forward_scope = activations_[0]->scope();
    Tensor embedding_out = activations_[0]->alloc(DType::BF16, {kHidden, T});
    ops::embedding(ids, model.embedding, embedding_out, stream_);

    Tensor residual    = activations_[0]->alloc(DType::BF16, {kHidden, kStreams, T, kBatch});
    Tensor normalized  = activations_[0]->alloc(DType::BF16, {kHidden, kStreams, T, kBatch});
    Tensor stage       = activations_[0]->alloc(DType::BF16, {kHidden, kStreams, T, kBatch});
    Tensor block_input = activations_[0]->alloc(DType::BF16, {kHidden, T});
    Tensor injection   = activations_[0]->alloc(DType::FP32, {kStreams, T});
    Tensor mixer_out   = activations_[0]->alloc(DType::BF16, {kHidden, T});
    Tensor moe_out     = activations_[0]->alloc(DType::BF16, {kHidden, T});
    Tensor sample      = activations_[0]->alloc(DType::BF16, {kHidden, T});
    Tensor logits      = activations_[0]->alloc(DType::BF16, {TextConfig::vocab, T});
    Tensor positions   = activations_[0]->alloc(DType::I32, {T, 1, kBatch});
    Tensor valid_columns = activations_[0]->alloc(DType::I32, {kBatch});
    Tensor table_rows    = activations_[0]->alloc(DType::I32, {kBatch});

    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(T));
    for (std::int32_t t = 0; t < T; ++t) {
        host_positions[static_cast<std::size_t>(t)] =
            static_cast<std::int32_t>(state.next_position() + static_cast<std::uint32_t>(t));
    }
    CUDA_CHECK(cudaMemcpyAsync(positions.data, host_positions.data(),
                               static_cast<std::size_t>(T) * 4, cudaMemcpyHostToDevice, stream_));
    const std::int32_t live = T;
    const std::int32_t row  = 0;
    CUDA_CHECK(cudaMemcpyAsync(valid_columns.data, &live, 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(table_rows.data, &row, 4, cudaMemcpyHostToDevice, stream_));

    ops::hc_expand_norm(embedding_out, model.layers[0].attention_hc.norm, TextConfig::rms_epsilon,
                        residual, normalized, stream_);

    for (std::size_t layer = 0; layer < model.loaded_layers; ++layer) {
        const auto layer_scope = activations_[0]->scope();
        const auto& attention_hc = model.layers[layer].attention_hc;
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("residual_in", layer, residual);
        NINFER_QWEN4EXP_LAYER_DUMP("normalized_in", layer, normalized);
#endif
        ops::hc_mix(normalized, ops::HcMixWeights{attention_hc.down_injection, attention_hc.up},
                    workspace_, block_input, injection, stream_);
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("attention_block", layer, block_input);
        NINFER_QWEN4EXP_LAYER_DUMP("attention_injection", layer, injection);
#endif

        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            run_qsa(layer, T, block_input, positions, valid_columns, table_rows, mixer_out);
        } else {
            run_gdn(layer, T, block_input, mixer_out);
        }
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("mixer", layer, mixer_out);
#endif

        const auto& moe_hc = model.layers[layer].moe_hc;
        ops::hc_combine_norm(residual, mixer_out, injection, moe_hc.norm, TextConfig::rms_epsilon,
                             stage, normalized, stream_);
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("stage", layer, stage);
        NINFER_QWEN4EXP_LAYER_DUMP("normalized_moe", layer, normalized);
#endif

        ops::hc_mix(normalized, ops::HcMixWeights{moe_hc.down_injection, moe_hc.up}, workspace_,
                    block_input, injection, stream_);
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("moe_block", layer, block_input);
        NINFER_QWEN4EXP_LAYER_DUMP("moe_injection", layer, injection);
#endif

        CUDA_CHECK(cudaMemsetAsync(moe_out.data, 0, moe_out.bytes(), stream_));
        Variant::post_mixer(block_input, model.layers[layer].post_mixer, moe_out, workspace_, stream_);
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("moe", layer, moe_out);
#endif

        if (layer + 1 == static_cast<std::size_t>(TextConfig::ple_layer)) {
            ops::hc_combine(stage, moe_out, injection, residual, stream_);
            run_ple(T, residual, stage, normalized);
            std::swap(residual, stage);
        } else {
            const Tensor& next_norm = layer + 1 < model.loaded_layers
                                          ? model.layers[layer + 1].attention_hc.norm
                                          : model.tail_norm;
            ops::hc_combine_norm(stage, moe_out, injection, next_norm, TextConfig::rms_epsilon,
                                 residual, normalized, stream_);
        }
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("residual_out", layer, residual);
        NINFER_QWEN4EXP_LAYER_DUMP("normalized_out", layer, normalized);
#endif
    }

    ops::hc_final_mix(normalized,
                      ops::HcFinalMixWeights{model.final_hc.down_injection, model.final_hc.up},
                      workspace_, sample, stream_);
    ops::linear(sample, model.output_head, logits, ops::LinearPolicy::A16Only, workspace_, stream_);
    NINFER_QWEN4EXP_LAST_STAGE(sample, logits);

    Tensor winner = activations_[0]->alloc(DType::I32, {T});
    ops::argmax(logits, winner, TextConfig::vocab, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::int32_t token = 0;
    CUDA_CHECK(cudaMemcpy(&token,
                          static_cast<const std::byte*>(winner.data) +
                              static_cast<std::size_t>(T - 1) * 4,
                          4, cudaMemcpyDeviceToHost));
    state.advance_position(static_cast<std::uint32_t>(T));
    return static_cast<TokenId>(token);
}

inline void TextTrunk::run_gdn_parallel(std::size_t layer, std::int32_t T,
                                        const std::array<Tensor, kMaximumDevices>& block_input,
                                        std::array<Tensor, kMaximumDevices>& mixer_out,
                                        const std::array<Tensor, kMaximumDevices>& staging) {
    std::array<WorkspaceArena*, kMaximumDevices> arena_ptrs{};
    for (int rank = 0; rank < tp_; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        arena_ptrs[slot] = workspaces_[slot];
    }
    std::array<Tensor, kMaximumDevices> staging_out{};
    std::array<Weight, kMaximumDevices> output_weight{};
    std::array<Tensor, kMaximumDevices> projected{};
    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        WorkspaceArena& workspace_ = *workspaces_[slot];
        const auto& weights = models_[slot]->layers[layer].gdn_projection;
        TextState& state    = *states_[slot];
        const std::int32_t key_width   = state.gdn_key_heads() * TextConfig::gdn_head_dim;
        const std::int32_t value_width = state.gdn_value_heads() * TextConfig::gdn_head_dim;
        const std::int32_t conv_channels = state.conv_channels();

        Tensor q    = activations_[slot]->alloc(DType::BF16, {key_width, T});
        Tensor k    = activations_[slot]->alloc(DType::BF16, {key_width, T});
        Tensor v    = activations_[slot]->alloc(DType::BF16, {value_width, T});
        Tensor z    = activations_[slot]->alloc(DType::BF16, {value_width, T});
        Tensor g    = activations_[slot]->alloc(DType::FP32, {TextConfig::gdn_value_heads, T});
        Tensor beta = activations_[slot]->alloc(DType::FP32, {TextConfig::gdn_value_heads, T});
        Variant::gdn_projection_control(block_input[slot], weights, q, k, v, z, g, beta, workspace_,
                                        stream_for(rank));

        Tensor conv_kernel_rank = activations_[slot]->alloc(DType::BF16, {conv_channels, 4});
        const Tensor conv_full = weights.convolution;
        const auto copy_kernel = [&](std::int32_t source_column, std::int32_t width,
                                     std::int32_t destination_column) {
            for (std::int32_t tap = 0; tap < 4; ++tap) {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::byte*>(conv_kernel_rank.data) +
                        (static_cast<std::size_t>(tap) * conv_channels + destination_column) *
                            sizeof(std::uint16_t),
                    static_cast<const std::byte*>(conv_full.data) +
                        (static_cast<std::size_t>(tap) * TextConfig::gdn_conv_channels +
                         source_column) *
                            sizeof(std::uint16_t),
                    static_cast<std::size_t>(width) * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToDevice, stream_for(rank)));
            }
        };
        copy_kernel(rank * key_width, key_width, 0);
        copy_kernel(TextConfig::gdn_key_width + rank * key_width, key_width, key_width);
        copy_kernel(2 * TextConfig::gdn_key_width + rank * value_width, value_width, 2 * key_width);

        Tensor conv_in  = activations_[slot]->alloc(DType::BF16, {conv_channels, T});
        Tensor conv_out = activations_[slot]->alloc(DType::BF16, {conv_channels, T});
        const auto copy_segment = [&](const Tensor& source, std::int32_t column) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                static_cast<std::byte*>(conv_in.data) +
                    static_cast<std::size_t>(column) * sizeof(std::uint16_t),
                static_cast<std::size_t>(conv_channels) * sizeof(std::uint16_t),
                source.data, static_cast<std::size_t>(source.ne[0]) * sizeof(std::uint16_t),
                static_cast<std::size_t>(source.ne[0]) * sizeof(std::uint16_t),
                static_cast<std::size_t>(T), cudaMemcpyDeviceToDevice, stream_for(rank)));
        };
        copy_segment(q, 0);
        copy_segment(k, key_width);
        copy_segment(v, 2 * key_width);

        const std::optional<std::size_t> state_slot = state.gdn_state_index(layer);
        if (!state_slot.has_value()) { throw std::logic_error("a QSA layer has no GDN state slot"); }
        const std::size_t gdn_index = *state_slot;
        Tensor conv_state = state.gdn_conv_state(gdn_index);
        ops::causal_conv1d_silu(conv_in, conv_kernel_rank, conv_state, conv_state, conv_out,
                                stream_for(rank));
        ops::extract_bf16_columns(conv_out, 0, q, stream_for(rank));
        ops::extract_bf16_columns(conv_out, key_width, k, stream_for(rank));
        ops::extract_bf16_columns(conv_out, 2 * key_width, v, stream_for(rank));

        Tensor o  = activations_[slot]->alloc(DType::BF16, {value_width, T});
        Tensor on = activations_[slot]->alloc(DType::BF16, {value_width, T});
        Tensor q_recurrent = q.view({128, state.gdn_key_heads(), T});
        Tensor k_recurrent = k.view({128, state.gdn_key_heads(), T});
        Tensor v_recurrent = v.view({128, state.gdn_value_heads(), T});
        Tensor o_recurrent = o.view({128, state.gdn_value_heads(), T});
        Tensor o_normalized = on.view({128, state.gdn_value_heads(), T});
        Tensor recurrent_state = state.gdn_recurrent_state(gdn_index);
        Tensor g_rank    = activations_[slot]->alloc(DType::FP32, {state.gdn_value_heads(), T});
        Tensor beta_rank = activations_[slot]->alloc(DType::FP32, {state.gdn_value_heads(), T});
        CUDA_CHECK(cudaMemcpy2DAsync(
            g_rank.data, static_cast<std::size_t>(state.gdn_value_heads()) * 4,
            static_cast<const std::byte*>(g.data) +
                static_cast<std::size_t>(rank * state.gdn_value_heads()) * 4,
            static_cast<std::size_t>(TextConfig::gdn_value_heads) * 4,
            static_cast<std::size_t>(state.gdn_value_heads()) * 4, static_cast<std::size_t>(T),
            cudaMemcpyDeviceToDevice, stream_for(rank)));
        CUDA_CHECK(cudaMemcpy2DAsync(
            beta_rank.data, static_cast<std::size_t>(state.gdn_value_heads()) * 4,
            static_cast<const std::byte*>(beta.data) +
                static_cast<std::size_t>(rank * state.gdn_value_heads()) * 4,
            static_cast<std::size_t>(TextConfig::gdn_value_heads) * 4,
            static_cast<std::size_t>(state.gdn_value_heads()) * 4, static_cast<std::size_t>(T),
            cudaMemcpyDeviceToDevice, stream_for(rank)));
        ops::gated_delta_net(q_recurrent, k_recurrent, v_recurrent, g_rank, beta_rank, kGdnScale,
                             true, workspace_, recurrent_state, o_recurrent, stream_for(rank));
        ops::gated_rmsnorm(o_recurrent, weights.norm,
                           z.view({128, state.gdn_value_heads(), T}), TextConfig::rms_epsilon,
                           o_normalized, stream_for(rank));
        projected[slot] = on.view({value_width, T});
        output_weight[slot] = weights.output;
        staging_out[slot] = mixer_out[slot];
        arena_ptrs[slot] = workspaces_[slot];
    });
    ops::linear_row_parallel(projected, output_weight, staging_out, staging,
                             ops::LinearPolicy::A16Only, arena_ptrs, *execution_, *comms_);
}

inline void TextTrunk::run_qsa_parallel(
    std::size_t layer, std::int32_t T, const std::array<Tensor, kMaximumDevices>& block_input,
    const std::array<Tensor, kMaximumDevices>& positions,
    const std::array<Tensor, kMaximumDevices>& valid_columns,
    const std::array<Tensor, kMaximumDevices>& table_rows,
    std::array<Tensor, kMaximumDevices>& mixer_out,
    const std::array<Tensor, kMaximumDevices>& staging) {
    const std::int32_t index_dim = TextConfig::index_head_dim;
    std::array<Tensor, kMaximumDevices> projected{};
    std::array<Weight, kMaximumDevices> output_weight{};
    std::array<Tensor, kMaximumDevices> staging_out{};
    std::array<WorkspaceArena*, kMaximumDevices> arena_ptrs{};

    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        WorkspaceArena& workspace_ = *workspaces_[slot];
        const auto& weights = models_[slot]->layers[layer].attention_projection;
        TextState& state    = *states_[slot];
        const std::int32_t query_heads = TextConfig::query_heads / tp_;
        const std::int32_t kv_heads    = state.kv_heads();
        const std::int32_t query_width = query_heads * TextConfig::head_dim;
        const std::int32_t kv_width    = kv_heads * TextConfig::head_dim;

        Tensor query_flat       = activations_[slot]->alloc(DType::BF16, {query_width, T});
        Tensor output_gate_flat = activations_[slot]->alloc(DType::BF16, {query_width, T});
        Tensor key_flat         = activations_[slot]->alloc(DType::BF16, {kv_width, T});
        Tensor value_flat       = activations_[slot]->alloc(DType::BF16, {kv_width, T});
        Tensor index_query_flat =
            activations_[slot]->alloc(DType::BF16, {index_dim * TextConfig::index_heads, T});
        Tensor index_key_flat =
            activations_[slot]->alloc(DType::BF16, {index_dim * TextConfig::index_kv_heads, T});
        Tensor query       = query_flat.view({kHeadDim, query_heads, T, kBatch});
        Tensor output_gate = output_gate_flat.view({kHeadDim, query_heads, T, kBatch});
        Tensor key         = key_flat.view({kHeadDim, kv_heads, T, kBatch});
        Tensor value       = value_flat.view({kHeadDim, kv_heads, T, kBatch});
        Tensor index_query = index_query_flat.view({index_dim, TextConfig::index_heads, T, kBatch});
        Tensor index_key   = index_key_flat.view({index_dim, TextConfig::index_kv_heads, T, kBatch});
        Variant::attention_projection(block_input[slot], weights, query_flat, output_gate_flat,
                                      key_flat, value_flat, index_query_flat, index_key_flat,
                                      workspace_, stream_for(rank));

        Tensor rotated_query =
            activations_[slot]->alloc(DType::BF16, {kHeadDim, query_heads, T, kBatch});
        Tensor rotated_index_query =
            activations_[slot]->alloc(DType::BF16, {index_dim, TextConfig::index_heads, T, kBatch});

        const ops::QsaPrepareParameters parameters{
            .query_norm        = weights.query_norm,
            .key_norm          = weights.key_norm,
            .index_query_norm  = weights.index_query_norm,
            .index_key_norm    = weights.index_key_norm,
            .rms_epsilon       = TextConfig::rms_epsilon,
            .rotary_dim        = TextConfig::rotary_dim,
            .theta             = static_cast<float>(TextConfig::rope_theta),
        };
        const ops::QsaBatchView batch{.positions = positions[slot],
                                      .valid_columns = valid_columns[slot],
                                      .table_rows = table_rows[slot]};
        ops::QsaRawRecordWriteView records = state.records(static_cast<std::size_t>(T));
        const ops::QsaCacheView cache = state.cache();

        ops::qsa_prepare(ops::QsaProjectionView{query, key, value, index_query, index_key},
                         parameters, batch, state.before_tail(), cache, workspace_, records,
                         rotated_query, rotated_index_query, stream_for(rank));

        ops::QsaBlockSelection selected{
            .selection = activations_[slot]->alloc(DType::I32,
                                                    {TextConfig::index_block_budget, T, kBatch}),
            .counts    = activations_[slot]->alloc(DType::I32, {T, kBatch}),
        };
        ops::qsa_indexer(rotated_index_query,
                         ops::ConstQsaIndexCacheView{cache.index_k_pages, cache.page_table}, batch,
                         static_cast<std::uint32_t>(T), workspace_, selected, stream_for(rank));
#ifdef NINFER_QWEN4EXP_LAYER_DUMP
        NINFER_QWEN4EXP_LAYER_DUMP("index_query", layer, rotated_index_query);
        NINFER_QWEN4EXP_LAYER_DUMP("index_keys", layer, cache.index_k_pages);
#endif

        Tensor attention =
            activations_[slot]->alloc(DType::BF16, {kHeadDim, query_heads, T, kBatch});
        ops::qsa_sparse_gqa(rotated_query, output_gate, cache, batch, selected,
                            static_cast<std::uint32_t>(T), workspace_, attention, stream_for(rank));

        const ops::QsaRawRecordReadView read_records{records.raw_keys, records.rope_positions};
        ops::QsaTailStateView after = state.tail();
        ops::qsa_tail_extract(state.const_tail(), read_records, records.base_positions,
                              valid_columns[slot], after, stream_for(rank));
        state.publish_tail();

        projected[slot] = attention.view({query_width, T});
        output_weight[slot] = weights.output;
        staging_out[slot] = mixer_out[slot];
        arena_ptrs[slot] = workspaces_[slot];
    });
    ops::linear_row_parallel(projected, output_weight, staging_out, staging,
                             ops::LinearPolicy::A16Only, arena_ptrs, *execution_, *comms_);
}

inline void TextTrunk::run_ple_parallel(std::int32_t T,
                                        std::array<Tensor, kMaximumDevices>& residual,
                                        std::array<Tensor, kMaximumDevices>& updated,
                                        std::array<Tensor, kMaximumDevices>& normalized) {
    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        WorkspaceArena& workspace_ = *workspaces_[slot];
        const auto& ple = models_[slot]->ple;
        TextState& state = *states_[slot];
        const std::int32_t entries = static_cast<std::int32_t>(ops::kPleHeads) * T * kBatch;

        Tensor embedding = activations_[slot]->alloc(DType::BF16, {TextConfig::ple_embed_dim, T, kBatch});
        ops::ple_lookup_rows(state.ple_table(), state.ple_rows(),
                             static_cast<std::size_t>(entries), state.ple_staged_rows(), embedding,
                             stream_for(rank));

        Tensor projected_key   = activations_[slot]->alloc(DType::BF16, {kChannels, T, kBatch});
        Tensor projected_value = activations_[slot]->alloc(DType::BF16, {kHidden, T, kBatch});
        ops::linear(embedding, ple.key, projected_key, ops::LinearPolicy::A16Only, workspace_,
                    stream_for(rank));
        ops::linear(embedding, ple.value, projected_value, ops::LinearPolicy::A16Only, workspace_,
                    stream_for(rank));

        Tensor gated_value     = activations_[slot]->alloc(DType::BF16, {kChannels, T, kBatch});
        Tensor normalized_conv = activations_[slot]->alloc(DType::BF16, {kChannels, T, kBatch});
        Tensor offsets         = activations_[slot]->alloc(DType::I32, {kBatch});
        Tensor counts          = activations_[slot]->alloc(DType::I32, {kBatch});
        const std::int32_t live         = T;
        const std::int32_t column_start = 0;
        CUDA_CHECK(cudaMemcpyAsync(offsets.data, &column_start, 4, cudaMemcpyHostToDevice,
                                   stream_for(rank)));
        CUDA_CHECK(cudaMemcpyAsync(counts.data, &live, 4, cudaMemcpyHostToDevice,
                                   stream_for(rank)));

        ops::ple_gate(projected_key, projected_value, residual[slot], ple.norm_key, ple.norm_query,
                      ple.norm_conv, TextConfig::rms_epsilon, gated_value, normalized_conv,
                      stream_for(rank));

        const Tensor next_norm =
            models_[slot]->layers[TextConfig::ple_layer].attention_hc.norm.view({kChannels});
        ops::ple_conv_add_norm(state.ple_history(), normalized_conv, residual[slot], gated_value,
                               ple.kernel, offsets, counts, next_norm, TextConfig::rms_epsilon,
                               updated[slot], normalized[slot], stream_for(rank));

        Tensor after_history = activations_[slot]->alloc(
            DType::BF16, {kChannels, static_cast<std::int32_t>(ops::kPleConvWindow), kBatch});
        ops::ple_state_extract(state.ple_history(), normalized_conv, counts, after_history,
                               stream_for(rank));
        CUDA_CHECK(cudaMemcpyAsync(
            state.ple_history().data, after_history.data,
            static_cast<std::size_t>(kChannels) * ops::kPleConvWindow * kBatch * 2,
            cudaMemcpyDeviceToDevice, stream_for(rank)));
    });
}

inline TokenId TextTrunk::forward_parallel(std::span<const TokenId> tokens) {
    const std::int32_t T = static_cast<std::int32_t>(tokens.size());
    for_each_rank([&](int rank) { states_[static_cast<std::size_t>(rank)]->prepare_ple_rows(tokens); });

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
        ids[slot] = activations_[slot]->alloc(DType::I32, {T});
        CUDA_CHECK(cudaMemcpyAsync(ids[slot].data, tokens.data(), static_cast<std::size_t>(T) * 4,
                                   cudaMemcpyHostToDevice, stream));
        embedding_out[slot] = activations_[slot]->alloc(DType::BF16, {kHidden, T});
        ops::embedding(ids[slot], models_[slot]->embedding, embedding_out[slot], stream);

        residual[slot]    = activations_[slot]->alloc(DType::BF16, {kHidden, kStreams, T, kBatch});
        normalized[slot]  = activations_[slot]->alloc(DType::BF16, {kHidden, kStreams, T, kBatch});
        stage[slot]       = activations_[slot]->alloc(DType::BF16, {kHidden, kStreams, T, kBatch});
        block_input[slot] = activations_[slot]->alloc(DType::BF16, {kHidden, T});
        injection[slot]   = activations_[slot]->alloc(DType::FP32, {kStreams, T});
        mixer_out[slot]   = activations_[slot]->alloc(DType::BF16, {kHidden, T});
        moe_out[slot]     = activations_[slot]->alloc(DType::BF16, {kHidden, T});
        sample[slot]      = activations_[slot]->alloc(DType::BF16, {kHidden, T});
        logits[slot]      = activations_[slot]->alloc(DType::BF16, {TextConfig::vocab, T});
        logits_part[slot] =
            activations_[slot]->alloc(DType::BF16, {TextConfig::vocab / tp_, T});
        positions[slot]     = activations_[slot]->alloc(DType::I32, {T, 1, kBatch});
        valid_columns[slot] = activations_[slot]->alloc(DType::I32, {kBatch});
        table_rows[slot]    = activations_[slot]->alloc(DType::I32, {kBatch});
        staging[slot]       = workspaces_[slot]->alloc(DType::BF16, {kHidden, T});
        CUDA_CHECK(cudaMemcpyAsync(positions[slot].data, host_positions.data(),
                                   static_cast<std::size_t>(T) * 4, cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(valid_columns[slot].data, &live, 4, cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(table_rows[slot].data, &row, 4, cudaMemcpyHostToDevice, stream));
        ops::hc_expand_norm(embedding_out[slot],
                            models_[slot]->layers[0].attention_hc.norm, TextConfig::rms_epsilon,
                            residual[slot], normalized[slot], stream);
    });

    std::array<WorkspaceArena*, kMaximumDevices> workspace_ptrs{};
    for (int rank = 0; rank < tp_; ++rank) {
        workspace_ptrs[static_cast<std::size_t>(rank)] = workspaces_[static_cast<std::size_t>(rank)];
    }

    for (std::size_t layer = 0; layer < models_[0]->loaded_layers; ++layer) {
        std::array<std::optional<DeviceArena::Scope>, kMaximumDevices> layer_activation_scopes;
        std::array<std::optional<DeviceArena::Scope>, kMaximumDevices> layer_workspace_scopes;
        for (int rank = 0; rank < tp_; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            layer_activation_scopes[slot].emplace(activations_[slot]->scope());
            layer_workspace_scopes[slot].emplace(workspaces_[slot]->scope());
        }
        for_each_rank([&](int rank) {
            const auto slot = static_cast<std::size_t>(rank);
            const auto& attention_hc = models_[slot]->layers[layer].attention_hc;
            ops::hc_mix(normalized[slot],
                        ops::HcMixWeights{attention_hc.down_injection, attention_hc.up},
                        *workspaces_[slot], block_input[slot], injection[slot], stream_for(rank));
        });
        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            run_qsa_parallel(layer, T, block_input, positions, valid_columns, table_rows,
                             mixer_out, staging);
        } else {
            run_gdn_parallel(layer, T, block_input, mixer_out, staging);
        }
        for_each_rank([&](int rank) {
            const auto slot = static_cast<std::size_t>(rank);
            const auto& moe_hc = models_[slot]->layers[layer].moe_hc;
            ops::hc_combine_norm(residual[slot], mixer_out[slot], injection[slot], moe_hc.norm,
                                 TextConfig::rms_epsilon, stage[slot], normalized[slot],
                                 stream_for(rank));
            ops::hc_mix(normalized[slot],
                        ops::HcMixWeights{moe_hc.down_injection, moe_hc.up}, *workspaces_[slot],
                        block_input[slot], injection[slot], stream_for(rank));
            CUDA_CHECK(cudaMemsetAsync(moe_out[slot].data, 0, moe_out[slot].bytes(),
                                       stream_for(rank)));
        });

        std::array<ops::MoeNvfp4RouteView, kMaximumDevices> routes{};
        for_each_rank([&](int rank) {
            const auto slot = static_cast<std::size_t>(rank);
            const std::int32_t top_k = TextConfig::experts_per_token;
            Tensor route_ids = activations_[slot]->alloc(DType::I32, {top_k, T});
            Tensor route_coefficients = activations_[slot]->alloc(DType::FP32, {top_k, T});
            Tensor route_gate = activations_[slot]->alloc(DType::FP32, {T});
            ops::moe_nvfp4_route(block_input[slot], models_[slot]->layers[layer].post_mixer,
                                 route_ids, route_coefficients, route_gate, stream_for(rank));
            routes[slot] = ops::MoeNvfp4RouteView{route_ids, route_coefficients, route_gate};
            Variant::post_mixer_routed(block_input[slot],
                                       models_[slot]->layers[layer].post_mixer, routes[slot],
                                       moe_out[slot], stream_for(rank));
        });
        ops::allreduce_sum(moe_out, staging, *execution_, *comms_);
        for_each_rank([&](int rank) {
            const auto slot = static_cast<std::size_t>(rank);
            Variant::post_mixer_shared(block_input[slot],
                                       models_[slot]->layers[layer].post_mixer, routes[slot],
                                       moe_out[slot], stream_for(rank));
        });

        if (layer + 1 == static_cast<std::size_t>(TextConfig::ple_layer)) {
            for_each_rank([&](int rank) {
                const auto slot = static_cast<std::size_t>(rank);
                ops::hc_combine(stage[slot], moe_out[slot], injection[slot], residual[slot],
                                stream_for(rank));
            });
            run_ple_parallel(T, residual, stage, normalized);
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
                ops::hc_combine_norm(stage[slot], moe_out[slot], injection[slot], next_norm,
                                     TextConfig::rms_epsilon, residual[slot], normalized[slot],
                                     stream_for(rank));
            });
        }
    }

    std::array<Weight, kMaximumDevices> head{};
    std::array<Tensor, kMaximumDevices> winner{};
    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        ops::hc_final_mix(normalized[slot],
                          ops::HcFinalMixWeights{models_[slot]->final_hc.down_injection,
                                                 models_[slot]->final_hc.up},
                          *workspaces_[slot], sample[slot], stream_for(rank));
        head[slot] = models_[slot]->output_head;
        ops::linear(sample[slot], head[slot], logits_part[slot], ops::LinearPolicy::A16Only,
                    *workspaces_[slot], stream_for(rank));
    });
    const std::int32_t vocab = TextConfig::vocab;
    for (std::int32_t token = 0; token < T; ++token) {
        std::array<Tensor, kMaximumDevices> destination_token{};
        std::array<Tensor, kMaximumDevices> part_token{};
        for (int rank = 0; rank < tp_; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            destination_token[slot] = Tensor(
                static_cast<std::byte*>(logits[slot].data) +
                    static_cast<std::size_t>(token) * vocab * 2,
                DType::BF16, {1, vocab});
            part_token[slot] = Tensor(
                static_cast<std::byte*>(logits_part[slot].data) +
                    static_cast<std::size_t>(token) * (vocab / tp_) * 2,
                DType::BF16, {1, vocab / tp_});
        }
        ops::allgather_rows(destination_token, part_token, *execution_, *comms_);
    }
    for_each_rank([&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        NINFER_QWEN4EXP_LAST_STAGE(sample[slot], logits[slot]);
        winner[slot] = activations_[slot]->alloc(DType::I32, {T});
        ops::argmax(logits[slot], winner[slot], TextConfig::vocab, stream_for(rank));
    });

    CUDA_CHECK(cudaSetDevice(execution_->dev[0]->device));
    CUDA_CHECK(cudaStreamSynchronize(execution_->dev[0]->stream));
    std::int32_t token = 0;
    CUDA_CHECK(cudaMemcpy(&token,
                          static_cast<const std::byte*>(winner[0].data) +
                              static_cast<std::size_t>(T - 1) * 4,
                          4, cudaMemcpyDeviceToHost));
    for (int rank = 0; rank < tp_; ++rank) {
        states_[static_cast<std::size_t>(rank)]->advance_position(static_cast<std::uint32_t>(T));
    }
    return static_cast<TokenId>(token);
}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
