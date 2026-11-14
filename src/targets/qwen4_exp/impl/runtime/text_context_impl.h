#pragma once

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/scatter.h"
#include "targets/qwen4_exp/impl/runtime/text_context.h"

#include <algorithm>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

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
    : device_scope_(device.device), ple_source_(model.ple.rows), stream_(device.stream),
      layers_(model.loaded_layers), forward_tokens_(forward_tokens == 0 ? 1 : forward_tokens),
      pages_(kv_capacity_tokens / 64 + 1),
      gdn_layers_(layers_), gdn_index_(layers_),
      conv_states_(layers_ * static_cast<std::size_t>(TextConfig::gdn_conv_channels) * 3 * 2),
      recurrent_states_(layers_ * 128 * 128 * static_cast<std::size_t>(TextConfig::gdn_value_heads) * 4),
      k_pages_(static_cast<std::size_t>(kHeadDim) * 64 * TextConfig::kv_heads * pages_ * 2),
      v_pages_(static_cast<std::size_t>(kHeadDim) * 64 * TextConfig::kv_heads * pages_ * 2),
      index_k_pages_(static_cast<std::size_t>(TextConfig::index_head_dim) * 16 *
                     TextConfig::index_kv_heads * pages_ * 2),
      page_table_(pages_ * 4),
      tail_raw_keys_(static_cast<std::size_t>(TextConfig::index_head_dim) * 4 * kBatch * 2),
      tail_first_rope_(3 * kBatch * 4),
      after_raw_keys_(static_cast<std::size_t>(TextConfig::index_head_dim) * 4 * kBatch * 2),
      after_first_rope_(3 * kBatch * 4),
      record_raw_keys_(static_cast<std::size_t>(TextConfig::index_head_dim) * forward_tokens_ * kBatch * 2),
      record_rope_positions_(3 * forward_tokens_ * kBatch * 4),
      record_base_positions_(kBatch * 4),
      ple_history_(static_cast<std::size_t>(kChannels) * ops::kPleConvWindow * kBatch * 2),
      ple_rows_(ops::kPleHeads * forward_tokens_ * kBatch * 4),
      ple_staged_rows_(ops::kPleHeads * forward_tokens_ * kBatch * 128),
      ple_divisors_(ops::kPleShards * 4) {
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
    std::int32_t base = 0;
    record_base_positions_.copy_to_host(&base, 4);
    return base;
}

inline Tensor TextState::gdn_conv_state(std::size_t gdn_index) const {
    const std::size_t layer = gdn_layers_.at(gdn_index);
    return Tensor(conv_states_.p + layer * static_cast<std::size_t>(TextConfig::gdn_conv_channels) * 3 * 2,
                  DType::BF16, {3, TextConfig::gdn_conv_channels});
}

inline Tensor TextState::gdn_recurrent_state(std::size_t gdn_index) const {
    const std::size_t layer = gdn_layers_.at(gdn_index);
    return Tensor(recurrent_states_.p +
                      layer * 128 * 128 * static_cast<std::size_t>(TextConfig::gdn_value_heads) * 4,
                  DType::FP32, {128, 128, TextConfig::gdn_value_heads});
}

inline std::optional<std::size_t> TextState::gdn_state_index(std::size_t layer) const {
    return gdn_index_.at(layer);
}

inline std::size_t TextState::gdn_state_entries() const noexcept { return gdn_layers_.size(); }

inline ops::QsaCacheView TextState::cache() const {
    const std::int32_t pages = static_cast<std::int32_t>(pages_);
    return ops::QsaCacheView{
        .k_pages       = Tensor(k_pages_.p, DType::BF16, {kHeadDim, 64, TextConfig::kv_heads, pages}),
        .v_pages       = Tensor(v_pages_.p, DType::FP16, {kHeadDim, 64, TextConfig::kv_heads, pages}),
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
    CUDA_CHECK(cudaMemcpyAsync(tail_raw_keys_.p, after_raw_keys_.p,
                               static_cast<std::size_t>(TextConfig::index_head_dim) * 4 * kBatch * 2,
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(tail_first_rope_.p, after_first_rope_.p, 3 * kBatch * 4,
                               cudaMemcpyDeviceToDevice, stream_));
}

inline TextTrunk::TextTrunk(const ModelView& model, TextState& state, WorkspaceArena& activations,
                            WorkspaceArena& workspace, cudaStream_t stream)
    : model_(model), state_(state), activations_(activations), workspace_(workspace),
      stream_(stream) {}

inline TokenId TextTrunk::prefill(std::span<const TokenId> tokens) { return forward(tokens); }

inline TokenId TextTrunk::decode(TokenId token) {
    return forward(std::span<const TokenId>{&token, 1});
}

inline void TextTrunk::run_gdn(std::size_t layer, std::int32_t T, const Tensor& block_input,
                               Tensor& mixer_out) {
    const auto& weights = model_.layers[layer].gdn_projection;
    const std::int32_t key_width   = TextConfig::gdn_key_width;
    const std::int32_t value_width = TextConfig::gdn_value_width;

    Tensor q    = activations_.alloc(DType::BF16, {key_width, T});
    Tensor k    = activations_.alloc(DType::BF16, {key_width, T});
    Tensor v    = activations_.alloc(DType::BF16, {value_width, T});
    Tensor z    = activations_.alloc(DType::BF16, {value_width, T});
    Tensor g    = activations_.alloc(DType::FP32, {TextConfig::gdn_value_heads, T});
    Tensor beta = activations_.alloc(DType::FP32, {TextConfig::gdn_value_heads, T});
    Variant::gdn_projection_control(block_input, weights, q, k, v, z, g, beta, workspace_, stream_);

    Tensor conv_in  = activations_.alloc(DType::BF16, {TextConfig::gdn_conv_channels, T});
    Tensor conv_out = activations_.alloc(DType::BF16, {TextConfig::gdn_conv_channels, T});
    const auto copy_segment = [&](const Tensor& source, std::int32_t column) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            static_cast<std::byte*>(conv_in.data) +
                static_cast<std::size_t>(column) * sizeof(std::uint16_t),
            static_cast<std::size_t>(TextConfig::gdn_conv_channels) * sizeof(std::uint16_t),
            source.data, static_cast<std::size_t>(source.ne[0]) * sizeof(std::uint16_t),
            static_cast<std::size_t>(source.ne[0]) * sizeof(std::uint16_t), static_cast<std::size_t>(T),
            cudaMemcpyDeviceToDevice, stream_));
    };
    copy_segment(q, 0);
    copy_segment(k, key_width);
    copy_segment(v, 2 * key_width);

    const std::optional<std::size_t> slot = state_.gdn_state_index(layer);
    if (!slot.has_value()) {
        throw std::logic_error("a QSA layer has no GDN state slot");
    }
    const std::size_t gdn_index = *slot;
    Tensor conv_state = state_.gdn_conv_state(gdn_index);
    Tensor conv_kernel =
        weights.convolution.view({TextConfig::gdn_conv_channels, TextConfig::gdn_conv_kernel});
    ops::causal_conv1d_silu(conv_in, conv_kernel, conv_state, conv_state, conv_out, stream_);

    ops::extract_bf16_columns(conv_out, 0, q, stream_);
    ops::extract_bf16_columns(conv_out, key_width, k, stream_);
    ops::extract_bf16_columns(conv_out, 2 * key_width, v, stream_);

    Tensor o = activations_.alloc(DType::BF16, {value_width, T});
    Tensor on = activations_.alloc(DType::BF16, {value_width, T});
    Tensor q_recurrent = q.view({128, TextConfig::gdn_key_heads, T});
    Tensor k_recurrent = k.view({128, TextConfig::gdn_key_heads, T});
    Tensor v_recurrent = v.view({128, TextConfig::gdn_value_heads, T});
    Tensor o_recurrent = o.view({128, TextConfig::gdn_value_heads, T});
    Tensor o_normalized = on.view({128, TextConfig::gdn_value_heads, T});
    Tensor recurrent_state = state_.gdn_recurrent_state(gdn_index);
    ops::gated_delta_net(q_recurrent, k_recurrent, v_recurrent, g, beta, kGdnScale, true, workspace_,
                         recurrent_state, o_recurrent, stream_);
    ops::gated_rmsnorm(o_recurrent, weights.norm, z.view({128, TextConfig::gdn_value_heads, T}),
                       TextConfig::rms_epsilon, o_normalized, stream_);
    ops::linear(on, weights.output, mixer_out, ops::LinearPolicy::A16Only, workspace_, stream_);
}

inline void TextTrunk::run_qsa(std::size_t layer, std::int32_t T, const Tensor& block_input,
                               const Tensor& positions, const Tensor& valid_columns,
                               const Tensor& table_rows, Tensor& mixer_out) {
    const auto& weights = model_.layers[layer].attention_projection;
    const std::int32_t index_dim = TextConfig::index_head_dim;

    const std::int32_t query_rows = kHeadDim * TextConfig::query_heads;
    Tensor query_flat       = activations_.alloc(DType::BF16, {query_rows, T});
    Tensor output_gate_flat = activations_.alloc(DType::BF16, {query_rows, T});
    Tensor key_flat         = activations_.alloc(DType::BF16, {kHeadDim * TextConfig::kv_heads, T});
    Tensor value_flat       = activations_.alloc(DType::BF16, {kHeadDim * TextConfig::kv_heads, T});
    Tensor index_query_flat =
        activations_.alloc(DType::BF16, {index_dim * TextConfig::index_heads, T});
    Tensor index_key_flat =
        activations_.alloc(DType::BF16, {index_dim * TextConfig::index_kv_heads, T});
    Tensor query       = query_flat.view({kHeadDim, TextConfig::query_heads, T, kBatch});
    Tensor output_gate = output_gate_flat.view({kHeadDim, TextConfig::query_heads, T, kBatch});
    Tensor key         = key_flat.view({kHeadDim, TextConfig::kv_heads, T, kBatch});
    Tensor value       = value_flat.view({kHeadDim, TextConfig::kv_heads, T, kBatch});
    Tensor index_query = index_query_flat.view({index_dim, TextConfig::index_heads, T, kBatch});
    Tensor index_key   = index_key_flat.view({index_dim, TextConfig::index_kv_heads, T, kBatch});
    Variant::attention_projection(block_input, weights, query_flat, output_gate_flat, key_flat,
                                  value_flat, index_query_flat, index_key_flat, workspace_, stream_);

    Tensor rotated_query = activations_.alloc(DType::BF16, {kHeadDim, TextConfig::query_heads, T, kBatch});
    Tensor rotated_index_query =
        activations_.alloc(DType::BF16, {index_dim, TextConfig::index_heads, T, kBatch});

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
    ops::QsaRawRecordWriteView records = state_.records(static_cast<std::size_t>(T));
    const ops::QsaCacheView cache = state_.cache();

    ops::qsa_prepare(ops::QsaProjectionView{query, key, value, index_query, index_key}, parameters,
                     batch, state_.before_tail(), cache, workspace_, records, rotated_query,
                     rotated_index_query, stream_);

    ops::QsaBlockSelection selected{
        .selection = activations_.alloc(DType::I32, {TextConfig::index_block_budget, T, kBatch}),
        .counts    = activations_.alloc(DType::I32, {T, kBatch}),
    };
    ops::qsa_indexer(rotated_index_query, ops::ConstQsaIndexCacheView{cache.index_k_pages,
                                                                      cache.page_table},
                     batch, static_cast<std::uint32_t>(T), workspace_, selected, stream_);

    Tensor attention = activations_.alloc(DType::BF16, {kHeadDim, TextConfig::query_heads, T, kBatch});
    ops::qsa_sparse_gqa(rotated_query, output_gate, cache, batch, selected,
                        static_cast<std::uint32_t>(T), workspace_, attention, stream_);
    ops::linear(attention.view({TextConfig::query_width, T}), weights.output, mixer_out,
                ops::LinearPolicy::A16Only, workspace_, stream_);

    const ops::QsaRawRecordReadView read_records{records.raw_keys, records.rope_positions};
    ops::QsaTailStateView after = state_.tail();
    ops::qsa_tail_extract(state_.const_tail(), read_records, records.base_positions, valid_columns,
                          after, stream_);
    state_.publish_tail();
}

inline void TextTrunk::run_ple(std::int32_t T, const Tensor& residual, Tensor& updated,
                               Tensor& normalized) {
    const auto& ple = model_.ple;
    const std::int32_t entries = static_cast<std::int32_t>(ops::kPleHeads) * T * kBatch;

    Tensor embedding = activations_.alloc(DType::BF16, {TextConfig::ple_embed_dim, T, kBatch});
    ops::ple_lookup_rows(state_.ple_table(), state_.ple_rows(), static_cast<std::size_t>(entries),
                         state_.ple_staged_rows(), embedding, stream_);

    Tensor projected_key   = activations_.alloc(DType::BF16, {kChannels, T, kBatch});
    Tensor projected_value = activations_.alloc(DType::BF16, {kHidden, T, kBatch});
    ops::linear(embedding, ple.key, projected_key, ops::LinearPolicy::A16Only, workspace_, stream_);
    ops::linear(embedding, ple.value, projected_value, ops::LinearPolicy::A16Only, workspace_, stream_);

    Tensor gated_value     = activations_.alloc(DType::BF16, {kChannels, T, kBatch});
    Tensor normalized_conv = activations_.alloc(DType::BF16, {kChannels, T, kBatch});
    Tensor offsets         = activations_.alloc(DType::I32, {kBatch});
    Tensor counts          = activations_.alloc(DType::I32, {kBatch});
    const std::int32_t live = T;
    CUDA_CHECK(cudaMemcpyAsync(offsets.data, &live, 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(counts.data, &live, 4, cudaMemcpyHostToDevice, stream_));

    ops::ple_gate(projected_key, projected_value, residual, ple.norm_key, ple.norm_query,
                  ple.norm_conv, TextConfig::rms_epsilon, gated_value, normalized_conv, stream_);

    const Tensor next_norm =
        model_.layers[TextConfig::ple_layer].attention_hc.norm.view({kChannels});
    ops::ple_conv_add_norm(state_.ple_history(), normalized_conv, residual, gated_value, ple.kernel,
                           offsets, counts, next_norm, TextConfig::rms_epsilon, updated, normalized,
                           stream_);

    Tensor after_history = activations_.alloc(DType::BF16,
                                            {kChannels, static_cast<std::int32_t>(ops::kPleConvWindow),
                                             kBatch});
    ops::ple_state_extract(state_.ple_history(), normalized_conv, counts, after_history, stream_);
    CUDA_CHECK(cudaMemcpyAsync(state_.ple_history().data, after_history.data,
                               static_cast<std::size_t>(kChannels) * ops::kPleConvWindow * kBatch * 2,
                               cudaMemcpyDeviceToDevice, stream_));
}

inline TokenId TextTrunk::forward(std::span<const TokenId> tokens) {
    const std::int32_t T = static_cast<std::int32_t>(tokens.size());
    auto scope = workspace_.scope();
    state_.prepare_ple_rows(tokens);

    Tensor ids = activations_.alloc(DType::I32, {T});
    CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data(), static_cast<std::size_t>(T) * 4,
                               cudaMemcpyHostToDevice, stream_));

    const auto forward_scope = activations_.scope();
    Tensor embedding_out = activations_.alloc(DType::BF16, {kHidden, T});
    ops::embedding(ids, model_.embedding, embedding_out, stream_);

    Tensor residual    = activations_.alloc(DType::BF16, {kHidden, kStreams, T, kBatch});
    Tensor normalized  = activations_.alloc(DType::BF16, {kHidden, kStreams, T, kBatch});
    Tensor stage       = activations_.alloc(DType::BF16, {kHidden, kStreams, T, kBatch});
    Tensor block_input = activations_.alloc(DType::BF16, {kHidden, T});
    Tensor injection   = activations_.alloc(DType::FP32, {kStreams, T});
    Tensor mixer_out   = activations_.alloc(DType::BF16, {kHidden, T});
    Tensor moe_out     = activations_.alloc(DType::BF16, {kHidden, T});
    Tensor sample      = activations_.alloc(DType::BF16, {kHidden, T});
    Tensor logits      = activations_.alloc(DType::BF16, {TextConfig::vocab, T});
    Tensor positions   = activations_.alloc(DType::I32, {T, 1, kBatch});
    Tensor valid_columns = activations_.alloc(DType::I32, {kBatch});
    Tensor table_rows    = activations_.alloc(DType::I32, {kBatch});

    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(T));
    for (std::int32_t t = 0; t < T; ++t) {
        host_positions[static_cast<std::size_t>(t)] =
            static_cast<std::int32_t>(state_.next_position() + static_cast<std::uint32_t>(t));
    }
    CUDA_CHECK(cudaMemcpyAsync(positions.data, host_positions.data(),
                               static_cast<std::size_t>(T) * 4, cudaMemcpyHostToDevice, stream_));
    const std::int32_t live = T;
    const std::int32_t row  = 0;
    CUDA_CHECK(cudaMemcpyAsync(valid_columns.data, &live, 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(table_rows.data, &row, 4, cudaMemcpyHostToDevice, stream_));

    ops::hc_expand_norm(embedding_out, model_.layers[0].attention_hc.norm, TextConfig::rms_epsilon,
                        residual, normalized, stream_);

    for (std::size_t layer = 0; layer < model_.loaded_layers; ++layer) {
        const auto layer_scope = activations_.scope();
        const auto& attention_hc = model_.layers[layer].attention_hc;
        ops::hc_mix(normalized, ops::HcMixWeights{attention_hc.down_injection, attention_hc.up},
                    workspace_, block_input, injection, stream_);

        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            run_qsa(layer, T, block_input, positions, valid_columns, table_rows, mixer_out);
        } else {
            run_gdn(layer, T, block_input, mixer_out);
        }

        const auto& moe_hc = model_.layers[layer].moe_hc;
        ops::hc_combine_norm(residual, mixer_out, injection, moe_hc.norm, TextConfig::rms_epsilon,
                             stage, normalized, stream_);

        ops::hc_mix(normalized, ops::HcMixWeights{moe_hc.down_injection, moe_hc.up}, workspace_,
                    block_input, injection, stream_);

        CUDA_CHECK(cudaMemsetAsync(moe_out.data, 0, moe_out.bytes(), stream_));
        Variant::post_mixer(block_input, model_.layers[layer].post_mixer, moe_out, workspace_, stream_);

        if (layer + 1 == static_cast<std::size_t>(TextConfig::ple_layer)) {
            ops::hc_combine(stage, moe_out, injection, residual, stream_);
            run_ple(T, residual, stage, normalized);
            std::swap(residual, stage);
        } else {
            const Tensor& next_norm = layer + 1 < model_.loaded_layers
                                          ? model_.layers[layer + 1].attention_hc.norm
                                          : model_.tail_norm;
            ops::hc_combine_norm(stage, moe_out, injection, next_norm, TextConfig::rms_epsilon,
                                 residual, normalized, stream_);
        }
    }

    ops::hc_final_mix(normalized,
                      ops::HcFinalMixWeights{model_.final_hc.down_injection, model_.final_hc.up},
                      workspace_, sample, stream_);
    ops::linear(sample, model_.output_head, logits, ops::LinearPolicy::A16Only, workspace_, stream_);

    Tensor winner = activations_.alloc(DType::I32, {T});
    ops::argmax(logits, winner, TextConfig::vocab, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::int32_t token = 0;
    CUDA_CHECK(cudaMemcpy(&token,
                          static_cast<const std::byte*>(winner.data) +
                              static_cast<std::size_t>(T - 1) * 4,
                          4, cudaMemcpyDeviceToHost));
    state_.advance_position(static_cast<std::uint32_t>(T));
    return static_cast<TokenId>(token);
}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
