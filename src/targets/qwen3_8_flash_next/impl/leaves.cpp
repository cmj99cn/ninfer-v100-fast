#include "targets/qwen3_8_flash_next/impl/leaves.h"

#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/moe_nvfp4.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_runtime.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <span>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr std::int32_t kQueryGateRows      = 2 * TextConfig::query_width;
constexpr std::int32_t kKeyValueRows       = 2 * TextConfig::kv_width + TextConfig::index_query_width +
                                            TextConfig::index_key_width;
constexpr std::int32_t kGdnQkvzRows        = TextConfig::gdn_conv_channels + TextConfig::gdn_value_width;

std::array<ops::LinearSplitSegment, 2> query_gate_segments(const Tensor& query,
                                                           const Tensor& output_gate) {
    return {{
        {0, query.ne[0]},
        {query.ne[0], output_gate.ne[0]},
    }};
}

std::array<ops::LinearSplitSegment, 4> key_value_segments(const Tensor& key, const Tensor& value,
                                                          const Tensor& index_query,
                                                          const Tensor& index_key) {
    return {{
        {0, key.ne[0]},
        {key.ne[0], value.ne[0]},
        {key.ne[0] + value.ne[0], index_query.ne[0]},
        {key.ne[0] + value.ne[0] + index_query.ne[0], index_key.ne[0]},
    }};
}

std::array<ops::LinearSplitSegment, 4> gdn_segments(const Tensor& query, const Tensor& key,
                                                    const Tensor& value, const Tensor& z) {
    return {{
        {0, query.ne[0]},
        {query.ne[0], key.ne[0]},
        {query.ne[0] + key.ne[0], value.ne[0]},
        {query.ne[0] + key.ne[0] + value.ne[0], z.ne[0]},
    }};
}

} // namespace

void AttentionProjectionLeaf::project(const Tensor& hidden, const AttentionProjectionWeights& weights,
                                     Tensor& query, Tensor& output_gate, Tensor& key, Tensor& value,
                                     Tensor& index_query, Tensor& index_key,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();

    const auto query_gate = query_gate_segments(query, output_gate);
    std::array<Tensor, 2> query_gate_outputs{query, output_gate};
    ops::linear_split(hidden, weights.query_gate, query_gate, query_gate_outputs,
                      ops::LinearPolicy::A16Only, workspace, stream);

    const auto key_value = key_value_segments(key, value, index_query, index_key);
    std::array<Tensor, 4> key_value_outputs{key, value, index_query, index_key};
    ops::linear_split(hidden, weights.key_value, key_value, key_value_outputs,
                      ops::LinearPolicy::A16Only, workspace, stream);
}

std::size_t AttentionProjectionLeaf::workspace_capacity_bytes(std::int32_t tokens) {
    constexpr std::array<ops::LinearSplitSegment, 2> query_gate_segments{{
        {0, TextConfig::query_width},
        {TextConfig::query_width, TextConfig::query_width},
    }};
    constexpr std::array<ops::LinearSplitSegment, 4> key_value_segments{{
        {0, TextConfig::kv_width},
        {TextConfig::kv_width, TextConfig::kv_width},
        {2 * TextConfig::kv_width, TextConfig::index_query_width},
        {2 * TextConfig::kv_width + TextConfig::index_query_width, TextConfig::index_key_width},
    }};
    const std::size_t query_gate = ops::linear_split_workspace_capacity_bytes(
        ninfer::QType::W8G32_F16S, kQueryGateRows, TextConfig::hidden, query_gate_segments,
        ops::LinearPolicy::A16Only, tokens, tokens);
    const std::size_t key_value = ops::linear_split_workspace_capacity_bytes(
        ninfer::QType::BF16_CTRL, kKeyValueRows, TextConfig::hidden, key_value_segments,
        ops::LinearPolicy::A16Only, tokens, tokens);
    return query_gate > key_value ? query_gate : key_value;
}

void GdnProjectionLeaf::project(const Tensor& hidden, const GdnProjectionWeights& weights,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z, Tensor& g,
                                Tensor& beta, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();

    const auto gdn = gdn_segments(query, key, value, z);
    std::array<Tensor, 4> outputs{query, key, value, z};
    ops::linear_split(hidden, weights.qkvz, gdn, outputs, ops::LinearPolicy::A16Only, workspace,
                      stream);
    ops::gdn_gating_proj(hidden, weights.control, weights.a_log, weights.dt_bias, workspace, g,
                         beta, stream);
}

std::size_t GdnProjectionLeaf::workspace_capacity_bytes(std::int32_t tokens) {
    constexpr std::array<ops::LinearSplitSegment, 4> gdn_segments{{
        {0, TextConfig::gdn_key_width},
        {TextConfig::gdn_key_width, TextConfig::gdn_key_width},
        {2 * TextConfig::gdn_key_width, TextConfig::gdn_value_width},
        {TextConfig::gdn_conv_channels, TextConfig::gdn_value_width},
    }};
    return ops::linear_split_workspace_capacity_bytes(ninfer::QType::W8G32_F16S, kGdnQkvzRows,
                                                      TextConfig::hidden, gdn_segments,
                                                      ops::LinearPolicy::A16Only, tokens, tokens) +
           ops::gdn_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                         TextConfig::hidden, tokens, tokens);
}

void PostMixerLeaf::routed(const Tensor& hidden, const MoeWeights& weights,
                           const ops::MoeNvfp4RouteView& route, Tensor& destination,
                           cudaStream_t stream) {
    if (weights.cache != nullptr) {
        weights.cache->routed(weights.layer, hidden, route, destination, stream);
        return;
    }
    ops::moe_routed_nvfp4_partial(hidden, weights, route, destination, stream);
}

void PostMixerLeaf::prefetch(const MoeWeights& weights, cudaStream_t stream) {
    if (weights.cache != nullptr) { weights.cache->prefetch_previous_round(stream); }
}

void PostMixerLeaf::shared(const Tensor& hidden, const MoeWeights& weights,
                           const ops::MoeNvfp4RouteView& route, Tensor& destination,
                           cudaStream_t stream) {
    const ops::MoeNvfp4ExpertScratch scratch{
        weights.cache != nullptr ? weights.cache->activation_scratch() : nullptr};
    ops::moe_shared_nvfp4(hidden, weights, scratch, route, destination, stream);
}

std::size_t PostMixerLeaf::workspace_capacity_bytes(std::int32_t tokens) {
    return ops::moe_nvfp4_workspace_capacity_bytes(tokens, tokens);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
