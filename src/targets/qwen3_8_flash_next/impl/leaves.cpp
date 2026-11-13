#include "targets/qwen3_8_flash_next/impl/leaves.h"

#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/moe_nvfp4.h"

#include <array>
#include <span>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr std::int32_t kQueryGateRows      = 2 * TextConfig::query_width;
constexpr std::int32_t kKeyValueRows       = 2 * TextConfig::kv_width + TextConfig::index_query_width +
                                            TextConfig::index_key_width;
constexpr std::int32_t kGdnQkvzRows        = TextConfig::gdn_conv_channels + TextConfig::gdn_value_width;

constexpr std::array<ops::LinearSplitSegment, 2> kQueryGateSegments{{
    {0, TextConfig::query_width},
    {TextConfig::query_width, TextConfig::query_width},
}};

constexpr std::array<ops::LinearSplitSegment, 4> kKeyValueSegments{{
    {0, TextConfig::kv_width},
    {TextConfig::kv_width, TextConfig::kv_width},
    {2 * TextConfig::kv_width, TextConfig::index_query_width},
    {2 * TextConfig::kv_width + TextConfig::index_query_width, TextConfig::index_key_width},
}};

constexpr std::array<ops::LinearSplitSegment, 4> kGdnSegments{{
    {0, TextConfig::gdn_key_width},
    {TextConfig::gdn_key_width, TextConfig::gdn_key_width},
    {2 * TextConfig::gdn_key_width, TextConfig::gdn_value_width},
    {TextConfig::gdn_conv_channels, TextConfig::gdn_value_width},
}};

} // namespace

void AttentionProjectionLeaf::project(const Tensor& hidden, const AttentionProjectionWeights& weights,
                                     Tensor& query, Tensor& output_gate, Tensor& key, Tensor& value,
                                     Tensor& index_query, Tensor& index_key,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();

    std::array<Tensor, 2> query_gate_outputs{query, output_gate};
    ops::linear_split(hidden, weights.query_gate, kQueryGateSegments, query_gate_outputs,
                      ops::LinearPolicy::A16Only, workspace, stream);

    std::array<Tensor, 4> key_value_outputs{key, value, index_query, index_key};
    ops::linear_split(hidden, weights.key_value, kKeyValueSegments, key_value_outputs,
                      ops::LinearPolicy::A16Only, workspace, stream);
}

std::size_t AttentionProjectionLeaf::workspace_capacity_bytes(std::int32_t tokens) {
    const std::size_t query_gate = ops::linear_split_workspace_capacity_bytes(
        ninfer::QType::W8G32_F16S, kQueryGateRows, TextConfig::hidden, kQueryGateSegments,
        ops::LinearPolicy::A16Only, tokens, tokens);
    const std::size_t key_value = ops::linear_split_workspace_capacity_bytes(
        ninfer::QType::BF16_CTRL, kKeyValueRows, TextConfig::hidden, kKeyValueSegments, ops::LinearPolicy::A16Only,
        tokens, tokens);
    return query_gate > key_value ? query_gate : key_value;
}

void GdnProjectionLeaf::project(const Tensor& hidden, const GdnProjectionWeights& weights,
                                Tensor& qkv, Tensor& z, Tensor& g, Tensor& beta,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();

    std::array<Tensor, 4> outputs{
        qkv.slice(0, 0, TextConfig::gdn_key_width),
        qkv.slice(0, TextConfig::gdn_key_width, TextConfig::gdn_key_width),
        qkv.slice(0, 2 * TextConfig::gdn_key_width, TextConfig::gdn_value_width),
        z,
    };
    ops::linear_split(hidden, weights.qkvz, kGdnSegments, outputs, ops::LinearPolicy::A16Only, workspace,
                      stream);
    ops::gdn_gating_proj(hidden, weights.control, weights.a_log, weights.dt_bias, workspace, g, beta,
                         stream);
}

std::size_t GdnProjectionLeaf::workspace_capacity_bytes(std::int32_t tokens) {
    return ops::linear_split_workspace_capacity_bytes(ninfer::QType::W8G32_F16S, kGdnQkvzRows,
                                                      TextConfig::hidden, kGdnSegments,
                                                      ops::LinearPolicy::A16Only, tokens, tokens) +
           ops::gdn_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                         TextConfig::hidden, tokens, tokens);
}

void PostMixerLeaf::mix(const Tensor& hidden, const MoeWeights& weights, Tensor& destination,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    ops::moe_routed_nvfp4(hidden, weights, destination, workspace, stream);
}

std::size_t PostMixerLeaf::workspace_capacity_bytes(std::int32_t tokens) {
    return ops::moe_nvfp4_workspace_capacity_bytes(tokens, tokens);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
