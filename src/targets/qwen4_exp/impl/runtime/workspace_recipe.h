#pragma once

#include "ninfer/ops/ple.h"

#include "targets/qwen4_exp/impl/runtime/instance.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

inline constexpr std::size_t kBf16Bytes = 2;

struct WorkspaceRecipe {
    std::size_t lane_capacity_bytes;
    std::size_t request_transient_capacity_bytes;
};

[[nodiscard]] inline std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    if (alignment == 0) { return value; }
    return (value + alignment - 1) / alignment * alignment;
}

[[nodiscard]] inline WorkspaceRecipe text_workspace_recipe(std::uint32_t max_concurrency) noexcept {
    const std::size_t hidden = static_cast<std::size_t>(TextConfig::hidden) * kBf16Bytes;
    const std::size_t residual = static_cast<std::size_t>(TextConfig::residual_width) * kBf16Bytes;
    const std::size_t qsa =
        (static_cast<std::size_t>(TextConfig::index_query_width) +
         static_cast<std::size_t>(TextConfig::index_key_width) +
         static_cast<std::size_t>(TextConfig::index_block_budget)) * kBf16Bytes;
    const std::size_t moe =
        (static_cast<std::size_t>(TextConfig::experts_per_token) * TextConfig::expert_intermediate +
         static_cast<std::size_t>(TextConfig::shared_intermediate)) * kBf16Bytes;
    const std::size_t ple =
        static_cast<std::size_t>(TextConfig::ple_heads) * TextConfig::ple_head_width *
        TextConfig::ple_conv_kernel * kBf16Bytes;

    const std::size_t lane = align_up(hidden + residual + qsa + moe + ple, 16);
    const std::size_t transient = align_up(ple, 16);
    return WorkspaceRecipe{
        .lane_capacity_bytes = lane * max_concurrency,
        .request_transient_capacity_bytes = transient,
    };
}

[[nodiscard]] inline std::size_t text_activation_capacity_bytes(std::uint32_t prefill_chunk) noexcept {
    const std::size_t tokens   = prefill_chunk == 0 ? 1 : prefill_chunk;
    const std::size_t hidden   = TextConfig::hidden;
    const std::size_t channels = TextConfig::residual_width;
    const std::size_t streams  = TextConfig::hc_streams;

    const std::size_t trunk_per_token =
        (17 * hidden + static_cast<std::size_t>(TextConfig::vocab)) * kBf16Bytes +
        (streams + 2) * sizeof(std::uint32_t);

    const std::size_t qsa_per_token =
        (4 * static_cast<std::size_t>(TextConfig::query_width) +
         2 * static_cast<std::size_t>(TextConfig::kv_width) +
         2 * static_cast<std::size_t>(TextConfig::index_query_width) +
         static_cast<std::size_t>(TextConfig::index_key_width)) *
            kBf16Bytes +
        (static_cast<std::size_t>(TextConfig::index_block_budget) + 1) * sizeof(std::uint32_t);

    const std::size_t gdn_per_token =
        (2 * static_cast<std::size_t>(TextConfig::gdn_key_width) +
         4 * static_cast<std::size_t>(TextConfig::gdn_value_width) +
         2 * static_cast<std::size_t>(TextConfig::gdn_conv_channels)) *
            kBf16Bytes +
        2 * static_cast<std::size_t>(TextConfig::gdn_value_heads) * sizeof(float);

    const std::size_t ple_per_token =
        (2 * static_cast<std::size_t>(TextConfig::ple_embed_dim) + 3 * channels) * kBf16Bytes;
    const std::size_t ple_history =
        channels * static_cast<std::size_t>(ops::kPleConvWindow) * kBf16Bytes;

    const std::size_t layer_per_token = std::max(qsa_per_token, gdn_per_token + ple_per_token);
    return align_up((trunk_per_token + layer_per_token) * tokens + ple_history + 8, 16);
}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
