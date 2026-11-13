#pragma once

#include "targets/qwen4_exp/impl/runtime/instance.h"

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

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
