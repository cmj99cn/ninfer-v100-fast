#pragma once

#include "targets/qwen3_8_flash_next/impl/config.h"
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct Variant {
    using WeightsProfile        = detail::WeightsProfile;
    using TextConfig            = detail::TextConfig;
    using ModelView             = detail::RuntimeModelView;
    using GraphExecutionProfile = qwen4_exp::GraphExecutionProfile;

    static constexpr float attention_scale             = 1.0F;
    static constexpr float gdn_scale                   = 1.0F;
    static constexpr std::uint32_t prefill_chunk_alignment = 64;
    static constexpr std::uint32_t maximum_context     = TextConfig::max_position_embeddings;

    static constexpr std::uint32_t kv_page_tokens = 64;
    static constexpr std::uint32_t kv_group_size = 64;
    static constexpr std::size_t bytes_per_main_page_group =
        kv_page_tokens * (static_cast<std::size_t>(TextConfig::kv_width) * 2 +
                          static_cast<std::size_t>(TextConfig::kv_width) / kv_group_size * 2);
    static constexpr std::size_t minimum_device_reservation_bytes =
        bytes_per_main_page_group;
    static constexpr std::size_t bytes_per_additional_main_page_group =
        bytes_per_main_page_group;

    static void post_mixer() {
        throw std::logic_error("qwen4_exp post-mixer leaf is not implemented");
    }
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
