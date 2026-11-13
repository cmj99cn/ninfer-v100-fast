#pragma once

#include "targets/qwen3_8_flash_next/impl/config.h"
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct PostMixerWeights {};

struct Variant {
    using WeightsProfile        = detail::WeightsProfile;
    using TextConfig            = detail::TextConfig;
    using ModelView             = detail::RuntimeModelView;
    using PostMixerWeights      = detail::PostMixerWeights;
    using GraphExecutionProfile = qwen4_exp::GraphExecutionProfile;

    static constexpr float attention_scale             = 1.0F;
    static constexpr float gdn_scale                   = 1.0F;
    static constexpr std::uint32_t prefill_chunk_alignment = 64;
    static constexpr std::uint32_t maximum_context     = TextConfig::max_position_embeddings;

    static void post_mixer() {
        throw std::logic_error("qwen4_exp post-mixer leaf is not implemented");
    }
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
