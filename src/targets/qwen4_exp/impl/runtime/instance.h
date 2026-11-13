#pragma once

#ifndef NINFER_QWEN4EXP_VARIANT
#    error "NINFER_QWEN4EXP_VARIANT must name the complete exact Variant"
#endif
#ifndef NINFER_QWEN4EXP_RUNTIME_NS
#    error "NINFER_QWEN4EXP_RUNTIME_NS must be a unique identifier for this instantiation"
#endif

#include <ninfer/targets/qwen4_exp/runtime.h>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

using Variant             = NINFER_QWEN4EXP_VARIANT;
using WeightsProfile      = typename Variant::WeightsProfile;
using TextConfig          = typename Variant::TextConfig;
using ModelView           = typename Variant::ModelView;
using PostMixerWeights    = typename Variant::PostMixerWeights;
using GraphExecutionProfile = typename Variant::GraphExecutionProfile;

using SequencePlan    = qwen4_exp::SequencePlan<Variant>;
using SequencePlanner = qwen4_exp::SequencePlanner<Variant>;
using RequestBasePlan = qwen4_exp::RequestBasePlan<Variant>;
using RequestPlan     = qwen4_exp::RequestPlan<Variant>;
using Program         = qwen4_exp::Program<Variant>;

inline constexpr float kAttentionScale                = Variant::attention_scale;
inline constexpr float kGdnScale                      = Variant::gdn_scale;
inline constexpr std::uint32_t kPrefillChunkAlignment = Variant::prefill_chunk_alignment;
inline constexpr std::uint32_t kMaximumContext        = Variant::maximum_context;

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
