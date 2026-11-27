#pragma once

#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

namespace ninfer::targets::qwen4_exp {

[[nodiscard]] inline ops::SamplingConfig
translate_sampling(const ResolvedSamplingParameters& source) noexcept {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = static_cast<unsigned long long>(source.seed);
    out.token_counts      = nullptr;
    return out;
}

} // namespace ninfer::targets::qwen4_exp
