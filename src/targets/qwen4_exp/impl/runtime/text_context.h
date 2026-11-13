#pragma once

#include "targets/qwen4_exp/impl/runtime/instance.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

struct TextLaneContext {
    std::vector<TokenId> prompt_tokens;
    std::uint32_t processed_prompt_tokens = 0;
    bool decode_ready = false;
};

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
