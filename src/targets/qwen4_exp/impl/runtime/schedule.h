#pragma once

#include "runtime/contract/types.h"
#include "targets/qwen4_exp/impl/runtime/instance.h"

#include <cstdint>
#include <span>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

[[nodiscard]] inline std::uint32_t prefill_text_chunk(std::uint32_t remaining) {
    return remaining;
}

[[nodiscard]] inline std::uint32_t ordinary_decode_batch(std::span<const TokenId> last_tokens) {
    return static_cast<std::uint32_t>(last_tokens.size());
}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
