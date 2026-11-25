#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "runtime/contract/types.h"
#include "targets/qwen3_8_flash_next/impl/diag_switches.h"

#include <ninfer/types.h>

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class MtpDraft;

std::vector<TokenId> draft_chain(MtpDraft& draft,
                                 const std::array<Tensor, kMaximumDevices>& conditioning,
                                 TokenId first_token, std::uint32_t steps,
                                 std::int32_t start_position = 0, float min_p = draft_min_p());

using ResidualSinkObserver = std::function<void(const Tensor&)>;

inline ResidualSinkObserver& residual_sink_observer() {
    static ResidualSinkObserver observer;
    return observer;
}

inline void observe_residual_sink(const Tensor& sink) {
    if (const auto& observer = residual_sink_observer(); observer) { observer(sink); }
}

std::uint32_t accepted_prefix(std::span<const TokenId> drafts,
                              std::span<const TokenId> target_predictions);

std::vector<TokenId> commit_tokens(std::span<const TokenId> drafts,
                                   std::span<const TokenId> target_predictions);

std::vector<TokenId> verify_sequential(
    std::span<const TokenId> verify_tokens,
    const std::function<TokenId(TokenId)>& forward_step);

std::uint32_t verify_extent(std::uint32_t draft_count, std::uint32_t draft_window,
                            const ninfer::runtime::RoundBudget& budget, std::uint32_t frontier,
                            std::uint32_t capacity);

bool round_finished(const ninfer::runtime::RoundBudget& budget, std::uint32_t committed_tokens);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
