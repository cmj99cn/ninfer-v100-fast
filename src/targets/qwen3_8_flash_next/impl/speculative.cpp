#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

std::vector<TokenId> draft_chain(MtpDraft& draft,
                                 const std::array<Tensor, kMaximumDevices>& conditioning,
                                 TokenId token, std::uint32_t steps, std::int32_t start_position,
                                 float min_p) {
    return draft.run_chain(conditioning, token, steps, start_position, min_p);
}

std::uint32_t accepted_prefix(std::span<const TokenId> drafts,
                              std::span<const TokenId> target_predictions) {
    const std::size_t limit = std::min(drafts.size(), target_predictions.size());
    std::uint32_t accepted = 0;
    while (accepted < limit && drafts[accepted] == target_predictions[accepted]) { ++accepted; }
    return accepted;
}

std::vector<TokenId> commit_tokens(std::span<const TokenId> drafts,
                                   std::span<const TokenId> target_predictions) {
    const std::uint32_t accepted = accepted_prefix(drafts, target_predictions);
    std::vector<TokenId> committed(drafts.begin(),
                                   drafts.begin() + static_cast<std::ptrdiff_t>(accepted));
    if (accepted < target_predictions.size()) { committed.push_back(target_predictions[accepted]); }
    return committed;
}

std::vector<TokenId> verify_sequential(
    std::span<const TokenId> verify_tokens,
    const std::function<TokenId(TokenId)>& forward_step) {
    std::vector<TokenId> predictions;
    predictions.reserve(verify_tokens.size());
    for (const TokenId token : verify_tokens) { predictions.push_back(forward_step(token)); }
    return predictions;
}

std::uint32_t verify_extent(std::uint32_t draft_count, std::uint32_t draft_window,
                            const ninfer::runtime::RoundBudget& budget, std::uint32_t frontier,
                            std::uint32_t capacity) {
    const std::uint32_t max_by_budget =
        budget.generated_tokens_remaining > 1 ? budget.generated_tokens_remaining - 1 : 0;
    const std::uint32_t max_by_capacity = capacity > frontier + 1 ? capacity - frontier - 1 : 0;
    return std::min({draft_count, draft_window, max_by_budget, max_by_capacity});
}

bool round_finished(const ninfer::runtime::RoundBudget& budget, std::uint32_t committed_tokens) {
    return committed_tokens >= budget.generated_tokens_remaining;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
