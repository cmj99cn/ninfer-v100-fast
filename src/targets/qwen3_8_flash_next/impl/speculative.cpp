#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

std::vector<TokenId> draft_chain(MtpDraft& draft, const Tensor& conditioning, TokenId token,
                                 std::uint32_t steps, std::int32_t start_position) {
    std::vector<TokenId> tokens;
    tokens.reserve(steps);
    if (steps == 0) { return tokens; }

    const std::size_t bytes = static_cast<std::size_t>(MtpDraft::hidden_elements) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaSetDevice(draft.device()));
    ninfer::DeviceBuffer current(bytes);
    ninfer::DeviceBuffer next(bytes);
    CUDA_CHECK(cudaMemcpy(current.p, conditioning.data, bytes, cudaMemcpyDeviceToDevice));
    for (std::uint32_t step = 0; step < steps; ++step) {
        const Tensor current_view(current.p, DType::FP16, {2560, 4});
        Tensor next_view(next.p, DType::FP16, {2560, 4});
        token = draft.run(current_view, token, start_position + static_cast<std::int32_t>(step),
                          &next_view);
        tokens.push_back(token);
        std::swap(current, next);
    }
    return tokens;
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
