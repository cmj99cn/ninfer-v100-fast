#pragma once

#include "core/tensor.h"
#include "runtime/contract/types.h"

#include <ninfer/types.h>

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class MtpDraft;

std::vector<TokenId> draft_chain(MtpDraft& draft, const Tensor& conditioning, TokenId first_token,
                                 std::uint32_t steps, std::int32_t start_position = 0);

std::uint32_t accepted_prefix(std::span<const TokenId> drafts,
                              std::span<const TokenId> target_predictions);

std::vector<TokenId> commit_tokens(std::span<const TokenId> drafts,
                                   std::span<const TokenId> target_predictions);

std::uint32_t verify_extent(std::uint32_t draft_count, std::uint32_t draft_window,
                            const ninfer::runtime::RoundBudget& budget, std::uint32_t frontier,
                            std::uint32_t capacity);

bool round_finished(const ninfer::runtime::RoundBudget& budget, std::uint32_t committed_tokens);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
