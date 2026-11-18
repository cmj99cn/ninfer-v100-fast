#pragma once

#include "core/tensor.h"

#include <ninfer/types.h>

#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class MtpDraft;

std::vector<TokenId> draft_chain(MtpDraft& draft, const Tensor& conditioning, TokenId first_token,
                                 std::uint32_t steps);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
