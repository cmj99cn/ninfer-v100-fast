#pragma once

#include "core/device.h"

#include <ninfer/types.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::flash_next_runtime {
class TextState;
}

namespace ninfer::targets::qwen3_8_flash_next::detail {

class StateTransaction {
public:
    using State = ninfer::targets::qwen4_exp::detail::flash_next_runtime::TextState;
    using Replay = std::function<void(TokenId)>;

    explicit StateTransaction(std::array<State*, kMaximumDevices> states, int tp);
    ~StateTransaction();

    StateTransaction(const StateTransaction&)            = delete;
    StateTransaction& operator=(const StateTransaction&) = delete;
    StateTransaction(StateTransaction&&)                 = delete;
    StateTransaction& operator=(StateTransaction&&)      = delete;

    void begin();
    void commit(std::span<const TokenId> committed, const Replay& replay);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
