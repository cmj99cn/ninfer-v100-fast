#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen4_exp/impl/runtime/text_context.h"

#include <ninfer/types.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

class StateTransaction {
public:
    using State  = TextState;
    using Replay = std::function<void(TokenId)>;

    StateTransaction(std::array<State*, kMaximumDevices> states, int tp,
                     std::uint32_t verify_extent);
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

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
