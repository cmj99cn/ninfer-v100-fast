#pragma once

#include "core/decode_graph.h"
#include "graph_decode.h"

#include <array>
#include <cstdint>
#include <functional>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

class TextState;

class GraphDecodeController {
public:
    GraphDecodeController(ExecutionContext& execution,
                          std::array<TextState*, kMaximumDevices> states,
                          std::array<WorkspaceArena*, kMaximumDevices> activations,
                          std::array<WorkspaceArena*, kMaximumDevices> workspaces, int ranks);
    ~GraphDecodeController() noexcept;

    [[nodiscard]] bool ready() const noexcept { return executable_.ready(); }
    void capture(TokenId token, const std::function<void()>& body);
    [[nodiscard]] TokenId replay(TokenId token, std::uint32_t position);
    [[nodiscard]] const std::uint16_t* logits() const noexcept;

private:
    void fill_inputs(TokenId token, std::uint32_t position);

    ExecutionContext& execution_;
    std::array<TextState*, kMaximumDevices> states_{};
    std::array<WorkspaceArena*, kMaximumDevices> activations_{};
    std::array<WorkspaceArena*, kMaximumDevices> workspaces_{};
    int ranks_ = 0;
    std::array<int, kMaximumDevices> devices_{};
    std::array<cudaStream_t, kMaximumDevices> streams_{};
    DecodeGraphDefinition definition_;
    DecodeGraphExecutable executable_;
};

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
