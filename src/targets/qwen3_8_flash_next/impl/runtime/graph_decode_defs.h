#pragma once

#include "graph_decode_impl.h"

#include <ninfer/ops/ple.h>

#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

inline GraphDecodeController::GraphDecodeController(
    ExecutionContext& execution, std::array<TextState*, kMaximumDevices> states,
    std::array<WorkspaceArena*, kMaximumDevices> activations,
    std::array<WorkspaceArena*, kMaximumDevices> workspaces, int ranks)
    : execution_(execution), states_(states), activations_(activations), workspaces_(workspaces),
      ranks_(ranks) {
    auto& inputs = qwen3_8_flash_next::detail::graph_decode_inputs();
    for (int rank = 0; rank < ranks_; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        inputs.ids[slot]       = std::make_unique<PinnedHostBuffer>(16);
        inputs.positions[slot] = std::make_unique<PinnedHostBuffer>(16);
        inputs.scalars[slot]   = std::make_unique<PinnedHostBuffer>(16);
        inputs.winner[slot]    = std::make_unique<PinnedHostBuffer>(16);
        devices_[slot]         = execution_.dev[slot]->device;
        streams_[slot]         = execution_.dev[slot]->stream;
    }
    inputs.logits_buffer = std::make_unique<PinnedHostBuffer>(
        static_cast<std::size_t>(TextConfig::vocab) * 2);
}

inline GraphDecodeController::~GraphDecodeController() noexcept {
    qwen3_8_flash_next::detail::graph_decode_inputs().active = false;
}

inline void GraphDecodeController::fill_inputs(TokenId token, std::uint32_t position) {
    auto& inputs = qwen3_8_flash_next::detail::graph_decode_inputs();
    const std::int32_t token_bits = static_cast<std::int32_t>(token);
    const std::int32_t position_bits = static_cast<std::int32_t>(position);
    const std::int32_t scalars[4]{0, 1, 1, 0};
    for (int rank = 0; rank < ranks_; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        std::memcpy(inputs.ids[slot]->data(), &token_bits, 4);
        std::memcpy(inputs.positions[slot]->data(), &position_bits, 4);
        std::memcpy(inputs.scalars[slot]->data(), scalars, sizeof(scalars));
        std::memset(inputs.winner[slot]->data(), 0, 4);
    }
}

inline void GraphDecodeController::capture(TokenId token, const std::function<void()>& body) {
    for (int rank = 0; rank < ranks_; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        activations_[slot]->reset();
        workspaces_[slot]->reset();
    }
    qwen3_8_flash_next::detail::graph_decode_inputs().active = true;
    fill_inputs(token, 0);
    definition_.capture(std::span<const int>(devices_.data(), static_cast<std::size_t>(ranks_)),
                        std::span<cudaStream_t>(streams_.data(),
                                                static_cast<std::size_t>(ranks_)),
                        body);
    qwen3_8_flash_next::detail::graph_decode_inputs().active = false;
    executable_.instantiate(
        definition_, std::span<const int>(devices_.data(), static_cast<std::size_t>(ranks_)),
        std::span<cudaStream_t>(streams_.data(), static_cast<std::size_t>(ranks_)));
    executable_.upload();
}

inline TokenId GraphDecodeController::replay(TokenId token, std::uint32_t position) {
    for (int rank = 0; rank < ranks_; ++rank) {
        const TokenId one[1]{token};
        states_[static_cast<std::size_t>(rank)]->prepare_ple_rows(
            std::span<const TokenId>{one, 1});
    }
    for (int rank = 0; rank < ranks_; ++rank) {
        states_[static_cast<std::size_t>(rank)]->enqueue_ple_wait();
        fill_inputs(token, position);
    }
    executable_.launch();
    for (int rank = 0; rank < ranks_; ++rank) { execution_.dev[rank]->synchronize(); }
    const auto next = static_cast<TokenId>(
        *static_cast<const std::int32_t*>(qwen3_8_flash_next::detail::graph_decode_inputs().winner[0]->data()));
    for (int rank = 0; rank < ranks_; ++rank) {
        states_[static_cast<std::size_t>(rank)]->advance_position(1);
    }
    return next;
}

inline const std::uint16_t* GraphDecodeController::logits() const noexcept {
    return static_cast<const std::uint16_t*>(qwen3_8_flash_next::detail::graph_decode_inputs().logits_buffer->data());
}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
