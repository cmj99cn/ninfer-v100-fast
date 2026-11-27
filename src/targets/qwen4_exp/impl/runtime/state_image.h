#pragma once

// Ported and adapted from Neroued/ninfer, commit 40dec5ff
// (https://github.com/Neroued/ninfer), Apache License, Version 2.0.
// The upstream qwen4_exp StateLayout/StateViews were mapped onto this fork's TextState fields:
// GDN conv/recurrent, QSA before-tails and tail, PLE history, route counter and position scalars.
// The same ordering drives MTP verify snapshots and pinned-host prefix-cache publication.

#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen4_exp/impl/runtime/text_context.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {
namespace state_transaction_detail {

inline std::size_t page_bytes(const Tensor& plane) {
    return static_cast<std::size_t>(plane.ne[0]) * static_cast<std::size_t>(plane.ne[1]) *
           static_cast<std::size_t>(plane.ne[2]) * 2;
}

inline void require_room(std::size_t cursor, std::size_t bytes, std::size_t limit) {
    if (cursor + bytes > limit) {
        throw std::runtime_error("MTP state snapshot exceeds its device buffer");
    }
}

inline void copy_out(const Tensor& tensor, std::byte* base, std::size_t& cursor,
                     std::size_t limit, cudaStream_t stream) {
    const std::size_t bytes = tensor.bytes();
    require_room(cursor, bytes, limit);
    CUDA_CHECK(cudaMemcpyAsync(base + cursor, tensor.data, bytes, cudaMemcpyDeviceToDevice, stream));
    cursor += bytes;
}

inline void copy_in(const Tensor& tensor, const std::byte* base, std::size_t& cursor,
                    std::size_t limit, cudaStream_t stream) {
    const std::size_t bytes = tensor.bytes();
    require_room(cursor, bytes, limit);
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, base + cursor, bytes, cudaMemcpyDeviceToDevice, stream));
    cursor += bytes;
}

inline void copy_raw_out(const void* source, std::size_t bytes, std::byte* base,
                         std::size_t& cursor, std::size_t limit, cudaStream_t stream) {
    require_room(cursor, bytes, limit);
    CUDA_CHECK(cudaMemcpyAsync(base + cursor, source, bytes, cudaMemcpyDeviceToDevice, stream));
    cursor += bytes;
}

inline void copy_raw_in(void* destination, std::size_t bytes, const std::byte* base,
                        std::size_t& cursor, std::size_t limit, cudaStream_t stream) {
    require_room(cursor, bytes, limit);
    CUDA_CHECK(cudaMemcpyAsync(destination, base + cursor, bytes, cudaMemcpyDeviceToDevice, stream));
    cursor += bytes;
}

inline void skip_bytes(std::size_t bytes, std::size_t& cursor, std::size_t limit) {
    require_room(cursor, bytes, limit);
    cursor += bytes;
}

// Host scalars accompanying a captured image. Tensor bytes follow a fixed order: GDN
// conv/recurrent, QSA before-tails, QSA tail, PLE history, route counter. Full KV history is stored
// separately as 64-token blocks.
struct StateImageHeader {
    std::uint32_t position = 0;
    std::uint64_t previous = 0;
    std::uint64_t current  = 0;
};

inline std::size_t state_image_tensor_bytes(const TextState& state) {
    std::size_t bytes = sizeof(std::int32_t);
    for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
        bytes += state.gdn_conv_state(index).bytes() + state.gdn_recurrent_state(index).bytes();
    }
    for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
        const ops::QsaTailStateView tail = state.before_tail(slot);
        bytes += tail.raw_keys.bytes() + tail.first_rope.bytes();
    }
    const ops::QsaTailStateView tail = state.tail(0);
    bytes += tail.raw_keys.bytes() + tail.first_rope.bytes();
    return bytes + state.ple_history().bytes();
}

[[nodiscard]] inline StateImageHeader
capture_state_image(TextState& state, std::byte* base, std::size_t& cursor, std::size_t limit,
                    cudaStream_t stream) {
    StateImageHeader header{state.next_position(), state.ple_tail(0), state.ple_tail(1)};
    for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
        copy_out(state.gdn_conv_state(index), base, cursor, limit, stream);
        copy_out(state.gdn_recurrent_state(index), base, cursor, limit, stream);
    }
    for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
        const ops::QsaTailStateView tail = state.before_tail(slot);
        copy_out(tail.raw_keys, base, cursor, limit, stream);
        copy_out(tail.first_rope, base, cursor, limit, stream);
    }
    {
        const ops::QsaTailStateView tail = state.tail(0);
        copy_out(tail.raw_keys, base, cursor, limit, stream);
        copy_out(tail.first_rope, base, cursor, limit, stream);
    }
    copy_out(state.ple_history(), base, cursor, limit, stream);
    copy_raw_out(state.route_counter(), sizeof(std::int32_t), base, cursor, limit, stream);
    return header;
}

inline void restore_state_image(TextState& state, const std::byte* base, std::size_t& cursor,
                                std::size_t limit, cudaStream_t stream) {
    for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
        copy_in(state.gdn_conv_state(index), base, cursor, limit, stream);
        copy_in(state.gdn_recurrent_state(index), base, cursor, limit, stream);
    }
    for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
        const ops::QsaTailStateView tail = state.before_tail(slot);
        copy_in(tail.raw_keys, base, cursor, limit, stream);
        copy_in(tail.first_rope, base, cursor, limit, stream);
    }
    {
        const ops::QsaTailStateView tail = state.tail(0);
        copy_in(tail.raw_keys, base, cursor, limit, stream);
        copy_in(tail.first_rope, base, cursor, limit, stream);
    }
    copy_in(state.ple_history(), base, cursor, limit, stream);
    copy_raw_in(state.route_counter(), sizeof(std::int32_t), base, cursor, limit, stream);
}

} // namespace state_transaction_detail
} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
