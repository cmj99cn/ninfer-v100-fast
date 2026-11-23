#pragma once

#include "targets/qwen4_exp/impl/runtime/state_transaction.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

namespace state_transaction_detail {

struct RankSnapshot {
    std::uint32_t position = 0;
    std::uint64_t previous = 0;
    std::uint64_t current  = 0;
    DeviceBuffer buffer;
};

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

inline void copy_plane_out(const Tensor& plane, int page, std::size_t bytes, std::byte* base,
                           std::size_t& cursor, std::size_t limit, cudaStream_t stream) {
    copy_raw_out(static_cast<const std::byte*>(plane.data) + static_cast<std::size_t>(page) * bytes,
                 bytes, base, cursor, limit, stream);
}

inline void copy_plane_in(const Tensor& plane, int page, std::size_t bytes, const std::byte* base,
                          std::size_t& cursor, std::size_t limit, cudaStream_t stream) {
    copy_raw_in(static_cast<std::byte*>(plane.data) + static_cast<std::size_t>(page) * bytes,
                bytes, base, cursor, limit, stream);
}

} // namespace state_transaction_detail

struct StateTransaction::Impl {
    using RankSnapshot = state_transaction_detail::RankSnapshot;

    std::array<State*, kMaximumDevices> states{};
    int tp = 0;
    std::uint32_t verify_extent = 0;
    std::size_t max_kv_pages = 0;
    std::vector<RankSnapshot> snapshots;

    Impl(std::array<State*, kMaximumDevices> tracked, int width, std::uint32_t extent)
        : states(tracked), tp(width), verify_extent(extent) {
        const std::size_t page = static_cast<std::size_t>(ops::kQsaTokenPageSize);
        max_kv_pages = (static_cast<std::size_t>(verify_extent) + page - 1) / page + 1;
        std::vector<std::size_t> rank_bytes(static_cast<std::size_t>(tp), 0);
        for (int rank = 0; rank < tp; ++rank) {
            State& state = *states[static_cast<std::size_t>(rank)];
            std::size_t bytes = 0;
            for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
                bytes += state.gdn_conv_state(index).bytes();
                bytes += state.gdn_recurrent_state(index).bytes();
            }
            for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
                const ops::QsaTailStateView tail = state.before_tail(slot);
                bytes += tail.raw_keys.bytes() + tail.first_rope.bytes();
            }
            const ops::QsaTailStateView tail = state.tail(0);
            bytes += tail.raw_keys.bytes() + tail.first_rope.bytes();
            bytes += state.ple_history().bytes();
            bytes += sizeof(std::int32_t);
            for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
                const ops::QsaCacheView cache = state.cache(slot);
                bytes += max_kv_pages * (state_transaction_detail::page_bytes(cache.k_pages) +
                                         state_transaction_detail::page_bytes(cache.v_pages) +
                                         state_transaction_detail::page_bytes(cache.index_k_pages));
            }
            rank_bytes[static_cast<std::size_t>(rank)] = bytes;
        }
        snapshots.resize(static_cast<std::size_t>(tp));
        for (int rank = 0; rank < tp; ++rank) {
            State& state = *states[static_cast<std::size_t>(rank)];
            CUDA_CHECK(cudaSetDevice(state.device_id()));
            snapshots[static_cast<std::size_t>(rank)].buffer =
                DeviceBuffer(rank_bytes[static_cast<std::size_t>(rank)]);
        }
    }

    void copy_kv(State& state, const RankSnapshot& snapshot, std::byte* base,
                 std::size_t& cursor, std::size_t limit, bool outbound, bool include) {
        if (verify_extent == 0) { return; }
        const std::uint32_t last = snapshot.position + verify_extent - 1;
        const int first_page = static_cast<int>(snapshot.position /
                                                static_cast<std::uint32_t>(ops::kQsaTokenPageSize));
        const int last_page  = static_cast<int>(last /
                                                static_cast<std::uint32_t>(ops::kQsaTokenPageSize));
        const cudaStream_t stream = state.stream();
        for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
            const ops::QsaCacheView cache = state.cache(slot);
            const std::size_t token_bytes = state_transaction_detail::page_bytes(cache.k_pages);
            const std::size_t index_bytes = state_transaction_detail::page_bytes(cache.index_k_pages);
            for (int page = first_page; page <= last_page; ++page) {
                if (!include) {
                    state_transaction_detail::skip_bytes(token_bytes, cursor, limit);
                    state_transaction_detail::skip_bytes(token_bytes, cursor, limit);
                    state_transaction_detail::skip_bytes(index_bytes, cursor, limit);
                } else if (outbound) {
                    state_transaction_detail::copy_plane_out(cache.k_pages, page, token_bytes, base,
                                                             cursor, limit, stream);
                    state_transaction_detail::copy_plane_out(cache.v_pages, page, token_bytes, base,
                                                             cursor, limit, stream);
                    state_transaction_detail::copy_plane_out(cache.index_k_pages, page, index_bytes,
                                                             base, cursor, limit, stream);
                } else {
                    state_transaction_detail::copy_plane_in(cache.k_pages, page, token_bytes, base,
                                                            cursor, limit, stream);
                    state_transaction_detail::copy_plane_in(cache.v_pages, page, token_bytes, base,
                                                            cursor, limit, stream);
                    state_transaction_detail::copy_plane_in(cache.index_k_pages, page, index_bytes,
                                                            base, cursor, limit, stream);
                }
            }
        }
    }

    void capture_state(State& state, RankSnapshot& snapshot, bool include_gdn, bool include_kv) {
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        snapshot.position = state.next_position();
        snapshot.previous = state.ple_tail(0);
        snapshot.current  = state.ple_tail(1);
        auto* base = static_cast<std::byte*>(snapshot.buffer.p);
        const std::size_t limit = snapshot.buffer.bytes;
        std::size_t cursor = 0;
        const cudaStream_t stream = state.stream();
        for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
            if (include_gdn) {
                state_transaction_detail::copy_out(state.gdn_conv_state(index), base, cursor, limit, stream);
                state_transaction_detail::copy_out(state.gdn_recurrent_state(index), base, cursor, limit, stream);
            } else {
                state_transaction_detail::skip_bytes(state.gdn_conv_state(index).bytes(), cursor,
                                                     limit);
                state_transaction_detail::skip_bytes(state.gdn_recurrent_state(index).bytes(), cursor,
                                                     limit);
            }
        }
        for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
            const ops::QsaTailStateView tail = state.before_tail(slot);
            state_transaction_detail::copy_out(tail.raw_keys, base, cursor, limit, stream);
            state_transaction_detail::copy_out(tail.first_rope, base, cursor, limit, stream);
        }
        {
            const ops::QsaTailStateView tail = state.tail(0);
            state_transaction_detail::copy_out(tail.raw_keys, base, cursor, limit, stream);
            state_transaction_detail::copy_out(tail.first_rope, base, cursor, limit, stream);
        }
        state_transaction_detail::copy_out(state.ple_history(), base, cursor, limit, stream);
        state_transaction_detail::copy_raw_out(state.route_counter(), sizeof(std::int32_t), base,
                                               cursor, limit, stream);
        copy_kv(state, snapshot, base, cursor, limit, true, include_kv);
    }

    void restore_state(State& state, RankSnapshot& snapshot, bool include_gdn, bool include_kv) {
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        state.rewind_scalars(snapshot.position, snapshot.previous, snapshot.current);
        auto* base = static_cast<std::byte*>(snapshot.buffer.p);
        const std::size_t limit = snapshot.buffer.bytes;
        std::size_t cursor = 0;
        const cudaStream_t stream = state.stream();
        for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
            if (include_gdn) {
                state_transaction_detail::copy_in(state.gdn_conv_state(index), base, cursor, limit, stream);
                state_transaction_detail::copy_in(state.gdn_recurrent_state(index), base, cursor, limit, stream);
            } else {
                state_transaction_detail::skip_bytes(state.gdn_conv_state(index).bytes(), cursor,
                                                     limit);
                state_transaction_detail::skip_bytes(state.gdn_recurrent_state(index).bytes(), cursor,
                                                     limit);
            }
        }
        for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
            const ops::QsaTailStateView tail = state.before_tail(slot);
            state_transaction_detail::copy_in(tail.raw_keys, base, cursor, limit, stream);
            state_transaction_detail::copy_in(tail.first_rope, base, cursor, limit, stream);
        }
        {
            const ops::QsaTailStateView tail = state.tail(0);
            state_transaction_detail::copy_in(tail.raw_keys, base, cursor, limit, stream);
            state_transaction_detail::copy_in(tail.first_rope, base, cursor, limit, stream);
        }
        state_transaction_detail::copy_in(state.ple_history(), base, cursor, limit, stream);
        state_transaction_detail::copy_raw_in(state.route_counter(), sizeof(std::int32_t), base,
                                              cursor, limit, stream);
        copy_kv(state, snapshot, base, cursor, limit, false, include_kv);
    }
};

StateTransaction::StateTransaction(std::array<State*, kMaximumDevices> states, int tp,
                                   std::uint32_t verify_extent)
    : impl_(std::make_unique<Impl>(states, tp, verify_extent)) {}

StateTransaction::~StateTransaction() = default;

void StateTransaction::begin() {
    for (int rank = 0; rank < impl_->tp; ++rank) {
        impl_->capture_state(*impl_->states[static_cast<std::size_t>(rank)],
                             impl_->snapshots[static_cast<std::size_t>(rank)], true, true);
    }
}

void StateTransaction::commit(std::span<const TokenId> committed, const Replay& replay) {
    for (int rank = 0; rank < impl_->tp; ++rank) {
        impl_->restore_state(*impl_->states[static_cast<std::size_t>(rank)],
                             impl_->snapshots[static_cast<std::size_t>(rank)], true, true);
    }
    for (const TokenId token : committed) { replay(token); }
}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
