#pragma once

#include "targets/qwen4_exp/impl/runtime/state_transaction.h"

#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

namespace state_transaction_detail {

using Words = std::vector<std::uint16_t>;

inline Words capture_bytes(const Tensor& tensor, std::size_t offset, std::size_t bytes) {
    Words words(bytes / 2);
    CUDA_CHECK(cudaMemcpy(words.data(), static_cast<const std::byte*>(tensor.data) + offset, bytes,
                          cudaMemcpyDeviceToHost));
    return words;
}

inline void restore_bytes(const Tensor& tensor, std::size_t offset, const Words& words) {
    CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(tensor.data) + offset, words.data(),
                          words.size() * 2, cudaMemcpyHostToDevice));
}

inline void capture(std::vector<Words>& out, const Tensor& tensor) {
    out.push_back(capture_bytes(tensor, 0, tensor.bytes()));
}

inline void restore(std::vector<Words>& in, const Tensor& tensor) {
    restore_bytes(tensor, 0, in.front());
    in.erase(in.begin());
}

inline void capture_device(std::vector<Words>& out, void* data, std::size_t bytes) {
    Words words(bytes / 2);
    CUDA_CHECK(cudaMemcpy(words.data(), data, bytes, cudaMemcpyDeviceToHost));
    out.push_back(std::move(words));
}

inline void restore_device(std::vector<Words>& in, void* data, std::size_t bytes) {
    const Words& words = in.front();
    CUDA_CHECK(cudaMemcpy(data, words.data(), bytes, cudaMemcpyHostToDevice));
    in.erase(in.begin());
}

struct RankSnapshot {
    std::uint32_t position = 0;
    std::uint64_t previous = 0;
    std::uint64_t current  = 0;
    std::vector<Words> buffers;
    std::vector<std::vector<Words>> kv_pages;
};

inline std::size_t page_bytes(const Tensor& plane) {
    return static_cast<std::size_t>(plane.ne[0]) * static_cast<std::size_t>(plane.ne[1]) *
           static_cast<std::size_t>(plane.ne[2]) * 2;
}

} // namespace state_transaction_detail

struct StateTransaction::Impl {
    using Words = state_transaction_detail::Words;
    using RankSnapshot = state_transaction_detail::RankSnapshot;

    std::array<State*, kMaximumDevices> states{};
    int tp = 0;
    std::uint32_t verify_extent = 0;
    std::vector<RankSnapshot> snapshots;

    Impl(std::array<State*, kMaximumDevices> tracked, int width, std::uint32_t extent)
        : states(tracked), tp(width), verify_extent(extent) {}

    void capture_kv(State& state, RankSnapshot& snapshot) {
        if (verify_extent == 0) { return; }
        const std::uint32_t last = snapshot.position + verify_extent - 1;
        const int first_page = static_cast<int>(snapshot.position / static_cast<std::uint32_t>(ops::kQsaTokenPageSize));
        const int last_page  = static_cast<int>(last / static_cast<std::uint32_t>(ops::kQsaTokenPageSize));
        for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
            const ops::QsaCacheView cache = state.cache(slot);
            const std::size_t token_bytes = state_transaction_detail::page_bytes(cache.k_pages);
            const std::size_t index_bytes = state_transaction_detail::page_bytes(cache.index_k_pages);
            std::vector<Words> slot_pages;
            for (int page = first_page; page <= last_page; ++page) {
                slot_pages.push_back(state_transaction_detail::capture_bytes(cache.k_pages,
                                                   static_cast<std::size_t>(page) * token_bytes,
                                                   token_bytes));
                slot_pages.push_back(state_transaction_detail::capture_bytes(cache.v_pages,
                                                   static_cast<std::size_t>(page) * token_bytes,
                                                   token_bytes));
                slot_pages.push_back(state_transaction_detail::capture_bytes(cache.index_k_pages,
                                                   static_cast<std::size_t>(page) * index_bytes,
                                                   index_bytes));
            }
            snapshot.kv_pages.push_back(std::move(slot_pages));
        }
    }

    void restore_kv(State& state, RankSnapshot& snapshot) {
        if (verify_extent == 0) { return; }
        const std::uint32_t last = snapshot.position + verify_extent - 1;
        const int first_page = static_cast<int>(snapshot.position / static_cast<std::uint32_t>(ops::kQsaTokenPageSize));
        const int last_page  = static_cast<int>(last / static_cast<std::uint32_t>(ops::kQsaTokenPageSize));
        for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
            const ops::QsaCacheView cache = state.cache(slot);
            const std::size_t token_bytes = state_transaction_detail::page_bytes(cache.k_pages);
            const std::size_t index_bytes = state_transaction_detail::page_bytes(cache.index_k_pages);
            const std::vector<Words>& slot_pages = snapshot.kv_pages[static_cast<std::size_t>(slot)];
            std::size_t cursor = 0;
            for (int page = first_page; page <= last_page; ++page) {
                state_transaction_detail::restore_bytes(cache.k_pages, static_cast<std::size_t>(page) * token_bytes,
                              slot_pages[cursor++]);
                state_transaction_detail::restore_bytes(cache.v_pages, static_cast<std::size_t>(page) * token_bytes,
                              slot_pages[cursor++]);
                state_transaction_detail::restore_bytes(cache.index_k_pages, static_cast<std::size_t>(page) * index_bytes,
                              slot_pages[cursor++]);
            }
        }
    }

    void capture_state(State& state, RankSnapshot& snapshot) {
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        CUDA_CHECK(cudaStreamSynchronize(state.stream()));
        snapshot.position = state.next_position();
        snapshot.previous = state.ple_tail(0);
        snapshot.current  = state.ple_tail(1);
        for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
            state_transaction_detail::capture(snapshot.buffers, state.gdn_conv_state(index));
            state_transaction_detail::capture(snapshot.buffers, state.gdn_recurrent_state(index));
        }
        for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
            state_transaction_detail::capture(snapshot.buffers, state.before_tail(slot).raw_keys);
            state_transaction_detail::capture(snapshot.buffers, state.before_tail(slot).first_rope);
        }
        state_transaction_detail::capture(snapshot.buffers, state.tail(0).raw_keys);
        state_transaction_detail::capture(snapshot.buffers, state.tail(0).first_rope);
        state_transaction_detail::capture(snapshot.buffers, state.ple_history());
        state_transaction_detail::capture_device(snapshot.buffers, state.route_counter(), sizeof(std::int32_t));
        capture_kv(state, snapshot);
    }

    void restore_state(State& state, RankSnapshot& snapshot) {
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        CUDA_CHECK(cudaStreamSynchronize(state.stream()));
        state.rewind_scalars(snapshot.position, snapshot.previous, snapshot.current);
        for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
            state_transaction_detail::restore(snapshot.buffers, state.gdn_conv_state(index));
            state_transaction_detail::restore(snapshot.buffers, state.gdn_recurrent_state(index));
        }
        for (int slot = 0; slot < TextConfig::qsa_layers(); ++slot) {
            state_transaction_detail::restore(snapshot.buffers, state.before_tail(slot).raw_keys);
            state_transaction_detail::restore(snapshot.buffers, state.before_tail(slot).first_rope);
        }
        state_transaction_detail::restore(snapshot.buffers, state.tail(0).raw_keys);
        state_transaction_detail::restore(snapshot.buffers, state.tail(0).first_rope);
        state_transaction_detail::restore(snapshot.buffers, state.ple_history());
        state_transaction_detail::restore_device(snapshot.buffers, state.route_counter(), sizeof(std::int32_t));
        restore_kv(state, snapshot);
    }
};

StateTransaction::StateTransaction(std::array<State*, kMaximumDevices> states, int tp,
                                          std::uint32_t verify_extent)
    : impl_(std::make_unique<Impl>(states, tp, verify_extent)) {}

StateTransaction::~StateTransaction() = default;

void StateTransaction::begin() {
    impl_->snapshots.clear();
    impl_->snapshots.reserve(static_cast<std::size_t>(impl_->tp));
    for (int rank = 0; rank < impl_->tp; ++rank) {
        state_transaction_detail::RankSnapshot snapshot;
        impl_->capture_state(*impl_->states[static_cast<std::size_t>(rank)], snapshot);
        impl_->snapshots.push_back(std::move(snapshot));
    }
}

void StateTransaction::commit(std::span<const TokenId> committed, const Replay& replay) {
    for (int rank = 0; rank < impl_->tp; ++rank) {
        impl_->restore_state(*impl_->states[static_cast<std::size_t>(rank)],
                             impl_->snapshots[static_cast<std::size_t>(rank)]);
    }
    for (const TokenId token : committed) { replay(token); }
}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
