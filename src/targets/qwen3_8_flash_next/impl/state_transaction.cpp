#include "targets/qwen3_8_flash_next/impl/state_transaction.h"

#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/variant.h"

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_runtime
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#undef NINFER_QWEN4EXP_RUNTIME_NS
#undef NINFER_QWEN4EXP_VARIANT

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace state_rt = ninfer::targets::qwen4_exp::detail::flash_next_runtime;

namespace {

using Words = std::vector<std::uint16_t>;

void capture(std::vector<Words>& out, const Tensor& tensor) {
    Words words(tensor.bytes() / 2);
    CUDA_CHECK(cudaMemcpy(words.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    out.push_back(std::move(words));
}

void restore(std::vector<Words>& in, const Tensor& tensor) {
    const Words& words = in.front();
    CUDA_CHECK(cudaMemcpy(tensor.data, words.data(), words.size() * 2, cudaMemcpyHostToDevice));
    in.erase(in.begin());
}

void capture_device(std::vector<Words>& out, void* data, std::size_t bytes) {
    Words words(bytes / 2);
    CUDA_CHECK(cudaMemcpy(words.data(), data, bytes, cudaMemcpyDeviceToHost));
    out.push_back(std::move(words));
}

void restore_device(std::vector<Words>& in, void* data, std::size_t bytes) {
    const Words& words = in.front();
    CUDA_CHECK(cudaMemcpy(data, words.data(), bytes, cudaMemcpyHostToDevice));
    in.erase(in.begin());
}

struct RankSnapshot {
    std::uint32_t position = 0;
    std::uint64_t previous = 0;
    std::uint64_t current  = 0;
    std::vector<Words> buffers;
};

} // namespace

struct StateTransaction::Impl {
    std::array<State*, kMaximumDevices> states{};
    int tp = 0;
    std::vector<RankSnapshot> snapshots;

    Impl(std::array<State*, kMaximumDevices> tracked, int width)
        : states(tracked), tp(width) {}

    void capture_state(State& state, RankSnapshot& snapshot) {
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        CUDA_CHECK(cudaStreamSynchronize(state.stream()));
        snapshot.position = state.next_position();
        snapshot.previous = state.ple_tail(0);
        snapshot.current  = state.ple_tail(1);
        for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
            capture(snapshot.buffers, state.gdn_conv_state(index));
            capture(snapshot.buffers, state.gdn_recurrent_state(index));
        }
        for (int slot = 0; slot < state_rt::TextConfig::qsa_layers(); ++slot) {
            capture(snapshot.buffers, state.before_tail(slot).raw_keys);
            capture(snapshot.buffers, state.before_tail(slot).first_rope);
        }
        capture(snapshot.buffers, state.tail(0).raw_keys);
        capture(snapshot.buffers, state.tail(0).first_rope);
        capture(snapshot.buffers, state.ple_history());
        capture_device(snapshot.buffers, state.route_counter(), sizeof(std::int32_t));
    }

    void restore_state(State& state, RankSnapshot& snapshot) {
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        CUDA_CHECK(cudaStreamSynchronize(state.stream()));
        state.rewind_scalars(snapshot.position, snapshot.previous, snapshot.current);
        for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
            restore(snapshot.buffers, state.gdn_conv_state(index));
            restore(snapshot.buffers, state.gdn_recurrent_state(index));
        }
        for (int slot = 0; slot < state_rt::TextConfig::qsa_layers(); ++slot) {
            restore(snapshot.buffers, state.before_tail(slot).raw_keys);
            restore(snapshot.buffers, state.before_tail(slot).first_rope);
        }
        restore(snapshot.buffers, state.tail(0).raw_keys);
        restore(snapshot.buffers, state.tail(0).first_rope);
        restore(snapshot.buffers, state.ple_history());
        restore_device(snapshot.buffers, state.route_counter(), sizeof(std::int32_t));
    }
};

StateTransaction::StateTransaction(std::array<State*, kMaximumDevices> states, int tp)
    : impl_(std::make_unique<Impl>(states, tp)) {}

StateTransaction::~StateTransaction() = default;

void StateTransaction::begin() {
    impl_->snapshots.clear();
    impl_->snapshots.reserve(static_cast<std::size_t>(impl_->tp));
    for (int rank = 0; rank < impl_->tp; ++rank) {
        RankSnapshot snapshot;
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

} // namespace ninfer::targets::qwen3_8_flash_next::detail
