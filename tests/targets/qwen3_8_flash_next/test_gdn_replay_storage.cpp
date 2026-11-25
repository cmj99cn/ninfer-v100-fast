#include "flash_model_harness.h"
#include "ninfer/ops/gdn_replay.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using flash_harness::FlashHarness;
using flash_harness::require;
using ninfer::TokenId;
namespace rt = flash_harness::rt;
namespace ops = ninfer::ops;

constexpr std::int32_t kLayers     = 36;
constexpr std::int32_t kQkHeads    = 8;
constexpr std::int32_t kValueHeads = 24;
constexpr std::int32_t kChannels   = 5120;
constexpr std::int32_t kStateDim   = 128;
// The harness forward width is 8; replay width must be the explicitly passed verify span instead.
constexpr std::int32_t kReplayWidth = 4;
constexpr std::int32_t kCapacity   = 8;

void check_shape(const ninfer::Tensor& tensor, ninfer::DType dtype,
                 std::initializer_list<std::int32_t> shape, const std::string& label) {
    require(tensor.data != nullptr && tensor.dtype == dtype && tensor.is_contiguous(),
            label + ": invalid tensor view");
    const ninfer::Tensor expected(nullptr, dtype, shape);
    for (int dim = 0; dim < 4; ++dim) {
        require(tensor.ne[dim] == expected.ne[dim], label + ": unexpected extent");
    }
}

void check_rank(rt::TextState& state, int rank) {
    const std::string tag = " on rank " + std::to_string(rank);
    const auto& records    = state.gdn_replay_records();
    const auto& spec       = records.spec;
    require(spec.activation_dtype == ninfer::DType::FP16, "activation dtype" + tag);
    require(spec.layers == kLayers, "layers" + tag);
    require(spec.qk_heads == kQkHeads, "qk heads" + tag);
    require(spec.value_heads == kValueHeads, "value heads" + tag);
    require(spec.conv_channels == kChannels, "conv channels" + tag);
    require(spec.key_dim == kStateDim && spec.value_dim == kStateDim, "state dims" + tag);
    require(spec.width == kReplayWidth, "replay width" + tag);
    require(spec.width != static_cast<std::int32_t>(flash_harness::kForwardTokens),
            "replay width must not follow forward_tokens" + tag);
    require(spec.record_capacity == kCapacity, "record capacity" + tag);

    const auto first = records.layer(0, 1);
    check_shape(first.conv, spec.activation_dtype, {kChannels, kReplayWidth, 1},
                "conv layer 0" + tag);
    check_shape(first.key, spec.activation_dtype, {kStateDim, kQkHeads, kReplayWidth, 1},
                "key layer 0" + tag);
    check_shape(first.value, spec.activation_dtype, {kStateDim, kValueHeads, kReplayWidth, 1},
                "value layer 0" + tag);
    check_shape(first.gate, ninfer::DType::FP32, {2, kValueHeads, kReplayWidth, 1},
                "gate layer 0" + tag);
    const auto last = records.layer(kLayers - 1, kCapacity);
    check_shape(last.conv, spec.activation_dtype, {kChannels, kReplayWidth, kCapacity},
                "last conv layer" + tag);
    (void)last.key;
    (void)last.value;
    (void)last.gate;

    const auto view = state.gdn_all_layers_view();
    require(view.spec.layers == static_cast<std::uint32_t>(kLayers), "view layers" + tag);
    require(view.spec.conv_channels == kChannels && view.spec.conv_width == 3,
            "view conv geometry" + tag);
    require(view.spec.value_heads == kValueHeads && view.spec.value_head_dim == kStateDim &&
                view.spec.key_head_dim == kStateDim,
            "view head geometry" + tag);
    require(view.spec.slot_count == 1, "view slot count" + tag);
    require(view.spec.conv_dtype == spec.activation_dtype, "view conv dtype" + tag);
    check_shape(view.conv_layer0, spec.activation_dtype, {3, kChannels, 1}, "conv layer0" + tag);
    check_shape(view.recurrent_layer0, ninfer::DType::FP32, {kStateDim, kStateDim, kValueHeads, 1},
                "recurrent layer0" + tag);
    require(view.conv_layer_stride_bytes == static_cast<std::int64_t>(kChannels) * 3 * 2,
            "conv layer stride" + tag);
    require(view.recurrent_layer_stride_bytes ==
                static_cast<std::int64_t>(kStateDim) * kStateDim * kValueHeads * 4,
            "recurrent layer stride" + tag);

    const std::uint64_t before = flash_harness::state_digest(state);
    const ops::GdnReplayFoldRow rows[] = {{0, 0}};
    ops::gdn_replay_fold(records, view, rows, state.stream());
    const std::uint64_t after = flash_harness::state_digest(state);
    require(before == after, "commit_columns=0 changed persistent state" + tag);
}

std::uint64_t gdn_state_digest(rt::TextState& state) {
    CUDA_CHECK(cudaSetDevice(state.device_id()));
    CUDA_CHECK(cudaStreamSynchronize(state.stream()));
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
        hash = flash_harness::hash_tensor(hash, state.gdn_conv_state(index));
        hash = flash_harness::hash_tensor(hash, state.gdn_recurrent_state(index));
    }
    return hash;
}

std::array<std::uint64_t, 2> gdn_digests(FlashHarness& harness) {
    return {gdn_state_digest(*harness.states[0]), gdn_state_digest(*harness.states[1])};
}

bool plane_has_data(const ninfer::Tensor& plane) {
    std::vector<std::uint8_t> bytes(plane.bytes());
    CUDA_CHECK(cudaMemcpy(bytes.data(), plane.data, bytes.size(), cudaMemcpyDeviceToHost));
    return std::any_of(bytes.begin(), bytes.end(), [](std::uint8_t byte) { return byte != 0; });
}

void check_record_bypass(FlashHarness& harness) {
    const std::vector<TokenId> prompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};
    harness.reset();
    const TokenId anchor = harness.prefill(prompt);
    const TokenId next   = harness.decode(anchor);

    const auto before       = gdn_digests(harness);
    const std::uint64_t all = harness.digest();
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::RecordForReplay);
    const std::vector<TokenId> span{anchor, next, anchor, next};
    const auto winners = harness.trunk->decode_span(span);
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::UpdateInPlace);
    require(winners.size() == span.size(), "record span returned the wrong winner count");

    const auto after = gdn_digests(harness);
    for (int rank = 0; rank < 2; ++rank) {
        require(before[rank] == after[rank],
                "RecordForReplay changed persistent GDN state on rank " + std::to_string(rank));
    }
    require(harness.digest() != all, "the record span did not advance the rest of the state");

    for (int rank = 0; rank < 2; ++rank) {
        auto& state = *harness.states[static_cast<std::size_t>(rank)];
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        const std::string tag = " record on rank " + std::to_string(rank);
        const auto& records    = state.gdn_replay_records();
        require(records.conv.data != nullptr && records.key.data != nullptr &&
                    records.value.data != nullptr && records.gate.data != nullptr,
                "null record plane" + tag);
        const auto layer = records.layer(0, 1);
        check_shape(layer.conv, records.spec.activation_dtype,
                    {kChannels, records.spec.width, 1}, "conv" + tag);
        check_shape(layer.key, records.spec.activation_dtype,
                    {kStateDim, kQkHeads, records.spec.width, 1}, "key" + tag);
        check_shape(layer.value, records.spec.activation_dtype,
                    {kStateDim, kValueHeads, records.spec.width, 1}, "value" + tag);
        check_shape(layer.gate, ninfer::DType::FP32,
                    {2, kValueHeads, records.spec.width, 1}, "gate" + tag);
        require(plane_has_data(layer.conv) && plane_has_data(layer.key) &&
                    plane_has_data(layer.value) && plane_has_data(layer.gate),
                "empty record plane" + tag);
    }

    const auto inplace_before = gdn_digests(harness);
    (void)harness.trunk->decode_span(std::span<const TokenId>{span.data(), 2});
    const auto inplace_after = gdn_digests(harness);
    for (int rank = 0; rank < 2; ++rank) {
        require(inplace_before[rank] != inplace_after[rank],
                "UpdateInPlace stopped committing GDN state on rank " + std::to_string(rank));
    }
}

void check_reset_zeroes_records(FlashHarness& harness) {
    for (int rank = 0; rank < 2; ++rank) {
        auto& state = *harness.states[rank];
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        const auto& records = state.gdn_replay_records();
        CUDA_CHECK(cudaMemset(records.gate.data, 0x7F, records.gate.bytes()));
    }
    harness.reset();
    for (int rank = 0; rank < 2; ++rank) {
        auto& state = *harness.states[rank];
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        const auto& records = state.gdn_replay_records();
        std::vector<std::uint8_t> bytes(records.gate.bytes());
        CUDA_CHECK(cudaMemcpy(bytes.data(), records.gate.data, bytes.size(),
                              cudaMemcpyDeviceToHost));
        for (const std::uint8_t byte : bytes) {
            require(byte == 0, "reset must zero the replay records on rank " +
                                   std::to_string(rank));
        }
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (artifact == nullptr || artifact[0] == '\0') {
        std::cout << "skip: replay storage needs the artifact\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }
    try {
        FlashHarness harness(artifact, static_cast<std::uint32_t>(kReplayWidth));
        check_rank(*harness.states[0], 0);
        check_rank(*harness.states[1], 1);
        check_reset_zeroes_records(harness);
        check_record_bypass(harness);
        require(harness.states[0]->gdn_replay_records().conv.data !=
                    harness.states[1]->gdn_replay_records().conv.data,
                "TP ranks must own distinct replay backing allocations");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK flash-next GDN replay storage: 36-layer records, dense all-layers view, "
                 "per-rank backing, reset zeroing, fold validation accepts commit_columns=0, "
                 "record bypass preserves persistent GDN state and UpdateInPlace still commits\n";
    return 0;
}
