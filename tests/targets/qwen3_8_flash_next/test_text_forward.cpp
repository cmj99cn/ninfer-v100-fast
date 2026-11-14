#include "fixture.h"

#include "artifact/typed_binding.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/ple.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <variant>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_text_test_runtime
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

namespace {

using namespace ninfer;
namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;
namespace detail  = ninfer::targets::qwen3_8_flash_next::detail;
namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_text_test_runtime;
namespace qwen4_exp = ninfer::targets::qwen4_exp;

constexpr std::size_t kLoadedLayers = 4;
constexpr std::size_t kPromptTokens = 8;
constexpr std::size_t kDecodeRounds = 3;
constexpr std::uint32_t kKvCapacityTokens = 64;

constexpr std::array<std::int32_t, kPromptTokens> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void place_reduced(const detail::BindingPlan& plan, artifact::Binder& binder) {
    std::size_t object_slots = 0;
    for (const auto& object : plan.objects) {
        object_slots = std::max(object_slots, object.handle.index + 1);
    }
    std::vector<bool> reduced(object_slots, false);
    for (const auto* parameter : plan.parameters()) {
        if (parameter->role == roles::Role::PleTable) { continue; }
        const bool prefix_layer = parameter->layer < kLoadedLayers;
        const bool prefix_tail = parameter->layer == kLoadedLayers &&
                                 parameter->role == roles::Role::AttentionHcNorm;
        if (!prefix_layer && !prefix_tail) { continue; }
        for (const auto& part : parameter->parts) {
            reduced[plan.objects[static_cast<std::size_t>(part.object)].handle.index] = true;
        }
        if (parameter->library.has_value()) {
            reduced[parameter->library->handle.index] = true;
        }
    }
    for (const auto& object : plan.objects) {
        const auto handle = object.handle;
        if (reduced[static_cast<std::size_t>(handle.index)] &&
            std::holds_alternative<artifact::TensorDescriptor>(binder.descriptor(handle))) {
            binder.materialize_on_device(handle);
        } else {
            binder.validate_only(handle);
        }
    }
}

std::size_t gdn_state_signature(const rt::TextState& state) {
    std::size_t hash = 0;
    for (std::size_t entry = 0; entry < state.gdn_state_entries(); ++entry) {
        const Tensor state_tensor = state.gdn_recurrent_state(entry);
        std::vector<std::byte> bytes(state_tensor.bytes());
        CUDA_CHECK(cudaMemcpy(bytes.data(), state_tensor.data, bytes.size(), cudaMemcpyDeviceToHost));
        for (const std::byte byte : bytes) {
            hash = hash * 1000003U + static_cast<std::size_t>(byte);
        }
    }
    return hash;
}

std::size_t tail_signature(const rt::TextState& state) {
    const ops::ConstQsaTailStateView tail = state.const_tail();
    std::vector<std::byte> keys(tail.raw_keys.bytes());
    std::vector<std::byte> rope(tail.first_rope.bytes());
    CUDA_CHECK(cudaMemcpy(keys.data(), tail.raw_keys.data, keys.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rope.data(), tail.first_rope.data, rope.size(), cudaMemcpyDeviceToHost));
    std::size_t hash = 0;
    for (const std::byte byte : keys) { hash = hash * 1000003U + static_cast<std::size_t>(byte); }
    for (const std::byte byte : rope) { hash = hash * 1000003U + static_cast<std::size_t>(byte); }
    return hash;
}

struct RunOutcome {
    TokenId token = 0;
    bool gdn_updated = false;
    std::size_t gdn_entries = 0;
    std::array<std::size_t, kDecodeRounds + 1> tail_signatures{};
    std::array<std::int32_t, kDecodeRounds + 1> base_positions{};
};

void check_gdn_slot_mapping(DeviceContext& device, const detail::Variant::ModelView& view) {
    const rt::TextState state(device, view, kKvCapacityTokens, kPromptTokens);
    std::vector<std::size_t> slots;
    for (std::size_t layer = 0; layer < kLoadedLayers; ++layer) {
        const std::optional<std::size_t> slot = state.gdn_state_index(layer);
        if (detail::TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            require(!slot.has_value(), "a QSA layer mapped to a GDN state slot");
            continue;
        }
        require(slot.has_value(), "a GDN layer has no state slot");
        require(*slot < state.gdn_state_entries(), "a GDN state slot is outside the pool");
        slots.push_back(*slot);
    }
    require(slots.size() == state.gdn_state_entries(), "the GDN slots do not cover the pool");
    std::sort(slots.begin(), slots.end());
    for (std::size_t index = 0; index < slots.size(); ++index) {
        require(slots[index] == index, "the GDN state slots are not a permutation");
    }
}

RunOutcome run_prompt(DeviceContext& device, const detail::Variant::ModelView& view) {
    using ModelView = detail::Variant::ModelView;
    rt::TextState state(device, view, kKvCapacityTokens, kPromptTokens);

    const auto recipe = rt::text_workspace_recipe(kPromptTokens, 1);
    DeviceArena activations(rt::text_activation_capacity_bytes(kPromptTokens));
    DeviceArena workspace(recipe.lane_capacity_bytes);
    rt::TextTrunk trunk(view, state, activations, workspace, device.stream);

    const std::span<const TokenId> prompt{kPrompt.data(), kPrompt.size()};
    const TokenId first = trunk.prefill(prompt);

    RunOutcome outcome;
    outcome.token = first;
    outcome.tail_signatures[0] = tail_signature(state);
    outcome.base_positions[0] = state.base_position();
    const std::size_t before = gdn_state_signature(state);
    TokenId token = first;
    for (std::size_t round = 0; round < kDecodeRounds; ++round) {
        token = trunk.decode(token);
        outcome.tail_signatures[round + 1] = tail_signature(state);
        outcome.base_positions[round + 1] = state.base_position();
    }
    outcome.gdn_updated = gdn_state_signature(state) != before;
    outcome.gdn_entries = state.gdn_state_entries();
    return outcome;
}

struct ShiftedPleRows {
    qwen4_exp::PleRowSource base;
};

void fetch_shifted_rows(void* context, const ops::PleRowAddress* addresses, std::size_t count,
                        std::byte* destination) {
    auto* shifted = static_cast<ShiftedPleRows*>(context);
    std::vector<ops::PleRowAddress> moved(addresses, addresses + count);
    const std::uint32_t rows = static_cast<std::uint32_t>(detail::TextConfig::ple_shard_rows);
    for (auto& address : moved) { address.row = (address.row + 1) % rows; }
    shifted->base.fetch(shifted->base.context, moved.data(), moved.size(), destination);
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (path == nullptr) {
        std::cout << "SKIP: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
        return 77;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }

    const char* reference_text = std::getenv("NINFER_FLASH_NEXT_REFERENCE_TOKEN");
    const std::optional<TokenId> reference =
        reference_text != nullptr ? std::optional<TokenId>{static_cast<TokenId>(std::stoll(reference_text))}
                                  : std::nullopt;

    DeviceContext device(0);
    artifact::Reader reader(path);
    artifact::Binder binder(reader, 1);

    const detail::BindingPlan plan = detail::bind_artifact(binder, detail::WeightsProfile::Nvfp4, 1);
    place_reduced(plan, binder);
    binder.validate_unconsumed_matching();
    const artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, binder.finish(), device);
    const std::filesystem::path artifact_path(path);
    const detail::LoadedModelData data(plan, materialized, kLoadedLayers, artifact_path);

    const auto key_bytes = static_cast<std::uint64_t>(detail::TextConfig::residual_width) *
                           detail::TextConfig::hidden * 2;
    const auto key_base = static_cast<const std::byte*>(data.view().ple.key.payload);
    const auto value_base = static_cast<const std::byte*>(data.view().ple.value.payload);
    require(value_base - key_base == static_cast<std::ptrdiff_t>(key_bytes),
            "the PLE value view does not start at the value rows");

    check_gdn_slot_mapping(device, data.view());
    const RunOutcome outcome = run_prompt(device, data.view());
    require(outcome.token >= 0 && outcome.token < static_cast<TokenId>(detail::TextConfig::vocab),
            "the greedy token is outside the vocabulary");
    const RunOutcome repeated = run_prompt(device, data.view());
    require(repeated.token == outcome.token, "the trunk is not deterministic across repeated runs");

    require(outcome.gdn_entries == 3, "the GDN state pool does not match the loaded GDN layers");
    require(outcome.gdn_updated, "the GDN recurrent state was not updated");

    for (std::size_t round = 0; round < outcome.base_positions.size(); ++round) {
        const std::int32_t expected =
            round == 0 ? 0 : static_cast<std::int32_t>(kPromptTokens + round - 1);
        require(outcome.base_positions[round] == expected,
                "the QSA record base position does not follow the round");
    }
    for (std::size_t first = 0; first < outcome.tail_signatures.size(); ++first) {
        for (std::size_t other = first + 1; other < outcome.tail_signatures.size(); ++other) {
            require(outcome.tail_signatures[first] != outcome.tail_signatures[other],
                    "the QSA canonical tail did not advance between rounds");
        }
    }

    ShiftedPleRows shifted{qwen4_exp::PleRowSource{data.view().ple.rows.context,
                                                   data.view().ple.rows.fetch,
                                                   data.view().ple.rows.divisors}};
    auto shifted_view = data.view();
    shifted_view.ple.rows.context = &shifted;
    shifted_view.ple.rows.fetch = &fetch_shifted_rows;
    const RunOutcome shifted_outcome = run_prompt(device, shifted_view);
    require(shifted_outcome.token != outcome.token,
            "the PLE row source did not change the sampled token");

    auto zeroed_view = data.view();
    std::uint64_t mixer_bytes = 16;
    {
        const auto encoded = artifact::tensor_encoded_size(
            artifact::StorageLayout::RowSplitK128V1, artifact::NumericFormat::W8G32_F16S,
            std::array<std::uint64_t, 2>{static_cast<std::uint64_t>(detail::TextConfig::hidden),
                                         static_cast<std::uint64_t>(detail::TextConfig::gdn_value_width)});
        mixer_bytes = encoded;
    }
    DeviceBuffer zeroed_mixer(mixer_bytes);
    zeroed_mixer.fill(0);
    zeroed_view.layers[1].gdn_projection.output = artifact::staged_weight_view(
        artifact::NumericFormat::W8G32_F16S, zeroed_mixer.p, detail::TextConfig::hidden,
        detail::TextConfig::gdn_value_width);
    const RunOutcome zeroed_outcome = run_prompt(device, zeroed_view);
    require(zeroed_outcome.token != outcome.token,
            "zeroing a mixer output did not change the sampled token");

    if (reference.has_value()) {
        require(outcome.token == *reference, "the greedy token " + std::to_string(outcome.token) +
                                                 " does not match the reference token " +
                                                 std::to_string(*reference));
    } else {
        std::cout << "SKIP: the reference token parity check (NINFER_FLASH_NEXT_REFERENCE_TOKEN)\n";
    }

    std::cout << "OK flash-next text trunk: token " << outcome.token << " deterministic\n";
    return 0;
}
