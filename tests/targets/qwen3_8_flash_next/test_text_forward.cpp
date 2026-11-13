#include "fixture.h"

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

namespace {

using namespace ninfer;
using ninfer_flash_next_fixture::text_config;

namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;
namespace detail  = ninfer::targets::qwen3_8_flash_next::detail;
namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_text_test_runtime;

constexpr std::size_t kLoadedLayers = 4;
constexpr std::size_t kPromptTokens = 8;
constexpr std::size_t kDecodeRounds = 2;

constexpr std::array<std::int32_t, kPromptTokens> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint64_t descriptor_offset(const artifact::ObjectDescriptor& descriptor) {
    return std::visit([](const auto& d) { return d.offset; }, descriptor);
}

std::vector<std::byte> read_range(const artifact::Reader& reader, const std::string& object,
                                  std::uint64_t offset, std::uint64_t bytes) {
    const auto* descriptor = reader.find(object);
    if (descriptor == nullptr) { throw std::runtime_error("the artifact does not carry " + object); }
    std::vector<std::byte> out(bytes);
    const auto read = reader.read_direct(reader.payload_offset() + descriptor_offset(*descriptor) + offset,
                                         out);
    if (read != bytes) { throw std::runtime_error("short read for " + object); }
    return out;
}

std::vector<std::string> ple_shard_names(const detail::BindingPlan& plan) {
    std::vector<std::string> names;
    for (const auto& part : plan.ple.table.parts) {
        names.push_back(plan.objects[part.object].name);
    }
    return names;
}

void place_reduced(const detail::BindingPlan& plan, artifact::Binder& binder) {
    std::size_t object_slots = 0;
    for (const auto& object : plan.objects) { object_slots = std::max(object_slots, object.handle.index + 1); }
    std::vector<bool> reduced(object_slots, false);
    for (const auto* parameter : plan.parameters()) {
        if (parameter->role == roles::Role::PleTable) { continue; }
        if (parameter->layer >= kLoadedLayers) { continue; }
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
    for (std::size_t layer = 0; layer < state.gdn_state_entries(); ++layer) {
        const Tensor state_tensor = state.recurrent_state(layer);
        std::vector<std::byte> bytes(state_tensor.bytes());
        CUDA_CHECK(cudaMemcpy(bytes.data(), state_tensor.data, bytes.size(), cudaMemcpyDeviceToHost));
        for (const std::byte byte : bytes) {
            hash = hash * 1000003U + static_cast<std::size_t>(byte);
        }
    }
    return hash;
}

TokenId run_prompt(DeviceContext& device, const detail::BindingPlan& plan,
                   const artifact::MaterializedArtifact& materialized,
                   const ops::PleHashControls& controls, const std::vector<std::byte>& staged,
                   const std::vector<std::uint32_t>& rows, const std::vector<float>& divisors) {
    detail::LoadedModelData data(plan, materialized, kLoadedLayers);
    rt::TextState state(device, data.view(), static_cast<std::uint32_t>(kPromptTokens));
    state.set_ple_controls(controls);

    DeviceBuffer device_divisors(divisors.size() * 4);
    device_divisors.copy_from_host(divisors.data(), divisors.size() * 4);
    state.set_ple_divisors(static_cast<const float*>(device_divisors.p));

    DeviceBuffer staged_rows(staged.size());
    staged_rows.copy_from_host(staged.data(), staged.size());
    DeviceBuffer rows_buffer(rows.size() * 4);
    rows_buffer.copy_from_host(rows.data(), rows.size() * 4);
    state.set_ple_rows(static_cast<const std::uint32_t*>(rows_buffer.p), rows.size());
    state.set_ple_staged_rows(static_cast<const std::uint8_t*>(staged_rows.p), staged.size());

    DeviceArena activations(1 << 28);
    DeviceArena workspace(1 << 26);
    rt::TextTrunk trunk(data.view(), state, activations, workspace, device.stream);

    const std::span<const TokenId> prompt{kPrompt.data(), kPrompt.size()};
    const TokenId first = trunk.prefill(prompt);

    const std::size_t before = gdn_state_signature(state);
    TokenId token = first;
    for (std::size_t round = 0; round < kDecodeRounds; ++round) {
        token = trunk.decode(token);
    }
    require(gdn_state_signature(state) != before, "the GDN recurrent state was not updated");
    require(state.gdn_state_entries() == kLoadedLayers, "the GDN state pool does not grow with layers");
    return first;
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

    const auto controls_payload = reader.payload("resource/text/ple_hash_controls");
    const ops::PleHashControls controls = ops::parse_ple_hash_controls(controls_payload.data);

    const std::vector<std::string> shards = ple_shard_names(plan);
    const std::array<std::uint64_t, 2> table_shape{detail::TextConfig::ple_shard_rows,
                                                   detail::TextConfig::ple_head_width};
    const artifact::RowPagedNf4Geometry geometry =
        artifact::row_paged_nf4_geometry(artifact::NumericFormat::NF4_G16_E4M3S, table_shape);

    std::vector<float> divisors(ops::kPleShards, 1.0F);
    for (std::size_t shard = 0; shard < ops::kPleShards; ++shard) {
        const auto word = read_range(reader, shards[shard], geometry.matrix_scale_offset, 4);
        float value = 0.0F;
        std::memcpy(&value, word.data(), sizeof(value));
        divisors[shard] = value;
    }

    std::vector<std::uint64_t> ids(kPrompt.begin(), kPrompt.end());
    std::array<std::uint64_t, 2> tail{detail::TextConfig::eos_token_id, detail::TextConfig::eos_token_id};
    std::vector<std::uint32_t> rows;
    for (std::uint64_t id : ids) {
        std::vector<std::uint32_t> token_rows(ops::kPleHeads);
        ops::ple_hash_tokens(std::span<const std::uint64_t>{&id, 1}, tail, detail::TextConfig::eos_token_id,
                             controls, token_rows);
        for (std::uint32_t row : token_rows) { rows.push_back(row); }
        tail[0] = tail[1];
        tail[1] = id;
    }

    std::vector<std::byte> staged;
    for (std::uint32_t row : rows) {
        const ops::PleRowAddress address = ops::ple_row_address(
            ops::PleNf4Table{geometry, ops::kPleShards, nullptr}, row);
        const std::uint64_t offset = artifact::row_paged_nf4_row_offset(geometry, address.row);
        const auto bytes = read_range(reader, shards[address.shard], offset, geometry.row_bytes);
        staged.insert(staged.end(), bytes.begin(), bytes.end());
    }

    const TokenId first = run_prompt(device, plan, materialized, controls, staged, rows, divisors);
    require(first >= 0 && first < static_cast<TokenId>(detail::TextConfig::vocab),
           "the greedy token is outside the vocabulary");
    const TokenId repeated = run_prompt(device, plan, materialized, controls, staged, rows, divisors);
    require(repeated == first, "the trunk is not deterministic across repeated runs");

    if (reference.has_value()) {
        require(first == *reference, "the greedy token " + std::to_string(first) +
                                         " does not match the reference token " +
                                         std::to_string(*reference));
    } else {
        std::cout << "SKIP: the reference token parity check (NINFER_FLASH_NEXT_REFERENCE_TOKEN)\n";
    }

    std::cout << "OK flash-next text trunk: token " << first << " deterministic\n";
    return 0;
}
