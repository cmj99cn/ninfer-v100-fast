#include "fixture.h"

#include "artifact/typed_binding.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen4_exp/impl/runtime/activation_dtype.h"

#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <algorithm>
#include <array>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <variant>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

struct QsaDumpEntry {
    std::int32_t position = 0;
    std::vector<std::int32_t> blocks;
};

std::vector<QsaDumpEntry> g_qsa_entries;
std::vector<float> g_index_query;
std::vector<float> g_index_keys;
std::int32_t g_index_tokens = 0;
std::int32_t g_index_pages  = 0;
bool g_qsa_recording = false;

void flash_next_record_qsa(const ninfer::ops::QsaBatchView& batch,
                           const ninfer::ops::QsaBlockSelection& selected,
                           const ninfer::Tensor& index_query,
                           const ninfer::ops::ConstQsaIndexCacheView& cache, cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::int32_t tokens = batch.positions.ne[0];
    const std::int32_t axes   = batch.positions.ne[1];
    const std::int32_t rows   = batch.positions.ne[2];
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens) * axes * rows);
    std::vector<std::int32_t> counts(static_cast<std::size_t>(tokens) * rows);
    std::vector<std::int32_t> selection(static_cast<std::size_t>(512) * tokens * rows);
    CUDA_CHECK(cudaMemcpy(positions.data(), batch.positions.data, positions.size() * 4,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts.data(), selected.counts.data, counts.size() * 4,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(selection.data(), selected.selection.data, selection.size() * 4,
                          cudaMemcpyDeviceToHost));
    const auto widen = [](const ninfer::Tensor& tensor) {
        std::vector<std::uint16_t> words(tensor.bytes() / 2);
        if (!words.empty()) {
            CUDA_CHECK(cudaMemcpy(words.data(), tensor.data, tensor.bytes(),
                                  cudaMemcpyDeviceToHost));
        }
        std::vector<float> values(words.size());
        for (std::size_t index = 0; index < words.size(); ++index) {
            if (tensor.dtype == ninfer::DType::FP16) {
                __half value;
                std::memcpy(&value, &words[index], sizeof(value));
                values[index] = __half2float(value);
            } else {
                __nv_bfloat16 value;
                std::memcpy(&value, &words[index], sizeof(value));
                values[index] = __bfloat162float(value);
            }
        }
        return values;
    };
    g_index_query = widen(index_query);
    g_index_keys  = widen(cache.index_k_pages);
    g_index_tokens = tokens;
    g_index_pages  = cache.index_k_pages.ne[3];
    for (std::int32_t b = 0; b < rows; ++b) {
        for (std::int32_t t = 0; t < tokens; ++t) {
            QsaDumpEntry entry;
            entry.position = positions[static_cast<std::size_t>(t * axes) * rows + b];
            const std::int32_t count = counts[static_cast<std::size_t>(t) + tokens * b];
            const auto begin = selection.begin() + static_cast<std::ptrdiff_t>(512) * (t + tokens * b);
            entry.blocks.assign(begin, begin + count);
            g_qsa_entries.push_back(std::move(entry));
        }
    }
}

void flash_next_layer_dump(const char* name, std::size_t layer, const ninfer::Tensor& tensor,
                           cudaStream_t stream) {
    static std::string directory;
    if (directory.empty()) {
        const char* configured = std::getenv("NINFER_FLASH_NEXT_LAYER_DUMP");
        if (configured == nullptr || configured[0] == '\0') { return; }
        directory = configured;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::byte> host(tensor.bytes());
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    }
    const std::string stem =
        directory + "/L" + std::to_string(layer) + "_" + name;
    std::FILE* data_file = std::fopen(stem.c_str(), "wb");
    if (data_file == nullptr) { throw std::runtime_error("cannot open the layer dump path"); }
    if (!host.empty()) { std::fwrite(host.data(), 1, host.size(), data_file); }
    std::fclose(data_file);
    std::FILE* manifest = std::fopen((directory + "/manifest.txt").c_str(), "a");
    if (manifest == nullptr) { throw std::runtime_error("cannot open the layer dump manifest"); }
    std::fprintf(manifest, "%zu %s %d 4 %d %d %d %d %zu\n", layer, name,
                 static_cast<int>(tensor.dtype), tensor.ne[0], tensor.ne[1], tensor.ne[2],
                 tensor.ne[3], host.size());
    std::fclose(manifest);
}

namespace ninfer::ops {
inline void ninfer_flash_next_dump_qsa_indexer(const Tensor& index_query,
                                               const ConstQsaIndexCacheView& cache,
                                               const QsaBatchView& batch,
                                               std::uint32_t max_visible_tokens,
                                               WorkspaceArena& workspace,
                                               QsaBlockSelection& selected, cudaStream_t stream) {
    qsa_indexer(index_query, cache, batch, max_visible_tokens, workspace, selected, stream);
    if (g_qsa_recording) { flash_next_record_qsa(batch, selected, index_query, cache, stream); }
}
} // namespace ninfer::ops

#define qsa_indexer ninfer_flash_next_dump_qsa_indexer
#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_text_test_runtime
#define NINFER_QWEN4EXP_LAYER_DUMP(name, layer, tensor) ::flash_next_layer_dump((name), (layer), (tensor), stream_)
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"
#undef qsa_indexer

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

bool g_dump_written = false;

void write_parity_dump(const char* path, const rt::TextState& state,
                       std::span<const TokenId> prompt) {
    const std::size_t tokens = prompt.size();
    const std::size_t heads  = 16;
    std::vector<std::uint32_t> rows(heads * tokens);
    CUDA_CHECK(cudaMemcpy(rows.data(), state.ple_rows(), rows.size() * 4, cudaMemcpyDeviceToHost));
    std::FILE* file = std::fopen(path, "w");
    if (file == nullptr) { throw std::runtime_error("cannot open the parity dump path"); }
    std::fprintf(file, "flash-next-dump v1\n");
    std::fprintf(file, "prompt %zu", tokens);
    for (const TokenId token : prompt) { std::fprintf(file, " %d", static_cast<int>(token)); }
    std::fprintf(file, "\nple %zu\n", tokens);
    for (std::size_t t = 0; t < tokens; ++t) {
        std::fprintf(file, "%zu", t);
        for (std::size_t head = 0; head < heads; ++head) {
            std::fprintf(file, " %u", rows[head + heads * t]);
        }
        std::fprintf(file, "\n");
    }
    std::fprintf(file, "qsa %zu\n", g_qsa_entries.size());
    for (const QsaDumpEntry& entry : g_qsa_entries) {
        std::fprintf(file, "%d %zu", static_cast<int>(entry.position), entry.blocks.size());
        for (const std::int32_t block : entry.blocks) { std::fprintf(file, " %d", block); }
        std::fprintf(file, "\n");
    }
    std::fclose(file);
    const std::string index_path = std::string(path) + ".index";
    std::FILE* index_file = std::fopen(index_path.c_str(), "wb");
    if (index_file == nullptr) { throw std::runtime_error("cannot open the index dump path"); }
    const std::uint64_t query_words = g_index_query.size();
    const std::uint64_t key_words   = g_index_keys.size();
    std::fwrite(&g_index_tokens, sizeof(g_index_tokens), 1, index_file);
    std::fwrite(&g_index_pages, sizeof(g_index_pages), 1, index_file);
    std::fwrite(&query_words, sizeof(query_words), 1, index_file);
    std::fwrite(&key_words, sizeof(key_words), 1, index_file);
    if (!g_index_query.empty()) {
        std::fwrite(g_index_query.data(), sizeof(float), g_index_query.size(), index_file);
    }
    if (!g_index_keys.empty()) {
        std::fwrite(g_index_keys.data(), sizeof(float), g_index_keys.size(), index_file);
    }
    std::fclose(index_file);
}

std::vector<bool> reduced_object_mask(const detail::BindingPlan& plan) {
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
    return reduced;
}

void place_reduced(const detail::BindingPlan& plan, artifact::Binder& binder) {
    const std::vector<bool> reduced = reduced_object_mask(plan);
    for (const auto& object : plan.objects) {
        const auto handle = object.handle;
        if (static_cast<std::size_t>(handle.index) < reduced.size() &&
            reduced[static_cast<std::size_t>(handle.index)] &&
            std::holds_alternative<artifact::TensorDescriptor>(binder.descriptor(handle))) {
            binder.materialize_on_device(handle);
        } else {
            binder.validate_only(handle);
        }
    }
}

artifact::MaterializationPlan reduce_plan(const artifact::MaterializationPlan& plan,
                                          const std::vector<bool>& keep) {
    artifact::MaterializationPlan reduced;
    reduced.object_count = plan.object_count;
    reduced.device_count = plan.device_count;
    std::array<std::uint64_t, artifact::kMaximumDevices> cursor{};
    for (const auto& placement : plan.device_objects) {
        if (static_cast<std::size_t>(placement.object.index) >= keep.size() ||
            !keep[static_cast<std::size_t>(placement.object.index)]) {
            continue;
        }
        const auto device = static_cast<std::size_t>(placement.device);
        const std::uint64_t alignment = placement.alignment == 0 ? 1 : placement.alignment;
        const std::uint64_t aligned = (cursor[device] + alignment - 1) / alignment * alignment;
        artifact::DeviceMaterialization narrowed = placement;
        narrowed.offset = aligned;
        cursor[device] = aligned + placement.bytes;
        reduced.device_objects.push_back(std::move(narrowed));
    }
    reduced.device_capacity_bytes = cursor;
    return reduced;
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
    const ops::ConstQsaTailStateView tail = state.const_tail(0);
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
    const char* dump_path = std::getenv("NINFER_FLASH_NEXT_DUMP");
    const bool dump_here  = dump_path != nullptr && !g_dump_written;
    if (dump_here) {
        g_qsa_entries.clear();
        g_qsa_recording = true;
    }
    const TokenId first = trunk.prefill(prompt);
    if (dump_here) {
        g_qsa_recording = false;
        write_parity_dump(dump_path, state, prompt);
        g_dump_written = true;
    }

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

TokenId run_long_prompt(DeviceContext& device, const detail::Variant::ModelView& view,
                        std::span<const TokenId> prompt, const char* dump_path) {
    using ModelView = detail::Variant::ModelView;
    const std::uint32_t tokens = static_cast<std::uint32_t>(prompt.size());
    rt::TextState state(device, view, tokens, tokens);

    const auto recipe = rt::text_workspace_recipe(tokens, 1);
    DeviceArena activations(rt::text_activation_capacity_bytes(tokens));
    DeviceArena workspace(recipe.lane_capacity_bytes);
    rt::TextTrunk trunk(view, state, activations, workspace, device.stream);

    g_qsa_entries.clear();
    g_qsa_recording = true;
    const TokenId token = trunk.prefill(prompt);
    g_qsa_recording = false;
    if (dump_path != nullptr) { write_parity_dump(dump_path, state, prompt); }
    return token;
}

int run_plan_load(DeviceContext& device, const char* path, std::optional<TokenId> reference) {
    artifact::Reader reader(path);
    artifact::Binder binder(reader, 1);
    ninfer::EngineOptions options;
    options.device        = device.device;
    options.tp            = 1;
    options.max_context   = kKvCapacityTokens;
    options.max_concurrency = 1;
    options.artifact_path = path;
    auto load_plan = ninfer::targets::qwen3_8_flash_next::Package::plan_load(
        binder, options, detail::WeightsProfile::Nvfp4);

    const detail::BindingPlan& bindings = load_plan.bindings();
    const artifact::MaterializationPlan& plan = load_plan.materialization();

    std::vector<bool> placed(plan.object_count, false);
    for (const auto& placement : plan.device_objects) {
        placed[static_cast<std::size_t>(placement.object.index)] = true;
    }
    const bool expect_resident =
        ninfer::targets::qwen4_exp::detail::kTextActivationDtype == DType::BF16;
    for (const auto& object : bindings.objects) {
        if (!detail::is_fp16_control_object(bindings, object)) { continue; }
        const bool resident = placed[static_cast<std::size_t>(object.handle.index)];
        require(resident == expect_resident,
                "the fp16 control object " + object.name +
                    " residency does not match the build activation dtype");
    }

    const std::vector<bool> keep = reduced_object_mask(bindings);
    artifact::MaterializationPlan reduced = reduce_plan(plan, keep);
    reduced.device_capacity_bytes[0] +=
        detail::fp16_control_capacity_bytes(bindings, 1, kLoadedLayers)[0];
    const artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, reduced, device);
    (void)materialized;
    (void)reference;
    std::cout << "OK flash-next plan load: fp16 control residency and reduced placement\n";
    return 0;
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
    if (std::getenv("NINFER_FLASH_NEXT_PLAN_LOAD") != nullptr) {
        return run_plan_load(device, path, reference);
    }
    artifact::Reader reader(path);
    artifact::Binder binder(reader, 1);

    const detail::BindingPlan plan = detail::bind_artifact(binder, detail::WeightsProfile::Nvfp4, 1);
    place_reduced(plan, binder);
    binder.validate_unconsumed_matching();
    artifact::MaterializationPlan materialization = binder.finish();
    materialization.device_capacity_bytes[0] +=
        detail::fp16_control_capacity_bytes(plan, 1, kLoadedLayers)[0];
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, materialization, device);
    const std::filesystem::path artifact_path(path);
    const detail::LoadedModelData data(plan, materialized, kLoadedLayers, 1, artifact_path);

    const char* long_count_text = std::getenv("NINFER_FLASH_NEXT_LONG_PROMPT");
    if (long_count_text != nullptr) {
        const std::size_t count = static_cast<std::size_t>(std::stoul(long_count_text));
        require(count > 0, "the long prompt needs tokens");
        std::vector<TokenId> tokens(count);
        for (std::size_t index = 0; index < count; ++index) {
            tokens[index] = static_cast<TokenId>(1024 + (index * 7) % 200000);
        }
        const std::span<const TokenId> prompt{tokens.data(), tokens.size()};
        const TokenId token =
            run_long_prompt(device, data.view(0), prompt, std::getenv("NINFER_FLASH_NEXT_DUMP"));
        require(token >= 0 && token < static_cast<TokenId>(detail::TextConfig::vocab),
                "the long prompt greedy token is outside the vocabulary");
        if (reference.has_value()) {
            require(token == *reference, "the long prompt token " + std::to_string(token) +
                                             " does not match the reference token " +
                                             std::to_string(*reference));
        } else {
            std::cout << "SKIP: the long prompt reference token parity check\n";
        }
        std::cout << "OK flash-next long prompt: token " << token << " tokens " << count << "\n";
        return 0;
    }

    const auto key_bytes = static_cast<std::uint64_t>(detail::TextConfig::residual_width) *
                           detail::TextConfig::hidden * 2;
    const auto key_base = static_cast<const std::byte*>(data.view(0).ple.key.payload);
    const auto value_base = static_cast<const std::byte*>(data.view(0).ple.value.payload);
    require(value_base - key_base == static_cast<std::ptrdiff_t>(key_bytes),
            "the PLE value view does not start at the value rows");

    check_gdn_slot_mapping(device, data.view(0));
    const RunOutcome outcome = run_prompt(device, data.view(0));
    require(outcome.token >= 0 && outcome.token < static_cast<TokenId>(detail::TextConfig::vocab),
            "the greedy token is outside the vocabulary");
    const RunOutcome repeated = run_prompt(device, data.view(0));
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

    ShiftedPleRows shifted{qwen4_exp::PleRowSource{data.view(0).ple.rows.context,
                                                   data.view(0).ple.rows.fetch,
                                                   data.view(0).ple.rows.divisors}};
    auto shifted_view = data.view(0);
    shifted_view.ple.rows.context = &shifted;
    shifted_view.ple.rows.fetch = &fetch_shifted_rows;
    const RunOutcome shifted_outcome = run_prompt(device, shifted_view);
    require(shifted_outcome.token != outcome.token,
            "the PLE row source did not change the sampled token");

    auto zeroed_view = data.view(0);
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
