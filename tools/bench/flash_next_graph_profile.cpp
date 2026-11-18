#include "core/arena.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "core/tensor.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include "targets/qwen4_exp/impl/runtime/activation_dtype.h"

#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen4_exp/runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace artifact = ninfer::artifact;
namespace detail = ninfer::targets::qwen3_8_flash_next::detail;
using ninfer::DecodeGraphDefinition;
using ninfer::DecodeGraphExecutable;
using ninfer::DeviceArena;
using ninfer::ExecutionContext;
using ninfer::PinnedHostBuffer;
using ninfer::TokenId;
using Package = ninfer::targets::qwen3_8_flash_next::Package;

constexpr std::uint32_t kRanks = 2;
constexpr std::uint32_t kKvCapacity = 262144;
constexpr std::uint32_t kForwardTokens = 512;
constexpr std::uint32_t kWarmupTokens = 192;
constexpr std::uint32_t kReplayTokens = 6;

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct GraphInputBlock {
    std::vector<TokenId> ids;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> scalars;
    std::vector<std::int32_t> winner;
};

std::array<GraphInputBlock, kRanks> g_graph_inputs;
std::array<std::unique_ptr<PinnedHostBuffer>, kRanks> g_ids_host;
std::array<std::unique_ptr<PinnedHostBuffer>, kRanks> g_positions_host;
std::array<std::unique_ptr<PinnedHostBuffer>, kRanks> g_scalars_host;
std::array<std::unique_ptr<PinnedHostBuffer>, kRanks> g_winner_host;

inline std::int32_t* graph_scalar(int rank, const char* label) {
    auto& scalars = g_graph_inputs[static_cast<std::size_t>(rank)].scalars;
    if (std::strcmp(label, "ple_offset") == 0) return scalars.data() + 0;
    if (std::strcmp(label, "ple_count") == 0) return scalars.data() + 1;
    if (std::strcmp(label, "live") == 0) return scalars.data() + 2;
    if (std::strcmp(label, "row") == 0) return scalars.data() + 3;
    throw std::runtime_error(std::string("unknown graph scalar label: ") + label);
}

inline void fill_graph_inputs(int rank, TokenId token, std::uint32_t position) {
    auto& block = g_graph_inputs[static_cast<std::size_t>(rank)];
    block.ids.assign(1, token);
    block.positions.assign(1, static_cast<std::int32_t>(position));
    if (block.scalars.empty()) block.scalars.assign(4, 0);
    block.scalars[0] = 0;
    block.scalars[1] = 1;
    block.scalars[2] = 1;
    block.scalars[3] = 0;
    block.winner.assign(1, 0);
    std::memcpy(g_ids_host[static_cast<std::size_t>(rank)]->data(), block.ids.data(), 4);
    std::memcpy(g_positions_host[static_cast<std::size_t>(rank)]->data(), block.positions.data(), 4);
    std::memcpy(g_scalars_host[static_cast<std::size_t>(rank)]->data(), block.scalars.data(),
                block.scalars.size() * 4);
}

std::vector<TokenId> make_prompt(std::uint32_t prompt_tokens) {
    constexpr std::array<TokenId, 8> base{1024, 7, 98140, 33, 4097, 512, 100, 2048};
    std::vector<TokenId> prompt(prompt_tokens);
    for (std::uint32_t index = 0; index < prompt_tokens; ++index) {
        prompt[index] = base[index % base.size()];
    }
    return prompt;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_graph_capture_runtime
#define NINFER_QWEN4EXP_GRAPH_IDS_SRC(rank, tokens) (g_ids_host[rank]->data())
#define NINFER_QWEN4EXP_GRAPH_POSITIONS_SRC(rank, host_positions) (g_positions_host[rank]->data())
#define NINFER_QWEN4EXP_GRAPH_SCALARS_SRC(rank, value, label) (graph_scalar((rank), (label)))
#define NINFER_QWEN4EXP_GRAPH_PREPARE_PLE(rank, state, tokens)
#define NINFER_QWEN4EXP_GRAPH_PLE_AWAIT(rank, state)
#define NINFER_QWEN4EXP_GRAPH_ROUTED_FINISH(weights, stream)
#define NINFER_QWEN4EXP_GRAPH_FINISH_SYNC()
#define NINFER_QWEN4EXP_GRAPH_WINNER_PTR() (g_winner_host[0]->data())
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

namespace {

namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_graph_capture_runtime;

struct Trunks {
    std::array<std::unique_ptr<rt::TextState>, ninfer::kMaximumDevices> states;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> activations;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> workspaces;
    std::array<rt::TextState*, ninfer::kMaximumDevices> state_ptrs{};
    std::array<const rt::Variant::ModelView*, ninfer::kMaximumDevices> view_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> activation_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> workspace_ptrs{};
    std::unique_ptr<rt::TextTrunk> trunk;
};

void run(const char* path) {
    ExecutionContext execution({0, 1});
    artifact::Reader reader(path);
    artifact::Binder binder(reader, 2);

    ninfer::EngineOptions options;
    options.artifact_path = path;
    options.device = 0;
    options.tp = 2;
    options.devices = {0, 1};
    options.max_context = kKvCapacity;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
    options.max_concurrency = 1;
    options.prefill_chunk = kForwardTokens;
    options.expert_cache_mib = 20000;

    auto load_plan = Package::plan_load(binder, options, detail::WeightsProfile::Nvfp4);
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, load_plan.materialization(), execution);
    detail::LoadedModelData data(load_plan.bindings(), materialized, detail::TextConfig::layers, 2,
                                 std::filesystem::path(path));
    data.allocate_expert_cache(options, execution);

    const auto recipe = rt::text_workspace_recipe(kForwardTokens, 1);
    auto comms = std::make_unique<ninfer::ops::CollectiveComms>(execution);

    Trunks trunks;
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
        trunks.states[slot] = std::make_unique<rt::TextState>(
            *execution.dev[slot], data.view(rank), kKvCapacity, kForwardTokens, 2, rank);
        trunks.activations[slot] =
            std::make_unique<DeviceArena>(rt::text_activation_capacity_bytes(kForwardTokens));
        trunks.workspaces[slot] = std::make_unique<DeviceArena>(recipe.lane_capacity_bytes);
        trunks.state_ptrs[slot] = trunks.states[slot].get();
        trunks.view_ptrs[slot] = &data.view(rank);
        trunks.activation_ptrs[slot] = trunks.activations[slot].get();
        trunks.workspace_ptrs[slot] = trunks.workspaces[slot].get();
        g_ids_host[slot] = std::make_unique<PinnedHostBuffer>(64);
        g_positions_host[slot] = std::make_unique<PinnedHostBuffer>(64);
        g_scalars_host[slot] = std::make_unique<PinnedHostBuffer>(64);
        g_winner_host[slot] = std::make_unique<PinnedHostBuffer>(64);
        CUDA_CHECK(cudaSetDevice(previous));
    }
    trunks.trunk = std::make_unique<rt::TextTrunk>(trunks.view_ptrs, trunks.state_ptrs,
                                                    trunks.activation_ptrs, trunks.workspace_ptrs,
                                                    execution, *comms);

    const std::vector<TokenId> prompt = make_prompt(512);
    TokenId token = 0;
    for (std::size_t offset = 0; offset < prompt.size(); offset += kForwardTokens) {
        const std::size_t count = std::min<std::size_t>(kForwardTokens, prompt.size() - offset);
        token = trunks.trunk->prefill(std::span<const TokenId>{prompt.data() + offset, count});
    }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { execution.dev[rank]->synchronize(); }
    for (std::uint32_t index = 0; index < kWarmupTokens; ++index) { token = trunks.trunk->decode(token); }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { execution.dev[rank]->synchronize(); }
    const auto warm = data.expert_cache_stats(0);
    std::cout << "{\"warm_hits\":" << warm.hits << ",\"warm_misses\":" << warm.miss_count << ",\n";

    std::array<int, kRanks> devices{};
    std::array<cudaStream_t, kRanks> streams{};
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
        devices[rank] = execution.dev[rank]->device;
        streams[rank] = execution.dev[rank]->stream;
    }
    DecodeGraphDefinition definition;
    const std::uint32_t position = trunks.states[0]->next_position();
    try {
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) { fill_graph_inputs(rank, token, position); }
        trunks.activations[0]->reset();
        trunks.workspaces[0]->reset();
        for (std::uint32_t rank = 1; rank < kRanks; ++rank) {
            trunks.activations[rank]->reset();
            trunks.workspaces[rank]->reset();
        }
        definition.capture(
            std::span<const int>(devices.data(), kRanks),
            std::span<cudaStream_t>(streams.data(), kRanks),
            [&] { trunks.trunk->decode(token); });
        std::cout << "\"capture_ok\":true,\"node_count\":" << definition.node_count() << ",\n";
    } catch (const std::exception& error) {
        std::cout << "\"capture_ok\":false,\"error\":\"" << error.what() << "\"}\n";
        return;
    }
    DecodeGraphExecutable executable;
    executable.instantiate(definition, std::span<const int>(devices.data(), kRanks),
                           std::span<cudaStream_t>(streams.data(), kRanks));
    executable.upload();

    std::vector<double> walls;
    std::vector<TokenId> replay_tokens;
    for (std::uint32_t index = 0; index < kReplayTokens; ++index) {
        const std::uint32_t pos = trunks.states[0]->next_position();
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) { fill_graph_inputs(rank, token, pos); }
        const std::int64_t begin = now_ns();
        executable.launch();
        execution.dev[0]->synchronize();
        const double wall = static_cast<double>(now_ns() - begin) / 1.0e6;
        TokenId result = static_cast<TokenId>(
            *static_cast<const std::int32_t*>(g_winner_host[0]->data()));
        walls.push_back(wall);
        replay_tokens.push_back(result);
        token = result;
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            trunks.states[rank]->advance_position(1);
            trunks.states[rank]->advance_ple_tail(token);
        }
    }
    double sum = 0.0;
    for (double wall : walls) { sum += wall; }
    std::cout << "\"replay_wall_ms_avg\":" << (sum / walls.size()) << ",\n";
    std::cout << "\"replay_tokens\":[";
    for (std::size_t i = 0; i < replay_tokens.size(); ++i) {
        std::cout << (i ? "," : "") << replay_tokens[i];
    }
    std::cout << "]}\n";
}

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (path == nullptr || path[0] == '\0') {
        std::cerr << "flash_next_graph_profile: an artifact path is required\n";
        return 1;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cerr << "flash_next_graph_profile: two devices are required\n";
        return 1;
    }
    try {
        run(path);
    } catch (const std::exception& error) {
        std::cerr << "flash_next_graph_profile: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
