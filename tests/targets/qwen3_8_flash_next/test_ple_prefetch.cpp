#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_runtime.h"
#include "targets/qwen4_exp/impl/runtime/activation_dtype.h"

#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen4_exp/runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::DeviceArena;
using ninfer::ExecutionContext;
using ninfer::Tensor;
using ninfer::TokenId;

namespace artifact = ninfer::artifact;
namespace detail    = ninfer::targets::qwen3_8_flash_next::detail;
namespace qwen4_exp = ninfer::targets::qwen4_exp;
using Package       = ninfer::targets::qwen3_8_flash_next::Package;

constexpr std::size_t kPromptTokens  = 8;
constexpr std::uint32_t kKvCapacity  = 64;
constexpr std::uint32_t kExpertCacheMib = 675;
constexpr std::array<TokenId, kPromptTokens> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::uint64_t thread_hash() {
    return static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

struct PrefetchProbe {
    std::atomic<std::uint32_t> launched{0};
    std::atomic<std::uint32_t> awaited{0};
    std::atomic<std::uint32_t> fetch_begins{0};
    std::atomic<std::uint32_t> fetch_ends{0};
    std::atomic<std::uint32_t> rows{0};
    std::atomic<std::int32_t> active{0};
    std::atomic<std::uint64_t> fetch_thread{0};
    std::atomic<std::int64_t> launch_ns{0};
    std::atomic<std::int64_t> fetch_begin_ns{0};
    std::atomic<std::int64_t> fetch_end_ns{0};
    std::atomic<std::int64_t> await_ns{0};
    std::atomic<std::uint32_t> expert_layer0_windows{0};
    std::atomic<std::int64_t> expert_begin_ns{0};
    std::atomic<std::int64_t> expert_end_ns{0};
    std::atomic<std::int32_t> phase_at_expert_begin{-1};
    std::atomic<std::int32_t> phase_at_expert_end{-1};
    std::atomic<std::uint64_t> host_reads_begin{0};
    std::atomic<std::uint64_t> host_reads_end{0};

    std::uint64_t main_thread = 0;

    [[nodiscard]] std::int32_t outstanding() const {
        return launched.load() > awaited.load() ? 1 : 0;
    }

    void reset() {
        launched = 0;
        awaited = 0;
        fetch_begins = 0;
        fetch_ends = 0;
        rows = 0;
        active = 0;
        fetch_thread = 0;
        launch_ns = 0;
        fetch_begin_ns = 0;
        fetch_end_ns = 0;
        await_ns = 0;
        expert_layer0_windows = 0;
        expert_begin_ns = 0;
        expert_end_ns = 0;
        phase_at_expert_begin = -1;
        phase_at_expert_end = -1;
        host_reads_begin = 0;
        host_reads_end = 0;
    }
};

PrefetchProbe g_probe;

void probe_launch() {
    const std::int64_t stamp = now_ns();
    g_probe.launch_ns = stamp;
    ++g_probe.launched;
}

void probe_fetch_begin(std::size_t count) {
    const std::int64_t stamp = now_ns();
    g_probe.fetch_begin_ns = stamp;
    g_probe.fetch_thread = thread_hash();
    g_probe.rows = static_cast<std::uint32_t>(count);
    ++g_probe.fetch_begins;
    ++g_probe.active;
}

void probe_fetch_end() {
    const std::int64_t stamp = now_ns();
    g_probe.fetch_end_ns = stamp;
    --g_probe.active;
    ++g_probe.fetch_ends;
}

void probe_await() {
    const std::int64_t stamp = now_ns();
    g_probe.await_ns = stamp;
    ++g_probe.awaited;
}

void probe_expert_begin(std::size_t layer, int rank, const detail::MoeWeights& weights) {
    if (rank != 0 || layer != 0) { return; }
    g_probe.expert_begin_ns = now_ns();
    g_probe.phase_at_expert_begin = g_probe.outstanding();
    g_probe.host_reads_begin =
        weights.cache != nullptr
            ? weights.cache->cache().source().reads_started()
            : 0;
    ++g_probe.expert_layer0_windows;
}

void probe_expert_end(std::size_t layer, int rank, const detail::MoeWeights& weights) {
    if (rank != 0 || layer != 0) { return; }
    g_probe.expert_end_ns = now_ns();
    g_probe.phase_at_expert_end = g_probe.outstanding();
    g_probe.host_reads_end =
        weights.cache != nullptr
            ? weights.cache->cache().source().reads_finished()
            : 0;
}

} // namespace

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_ple_prefetch_test_runtime
#define NINFER_QWEN4EXP_PLE_PREFETCH_LAUNCH() ::probe_launch()
#define NINFER_QWEN4EXP_PLE_FETCH_BEGIN(count) ::probe_fetch_begin((count))
#define NINFER_QWEN4EXP_PLE_FETCH_END() ::probe_fetch_end()
#define NINFER_QWEN4EXP_PLE_PREFETCH_AWAIT() ::probe_await()
#define NINFER_QWEN4EXP_EXPERT_READ_BEGIN(layer) \
    ::probe_expert_begin((layer), (rank), (models_[slot]->layers[layer].post_mixer))
#define NINFER_QWEN4EXP_EXPERT_READ_END(layer) \
    ::probe_expert_end((layer), (rank), (models_[slot]->layers[layer].post_mixer))
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

namespace {

namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_ple_prefetch_test_runtime;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void run(const char* path) {
    ExecutionContext execution({0, 1});
    artifact::Reader reader(path);
    artifact::Binder binder(reader, 2);

    ninfer::EngineOptions options;
    options.artifact_path    = path;
    options.device           = 0;
    options.tp               = 2;
    options.devices          = {0, 1};
    options.max_context      = kKvCapacity;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
    options.max_concurrency  = 1;
    options.prefill_chunk    = 64;
    options.expert_cache_mib = kExpertCacheMib;

    auto load_plan = Package::plan_load(binder, options, detail::WeightsProfile::Nvfp4);
    const detail::BindingPlan& bindings = load_plan.bindings();
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, load_plan.materialization(), execution);
    detail::LoadedModelData data(bindings, materialized, detail::TextConfig::layers, 2,
                                 std::filesystem::path(path));
    data.allocate_expert_cache(options, execution);
    require(data.expert_cache_stats(0).residency == 0,
            "the prefetch test needs a cold expert cache at its first round");

    const auto recipe = rt::text_workspace_recipe(kPromptTokens, 1);
    std::unique_ptr<ninfer::ops::CollectiveComms> comms =
        std::make_unique<ninfer::ops::CollectiveComms>(execution);
    std::array<std::unique_ptr<rt::TextState>, ninfer::kMaximumDevices> states;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> activations;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> workspaces;
    std::array<rt::TextState*, ninfer::kMaximumDevices> state_ptrs{};
    std::array<const rt::Variant::ModelView*, ninfer::kMaximumDevices> view_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> activation_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> workspace_ptrs{};
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        int previous    = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
        states[slot] = std::make_unique<rt::TextState>(*execution.dev[slot], data.view(rank),
                                                       kKvCapacity, kPromptTokens, 2, rank);
        activations[slot] =
            std::make_unique<DeviceArena>(rt::text_activation_capacity_bytes(kPromptTokens));
        workspaces[slot] = std::make_unique<DeviceArena>(recipe.lane_capacity_bytes);
        state_ptrs[slot]      = states[slot].get();
        view_ptrs[slot]       = &data.view(rank);
        activation_ptrs[slot] = activations[slot].get();
        workspace_ptrs[slot]  = workspaces[slot].get();
        CUDA_CHECK(cudaSetDevice(previous));
    }
    rt::TextTrunk trunk(view_ptrs, state_ptrs, activation_ptrs, workspace_ptrs, execution, *comms);

    g_probe.main_thread = thread_hash();
    const TokenId first =
        trunk.prefill(std::span<const TokenId>{kPrompt.data(), kPrompt.size()});
    for (int rank = 0; rank < 2; ++rank) { execution.dev[rank]->synchronize(); }

    const detail::CacheStats before = data.expert_cache_stats(0);
    g_probe.reset();
    const TokenId token = trunk.decode(first);
    for (int rank = 0; rank < 2; ++rank) { execution.dev[rank]->synchronize(); }
    const detail::CacheStats after = data.expert_cache_stats(0);
    require(token >= 0 && token < static_cast<TokenId>(detail::TextConfig::vocab),
            "the greedy token is outside the vocabulary");

    constexpr std::uint32_t kRanks = 2;
    require(g_probe.launched.load() == kRanks,
            "the decode round launched " + std::to_string(g_probe.launched.load()) +
                " PLE prefetches; a round launches one per rank");
    require(g_probe.awaited.load() == kRanks,
            "the decode round awaited " + std::to_string(g_probe.awaited.load()) +
                " PLE prefetches; a round awaits one per rank");
    require(g_probe.fetch_begins.load() == kRanks && g_probe.fetch_ends.load() == kRanks,
            "the decode round ran " + std::to_string(g_probe.fetch_begins.load()) +
                " PLE row fetches; it runs one per rank");
    require(g_probe.active.load() == 0, "the PLE fetch was still active after the await");
    require(g_probe.rows.load() == static_cast<std::uint32_t>(detail::TextConfig::ple_heads),
            "the decode round fetched " + std::to_string(g_probe.rows.load()) +
                " PLE rows; a token addresses " +
                std::to_string(detail::TextConfig::ple_heads));
    require(g_probe.fetch_thread.load() != 0 && g_probe.fetch_thread.load() != g_probe.main_thread,
            "the PLE row fetch ran on the request thread, so it was not overlapped");

    require(g_probe.launch_ns.load() < g_probe.fetch_begin_ns.load(),
            "the PLE prefetch was launched after its own fetch began");
    require(g_probe.fetch_end_ns.load() <= g_probe.await_ns.load(),
            "the PLE fetch completed after the consumption await");
    require(g_probe.launch_ns.load() < g_probe.expert_end_ns.load(),
            "the PLE prefetch was launched after the first GDN layer's expert reads ended");
    require(g_probe.fetch_end_ns.load() <= g_probe.expert_begin_ns.load(),
            "the PLE worker fetch did not finish before the layer-0 expert host reads began");

    require(g_probe.expert_layer0_windows.load() == 1,
            "the decode round recorded " +
                std::to_string(g_probe.expert_layer0_windows.load()) +
                " layer-0 expert read windows; it records one");
    require(g_probe.phase_at_expert_begin.load() == 1 &&
                g_probe.phase_at_expert_end.load() == 1,
            "the PLE prefetch was awaited inside the layer-0 expert read window");
    require(g_probe.launch_ns.load() < g_probe.expert_begin_ns.load() &&
                g_probe.expert_end_ns.load() < g_probe.await_ns.load(),
            "the PLE prefetch lifecycle does not contain the layer-0 expert reads");
    require(after.miss_count > before.miss_count,
            "the decode round did not exercise the expert miss path");
    require(g_probe.host_reads_end.load() > g_probe.host_reads_begin.load(),
            "the layer-0 host pair source recorded no completed miss reads inside the window");

    const std::int64_t base = g_probe.launch_ns.load();
    std::cout << "flash-next PLE prefetch: lifecycle "
              << (g_probe.await_ns.load() - base) << " ns contains layer-0 expert window "
              << (g_probe.expert_end_ns.load() - g_probe.expert_begin_ns.load())
              << " ns; worker fetch " << (g_probe.fetch_end_ns.load() - g_probe.fetch_begin_ns.load())
              << " ns ended " << (g_probe.expert_begin_ns.load() - g_probe.fetch_end_ns.load())
              << " ns before the expert host reads began\n";
    std::cout << "OK flash-next PLE prefetch: fetch " << g_probe.rows.load()
              << " rows off the request thread, completed before the layer-0 expert host reads; "
              << "lifecycle contains " << (g_probe.host_reads_end.load() - g_probe.host_reads_begin.load())
              << " completed miss reads\n";
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (path == nullptr || path[0] == '\0') {
        std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }
    try {
        run(path);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
