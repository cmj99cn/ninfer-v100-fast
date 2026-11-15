#include "core/arena.h"
#include "core/device.h"
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
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace artifact  = ninfer::artifact;
namespace detail    = ninfer::targets::qwen3_8_flash_next::detail;
namespace qwen4_exp = ninfer::targets::qwen4_exp;
using ninfer::DeviceArena;
using ninfer::ExecutionContext;
using ninfer::TokenId;
using Package = ninfer::targets::qwen3_8_flash_next::Package;

constexpr std::uint32_t kRanks        = 2;
constexpr std::uint32_t kKvCapacity   = 262144;
constexpr std::uint32_t kForwardTokens = 512;
constexpr std::uint32_t kDecodeTokens  = 6;
constexpr int kEventPool              = 4096;

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct MarkPool {
    std::array<cudaEvent_t, kEventPool> events{};
    std::array<std::int64_t, kEventPool> host{};
    std::array<const char*, kEventPool> tags{};
    int count = 0;
};

std::array<MarkPool, kRanks> g_pool;
bool g_enabled = false;

struct Aggregate {
    double device_ms = 0.0;
    double host_ms   = 0.0;
    std::uint64_t phases = 0;
};

void init_pools() {
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(static_cast<int>(rank)));
        for (int index = 0; index < kEventPool; ++index) {
            CUDA_CHECK(cudaEventCreateWithFlags(&g_pool[rank].events[index], cudaEventDefault));
        }
        CUDA_CHECK(cudaSetDevice(previous));
    }
}

void destroy_pools() {
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(static_cast<int>(rank)));
        for (int index = 0; index < kEventPool; ++index) {
            CUDA_CHECK(cudaEventDestroy(g_pool[rank].events[index]));
        }
        CUDA_CHECK(cudaSetDevice(previous));
    }
}

void begin_round() {
    for (auto& pool : g_pool) { pool.count = 0; }
}

void mark(const char* tag, int rank, cudaStream_t stream) {
    if (!g_enabled) { return; }
    MarkPool& pool = g_pool[static_cast<std::size_t>(rank)];
    if (pool.count >= kEventPool) { return; }
    const int slot = pool.count++;
    CUDA_CHECK(cudaEventRecord(pool.events[slot], stream));
    pool.host[slot] = now_ns();
    pool.tags[slot] = tag;
}

struct TokenReport {
    std::map<std::string, Aggregate> phases;
    std::array<std::map<std::string, Aggregate>, kRanks> per_rank;
    double anchored_device_ms = 0.0;
    double anchored_host_ms   = 0.0;
    double wall_ms            = 0.0;
    int marks[2]              = {0, 0};
};

TokenReport collect_token(double wall_ms) {
    TokenReport report;
    report.wall_ms = wall_ms;
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
        MarkPool& pool = g_pool[rank];
        report.marks[rank] = pool.count;
        if (pool.count < 2) { continue; }
        int previous_device = 0;
        CUDA_CHECK(cudaGetDevice(&previous_device));
        CUDA_CHECK(cudaSetDevice(static_cast<int>(rank)));
        const std::int64_t first_host = pool.host[0];
        float previous_ms = 0.0F;
        for (int index = 0; index < pool.count; ++index) {
            float cumulative_ms = 0.0F;
            if (index != 0) {
                CUDA_CHECK(cudaEventElapsedTime(&cumulative_ms, pool.events[0], pool.events[index]));
            }
            const double device_ms = static_cast<double>(cumulative_ms - previous_ms);
            previous_ms = cumulative_ms;
            const double host_ms = static_cast<double>(pool.host[index] -
                                                       (index == 0 ? pool.host[0] : pool.host[index - 1])) /
                                  1.0e6;
            Aggregate& aggregate = report.phases[pool.tags[index]];
            aggregate.device_ms += device_ms;
            aggregate.host_ms += host_ms;
            ++aggregate.phases;
            Aggregate& rank_aggregate = report.per_rank[rank][pool.tags[index]];
            rank_aggregate.device_ms += device_ms;
            rank_aggregate.host_ms += host_ms;
            ++rank_aggregate.phases;
        }
        float span_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&span_ms, pool.events[0], pool.events[pool.count - 1]));
        report.anchored_device_ms += span_ms;
        report.anchored_host_ms +=
            static_cast<double>(pool.host[pool.count - 1] - first_host) / 1.0e6;
        CUDA_CHECK(cudaSetDevice(previous_device));
    }
    return report;
}

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<TokenId> make_prompt(std::uint32_t prompt_tokens) {
    constexpr std::array<TokenId, 8> base{1024, 7, 98140, 33, 4097, 512, 100, 2048};
    std::vector<TokenId> prompt(prompt_tokens);
    for (std::uint32_t index = 0; index < prompt_tokens; ++index) {
        prompt[index] = base[index % base.size()];
    }
    return prompt;
}

} // namespace

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_phase_profile_runtime
#define NINFER_QWEN4EXP_DIAG_MARK(label) ::mark((label), (rank), stream_for(rank))
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

namespace {

namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_phase_profile_runtime;

void run(const char* path, std::uint32_t prompt_tokens, std::uint32_t warmup_tokens) {
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
    options.prefill_chunk    = kForwardTokens;
    options.expert_cache_mib = 0;

    auto load_plan = Package::plan_load(binder, options, detail::WeightsProfile::Nvfp4);
    const detail::BindingPlan& bindings = load_plan.bindings();
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, load_plan.materialization(), execution);
    detail::LoadedModelData data(bindings, materialized, detail::TextConfig::layers, 2,
                                 std::filesystem::path(path));
    data.allocate_expert_cache(options, execution);

    const auto recipe = rt::text_workspace_recipe(kForwardTokens, 1);
    std::unique_ptr<ninfer::ops::CollectiveComms> comms =
        std::make_unique<ninfer::ops::CollectiveComms>(execution);
    std::array<std::unique_ptr<rt::TextState>, ninfer::kMaximumDevices> states;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> activations;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> workspaces;
    std::array<rt::TextState*, ninfer::kMaximumDevices> state_ptrs{};
    std::array<const rt::Variant::ModelView*, ninfer::kMaximumDevices> view_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> activation_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> workspace_ptrs{};
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        int previous    = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
        states[slot] = std::make_unique<rt::TextState>(*execution.dev[slot], data.view(rank),
                                                       kKvCapacity, kForwardTokens, 2, rank);
        activations[slot] =
            std::make_unique<DeviceArena>(rt::text_activation_capacity_bytes(kForwardTokens));
        workspaces[slot] = std::make_unique<DeviceArena>(recipe.lane_capacity_bytes);
        state_ptrs[slot]      = states[slot].get();
        view_ptrs[slot]       = &data.view(rank);
        activation_ptrs[slot] = activations[slot].get();
        workspace_ptrs[slot]  = workspaces[slot].get();
        CUDA_CHECK(cudaSetDevice(previous));
    }
    rt::TextTrunk trunk(view_ptrs, state_ptrs, activation_ptrs, workspace_ptrs, execution, *comms);

    const std::vector<TokenId> prompt = make_prompt(prompt_tokens);
    const detail::CacheStats cold = data.expert_cache_stats(0);
    const std::int64_t prefill_begin = now_ns();
    TokenId token = 0;
    for (std::size_t offset = 0; offset < prompt.size(); offset += kForwardTokens) {
        const std::size_t count = std::min<std::size_t>(kForwardTokens, prompt.size() - offset);
        token = trunk.prefill(std::span<const TokenId>{prompt.data() + offset, count});
    }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { execution.dev[rank]->synchronize(); }
    const double prefill_ms = static_cast<double>(now_ns() - prefill_begin) / 1.0e6;
    for (std::uint32_t index = 0; index < warmup_tokens; ++index) { token = trunk.decode(token); }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { execution.dev[rank]->synchronize(); }
    const detail::CacheStats warm = data.expert_cache_stats(0);

    init_pools();
    g_enabled = true;

    std::vector<TokenReport> tokens;
    std::array<std::uint64_t, kDecodeTokens> hits{};
    std::array<std::uint64_t, kDecodeTokens> misses{};
    for (std::uint32_t index = 0; index < kDecodeTokens; ++index) {
        const detail::CacheStats before = data.expert_cache_stats(0);
        begin_round();
        const std::int64_t decode_begin = now_ns();
        token = trunk.decode(token);
        const double decode_ms = static_cast<double>(now_ns() - decode_begin) / 1.0e6;
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            execution.dev[rank]->synchronize();
        }
        const detail::CacheStats after = data.expert_cache_stats(0);
        hits[index]   = after.hits - before.hits;
        misses[index] = after.miss_count - before.miss_count;
        tokens.push_back(collect_token(decode_ms));
    }
    g_enabled = false;

    std::cout << "{\n";
    std::cout << "  \"artifact\": \"" << path << "\",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"decode_tokens\": " << kDecodeTokens << ",\n";
    std::cout << "  \"warmup_tokens\": " << warmup_tokens << ",\n";
    std::cout << "  \"prefill_ms\": " << prefill_ms << ",\n";
    std::cout << "  \"cold_misses\": " << (warm.miss_count - cold.miss_count) << ",\n";
    std::cout << "  \"tokens\": [\n";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const TokenReport& report = tokens[index];
        std::cout << "    {\"index\": " << index << ", \"wall_ms\": " << report.wall_ms
                  << ", \"anchored_device_ms\": " << report.anchored_device_ms
                  << ", \"anchored_host_ms\": " << report.anchored_host_ms
                  << ", \"ranks\": [" << report.marks[0] << ", " << report.marks[1] << "]"
                  << ", \"hits\": " << hits[index] << ", \"misses\": " << misses[index]
                  << ", \"phases\": {";
        bool first = true;
        for (const auto& [tag, aggregate] : report.phases) {
            if (!first) { std::cout << ", "; }
            first = false;
            std::cout << "\"" << tag << "\": {\"dev_ms\": " << aggregate.device_ms
                      << ", \"host_ms\": " << aggregate.host_ms
                      << ", \"n\": " << aggregate.phases << "}";
        }
        std::cout << "}, \"per_rank\": [";
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            if (rank != 0) { std::cout << ", "; }
            std::cout << "{";
            bool rank_first = true;
            for (const auto& [tag, aggregate] : report.per_rank[rank]) {
                if (!rank_first) { std::cout << ", "; }
                rank_first = false;
                std::cout << "\"" << tag << "\": {\"dev_ms\": " << aggregate.device_ms
                          << ", \"host_ms\": " << aggregate.host_ms
                          << ", \"n\": " << aggregate.phases << "}";
            }
            std::cout << "}";
        }
        std::cout << "]}\n" << (index + 1 == tokens.size() ? "" : ",");
    }
    std::cout << "  ]\n}\n";

    destroy_pools();
}

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (path == nullptr || path[0] == '\0') {
        std::cerr << "flash_next_phase_profile: an artifact path is required\n";
        return 1;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cerr << "flash_next_phase_profile: two devices are required\n";
        return 1;
    }
    std::uint32_t prompt_tokens = kForwardTokens;
    if (argc > 2) { prompt_tokens = static_cast<std::uint32_t>(std::stoul(argv[2])); }
    std::uint32_t warmup_tokens = 0;
    if (argc > 3) { warmup_tokens = static_cast<std::uint32_t>(std::stoul(argv[3])); }
    try {
        run(path, prompt_tokens, warmup_tokens);
    } catch (const std::exception& error) {
        std::cerr << "flash_next_phase_profile: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
