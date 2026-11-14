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

#include <cuda_fp16.h>
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::DeviceArena;
using ninfer::ExecutionContext;
using ninfer::MemorySummary;
using ninfer::Tensor;
using ninfer::TokenId;

namespace artifact = ninfer::artifact;
namespace detail    = ninfer::targets::qwen3_8_flash_next::detail;
namespace qwen4_exp = ninfer::targets::qwen4_exp;
using Package       = ninfer::targets::qwen3_8_flash_next::Package;

constexpr std::size_t kPromptTokens    = 8;
constexpr std::uint32_t kKvCapacity    = 262144;
constexpr std::uint32_t kSmallCacheMib = 700;
constexpr std::uint64_t kDeviceBudget  = 32ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kHostBudget    = static_cast<std::uint64_t>(94.0e9 * 0.8);

constexpr std::array<TokenId, kPromptTokens> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};

std::vector<std::uint16_t> g_logits;
std::int32_t g_logits_vocab  = 0;
std::int32_t g_logits_tokens = 0;
bool g_capture               = false;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint64_t peak_host_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) { return 0; }
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
}

std::size_t device_used_bytes(int device) {
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    CUDA_CHECK(cudaSetDevice(device));
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    CUDA_CHECK(cudaSetDevice(previous));
    return total_bytes - free_bytes;
}

std::vector<std::byte> read_artifact_row(const artifact::Reader& reader, const std::string& object,
                                         std::uint64_t offset, std::uint64_t bytes) {
    const auto* descriptor = reader.find(object);
    if (descriptor == nullptr) { throw std::runtime_error("the artifact has no object " + object); }
    const std::uint64_t absolute =
        reader.payload_offset() + artifact::object_offset(*descriptor) + offset;
    std::vector<std::byte> out(static_cast<std::size_t>(bytes));
    std::size_t copied = 0;
    for (const auto& view : reader.payload_views(absolute, bytes)) {
        std::copy(view.data.begin(), view.data.end(),
                  out.begin() + static_cast<std::ptrdiff_t>(copied));
        copied += view.data.size();
    }
    require(copied == bytes, "the artifact row was read short");
    return out;
}

} // namespace

void flash_next_capture_logits(const Tensor& logits, cudaStream_t stream, int rank, int) {
    if (!g_capture || rank != 0) { return; }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    g_logits_vocab  = logits.ne[0];
    g_logits_tokens = logits.ne[1];
    g_logits.resize(logits.bytes() / 2);
    CUDA_CHECK(cudaMemcpy(g_logits.data(), logits.data, logits.bytes(), cudaMemcpyDeviceToHost));
}

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_full_model_test_runtime
#define NINFER_QWEN4EXP_LAST_STAGE(sample, logits) \
    ::flash_next_capture_logits((logits), stream_for(rank), rank, T)
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

namespace {

namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_full_model_test_runtime;

struct RunResult {
    TokenId token_first  = 0;
    TokenId token_second = 0;
    std::vector<std::uint16_t> logits_first;
    std::vector<std::uint16_t> logits_second;
    std::vector<std::uint16_t> logits_third;
    MemorySummary memory;
    detail::CacheStats stats_first;
    detail::CacheStats stats_second;
    detail::CacheStats stats_third;
    std::uint64_t cache_bytes   = 0;
    std::uint64_t weights_bytes = 0;
    std::uint64_t host_bytes    = 0;
    std::uint64_t measured_used[2] = {0, 0};
    std::uint64_t peak_host = 0;
};

RunResult run_variant(const char* path, std::uint32_t expert_cache_mib) {
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
    options.expert_cache_mib = expert_cache_mib;

    auto load_plan = Package::plan_load(binder, options, detail::WeightsProfile::Nvfp4);
    const detail::BindingPlan& bindings = load_plan.bindings();
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, load_plan.materialization(), execution);
    detail::LoadedModelData data(bindings, materialized, detail::TextConfig::layers, 2,
                                 std::filesystem::path(path));
    data.allocate_expert_cache(options, execution);

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

    auto run_prefill = [&](std::vector<std::uint16_t>& logits) {
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            int previous    = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
            states[slot]->reset();
            activations[slot]->reset();
            workspaces[slot]->reset();
            CUDA_CHECK(cudaSetDevice(previous));
        }
        g_logits.clear();
        g_capture = true;
        const TokenId token = trunk.prefill(std::span<const TokenId>{kPrompt.data(), kPrompt.size()});
        g_capture = false;
        for (int rank = 0; rank < 2; ++rank) { execution.dev[rank]->synchronize(); }
        logits = g_logits;
        return token;
    };

    RunResult result;
    result.token_first = run_prefill(result.logits_first);
    result.stats_first = data.expert_cache_stats(0);
    result.token_second = run_prefill(result.logits_second);
    result.stats_second = data.expert_cache_stats(0);
    (void)run_prefill(result.logits_third);
    result.stats_third = data.expert_cache_stats(0);

    auto planner = Package::make_sequence_planner(execution.primary(), options,
                                                  detail::WeightsProfile::Nvfp4);
    auto sequence_plan =
        std::move(planner).finalize(planner.capacity_curve().minimum_main_page_groups);
    std::array<const detail::Variant::ModelView*, ninfer::kMaximumDevices> models{};
    models[0] = &data.view(0);
    models[1] = &data.view(1);
    auto program = qwen4_exp::create_program<detail::Variant>(models, detail::WeightsProfile::Nvfp4,
                                                              std::move(sequence_plan), execution,
                                                              options);
    result.memory      = program->memory_summary();
    result.cache_bytes = data.expert_cache_bytes(0);
    result.host_bytes  = data.expert_host_bytes(0);
    result.weights_bytes = data.view(0).resident->weights_capacity_bytes;
    result.peak_host   = peak_host_bytes();
    for (int rank = 0; rank < 2; ++rank) {
        result.measured_used[rank] = device_used_bytes(execution.dev[rank]->device);
    }

    const ninfer::Weight& embedding = data.view(0).embedding;
    require(embedding.qtype == ninfer::QType::BF16_CTRL &&
                embedding.n == detail::TextConfig::vocab &&
                embedding.k == detail::TextConfig::hidden,
            "the token embedding view is not the replicated BF16 table");
    const std::uint32_t token = 248319;
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(detail::TextConfig::hidden) * 2;
    std::vector<std::byte> device_row(row_bytes);
    CUDA_CHECK(cudaMemcpy(device_row.data(),
                          static_cast<const std::byte*>(embedding.qdata) + token * row_bytes,
                          row_bytes, cudaMemcpyDeviceToHost));
    const auto& bound  = bindings.token_embedding.parts.front();
    const auto& object = bindings.objects.at(bound.object);
    const std::vector<std::byte> artifact_row =
        read_artifact_row(reader, object.name, static_cast<std::uint64_t>(token) * row_bytes,
                          row_bytes);
    require(device_row == artifact_row,
            "the device token embedding row differs from the artifact row");

    return result;
}

void check_variant(const RunResult& result, const std::string& label) {
    require(result.logits_first.size() == result.logits_second.size(),
            label + ": the two prefill runs produced different logits extents");
    require(result.logits_second == result.logits_third,
            label + ": the last two prefill runs are not bit-identical");
    require(result.logits_first == result.logits_second,
            label + ": the cold and warm prefill runs are not bit-identical");
    require(result.memory.weights.capacity_bytes != 0, label + ": the weights partition is empty");
    require(result.memory.expert_cache_bytes == result.cache_bytes,
            label + ": memory_summary does not report the expert cache bytes");
    require(result.memory.sequence.capacity_bytes + result.memory.workspace.capacity_bytes +
                    result.memory.weights.capacity_bytes + result.memory.expert_cache_bytes <=
                kDeviceBudget,
            label + ": the per-device plan exceeds 32 GiB");
    for (int rank = 0; rank < 2; ++rank) {
        require(result.measured_used[rank] <= kDeviceBudget,
                label + ": device " + std::to_string(rank) + " uses more than 32 GiB");
    }
    require(result.peak_host <= kHostBudget,
            label + ": the process peak host RSS exceeds 80% of 94 GB");
    for (std::size_t index = 0; index < result.logits_first.size(); ++index) {
        __half value;
        std::memcpy(&value, &result.logits_first[index], sizeof(value));
        require(__half2float(value) == __half2float(value),
                label + ": the prefill logits are not finite");
    }
    const std::uint64_t lookups =
        result.stats_third.lookups - result.stats_second.lookups;
    const std::uint64_t hits = result.stats_third.hits - result.stats_second.hits;
    std::cout << "flash-next full model: " << label << " weights " << result.weights_bytes
              << " expert-cache " << result.cache_bytes << " pairs "
              << result.stats_second.residency << " second-run lookups " << lookups << " hits "
              << hits << " miss-bytes "
              << (result.stats_third.miss_count - result.stats_second.miss_count) *
                     detail::kPairBytes
              << " host " << result.host_bytes << " rss " << result.peak_host << " device-used "
              << result.measured_used[0] << " " << result.measured_used[1] << '\n';
}

void run(const char* path) {
    const RunResult small = run_variant(path, kSmallCacheMib);
    const RunResult large = run_variant(path, 0);

    check_variant(small, "small-cache");
    check_variant(large, "auto-cache");

    const std::uint64_t large_misses =
        large.stats_third.miss_count - large.stats_second.miss_count;
    const std::uint64_t large_hits = large.stats_third.hits - large.stats_second.hits;
    const std::uint64_t small_misses =
        small.stats_third.miss_count - small.stats_second.miss_count;
    require(large_misses == 0,
            "the auto cache still missed on the warm run: " + std::to_string(large_misses));
    require(large_hits != 0, "the auto cache recorded no warm hit");
    require(small_misses != 0,
            "the small cache recorded no miss on the second run, so the miss path was not exercised");
    require(small.logits_third == large.logits_third,
            "the small and auto caches produced different warm logits");
    require(small.cache_bytes < large.cache_bytes,
            "the small cache is not smaller than the auto cache");
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
    std::cout << "OK flash-next full model: 48 layers over two ranks, deterministic, cached\n";
    return 0;
}
