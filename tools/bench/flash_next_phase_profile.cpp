#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"
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
namespace flash     = ninfer::targets::qwen3_8_flash_next::detail;
namespace qwen4_exp = ninfer::targets::qwen4_exp;
using ninfer::DeviceArena;
using ninfer::DeviceBuffer;
using ninfer::ExecutionContext;
using ninfer::Tensor;
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

struct PhaseAccum {
    double draft_ms     = 0.0;
    double verify_ms    = 0.0;
    double baseline_ms  = 0.0;
    double action_ms    = 0.0;
    double span_ms      = 0.0;
    double accepting_ms = 0.0;
    double commit_ms    = 0.0;
    double sync_ms      = 0.0;
    double rest_ms      = 0.0;
    double total_ms     = 0.0;
    std::uint64_t rounds   = 0;
    std::uint64_t drafted  = 0;
    std::uint64_t accepted = 0;
};

void add_ms(double& destination, std::int64_t begin, std::int64_t end) {
    destination += static_cast<double>(end - begin) / 1.0e6;
}

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
    if (const char* cache_mib = std::getenv("NINFER_FLASH_PHASE_CACHE_MIB")) {
        options.expert_cache_mib = static_cast<std::uint64_t>(std::stoull(cache_mib));
    }

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
    std::array<std::uint64_t, kDecodeTokens> prefetched{};
    std::array<std::uint64_t, kDecodeTokens> prefetch_hits{};
    std::array<std::uint64_t, kDecodeTokens> prefetch_waste{};
    for (std::uint32_t index = 0; index < kDecodeTokens; ++index) {
        const detail::CacheStats before = data.expert_cache_stats(0);
        begin_round();
        if (index == 0) { setenv("NINFER_LAUNCH_GATE", "1", 1); }
        const std::int64_t decode_begin = now_ns();
        token = trunk.decode(token);
        const double decode_ms = static_cast<double>(now_ns() - decode_begin) / 1.0e6;
        if (index == 0) { setenv("NINFER_LAUNCH_GATE", "0", 1); }
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            execution.dev[rank]->synchronize();
        }
        const detail::CacheStats after = data.expert_cache_stats(0);
        hits[index]   = after.hits - before.hits;
        misses[index] = after.miss_count - before.miss_count;
        prefetched[index]     = after.prefetch_pairs - before.prefetch_pairs;
        prefetch_hits[index]  = after.prefetch_hits - before.prefetch_hits;
        prefetch_waste[index] = after.prefetch_waste - before.prefetch_waste;
        tokens.push_back(collect_token(decode_ms));
    }
    const detail::CacheStats final = data.expert_cache_stats(0);
    g_enabled = false;

    std::cout << "{\n";
    std::cout << "  \"artifact\": \"" << path << "\",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"decode_tokens\": " << kDecodeTokens << ",\n";
    std::cout << "  \"warmup_tokens\": " << warmup_tokens << ",\n";
    std::cout << "  \"prefill_ms\": " << prefill_ms << ",\n";
    std::cout << "  \"cold_misses\": " << (warm.miss_count - cold.miss_count) << ",\n";
    std::cout << "  \"prefetch_pairs\": " << (final.prefetch_pairs - warm.prefetch_pairs)
              << ",\n";
    std::cout << "  \"prefetch_hits\": " << (final.prefetch_hits - warm.prefetch_hits) << ",\n";
    std::cout << "  \"prefetch_waste\": " << (final.prefetch_waste - warm.prefetch_waste)
              << ",\n";
    std::cout << "  \"tokens\": [\n";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const TokenReport& report = tokens[index];
        std::cout << "    {\"index\": " << index << ", \"wall_ms\": " << report.wall_ms
                  << ", \"anchored_device_ms\": " << report.anchored_device_ms
                  << ", \"anchored_host_ms\": " << report.anchored_host_ms
                  << ", \"ranks\": [" << report.marks[0] << ", " << report.marks[1] << "]"
                  << ", \"hits\": " << hits[index] << ", \"misses\": " << misses[index]
                  << ", \"prefetch_pairs\": " << prefetched[index]
                  << ", \"prefetch_hits\": " << prefetch_hits[index]
                  << ", \"prefetch_waste\": " << prefetch_waste[index]
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

void run_mtp(const char* path, std::uint32_t prompt_tokens, std::uint32_t warmup_tokens,
             std::uint32_t decode_tokens, const std::string& profile, std::uint64_t cache_mib,
             bool recording) {
    ExecutionContext execution({0, 1});
    artifact::Reader reader(path);
    artifact::Binder binder(reader, 2);

    ninfer::EngineOptions options;
    options.artifact_path        = path;
    options.device               = 0;
    options.tp                   = 2;
    options.devices              = {0, 1};
    options.max_context          = kKvCapacity;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
    options.max_concurrency      = 1;
    options.prefill_chunk        = kForwardTokens;
    options.expert_cache_mib     = cache_mib;
    options.expert_cache_profile = profile;

    auto load_plan = Package::plan_load(binder, options, detail::WeightsProfile::Nvfp4);
    const detail::BindingPlan& bindings = load_plan.bindings();
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, load_plan.materialization(), execution);
    detail::LoadedModelData data(bindings, materialized, detail::TextConfig::layers, 2,
                                 std::filesystem::path(path));
    data.allocate_expert_cache(options, execution);

    constexpr std::uint32_t kReplayWidth = 2; // draft=1 -> [next_input, proposal]
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
        states[slot] = std::make_unique<rt::TextState>(
            *execution.dev[slot], data.view(rank), kKvCapacity, kForwardTokens, 2,
            static_cast<int>(rank), kReplayWidth);
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

    flash::MtpDraft draft(path, execution, *comms);
    const std::size_t sink_bytes = static_cast<std::size_t>(kForwardTokens) *
                                   static_cast<std::size_t>(detail::TextConfig::residual_width) * 2;
    std::array<DeviceBuffer, kRanks> sinks{};
    flash::MtpDraft::RankTensors conditioning{};
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
        sinks[slot] = DeviceBuffer(sink_bytes);
        CUDA_CHECK(cudaSetDevice(previous));
        trunk.set_residual_sink(static_cast<int>(rank), sinks[slot].p, sink_bytes);
        conditioning[slot] = Tensor(sinks[slot].p, qwen4_exp::detail::kTextActivationDtype,
                                    {detail::TextConfig::residual_width, 1});
    }

    const std::vector<TokenId> prompt = make_prompt(prompt_tokens);
    TokenId token = 0;
    for (std::size_t offset = 0; offset < prompt.size(); offset += kForwardTokens) {
        const std::size_t count = std::min<std::size_t>(kForwardTokens, prompt.size() - offset);
        token = trunk.prefill(std::span<const TokenId>{prompt.data() + offset, count});
    }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { execution.dev[rank]->synchronize(); }

    auto mtp_round = [&](TokenId next_input, std::uint32_t remaining, PhaseAccum* acc) -> TokenId {
        const std::uint32_t frontier = states[0]->next_position();
        std::uint32_t drafts = 0;
        if (remaining > 1 && kKvCapacity > frontier + 1) { drafts = 1; }
        const std::int64_t t0 = now_ns();
        const std::vector<TokenId> proposals = flash::draft_chain(
            draft, conditioning, next_input, drafts, static_cast<std::int32_t>(frontier));
        const std::int64_t t1 = now_ns();
        TokenId result = 0;
        std::uint32_t accepted = 0;
        std::int64_t t4 = t1;
        if (drafts == 0) {
            result = trunk.decode(next_input);
            if (acc != nullptr) { add_ms(acc->draft_ms, t0, t1); }
        } else if (!recording) {
            std::vector<TokenId> verify_tokens;
            verify_tokens.reserve(proposals.size() + 1);
            verify_tokens.push_back(next_input);
            verify_tokens.insert(verify_tokens.end(), proposals.begin(), proposals.end());
            const std::int64_t vs = now_ns();
            const std::vector<TokenId> predictions = trunk.decode_span(verify_tokens);
            const std::int64_t vp = now_ns();
            accepted = flash::accepted_prefix(proposals, predictions);
            std::vector<TokenId> committed = flash::commit_tokens(proposals, predictions);
            const std::int64_t vc = now_ns();
            if (committed.empty()) { committed.push_back(predictions.front()); }
            int previous = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            if (previous != 0) { CUDA_CHECK(cudaSetDevice(0)); }
            CUDA_CHECK(cudaStreamSynchronize(states[0]->stream()));
            if (previous != 0) { CUDA_CHECK(cudaSetDevice(previous)); }
            t4 = now_ns();
            result = committed.back();
            if (acc != nullptr) {
                add_ms(acc->draft_ms, t0, t1);
                add_ms(acc->span_ms, vs, vp);
                add_ms(acc->accepting_ms, vp, vc);
                add_ms(acc->verify_ms, vs, vc);
                add_ms(acc->sync_ms, vc, t4);
                acc->drafted += drafts;
                acc->accepted += accepted;
            }
        } else {
            const rt::TextTrunk::VerificationBaseline baseline =
                trunk.capture_verification_baseline();
            const std::int64_t va = now_ns();
            struct ActionGuard {
                rt::TextTrunk* trunk;
                ~ActionGuard() { trunk->set_gdn_state_action(rt::GdnStateAction::UpdateInPlace); }
            } guard{&trunk};
            trunk.set_gdn_state_action(rt::GdnStateAction::RecordForReplay);
            const std::int64_t vd = now_ns();
            std::vector<TokenId> verify_tokens;
            verify_tokens.reserve(proposals.size() + 1);
            verify_tokens.push_back(next_input);
            verify_tokens.insert(verify_tokens.end(), proposals.begin(), proposals.end());
            const std::vector<TokenId> predictions = trunk.decode_span(verify_tokens);
            const std::int64_t vp = now_ns();
            accepted = flash::accepted_prefix(proposals, predictions);
            std::vector<TokenId> committed = flash::commit_tokens(proposals, predictions);
            const std::int64_t vc = now_ns();
            const std::uint32_t committed_count = accepted + 1;
            if (committed.empty()) { committed.push_back(predictions.front()); }
            trunk.commit_verified_span(verify_tokens, committed_count, baseline);
            const std::int64_t t3 = now_ns();
            int previous = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            if (previous != 0) { CUDA_CHECK(cudaSetDevice(0)); }
            CUDA_CHECK(cudaStreamSynchronize(states[0]->stream()));
            if (previous != 0) { CUDA_CHECK(cudaSetDevice(previous)); }
            t4 = now_ns();
            result = committed.back();
            if (acc != nullptr) {
                add_ms(acc->draft_ms, t0, t1);
                add_ms(acc->baseline_ms, t1, va);
                add_ms(acc->action_ms, va, vd);
                add_ms(acc->span_ms, vd, vp);
                add_ms(acc->accepting_ms, vp, vc);
                add_ms(acc->verify_ms, t1, vc);
                add_ms(acc->commit_ms, vc, t3);
                add_ms(acc->sync_ms, t3, t4);
                acc->drafted += drafts;
                acc->accepted += accepted;
            }
        }
        const std::int64_t t5 = now_ns();
        if (acc != nullptr) {
            add_ms(acc->rest_ms, t4, t5);
            add_ms(acc->total_ms, t0, t5);
            ++acc->rounds;
        }
        return result;
    };

    for (std::uint32_t index = 0; index < warmup_tokens; ++index) {
        token = mtp_round(token, static_cast<std::uint32_t>(-1), nullptr);
    }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { execution.dev[rank]->synchronize(); }

    const detail::CacheStats before = data.expert_cache_stats(0);
    PhaseAccum acc;
    for (std::uint32_t index = 0; index < decode_tokens; ++index) {
        token = mtp_round(token, decode_tokens - index, &acc);
    }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { execution.dev[rank]->synchronize(); }
    const detail::CacheStats after = data.expert_cache_stats(0);

    const double round_count = static_cast<double>(acc.rounds);
    const auto avg = [round_count](double value) {
        return round_count == 0.0 ? 0.0 : value / round_count;
    };
    const double total = avg(acc.total_ms);
    const double verify = avg(acc.verify_ms);
    const auto share = [total](double value) {
        return total == 0.0 ? 0.0 : 100.0 * value / total;
    };
    const auto verify_share = [verify](double value) {
        return verify == 0.0 ? 0.0 : 100.0 * value / verify;
    };
    std::cout << "{\n";
    std::cout << "  \"mode\": \"" << (recording ? "mtp_verify_subphase" : "mtp_span_norecord")
              << "\",\n";
    std::cout << "  \"artifact\": \"" << path << "\",\n";
    std::cout << "  \"draft_window\": 1,\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"warmup_rounds\": " << warmup_tokens << ",\n";
    std::cout << "  \"decode_rounds\": " << decode_tokens << ",\n";
    std::cout << "  \"expert_cache_mib\": " << cache_mib << ",\n";
    std::cout << "  \"expert_cache_profile\": \"" << profile << "\",\n";
    std::cout << "  \"rounds\": " << acc.rounds << ",\n";
    std::cout << "  \"drafted\": " << acc.drafted << ",\n";
    std::cout << "  \"accepted\": " << acc.accepted << ",\n";
    std::cout << "  \"acceptance\": "
              << (acc.drafted == 0
                      ? 0.0
                      : static_cast<double>(acc.accepted) / static_cast<double>(acc.drafted))
              << ",\n";
    std::cout << "  \"misses\": " << (after.miss_count - before.miss_count) << ",\n";
    std::cout << "  \"avg_total_ms\": " << total << ",\n";
    std::cout << "  \"phases\": {\n";
    struct Row { const char* name; double ms; };
    const Row rows[5] = {{"draft_chain", avg(acc.draft_ms)},
                         {"verify_decode_span", avg(acc.verify_ms)},
                         {"commit_verified_span", avg(acc.commit_ms)},
                         {"rank0_sync", avg(acc.sync_ms)},
                         {"rest", avg(acc.rest_ms)}};
    for (std::size_t index = 0; index < 5; ++index) {
        std::cout << "    \"" << rows[index].name << "\": {\"avg_ms\": " << rows[index].ms
                  << ", \"pct_of_total\": " << share(rows[index].ms) << "}"
                  << (index + 1 == 5 ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"verify_subphases\": {\n";
    struct SubRow { const char* name; double ms; };
    const SubRow subrows[4] = {{"baseline_capture", avg(acc.baseline_ms)},
                               {"action_switch", avg(acc.action_ms)},
                               {"decode_span", avg(acc.span_ms)},
                               {"accepting", avg(acc.accepting_ms)}};
    for (std::size_t index = 0; index < 4; ++index) {
        std::cout << "    \"" << subrows[index].name << "\": {\"avg_ms\": " << subrows[index].ms
                  << ", \"pct_of_verify\": " << verify_share(subrows[index].ms)
                  << ", \"pct_of_total\": " << share(subrows[index].ms) << "}"
                  << (index + 1 == 4 ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"sum_phases_ms\": "
              << (avg(acc.draft_ms) + avg(acc.verify_ms) + avg(acc.commit_ms) + avg(acc.sync_ms) +
                  avg(acc.rest_ms))
              << "\n";
    std::cout << "}\n";
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
    const std::string mode = argc > 4 ? argv[4] : "";
    const bool mtp_mode = mode == "mtp" || mode == "mtp_norecord";
    try {
        if (mtp_mode) {
            const std::uint32_t decode_tokens =
                argc > 5 ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 128;
            std::string profile;
            if (argc > 6) {
                profile = argv[6];
            } else if (const char* value = std::getenv("NINFER_FLASH_PHASE_PROFILE")) {
                profile = value;
            }
            std::uint64_t cache_mib = 15000;
            if (const char* value = std::getenv("NINFER_FLASH_PHASE_CACHE_MIB")) {
                cache_mib = std::stoull(value);
            }
            run_mtp(path, prompt_tokens, warmup_tokens, decode_tokens, profile, cache_mib,
                    mode == "mtp");
        } else {
            run(path, prompt_tokens, warmup_tokens);
        }
    } catch (const std::exception& error) {
        std::cerr << "flash_next_phase_profile: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
