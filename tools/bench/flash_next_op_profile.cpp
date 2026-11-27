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
#include <iomanip>
#include <optional>
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
constexpr std::uint32_t kDecodeTokens  = 8;
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
    std::array<std::uint8_t, kEventPool> kind{};
    int count = 0;
};

std::array<MarkPool, kRanks> g_pool;
bool g_enabled = false;

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

void op_mark(const char* tag, std::uint8_t kind, int rank, cudaStream_t stream) {
    if (!g_enabled) { return; }
    MarkPool& pool = g_pool[static_cast<std::size_t>(rank)];
    if (pool.count >= kEventPool) { return; }
    const int slot = pool.count++;
    CUDA_CHECK(cudaEventRecord(pool.events[slot], stream));
    pool.host[slot] = now_ns();
    pool.tags[slot] = tag;
    pool.kind[slot] = kind;
}

// Logical bytes read per op call per tensor-parallel rank, decode T = 1 (text decode).
// Derivation (see .superpowers/sdd/2026-11-19-performance-review/per-op-floor.md for the full
// formulas and artifact citations):
//   NVFP4 [N,K]  : align256(N*K/2) + N*K/16 + 4                (storage-layouts.md §4/§11)
//   Q8G32_F16S   : align256(N*(K_pad/32)*32) + N*(K_pad/32)*2, K_pad = align128(K)
//                                                             (storage-layouts.md §3)
//   BF16         : N*K*2                                        (storage-layouts.md §2)
// TP2 splits (src/targets/qwen3_8_flash_next/impl/load/sharding.cpp): head-axis roles split by
// head; vocabs split by half; experts split 256/256; everything else replicated.
// -1 means "computed at run time from the expert-cache counters (hits * 2,764,800 B)".
struct OpBytes { const char* name; double bytes_per_call; };
constexpr OpBytes kOpBytes[] = {
    {"embedding", 5120.0},          // one BF16 token row [2560]; full matrix is NOT streamed
    {"stem", 20480.0},              // attention_hc/norm [10240] BF16 replicated
    {"attn_hc", 13189120.0},        // [324,10240] down+inject (6,635,520) + up [10240,320] (6,553,600), replicated
    {"gdn_proj", 22774144.0},       // Q8 qkvz rows 8192/rank (8192*2720=22,282,240) + control BF16 [96,2560] (491,520) + a_log+dt_bias FP32 (384), replicated
    {"gdn_stage", 0.0},             // state/activation only
    {"gdn_conv", 204800.0},         // conv kernel BF16 [4,10240] (81,920) + conv state r/w (2*3*10240*2 = 122,880), replicated
    {"gdn_extract", 0.0},           // activation only
    {"gdn_recurrent", 3145728.0},   // recurrent state FP32 r+w: 24 heads/rank * 128*128 * 4 B * 2
    {"gdn_norm", 256.0},            // gain [128] BF16 replicated
    {"gdn_out", 8355840.0},         // Q8 [2560,6144] Split::Input K/2=3072: base 2560*96*32=7,864,320 + scale 2560*96*2=491,520
    {"qsa_proj", 22609920.0},       // Q8 query+gate halves 6144 rows/rank (6144*2720=16,711,680) + BF16 K/V 512 rows (2,621,440) + index_query [512,2560] (2,621,440) + index_key [128,2560] (655,360); index_* replicated
    {"qsa_prep", 1536.0},           // query/key/index norms BF16 (512+512+256+256), replicated
    {"qsa_index", 0.0},             // indexer reads index_k_pages state, not weights
    {"qsa_attn", 0.0},              // reads QSA KV pages; context-dependent, excluded
    {"qsa_tail", 0.0},              // tail state only
    {"qsa_out", 8355840.0},         // Q8 [2560,6144] Split::Input K/2=3072 (same geometry as gdn_out)
    {"route", 2626560.0},           // router+shared_score rows [513,2560] BF16 replicated
    {"routed", -1.0},               // runtime: hits * 2,764,800 B per token (expert pair = 3*(640*2560/2 + 640*2560/16))
    {"moe_reduce", 5120.0},         // allreduce of [2560] BF16 partials; NCCL-latency bound, bytes nominal
    {"moe_shared", 9830400.0},      // shared gate_up rows [1280,2560] BF16 (6,553,600) + down [2560,640] BF16 (3,276,800), replicated
    {"moe_hc_combine", 20480.0},    // moe_hc/norm [10240] BF16 replicated
    {"moe_hc_mix", 13189120.0},     // moe_hc down+inject [324,10240] (6,635,520) + up [10240,320] (6,553,600), replicated
    {"layer_end_combine", 20480.0}, // next-layer attention_hc/norm (layers<47) or tail_norm (layer 47), BF16 [10240] replicated
    {"pre_ple", 0.0},               // no weights (hc_combine)
    {"ple_lookup", 1440.0},         // 16 staged PLE rows (16 heads x 90 B)
    {"ple_proj", 65536000.0},       // ple.key rows [10240,2560] BF16 (52,428,800) + ple.value [2560,2560] BF16 (13,107,200), replicated
    {"ple_gate", 61440.0},          // 3 norms [10240] BF16 replicated
    {"ple_conv", 102400.0},         // ple.kernel BF16 [4,10240] (81,920) + next-layer norm view [10240] BF16 (20,480)
    {"head_mix", 13107200.0},       // final_hc down [320,10240] (6,553,600) + up [10240,320] (6,553,600), replicated
    {"head_linear", 635699200.0},   // output_head [248320,2560] BF16 Vocab-split: 124160 rows/rank * 2560 * 2
    {"head_gather", 248320.0},      // allgather of [124160] BF16 activations (received half)
    {"argmax", 496640.0},           // reads logits [248320] BF16
};
constexpr double kPairBytes = 2764800.0;
constexpr double kBytesPerGBs = 850.0; // floor bandwidth hypothesis, GB/s

double op_bytes_per_call(const std::string& name) {
    for (const auto& entry : kOpBytes) {
        if (name == entry.name) { return entry.bytes_per_call; }
    }
    return 0.0;
}

struct OpAggregate {
    double device_ms = 0.0;
    std::uint64_t calls = 0;
};

struct TokenReport {
    std::array<std::map<std::string, OpAggregate>, kRanks> per_rank;
    std::array<double, kRanks> rank_span_ms{};
    double anchored_device_ms = 0.0;
    double wall_ms            = 0.0;
    int marks[2]              = {0, 0};
    std::uint64_t mismatched_pairs = 0;
};

// One rank's marker stream is a sequence of BEGIN/END pairs in program order; pairs never nest
// (every op is wrapped on the same stream). The next marker after a BEGIN is its END, so pairing
// is a linear scan with a small stack guard for safety.
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
        std::vector<int> begin_stack;
        for (int index = 0; index < pool.count; ++index) {
            if (pool.kind[index] == 0) {
                begin_stack.push_back(index);
                continue;
            }
            if (begin_stack.empty()) { ++report.mismatched_pairs; continue; }
            int begin = begin_stack.back();
            if (std::string(pool.tags[begin]) != pool.tags[index]) {
                ++report.mismatched_pairs;
                begin_stack.clear();
                continue;
            }
            begin_stack.pop_back();
            float elapsed_ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, pool.events[begin], pool.events[index]));
            OpAggregate& aggregate = report.per_rank[rank][pool.tags[index]];
            aggregate.device_ms += static_cast<double>(elapsed_ms);
            ++aggregate.calls;
        }
        float span_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&span_ms, pool.events[0], pool.events[pool.count - 1]));
        report.rank_span_ms[rank] = static_cast<double>(span_ms);
        report.anchored_device_ms += static_cast<double>(span_ms);
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
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_op_profile_runtime
#define NINFER_QWEN4EXP_DIAG_OP_BEGIN(name) ::op_mark((name), /*kind=*/0, (rank), stream_for(rank))
#define NINFER_QWEN4EXP_DIAG_OP_END(name) ::op_mark((name), /*kind=*/1, (rank), stream_for(rank))
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

namespace {

namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_op_profile_runtime;

struct OpTotal {
    std::array<double, kRanks> rank_ms{};
    std::array<std::uint64_t, kRanks> calls{};
};
using OpTotals = std::map<std::string, OpTotal>;

OpTotals accumulate_totals(const std::vector<TokenReport>& reports) {
    OpTotals totals;
    for (const TokenReport& report : reports) {
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            for (const auto& [op, aggregate] : report.per_rank[rank]) {
                OpTotal& total = totals[op];
                total.rank_ms[rank] += aggregate.device_ms;
                total.calls[rank] += aggregate.calls;
            }
        }
    }
    return totals;
}

double op_bytes_per_token(const std::string& op, std::uint64_t calls_rank0, double n_tokens,
                          double avg_hits_per_rank) {
    if (op == "routed") { return avg_hits_per_rank * kPairBytes; }
    return op_bytes_per_call(op) * static_cast<double>(calls_rank0) / n_tokens;
}

struct AggregateMetrics {
    double sum_two_rank_ms    = 0.0;
    double sum_fixed_bytes    = 0.0;
};

AggregateMetrics emit_aggregate_body(const OpTotals& totals, std::uint64_t n_tokens,
                                     double wall_total_ms, double wall_avg_ms,
                                     double avg_hits_per_rank) {
    const double n = static_cast<double>(n_tokens);
    AggregateMetrics metrics;
    std::cout << "    \"tokens\": " << n_tokens << ",\n";
    std::cout << "    \"wall_ms_total\": " << wall_total_ms << ",\n";
    std::cout << "    \"wall_ms_avg\": " << wall_avg_ms << ",\n";
    std::cout << "    \"ops\": [\n";
    bool op_first = true;
    for (const auto& [op, total] : totals) {
        const double bytes_per_token =
            op_bytes_per_token(op, total.calls[0], n, avg_hits_per_rank);
        const double two_rank_ms = total.rank_ms[0] + total.rank_ms[1];
        const double wall_ms = std::max(total.rank_ms[0], total.rank_ms[1]);
        const double rank_ms_per_token[kRanks] = {total.rank_ms[0] / n,
                                                  total.rank_ms[1] / n};
        double gbps[kRanks] = {0.0, 0.0};
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            if (rank_ms_per_token[rank] > 0.0) {
                gbps[rank] = bytes_per_token / (rank_ms_per_token[rank] * 1.0e-3) / 1.0e9;
            }
        }
        const double floor_850_ms = bytes_per_token / (kBytesPerGBs * 1.0e9) * 1.0e3;
        const double floor_ratio_r0 =
            floor_850_ms > 0.0 ? rank_ms_per_token[0] / floor_850_ms : 0.0;
        const double floor_ratio_r1 =
            floor_850_ms > 0.0 ? rank_ms_per_token[1] / floor_850_ms : 0.0;
        if (!op_first) { std::cout << ",\n"; }
        op_first = false;
        const double calls_per_token = static_cast<double>(total.calls[0]) / n;
        std::cout << "      {\"op\": \"" << op << "\", \"calls_total\": [" << total.calls[0]
                  << ", " << total.calls[1] << "], \"calls_per_token\": " << calls_per_token
                  << ", \"rank_ms\": [" << total.rank_ms[0] << ", " << total.rank_ms[1]
                  << "], \"rank_ms_per_token\": [" << rank_ms_per_token[0] << ", "
                  << rank_ms_per_token[1] << "], \"two_rank_ms\": " << two_rank_ms
                  << ", \"two_rank_ms_per_token\": " << two_rank_ms / n
                  << ", \"wall_ms\": " << wall_ms
                  << ", \"wall_ms_per_token\": " << wall_ms / n
                  << ", \"bytes_per_token_per_card\": " << bytes_per_token
                  << ", \"gb_per_s_rank\": [" << gbps[0] << ", " << gbps[1] << "]"
                  << ", \"floor_850_ms_per_token\": " << floor_850_ms
                  << ", \"floor_ratio_rank\": [" << floor_ratio_r0 << ", " << floor_ratio_r1
                  << "]}";
        metrics.sum_two_rank_ms += two_rank_ms;
        metrics.sum_fixed_bytes += bytes_per_token;
    }
    std::cout << "\n    ],\n";
    std::cout << "    \"bytes_per_token_per_card_total\": " << metrics.sum_fixed_bytes << ",\n";
    std::cout << "    \"floor_850_ms_per_token_total\": "
              << metrics.sum_fixed_bytes / (kBytesPerGBs * 1.0e9) * 1.0e3 << ",\n";
    std::cout << "    \"sum_two_rank_ms\": " << metrics.sum_two_rank_ms << ",\n";
    std::cout << "    \"sum_two_rank_ms_per_token\": " << metrics.sum_two_rank_ms / n;
    return metrics;
}

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
    if (const char* cache_mib = std::getenv("NINFER_FLASH_OP_CACHE_MIB")) {
        options.expert_cache_mib = static_cast<std::uint64_t>(std::stoull(cache_mib));
    } else if (const char* cache_mib = std::getenv("NINFER_FLASH_PHASE_CACHE_MIB")) {
        options.expert_cache_mib = static_cast<std::uint64_t>(std::stoull(cache_mib));
    }
    if (const char* profile = std::getenv("NINFER_FLASH_OP_EXPERT_PROFILE")) {
        options.expert_cache_profile = profile;
    }

    std::uint32_t span_t = 1;
    if (const char* span = std::getenv("NINFER_FLASH_OP_SPAN_T")) {
        span_t = static_cast<std::uint32_t>(std::stoul(span));
        if (span_t < 1) { throw std::runtime_error("NINFER_FLASH_OP_SPAN_T must be >= 1"); }
    }
    const char* prefill_flag = std::getenv("NINFER_FLASH_OP_PREFILL");
    const bool profile_prefill =
        prefill_flag != nullptr && prefill_flag[0] != '\0' && std::string(prefill_flag) != "0";

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
    if (profile_prefill) {
        init_pools();
        begin_round();
        g_enabled = true;
    }
    const std::int64_t prefill_begin = now_ns();
    TokenId token = 0;
    for (std::size_t offset = 0; offset < prompt.size(); offset += kForwardTokens) {
        const std::size_t count = std::min<std::size_t>(kForwardTokens, prompt.size() - offset);
        token = trunk.prefill(std::span<const TokenId>{prompt.data() + offset, count});
    }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { execution.dev[rank]->synchronize(); }
    const double prefill_ms = static_cast<double>(now_ns() - prefill_begin) / 1.0e6;
    std::optional<TokenReport> prefill_report;
    if (profile_prefill) {
        g_enabled = false;
        prefill_report = collect_token(prefill_ms);
        destroy_pools();
        init_pools();
    }
    for (std::uint32_t index = 0; index < warmup_tokens; ++index) { token = trunk.decode(token); }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { execution.dev[rank]->synchronize(); }
    const detail::CacheStats warm = data.expert_cache_stats(0);

    if (!profile_prefill) { init_pools(); }
    g_enabled = true;

    std::vector<TokenReport> tokens;
    std::array<std::array<std::uint64_t, kRanks>, kDecodeTokens> hits{};
    std::array<std::array<std::uint64_t, kRanks>, kDecodeTokens> misses{};
    std::array<std::uint64_t, kDecodeTokens> prefetched{};
    std::array<std::uint64_t, kDecodeTokens> prefetch_hits{};
    std::array<std::uint64_t, kDecodeTokens> prefetch_waste{};
    for (std::uint32_t index = 0; index < kDecodeTokens; ++index) {
        std::array<detail::CacheStats, kRanks> before{data.expert_cache_stats(0),
                                                      data.expert_cache_stats(1)};
        begin_round();
        const std::int64_t decode_begin = now_ns();
        if (span_t > 1) {
            const std::vector<TokenId> span(span_t, token);
            const std::vector<TokenId> predictions = trunk.decode_span(span);
            if (!predictions.empty()) { token = predictions.back(); }
        } else {
            token = trunk.decode(token);
        }
        const double decode_ms = static_cast<double>(now_ns() - decode_begin) / 1.0e6;
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            execution.dev[rank]->synchronize();
        }
        std::array<detail::CacheStats, kRanks> after{data.expert_cache_stats(0),
                                                     data.expert_cache_stats(1)};
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            hits[index][rank]   = after[rank].hits - before[rank].hits;
            misses[index][rank] = after[rank].miss_count - before[rank].miss_count;
        }
        prefetched[index]     = after[0].prefetch_pairs - before[0].prefetch_pairs;
        prefetch_hits[index]  = after[0].prefetch_hits - before[0].prefetch_hits;
        prefetch_waste[index] = after[0].prefetch_waste - before[0].prefetch_waste;
        tokens.push_back(collect_token(decode_ms));
    }
    const detail::CacheStats final = data.expert_cache_stats(0);
    g_enabled = false;

    // Aggregate over the decode window: per op per rank totals, two-rank sum, wall (max rank),
    // per-card bytes, achieved GB/s and the floor at 850 GB/s.
    const OpTotals totals = accumulate_totals(tokens);
    std::array<double, kRanks> sum_rank_ms{};
    for (const auto& [op, total] : totals) {
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            sum_rank_ms[rank] += total.rank_ms[rank];
        }
    }
    std::array<std::uint64_t, kDecodeTokens> token_hits{};
    for (std::uint32_t index = 0; index < kDecodeTokens; ++index) {
        token_hits[index] = hits[index][0] + hits[index][1];
    }
    std::uint64_t total_routed_hits = 0;
    for (std::uint32_t index = 0; index < kDecodeTokens; ++index) { total_routed_hits += token_hits[index]; }
    const double mean_routed_hits_per_token =
        kDecodeTokens == 0 ? 0.0 : static_cast<double>(total_routed_hits) / kDecodeTokens;

    double wall_total = 0.0;
    for (const auto& report : tokens) { wall_total += report.wall_ms; }
    const double wall_avg = wall_total / static_cast<double>(tokens.size());
    double sum_two_rank_ms = 0.0;
    for (const auto& [op, total] : totals) {
        sum_two_rank_ms += total.rank_ms[0] + total.rank_ms[1];
    }
    const double n_tokens = static_cast<double>(kDecodeTokens);
    double avg_misses[kRanks] = {0.0, 0.0};
    for (std::uint32_t index = 0; index < kDecodeTokens; ++index) {
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            avg_misses[rank] += static_cast<double>(misses[index][rank]);
        }
    }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) { avg_misses[rank] /= n_tokens; }
    const double avg_hits_per_rank =
        static_cast<double>(total_routed_hits) / (2.0 * n_tokens);
    double sum_two_rank_check = 0.0;
    double sum_fixed_bytes = 0.0;
    for (const auto& [op, total] : totals) {
        sum_two_rank_check += total.rank_ms[0] + total.rank_ms[1];
        sum_fixed_bytes += op_bytes_per_token(op, total.calls[0], n_tokens, avg_hits_per_rank);
    }

    std::cout << std::setprecision(12);
    std::cout << "{\n";
    std::cout << "  \"artifact\": \"" << path << "\",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"decode_tokens\": " << kDecodeTokens << ",\n";
    std::cout << "  \"span_t\": " << span_t << ",\n";
    std::cout << "  \"warmup_tokens\": " << warmup_tokens << ",\n";
    std::cout << "  \"expert_cache_mib\": " << options.expert_cache_mib << ",\n";
    std::cout << "  \"prefill_ms\": " << prefill_ms << ",\n";
    std::cout << "  \"cold_misses\": " << (warm.miss_count - cold.miss_count) << ",\n";
    std::cout << "  \"prefetch_pairs\": " << (final.prefetch_pairs - warm.prefetch_pairs)
              << ",\n";
    std::cout << "  \"prefetch_hits\": " << (final.prefetch_hits - warm.prefetch_hits) << ",\n";
    std::cout << "  \"prefetch_waste\": " << (final.prefetch_waste - warm.prefetch_waste)
              << ",\n";
    std::uint64_t mismatched = 0;
    for (const auto& report : tokens) { mismatched += report.mismatched_pairs; }
    std::cout << "  \"mismatched_pairs\": " << mismatched << ",\n";
    std::cout << "  \"wall_ms_avg\": " << wall_avg << ",\n";
    std::cout << "  \"sum_two_rank_ms_per_token\": " << sum_two_rank_ms / n_tokens << ",\n";
    std::cout << "  \"sum_rank_ms_per_token\": [" << sum_rank_ms[0] / n_tokens << ", "
              << sum_rank_ms[1] / n_tokens << "],\n";
    std::cout << "  \"routed_miss_h2d_bytes_per_token_rank\": [" << avg_misses[0] * kPairBytes
              << ", " << avg_misses[1] * kPairBytes << "],\n";
    std::cout << "  \"mean_routed_hits_per_rank_per_token\": " << avg_hits_per_rank << ",\n";
    std::cout << "  \"routed_bytes_per_token_rank\": [" << avg_hits_per_rank * kPairBytes
              << ", " << avg_hits_per_rank * kPairBytes << "],\n";
    std::cout << "  \"tokens\": [\n";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const TokenReport& report = tokens[index];
        std::cout << "    {\"index\": " << index << ", \"wall_ms\": " << report.wall_ms
                  << ", \"anchored_device_ms\": " << report.anchored_device_ms
                  << ", \"rank_span_ms\": [" << report.rank_span_ms[0] << ", "
                  << report.rank_span_ms[1] << "]"
                  << ", \"ranks\": [" << report.marks[0] << ", " << report.marks[1] << "]"
                  << ", \"hits\": [" << hits[index][0] << ", " << hits[index][1] << "]"
                  << ", \"misses\": [" << misses[index][0] << ", " << misses[index][1] << "]"
                  << ", \"prefetch_pairs\": " << prefetched[index]
                  << ", \"prefetch_hits\": " << prefetch_hits[index]
                  << ", \"prefetch_waste\": " << prefetch_waste[index]
                  << ", \"per_rank\": [";
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            if (rank != 0) { std::cout << ", "; }
            std::cout << "{";
            bool rank_first = true;
            for (const auto& [op, aggregate] : report.per_rank[rank]) {
                if (!rank_first) { std::cout << ", "; }
                rank_first = false;
                std::cout << "\"" << op << "\": {\"ms\": " << aggregate.device_ms
                          << ", \"calls\": " << aggregate.calls << "}";
            }
            std::cout << "}";
        }
        std::cout << "]}\n" << (index + 1 == tokens.size() ? "" : ",");
    }
    std::cout << "  ],\n";
    std::cout << "  \"aggregate\": {\n";
    emit_aggregate_body(totals, kDecodeTokens, wall_total, wall_avg, avg_hits_per_rank);
    std::cout << "  }";
    if (prefill_report.has_value()) {
        const OpTotals prefill_totals = accumulate_totals({*prefill_report});
        std::cout << ",\n  \"prefill_aggregate\": {\n";
        emit_aggregate_body(prefill_totals, prompt_tokens, prefill_ms, prefill_ms, 0.0);
        std::cout << "  }";
    }
    std::cout << "\n}\n";

    destroy_pools();
}

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (path == nullptr || path[0] == '\0') {
        std::cerr << "flash_next_op_profile: an artifact path is required\n";
        return 1;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cerr << "flash_next_op_profile: two devices are required\n";
        return 1;
    }
    std::uint32_t prompt_tokens = kForwardTokens;
    if (argc > 2) { prompt_tokens = static_cast<std::uint32_t>(std::stoul(argv[2])); }
    std::uint32_t warmup_tokens = 0;
    if (argc > 3) { warmup_tokens = static_cast<std::uint32_t>(std::stoul(argv[3])); }
    try {
        run(path, prompt_tokens, warmup_tokens);
    } catch (const std::exception& error) {
        std::cerr << "flash_next_op_profile: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
