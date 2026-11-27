#include "targets/qwen3_8_flash_next/impl/load/expert_runtime.h"

#include "core/device.h"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

ExpertCacheOptions with_expert_begin(const ExpertCacheOptions& options, std::uint32_t expert_begin) {
    ExpertCacheOptions result = options;
    result.expert_begin        = expert_begin;
    return result;
}


class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        if (previous_ != device) { CUDA_CHECK(cudaSetDevice(device)); }
    }
    ~ScopedDevice() { (void)cudaSetDevice(previous_); }
    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = 0;
};

constexpr std::int32_t kExperts      = TextConfig::experts;
constexpr std::int32_t kTopK         = TextConfig::experts_per_token;
constexpr std::size_t kLookupEntries = static_cast<std::size_t>(kTextLayers) * kExperts;

// ---------------------------------------------------------------------------
// DFlash2 cross-column route-overlap diagnostic (NINFER_DFLASH2_ROUTE_OVERLAP=1).
//
// For every eager multi-column target forward (the T in [2,8] verify/decoded
// span) this records, per rank and layer, how many of the kTopK*T routed slots
// collapse onto the same expert. Routing is replicated across tensor-parallel
// ranks (moe/router is Axis::Replicated) and the ids are global, so the metric
// is reported two ways:
//   * global  - distinct ids over the full kTopK*T slab (rank-invariant);
//   * owned   - only the slots whose global id falls in this rank's expert
//               range, i.e. the experts this rank actually reads. This is the
//               ratio that bounds a grouped (one-weight-read-per-expert) path.
// The gate is evaluated once; when it is off nothing is copied, synchronized or
// accumulated, so the default execution path is byte-for-byte unchanged.
// ---------------------------------------------------------------------------
constexpr std::int32_t kOverlapMaxTokens = 8;
constexpr std::uint32_t kOverlapRanks    = kTp;
constexpr std::uint64_t kOverlapWarmupRounds = 16;

bool route_overlap_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DFLASH2_ROUTE_OVERLAP");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

// DFlash2 grouped multi-column routed path. The gate is evaluated once; with it off no grouped
// buffer is allocated and the cached kernel sequence is unchanged.
bool grouped_routed_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DFLASH2_GROUPED_ROUTED");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

struct RouteOverlapAccum {
    std::uint64_t calls           = 0;
    std::uint64_t global_slots    = 0;
    std::uint64_t global_distinct = 0;
    std::uint64_t owned_slots     = 0;
    std::uint64_t owned_distinct  = 0;
};

struct RouteOverlapRegistry {
    std::mutex mutex;
    std::array<std::array<std::uint64_t, kOverlapMaxTokens + 1>, kOverlapRanks> records{};
    std::array<std::array<std::array<RouteOverlapAccum, kTextLayers>, kOverlapMaxTokens + 1>,
               kOverlapRanks>
        accum{};
    bool at_exit_registered = false;
};

RouteOverlapRegistry& route_overlap_registry() {
    static RouteOverlapRegistry registry;
    return registry;
}

void route_overlap_print() {
    RouteOverlapRegistry& registry = route_overlap_registry();
    std::lock_guard lock(registry.mutex);
    for (std::uint32_t rank = 0; rank < kOverlapRanks; ++rank) {
        for (std::int32_t tokens = 2; tokens <= kOverlapMaxTokens; ++tokens) {
            const auto& by_layer = registry.accum[rank][static_cast<std::size_t>(tokens)];
            std::uint64_t calls = 0, global_slots = 0, global_distinct = 0;
            std::uint64_t owned_slots = 0, owned_distinct = 0, layers = 0;
            double global_min = 2.0, global_max = -1.0;
            double owned_min = 2.0, owned_max = -1.0;
            for (const RouteOverlapAccum& layer : by_layer) {
                if (layer.calls == 0) { continue; }
                ++layers;
                calls += layer.calls;
                global_slots += layer.global_slots;
                global_distinct += layer.global_distinct;
                owned_slots += layer.owned_slots;
                owned_distinct += layer.owned_distinct;
                if (layer.global_slots != 0) {
                    const double overlap =
                        1.0 - static_cast<double>(layer.global_distinct) /
                                  static_cast<double>(layer.global_slots);
                    global_min = std::min(global_min, overlap);
                    global_max = std::max(global_max, overlap);
                }
                if (layer.owned_slots != 0) {
                    const double overlap =
                        1.0 - static_cast<double>(layer.owned_distinct) /
                                  static_cast<double>(layer.owned_slots);
                    owned_min = std::min(owned_min, overlap);
                    owned_max = std::max(owned_max, overlap);
                }
            }
            if (layers == 0 || global_slots == 0) { continue; }
            const double global_ratio = static_cast<double>(global_distinct) /
                                        static_cast<double>(global_slots);
            std::fprintf(
                stderr,
                "[dflash2-route-overlap] rank=%u tokens=%d layers=%llu calls=%llu "
                "global distinct/total=%.5f overlap=%.5f per-layer[min=%.5f max=%.5f] "
                "owned distinct/total=%.5f overlap=%.5f per-layer[min=%.5f max=%.5f]\n",
                rank, tokens, static_cast<unsigned long long>(layers),
                static_cast<unsigned long long>(calls), global_ratio, 1.0 - global_ratio,
                global_min, global_max,
                owned_slots == 0 ? 1.0
                                 : static_cast<double>(owned_distinct) /
                                       static_cast<double>(owned_slots),
                owned_slots == 0 ? 0.0
                                 : 1.0 - static_cast<double>(owned_distinct) /
                                             static_cast<double>(owned_slots),
                owned_slots == 0 ? 0.0 : owned_min, owned_slots == 0 ? 0.0 : owned_max);
        }
    }
    // Combined across ranks for the canonical verify width (tokens == 6).
    constexpr std::int32_t kVerifyTokens = 6;
    std::uint64_t global_slots = 0, global_distinct = 0, owned_slots = 0, owned_distinct = 0;
    std::array<std::uint64_t, kTextLayers> layer_global_slots{};
    std::array<std::uint64_t, kTextLayers> layer_global_distinct{};
    std::array<std::array<double, kTextLayers>, kOverlapRanks> layer_owned_overlap{};
    for (std::uint32_t rank = 0; rank < kOverlapRanks; ++rank) {
        const auto& by_layer = registry.accum[rank][kVerifyTokens];
        for (std::uint32_t layer = 0; layer < kTextLayers; ++layer) {
            const RouteOverlapAccum& value = by_layer[layer];
            layer_global_slots[layer] += value.global_slots;
            layer_global_distinct[layer] += value.global_distinct;
            global_slots += value.global_slots;
            global_distinct += value.global_distinct;
            owned_slots += value.owned_slots;
            owned_distinct += value.owned_distinct;
            if (value.owned_slots != 0) {
                layer_owned_overlap[rank][layer] =
                    1.0 - static_cast<double>(value.owned_distinct) /
                              static_cast<double>(value.owned_slots);
            }
        }
    }
    if (global_slots == 0) { return; }
    double layer_min = 2.0, layer_max = -1.0;
    for (std::uint32_t layer = 0; layer < kTextLayers; ++layer) {
        if (layer_global_slots[layer] == 0) { continue; }
        const double overlap = 1.0 - static_cast<double>(layer_global_distinct[layer]) /
                                          static_cast<double>(layer_global_slots[layer]);
        layer_min = std::min(layer_min, overlap);
        layer_max = std::max(layer_max, overlap);
        std::fprintf(stderr,
                     "[dflash2-route-overlap] layer=%-2u tokens=%d global_overlap=%.5f "
                     "owned_overlap rank0=%.5f rank1=%.5f\n",
                     layer, kVerifyTokens, overlap, layer_owned_overlap[0][layer],
                     layer_owned_overlap[kOverlapRanks - 1][layer]);
    }
    const double ratio = static_cast<double>(global_distinct) / static_cast<double>(global_slots);
    std::fprintf(stderr,
                 "[dflash2-route-overlap] COMBINED tokens=%d ranks=%u "
                 "total_slots=%llu distinct=%llu distinct/total=%.5f overlap=%.5f "
                 "per-layer[min=%.5f max=%.5f] owned distinct/total=%.5f\n",
                 kVerifyTokens, kOverlapRanks,
                 static_cast<unsigned long long>(global_slots),
                 static_cast<unsigned long long>(global_distinct), ratio, 1.0 - ratio, layer_min,
                 layer_max,
                 owned_slots == 0 ? 1.0
                                  : static_cast<double>(owned_distinct) /
                                        static_cast<double>(owned_slots));
}

void route_overlap_record(std::uint32_t rank, std::uint32_t layer, std::int32_t tokens,
                          std::uint32_t expert_begin, const std::int32_t* ids) {
    if (rank >= kOverlapRanks || layer >= kTextLayers || tokens < 2 ||
        tokens > kOverlapMaxTokens) {
        return;
    }
    RouteOverlapRegistry& registry = route_overlap_registry();
    std::lock_guard lock(registry.mutex);
    if (!registry.at_exit_registered) {
        registry.at_exit_registered = true;
        std::atexit(route_overlap_print);
    }
    const std::uint64_t index = registry.records[rank][static_cast<std::size_t>(tokens)]++;
    if (index / kTextLayers < kOverlapWarmupRounds) { return; }

    const std::int32_t slots = kTopK * tokens;
    const std::int32_t begin = static_cast<std::int32_t>(expert_begin);
    const std::int32_t end   = begin + static_cast<std::int32_t>(kRankExperts);
    std::array<bool, static_cast<std::size_t>(kExperts)> seen{};
    std::array<bool, static_cast<std::size_t>(kExperts)> owned_seen{};
    std::uint64_t distinct = 0, owned_slots = 0, owned_distinct = 0;
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        const std::int32_t id = ids[slot];
        if (id < 0 || id >= kExperts) { continue; }
        if (!seen[static_cast<std::size_t>(id)]) {
            seen[static_cast<std::size_t>(id)] = true;
            ++distinct;
        }
        if (id >= begin && id < end) {
            ++owned_slots;
            if (!owned_seen[static_cast<std::size_t>(id)]) {
                owned_seen[static_cast<std::size_t>(id)] = true;
                ++owned_distinct;
            }
        }
    }
    RouteOverlapAccum& accum = registry.accum[rank][static_cast<std::size_t>(tokens)][layer];
    accum.calls += 1;
    accum.global_slots += static_cast<std::uint64_t>(slots);
    accum.global_distinct += distinct;
    accum.owned_slots += owned_slots;
    accum.owned_distinct += owned_distinct;
}

} // namespace

ExpertRuntime::ExpertRuntime(const std::filesystem::path& artifact_path, int device,
                             std::uint32_t expert_begin, const ExpertCacheOptions& options,
                             std::uint64_t available_bytes, qwen4_exp::TextResidentStats* stats)
    : device_(device), expert_begin_(expert_begin), stats_(stats),
      source_(artifact_path, kPairBytes),
      cache_(with_expert_begin(options, expert_begin), kPairBytes, available_bytes, source_) {
    const ScopedDevice scope(device_);
    slab_    = DeviceBuffer(static_cast<std::size_t>(cache_.plan().bytes));
    lookup_  = DeviceBuffer(kLookupEntries * 4);
    divisors_ = DeviceBuffer(static_cast<std::size_t>(cache_.slots()) * 3 * 4);
    activation_scratch_ = DeviceBuffer(
        static_cast<std::size_t>(ops::detail::kMoeMaxSmallT) *
        static_cast<std::size_t>(kTopK) *
        static_cast<std::size_t>(TextConfig::expert_intermediate) * sizeof(float));
    if (grouped_routed_enabled()) {
        const std::size_t bucket_slots = static_cast<std::size_t>(cache_.slots());
        grouped_buckets_ = DeviceBuffer(bucket_slots *
            static_cast<std::size_t>(ops::detail::kMoeGroupBucketStride) * sizeof(std::int32_t));
        grouped_meta_ = DeviceBuffer(
            (1 + 2 * static_cast<std::size_t>(ops::detail::kMoeGroupMaxGroups)) *
            sizeof(std::int32_t));
        grouped_partial_ = DeviceBuffer(
            static_cast<std::size_t>(ops::detail::kMoeMaxSmallT) * static_cast<std::size_t>(kTopK) *
            static_cast<std::size_t>(TextConfig::hidden) * sizeof(float));
    }
    rollback_ = DeviceBuffer(static_cast<std::size_t>(TextConfig::hidden) * sizeof(std::uint16_t));
    lookup_host_ = std::make_unique<PinnedHostBuffer>(kLookupEntries * 4);
    route_history_ = DeviceBuffer(kRouteHistoryBytes);
    route_history_.fill(0xFF);
    route_history_host_ = std::make_unique<PinnedHostBuffer>(kRouteHistoryBytes);
    std::memset(route_history_host_->data(), 0xFF, kRouteHistoryBytes);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&probe_host_), kProbeBytes,
                             cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(probe_host_, 0, kProbeBytes);
    probe_miss_count_   = probe_host_;
    probe_active_count_ = probe_host_ + 1;
    probe_miss_ids_     = probe_host_ + 2;
    probe_active_ids_   = probe_host_ + 2 + kExperts;
    void* probe_dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&probe_dev, probe_host_, 0));
    probe_miss_count_dev_   = static_cast<std::int32_t*>(probe_dev);
    probe_active_count_dev_ = probe_miss_count_dev_ + 1;
    probe_miss_ids_dev_     = probe_miss_count_dev_ + 2;
    probe_active_ids_dev_   = probe_miss_ids_dev_ + kExperts;
    for (auto& staging : staging_) {
        staging = std::make_unique<PinnedHostBuffer>(static_cast<std::size_t>(kPairBytes));
    }
    for (auto& staging : prefetch_staging_) {
        staging = std::make_unique<PinnedHostBuffer>(static_cast<std::size_t>(kPairBytes));
    }
    for (auto& event : stage_ready_) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    for (auto& event : prefetch_stage_ready_) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    for (auto& event : prefetch_layer_event_) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&copy_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&probe_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&down_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    prefetch_workers_ = std::make_unique<HostWorkerPool>(1, 2);
    refresh_lookup();
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    if (options.profile.has_value()) { prime_static_residency(); }
    if (stats_ != nullptr) {
        stats_->expert_cache_bytes = cache_bytes();
        stats_->expert_scratch_bytes = scratch_bytes();
        stats_->expert_cache_pairs = cache_.plan().pairs;
        stats_->expert_host_bytes  = host_bytes();
    }
}

ExpertRuntime::~ExpertRuntime() {
    const ScopedDevice scope(device_);
    if (prefetch_future_.valid()) { prefetch_future_.get(); }
    prefetch_workers_.reset();
    if (probe_host_ != nullptr) { (void)cudaFreeHost(probe_host_); }
    if (copy_stream_ != nullptr) { (void)cudaStreamDestroy(copy_stream_); }
    if (probe_done_ != nullptr) { (void)cudaEventDestroy(probe_done_); }
    if (down_done_ != nullptr) { (void)cudaEventDestroy(down_done_); }
    if (copy_done_ != nullptr) { (void)cudaEventDestroy(copy_done_); }
    for (cudaEvent_t& event : stage_ready_) {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }
    for (cudaEvent_t& event : prefetch_stage_ready_) {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }
    for (cudaEvent_t& event : prefetch_layer_event_) {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }
}

void ExpertRuntime::launch_pull(std::uint32_t layer, std::uint32_t expert, std::uint32_t slot) {
    const std::uint32_t stage = stage_cursor_;
    stage_cursor_ = (stage_cursor_ + 1) % kCopyStages;
    CUDA_CHECK(cudaEventSynchronize(stage_ready_[stage]));
    PinnedHostBuffer& staging = *staging_[stage];
    const std::uint32_t global = expert_begin_ + expert;
    source_.read(layer, global,
                 std::span<std::byte>(static_cast<std::byte*>(staging.data()), kPairBytes));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(slab_.p) + slot * kPairBytes,
                               staging.data(), kPairBytes, cudaMemcpyHostToDevice, copy_stream_));
    float divisors[3]{};
    source_.read_divisors(layer, global, divisors);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(divisors_.p) + static_cast<std::size_t>(slot) * 3,
                               divisors, sizeof(divisors), cudaMemcpyHostToDevice, copy_stream_));
    CUDA_CHECK(cudaEventRecord(stage_ready_[stage], copy_stream_));
    CUDA_CHECK(cudaEventRecord(copy_done_, copy_stream_));
}

void ExpertRuntime::build_lookup() {
    auto* table = static_cast<std::int32_t*>(lookup_host_->data());
    std::fill(table, table + kLookupEntries, -1);
    for (std::uint32_t layer = 0; layer < kTextLayers; ++layer) {
        std::int32_t* row = table + static_cast<std::size_t>(layer) * kExperts + expert_begin_;
        for (std::uint32_t expert = 0; expert < kRankExperts; ++expert) {
            const std::int32_t slot = cache_.ready_slot(layer, expert);
            if (slot >= 0) { row[expert] = slot; }
        }
    }
}

void ExpertRuntime::refresh_lookup() {
    build_lookup();
    CUDA_CHECK(cudaMemcpyAsync(lookup_.p, lookup_host_->data(), kLookupEntries * 4,
                               cudaMemcpyHostToDevice, copy_stream_));
    CUDA_CHECK(cudaEventRecord(copy_done_, copy_stream_));
}

void ExpertRuntime::prime_static_residency() {
    const ScopedDevice scope(device_);
    const std::uint32_t residency = cache_.residency();
    for (std::uint32_t slot = 0; slot < residency; ++slot) {
        const auto [layer, expert] = cache_.resident_pair(slot);
        launch_pull(layer, expert, slot);
    }
    refresh_lookup();
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    cache_.freeze();
    frozen_ = true;
}

void ExpertRuntime::record_route(std::uint32_t layer, const ops::MoeNvfp4RouteView& route,
                                 cudaStream_t stream) {
    const std::int32_t tokens = route.expert_ids.ne[1];
    if (tokens < 1 || tokens > kRouteHistoryTokens) { return; }
    auto* destination = static_cast<std::byte*>(route_history_.p) +
                        static_cast<std::size_t>(layer) * kRouteHistoryStride;
    CUDA_CHECK(cudaMemsetAsync(destination, 0xFF, kRouteHistoryStride, stream));
    CUDA_CHECK(cudaMemcpyAsync(destination, route.expert_ids.data,
                                static_cast<std::size_t>(kTopK) * tokens * 4,
                                cudaMemcpyDeviceToDevice, stream));
}

void ExpertRuntime::prefetch_previous_round(cudaStream_t stream) {
    if (frozen_) { return; }
    const ScopedDevice scope(device_);
    if (prefetch_future_.valid()) { prefetch_future_.get(); }
    CUDA_CHECK(cudaMemcpyAsync(route_history_host_->data(), route_history_.p, kRouteHistoryBytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto* history = static_cast<const std::int32_t*>(route_history_host_->data());
    for (const auto& [layer, local] : prefetch_predicted_) {
        bool used = false;
        const auto* layer_history =
            history + static_cast<std::size_t>(layer) * (kTopK * kRouteHistoryTokens);
        for (std::int32_t token = 0; token < kRouteHistoryTokens && !used; ++token) {
            for (std::int32_t index = 0; index < kTopK; ++index) {
                if (layer_history[static_cast<std::size_t>(token) * kTopK + index] ==
                    static_cast<std::int32_t>(expert_begin_ + local)) {
                    used = true;
                    break;
                }
            }
        }
        cache_.note_prefetch_result(used);
    }
    prefetch_predicted_.clear();

    std::vector<PrefetchJob> jobs;
    const std::size_t max_jobs =
        cache_.slots() > 64 ? static_cast<std::size_t>(cache_.slots() - 64) : 1;
    for (std::uint32_t layer = 0; layer < kTextLayers && jobs.size() < max_jobs; ++layer) {
        std::array<bool, kRankExperts> seen{};
        const auto* layer_history =
            history + static_cast<std::size_t>(layer) * (kTopK * kRouteHistoryTokens);
        for (std::int32_t token = 0; token < kRouteHistoryTokens && jobs.size() < max_jobs; ++token) {
            for (std::int32_t index = 0; index < kTopK && jobs.size() < max_jobs; ++index) {
                const std::int32_t global =
                    layer_history[static_cast<std::size_t>(token) * kTopK + index];
                if (global < static_cast<std::int32_t>(expert_begin_) ||
                    global >= static_cast<std::int32_t>(expert_begin_ + kRankExperts)) {
                    continue;
                }
                const auto local = static_cast<std::uint32_t>(global - expert_begin_);
                if (seen[local]) { continue; }
                seen[local] = true;
                if (cache_.resident_slot(layer, local) >= 0) { continue; }
                const std::uint32_t slot = cache_.prefetch_place(layer, local);
                jobs.push_back(PrefetchJob{layer, local, slot});
                prefetch_pairs_[layer].emplace_back(local, slot);
                prefetch_pending_[layer] = true;
                prefetch_predicted_.emplace_back(layer, local);
            }
        }
    }
    if (jobs.empty()) { return; }

    build_lookup();
    CUDA_CHECK(cudaMemcpyAsync(lookup_.p, lookup_host_->data(), kLookupEntries * 4,
                               cudaMemcpyHostToDevice, stream));
    {
        std::lock_guard lock(prefetch_mutex_);
        for (std::uint32_t layer = 0; layer < kTextLayers; ++layer) {
            prefetch_recorded_[layer] = !prefetch_pending_[layer];
        }
    }
    prefetch_future_ =
        prefetch_workers_->submit([this, jobs = std::move(jobs)]() mutable {
            run_prefetch(std::move(jobs));
        });
}

void ExpertRuntime::run_prefetch(std::vector<PrefetchJob> jobs) {
    try {
        run_prefetch_jobs(std::move(jobs));
    } catch (...) {
        {
            std::lock_guard lock(prefetch_mutex_);
            prefetch_failed_ = true;
            for (bool& recorded : prefetch_recorded_) { recorded = true; }
        }
        prefetch_recorded_cv_.notify_all();
        throw;
    }
}

void ExpertRuntime::run_prefetch_jobs(std::vector<PrefetchJob> jobs) {
    CUDA_CHECK(cudaSetDevice(device_));
    const std::uint32_t finish = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t active       = finish;
    for (const PrefetchJob& job : jobs) {
        if (job.layer != active) {
            if (active != finish) {
                CUDA_CHECK(cudaEventRecord(prefetch_layer_event_[active], copy_stream_));
                {
                    std::lock_guard lock(prefetch_mutex_);
                    prefetch_recorded_[active] = true;
                }
                prefetch_recorded_cv_.notify_all();
            }
            active = job.layer;
        }
        const std::uint32_t stage = prefetch_stage_cursor_;
        prefetch_stage_cursor_    = (prefetch_stage_cursor_ + 1) % kPrefetchStages;
        CUDA_CHECK(cudaEventSynchronize(prefetch_stage_ready_[stage]));
        PinnedHostBuffer& staging = *prefetch_staging_[stage];
        const std::uint32_t global = expert_begin_ + job.expert;
        source_.read(job.layer, global,
                     std::span<std::byte>(static_cast<std::byte*>(staging.data()), kPairBytes));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(slab_.p) +
                                       static_cast<std::size_t>(job.slot) * kPairBytes,
                                   staging.data(), kPairBytes, cudaMemcpyHostToDevice,
                                   copy_stream_));
        float divisors[3]{};
        source_.read_divisors(job.layer, global, divisors);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(divisors_.p) +
                                       static_cast<std::size_t>(job.slot) * 3,
                                   divisors, sizeof(divisors), cudaMemcpyHostToDevice,
                                   copy_stream_));
        CUDA_CHECK(cudaEventRecord(prefetch_stage_ready_[stage], copy_stream_));
    }
    if (active != finish) {
        CUDA_CHECK(cudaEventRecord(prefetch_layer_event_[active], copy_stream_));
        {
            std::lock_guard lock(prefetch_mutex_);
            prefetch_recorded_[active] = true;
        }
        prefetch_recorded_cv_.notify_all();
    }
}

void ExpertRuntime::publish_prefetch(std::uint32_t layer, cudaStream_t stream) {
    if (!prefetch_pending_[layer]) { return; }
    {
        std::unique_lock lock(prefetch_mutex_);
        prefetch_recorded_cv_.wait(lock, [&] { return prefetch_recorded_[layer]; });
        if (prefetch_failed_) { throw std::runtime_error("the expert prefetch worker failed"); }
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, prefetch_layer_event_[layer], 0));
    auto* base = static_cast<std::int32_t*>(lookup_.p) +
                 static_cast<std::size_t>(layer) * kExperts + expert_begin_;
    auto* host = static_cast<std::int32_t*>(lookup_host_->data()) +
                 static_cast<std::size_t>(layer) * kExperts + expert_begin_;
    for (const auto& [local, slot] : prefetch_pairs_[layer]) {
        if (cache_.resident_slot(layer, local) != static_cast<std::int32_t>(slot)) { continue; }
        cache_.release_prefetch(slot);
        host[local] = static_cast<std::int32_t>(slot);
    }
    CUDA_CHECK(cudaMemcpyAsync(base, host, kRankExperts * 4, cudaMemcpyHostToDevice, stream));
    prefetch_pairs_[layer].clear();
    prefetch_pending_[layer] = false;
}

void ExpertRuntime::routed_cached_launch(std::uint32_t layer, const Tensor& x,
                                          const ops::MoeNvfp4RouteView& route, Tensor& destination,
                                          cudaStream_t stream) {
    const std::int32_t tokens = route.expert_ids.ne[1];
    const auto* table_row = static_cast<const std::int32_t*>(lookup_.p) +
                            static_cast<std::size_t>(layer) * kExperts;
    const ops::detail::MoeNvfp4RouteOutputs outputs{
        static_cast<std::int32_t*>(route.expert_ids.data),
        static_cast<float*>(route.coefficients.data),
        static_cast<float*>(route.shared_gate.data)};
    const ops::MoeNvfp4ExpertSlots bank{
        .payload    = static_cast<const std::uint8_t*>(slab_.p),
        .slot_bytes = kPairBytes,
        .slot_count = static_cast<std::int32_t>(cache_.slots()),
        .lookup     = table_row,
        .divisors   = static_cast<const float*>(divisors_.p),
    };
    const ops::MoeNvfp4ExpertScratch scratch{.activation = activation_scratch()};
    if (grouped_routed_enabled() && tokens >= 2 &&
        tokens <= ops::detail::kMoeMaxSmallT) {
        const ops::detail::MoeNvfp4Scratch detail_scratch{.activation = activation_scratch()};
        const ops::detail::MoeNvfp4GroupedScratch grouped{
            .buckets  = static_cast<std::int32_t*>(grouped_buckets_.p),
            .meta     = static_cast<std::int32_t*>(grouped_meta_.p),
            .partial  = static_cast<float*>(grouped_partial_.p),
        };
        ops::detail::moe_nvfp4_grouped_cached_launch(
            x, static_cast<const std::uint8_t*>(slab_.p), kPairBytes,
            static_cast<std::int32_t>(cache_.slots()), table_row,
            static_cast<const float*>(divisors_.p), detail_scratch, grouped, outputs, destination,
            stream);
        return;
    }
    ops::moe_routed_nvfp4_cached(x, bank, scratch, route, destination, stream);
}

void ExpertRuntime::routed_launch(std::uint32_t layer, const Tensor& x,
                                   const ops::MoeNvfp4RouteView& route, Tensor& destination,
                                   cudaStream_t stream) {
    const ScopedDevice scope(device_);
    const std::int32_t tokens = route.expert_ids.ne[1];
    if (tokens < 1) { return; }

    publish_prefetch(layer, stream);
    record_route(layer, route, stream);

    const std::int32_t expert_begin = static_cast<std::int32_t>(expert_begin_);
    const std::int32_t expert_end   = expert_begin + static_cast<std::int32_t>(kRankExperts);
    const auto* table_row = static_cast<const std::int32_t*>(lookup_.p) +
                             static_cast<std::size_t>(layer) * kExperts;
    const ops::detail::MoeNvfp4RouteOutputs outputs{
        static_cast<std::int32_t*>(route.expert_ids.data),
        static_cast<float*>(route.coefficients.data),
        static_cast<float*>(route.shared_gate.data)};
    const ops::MoeNvfp4ExpertScratch scratch{.activation = activation_scratch()};
    const ops::detail::MoeNvfp4Scratch detail_scratch{.activation = activation_scratch()};
    const ops::MoeNvfp4ExpertSlots bank{
        .payload    = static_cast<const std::uint8_t*>(slab_.p),
        .slot_bytes = kPairBytes,
        .slot_count = static_cast<std::int32_t>(cache_.slots()),
        .lookup     = table_row,
        .divisors   = static_cast<const float*>(divisors_.p),
    };

    if (tokens == 1) {
        const ops::detail::MoeNvfp4Probe probe{
            .active_ids   = probe_active_ids_dev_,
            .miss_ids     = probe_miss_ids_dev_,
            .active_count = probe_active_count_dev_,
            .miss_count   = probe_miss_count_dev_,
            .expert_begin = expert_begin,
            .expert_end   = expert_end,
        };
        const ops::detail::MoeNvfp4Probe* active_probe = graph_mode_ ? nullptr : &probe;
        ops::detail::moe_nvfp4_cached_gate_up_launch(
            x, static_cast<const std::uint8_t*>(slab_.p), kPairBytes, table_row,
            static_cast<const float*>(divisors_.p), detail_scratch, outputs, active_probe, stream);
        if (!graph_mode_) {
            CUDA_CHECK(cudaEventRecord(probe_done_, stream));
            CUDA_CHECK(cudaMemcpyAsync(rollback_.p, destination.data,
                                       static_cast<std::size_t>(TextConfig::hidden) *
                                           sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        ops::detail::moe_nvfp4_cached_down_launch(
            static_cast<const std::uint8_t*>(slab_.p), kPairBytes, table_row,
            static_cast<const float*>(divisors_.p), detail_scratch, outputs, destination, stream);
        if (!graph_mode_) { CUDA_CHECK(cudaEventRecord(down_done_, stream)); }
        pending_.active      = !graph_mode_;
        pending_.layer       = layer;
        pending_.tokens      = 1;
        pending_.x           = x;
        pending_.destination = destination;
        pending_.route       = route;
        return;
    }

    if (!graph_mode_ && tokens <= kRouteHistoryTokens) {
        if (route_overlap_enabled()) {
            overlap_ids_.resize(static_cast<std::size_t>(kTopK) * static_cast<std::size_t>(tokens));
            CUDA_CHECK(cudaMemcpyAsync(overlap_ids_.data(), route.expert_ids.data,
                                       overlap_ids_.size() * 4, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::uint32_t rank =
                static_cast<std::uint32_t>(expert_begin_) / kRankExperts;
            route_overlap_record(rank, layer, tokens, expert_begin_, overlap_ids_.data());
        }
        CUDA_CHECK(cudaMemsetAsync(probe_miss_count_dev_, 0, 2 * sizeof(std::int32_t), stream));
        ops::detail::moe_nvfp4_route_probe_launch(
            static_cast<const std::int32_t*>(route.expert_ids.data), kTopK * tokens, table_row,
            expert_begin, expert_end, probe_active_ids_dev_, probe_miss_ids_dev_,
            probe_active_count_dev_, probe_miss_count_dev_, stream);
        CUDA_CHECK(cudaEventRecord(probe_done_, stream));
        routed_cached_launch(layer, x, route, destination, stream);
        CUDA_CHECK(cudaEventRecord(down_done_, stream));
        pending_.active      = true;
        pending_.layer       = layer;
        pending_.tokens      = static_cast<std::uint32_t>(tokens);
        pending_.x           = x;
        pending_.destination = destination;
        pending_.route       = route;
        return;
    }

    host_ids_.resize(static_cast<std::size_t>(kTopK) * tokens);
    CUDA_CHECK(cudaMemcpyAsync(host_ids_.data(), route.expert_ids.data, host_ids_.size() * 4,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::pair<std::uint32_t, std::int32_t>> needed;
    bool any_missing = false;
    std::vector<bool> seen(static_cast<std::size_t>(kRankExperts), false);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t slot = 0; slot < kTopK; ++slot) {
            const std::int32_t global =
                host_ids_[static_cast<std::size_t>(slot) + static_cast<std::size_t>(kTopK) * token];
            if (global < expert_begin || global >= expert_end) { continue; }
            const auto local = static_cast<std::uint32_t>(global - expert_begin);
            if (seen[local]) { continue; }
            seen[local] = true;
            needed.emplace_back(local, global);
            if (cache_.resident_slot(layer, local) < 0) { any_missing = true; }
        }
    }
    if (frozen_ || !any_missing) {
        cache_.record_resident_lookups(static_cast<std::uint64_t>(needed.size()));
    } else {
        if (needed.size() > cache_.slots()) {
            throw std::logic_error("one layer selected more experts than the cache can hold");
        }
        std::vector<std::pair<std::uint32_t, std::uint32_t>> protected_pairs;
        std::uint32_t missing = 0;
        for (const auto& [local, global] : needed) {
            if (cache_.resident_slot(layer, local) >= 0) {
                protected_pairs.emplace_back(layer, local);
            } else {
                ++missing;
            }
        }
        cache_.reserve(missing, protected_pairs);
        for (const auto& [local, global] : needed) {
            std::int32_t slot = cache_.resident_slot(layer, local);
            if (slot < 0) {
                slot = static_cast<std::int32_t>(cache_.load(layer, local));
                launch_pull(layer, local, static_cast<std::uint32_t>(slot));
            } else {
                (void)cache_.find(layer, local);
            }
        }
        refresh_lookup();
        CUDA_CHECK(cudaStreamWaitEvent(stream, copy_done_, 0));
    }

    routed_cached_launch(layer, x, route, destination, stream);

    if (stats_ != nullptr) {
        const CacheStats stats = cache_.stats();
        stats_->expert_cache_hits       = stats.hits;
        stats_->expert_cache_misses     = stats.miss_count;
        stats_->expert_cache_miss_bytes = stats.miss_count * kPairBytes;
    }
}

void ExpertRuntime::routed_finish(cudaStream_t stream) {
    if (!pending_.active) { return; }
    if (frozen_) { pending_.active = false; return; }
    const ScopedDevice scope(device_);
    const std::uint32_t layer = pending_.layer;
    const Tensor& x = pending_.x;
    const auto& route = pending_.route;
    Tensor& destination = pending_.destination;

    const std::int32_t expert_begin = static_cast<std::int32_t>(expert_begin_);
    const std::int32_t expert_end   = expert_begin + static_cast<std::int32_t>(kRankExperts);
    const auto* table_row = static_cast<const std::int32_t*>(lookup_.p) +
                            static_cast<std::size_t>(layer) * kExperts;
    const ops::detail::MoeNvfp4RouteOutputs outputs{
        static_cast<std::int32_t*>(route.expert_ids.data),
        static_cast<float*>(route.coefficients.data),
        static_cast<float*>(route.shared_gate.data)};
    const ops::detail::MoeNvfp4Scratch detail_scratch{.activation = activation_scratch()};

    const cudaError_t probe_status = cudaEventQuery(probe_done_);
    if (probe_status == cudaErrorNotReady) {
        (void)cudaGetLastError();
        CUDA_CHECK(cudaEventSynchronize(probe_done_));
    } else {
        CUDA_CHECK(probe_status);
    }
    const std::int32_t active_count = *probe_active_count_;
    if (*probe_miss_count_ == 0) {
        cache_.record_resident_lookups(static_cast<std::uint64_t>(active_count));
        if (stats_ != nullptr) {
            const CacheStats stats = cache_.stats();
            stats_->expert_cache_hits       = stats.hits;
            stats_->expert_cache_misses     = stats.miss_count;
            stats_->expert_cache_miss_bytes = stats.miss_count * kPairBytes;
        }
        pending_.active = false;
        return;
    }

    std::vector<std::pair<std::uint32_t, std::int32_t>> needed;
    needed.reserve(static_cast<std::size_t>(active_count));
    for (std::int32_t index = 0; index < *probe_miss_count_; ++index) {
        const std::int32_t global = probe_miss_ids_[index];
        needed.emplace_back(static_cast<std::uint32_t>(global - expert_begin), global);
    }
    for (std::int32_t index = 0; index < active_count; ++index) {
        const std::int32_t global = probe_active_ids_[index];
        const auto local = static_cast<std::uint32_t>(global - expert_begin);
        if (cache_.resident_slot(layer, local) >= 0) { needed.emplace_back(local, global); }
    }

    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, down_done_, 0));
    if (needed.size() > cache_.slots()) {
        throw std::logic_error("one layer selected more experts than the cache can hold");
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> protected_pairs;
    std::uint32_t missing = 0;
    for (const auto& [local, global] : needed) {
        if (cache_.resident_slot(layer, local) >= 0) {
            protected_pairs.emplace_back(layer, local);
        } else {
            ++missing;
        }
    }
    cache_.reserve(missing, protected_pairs);
    for (const auto& [local, global] : needed) {
        std::int32_t slot = cache_.resident_slot(layer, local);
        if (slot < 0) {
            slot = static_cast<std::int32_t>(cache_.load(layer, local));
            launch_pull(layer, local, static_cast<std::uint32_t>(slot));
        } else {
            (void)cache_.find(layer, local);
        }
    }
    refresh_lookup();
    CUDA_CHECK(cudaStreamWaitEvent(stream, copy_done_, 0));
    if (pending_.tokens == 1) {
        CUDA_CHECK(cudaMemcpyAsync(destination.data, rollback_.p,
                                    static_cast<std::size_t>(TextConfig::hidden) *
                                        sizeof(std::uint16_t),
                                    cudaMemcpyDeviceToDevice, stream));
        ops::detail::moe_nvfp4_cached_gate_up_launch(
            x, static_cast<const std::uint8_t*>(slab_.p), kPairBytes, table_row,
            static_cast<const float*>(divisors_.p), detail_scratch, outputs, nullptr, stream);
        ops::detail::moe_nvfp4_cached_down_launch(
            static_cast<const std::uint8_t*>(slab_.p), kPairBytes, table_row,
            static_cast<const float*>(divisors_.p), detail_scratch, outputs, destination, stream);
    } else {
        const ops::MoeNvfp4ExpertSlots bank{
            .payload    = static_cast<const std::uint8_t*>(slab_.p),
            .slot_bytes = kPairBytes,
            .slot_count = static_cast<std::int32_t>(cache_.slots()),
            .lookup     = table_row,
            .divisors   = static_cast<const float*>(divisors_.p),
        };
        const ops::MoeNvfp4ExpertScratch scratch{.activation = activation_scratch()};
        CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(), stream));
        routed_cached_launch(layer, x, route, destination, stream);
    }
    pending_.active = false;

    if (stats_ != nullptr) {
        const CacheStats stats = cache_.stats();
        stats_->expert_cache_hits       = stats.hits;
        stats_->expert_cache_misses     = stats.miss_count;
        stats_->expert_cache_miss_bytes = stats.miss_count * kPairBytes;
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
