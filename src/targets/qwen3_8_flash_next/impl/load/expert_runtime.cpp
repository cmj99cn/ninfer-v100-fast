#include "targets/qwen3_8_flash_next/impl/load/expert_runtime.h"

#include "core/device.h"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

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

} // namespace

ExpertRuntime::ExpertRuntime(const std::filesystem::path& artifact_path, int device,
                             std::uint32_t expert_begin, const ExpertCacheOptions& options,
                             std::uint64_t available_bytes, qwen4_exp::TextResidentStats* stats)
    : device_(device), expert_begin_(expert_begin), stats_(stats),
      source_(artifact_path, kPairBytes),
      cache_(options, kPairBytes, available_bytes, source_) {
    const ScopedDevice scope(device_);
    slab_    = DeviceBuffer(static_cast<std::size_t>(cache_.plan().bytes));
    lookup_  = DeviceBuffer(kLookupEntries * 4);
    divisors_ = DeviceBuffer(static_cast<std::size_t>(cache_.slots()) * 3 * 4);
    activation_scratch_ = DeviceBuffer(
        static_cast<std::size_t>(kTopK) *
        static_cast<std::size_t>(TextConfig::expert_intermediate) * sizeof(float));
    lookup_host_ = std::make_unique<PinnedHostBuffer>(kLookupEntries * 4);
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
    for (auto& event : stage_ready_) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&copy_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&probe_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    refresh_lookup();
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    if (stats_ != nullptr) {
        stats_->expert_cache_bytes = cache_bytes();
        stats_->expert_scratch_bytes = scratch_bytes();
        stats_->expert_cache_pairs = cache_.plan().pairs;
        stats_->expert_host_bytes  = host_bytes();
    }
}

ExpertRuntime::~ExpertRuntime() {
    const ScopedDevice scope(device_);
    if (probe_host_ != nullptr) { (void)cudaFreeHost(probe_host_); }
    if (copy_stream_ != nullptr) { (void)cudaStreamDestroy(copy_stream_); }
    if (probe_done_ != nullptr) { (void)cudaEventDestroy(probe_done_); }
    if (copy_done_ != nullptr) { (void)cudaEventDestroy(copy_done_); }
    for (cudaEvent_t& event : stage_ready_) {
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

void ExpertRuntime::refresh_lookup() {
    auto* table = static_cast<std::int32_t*>(lookup_host_->data());
    std::fill(table, table + kLookupEntries, -1);
    for (std::uint32_t layer = 0; layer < kTextLayers; ++layer) {
        std::int32_t* row = table + static_cast<std::size_t>(layer) * kExperts + expert_begin_;
        for (std::uint32_t expert = 0; expert < kRankExperts; ++expert) {
            const std::int32_t slot = cache_.resident_slot(layer, expert);
            if (slot >= 0) { row[expert] = slot; }
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(lookup_.p, table, kLookupEntries * 4, cudaMemcpyHostToDevice,
                               copy_stream_));
    CUDA_CHECK(cudaEventRecord(copy_done_, copy_stream_));
}

void ExpertRuntime::routed(std::uint32_t layer, const Tensor& x,
                           const ops::MoeNvfp4RouteView& route, Tensor& destination,
                           cudaStream_t stream) {
    const ScopedDevice scope(device_);
    const std::int32_t tokens = route.expert_ids.ne[1];
    if (tokens < 1) { return; }

    const std::int32_t expert_begin = static_cast<std::int32_t>(expert_begin_);
    const std::int32_t expert_end   = expert_begin + static_cast<std::int32_t>(kRankExperts);
    const auto* table_row = static_cast<const std::int32_t*>(lookup_.p) +
                            static_cast<std::size_t>(layer) * kExperts;

    std::vector<std::pair<std::uint32_t, std::int32_t>> needed;
    bool any_missing = false;
    if (tokens == 1) {
        ops::detail::moe_nvfp4_route_probe_launch(
            static_cast<const std::int32_t*>(route.expert_ids.data), kTopK, table_row,
            expert_begin, expert_end, probe_active_ids_dev_, probe_miss_ids_dev_,
            probe_active_count_dev_, probe_miss_count_dev_, stream);
        CUDA_CHECK(cudaEventRecord(probe_done_, stream));
        CUDA_CHECK(cudaEventSynchronize(probe_done_));
        const std::int32_t active_count = *probe_active_count_;
        if (*probe_miss_count_ == 0) {
            cache_.record_resident_lookups(static_cast<std::uint64_t>(active_count));
        } else {
            any_missing = true;
            needed.reserve(static_cast<std::size_t>(active_count));
            for (std::int32_t index = 0; index < active_count; ++index) {
                const std::int32_t global = probe_active_ids_[index];
                needed.emplace_back(static_cast<std::uint32_t>(global - expert_begin), global);
            }
        }
    } else {
        host_ids_.resize(static_cast<std::size_t>(kTopK) * tokens);
        CUDA_CHECK(cudaMemcpyAsync(host_ids_.data(), route.expert_ids.data,
                                   host_ids_.size() * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

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
        if (!any_missing) {
            cache_.record_resident_lookups(static_cast<std::uint64_t>(needed.size()));
        }
    }

    bool copied = false;
    if (any_missing) {
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
        copied = true;
    }
    if (copied) { CUDA_CHECK(cudaStreamWaitEvent(stream, copy_done_, 0)); }

    const ops::MoeNvfp4ExpertSlots bank{
        .payload    = static_cast<const std::uint8_t*>(slab_.p),
        .slot_bytes = kPairBytes,
        .slot_count = static_cast<std::int32_t>(cache_.slots()),
        .lookup     = table_row,
        .divisors   = static_cast<const float*>(divisors_.p),
    };
    const ops::MoeNvfp4ExpertScratch scratch{
        .activation = static_cast<float*>(activation_scratch_.p),
    };
    ops::moe_routed_nvfp4_cached(x, bank, scratch, route, destination, stream);

    if (stats_ != nullptr) {
        const CacheStats stats = cache_.stats();
        stats_->expert_cache_hits       = stats.hits;
        stats_->expert_cache_misses     = stats.miss_count;
        stats_->expert_cache_miss_bytes = stats.miss_count * kPairBytes;
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
