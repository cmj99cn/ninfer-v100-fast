#include "targets/qwen3_8_flash_next/impl/load/expert_runtime.h"

#include "core/device.h"

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

} // namespace

ExpertRuntime::ExpertRuntime(const std::filesystem::path& artifact_path, int device,
                             std::uint32_t expert_begin, const ExpertCacheOptions& options,
                             std::uint64_t available_bytes, qwen4_exp::TextResidentStats* stats)
    : device_(device), expert_begin_(expert_begin), stats_(stats),
      source_(artifact_path, kPairBytes),
      cache_(options, kPairBytes, available_bytes, source_),
      staging_(static_cast<std::size_t>(kPairBytes)),
      host_lookup_(static_cast<std::size_t>(TextConfig::experts), -1),
      host_divisors_(static_cast<std::size_t>(cache_.slots()) * 3, 0.0F) {
    const ScopedDevice scope(device_);
    slab_    = DeviceBuffer(static_cast<std::size_t>(cache_.plan().bytes));
    lookup_  = DeviceBuffer(static_cast<std::size_t>(TextConfig::experts) * 4);
    divisors_ = DeviceBuffer(static_cast<std::size_t>(cache_.slots()) * 3 * 4);
    if (stats_ != nullptr) {
        stats_->expert_cache_bytes = device_bytes();
        stats_->expert_cache_pairs = cache_.plan().pairs;
        stats_->expert_host_bytes  = host_bytes();
    }
}

void ExpertRuntime::pull(std::uint32_t layer, std::uint32_t expert, std::uint32_t slot,
                         cudaStream_t stream) {
    const std::uint32_t global = expert_begin_ + expert;
    source_.read(layer, global,
                 std::span<std::byte>(static_cast<std::byte*>(staging_.data()), kPairBytes));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(slab_.p) + slot * kPairBytes,
                               staging_.data(), kPairBytes, cudaMemcpyHostToDevice, stream));
    float divisors[3]{};
    source_.read_divisors(layer, global, divisors);
    std::memcpy(host_divisors_.data() + static_cast<std::size_t>(slot) * 3, divisors,
                sizeof(divisors));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(divisors_.p) + static_cast<std::size_t>(slot) * 3,
                               divisors, sizeof(divisors), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void ExpertRuntime::routed(std::uint32_t layer, const Tensor& x,
                           const ops::MoeNvfp4RouteView& route, Tensor& destination,
                           cudaStream_t stream) {
    const ScopedDevice scope(device_);
    const std::int32_t top_k  = TextConfig::experts_per_token;
    const std::int32_t tokens = route.expert_ids.ne[1];
    if (tokens < 1) { return; }

    host_ids_.resize(static_cast<std::size_t>(top_k) * tokens);
    CUDA_CHECK(cudaMemcpyAsync(host_ids_.data(), route.expert_ids.data,
                               host_ids_.size() * 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::pair<std::uint32_t, std::int32_t>> needed;
    std::vector<bool> seen(static_cast<std::size_t>(kRankExperts), false);
    const std::int32_t expert_begin = static_cast<std::int32_t>(expert_begin_);
    const std::int32_t expert_end   = expert_begin + static_cast<std::int32_t>(kRankExperts);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t slot = 0; slot < top_k; ++slot) {
            const std::int32_t global = host_ids_[static_cast<std::size_t>(slot) + top_k * token];
            if (global < expert_begin || global >= expert_end) { continue; }
            const auto local = static_cast<std::uint32_t>(global - expert_begin);
            if (seen[local]) { continue; }
            seen[local] = true;
            needed.emplace_back(local, global);
        }
    }

    const std::uint32_t slots = cache_.slots();
    if (needed.size() > slots) {
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

    std::fill(host_lookup_.begin(), host_lookup_.end(), -1);
    for (const auto& [local, global] : needed) {
        std::int32_t slot = cache_.resident_slot(layer, local);
        if (slot < 0) {
            slot = static_cast<std::int32_t>(cache_.load(layer, local));
            pull(layer, local, static_cast<std::uint32_t>(slot), stream);
        } else {
            (void)cache_.find(layer, local);
        }
        host_lookup_[static_cast<std::size_t>(global)] = slot;
    }
    CUDA_CHECK(cudaMemcpyAsync(lookup_.p, host_lookup_.data(), host_lookup_.size() * 4,
                               cudaMemcpyHostToDevice, stream));
    const ops::MoeNvfp4ExpertSlots bank{
        .payload    = static_cast<const std::uint8_t*>(slab_.p),
        .slot_bytes = kPairBytes,
        .slot_count = static_cast<std::int32_t>(cache_.slots()),
        .lookup     = static_cast<const std::int32_t*>(lookup_.p),
        .divisors   = static_cast<const float*>(divisors_.p),
    };
    ops::moe_routed_nvfp4_cached(x, bank, route, destination, stream);

    if (stats_ != nullptr) {
        const CacheStats stats = cache_.stats();
        stats_->expert_cache_hits       = stats.hits;
        stats_->expert_cache_misses     = stats.miss_count;
        stats_->expert_cache_miss_bytes = stats.miss_count * kPairBytes;
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
