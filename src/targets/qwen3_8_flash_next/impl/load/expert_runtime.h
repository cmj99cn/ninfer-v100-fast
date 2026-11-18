#pragma once

#include "core/arena.h"
#include "core/host_worker_pool.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_nvfp4.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include <ninfer/targets/qwen4_exp/model_view.h>

#include <cuda_runtime.h>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class ExpertRuntime {
public:
    ExpertRuntime(const std::filesystem::path& artifact_path, int device,
                  std::uint32_t expert_begin, const ExpertCacheOptions& options,
                  std::uint64_t available_bytes, qwen4_exp::TextResidentStats* stats);

    ExpertRuntime(const ExpertRuntime&)            = delete;
    ExpertRuntime& operator=(const ExpertRuntime&) = delete;
    ExpertRuntime(ExpertRuntime&&)                 = delete;
    ExpertRuntime& operator=(ExpertRuntime&&)      = delete;
    ~ExpertRuntime();

    void routed_launch(std::uint32_t layer, const Tensor& x,
                       const ops::MoeNvfp4RouteView& route, Tensor& destination,
                       cudaStream_t stream);
    void routed_finish(cudaStream_t stream);
    void prefetch_previous_round(cudaStream_t stream);
    void prime_static_residency();
    void set_graph_mode(bool enabled) noexcept { graph_mode_ = enabled; }
    void freeze_current_residency() noexcept {
        cache_.freeze();
        frozen_ = true;
    }

    [[nodiscard]] const ExpertCache& cache() const noexcept { return cache_; }
    [[nodiscard]] float* activation_scratch() const noexcept {
        return static_cast<float*>(activation_scratch_.p);
    }
    [[nodiscard]] std::uint64_t cache_bytes() const noexcept {
        return static_cast<std::uint64_t>(slab_.bytes + lookup_.bytes + divisors_.bytes);
    }
    [[nodiscard]] std::uint64_t scratch_bytes() const noexcept {
        return static_cast<std::uint64_t>(activation_scratch_.bytes);
    }
    [[nodiscard]] std::uint64_t host_bytes() const noexcept {
        return static_cast<std::uint64_t>(kCopyStages + kPrefetchStages) * kPairBytes +
               lookup_host_->size() + route_history_host_->size() + kProbeBytes;
    }

private:
    struct PendingRoute {
        bool active = false;
        std::uint32_t layer = 0;
        Tensor x;
        Tensor destination;
        ops::MoeNvfp4RouteView route;
    };

    struct PrefetchJob {
        std::uint32_t layer;
        std::uint32_t expert;
        std::uint32_t slot;
    };

    static constexpr std::uint32_t kCopyStages     = 3;
    static constexpr std::uint32_t kPrefetchStages = 3;
    static constexpr std::size_t kProbeBytes =
        (2 + 2 * static_cast<std::size_t>(TextConfig::experts)) * sizeof(std::int32_t);
    static constexpr std::size_t kRouteHistoryBytes =
        static_cast<std::size_t>(kTextLayers) * TextConfig::experts_per_token *
        sizeof(std::int32_t);

    void launch_pull(std::uint32_t layer, std::uint32_t expert, std::uint32_t slot);
    void refresh_lookup();
    void build_lookup();
    void record_route(std::uint32_t layer, const ops::MoeNvfp4RouteView& route,
                      cudaStream_t stream);
    void publish_prefetch(std::uint32_t layer, cudaStream_t stream);
    void run_prefetch(std::vector<PrefetchJob> jobs);
    void run_prefetch_jobs(std::vector<PrefetchJob> jobs);

    int device_                              = 0;
    std::uint32_t expert_begin_              = 0;
    qwen4_exp::TextResidentStats* stats_     = nullptr;
    HostPairSource source_;
    ExpertCache cache_;
    std::array<std::unique_ptr<PinnedHostBuffer>, kCopyStages> staging_;
    std::array<cudaEvent_t, kCopyStages> stage_ready_{};
    std::uint32_t stage_cursor_              = 0;
    cudaStream_t copy_stream_                = nullptr;
    cudaEvent_t copy_done_                   = nullptr;
    cudaEvent_t probe_done_                  = nullptr;
    cudaEvent_t down_done_                   = nullptr;
    DeviceBuffer slab_;
    DeviceBuffer lookup_;
    DeviceBuffer divisors_;
    DeviceBuffer activation_scratch_;
    DeviceBuffer rollback_;
    std::unique_ptr<PinnedHostBuffer> lookup_host_;
    DeviceBuffer route_history_;
    std::unique_ptr<PinnedHostBuffer> route_history_host_;
    std::array<std::unique_ptr<PinnedHostBuffer>, kPrefetchStages> prefetch_staging_;
    std::array<cudaEvent_t, kPrefetchStages> prefetch_stage_ready_{};
    std::uint32_t prefetch_stage_cursor_ = 0;
    std::array<cudaEvent_t, kTextLayers> prefetch_layer_event_{};
    std::array<bool, kTextLayers> prefetch_pending_{};
    std::array<bool, kTextLayers> prefetch_recorded_{};
    std::array<std::vector<std::pair<std::uint32_t, std::uint32_t>>, kTextLayers>
        prefetch_pairs_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> prefetch_predicted_;
    std::unique_ptr<HostWorkerPool> prefetch_workers_;
    std::future<void> prefetch_future_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_recorded_cv_;
    bool prefetch_failed_ = false;
    std::vector<std::int32_t> host_ids_;
    bool frozen_                       = false;
    bool graph_mode_                   = false;
    std::int32_t* probe_host_          = nullptr;
    std::int32_t* probe_miss_count_    = nullptr;
    std::int32_t* probe_active_count_  = nullptr;
    std::int32_t* probe_miss_ids_      = nullptr;
    std::int32_t* probe_active_ids_    = nullptr;
    std::int32_t* probe_miss_count_dev_   = nullptr;
    std::int32_t* probe_active_count_dev_ = nullptr;
    std::int32_t* probe_miss_ids_dev_     = nullptr;
    std::int32_t* probe_active_ids_dev_   = nullptr;
    PendingRoute pending_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
