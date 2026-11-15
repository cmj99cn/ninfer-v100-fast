#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_nvfp4.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include <ninfer/targets/qwen4_exp/model_view.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
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

    void routed(std::uint32_t layer, const Tensor& x, const ops::MoeNvfp4RouteView& route,
                Tensor& destination, cudaStream_t stream);

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
        return static_cast<std::uint64_t>(kCopyStages) * kPairBytes + lookup_host_->size() +
               probe_host_->size();
    }

private:
    static constexpr std::uint32_t kCopyStages = 3;

    void launch_pull(std::uint32_t layer, std::uint32_t expert, std::uint32_t slot);
    void refresh_lookup();

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
    DeviceBuffer slab_;
    DeviceBuffer lookup_;
    DeviceBuffer divisors_;
    DeviceBuffer activation_scratch_;
    std::unique_ptr<PinnedHostBuffer> lookup_host_;
    std::unique_ptr<PinnedHostBuffer> probe_host_;
    std::int32_t* probe_ids_dev_   = nullptr;
    std::int32_t* probe_count_dev_ = nullptr;
    std::int32_t* probe_ids_host_   = nullptr;
    std::int32_t* probe_count_host_ = nullptr;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
