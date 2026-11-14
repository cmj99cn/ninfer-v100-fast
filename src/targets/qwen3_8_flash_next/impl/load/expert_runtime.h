#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_nvfp4.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include <ninfer/targets/qwen4_exp/model_view.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
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

    void routed(std::uint32_t layer, const Tensor& x, const ops::MoeNvfp4RouteView& route,
                Tensor& destination, cudaStream_t stream);

    [[nodiscard]] const ExpertCache& cache() const noexcept { return cache_; }
    [[nodiscard]] std::uint64_t device_bytes() const noexcept {
        return static_cast<std::uint64_t>(slab_.bytes + lookup_.bytes + divisors_.bytes);
    }
    [[nodiscard]] std::uint64_t host_bytes() const noexcept {
        return static_cast<std::uint64_t>(staging_.size());
    }

private:
    void pull(std::uint32_t layer, std::uint32_t expert, std::uint32_t slot, cudaStream_t stream);

    int device_                              = 0;
    std::uint32_t expert_begin_              = 0;
    qwen4_exp::TextResidentStats* stats_     = nullptr;
    HostPairSource source_;
    ExpertCache cache_;
    PinnedHostBuffer staging_;
    DeviceBuffer slab_;
    DeviceBuffer lookup_;
    DeviceBuffer divisors_;
    std::vector<std::int32_t> host_lookup_;
    std::vector<std::int32_t> host_ids_;
    std::vector<float> host_divisors_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
