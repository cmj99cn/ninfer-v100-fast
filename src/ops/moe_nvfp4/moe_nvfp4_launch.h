#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct MoeNvfp4RouteOutputs {
    std::int32_t* ids          = nullptr;
    float* coefficients        = nullptr;
    float* shared_gate         = nullptr;
};

struct MoeNvfp4BatchedPlanes {
    const std::uint8_t* codes    = nullptr;
    const std::uint8_t* scales   = nullptr;
    const float* divisors        = nullptr;
};

struct MoeNvfp4Scratch {
    float* activation = nullptr;
};

inline constexpr std::int32_t kMoeMaxSmallT = 4;

struct MoeNvfp4Probe {
    std::int32_t* active_ids   = nullptr;
    std::int32_t* miss_ids     = nullptr;
    std::int32_t* active_count = nullptr;
    std::int32_t* miss_count   = nullptr;
    std::int32_t expert_begin  = 0;
    std::int32_t expert_end    = 0;
};

void moe_nvfp4_route_launch(const Tensor& x, const void* control, MoeNvfp4RouteOutputs outputs,
                            float* logits, std::int32_t* counter, cudaStream_t stream);

void moe_nvfp4_grouped_launch(const Tensor& x, MoeNvfp4BatchedPlanes gate_up,
                              MoeNvfp4BatchedPlanes down, const MoeNvfp4RouteOutputs& route,
                              std::int32_t expert_begin, std::int32_t experts,
                              Tensor& destination, cudaStream_t stream);

void moe_nvfp4_cached_launch(const Tensor& x, const std::uint8_t* slots,
                             std::uint64_t slot_bytes, std::int32_t slot_count,
                             const std::int32_t* lookup, const float* divisors,
                             MoeNvfp4Scratch scratch, const MoeNvfp4RouteOutputs& route,
                             Tensor& destination, cudaStream_t stream);

void moe_nvfp4_cached_gate_up_launch(const Tensor& x, const std::uint8_t* slots,
                                     std::uint64_t slot_bytes, const std::int32_t* lookup,
                                     const float* divisors, MoeNvfp4Scratch scratch,
                                     const MoeNvfp4RouteOutputs& route,
                                     const MoeNvfp4Probe* probe, cudaStream_t stream);

void moe_nvfp4_cached_down_launch(const std::uint8_t* slots, std::uint64_t slot_bytes,
                                  const std::int32_t* lookup, const float* divisors,
                                  MoeNvfp4Scratch scratch, const MoeNvfp4RouteOutputs& route,
                                  Tensor& destination, cudaStream_t stream);

void moe_nvfp4_route_probe_launch(const std::int32_t* ids, std::int32_t entries,
                                  const std::int32_t* lookup, std::int32_t expert_begin,
                                  std::int32_t expert_end, std::int32_t* active_ids,
                                  std::int32_t* miss_ids, std::int32_t* active_count,
                                  std::int32_t* miss_count, cudaStream_t stream);

void moe_nvfp4_shared_launch(const Tensor& x, const void* control, const void* shared_down,
                             MoeNvfp4Scratch scratch, const MoeNvfp4RouteOutputs& route,
                             Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops::detail
