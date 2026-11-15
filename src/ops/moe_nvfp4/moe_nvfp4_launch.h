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

void moe_nvfp4_route_launch(const Tensor& x, const void* control, MoeNvfp4RouteOutputs outputs,
                            cudaStream_t stream);

void moe_nvfp4_grouped_launch(const Tensor& x, MoeNvfp4BatchedPlanes gate_up,
                              MoeNvfp4BatchedPlanes down, const MoeNvfp4RouteOutputs& route,
                              std::int32_t expert_begin, std::int32_t experts,
                              Tensor& destination, cudaStream_t stream);

void moe_nvfp4_cached_launch(const Tensor& x, const std::uint8_t* slots,
                             std::uint64_t slot_bytes, std::int32_t slot_count,
                             const std::int32_t* lookup, const float* divisors,
                             MoeNvfp4Scratch scratch, const MoeNvfp4RouteOutputs& route,
                             Tensor& destination, cudaStream_t stream);

void moe_nvfp4_shared_launch(const Tensor& x, const void* control, const void* shared_down,
                             MoeNvfp4Scratch scratch, const MoeNvfp4RouteOutputs& route,
                             Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops::detail
