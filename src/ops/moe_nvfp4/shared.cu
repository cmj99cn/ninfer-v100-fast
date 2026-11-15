#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <type_traits>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class A>
__device__ __forceinline__ float shared_control_dot(const std::uint32_t* weight_words,
                                                    const A* activation, std::int32_t pairs) {
    if constexpr (std::is_same_v<A, __half>) {
        return bf16_f16_warp_dot(
            weight_words, reinterpret_cast<const std::uint32_t*>(activation), pairs);
    } else {
        return bf16_warp_dot(
            weight_words, reinterpret_cast<const std::uint32_t*>(activation), pairs);
    }
}

template <class A>
__global__ __launch_bounds__(256) void moe_nvfp4_shared_kernel(
    const A* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    const __nv_bfloat16* __restrict__ shared_down, const float* __restrict__ shared_gate,
    A* __restrict__ destination) {
    const int token = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;

    __shared__ float gate_up[kMoeIntermediate * 2];
    __shared__ float activation[kMoeIntermediate];

    const A* x_column = x + static_cast<std::int64_t>(token) * kMoeHidden;

    for (int row = warp; row < kMoeIntermediate * 2; row += 8) {
        const std::uint32_t* weight_words =
            reinterpret_cast<const std::uint32_t*>(control +
                                                   static_cast<std::int64_t>(kMoeExperts + 1 + row) * kMoeHidden);
        const float value = shared_control_dot<A>(weight_words, x_column, kMoeHidden / 2);
        if (lane == 0) gate_up[row] = value;
    }
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < kMoeIntermediate; i += 256) {
        activation[i] = silu(gate_up[i]) * gate_up[kMoeIntermediate + i];
    }
    __syncthreads();

    const float gate = shared_gate[token];
    A* destination_column = destination + static_cast<std::int64_t>(token) * kMoeHidden;
    for (int row = warp; row < kMoeHidden; row += 8) {
        const std::uint32_t* weight_words =
            reinterpret_cast<const std::uint32_t*>(shared_down +
                                                   static_cast<std::int64_t>(row) * kMoeIntermediate);
        const float value = bf16_warp_dot_f32(weight_words, activation, kMoeIntermediate / 2);
        if (lane == 0) {
            destination_column[row] = MoeAct<A>::from_float(
                MoeAct<A>::to_float(destination_column[row]) + gate * value);
        }
    }
}

inline constexpr int kMoeSharedTile = 16;

template <class A>
__global__ __launch_bounds__(256) void moe_nvfp4_shared_gate_up_kernel(
    const A* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    float* __restrict__ activation) {
    const int row_base = static_cast<int>(blockIdx.x) * kMoeSharedTile;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;

    for (int row = row_base + warp; row < row_base + kMoeSharedTile; row += 8) {
        const std::uint32_t* gate_words = reinterpret_cast<const std::uint32_t*>(
            control + static_cast<std::int64_t>(kMoeExperts + 1 + row) * kMoeHidden);
        const std::uint32_t* up_words = reinterpret_cast<const std::uint32_t*>(
            control + static_cast<std::int64_t>(kMoeExperts + 1 + kMoeIntermediate + row) *
                          kMoeHidden);
        const float gate = shared_control_dot<A>(gate_words, x, kMoeHidden / 2);
        const float up   = shared_control_dot<A>(up_words, x, kMoeHidden / 2);
        if (lane == 0) activation[row] = silu(gate) * up;
    }
}

template <class A>
__global__ __launch_bounds__(256) void moe_nvfp4_shared_down_kernel(
    const __nv_bfloat16* __restrict__ shared_down, const float* __restrict__ shared_gate,
    const float* __restrict__ activation, A* __restrict__ destination) {
    const int row_base = static_cast<int>(blockIdx.x) * kMoeSharedTile;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;

    __shared__ float route_activation[kMoeIntermediate];
    for (int i = static_cast<int>(threadIdx.x); i < kMoeIntermediate; i += 256) {
        route_activation[i] = activation[i];
    }
    __syncthreads();

    const float gate = shared_gate[0];
    for (int row = row_base + warp; row < row_base + kMoeSharedTile; row += 8) {
        const std::uint32_t* weight_words = reinterpret_cast<const std::uint32_t*>(
            shared_down + static_cast<std::int64_t>(row) * kMoeIntermediate);
        const float value =
            bf16_warp_dot_f32(weight_words, route_activation, kMoeIntermediate / 2);
        if (lane == 0) {
            destination[row] = MoeAct<A>::from_float(
                MoeAct<A>::to_float(destination[row]) + gate * value);
        }
    }
}

} // namespace

void moe_nvfp4_shared_launch(const Tensor& x, const void* control, const void* shared_down,
                             MoeNvfp4Scratch scratch, const MoeNvfp4RouteOutputs& route,
                             Tensor& destination, cudaStream_t stream) {
    const bool parallel = x.ne[1] == 1 && scratch.activation != nullptr;
    if (x.dtype == DType::FP16) {
        if (parallel) {
            const dim3 gate_up_grid(kMoeIntermediate / kMoeSharedTile);
            moe_nvfp4_shared_gate_up_kernel<__half><<<gate_up_grid, 256, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const __nv_bfloat16*>(control),
                scratch.activation);
            const dim3 down_grid(kMoeHidden / kMoeSharedTile);
            moe_nvfp4_shared_down_kernel<__half><<<down_grid, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(shared_down), route.shared_gate,
                scratch.activation, static_cast<__half*>(destination.data));
        } else {
            moe_nvfp4_shared_kernel<__half><<<x.ne[1], 256, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const __nv_bfloat16*>(control),
                static_cast<const __nv_bfloat16*>(shared_down), route.shared_gate,
                static_cast<__half*>(destination.data));
        }
    } else {
        if (parallel) {
            const dim3 gate_up_grid(kMoeIntermediate / kMoeSharedTile);
            moe_nvfp4_shared_gate_up_kernel<__nv_bfloat16><<<gate_up_grid, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(control), scratch.activation);
            const dim3 down_grid(kMoeHidden / kMoeSharedTile);
            moe_nvfp4_shared_down_kernel<__nv_bfloat16><<<down_grid, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(shared_down), route.shared_gate,
                scratch.activation, static_cast<__nv_bfloat16*>(destination.data));
        } else {
            moe_nvfp4_shared_kernel<__nv_bfloat16><<<x.ne[1], 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(control),
                static_cast<const __nv_bfloat16*>(shared_down), route.shared_gate,
                static_cast<__nv_bfloat16*>(destination.data));
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
