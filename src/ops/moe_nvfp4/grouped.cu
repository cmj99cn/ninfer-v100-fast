#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <type_traits>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class A>
__global__ __launch_bounds__(256) void moe_nvfp4_grouped_kernel(
    const A* __restrict__ x,
    const std::uint8_t* __restrict__ gate_up_codes, const std::uint8_t* __restrict__ gate_up_scales,
    const float* __restrict__ gate_up_divisors, const std::uint8_t* __restrict__ down_codes,
    const std::uint8_t* __restrict__ down_scales, const float* __restrict__ down_divisors,
    const std::int32_t* __restrict__ ids, const float* __restrict__ coefficients,
    std::int32_t expert_begin, std::int32_t experts, A* __restrict__ destination) {
    const int token = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;

    __shared__ float gate_up[kMoeIntermediate * 2];
    __shared__ float activation[kMoeIntermediate];
    __shared__ float accumulator[kMoeHidden];

    for (int i = static_cast<int>(threadIdx.x); i < kMoeHidden; i += 256) accumulator[i] = 0.0F;

    const A* x_column = x + static_cast<std::int64_t>(token) * kMoeHidden;

    for (int slot = 0; slot < kMoeTopK; ++slot) {
        const std::int32_t global     = ids[slot + kMoeTopK * token];
        const std::int32_t expert     = global - expert_begin;
        if (expert < 0 || expert >= experts) { continue; }
        const float coefficient       = coefficients[slot + kMoeTopK * token];
        const std::uint8_t* codes     = gate_up_codes +
                                        static_cast<std::size_t>(expert) * kMoeIntermediate * 2 * kMoeHidden / 2U;
        const std::uint8_t* scales    = gate_up_scales +
                                        static_cast<std::size_t>(expert) * kMoeIntermediate * 2 * kMoeHidden / 16U;
        const float inverse_divisor   = 1.0F / gate_up_divisors[expert];

        for (int row = warp; row < kMoeIntermediate * 2; row += 8) {
            const float value = batched_nvfp4_warp_dot<A>(codes, scales, inverse_divisor, row,
                                                          kMoeHidden / 16, x_column);
            if (lane == 0) gate_up[row] = value;
        }
        __syncthreads();
        for (int i = static_cast<int>(threadIdx.x); i < kMoeIntermediate; i += 256) {
            activation[i] = silu(gate_up[i]) * gate_up[kMoeIntermediate + i];
        }
        __syncthreads();

        const std::uint8_t* down_codes_e = down_codes +
                                           static_cast<std::size_t>(expert) * kMoeHidden * kMoeIntermediate / 2U;
        const std::uint8_t* down_scales_e = down_scales +
                                            static_cast<std::size_t>(expert) * kMoeHidden * kMoeIntermediate / 16U;
        const float down_inverse         = 1.0F / down_divisors[expert];

        for (int row = warp; row < kMoeHidden; row += 8) {
            const float value = coefficient * batched_nvfp4_warp_dot_f32(
                down_codes_e, down_scales_e, down_inverse, row, kMoeIntermediate / 16, activation);
            if (lane == 0) accumulator[row] += value;
        }
        __syncthreads();
    }

    A* destination_column = destination + static_cast<std::int64_t>(token) * kMoeHidden;
    for (int i = static_cast<int>(threadIdx.x); i < kMoeHidden; i += 256) {
        destination_column[i] = MoeAct<A>::from_float(
            accumulator[i] + MoeAct<A>::to_float(destination_column[i]));
    }
}

template <class A>
__global__ __launch_bounds__(256) void moe_nvfp4_cached_kernel(
    const A* __restrict__ x, const std::uint8_t* __restrict__ slots,
    std::uint64_t slot_bytes, const std::int32_t* __restrict__ lookup,
    const float* __restrict__ divisors, const std::int32_t* __restrict__ ids,
    const float* __restrict__ coefficients, A* __restrict__ destination) {
    const int token = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;

    __shared__ float gate_up[kMoeIntermediate * 2];
    __shared__ float activation[kMoeIntermediate];
    __shared__ float accumulator[kMoeHidden];

    for (int i = static_cast<int>(threadIdx.x); i < kMoeHidden; i += 256) accumulator[i] = 0.0F;

    const A* x_column = x + static_cast<std::int64_t>(token) * kMoeHidden;

    for (int route = 0; route < kMoeTopK; ++route) {
        const std::int32_t global = ids[route + kMoeTopK * token];
        const std::int32_t slot   = lookup[global];
        if (slot < 0) { continue; }
        const float coefficient     = coefficients[route + kMoeTopK * token];
        const std::uint8_t* base    = slots + static_cast<std::size_t>(slot) * slot_bytes;
        const float* slot_divisors  = divisors + static_cast<std::size_t>(slot) * 3;
        const float inverse_gate    = 1.0F / slot_divisors[0];
        const float inverse_up      = 1.0F / slot_divisors[1];
        const float inverse_down    = 1.0F / slot_divisors[2];

        const std::uint8_t* gate_codes  = base;
        const std::uint8_t* gate_scales = base + kMoeGateScale;
        const std::uint8_t* up_codes    = base + kMoeUpCode;
        const std::uint8_t* up_scales   = base + kMoeUpScale;
        const std::uint8_t* down_codes  = base + kMoeDownCode;
        const std::uint8_t* down_scales = base + kMoeDownScale;

        for (int row = warp; row < kMoeIntermediate; row += 8) {
            const float gate = batched_nvfp4_warp_dot<A>(gate_codes, gate_scales, inverse_gate,
                                                         row, kMoeHidden / 16, x_column);
            const float up = batched_nvfp4_warp_dot<A>(up_codes, up_scales, inverse_up, row,
                                                       kMoeHidden / 16, x_column);
            if (lane == 0) {
                gate_up[row]                    = gate;
                gate_up[kMoeIntermediate + row] = up;
            }
        }
        __syncthreads();
        for (int i = static_cast<int>(threadIdx.x); i < kMoeIntermediate; i += 256) {
            activation[i] = silu(gate_up[i]) * gate_up[kMoeIntermediate + i];
        }
        __syncthreads();

        for (int row = warp; row < kMoeHidden; row += 8) {
            const float value = coefficient * batched_nvfp4_warp_dot_f32(
                down_codes, down_scales, inverse_down, row, kMoeIntermediate / 16, activation);
            if (lane == 0) accumulator[row] += value;
        }
        __syncthreads();
    }

    A* destination_column = destination + static_cast<std::int64_t>(token) * kMoeHidden;
    for (int i = static_cast<int>(threadIdx.x); i < kMoeHidden; i += 256) {
        destination_column[i] = MoeAct<A>::from_float(
            accumulator[i] + MoeAct<A>::to_float(destination_column[i]));
    }
}

} // namespace

void moe_nvfp4_grouped_launch(const Tensor& x, MoeNvfp4BatchedPlanes gate_up,
                              MoeNvfp4BatchedPlanes down, const MoeNvfp4RouteOutputs& route,
                              std::int32_t expert_begin, std::int32_t experts,
                              Tensor& destination, cudaStream_t stream) {
    if (x.dtype == DType::FP16) {
        moe_nvfp4_grouped_kernel<__half><<<x.ne[1], 256, 0, stream>>>(
            static_cast<const __half*>(x.data), gate_up.codes, gate_up.scales, gate_up.divisors,
            down.codes, down.scales, down.divisors, route.ids, route.coefficients, expert_begin,
            experts, static_cast<__half*>(destination.data));
    } else {
        moe_nvfp4_grouped_kernel<__nv_bfloat16><<<x.ne[1], 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), gate_up.codes, gate_up.scales,
            gate_up.divisors, down.codes, down.scales, down.divisors, route.ids,
            route.coefficients, expert_begin, experts,
            static_cast<__nv_bfloat16*>(destination.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

void moe_nvfp4_cached_launch(const Tensor& x, const std::uint8_t* slots,
                             std::uint64_t slot_bytes, std::int32_t slot_count,
                             const std::int32_t* lookup, const float* divisors,
                             const MoeNvfp4RouteOutputs& route, Tensor& destination,
                             cudaStream_t stream) {
    (void)slot_count;
    if (x.dtype == DType::FP16) {
        moe_nvfp4_cached_kernel<__half><<<x.ne[1], 256, 0, stream>>>(
            static_cast<const __half*>(x.data), slots, slot_bytes, lookup, divisors, route.ids,
            route.coefficients, static_cast<__half*>(destination.data));
    } else {
        moe_nvfp4_cached_kernel<__nv_bfloat16><<<x.ne[1], 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), slots, slot_bytes, lookup, divisors,
            route.ids, route.coefficients, static_cast<__nv_bfloat16*>(destination.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
