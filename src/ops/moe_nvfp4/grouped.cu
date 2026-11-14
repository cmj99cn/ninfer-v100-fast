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

} // namespace ninfer::ops::detail
