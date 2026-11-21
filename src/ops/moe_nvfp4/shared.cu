#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <type_traits>

#include <cstdint>


namespace ninfer::ops::detail {
namespace {

// One pack = 8 elements. A BF16 weight pack is four uint32 words (16 B); an activation pack is
// one uint4 (8 x 2-byte A) or two float4 (8 x float). K must be a multiple of 8.
template <class A>
__device__ __forceinline__ float shared_pack_dot(const std::uint32_t* weight_words,
                                                 const A* activation, std::int32_t k_elements,
                                                 int lane) {
    const auto* activation_words = reinterpret_cast<const std::uint32_t*>(activation);
    float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const std::int32_t packs     = k_elements / 8;
    for (std::int32_t pack = lane; pack < packs; pack += 32) {
        const uint4 words       = load_vec<uint4>(weight_words + 4 * pack);
        const uint4 value_words = load_vec<uint4>(activation_words + 4 * pack);
        const std::uint32_t weights[4] = {words.x, words.y, words.z, words.w};
        const std::uint32_t values[4]  = {value_words.x, value_words.y, value_words.z,
                                          value_words.w};
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 weight = bf16x2_bits_to_float2(weights[pair]);
            const float2 value  = MoeAct<A>::pair_to_float2(values[pair]);
            accumulator[(2 * pair) & 3] =
                fmaf(weight.x, value.x, accumulator[(2 * pair) & 3]);
            accumulator[(2 * pair + 1) & 3] =
                fmaf(weight.y, value.y, accumulator[(2 * pair + 1) & 3]);
        }
    }
    float total = 0.0F;
#pragma unroll
    for (int chain = 0; chain < 4; ++chain) { total += accumulator[chain]; }
    return warp_reduce_sum(total);
}

__device__ __forceinline__ float shared_pack_dot_f32(const std::uint32_t* weight_words,
                                                     const float* activation,
                                                     std::int32_t k_elements, int lane) {
    float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const std::int32_t packs     = k_elements / 8;
    for (std::int32_t pack = lane; pack < packs; pack += 32) {
        const uint4 words       = load_vec<uint4>(weight_words + 4 * pack);
        const float4 value_low  = load_vec<float4>(activation + 8 * pack);
        const float4 value_high = load_vec<float4>(activation + 8 * pack + 4);
        const std::uint32_t weights[4] = {words.x, words.y, words.z, words.w};
        const float values[8] = {value_low.x, value_low.y, value_low.z, value_low.w,
                                 value_high.x, value_high.y, value_high.z, value_high.w};
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 weight = bf16x2_bits_to_float2(weights[pair]);
            accumulator[(2 * pair) & 3] =
                fmaf(weight.x, values[2 * pair], accumulator[(2 * pair) & 3]);
            accumulator[(2 * pair + 1) & 3] =
                fmaf(weight.y, values[2 * pair + 1], accumulator[(2 * pair + 1) & 3]);
        }
    }
    float total = 0.0F;
#pragma unroll
    for (int chain = 0; chain < 4; ++chain) { total += accumulator[chain]; }
    return warp_reduce_sum(total);
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
        const float value = shared_pack_dot<A>(weight_words, x_column, kMoeHidden, lane);
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
        const float value = shared_pack_dot_f32(weight_words, activation, kMoeIntermediate, lane);
        if (lane == 0) {
            destination_column[row] = MoeAct<A>::from_float(
                MoeAct<A>::to_float(destination_column[row]) + gate * value);
        }
    }
}

// Decode T == 1: one warp per gate/up dot; two dots per intermediate pair; partials through
// shared memory. Each CTA covers four pairs (eight warps), doubling SM occupancy vs pair-per-warp.
template <class A>
__global__ __launch_bounds__(256) void moe_nvfp4_shared_gate_up_kernel(
    const A* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    float* __restrict__ activation) {
    const auto* x_words = reinterpret_cast<const std::uint32_t*>(x);
    __shared__ float partial[4][2];

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    const int pair   = static_cast<int>(blockIdx.x) * 4 + (warp >> 1);
    const bool is_up = (warp & 1) != 0;
    if (pair >= kMoeIntermediate) { return; }

    const std::uint32_t* weight_words = reinterpret_cast<const std::uint32_t*>(
        control + static_cast<std::int64_t>(kMoeExperts + 1 + pair + (is_up ? kMoeIntermediate : 0)) *
                      kMoeHidden);

    float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (std::int32_t pack = lane; pack < kMoeHidden / 8; pack += 32) {
        const uint4 weight_pack = load_vec<uint4>(weight_words + 4 * pack);
        const uint4 value_pack  = __ldg(reinterpret_cast<const uint4*>(x_words) + pack);
        const std::uint32_t weights[4] = {weight_pack.x, weight_pack.y, weight_pack.z,
                                          weight_pack.w};
        const std::uint32_t values[4]  = {value_pack.x, value_pack.y, value_pack.z, value_pack.w};
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 weight = bf16x2_bits_to_float2(weights[pair]);
            const float2 value  = MoeAct<A>::pair_to_float2(values[pair]);
            accumulator[(2 * pair) & 3] =
                fmaf(weight.x, value.x, accumulator[(2 * pair) & 3]);
            accumulator[(2 * pair + 1) & 3] =
                fmaf(weight.y, value.y, accumulator[(2 * pair + 1) & 3]);
        }
    }
    float total = 0.0F;
#pragma unroll
    for (int chain = 0; chain < 4; ++chain) { total += accumulator[chain]; }
    total = warp_reduce_sum(total);
    if (lane == 0) { partial[warp >> 1][warp & 1] = total; }
    __syncthreads();
    if ((warp & 1) == 0 && lane == 0) {
        activation[pair] = silu(partial[warp >> 1][0]) * partial[warp >> 1][1];
    }
}

// Decode T == 1: one warp per four output rows, shared-staged fp32 activation, four
// accumulator sets held in registers so each lane keeps four independent weight packs in flight.
template <class A>
__global__ __launch_bounds__(256) void moe_nvfp4_shared_down_kernel(
    const __nv_bfloat16* __restrict__ shared_down, const float* __restrict__ shared_gate,
    const float* __restrict__ activation, A* __restrict__ destination) {
    const float gate = shared_gate[0];

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    const int row0 = static_cast<int>(blockIdx.x) * 32 + warp * 4;

    const std::uint32_t* weight_words =
        reinterpret_cast<const std::uint32_t*>(shared_down);
    float accumulator[4][4] = {};
#pragma unroll
    for (int pack = lane; pack < kMoeIntermediate / 8; pack += 32) {
        const float4 value_low  = __ldg(reinterpret_cast<const float4*>(activation) + 2 * pack);
        const float4 value_high = __ldg(reinterpret_cast<const float4*>(activation) + 2 * pack + 1);
        const float values[8] = {value_low.x, value_low.y, value_low.z, value_low.w,
                                 value_high.x, value_high.y, value_high.z, value_high.w};
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const uint4 words = load_vec<uint4>(
                weight_words + 4 * pack +
                static_cast<std::int64_t>(row0 + row) * (kMoeIntermediate / 2));
            const std::uint32_t bits[4] = {words.x, words.y, words.z, words.w};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float2 weight = bf16x2_bits_to_float2(bits[pair]);
                accumulator[row][(2 * pair) & 3] =
                    fmaf(weight.x, values[2 * pair], accumulator[row][(2 * pair) & 3]);
                accumulator[row][(2 * pair + 1) & 3] =
                    fmaf(weight.y, values[2 * pair + 1], accumulator[row][(2 * pair + 1) & 3]);
            }
        }
    }
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < 4; ++chain) { total += accumulator[row][chain]; }
        total = warp_reduce_sum(total);
        if (lane == 0) {
            destination[row0 + row] = MoeAct<A>::from_float(
                MoeAct<A>::to_float(destination[row0 + row]) + gate * total);
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
            const dim3 gate_up_grid(kMoeIntermediate / 4);
            moe_nvfp4_shared_gate_up_kernel<__half><<<gate_up_grid, 256, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const __nv_bfloat16*>(control),
                scratch.activation);
            const dim3 down_grid(kMoeHidden / 32);
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
            const dim3 gate_up_grid(kMoeIntermediate / 4);
            moe_nvfp4_shared_gate_up_kernel<__nv_bfloat16><<<gate_up_grid, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(control), scratch.activation);
            const dim3 down_grid(kMoeHidden / 32);
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
