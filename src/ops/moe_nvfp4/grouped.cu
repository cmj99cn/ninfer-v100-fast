#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <type_traits>

#include <cstdint>
#include <stdexcept>

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

inline constexpr int kMoeDecodeThreads    = 128;
inline constexpr int kMoeDecodeWarps      = kMoeDecodeThreads / 32;
inline constexpr int kMoeDecodeGateUpTile = kMoeDecodeWarps;
inline constexpr int kMoeDecodeDownTile   = 2 * kMoeDecodeWarps;
static_assert(kMoeIntermediate % kMoeDecodeGateUpTile == 0);
static_assert(kMoeHidden % kMoeDecodeDownTile == 0);

template <class A, int R>
__device__ __forceinline__ void cached_gate_up_rows(
    const std::uint8_t* __restrict__ gate_codes, const std::uint8_t* __restrict__ gate_scales,
    float inverse_gate, const std::uint8_t* __restrict__ up_codes,
    const std::uint8_t* __restrict__ up_scales, float inverse_up, const int (&rows)[R],
    const A* __restrict__ activation, const float2* __restrict__ decode_lut,
    const float* __restrict__ scale_lut, float (&gate)[R], float (&up)[R]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float gate_acc[R];
    float up_acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        gate_acc[r] = 0.0F;
        up_acc[r]   = 0.0F;
    }
    for (int group = lane; group < kMoeHidden / 16; group += 32) {
        const uint4 low  = load_vec<uint4>(activation + 16 * group);
        const uint4 high = load_vec<uint4>(activation + 16 * group + 8);
        const std::uint32_t words[8] = {low.x, low.y, low.z, low.w,
                                        high.x, high.y, high.z, high.w};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = rows[r];
            const float gate_scale =
                scale_lut[gate_scales[batched_nvfp4_scale_offset(row, group, kMoeHidden)]] *
                inverse_gate;
            const float up_scale =
                scale_lut[up_scales[batched_nvfp4_scale_offset(row, group, kMoeHidden)]] *
                inverse_up;
            const std::uint64_t gate_word = load_vec<std::uint64_t>(
                gate_codes + static_cast<std::size_t>(row) * (kMoeHidden / 2U) +
                static_cast<std::size_t>(group) * 8U);
            const std::uint64_t up_word = load_vec<std::uint64_t>(
                up_codes + static_cast<std::size_t>(row) * (kMoeHidden / 2U) +
                static_cast<std::size_t>(group) * 8U);
#pragma unroll
            for (int byte = 0; byte < 8; ++byte) {
                const float2 gate_code =
                    decode_lut[static_cast<std::uint8_t>(gate_word >> (8 * byte))];
                const float2 up_code =
                    decode_lut[static_cast<std::uint8_t>(up_word >> (8 * byte))];
                const float2 value = MoeAct<A>::pair_to_float2(words[byte]);
                gate_acc[r] = fmaf(gate_code.x * gate_scale, value.x, gate_acc[r]);
                gate_acc[r] = fmaf(gate_code.y * gate_scale, value.y, gate_acc[r]);
                up_acc[r] = fmaf(up_code.x * up_scale, value.x, up_acc[r]);
                up_acc[r] = fmaf(up_code.y * up_scale, value.y, up_acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        gate[r] = warp_reduce_sum(gate_acc[r]);
        up[r]   = warp_reduce_sum(up_acc[r]);
    }
}

template <int R>
__device__ __forceinline__ void cached_down_rows(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    float inverse, const int (&rows)[R], const float* __restrict__ activation,
    const float2* __restrict__ decode_lut, const float* __restrict__ scale_lut,
    float (&value)[R]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float accumulator[R];
#pragma unroll
    for (int r = 0; r < R; ++r) accumulator[r] = 0.0F;
    for (int group = lane; group < kMoeIntermediate / 16; group += 32) {
        const auto* values = reinterpret_cast<const float4*>(activation + 16 * group);
        const float4 a0 = load_vec<float4>(values);
        const float4 a1 = load_vec<float4>(values + 1);
        const float4 a2 = load_vec<float4>(values + 2);
        const float4 a3 = load_vec<float4>(values + 3);
        const float av[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                              a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = rows[r];
            const float scale =
                scale_lut[scales[batched_nvfp4_scale_offset(row, group, kMoeIntermediate)]] *
                inverse;
            const std::uint64_t codes_word = load_vec<std::uint64_t>(
                codes + static_cast<std::size_t>(row) * (kMoeIntermediate / 2U) +
                static_cast<std::size_t>(group) * 8U);
#pragma unroll
            for (int byte = 0; byte < 8; ++byte) {
                const float2 code = decode_lut[static_cast<std::uint8_t>(codes_word >> (8 * byte))];
                accumulator[r] = fmaf(code.x * scale, av[2 * byte], accumulator[r]);
                accumulator[r] = fmaf(code.y * scale, av[2 * byte + 1], accumulator[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) value[r] = warp_reduce_sum(accumulator[r]);
}

template <class A>
__global__ __launch_bounds__(kMoeDecodeThreads) void moe_nvfp4_cached_gate_up_kernel(
    const A* __restrict__ x, const std::uint8_t* __restrict__ slots,
    std::uint64_t slot_bytes, const std::int32_t* __restrict__ lookup,
    const float* __restrict__ divisors, const std::int32_t* __restrict__ ids,
    float* __restrict__ activation, std::int32_t expert_begin, std::int32_t expert_end,
    MoeNvfp4Probe probe) {
    constexpr int kRows = kMoeDecodeGateUpTile / kMoeDecodeWarps;
    const int token     = static_cast<int>(blockIdx.z);
    const int route_idx = static_cast<int>(blockIdx.x) + token * kMoeTopK;
    const int row_base  = static_cast<int>(blockIdx.y) * kMoeDecodeGateUpTile;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int lane      = static_cast<int>(threadIdx.x) & 31;

    __shared__ float2 decode_lut[256];
    __shared__ float scale_lut[256];
    for (int index = static_cast<int>(threadIdx.x); index < 256;
         index += static_cast<int>(blockDim.x)) {
        decode_lut[index] = decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(index));
        scale_lut[index]  = decode_nvfp4_e4m3(static_cast<std::uint8_t>(index));
    }
    __syncthreads();

    if (probe.active_count != nullptr && blockIdx.x == 0 && blockIdx.y == 0 &&
        blockIdx.z == 0 && threadIdx.x == 0) {
        moe_nvfp4_probe_routes(ids, kMoeTopK, lookup, expert_begin, expert_end, probe.active_ids,
                               probe.miss_ids, probe.active_count, probe.miss_count);
    }

    const std::int32_t global = ids[route_idx];
    const std::int32_t slot   = lookup[global];
    if (slot < 0) { return; }

    const std::uint8_t* base    = slots + static_cast<std::size_t>(slot) * slot_bytes;
    const float* slot_divisors  = divisors + static_cast<std::size_t>(slot) * 3;
    const float inverse_gate    = 1.0F / slot_divisors[0];
    const float inverse_up      = 1.0F / slot_divisors[1];

    int rows[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) rows[r] = row_base + warp + r * kMoeDecodeWarps;
    float gate[kRows];
    float up[kRows];
    cached_gate_up_rows<A, kRows>(base, base + kMoeGateScale, inverse_gate, base + kMoeUpCode,
                                  base + kMoeUpScale, inverse_up, rows,
                                  x + static_cast<std::size_t>(token) * kMoeHidden, decode_lut,
                                  scale_lut, gate, up);
    if (lane == 0) {
        float* activation_column = activation + static_cast<std::size_t>(route_idx) * kMoeIntermediate;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            activation_column[rows[r]] = silu(gate[r]) * up[r];
        }
    }
}

template <class A>
__global__ __launch_bounds__(kMoeDecodeThreads) void moe_nvfp4_cached_down_kernel(
    const std::uint8_t* __restrict__ slots, std::uint64_t slot_bytes,
    const std::int32_t* __restrict__ lookup, const float* __restrict__ divisors,
    const std::int32_t* __restrict__ ids, const float* __restrict__ coefficients,
    const float* __restrict__ activation, A* __restrict__ destination) {
    constexpr int kRows = kMoeDecodeDownTile / kMoeDecodeWarps;
    const int token    = static_cast<int>(blockIdx.y);
    const int row_base = static_cast<int>(blockIdx.x) * kMoeDecodeDownTile;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;

    __shared__ float2 decode_lut[256];
    __shared__ float scale_lut[256];
    for (int index = static_cast<int>(threadIdx.x); index < 256;
         index += static_cast<int>(blockDim.x)) {
        decode_lut[index] = decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(index));
        scale_lut[index]  = decode_nvfp4_e4m3(static_cast<std::uint8_t>(index));
    }
    __syncthreads();

    int rows[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) rows[r] = row_base + warp + r * kMoeDecodeWarps;
    float accumulator[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) accumulator[r] = 0.0F;

    const std::size_t route_base = static_cast<std::size_t>(token) * kMoeTopK;
    for (int route = 0; route < kMoeTopK; ++route) {
        const std::int32_t global = ids[route_base + static_cast<std::size_t>(route)];
        const std::int32_t slot   = lookup[global];
        if (slot < 0) { continue; }
        const float coefficient = coefficients[route_base + static_cast<std::size_t>(route)];
        const std::uint8_t* base = slots + static_cast<std::size_t>(slot) * slot_bytes;
        const float* slot_divisors = divisors + static_cast<std::size_t>(slot) * 3;
        const float inverse_down   = 1.0F / slot_divisors[2];
        const float* source = activation + (route_base + static_cast<std::size_t>(route)) *
                                              kMoeIntermediate;
        float value[kRows];
        cached_down_rows<kRows>(base + kMoeDownCode, base + kMoeDownScale, inverse_down, rows,
                                source, decode_lut, scale_lut, value);
#pragma unroll
        for (int r = 0; r < kRows; ++r) accumulator[r] += coefficient * value[r];
    }

    A* destination_column = destination + static_cast<std::size_t>(token) * kMoeHidden;
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = rows[r];
            destination_column[row] = MoeAct<A>::from_float(
                accumulator[r] + MoeAct<A>::to_float(destination_column[row]));
        }
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

void moe_nvfp4_cached_gate_up_launch(const Tensor& x, const std::uint8_t* slots,
                                     std::uint64_t slot_bytes, const std::int32_t* lookup,
                                     const float* divisors, MoeNvfp4Scratch scratch,
                                     const MoeNvfp4RouteOutputs& route,
                                     const MoeNvfp4Probe* probe, cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > kMoeMaxSmallT || scratch.activation == nullptr ||
        route.ids == nullptr) {
        throw std::invalid_argument("moe_nvfp4: the cached gate_up path needs one to eight tokens");
    }
    const MoeNvfp4Probe probe_value = probe != nullptr ? *probe : MoeNvfp4Probe{};
    const std::int32_t expert_begin = probe_value.expert_begin;
    const std::int32_t expert_end   = probe_value.expert_end;
    const dim3 gate_up_grid(kMoeTopK, kMoeIntermediate / kMoeDecodeGateUpTile, x.ne[1]);
    if (x.dtype == DType::FP16) {
        moe_nvfp4_cached_gate_up_kernel<__half><<<gate_up_grid, kMoeDecodeThreads, 0, stream>>>(
            static_cast<const __half*>(x.data), slots, slot_bytes, lookup, divisors, route.ids,
            scratch.activation, expert_begin, expert_end, probe_value);
    } else {
        moe_nvfp4_cached_gate_up_kernel<__nv_bfloat16>
            <<<gate_up_grid, kMoeDecodeThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data), slots, slot_bytes, lookup, divisors,
                route.ids, scratch.activation, expert_begin, expert_end, probe_value);
    }
    CUDA_CHECK(cudaGetLastError());
}

void moe_nvfp4_cached_down_launch(const std::uint8_t* slots, std::uint64_t slot_bytes,
                                  const std::int32_t* lookup, const float* divisors,
                                  MoeNvfp4Scratch scratch, const MoeNvfp4RouteOutputs& route,
                                  Tensor& destination, cudaStream_t stream) {
    if (destination.ne[1] < 1 || destination.ne[1] > kMoeMaxSmallT ||
        scratch.activation == nullptr || route.ids == nullptr) {
        throw std::invalid_argument("moe_nvfp4: the cached down path needs one to eight tokens");
    }
    const dim3 down_grid(kMoeHidden / kMoeDecodeDownTile, destination.ne[1]);
    if (destination.dtype == DType::FP16) {
        moe_nvfp4_cached_down_kernel<__half><<<down_grid, kMoeDecodeThreads, 0, stream>>>(
            slots, slot_bytes, lookup, divisors, route.ids, route.coefficients, scratch.activation,
            static_cast<__half*>(destination.data));
    } else {
        moe_nvfp4_cached_down_kernel<__nv_bfloat16><<<down_grid, kMoeDecodeThreads, 0, stream>>>(
            slots, slot_bytes, lookup, divisors, route.ids, route.coefficients, scratch.activation,
            static_cast<__nv_bfloat16*>(destination.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

void moe_nvfp4_cached_launch(const Tensor& x, const std::uint8_t* slots,
                             std::uint64_t slot_bytes, std::int32_t slot_count,
                             const std::int32_t* lookup, const float* divisors,
                             MoeNvfp4Scratch scratch, const MoeNvfp4RouteOutputs& route,
                             Tensor& destination, cudaStream_t stream) {
    (void)slot_count;
    if (x.ne[1] >= 1 && x.ne[1] <= kMoeMaxSmallT && scratch.activation != nullptr) {
        moe_nvfp4_cached_gate_up_launch(x, slots, slot_bytes, lookup, divisors, scratch, route,
                                        nullptr, stream);
        moe_nvfp4_cached_down_launch(slots, slot_bytes, lookup, divisors, scratch, route,
                                     destination, stream);
        return;
    }
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
