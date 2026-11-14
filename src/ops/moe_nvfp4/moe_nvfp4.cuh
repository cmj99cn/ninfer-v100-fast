#pragma once

#include "core/tensor.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kMoeExperts      = 512;
inline constexpr int kMoeTopK         = 10;
inline constexpr int kMoeHidden       = 2560;
inline constexpr int kMoeIntermediate = 640;

template <class A>
struct MoeAct;

template <>
struct MoeAct<__nv_bfloat16> {
    __device__ __forceinline__ static float to_float(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    __device__ __forceinline__ static __nv_bfloat16 from_float(float value) {
        return __float2bfloat16_rn(value);
    }
    __device__ __forceinline__ static float2 pair_to_float2(std::uint32_t bits) {
        return bf16x2_bits_to_float2(bits);
    }
};

template <>
struct MoeAct<__half> {
    __device__ __forceinline__ static float to_float(__half value) { return __half2float(value); }
    __device__ __forceinline__ static __half from_float(float value) {
        return __float2half_rn(value);
    }
    __device__ __forceinline__ static float2 pair_to_float2(std::uint32_t bits) {
        return __half22float2(half2_from_bits(bits));
    }
};

__device__ __forceinline__ std::size_t batched_nvfp4_scale_offset(std::int32_t row,
                                                                   std::int32_t group,
                                                                   std::int32_t columns) {
    return (static_cast<std::size_t>(row / 128) * (columns / 64) +
            static_cast<std::size_t>(group / 4)) * 512U +
           static_cast<std::size_t>(row % 32) * 16U +
           static_cast<std::size_t>((row % 128) / 32) * 4U + static_cast<std::size_t>(group % 4);
}

template <class A>
__device__ __forceinline__ float batched_nvfp4_warp_dot(const std::uint8_t* codes_row,
                                                         const std::uint8_t* matrix_scales,
                                                         float inverse_divisor, std::int32_t row,
                                                         std::int32_t groups,
                                                         const A* activation) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float accumulator = 0.0F;
    for (int group = lane; group < groups; group += 32) {
        const float scale =
            decode_nvfp4_e4m3(matrix_scales[batched_nvfp4_scale_offset(row, group, groups * 16)]) *
            inverse_divisor;
        const std::uint64_t codes_word =
            load_vec<std::uint64_t>(codes_row + static_cast<std::size_t>(row) * groups * 8U +
                                    static_cast<std::size_t>(group) * 8U);
        const uint4 low  = load_vec<uint4>(activation + 16 * group);
        const uint4 high = load_vec<uint4>(activation + 16 * group + 8);
        const std::uint32_t words[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            const float2 code =
                decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(codes_word >> (8 * byte)));
            const float2 value = MoeAct<A>::pair_to_float2(words[byte]);
            accumulator = fmaf(code.x * scale, value.x, accumulator);
            accumulator = fmaf(code.y * scale, value.y, accumulator);
        }
    }
    return warp_reduce_sum(accumulator);
}

__device__ __forceinline__ float batched_nvfp4_warp_dot_f32(const std::uint8_t* codes_row,
                                                             const std::uint8_t* matrix_scales,
                                                             float inverse_divisor, std::int32_t row,
                                                             std::int32_t groups,
                                                             const float* activation) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float accumulator = 0.0F;
    for (int group = lane; group < groups; group += 32) {
        const float scale =
            decode_nvfp4_e4m3(matrix_scales[batched_nvfp4_scale_offset(row, group, groups * 16)]) *
            inverse_divisor;
        const std::uint64_t codes_word =
            load_vec<std::uint64_t>(codes_row + static_cast<std::size_t>(row) * groups * 8U +
                                    static_cast<std::size_t>(group) * 8U);
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            const float2 code =
                decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(codes_word >> (8 * byte)));
            accumulator = fmaf(code.x * scale, activation[16 * group + 2 * byte], accumulator);
            accumulator = fmaf(code.y * scale, activation[16 * group + 2 * byte + 1], accumulator);
        }
    }
    return warp_reduce_sum(accumulator);
}

__device__ __forceinline__ float bf16_warp_dot(const std::uint32_t* weight_words,
                                               const std::uint32_t* activation_words,
                                               std::int32_t pairs) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float accumulator = 0.0F;
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 weight = bf16x2_bits_to_float2(weight_words[pair]);
        const float2 value  = bf16x2_bits_to_float2(activation_words[pair]);
        accumulator = fmaf(weight.x, value.x, accumulator);
        accumulator = fmaf(weight.y, value.y, accumulator);
    }
    return warp_reduce_sum(accumulator);
}

__device__ __forceinline__ float bf16_f16_warp_dot(const std::uint32_t* weight_words,
                                                    const std::uint32_t* activation_words,
                                                    std::int32_t pairs) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float accumulator = 0.0F;
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 weight = bf16x2_bits_to_float2(weight_words[pair]);
        const float2 value  = MoeAct<__half>::pair_to_float2(activation_words[pair]);
        accumulator = fmaf(weight.x, value.x, accumulator);
        accumulator = fmaf(weight.y, value.y, accumulator);
    }
    return warp_reduce_sum(accumulator);
}

__device__ __forceinline__ float bf16_warp_dot_f32(const std::uint32_t* weight_words,
                                                    const float* activation, std::int32_t pairs) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float accumulator = 0.0F;
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 weight = bf16x2_bits_to_float2(weight_words[pair]);
        accumulator = fmaf(weight.x, activation[2 * pair], accumulator);
        accumulator = fmaf(weight.y, activation[2 * pair + 1], accumulator);
    }
    return warp_reduce_sum(accumulator);
}

} // namespace ninfer::ops::detail
