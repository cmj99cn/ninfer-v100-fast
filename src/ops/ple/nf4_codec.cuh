#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

static __constant__ float kNf4Codebook[16] = {
    -1.0F,
    -0.6961928009986877F,
    -0.5250730514526367F,
    -0.39491748809814453F,
    -0.28444138169288635F,
    -0.18477343022823334F,
    -0.09105003625154495F,
    0.0F,
    0.07958029955625534F,
    0.16093020141124725F,
    0.24611230194568634F,
    0.33791524171829224F,
    0.44070982933044434F,
    0.5626170039176941F,
    0.7229568362236023F,
    1.0F,
};

__device__ __forceinline__ float2 decode_nf4x2(std::uint8_t storage) {
    float2 result;
    result.x = kNf4Codebook[storage & 0x0FU];
    result.y = kNf4Codebook[storage >> 4];
    return result;
}

} // namespace ninfer::ops::detail
