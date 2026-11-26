// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime: the fixed theta=1e7 frequency table is
// replaced by an explicit theta argument so the registered Flash-Next companion's config value
// (1e4) drives DFlash2 context RoPE. The angle reduction and split-half pairing are unchanged.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

// D128 split-half RoPE sin/cos for pair index in [0,64), evaluating
// theta^(-2*pair/128) at the absolute position. The angle is reduced modulo 2*pi in double before
// the float sincos so large positions keep usable precision.
__device__ __forceinline__ void dflash_rope_sincos(float theta, std::int32_t position, int pair,
                                                   float* sine, float* cosine) {
    constexpr double kInvTwoPi = 1.59154943091895336e-01;
    constexpr double kTwoPi    = 6.28318530717958648e+00;
    const double frequency =
        pow(static_cast<double>(theta), -static_cast<double>(pair) / 64.0);
    const double angle  = static_cast<double>(position) * frequency;
    const double turns  = angle * kInvTwoPi;
    const float reduced = static_cast<float>(angle - nearbyint(turns) * kTwoPi);
    sincosf(reduced, sine, cosine);
}

} // namespace ninfer::ops
