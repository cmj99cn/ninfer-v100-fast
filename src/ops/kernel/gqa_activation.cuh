#pragma once

// Activation element traits for the GQA kernels whose KV cache codec is independent of the
// activation dtype. The INT8/IQ4_NL cache path stores one code byte plus a per-group scale per
// element regardless of whether the model's hidden stream is BF16 or FP16, so the same kernel body
// serves both; only element load/store and the fp16 staging cast change.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

template <class T>
struct GqaActivation;

template <>
struct GqaActivation<__nv_bfloat16> {
    __device__ static float load(const __nv_bfloat16* p) { return __bfloat162float(*p); }
    __device__ static half as_half(const __nv_bfloat16* p) {
        return __float2half(__bfloat162float(*p));
    }
    __device__ static void store(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
    __device__ static std::uint32_t pack2(float lo, float hi) { return pack_bf16x2(lo, hi); }
};

template <>
struct GqaActivation<half> {
    __device__ static float load(const half* p) { return __half2float(*p); }
    __device__ static half as_half(const half* p) { return *p; }
    __device__ static void store(half* p, float v) { *p = __float2half(v); }
    __device__ static std::uint32_t pack2(float lo, float hi) {
        const __half2 packed = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<const std::uint32_t*>(&packed);
    }
};

template <class T>
struct GqaAppendInputOf {
    static constexpr bool writes_cache = true;
    const T* k;
    const T* v;
};

using GqaAppendInput = GqaAppendInputOf<__nv_bfloat16>;

} // namespace ninfer::ops
