#pragma once

// Activation element traits shared by the Op kernels whose weight codec is independent of the
// model activation dtype. BF16 and FP16 both occupy two bytes, so a kernel that only changes its
// element load/store (and the fp16 staging cast for tensor-core tiles) can serve both dtypes from
// one body.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

template <class T>
struct ActivationTraits;

template <>
struct ActivationTraits<__nv_bfloat16> {
    __device__ static float load(const __nv_bfloat16* p) { return __bfloat162float(*p); }
    __device__ static half as_half(const __nv_bfloat16* p) {
        return __float2half(__bfloat162float(*p));
    }
    __device__ static void store(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
    __device__ static std::uint32_t pack2(float lo, float hi) { return pack_bf16x2(lo, hi); }
};

template <>
struct ActivationTraits<half> {
    __device__ static float load(const half* p) { return __half2float(*p); }
    __device__ static half as_half(const half* p) { return *p; }
    __device__ static void store(half* p, float v) { *p = __float2half(v); }
    __device__ static std::uint32_t pack2(float lo, float hi) {
        const __half2 packed = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<const std::uint32_t*>(&packed);
    }
};

} // namespace ninfer::ops
