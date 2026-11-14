#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail {

template <class A>
struct HcAct;

template <>
struct HcAct<__nv_bfloat16> {
    __device__ __forceinline__ static float to_float(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    __device__ __forceinline__ static __nv_bfloat16 from_float(float value) {
        return __float2bfloat16_rn(value);
    }
};

template <>
struct HcAct<__half> {
    __device__ __forceinline__ static float to_float(__half value) { return __half2float(value); }
    __device__ __forceinline__ static __half from_float(float value) {
        return __float2half_rn(value);
    }
};

} // namespace ninfer::ops::detail
