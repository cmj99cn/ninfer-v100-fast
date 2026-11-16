#pragma once

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

inline void qsa_require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa: ") + message); }
}

inline void qsa_require_dtype(const Tensor& tensor, DType dtype, const char* label) {
    qsa_require(tensor.dtype == dtype, label);
}

inline void qsa_require_contiguous(const Tensor& tensor, const char* label) {
    qsa_require(tensor.data != nullptr && tensor.is_contiguous(), label);
}

inline bool qsa_has_valid_columns(const Tensor& valid_columns) {
    return valid_columns.data != nullptr;
}

inline bool qsa_is_activation(DType dtype) {
    return dtype == DType::BF16 || dtype == DType::FP16;
}

template <class A>
struct QsaAct;

template <>
struct QsaAct<__nv_bfloat16> {
    using pair = __nv_bfloat162;
    __device__ __forceinline__ static float to_float(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    __device__ __forceinline__ static float2 to_float2(pair value) {
        return __bfloat1622float2(value);
    }
    __device__ __forceinline__ static __nv_bfloat16 from_float(float value) {
        return __float2bfloat16_rn(value);
    }
};

template <>
struct QsaAct<__half> {
    using pair = __half2;
    __device__ __forceinline__ static float to_float(__half value) { return __half2float(value); }
    __device__ __forceinline__ static float2 to_float2(pair value) { return __half22float2(value); }
    __device__ __forceinline__ static __half from_float(float value) {
        return __float2half_rn(value);
    }
};

__device__ __forceinline__ float qsa_rope_dim(const float* z, int d, int rotary_dim, float theta,
                                              int rho0, int rho1, int rho2) {
    if (d >= rotary_dim) { return z[d]; }
    const int half  = rotary_dim / 2;
    const int i     = d < half ? d : d - half;
    const int axis  = i % 3;
    const float rho = axis == 0 ? static_cast<float>(rho0)
                                : (axis == 1 ? static_cast<float>(rho1) : static_cast<float>(rho2));
    const float frequency =
        powf(theta, -2.0F * static_cast<float>(i) / static_cast<float>(rotary_dim));
    float sine   = 0.0F;
    float cosine = 0.0F;
    sincosf(rho * frequency, &sine, &cosine);
    if (d < half) { return z[d] * cosine - z[d + half] * sine; }
    return z[d] * cosine + z[d - half] * sine;
}

__device__ __forceinline__ int qsa_position_axis(const int* positions, int t, int b, int batch,
                                                 int axes, int axis) {
    const int a = axes == 1 ? 0 : axis;
    return positions[(t * axes + a) * batch + b];
}

__device__ __forceinline__ int qsa_row_position(const int* positions, int t, int b, int batch,
                                                int axes) {
    return positions[(t * axes) * batch + b];
}

} // namespace ninfer::ops::detail
