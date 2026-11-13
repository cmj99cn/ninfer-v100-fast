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
