#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dynamic_grouped_conv_launch(const Tensor& x, const Tensor& coefficients,
                                 const Tensor& base_kernel, std::int32_t side, Tensor& out,
                                 cudaStream_t stream);

void prepare_normalize_launch(const Tensor& residual, const Tensor& norm_weight, float eps,
                              Tensor& normalized, std::int32_t hidden, std::int32_t columns,
                              cudaStream_t stream);

void prepare_conv_launch(const Tensor& normalized, const Tensor& coefficients_fp32,
                         const Tensor& base_kernel, Tensor& prepared, Tensor& finish_delta,
                         std::int32_t hidden, std::int32_t width, std::int32_t batch,
                         cudaStream_t stream);

void finish_add_launch(const Tensor& projected, const Tensor& base_kernel, const Tensor& finish_delta,
                       Tensor& residual, std::int32_t hidden, std::int32_t width,
                       std::int32_t batch, cudaStream_t stream);

} // namespace ninfer::ops::detail
