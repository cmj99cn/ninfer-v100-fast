#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// FP16 GDN control projection: x [K,T] FP16, a/b weights [H,K] FP16, A_log/dt_bias FP32 [H],
// g/beta FP32 [H,T]. Evaluation is FP32-accumulated; used by the full-FP16 GDN leaf.
void gdn_gating_proj_f16(const Tensor& x, const Tensor& a_weight, const Tensor& b_weight,
                         const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta,
                         cudaStream_t stream);

} // namespace ninfer::ops
