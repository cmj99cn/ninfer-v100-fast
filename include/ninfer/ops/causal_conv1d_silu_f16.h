#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// FP16 causal depthwise conv (width 3) + SiLU. x/out are [C,T] with token-major storage,
// weight is [C,4] with tap-major storage, state is [3,C].
void causal_conv1d_silu_f16(const Tensor& x, const Tensor& weight, Tensor& state, Tensor& out,
                            cudaStream_t stream);

} // namespace ninfer::ops
