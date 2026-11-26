#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.
#include "core/tensor.h"
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
void dflash2_attn_input_proj_launch(const Tensor& x, const Weight& parent, Tensor& q, Tensor& k,
                                    Tensor& v, cudaStream_t stream);
}
