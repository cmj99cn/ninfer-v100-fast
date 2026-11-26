#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

// ninfer::ops - DFlash2 fused Q/K/V input projection (W8 G32 row-split parent).

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: dflash2_attn_input_proj
 *
 * `x` is contiguous BF16 [2560,T] in the DFlash2 physical packing (token column outer, the 2560
 * features contiguous); the parent `query_key_value_weight` is W8G32_F16S RowSplit
 * [6144,2560] with rows [query 4096, key 1024, value 1024]. The outputs are contiguous BF16
 * [4096,T], [1024,T], [1024,T]. Every projection decodes the stored signed INT8 codes with the
 * exact FP16 group-32 scale and accumulates naively in FP32 from the represented BF16 activation.
 * Outputs use the same packing (token column outer, output rows contiguous). It performs no
 * normalization or rotation: q and k remain raw projection outputs. T is any
 * positive value. There is no workspace or persistent state side effect.
 */
void dflash2_attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q,
                             Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops
