#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

// ninfer::ops - DFlash2 fused gate/up projection and SwiGLU (W8 G32 row-split parent).

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops {

/**
 * Op: dflash2_linear_swiglu
 *
 * `x` is contiguous BF16 [2560,T] in the DFlash2 physical packing (token column outer, the 2560
 * features contiguous); the parent `gate_up_weight` is W8G32_F16S RowSplit [15360,2560]
 * with gate rows [0,7680) followed by their matching up rows [7680,15360). The output is
 * contiguous BF16 [7680,T]:
 *
 *   out[i,t] = SiLU(proj[i,t]) * proj[7680+i,t],
 *
 * where proj is the W8 projection decoded with the exact FP16 group-32 scale and accumulated in
 * FP32. `workspace` holds the gate half of the projection and is sized by
 * dflash2_linear_swiglu_workspace_capacity_bytes(); it must not overlap x, the weight, or out.
 */
[[nodiscard]] std::size_t dflash2_linear_swiglu_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                         std::int32_t max_tokens);

void dflash2_linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
