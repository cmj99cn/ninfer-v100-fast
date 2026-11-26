#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

// ninfer::ops - DFlash2 dynamic grouped causal convolution (kernel size 2, group size 16) and its
// fused prepare / linear-add forms.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: dynamic_grouped_conv
 *
 * One side of a two-tap grouped causal convolution. The flat storage of every [H,W,B] operand is
 * column-major with channel H contiguous (column index (b*W+i) outer), matching the existing
 * DFlash2 ops. `x` is contiguous BF16 or FP32 [H,W,B];
 * `coefficients` is contiguous FP32 or BF16 [4H/16,W,B] holding the four coefficient planes in
 * order (side 0 tap 0, side 0 tap 1, side 1 tap 0, side 1 tap 1); `base_kernel` is contiguous BF16
 * [H,2,2] whose flat order is [side][tap][channel]. `out` is contiguous BF16 [H,W,B].
 *
 * With G=H/16, column c=(b*W+i), group g=h/16 and coefficient row (side*2+tap)*G+g:
 *
 *   y = (base[side,0,h] + coeff[c,side,0,g]) * x[c,h]
 *     + (i > 0 ? (base[side,1,h] + coeff[c,side,1,g]) * x[c-1,h] : 0).
 *
 * For side==0, out = y. For side==1, out is first read as the incoming residual and is overwritten
 * with residual + y; it must not overlap x, the coefficients, or base_kernel. The channels are
 * independent. H must be a positive multiple of 16, W in [1,16] and B in [1,8].
 */
void dynamic_grouped_conv(const Tensor& x, const Tensor& coefficients, const Tensor& base_kernel,
                          std::int32_t side, Tensor& out, cudaStream_t stream);

/**
 * Op: rmsnorm_dynamic_grouped_conv_prepare
 *
 * A branch's prepared input. RMS-normalizes `residual` ([H,W,B] BF16, weight `norm_weight` [H],
 * epsilon `eps`) and projects the normalized value through the contiguous BF16
 * `kernel_projection_weight` [4H/16,H] to obtain the branch's four coefficient planes. Side 0 of
 * the resulting convolution is written to `prepared` [H,W,B] BF16, and side 1's two coefficient
 * planes are written, BF16-rounded, to `finish_delta` [H/16,2,W,B] for
 * linear_dynamic_grouped_conv_add. W is in [2,16] and B in [1,8].
 *
 * The projection is evaluated naively in FP32 from the represented BF16 inputs. `workspace` is
 * caller-owned transient storage sized by
 * rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(); it must not overlap any operand
 * or output. There is no persistent state side effect.
 */
[[nodiscard]] std::size_t rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t min_width, std::int32_t max_width, std::int32_t min_batch,
    std::int32_t max_batch);

void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Weight& kernel_projection_weight, Tensor& prepared,
                                          Tensor& finish_delta, WorkspaceArena& workspace,
                                          cudaStream_t stream);

/**
 * Op: linear_dynamic_grouped_conv_add
 *
 * Projects `x` ([C,W,B] BF16) through the row-split `projection_weight` (W8G32_F16S [H,C]) and
 * finishes side 1 of the dynamic convolution into an in-place residual:
 *
 *   z[c,h] = sum_i W8Weight[h,i] * x[c,i]
 *   residual[c,h] += (base[side=1,tap=0,h] + finish_delta[c,0,g]) * z[c,h]
 *                  + (i > 0 ? (base[side=1,tap=1,h] + finish_delta[c,1,g]) * z[c-1,h] : 0),
 *
 * with c=(b*W+i), g=h/16. `base_kernel` is BF16 [H,2,2] and `finish_delta` BF16 [H/16,2,W,B].
 * `workspace` is sized by linear_dynamic_grouped_conv_add_workspace_capacity_bytes() and holds the
 * projected BF16 plane; it must not overlap the operands, the weight, or the residual.
 */
[[nodiscard]] std::size_t linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t input_rows, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch, std::int32_t max_batch);

void linear_dynamic_grouped_conv_add(const Tensor& x, const Weight& projection_weight,
                                     const Tensor& base_kernel, const Tensor& finish_delta,
                                     Tensor& residual, WorkspaceArena& workspace,
                                     cudaStream_t stream);

} // namespace ninfer::ops
