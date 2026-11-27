#pragma once

// ninfer::ops - row-axis split of one packed linear projection into independent outputs.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * One output-feature row interval of the packed parent weight.
 *
 * row_begin >= 0, row_count > 0 and row_begin + row_count <= parent.n. The interval is half-open
 * on the parent's output-feature row axis; it is not a contraction-column interval.
 */
struct LinearSplitSegment {
    std::int32_t row_begin = 0;
    std::int32_t row_count = 0;
};

/**
 * Returns the transient capacity required by linear_split for every T in the inclusive
 * [min_tokens,max_tokens] interval. It is the maximum over segments of
 * linear_workspace_capacity_bytes() evaluated at that segment's row count and the parent's k.
 * Invalid profiles, segments or intervals throw.
 */
[[nodiscard]] std::size_t linear_split_workspace_capacity_bytes(
    QType qtype, std::int32_t parent_rows, std::int32_t input_rows,
    std::span<const LinearSplitSegment> segments, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: linear_split
 *
 * Math / indexing:
 *   For every segment s with rows [r_s, r_s + n_s) and every output column t:
 *     out_s[i,t] = sum_{k} dequant(w)[r_s+i,k] * FP32(x)[k,t]
 *   The split is on the parent's output-feature row axis. Each segment is an independent
 *   projection over the same activation; nothing is summed across segments.
 *
 * Logical shapes:
 *   x is contiguous BF16 or FP16 [K,T], out_s is contiguous [n_s,T] with the same dtype as x,
 *   and the parent weight is logical [N,K]. T is positive. The outputs' row counts must equal the
 *   segment counts.
 *
 * Supported domain:
 *   RowSplit K128 parents Q4G64_F16S, Q5G64_F16S, Q6G64_F16S and W8G32_F16S, plus contiguous
 *   BF16_CTRL or FP16_CTRL parents. Every segment must itself resolve as a registered linear
 *   problem of the parent's format; an unsupported format, layout or segment shape throws before
 *   any launch.
 *   Batched NVFP4, row-scaled FP8 and every other layout are rejected explicitly.
 *
 * Numeric:
 *   Each segment has the same numerical contract as ops::linear. The governing oracle is an
 *   independent naive FP64 evaluation of the logical formula from the represented public inputs,
 *   decoding every packed code with the exact stored scale over the segment's own row interval,
 *   compared with a criterion appropriate to the activation dtype's output rounding. Equality
 *   with an ops::linear()
 *   invocation on an equivalent standalone weight is supplementary evidence rather than the
 *   criterion, because the accumulation grouping depends on the row extent and bit-identical
 *   results are not guaranteed.
 *
 * Effects:
 *   Writes every output in full. x, the parent weight and the outputs are pairwise non-overlapping
 *   and read-only inputs are preserved.
 *
 * Workspace:
 *   The caller-owned transient capacity reported by linear_split_workspace_capacity_bytes().
 *   Each segment runs in a nested arena scope, so the capacity is the maximum segment requirement
 *   rather than their sum. The workspace is call-scoped and disjoint from every operand.
 *
 * Execution:
 *   Enqueues one projection per segment on the given stream, allocates no device memory and keeps
 *   no persistent state. It supports CUDA Graph capture and replay at bound addresses.
 */
void linear_split(const Tensor& x, const Weight& w, std::span<const LinearSplitSegment> segments,
                   std::span<Tensor> outputs, LinearPolicy policy, WorkspaceArena& workspace,
                   cudaStream_t stream);

/**
 * Op: linear_split_fused_f16 (sm_70 only)
 *
 * Same math, indexing, numeric oracle and effects as linear_split, restricted to W8G32_F16S
 * RowSplit parents with FP16 activations and outputs and large token counts. Each segment is
 * issued through the fused-dequant Volta m8n8k4 tensor-core route, which stages each weight row
 * once for a whole 32-column mma tile instead of replaying it once per 8 output columns. It is
 * numerically a different implementation profile (FP16 dequant operands, FP32 tensor-core
 * accumulation) and is validated directly against linear_split's FP64 oracle.
 *
 * Requires k divisible by 32 and 8, k >= 32, and a non-empty segment list. Throws on any other
 * format, dtype or geometry. Uses no workspace and keeps no state.
 */
void linear_split_fused_f16(const Tensor& x, const Weight& w,
                            std::span<const LinearSplitSegment> segments, std::span<Tensor> outputs,
                            cudaStream_t stream);

} // namespace ninfer::ops
