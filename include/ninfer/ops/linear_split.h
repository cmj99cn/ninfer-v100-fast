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
 *   x is contiguous BF16 [K,T], out_s is contiguous BF16 [n_s,T], and the parent weight is
 *   logical [N,K]. T is positive. The outputs' row counts must equal the segment counts.
 *
 * Supported domain:
 *   RowSplit K128 parents Q4G64_F16S, Q5G64_F16S, Q6G64_F16S and W8G32_F16S, plus contiguous
 *   BF16_CTRL parents. Every segment must itself resolve as a registered linear problem of the
 *   parent's format; an unsupported format, layout or segment shape throws before any launch.
 *   Batched NVFP4, row-scaled FP8 and every other layout are rejected explicitly.
 *
 * Numeric:
 *   Each segment has the same numerical contract as ops::linear on the equivalent standalone
 *   weight. The oracle is an independent ops::linear() invocation on a segment weight generated
 *   from the same logical parent rows; the two BF16 outputs are compared elementwise.
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

} // namespace ninfer::ops
