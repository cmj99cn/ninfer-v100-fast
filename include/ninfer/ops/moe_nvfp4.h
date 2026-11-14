#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The weights of the Qwen3.8-Flash-Next MoE.
 *
 *   router_shared   BF16 contiguous [1793, 2560]. Rows [0, 512) are the routed logits, row
 *                   512 is the shared logit, rows [513, 1153) are the shared gate, and rows
 *                   [1153, 1793) are the shared up projection.
 *   shared_down     BF16 contiguous [2560, 640], an independent object.
 *   routed_gate_up  batched NVFP4 [512, 1280, 2560]. Inside each expert's own matrix, rows
 *                   [0, 640) are the gate and rows [640, 1280) are the up projection.
 *   routed_down     batched NVFP4 [512, 2560, 640], the whole matrix of each expert.
 *
 * A batched NVFP4 weight carries one swizzled scale plane and one FP32 weight divisor per
 * matrix, so it uses the batched weight view rather than the single-matrix `Weight`.
 */
struct MoeNvfp4Weights {
    Weight router_shared;
    Weight shared_down;
    BatchedNvfp4Weight routed_gate_up;
    BatchedNvfp4Weight routed_down;
};

/**
 * Op: top-10 normalized routing control of the MoE.
 *
 * Math / indexing:
 *   l[d, t] = sum_{c = 0}^{H-1} router_shared[c, d] * x[c, t]          for d in [0, E + 1)
 *   ids[j, t] = the K greatest l[0:E, t], in descending order; at an exact tie the lower
 *   expert id comes first.
 *   coefficients[j, t] = exp(l[ids[j,t], t]) / sum_{j'} exp(l[ids[j',t], t]) — the softmax
 *   taken over the K selected logits only (norm_topk_prob), so the K coefficients sum to 1.
 *   shared_gate[t] = sigmoid(l[E, t]).
 *
 * Logical shapes:
 *   E  experts, 512
 *   K  selected experts per token, 10
 *   H  hidden width, 2560
 *   T  token extent, positive
 *   x            BF16 or FP16 [H, T] contiguous, dimension 0 fastest
 *   expert_ids   I32  [K, T] contiguous, dimension 0 fastest
 *   coefficients FP32 [K, T] contiguous, dimension 0 fastest
 *   shared_gate  FP32 [T]    contiguous
 *   Element (k, t) of a two-dimensional output is stored at k + K * t.
 *
 * Supported domain:
 *   The exact registered geometry above. 16-byte aligned bases. Every positive T.
 *
 * Numeric:
 *   The oracle evaluates the complete formula in FP64 from the represented activation.
 *   Scores, ranking, and the softmax accumulator are private execution choices; the published
 *   coefficients are FP32 outputs and are compared against the oracle. The tie rule is exact
 *   and observable.
 *
 * Effects:
 *   Writes all of expert_ids, all of coefficients, and all of shared_gate. The outputs are
 *   mutually disjoint and disjoint from x and the weights. Read-only inputs are preserved.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory, keeps no persistent state, and
 *   supports CUDA Graph capture and replay at bound addresses.
 */
void moe_nvfp4_route(const Tensor& x, const MoeNvfp4Weights& weights, Tensor& expert_ids,
                     Tensor& coefficients, Tensor& shared_gate, cudaStream_t stream);

/**
 * Op: the top-10 routed NVFP4 experts and the sigmoid-gated shared expert, accumulated into
 * destination.
 *
 * Math / indexing:
 *   For a token column t with route ids[j, t] and coefficients[j, t] as defined by the routing
 *   control above, and for the represented activation column x[:, t]:
 *
 *   F_e(v) = down_e * (SiLU(gate_e * v) * (up_e * v))
 *
 *   destination[d, t] += sum_{j = 0}^{K-1} coefficients[j, t] * F_{ids[j, t]}(x[:, t])
 *   destination[d, t] += shared_gate[t] * F_shared(x[:, t])
 *
 *   SiLU(v) = v * sigmoid(v). The shared expert contributes exactly once with its own
 *   sigmoid(l[E, t]) gate and is never scaled by the top-10 route coefficients. The routed
 *   contribution is the coefficient-weighted sum over the selected experts. The expert rows
 *   are consumed in place from the batched payload: each selected expert uses its own matrix
 *   row ranges (gate rows [0, 640), up rows [640, 1280), down rows whole) together with that
 *   matrix's scale plane and FP32 divisor. No selected-weight gather or repack occurs.
 *
 * Logical shapes:
 *   x           BF16 or FP16 [H, T] contiguous, dimension 0 fastest
 *   destination BF16 or FP16 [H, T] contiguous, dimension 0 fastest, matching x
 *   Element (d, t) is stored at d + H * t. The weights carry the geometry declared in
 *   MoeNvfp4Weights.
 *
 * Supported domain:
 *   The exact registered geometry above. 16-byte aligned bases. Every positive T.
 *
 * Numeric:
 *   The oracle evaluates the complete formula in FP64 from the represented inputs. The NVFP4
 *   weights are decoded with the stored codebook: the 4-bit E2M1 code, the E4M3 scale of the
 *   16-element group, and the FP32 weight divisor of that matrix. A mispaired scale plane or
 *   divisor is a correctness failure even when the totals look plausible. Scores, route
 *   weights, activation representation, workspace representation, and reduction association
 *   are private execution choices. destination is a BF16 or FP16 output, so its own storage
 *   rounding belongs to the Op's numerical criterion and not to the oracle; the routed and shared
 *   dot products accumulate in FP32 and only the final destination value is rounded.
 *
 * Effects:
 *   destination is the accumulation target: its incoming value is preserved and the routed and
 *   shared contributions are added to it. No other observable mutation. destination is
 *   disjoint from x, the weights, and the live workspace. Read-only inputs are preserved.
 *
 * Workspace:
 *   The caller-owned transient capacity reported by moe_nvfp4_workspace_capacity_bytes(): the
 *   per-token route ids, coefficients, and shared gate produced by the routing control and
 *   consumed by the expert paths. The workspace is disjoint from every operand, call-scoped,
 *   and sized by that query.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory of its own, keeps no persistent
 *   state, and supports CUDA Graph capture and replay at bound addresses.
 */
void moe_routed_nvfp4(const Tensor& x, const MoeNvfp4Weights& weights, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream);

/**
 * The routing control produced by moe_nvfp4_route, consumed by the sharded expert entries below.
 * All three are contiguous outputs of one moe_nvfp4_route call on the same activation:
 *   expert_ids   I32  [K, T]
 *   coefficients FP32 [K, T]
 *   shared_gate  FP32 [T]
 */
struct MoeNvfp4RouteView {
    Tensor expert_ids;
    Tensor coefficients;
    Tensor shared_gate;
};

/**
 * Op: the routed top-K contribution of ONE rank's expert subset, added into `destination`.
 *
 * The routing is the same global top-K selection moe_nvfp4_route publishes. Each batched weight
 * carries `expert_begin` (the global id of its first matrix) and `matrices` (the local count), so
 * a route slot whose global id falls outside [expert_begin, expert_begin + matrices) contributes
 * exactly zero on this rank. The whole rank group covers [0, 512); the caller combines the ranks
 * with one all-reduce. The shared expert is NOT part of this entry (see moe_shared_nvfp4), so an
 * all-reduce over the routed partials never counts it twice.
 *
 * Shapes, formats and effects match moe_routed_nvfp4 above. `route` must be the output of
 * moe_nvfp4_route for this same `x`; it is read-only. No workspace is required.
 */
void moe_routed_nvfp4_partial(const Tensor& x, const MoeNvfp4Weights& weights,
                              const MoeNvfp4RouteView& route, Tensor& destination,
                              cudaStream_t stream);

/**
 * Op: the sigmoid-gated shared-expert contribution, added into `destination`.
 *
 * This is the replicated half of the MoE: every rank carries the same shared weights and the same
 * routing control, so every rank adds the identical contribution exactly once after the routed
 * partials have been reduced. The routed experts are not part of this entry.
 */
void moe_shared_nvfp4(const Tensor& x, const MoeNvfp4Weights& weights,
                      const MoeNvfp4RouteView& route, Tensor& destination, cudaStream_t stream);

/**
 * Minimum call-local workspace capacity for every T in the inclusive
 * [min_tokens, max_tokens] interval. The Op takes one contiguous block aligned to 256 bytes at
 * the arena cursor holding the route ids, coefficients, and shared gate in that order; each
 * region is 4-byte aligned inside the block, so no internal alignment padding is added and the
 * reported capacity is that 256-aligned block size. Invalid intervals throw.
 */
[[nodiscard]] std::size_t moe_nvfp4_workspace_capacity_bytes(std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

} // namespace ninfer::ops
