#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: four-stream expansion with an offset RMSNorm.
 *
 * Math / indexing:
 *   residual[d,s,m]   = embedding[d,m]
 *   normalized[d,s,m] = embedding[d,m] * (1 + norm_weight[d,s])
 *                       / sqrt( sum_{d' = 0}^{H-1} embedding[d',m]^2 / H + epsilon )
 *
 * The residual is an exact copy of the embedding activation words, signed zero included. The four
 * streams share one statistic per column m. The gain 1 + norm_weight[d,s] is evaluated and
 * applied in FP32 and is never rounded before the multiplication.
 *
 * Logical shapes:
 *   H  hidden width, positive
 *   S  stream count, positive
 *   M  column extent (tokens * batch), positive
 *   embedding   BF16 or FP16 [H, M]    contiguous, dimension 0 fastest
 *   norm_weight BF16 [H, S]    contiguous, dimension 0 fastest
 *   residual    BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   normalized  BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   Element (d,s,m) of a three-dimensional tensor is stored at d + H * s + H * S * m, so the
 *   channel index of the four-stream residual is c = s * H + d. A [T,B] block flattens as
 *   M = T * B with m = t + T * b.
 *
 * Supported domain:
 *   BF16 or FP16 activation in and out (all activation operands sharing one dtype), 16-byte
 *   aligned bases, M > 0, epsilon positive and finite. The Qwen3.8-Flash-Next text package
 *   registers H = 2560 and S = 4.
 *
 * Numeric:
 *   The statistic must be accumulated in FP32; that accumulator precision is a required
 *   implementation profile. The oracle evaluates the formula in FP64 from the represented
 *   inputs; the activation output is promoted and compared against that result, so the output
 *   storage rounding belongs to the Op's numerical criterion and not to the oracle. Reduction
 *   association and staging are implementation choices.
 *
 * Effects:
 *   Writes all of residual and all of normalized. The two outputs are mutually disjoint and
 *   disjoint from every input. Read-only inputs may share backing and are preserved.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory, keeps no persistent state and
 *   supports CUDA Graph capture and replay at bound addresses.
 */
void hc_expand_norm(const Tensor& embedding, const Tensor& norm_weight, float epsilon,
                    Tensor& residual, Tensor& normalized, cudaStream_t stream);

/**
 * Op: per-stream offset RMSNorm of a four-stream residual.
 *
 * Math / indexing:
 *   normalized[d,s,m] = residual[d,s,m] * (1 + norm_weight[d,s])
 *                       / sqrt( sum_{d' = 0}^{H-1} residual[d',s,m]^2 / H + epsilon )
 *
 * Every stream owns its statistic and its affine weight. The gain 1 + norm_weight[d,s] is
 * evaluated and applied in FP32 and is never rounded to BF16 before the multiplication.
 *
 * Logical shapes:
 *   H  hidden width, positive
 *   S  stream count, positive
 *   M  column extent (tokens * batch), positive
 *   residual    BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   norm_weight BF16 [H, S]    contiguous, dimension 0 fastest
 *   normalized  BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   Element (d,s,m) is stored at d + H * s + H * S * m.
 *
 * Supported domain:
 *   BF16 or FP16 activation in and out (all activation operands sharing one dtype), 16-byte
 *   aligned bases, M > 0, epsilon positive and finite. The Qwen3.8-Flash-Next text package
 *   registers H = 2560 and S = 4.
 *
 * Numeric:
 *   The statistic must be accumulated in FP32; that accumulator precision is a required
 *   implementation profile. The oracle evaluates the formula in FP64 from the represented
 *   inputs; the activation output is promoted and compared against that result, so the output
 *   storage rounding belongs to the Op's numerical criterion and not to the oracle. Reduction
 *   association and staging are implementation choices.
 *
 * Effects:
 *   Writes all of normalized. The output is disjoint from every input. Read-only inputs may
 *   share backing and are preserved.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory, keeps no persistent state and
 *   supports CUDA Graph capture and replay at bound addresses.
 */
void hc_grouped_rmsnorm(const Tensor& residual, const Tensor& norm_weight, float epsilon,
                        Tensor& normalized, cudaStream_t stream);

/**
 * Op: four-stream residual update by an injected block output.
 *
 * Math / indexing:
 *   updated_residual[d,s,m] = residual[d,s,m] + injection[s,m] * block_output[d,m]
 *
 * The coefficient is applied directly: there is no second activation and no further scaling
 * at this step.
 *
 * Logical shapes:
 *   H  hidden width, positive
 *   S  stream count, positive
 *   M  column extent (tokens * batch), positive
 *   residual        BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   block_output   BF16 or FP16 [H, M]    contiguous, dimension 0 fastest
 *   injection       FP32 [S, M]    contiguous, dimension 0 fastest
 *   updated_residual BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   Element (d,s,m) is stored at d + H * s + H * S * m; injection[s,m] is stored at
 *   s + S * m and block_output[d,m] at d + H * m.
 *
 * Supported domain:
 *   BF16 or FP16 residual and block output (matching), FP32 injection, 16-byte aligned bases,
 *   M > 0. The injection holds finite, already activated coefficients in [0, 2]; that range is a
 *   caller precondition and is not enforced.
 *
 * Numeric:
 *   The update is evaluated in FP32 and stored in the activation dtype. The oracle evaluates the
 *   formula in FP64 from the represented inputs; the activation output is promoted and compared
 *   against that result, so the output storage rounding belongs to the Op's numerical criterion
 *   and not to the oracle.
 *
 * Effects:
 *   Writes the complete updated_residual. It may exactly alias the residual view or be
 *   disjoint from it; every other overlap between the output and an input is forbidden.
 *   Read-only inputs may share backing and are preserved.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory, keeps no persistent state and
 *   supports CUDA Graph capture and replay at bound addresses.
 */
void hc_combine(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                Tensor& updated_residual, cudaStream_t stream);

/**
 * Op: four-stream residual update followed by the per-stream offset RMSNorm of the update.
 *
 * Math / indexing:
 *   updated_residual[d,s,m] = residual[d,s,m] + injection[s,m] * block_output[d,m]
 *   normalized[d,s,m]       = updated_residual[d,s,m] * (1 + next_norm_weight[d,s])
 *                             / sqrt( sum_{d' = 0}^{H-1} updated_residual[d',s,m]^2 / H + epsilon )
 *
 * The normalization consumes the actual activation words of updated_residual, so the stored update
 * is the normalization's input and not an unrounded ideal value. The gain 1 +
 * next_norm_weight[d,s] is never rounded before the multiplication.
 *
 * Logical shapes:
 *   H  hidden width, positive
 *   S  stream count, positive
 *   M  column extent (tokens * batch), positive
 *   residual        BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   block_output   BF16 or FP16 [H, M]    contiguous, dimension 0 fastest
 *   injection       FP32 [S, M]    contiguous, dimension 0 fastest
 *   next_norm_weight BF16 [H, S]   contiguous, dimension 0 fastest
 *   updated_residual BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   normalized      BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   Element (d,s,m) is stored at d + H * s + H * S * m.
 *
 * Supported domain:
 *   BF16 or FP16 residual and block output (matching) with a BF16 norm weight, FP32 injection,
 *   16-byte aligned bases, M > 0, epsilon positive and finite. The injection holds finite,
 *   already activated coefficients in [0, 2]; that range is a caller precondition and is not
 *   enforced.
 *
 * Numeric:
 *   The statistic is accumulated in FP32 over the stored activation words. The oracle
 *   evaluates the complete formula in FP64 from the represented inputs and reproduces the
 *   activation-dtype storage rounding of updated_residual, which is an observable output of this
 *   Op; the normalized output is promoted and compared against that result, so its own storage
 *   rounding belongs to the Op's numerical criterion and not to the oracle.
 *
 * Effects:
 *   Writes all of updated_residual and all of normalized. updated_residual may exactly alias
 *   the residual view or be disjoint from it; every other overlap between an output and an
 *   input is forbidden, and normalized is disjoint from updated_residual. Read-only inputs
 *   may share backing and are preserved.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory, keeps no persistent state and
 *   supports CUDA Graph capture and replay at bound addresses.
 */
void hc_combine_norm(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                     const Tensor& next_norm_weight, float epsilon, Tensor& updated_residual,
                     Tensor& normalized, cudaStream_t stream);

/**
 * The low-rank mixing weights of a four-stream residual.
 *
 * Both members are contiguous BF16_CTRL or FP16_CTRL matrices whose second extent is the contiguous one, with
 * no bias and no padding, and with 16-byte aligned bases.
 *
 *   down_injection [R + S, H * S]
 *       Rows [0, R) are the low-rank down projection and rows [R, R + S) are the injection
 *       projection. R is the low-rank width and S is the stream count.
 *   up [H * S, R]
 *       Row s * H + d holds the gate of stream s and hidden channel d.
 */
struct HcMixWeights {
    Weight down_injection;
    Weight up;
};

/**
 * The low-rank mixing weights of the final four-stream residual, without injection rows.
 *
 * Both members are contiguous BF16_CTRL or FP16_CTRL matrices whose second extent is the contiguous one, with
 * no bias and no padding, and with 16-byte aligned bases.
 *
 *   down [R, H * S]
 *       The low-rank down projection.
 *   up [H * S, R]
 *       Row s * H + d holds the gate of stream s and hidden channel d.
 */
struct HcFinalMixWeights {
    Weight down;
    Weight up;
};

/**
 * Op: low-rank mixing of a four-stream residual into a block input and an injection.
 *
 * Math / indexing:
 *   flat[c,m]         = normalized[d,s,m] with c = s * H + d
 *   down[r,m]         = sum_{c} down_injection[r,c] * flat[c,m]           for r < R
 *   injection_row[s,m]= sum_{c} down_injection[R+s,c] * flat[c,m]         for s < S
 *   u[r,m]            = SiLU( down[r,m] / S )
 *   g[c,m]            = sum_{r} up[c,r] * u[r,m]
 *   block_input[d,m]  = ( sum_{s} sigmoid( g[s*H + d, m] ) * normalized[d,s,m] ) / S
 *   injection[s,m]    = 2 * sigmoid( injection_row[s,m] / S )
 *
 * SiLU(v) = v * sigmoid(v). The stream gates are independent, and the injection saturates at
 * the endpoints 0 and 2.
 *
 * Logical shapes:
 *   H  hidden width, positive
 *   S  stream count, positive
 *   M  column extent (tokens * batch), positive
 *   R  low-rank width, positive
 *   normalized BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   block_input BF16 or FP16 [H, M]   contiguous, dimension 0 fastest
 *   injection    FP32 [S, M]   contiguous, dimension 0 fastest
 *   The weights carry the geometry above: down_injection is [R + S, H * S] and up is
 *   [H * S, R]. R is the second extent of up, S is the second extent of normalized, and the
 *   remaining extents follow from the two relations down_injection.k = up.n = H * S and
 *   down_injection.n = R + S.
 *
 * Supported domain:
 *   BF16 or FP16 normalized with matching control weights (BF16_CTRL for BF16, FP16_CTRL for
 *   FP16), FP32 injection, 16-byte aligned bases, M > 0. The weights have exactly the declared
 *   native geometry, without bias or padding. The
 *   Qwen3.8-Flash-Next text package registers H = 2560, S = 4 and R = 320, which binds
 *   down_injection [324, 10240] and up [10240, 320]. M > 1 requires a Volta build or a linear
 *   problem registered for these geometries; the BF16 M = 1 path uses an
 *   architecture-independent GEMV.
 *
 * Numeric:
 *   The oracle evaluates the complete formula in FP64 from the represented inputs and keeps
 *   that result. The private down, activation and gate representations are not observable
 *   cast boundaries: the block input's private chain may round at the activation storage
 *   boundary because the block input itself is an activation output and its own storage rounding
 *   dominates.
 *   The injection is an FP32 output, so its projection over the injection rows is evaluated
 *   in FP32 and its accuracy is the FP32 accumulation profile of that contraction rather
 *   than an activation boundary. The block input and the injection are promoted and compared
 *   directly with the oracle results.
 *
 * Effects:
 *   Writes all of block_input and all of injection. The two outputs are mutually disjoint and
 *   disjoint from the input, the weights and the live workspace. Read-only inputs may share
 *   backing and are preserved. This Op performs no normalization and no residual combination.
 *
 * Workspace:
 *   The caller-owned transient capacity reported by hc_mix_workspace_capacity_bytes(). The
 *   workspace is disjoint from every operand, call-scoped and sized by that query.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory of its own, keeps no persistent
 *   state and supports CUDA Graph capture and replay at bound addresses.
 */
void hc_mix(const Tensor& normalized, const HcMixWeights& weights, WorkspaceArena& workspace,
            Tensor& block_input, Tensor& injection, cudaStream_t stream);

/**
 * Op: low-rank mixing of a four-stream residual into the final sample hidden state.
 *
 * Math / indexing:
 *   flat[c,m]        = normalized[d,s,m] with c = s * H + d
 *   down[r,m]        = sum_{c} down[r,c] * flat[c,m]                       for r < R
 *   u[r,m]           = SiLU( down[r,m] / S )
 *   g[c,m]           = sum_{r} up[c,r] * u[r,m]
 *   sample_hidden[d,m] = ( sum_{s} sigmoid( g[s*H + d, m] ) * normalized[d,s,m] ) / S
 *
 * SiLU(v) = v * sigmoid(v). The down weight has no injection rows, so this Op produces no
 * injection coefficient and performs no normalization and no residual combination.
 *
 * Logical shapes:
 *   H  hidden width, positive
 *   S  stream count, positive
 *   M  column extent (tokens * batch), positive
 *   R  low-rank width, positive
 *   normalized  BF16 or FP16 [H, S, M] contiguous, dimension 0 fastest
 *   sample_hidden BF16 or FP16 [H, M]  contiguous, dimension 0 fastest
 *   The weights carry the geometry above: down is [R, H * S] and up is [H * S, R]. R is the
 *   second extent of up, S is the second extent of normalized, and down.k = up.n = H * S.
 *
 * Supported domain:
 *   BF16 or FP16 normalized with matching control weights, 16-byte aligned bases, M > 0. The
 *   weights have exactly the declared native geometry, without bias or padding. The
 *   Qwen3.8-Flash-Next text
 *   package registers H = 2560, S = 4 and R = 320, which binds down [320, 10240] and up
 *   [10240, 320]. M > 1 requires a Volta build or a linear problem registered for these
 *   geometries; the BF16 M = 1 path uses an architecture-independent GEMV.
 *
 * Numeric:
 *   The oracle evaluates the complete formula in FP64 from the represented inputs and keeps
 *   that result. The private down, activation and gate representations are not observable
 *   cast boundaries, so the private chain may round at the activation storage boundary because
 *   the sample hidden state itself is an activation output. The sample hidden state is promoted and
 *   compared directly with the oracle result.
 *
 * Effects:
 *   Writes all of sample_hidden. The output is disjoint from the input, the weights and the
 *   live workspace. Read-only inputs may share backing and are preserved.
 *
 * Workspace:
 *   The caller-owned transient capacity reported by hc_final_mix_workspace_capacity_bytes().
 *   The workspace is disjoint from every operand, call-scoped and sized by that query.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory of its own, keeps no persistent
 *   state and supports CUDA Graph capture and replay at bound addresses.
 */
void hc_final_mix(const Tensor& normalized, const HcFinalMixWeights& weights,
                  WorkspaceArena& workspace, Tensor& sample_hidden, cudaStream_t stream);

/**
 * Minimum call-local workspace capacity for every M in the inclusive
 * [min_tokens, max_tokens] interval, with the supplied weight metadata. Measured from a
 * 256-byte-aligned arena cursor, including internal alignment but excluding trailing padding
 * and public tensors. Inspects host metadata only; device backing is not required. Invalid
 * weights or intervals throw.
 */
[[nodiscard]] std::size_t hc_mix_workspace_capacity_bytes(const HcMixWeights& weights,
                                                           std::int32_t min_tokens,
                                                           std::int32_t max_tokens);

/**
 * Minimum call-local workspace capacity for every M in the inclusive
 * [min_tokens, max_tokens] interval, with the supplied weight metadata. Measured from a
 * 256-byte-aligned arena cursor, including internal alignment but excluding trailing padding
 * and public tensors. Inspects host metadata only; device backing is not required. Invalid
 * weights or intervals throw.
 */
[[nodiscard]] std::size_t hc_final_mix_workspace_capacity_bytes(const HcFinalMixWeights& weights,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

} // namespace ninfer::ops
