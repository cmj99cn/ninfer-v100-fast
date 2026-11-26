#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

// ninfer::ops - DFlash2 candidate selection: the target head's exact top-16 candidates and the
// conditional lattice path over them.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kCandidateCount = 16;
inline constexpr std::int32_t kSelectorRank   = 256;

/**
 * Op: candidate_topk
 *
 * `logits` is contiguous BF16 or FP32 [rows, U]: each of the U columns is one draft row's logits
 * over the token vocabulary. The flat storage is column-major with the rows contiguous (column
 * index outer), the same physical packing the existing DFlash2 head/top-k ops consume. `candidate_ids` is contiguous I32 [16,U] and `candidate_scores` is
 * contiguous FP32 [16,U]. For every column this Op writes its 16 best rows in descending score
 * order, breaking equal scores by the lower token id; only rows v in [0,valid_rows) participate and
 * `valid_rows` is in [16,rows]. A column with fewer than 16 finite entries (for example all NaN)
 * fills the remaining slots with token 0, so a consumer never reads past the vocabulary.
 *
 * The comparison is exact: scores are the represented (BF16-promoted or FP32) values, not a
 * rounded projection. `workspace` is caller-owned transient storage sized by
 * candidate_topk_workspace_capacity_bytes(); it must not overlap the operands or outputs.
 */
[[nodiscard]] std::size_t candidate_topk_workspace_capacity_bytes(std::int32_t columns);

void candidate_topk(const Tensor& logits, std::int32_t valid_rows, Tensor& candidate_ids,
                    Tensor& candidate_scores, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: candidate_selector_path
 *
 * Walks one conditional path per batch row over the 16 candidate tokens of every draft step.
 * Shapes are candidate_ids/unary_scores [16,K,B], projected_hidden BF16 [rank,K,B], anchors I32
 * [B], base_positions I32 [B], drafts I32 [K,B] and proposal_q FP32 [16,K,B]; K is in [1,15], B in
 * [1,8] and rank is exactly 256. `predecessor_codebook` and `successor_codebook` are contiguous
 * BF16 [rows,rank] with rows >= the candidate token ids. `configs` is a device-resident
 * SamplingConfig[B] array (temperature and seed are the only fields this Op reads); `confidence`,
 * when non-null, is FP32 [K,B].
 *
 * At step k the edge of candidate c is
 *
 *   E_c = unary_scores[c,k,b] + sum_r P[parent,r] * projected_hidden[r,k,b] * S[c,r]
 *
 * where parent starts at anchors[b] and becomes the token id selected on the previous step, and
 * P/S are the two codebooks' rows named by the parent/candidate token. With edge_softcap > 0 each
 * edge is `softcap * tanh(E_c / softcap)` first.
 *
 * For configs[b].temperature <= 0 the selected candidate is the lowest slot index c whose edge
 * equals the maximum (the deterministic greedy tie-break). Otherwise proposal_q[:,k,b] is the
 * softmax over the 16 edges at that temperature and one candidate is drawn with the counter-based
 * uniform at key (seed, base_positions[b]+k, kSamplePurposeDFlash2Proposal, 0). `confidence[k,b]`
 * receives the drawn candidate's temperature-1 softmax probability. drafts[k,b] is the selected
 * token id; every other argument is read-only.
 *
 * num_selected_steps (K) chooses the route: K <= 4 evaluates each step's 16 edges directly, and a
 * larger K materialises the lattice of edges in caller-owned workspace first. Both routes produce
 * the same selected ids and proposal probabilities. `workspace` is sized by
 * candidate_selector_path_workspace_capacity_bytes() and must not overlap any operand, the
 * codebooks, configs, or the outputs.
 */
[[nodiscard]] std::size_t candidate_selector_path_workspace_capacity_bytes(int min_steps,
                                                                           int max_steps,
                                                                           int min_batch,
                                                                           int max_batch);

void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                             cudaStream_t stream, Tensor* confidence = nullptr,
                             float edge_softcap = 0.0f);

} // namespace ninfer::ops
