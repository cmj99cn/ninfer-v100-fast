#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

#include "ninfer/ops/candidate_selector.h"

namespace ninfer::ops::detail {

void candidate_topk_launch(const Tensor& logits, std::int32_t valid_rows, Tensor& candidate_ids,
                           Tensor& candidate_scores, DeviceSpan workspace, cudaStream_t stream);

void candidate_selector_path_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                    const Tensor& projected_hidden, const Tensor& anchors,
                                    const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook,
                                    const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, float* confidence, float edge_softcap,
                                    DeviceSpan workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
