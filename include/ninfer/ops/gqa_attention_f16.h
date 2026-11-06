#pragma once

#include "ninfer/ops/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// FP16 paged GQA for the registered D=256 geometry. q/out are [D,H,T] FP16, positions I32 [T],
// and cache pages are FP16 [D,page,H_KV,pages].
[[nodiscard]] std::size_t gqa_attention_f16_workspace_capacity_bytes(
    std::int32_t q_heads, GqaExecutionEnvelope envelope, std::int32_t min_width,
    std::int32_t max_width);

void gqa_attention_f16_cached(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

// FP16 paged K/V append. k/v are [D,H,T], positions I32 [T], cache pages FP16.
void gqa_kv_append_f16(const Tensor& k, const Tensor& v, const Tensor& positions,
                       PagedKVLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
