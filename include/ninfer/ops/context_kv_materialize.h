#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

// ninfer::ops - DFlash2 context K/V materialization into a cyclic ring.

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kContextKVMaterializeLayers = 5;
inline constexpr std::int32_t kContextKVSize              = 1024;
inline constexpr std::int32_t kContextKVHeadDim           = 128;
inline constexpr std::int32_t kContextKVHeads             = 8;
inline constexpr std::uint32_t kContextKVCapacity         = 2048;

/** One drafter layer's context-key/value weights, key norm, and its cyclic ring. */
struct ContextKVMaterializeLayerView {
    Weight key_weight;
    Weight value_weight;
    Tensor key_norm_weight;
    CyclicKVCacheLayerView cache;
};

/** Host execution-resource promise; it bounds launch count/workspace, never the admitted set. */
struct ContextKVMaterializeExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

/**
 * Op: context_kv_materialize
 *
 * Projects the target's tapped features into the drafter's per-layer context ring. `context` is
 * contiguous BF16 [H,W,B] with H=2560 in the DFlash2 physical packing (column (b*W+i) outer, H
 * contiguous); `positions` I32 [W,B], `counts` I32 [B] and `state_slots`
 * I32 [B]. Each layer's `key_weight` and `value_weight` are W8G32_F16S RowSplit [1024,H],
 * `key_norm_weight` is BF16 [128], and `cache` is the layer's cyclic ring.
 *
 * For a live column c=(b*W+i) with i<counts[b] and positions[i,b]=p:
 *
 *   key_raw[h*128+j] = sum_k K_weight[h*128+j,k] * context[k,c]
 *   key[h*128+j]     = rmsnorm_128(key_raw[h*128:], key_norm[h*128:], eps=1e-6) then split-half
 *                      RoPE at p with theta=rope_theta:
 *                        out[j]    = x[j]    * cos_j - x[j+64] * sin_j
 *                        out[j+64] = x[j+64] * cos_j + x[j]    * sin_j
 *   value[h*128+j]   = sum_k V_weight[h*128+j,k] * context[k,c]
 *
 * Key and value are stored BF16 at physical slot p & (capacity-1), head h, lane state_slots[b]
 * (K and V share the [D, padded_capacity, Hkv, lane] addressing). Columns outside [min_count,
 * max_count] or at/after counts[b] are inert. The Op allocates only from `workspace`, sized by
 * context_kv_materialize_workspace_capacity_bytes().
 */
[[nodiscard]] std::size_t context_kv_materialize_workspace_capacity_bytes(std::int32_t batch_size,
                                                                         std::int32_t min_width,
                                                                         std::int32_t max_width);

void context_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    float rope_theta, ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace,
    cudaStream_t stream);

} // namespace ninfer::ops
