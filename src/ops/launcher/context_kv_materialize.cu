// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime: ninfer::core Tensor/Weight/CyclicKVCacheLayer
// operands, sm_70 SIMT routes, BF16 K and V planes, an explicit rope_theta, and the registered
// Flash-Next companion geometry (H=2560, five layers, capacity 2048). The K/V projection, per-head
// 128-wide RMSNorm, split-half RoPE and cyclic slot addressing are unchanged.
#include "ops/launcher/context_kv_materialize.h"

#include "core/device.h"
#include "ops/common/dflash_rope.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/dflash2_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows    = kContextKVSize;
constexpr int kHeadDim = kContextKVHeadDim;
constexpr int kHeads   = kContextKVHeads;
constexpr float kEps   = 1.0e-6f;
constexpr int kBlock   = 256;

__device__ __forceinline__ std::int64_t cache_slot_offset(std::int32_t position, std::int32_t head,
                                                           std::int32_t lane,
                                                           std::int32_t padded_capacity,
                                                           std::int32_t capacity) {
    const auto slot = static_cast<std::int64_t>(position & (capacity - 1));
    return static_cast<std::int64_t>(kHeadDim) *
           (slot + static_cast<std::int64_t>(padded_capacity) *
                       (head + static_cast<std::int64_t>(kHeads) * lane));
}

__global__ void context_kv_post_kernel(const float* __restrict__ key_raw,
                                       const __nv_bfloat16* __restrict__ value_raw,
                                       const __nv_bfloat16* __restrict__ key_norm,
                                       __nv_bfloat16* __restrict__ cache_k,
                                       __nv_bfloat16* __restrict__ cache_v,
                                       const std::int32_t* __restrict__ positions,
                                       const std::int32_t* __restrict__ counts,
                                       const std::int32_t* __restrict__ slots, std::int32_t width,
                                       std::int32_t total_columns, std::int32_t max_count,
                                       std::int32_t min_count, std::int32_t padded_capacity,
                                       std::int32_t capacity, float rope_theta) {
    const int packed = blockIdx.x;
    const int batch  = packed / max_count;
    const int local  = packed - batch * max_count;
    const int count  = counts[batch];
    if (count < min_count || count > max_count || local >= count) { return; }
    const int physical = batch * width + local;
    const std::int32_t position = positions[physical];
    const int lane = threadIdx.x & 31;
    const int head = threadIdx.x >> 5;
    const int slot = slots[batch];

    const float* input = key_raw + static_cast<std::int64_t>(physical) * kRows +
                         head * kHeadDim;
    const int j0 = lane * 2;
    float x0 = input[j0], x1 = input[j0 + 1];
    float y0 = input[j0 + 64], y1 = input[j0 + 65];
    const float sum = warp_reduce_sum(x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1);
    const float inverse = rsqrtf(__shfl_sync(kFullWarpMask, sum, 0) / static_cast<float>(kHeadDim) +
                                 kEps);
    x0 *= inverse * __bfloat162float(key_norm[j0]);
    x1 *= inverse * __bfloat162float(key_norm[j0 + 1]);
    y0 *= inverse * __bfloat162float(key_norm[j0 + 64]);
    y1 *= inverse * __bfloat162float(key_norm[j0 + 65]);

    float sin0, cos0, sin1, cos1;
    dflash_rope_sincos(rope_theta, position, j0, &sin0, &cos0);
    dflash_rope_sincos(rope_theta, position, j0 + 1, &sin1, &cos1);
    const std::int64_t dst =
        cache_slot_offset(position, head, slot, padded_capacity, capacity);
    auto* out = reinterpret_cast<__nv_bfloat162*>(cache_k + dst);
    out[lane] = __floats2bfloat162_rn(x0 * cos0 - y0 * sin0, x1 * cos1 - y1 * sin1);
    out[lane + 32] = __floats2bfloat162_rn(y0 * cos0 + x0 * sin0, y1 * cos1 + x1 * sin1);

    for (int row = threadIdx.x; row < kRows; row += blockDim.x) {
        const int value_head = row / kHeadDim;
        const std::int64_t vdst =
            cache_slot_offset(position, value_head, slot, padded_capacity, capacity) +
            (row % kHeadDim);
        cache_v[vdst] = value_raw[static_cast<std::int64_t>(physical) * kRows + row];
    }
}

} // namespace

void context_kv_materialize_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    float rope_theta, ContextKVMaterializeExecutionEnvelope envelope, const Tensor& key_raw,
    const Tensor& value_raw, cudaStream_t stream) {
    if (envelope.max_count == 0) { return; }
    const std::int32_t hidden        = context.ne[0];
    const std::int32_t width         = context.ne[1];
    const std::int32_t batch         = context.ne[2];
    const std::int32_t total_columns = width * batch;
    const std::int32_t packed_columns = static_cast<std::int32_t>(envelope.max_count) * batch;
    for (const auto& layer : layers) {
        dflash2_w8_gemm_f32(context.data, static_cast<const std::uint8_t*>(layer.key_weight.qdata),
                            static_cast<const std::uint8_t*>(layer.key_weight.scales),
                            static_cast<float*>(key_raw.data), kRows, hidden, total_columns, stream);
        dflash2_w8_gemm(context.data, static_cast<const std::uint8_t*>(layer.value_weight.qdata),
                        static_cast<const std::uint8_t*>(layer.value_weight.scales),
                        value_raw.data, kRows, hidden, total_columns, stream);
        const auto& cache = layer.cache;
        context_kv_post_kernel<<<static_cast<unsigned>(packed_columns), kBlock, 0, stream>>>(
            static_cast<const float*>(key_raw.data),
            static_cast<const __nv_bfloat16*>(value_raw.data),
            static_cast<const __nv_bfloat16*>(layer.key_norm_weight.data),
            static_cast<__nv_bfloat16*>(cache.k.data), static_cast<__nv_bfloat16*>(cache.v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(counts.data),
            static_cast<const std::int32_t*>(state_slots.data), width, total_columns,
            static_cast<std::int32_t>(envelope.max_count),
            static_cast<std::int32_t>(envelope.min_count),
            static_cast<std::int32_t>(cache.padded_capacity),
            static_cast<std::int32_t>(cache.capacity), rope_theta);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
