// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

#include "ninfer/ops/context_kv_materialize.h"
#include <cmath>

#include "core/layout.h"
#include "ops/launcher/context_kv_materialize.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp = "context_kv_materialize";

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, std::int32_t n3, std::size_t alignment, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != n3 || !tensor.is_contiguous() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_w8(const Weight& weight, std::int32_t hidden, const char* name) {
    const std::uint64_t codes  = static_cast<std::uint64_t>(kContextKVSize) * hidden;
    const std::uint64_t scales = static_cast<std::uint64_t>(kContextKVSize) * (hidden / 32) * 2U;
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != kContextKVSize || weight.k != hidden ||
        weight.shape[0] != kContextKVSize || weight.shape[1] != hidden ||
        weight.padded_shape[0] != kContextKVSize || weight.padded_shape[1] != hidden ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < codes + scales || weight.qdata == nullptr ||
        weight.scales == nullptr || (hidden % 32) != 0 || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

std::size_t workspace_bytes(std::int32_t columns) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {kContextKVSize, columns}, 256);
    (void)layout.alloc(DType::BF16, {kContextKVSize, columns}, 256);
    return layout.peak_bytes();
}

} // namespace

std::size_t context_kv_materialize_workspace_capacity_bytes(std::int32_t batch_size,
                                                             std::int32_t min_width,
                                                             std::int32_t max_width) {
    if (batch_size < 1 || batch_size > 8 || min_width < 1 || max_width < min_width ||
        max_width > static_cast<std::int32_t>(kContextKVCapacity)) {
        throw std::invalid_argument(std::string(kOp) + " workspace: invalid W/B interval");
    }
    return workspace_bytes(max_width * batch_size);
}

void context_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    float rope_theta, ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace,
    cudaStream_t stream) {
    const std::int32_t hidden = context.ne[0];
    const std::int32_t width  = context.ne[1];
    const std::int32_t batch  = context.ne[2];
    if (hidden <= 0 || (hidden % 32) != 0 || width < 1 ||
        width > static_cast<std::int32_t>(kContextKVCapacity) || batch < 1 || batch > 8) {
        throw std::invalid_argument(std::string(kOp) + ": invalid W/B or H");
    }
    require_tensor(context, DType::BF16, hidden, width, batch, 1, 16, "context");
    require_tensor(positions, DType::I32, width, batch, 1, 1, alignof(std::int32_t), "positions");
    require_tensor(counts, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "counts");
    require_tensor(state_slots, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "state_slots");
    if (!(rope_theta > 0.0f) || !std::isfinite(rope_theta)) {
        throw std::invalid_argument(std::string(kOp) + ": rope_theta must be positive and finite");
    }
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(width) ||
        envelope.max_count > kContextKVCapacity) {
        throw std::invalid_argument(std::string(kOp) + ": invalid execution envelope");
    }
    for (const auto& layer : layers) {
        require_w8(layer.key_weight, hidden, "key weight");
        require_w8(layer.value_weight, hidden, "value weight");
        require_tensor(layer.key_norm_weight, DType::BF16, kContextKVHeadDim, 1, 1, 1, 4,
                       "key norm weight");
        const auto& cache = layer.cache;
        if (cache.capacity != kContextKVCapacity ||
            cache.padded_capacity < cache.capacity || cache.num_kv_heads != kContextKVHeads ||
            cache.head_dim != kContextKVHeadDim || cache.lane_capacity <= 0) {
            throw std::invalid_argument(std::string(kOp) + ": invalid cyclic cache geometry");
        }
        const auto padded = static_cast<std::int32_t>(cache.padded_capacity);
        require_tensor(cache.k, DType::BF16, kContextKVHeadDim, padded, kContextKVHeads,
                       cache.lane_capacity, 16, "cache K");
        require_tensor(cache.v, DType::BF16, kContextKVHeadDim, padded, kContextKVHeads,
                       cache.lane_capacity, 16, "cache V");
    }
    if (envelope.max_count == 0) { return; }

    auto scope = workspace.scope();
    Tensor key_raw   = workspace.alloc(DType::FP32, {kContextKVSize, width * batch});
    Tensor value_raw = workspace.alloc(DType::BF16, {kContextKVSize, width * batch});
    detail::context_kv_materialize_launch(context, positions, counts, state_slots, layers,
                                          rope_theta, envelope, key_raw, value_raw, stream);
}

} // namespace ninfer::ops
