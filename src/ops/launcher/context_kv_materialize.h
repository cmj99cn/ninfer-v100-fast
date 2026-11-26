#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

#include "ninfer/ops/context_kv_materialize.h"

namespace ninfer::ops::detail {

void context_kv_materialize_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    float rope_theta, ContextKVMaterializeExecutionEnvelope envelope, const Tensor& key_raw,
    const Tensor& value_raw, cudaStream_t stream);

} // namespace ninfer::ops::detail
