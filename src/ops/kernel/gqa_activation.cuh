#pragma once

// GQA-specific activation aliases. The element trait itself is the repo-wide ActivationTraits so
// linear/attention kernels share one BF16/FP16 load-store contract.

#include "ops/common/activation.cuh"

namespace ninfer::ops {

template <class T>
using GqaActivation = ActivationTraits<T>;

template <class T>
struct GqaAppendInputOf {
    static constexpr bool writes_cache = true;
    const T* k;
    const T* v;
};

using GqaAppendInput = GqaAppendInputOf<__nv_bfloat16>;

} // namespace ninfer::ops
