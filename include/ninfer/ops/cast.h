#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: cast_fp32_to_bf16
 *
 * Math / indexing:
 *   destination[i] = round_to_bf16_rne(source[i]) for every logical element i.
 *
 * Logical shapes:
 *   Source and destination have identical shapes and contiguous element order.
 *
 * Supported domain:
 *   Source is FP32 and destination is BF16. Contiguous aligned element counts divisible by four
 *   or two use vector routes; all other contiguous shapes use the scalar route.
 *
 * Numeric:
 *   Each FP32 value is converted independently with round-to-nearest-even.
 *
 * Effects:
 *   Writes the full destination. Source and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no state side effect beyond writing destination.
 */
void cast_fp32_to_bf16(const Tensor& source, Tensor& destination, cudaStream_t stream);

// Exact widening/narrowing casts used by the FP16 text-path prototype. BF16 -> FP16 is exact for
// every finite BF16 value inside FP16's range; FP16 -> BF16 rounds to nearest even.
void cast_bf16_to_fp16(const Tensor& source, Tensor& destination, cudaStream_t stream);
void cast_fp16_to_bf16(const Tensor& source, Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops
