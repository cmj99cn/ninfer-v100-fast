#pragma once

// Compile-time activation dtype for the text stream. The product contract keeps INT8 group-64 KV,
// BF16 weights/biases/norms, and the BF16 Vision tower fixed; only the text hidden/residual/
// activation workspaces and the ops that consume them follow this switch. Selected at configure
// time, never at runtime, so BF16 and FP16 are separate binaries with no per-token branch.

#include "core/tensor.h"

namespace ninfer::targets::qwen3_6::detail {

#if defined(NINFER_TEXT_DTYPE_F16)
inline constexpr DType kTextActivationDtype = DType::FP16;
#else
inline constexpr DType kTextActivationDtype = DType::BF16;
#endif

} // namespace ninfer::targets::qwen3_6::detail
