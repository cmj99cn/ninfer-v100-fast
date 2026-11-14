#pragma once

#include "core/tensor.h"

namespace ninfer::targets::qwen4_exp::detail {

#if defined(NINFER_TEXT_DTYPE_F16)
inline constexpr DType kTextActivationDtype = DType::FP16;
#else
inline constexpr DType kTextActivationDtype = DType::BF16;
#endif

} // namespace ninfer::targets::qwen4_exp::detail
