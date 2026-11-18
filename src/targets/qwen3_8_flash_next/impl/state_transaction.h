#pragma once

#ifndef NINFER_QWEN4EXP_VARIANT
#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_FLASH_NEXT_TXN_VARIANT
#endif
#ifndef NINFER_QWEN4EXP_RUNTIME_NS
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_runtime
#define NINFER_FLASH_NEXT_TXN_NS
#endif

#include "targets/qwen4_exp/impl/runtime/state_transaction.h"

#ifdef NINFER_FLASH_NEXT_TXN_NS
#undef NINFER_QWEN4EXP_RUNTIME_NS
#undef NINFER_FLASH_NEXT_TXN_NS
#endif
#ifdef NINFER_FLASH_NEXT_TXN_VARIANT
#undef NINFER_QWEN4EXP_VARIANT
#undef NINFER_FLASH_NEXT_TXN_VARIANT
#endif

namespace ninfer::targets::qwen3_8_flash_next::detail {

using StateTransaction =
    ninfer::targets::qwen4_exp::detail::flash_next_runtime::StateTransaction;

} // namespace ninfer::targets::qwen3_8_flash_next::detail
