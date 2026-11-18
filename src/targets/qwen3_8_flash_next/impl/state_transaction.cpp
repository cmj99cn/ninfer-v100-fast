#include "targets/qwen3_8_flash_next/impl/variant.h"
#include "targets/qwen3_8_flash_next/impl/state_transaction.h"

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_runtime
#include "targets/qwen4_exp/impl/runtime/state_transaction_impl.h"
#undef NINFER_QWEN4EXP_RUNTIME_NS
#undef NINFER_QWEN4EXP_VARIANT
