#include "targets/qwen3_8_flash_next/impl/variant.h"

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_runtime

#include "targets/qwen3_8_flash_next/impl/runtime/graph_decode.h"
#include "targets/qwen3_8_flash_next/impl/runtime/graph_decode_impl.h"
#include "targets/qwen4_exp/impl/runtime/instantiate.h"
#include "targets/qwen3_8_flash_next/impl/runtime/graph_decode_defs.h"
