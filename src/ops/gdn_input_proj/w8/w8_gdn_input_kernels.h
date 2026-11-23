#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kMaxQkvzDecodeTokens = 4;

void w8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);
void w8_gdn_input_decode_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);
void w8_gdn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);
void w8_gdn_input_splitk_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);
void w8_gdn_input_splitk_conv_record_launch(const Tensor& x, const Weight& weight,
                                            const Tensor& conv_weight, const Tensor& conv_states,
                                            const Tensor& valid_columns, const Tensor& initial_slot,
                                            Tensor& conv_record, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& z, cudaStream_t stream);
#ifdef NINFER_VOLTA_BUILD
// Volta: the [12288,2048] parent split into qkv/z row views, each driven through
// the general w8 SIMT kernel. Defined in w8_gdn_input_plan.cpp.
void w8_gdn_input_simt_row_view_split_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                             Tensor& z, cudaStream_t stream);
#endif // NINFER_VOLTA_BUILD

// Flash-Next decode fusion: one launch covers the whole per-rank Q8 qkvz parent
// (K=2560, Rows=8192), writes the four independent x-dtype outputs directly, and
// computes the 48 GDN gates from the BF16_CTRL [96,2560] control parent in the same
// grid (96 extra rows, one warp each, zero workspace). The gate formula and the
// FP64 oracle are unchanged from gdn_gating_proj.
void w8_gdn_qkvz_decode_launch(const Tensor& x, const Weight& weight, Tensor& query, Tensor& key,
                                Tensor& value, Tensor& z, const Weight& control,
                                const Tensor& a_log, const Tensor& dt_bias, Tensor& g,
                                Tensor& beta, cudaStream_t stream);
void w8_gdn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
