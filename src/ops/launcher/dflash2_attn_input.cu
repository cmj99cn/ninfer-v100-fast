// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime: the fused Q/K/V projection is expressed over
// this repository's W8G32_F16S RowSplit weight and an FP32-accumulating sm_70 SIMT route, at the
// real Flash-Next drafter geometry [6144,2560] with row split [4096,1024,1024]. The projection is
// otherwise the same raw linear map (no normalization or rotation).
#include "ops/launcher/dflash2_attn_input.h"

#include "ops/kernel/dflash2_gemm.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

void project_rows(const Tensor& x, const Weight& parent, Tensor& out, std::int32_t row_offset,
                  std::int32_t rows, std::int32_t columns, std::int32_t tokens,
                  cudaStream_t stream) {
    const auto* codes = static_cast<const std::uint8_t*>(parent.qdata) +
                        static_cast<std::int64_t>(row_offset) * columns;
    const auto* scales = static_cast<const std::uint8_t*>(parent.scales) +
                         static_cast<std::int64_t>(row_offset) * (columns / 32) * 2;
    dflash2_w8_gemm(x.data, codes, scales, out.data, rows, columns, tokens, stream);
}

} // namespace

void dflash2_attn_input_proj_launch(const Tensor& x, const Weight& parent, Tensor& q, Tensor& k,
                                    Tensor& v, cudaStream_t stream) {
    const std::int32_t columns = x.ne[0];
    const std::int32_t tokens  = x.ne[1];
    project_rows(x, parent, q, 0, 4096, columns, tokens, stream);
    project_rows(x, parent, k, 4096, 1024, columns, tokens, stream);
    project_rows(x, parent, v, 5120, 1024, columns, tokens, stream);
}

} // namespace ninfer::ops::detail
