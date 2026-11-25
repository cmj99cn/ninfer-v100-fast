#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using Fp16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

constexpr int kFp16SmallTMaxTokens = 8;

#ifdef NINFER_VOLTA_BUILD
// The single sm_70 tensor-core route for native FP16 RowMajor weights. No split-K: the contraction
// is short enough that one split covers every registered vision projection, so no workspace exists.
void launch_fp16_volta_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_fp16_volta_mma_f16(const Tensor& x, const Weight& w, Tensor& out,
                               cudaStream_t stream);
void launch_fp16_volta_mma_small_t(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream);
#endif
// Exact-geometry T==1 decode GEMV (Flash-Next output head / PLE projections). FP16 activation
// and FP16 weights only; any other activation dtype keeps the Volta MMA route.
void launch_fp16_decode(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

// True batched small-T counterpart: one weight load per row/phase reused across `columns` token
// columns, raw-bit identical to a sequential replay of launch_fp16_decode.
void launch_fp16_decode_multi(const Tensor& x, const Weight& w, Tensor& out, std::int32_t columns,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
