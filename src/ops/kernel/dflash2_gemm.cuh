#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Straightforward DFlash2 projection helpers. `x` is contiguous BF16 [K,T] and `out` is
// contiguous BF16 [N,T]. The W8 form decodes the stored signed INT8 codes with the exact FP16
// group-32 scale (row-major codes plane, then group scales) and accumulates in FP32. The BF16 form
// accumulates a contiguous BF16 [N,K] weight in FP32. Both are single-token-column SIMT routes
// used by the op layer; they do not repack or stage weights.

void dflash2_w8_gemm(const void* x, const std::uint8_t* codes, const std::uint8_t* scales,
                     void* out, std::int32_t n, std::int32_t k, std::int32_t t,
                     cudaStream_t stream);

void dflash2_bf16_f32_gemm(const void* x, const void* weight, float* out, std::int32_t n,
                           std::int32_t k, std::int32_t t, cudaStream_t stream);

// W8 [N,K] weight into an FP32 [N,T] plane (used where a following transform needs the exact
// decoded FP32 values).
void dflash2_w8_gemm_f32(const void* x, const std::uint8_t* codes, const std::uint8_t* scales,
                         float* out, std::int32_t n, std::int32_t k, std::int32_t t,
                         cudaStream_t stream);

// BF16 x [K,T] with BF16 weight [N,K] into BF16 out [N,T].
void dflash2_bf16_gemm(const void* x, const void* weight, void* out, std::int32_t n,
                       std::int32_t k, std::int32_t t, cudaStream_t stream);

} // namespace ninfer::ops::detail
