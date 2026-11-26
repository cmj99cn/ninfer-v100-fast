// The DFlash2 drafter's small-token projection kernels.
//
// One warp owns one output row and all of its columns. Lanes stride the contraction axis, so a
// row's W8 codes/scales (or BF16 weights) are read coalesced once per column tile instead of once
// per output element, and the partial sums fold through a warp tree. The activation columns of a
// tile are read by every row's warp and stay in cache. The result is written column-major, the same
// layout as the plain linear Op.
#include "ops/kernel/dflash2_gemm.cuh"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarpsPerBlock   = 8;
constexpr int kColumnsPerTile  = 8;
constexpr int kBlock           = kWarpsPerBlock * 32;

template <typename Out>
__device__ __forceinline__ void store_result(Out* out, std::int64_t index, float value) {
    if constexpr (std::is_same_v<Out, float>) {
        out[index] = value;
    } else {
        out[index] = __float2bfloat16_rn(value);
    }
}

// `quantized` selects the W8G32_F16S row-split weight (codes + per-32 FP16 scales) or a plain BF16
// row-major weight. Both are read row by row; `k` is a multiple of 32 in the quantized case.
template <typename Out, bool Quantized>
__global__ void __launch_bounds__(kBlock) gemm_row_warp_kernel(
    const __nv_bfloat16* __restrict__ x, const void* __restrict__ weight,
    const std::uint8_t* __restrict__ scales, Out* __restrict__ out, int n, int k, int t) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n) { return; }

    const std::uint8_t* codes = nullptr;
    const std::uint8_t* row_scales = nullptr;
    const __nv_bfloat16* dense = nullptr;
    if constexpr (Quantized) {
        codes      = static_cast<const std::uint8_t*>(weight) + static_cast<std::int64_t>(row) * k;
        row_scales = scales + static_cast<std::int64_t>(row) * (k / 32) * 2;
    } else {
        dense = static_cast<const __nv_bfloat16*>(weight) + static_cast<std::int64_t>(row) * k;
    }

    for (int tile = 0; tile < t; tile += kColumnsPerTile) {
        const int columns = min(kColumnsPerTile, t - tile);
        float acc[kColumnsPerTile];
#pragma unroll
        for (int j = 0; j < kColumnsPerTile; ++j) { acc[j] = 0.0F; }

        if constexpr (Quantized) {
            for (int group = 0; group < k / 32; ++group) {
                const float scale = __half2float(__ushort_as_half(
                    *reinterpret_cast<const std::uint16_t*>(row_scales + group * 2)));
                const int offset = group * 32 + lane;
                const float code = static_cast<float>(static_cast<std::int8_t>(codes[offset]));
                const float w    = code * scale;
#pragma unroll
                for (int j = 0; j < kColumnsPerTile; ++j) {
                    if (j >= columns) { continue; }
                    const __nv_bfloat16* column_x =
                        x + static_cast<std::int64_t>(tile + j) * k;
                    acc[j] = fmaf(w, __bfloat162float(column_x[offset]), acc[j]);
                }
            }
        } else {
            for (int offset = lane; offset < k; offset += 32) {
                const float w = __bfloat162float(dense[offset]);
#pragma unroll
                for (int j = 0; j < kColumnsPerTile; ++j) {
                    if (j >= columns) { continue; }
                    const __nv_bfloat16* column_x =
                        x + static_cast<std::int64_t>(tile + j) * k;
                    acc[j] = fmaf(w, __bfloat162float(column_x[offset]), acc[j]);
                }
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int j = 0; j < kColumnsPerTile; ++j) {
                acc[j] += __shfl_down_sync(0xffffffffU, acc[j], offset);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < kColumnsPerTile; ++j) {
                if (j >= columns) { continue; }
                const std::int64_t index =
                    static_cast<std::int64_t>(tile + j) * n + row;
                store_result(out, index, acc[j]);
            }
        }
    }
}

template <typename Out, bool Quantized>
void launch_gemm(const void* x, const void* weight, const std::uint8_t* scales, Out* out,
                 std::int32_t n, std::int32_t k, std::int32_t t, cudaStream_t stream) {
    const int blocks = (n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    gemm_row_warp_kernel<Out, Quantized><<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), weight, scales, out, n, k, t);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void dflash2_w8_gemm(const void* x, const std::uint8_t* codes, const std::uint8_t* scales,
                     void* out, std::int32_t n, std::int32_t k, std::int32_t t,
                     cudaStream_t stream) {
    launch_gemm<__nv_bfloat16, true>(x, codes, scales, static_cast<__nv_bfloat16*>(out), n, k, t,
                                     stream);
}

void dflash2_w8_gemm_f32(const void* x, const std::uint8_t* codes, const std::uint8_t* scales,
                         float* out, std::int32_t n, std::int32_t k, std::int32_t t,
                         cudaStream_t stream) {
    launch_gemm<float, true>(x, codes, scales, out, n, k, t, stream);
}

void dflash2_bf16_f32_gemm(const void* x, const void* weight, float* out, std::int32_t n,
                           std::int32_t k, std::int32_t t, cudaStream_t stream) {
    launch_gemm<float, false>(x, weight, nullptr, out, n, k, t, stream);
}

void dflash2_bf16_gemm(const void* x, const void* weight, void* out, std::int32_t n,
                       std::int32_t k, std::int32_t t, cudaStream_t stream) {
    launch_gemm<__nv_bfloat16, false>(x, weight, nullptr, static_cast<__nv_bfloat16*>(out), n, k, t,
                                      stream);
}

} // namespace ninfer::ops::detail
