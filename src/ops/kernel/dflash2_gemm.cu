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

template <typename Out>
__device__ __forceinline__ void store_result(Out* out, std::int64_t index, float value) {
    if constexpr (std::is_same_v<Out, float>) {
        out[index] = value;
    } else {
        out[index] = __float2bfloat16_rn(value);
    }
}

template <typename Out, bool Quantized, int Columns, int Warps>
__global__ void __launch_bounds__(Warps * 32) gemm_row_warp_kernel(
    const __nv_bfloat16* __restrict__ x, const void* __restrict__ weight,
    const std::uint8_t* __restrict__ scales, Out* __restrict__ out, int n, int k, int t) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * Warps + warp;
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

    for (int tile = 0; tile < t; tile += Columns) {
        float acc[Columns];
#pragma unroll
        for (int j = 0; j < Columns; ++j) { acc[j] = 0.0F; }

        if constexpr (Quantized) {
            for (int group = 0; group < k / 32; ++group) {
                const float scale = __half2float(__ushort_as_half(
                    *reinterpret_cast<const std::uint16_t*>(row_scales + group * 2)));
                const int offset = group * 32 + lane;
                const float code = static_cast<float>(static_cast<std::int8_t>(codes[offset]));
                const float w    = code * scale;
#pragma unroll
                for (int j = 0; j < Columns; ++j) {
                    const __nv_bfloat16* column_x =
                        x + static_cast<std::int64_t>(tile + j) * k;
                    acc[j] = fmaf(w, __bfloat162float(column_x[offset]), acc[j]);
                }
            }
        } else {
            for (int offset = lane; offset < k; offset += 32) {
                const float w = __bfloat162float(dense[offset]);
#pragma unroll
                for (int j = 0; j < Columns; ++j) {
                    const __nv_bfloat16* column_x =
                        x + static_cast<std::int64_t>(tile + j) * k;
                    acc[j] = fmaf(w, __bfloat162float(column_x[offset]), acc[j]);
                }
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int j = 0; j < Columns; ++j) {
                acc[j] += __shfl_down_sync(0xffffffffU, acc[j], offset);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < Columns; ++j) {
                const std::int64_t index =
                    static_cast<std::int64_t>(tile + j) * n + row;
                store_result(out, index, acc[j]);
            }
        }
    }
}

template <typename Out, bool Quantized, int Columns, int Warps>
void launch_gemm_impl(const void* x, const void* weight, const std::uint8_t* scales, Out* out,
                      std::int32_t n, std::int32_t k, std::int32_t t, cudaStream_t stream) {
    const int blocks = (n + Warps - 1) / Warps;
    gemm_row_warp_kernel<Out, Quantized, Columns, Warps><<<blocks, Warps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), weight, scales, out, n, k, t);
    CUDA_CHECK(cudaGetLastError());
}

inline int tile_columns(int t) {
    for (int columns = 8; columns > 1; --columns) {
        if (t % columns == 0) { return columns; }
    }
    return 1;
}

inline int warps_for_rows(int n) {
    if (n >= 2560) { return 8; }
    if (n >= 1280) { return 4; }
    if (n >= 640)  { return 2; }
    return 1;
}

template <typename Out, bool Quantized, int Columns>
void dispatch_warps(const void* x, const void* weight, const std::uint8_t* scales, Out* out,
                    std::int32_t n, std::int32_t k, std::int32_t t, int warps,
                    cudaStream_t stream) {
    switch (warps) {
        case 1: launch_gemm_impl<Out, Quantized, Columns, 1>(x, weight, scales, out, n, k, t, stream); break;
        case 2: launch_gemm_impl<Out, Quantized, Columns, 2>(x, weight, scales, out, n, k, t, stream); break;
        case 4: launch_gemm_impl<Out, Quantized, Columns, 4>(x, weight, scales, out, n, k, t, stream); break;
        default: launch_gemm_impl<Out, Quantized, Columns, 8>(x, weight, scales, out, n, k, t, stream); break;
    }
}

template <typename Out, bool Quantized>
void launch_gemm(const void* x, const void* weight, const std::uint8_t* scales, Out* out,
                 std::int32_t n, std::int32_t k, std::int32_t t, cudaStream_t stream) {
    if (n <= 0 || t <= 0) { return; }
    const int columns = tile_columns(t);
    const int warps   = warps_for_rows(n);
    switch (columns) {
        case 1: dispatch_warps<Out, Quantized, 1>(x, weight, scales, out, n, k, t, warps, stream); break;
        case 2: dispatch_warps<Out, Quantized, 2>(x, weight, scales, out, n, k, t, warps, stream); break;
        case 3: dispatch_warps<Out, Quantized, 3>(x, weight, scales, out, n, k, t, warps, stream); break;
        case 4: dispatch_warps<Out, Quantized, 4>(x, weight, scales, out, n, k, t, warps, stream); break;
        case 5: dispatch_warps<Out, Quantized, 5>(x, weight, scales, out, n, k, t, warps, stream); break;
        case 6: dispatch_warps<Out, Quantized, 6>(x, weight, scales, out, n, k, t, warps, stream); break;
        case 7: dispatch_warps<Out, Quantized, 7>(x, weight, scales, out, n, k, t, warps, stream); break;
        default: dispatch_warps<Out, Quantized, 8>(x, weight, scales, out, n, k, t, warps, stream); break;
    }
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
