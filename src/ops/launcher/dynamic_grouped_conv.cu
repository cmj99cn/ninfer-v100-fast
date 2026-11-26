// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime: ninfer::core Tensor/Weight operands, sm_70
// SIMT routes, and the Flash-Next draft geometry (H=2560, G=160) instead of the upstream 27B
// 5120/320 constants. The convolution mathematics (two causal taps over groups of 16, side 0
// prepare / side 1 residual finish) is unchanged.
#include "ops/launcher/dynamic_grouped_conv.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroupWidth = 16;
constexpr int kBlock      = 256;

template <typename T>
__device__ __forceinline__ float load(const T* p, std::int64_t i) {
    if constexpr (std::is_same_v<T, float>) {
        return p[i];
    } else {
        return __bfloat162float(p[i]);
    }
}

template <typename X, typename C>
__global__ void __launch_bounds__(kBlock)
    projected_conv_kernel(const X* __restrict__ x, const C* __restrict__ coefficients,
                          const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out,
                          std::int32_t hidden, std::int32_t width, std::int32_t side,
                          bool accumulate, std::int64_t elements) {
    const std::int64_t e = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= elements) return;
    const std::int64_t column = e / hidden;
    const std::int32_t h      = static_cast<std::int32_t>(e % hidden);
    const std::int32_t i      = static_cast<std::int32_t>(column % width);
    const std::int32_t groups = hidden / kGroupWidth;
    const std::int32_t g      = h / kGroupWidth;
    const float b0 = __bfloat162float(base[std::int64_t(side * 2 + 0) * hidden + h]);
    const C* row   = coefficients + column * 4 * groups;
    float y        = (b0 + load(row, std::int64_t(side * 2 + 0) * groups + g)) * load(x, e);
    if (i > 0) {
        const float b1 = __bfloat162float(base[std::int64_t(side * 2 + 1) * hidden + h]);
        y = fmaf(b1 + load(row, std::int64_t(side * 2 + 1) * groups + g), load(x, e - hidden), y);
    }
    out[e] = __float2bfloat16_rn(accumulate ? __bfloat162float(out[e]) + y : y);
}

__global__ void prepare_normalize_kernel(const __nv_bfloat16* __restrict__ residual,
                                         const __nv_bfloat16* __restrict__ norm_weight, float eps,
                                         __nv_bfloat16* __restrict__ normalized,
                                         std::int32_t hidden) {
    __shared__ float warp_sums[8];
    const int column = blockIdx.x;
    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const __nv_bfloat16* input = residual + static_cast<std::int64_t>(column) * hidden;
    float squares = 0.0f;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        const float value = __bfloat162float(input[h]);
        squares = fmaf(value, value, squares);
    }
    squares = warp_reduce_sum(squares);
    if (lane == 0) { warp_sums[warp] = squares; }
    __syncthreads();
    float total = (warp == 0 && lane < 8) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        total = warp_reduce_sum(total);
        if (lane == 0) { warp_sums[0] = total; }
    }
    __syncthreads();
    const float inverse =
        rsqrtf(warp_sums[0] / static_cast<float>(hidden) + eps);
    __nv_bfloat16* output = normalized + static_cast<std::int64_t>(column) * hidden;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        output[h] = __float2bfloat16_rn(__bfloat162float(input[h]) * inverse *
                                        __bfloat162float(norm_weight[h]));
    }
}

__global__ void prepare_conv_kernel(const __nv_bfloat16* __restrict__ normalized,
                                    const float* __restrict__ coefficients,
                                    const __nv_bfloat16* __restrict__ base,
                                    __nv_bfloat16* __restrict__ prepared, std::int32_t hidden,
                                    std::int32_t width, std::int64_t elements) {
    const std::int64_t e = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= elements) return;
    const std::int64_t column = e / hidden;
    const std::int32_t h      = static_cast<std::int32_t>(e % hidden);
    const std::int32_t i      = static_cast<std::int32_t>(column % width);
    const std::int32_t groups = hidden / kGroupWidth;
    const std::int32_t g      = h / kGroupWidth;
    const float* row          = coefficients + column * 4 * groups;
    float y = (__bfloat162float(base[std::int64_t(0) * hidden + h]) +
               row[std::int64_t(0) * groups + g]) *
              __bfloat162float(normalized[e]);
    if (i > 0) {
        y = fmaf(__bfloat162float(base[std::int64_t(1) * hidden + h]) +
                     row[std::int64_t(1) * groups + g],
                 __bfloat162float(normalized[e - hidden]), y);
    }
    prepared[e] = __float2bfloat16_rn(y);
}

__global__ void prepare_delta_kernel(const float* __restrict__ coefficients,
                                     __nv_bfloat16* __restrict__ finish_delta,
                                     std::int32_t groups, std::int64_t elements) {
    const std::int64_t e = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= elements) return;
    const std::int32_t g      = static_cast<std::int32_t>(e % groups);
    const std::int64_t column = e / groups;
    const float* row          = coefficients + column * 4 * groups;
    const std::int64_t dst    = column * 2 * groups + g;
    finish_delta[dst]         = __float2bfloat16_rn(row[std::int64_t(2) * groups + g]);
    finish_delta[dst + groups] = __float2bfloat16_rn(row[std::int64_t(3) * groups + g]);
}

__global__ void finish_add_kernel(const __nv_bfloat16* __restrict__ projected,
                                  const __nv_bfloat16* __restrict__ base,
                                  const __nv_bfloat16* __restrict__ finish_delta,
                                  __nv_bfloat16* __restrict__ residual, std::int32_t hidden,
                                  std::int32_t width, std::int64_t elements) {
    const std::int64_t e = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= elements) return;
    const std::int64_t column = e / hidden;
    const std::int32_t h      = static_cast<std::int32_t>(e % hidden);
    const std::int32_t i      = static_cast<std::int32_t>(column % width);
    const std::int32_t groups = hidden / kGroupWidth;
    const std::int32_t g      = h / kGroupWidth;
    const std::int64_t di     = column * 2 * groups + g;
    float value = fmaf(__bfloat162float(base[std::int64_t(2) * hidden + h]) +
                           __bfloat162float(finish_delta[di]),
                       __bfloat162float(projected[e]), __bfloat162float(residual[e]));
    if (i > 0) {
        value = fmaf(__bfloat162float(base[std::int64_t(3) * hidden + h]) +
                         __bfloat162float(finish_delta[di + groups]),
                     __bfloat162float(projected[e - hidden]), value);
    }
    residual[e] = __float2bfloat16_rn(value);
}

} // namespace

void dynamic_grouped_conv_launch(const Tensor& x, const Tensor& coefficients,
                                 const Tensor& base_kernel, std::int32_t side, Tensor& out,
                                 cudaStream_t stream) {
    const std::int64_t elements =
        static_cast<std::int64_t>(x.ne[0]) * x.ne[1] * x.ne[2];
    const unsigned blocks = static_cast<unsigned>((elements + kBlock - 1) / kBlock);
    const auto* base = static_cast<const __nv_bfloat16*>(base_kernel.data);
    auto* destination = static_cast<__nv_bfloat16*>(out.data);
    if (x.dtype == DType::FP32 && coefficients.dtype == DType::FP32) {
        projected_conv_kernel<float, float><<<blocks, kBlock, 0, stream>>>(
            static_cast<const float*>(x.data), static_cast<const float*>(coefficients.data), base,
            destination, x.ne[0], x.ne[1], side, side == 1, elements);
    } else if (x.dtype == DType::FP32) {
        projected_conv_kernel<float, __nv_bfloat16><<<blocks, kBlock, 0, stream>>>(
            static_cast<const float*>(x.data),
            static_cast<const __nv_bfloat16*>(coefficients.data), base, destination, x.ne[0],
            x.ne[1], side, side == 1, elements);
    } else if (coefficients.dtype == DType::FP32) {
        projected_conv_kernel<__nv_bfloat16, float><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(coefficients.data),
            base, destination, x.ne[0], x.ne[1], side, side == 1, elements);
    } else {
        projected_conv_kernel<__nv_bfloat16, __nv_bfloat16><<<blocks, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(coefficients.data), base, destination, x.ne[0],
            x.ne[1], side, side == 1, elements);
    }
    CUDA_CHECK(cudaGetLastError());
}

void prepare_normalize_launch(const Tensor& residual, const Tensor& norm_weight, float eps,
                              Tensor& normalized, std::int32_t hidden, std::int32_t columns,
                              cudaStream_t stream) {
    prepare_normalize_kernel<<<static_cast<unsigned>(columns), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(norm_weight.data), eps,
        static_cast<__nv_bfloat16*>(normalized.data), hidden);
    CUDA_CHECK(cudaGetLastError());
}

void prepare_conv_launch(const Tensor& normalized, const Tensor& coefficients_fp32,
                         const Tensor& base_kernel, Tensor& prepared, Tensor& finish_delta,
                         std::int32_t hidden, std::int32_t width, std::int32_t batch,
                         cudaStream_t stream) {
    const std::int64_t elements = static_cast<std::int64_t>(hidden) * width * batch;
    const unsigned blocks = static_cast<unsigned>((elements + kBlock - 1) / kBlock);
    prepare_conv_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const float*>(coefficients_fp32.data),
        static_cast<const __nv_bfloat16*>(base_kernel.data),
        static_cast<__nv_bfloat16*>(prepared.data), hidden, width, elements);
    CUDA_CHECK(cudaGetLastError());
    const std::int32_t groups = hidden / kGroupWidth;
    const std::int64_t delta_elements = static_cast<std::int64_t>(groups) * width * batch;
    const unsigned delta_blocks =
        static_cast<unsigned>((delta_elements + kBlock - 1) / kBlock);
    prepare_delta_kernel<<<delta_blocks, kBlock, 0, stream>>>(
        static_cast<const float*>(coefficients_fp32.data),
        static_cast<__nv_bfloat16*>(finish_delta.data), groups, delta_elements);
    CUDA_CHECK(cudaGetLastError());
}

void finish_add_launch(const Tensor& projected, const Tensor& base_kernel,
                       const Tensor& finish_delta, Tensor& residual, std::int32_t hidden,
                       std::int32_t width, std::int32_t batch, cudaStream_t stream) {
    const std::int64_t elements = static_cast<std::int64_t>(hidden) * width * batch;
    const unsigned blocks = static_cast<unsigned>((elements + kBlock - 1) / kBlock);
    finish_add_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(base_kernel.data),
        static_cast<const __nv_bfloat16*>(finish_delta.data),
        static_cast<__nv_bfloat16*>(residual.data), hidden, width, elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
