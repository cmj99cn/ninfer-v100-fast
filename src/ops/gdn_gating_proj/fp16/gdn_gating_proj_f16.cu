#include "ninfer/ops/gdn_gating_proj_f16.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ops/common/activation.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
__device__ __forceinline__ float softplus_f(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

template <class W>
__device__ __forceinline__ float weight_to_float(W v);
template <>
__device__ __forceinline__ float weight_to_float<__half>(__half v) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float weight_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// One CTA owns one output head and a tile of up to kWarpTile tokens. The head's two weight rows
// are staged once in shared memory (the naive sibling re-read them, uncoalesced, once per thread),
// then one warp per token performs a coalesced K reduction and a shuffle reduction. Decode (T=1)
// uses one warp; prefill uses all eight.
inline constexpr int kGatingThreads = 256;
inline constexpr int kGatingWarps   = kGatingThreads / 32;

template <class W>
__global__ void gdn_gating_proj_f16_tiled_kernel(const __half* __restrict__ x,
                                                 const W* __restrict__ a_weight,
                                                 const W* __restrict__ b_weight,
                                                 const float* __restrict__ A_log,
                                                 const float* __restrict__ dt_bias,
                                                 float* __restrict__ g, float* __restrict__ beta,
                                                 int heads, int input_rows, int tokens) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    W* sa = reinterpret_cast<W*>(smem_raw);
    W* sb = sa + input_rows;

    const int head = blockIdx.x;
    const W* arow = a_weight + static_cast<std::int64_t>(head) * input_rows;
    const W* brow = b_weight + static_cast<std::int64_t>(head) * input_rows;
    for (int k = threadIdx.x; k < input_rows; k += blockDim.x) {
        sa[k] = arow[k];
        sb[k] = brow[k];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int token = blockIdx.y * kGatingWarps + warp;
    if (token >= tokens) { return; }

    float a = 0.0f, b = 0.0f;
    const __half* xrow = x + static_cast<std::int64_t>(token) * input_rows;
    for (int k = lane; k < input_rows; k += 32) {
        const float xv = __half2float(xrow[k]);
        a = fmaf(weight_to_float<W>(sa[k]), xv, a);
        b = fmaf(weight_to_float<W>(sb[k]), xv, b);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, offset);
        b += __shfl_down_sync(0xffffffffu, b, offset);
    }
    if (lane != 0) { return; }

    const std::int64_t index = static_cast<std::int64_t>(token) * heads + head;
    g[index]                 = -expf(A_log[head]) * softplus_f(a + dt_bias[head]);
    beta[index]              = 1.0f / (1.0f + expf(-b));
}

void require_f16(const Tensor& t, const char* name) {
    if (t.dtype != DType::FP16 || !t.is_contiguous() || t.data == nullptr)
        throw std::invalid_argument(std::string("gdn_gating_proj_f16: invalid ") + name);
}
void require_weight(const Tensor& t, const char* name) {
    if ((t.dtype != DType::FP16 && t.dtype != DType::BF16) || !t.is_contiguous() ||
        t.data == nullptr) {
        throw std::invalid_argument(std::string("gdn_gating_proj_f16: invalid ") + name);
    }
}
void require_f32(const Tensor& t, const char* name) {
    if (t.dtype != DType::FP32 || !t.is_contiguous() || t.data == nullptr)
        throw std::invalid_argument(std::string("gdn_gating_proj_f16: invalid ") + name);
}
} // namespace

void gdn_gating_proj_f16(const Tensor& x, const Tensor& a_weight, const Tensor& b_weight,
                         const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta,
                         cudaStream_t stream) {
    require_f16(x, "x");
    require_weight(a_weight, "a_weight");
    require_weight(b_weight, "b_weight");
    if (a_weight.dtype != b_weight.dtype)
        throw std::invalid_argument("gdn_gating_proj_f16: weight dtype mismatch");
    require_f32(A_log, "A_log");
    require_f32(dt_bias, "dt_bias");
    require_f32(g, "g");
    require_f32(beta, "beta");
    const int input_rows = x.ne[0], tokens = x.ne[1], heads = a_weight.ne[0];
    if (b_weight.ne[0] != heads || a_weight.ne[1] != input_rows || b_weight.ne[1] != input_rows ||
        A_log.ne[0] != heads || dt_bias.ne[0] != heads || g.ne[0] != heads ||
        g.ne[1] != tokens || beta.ne[0] != heads || beta.ne[1] != tokens)
        throw std::invalid_argument("gdn_gating_proj_f16: shape mismatch");
    if (input_rows <= 0 || tokens <= 0 || heads <= 0) {
        throw std::invalid_argument("gdn_gating_proj_f16: empty problem");
    }
    const dim3 grid(static_cast<unsigned>(heads),
                    static_cast<unsigned>((tokens + kGatingWarps - 1) / kGatingWarps));
    const std::size_t shmem = static_cast<std::size_t>(2) * input_rows * sizeof(__half);
    if (a_weight.dtype == DType::BF16) {
        gdn_gating_proj_f16_tiled_kernel<__nv_bfloat16><<<grid, kGatingThreads, shmem, stream>>>(
            static_cast<const __half*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.data),
            static_cast<const __nv_bfloat16*>(b_weight.data),
            static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
            static_cast<float*>(g.data), static_cast<float*>(beta.data), heads, input_rows,
            tokens);
    } else {
        gdn_gating_proj_f16_tiled_kernel<__half><<<grid, kGatingThreads, shmem, stream>>>(
            static_cast<const __half*>(x.data), static_cast<const __half*>(a_weight.data),
            static_cast<const __half*>(b_weight.data), static_cast<const float*>(A_log.data),
            static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
            static_cast<float*>(beta.data), heads, input_rows, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
