#include "ninfer/ops/causal_conv1d_silu_f16.h"

#include "core/device.h"

#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

__global__ void conv_f16_kernel(const __half* x, const __half* w, const __half* state,
                                __half* out, int C, int T) {
    const int t = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    const __half x3 = x[static_cast<std::int64_t>(t) * C + c];
    const __half x0 = t >= 3 ? x[static_cast<std::int64_t>(t - 3) * C + c] : state[c];
    const __half x1 = t >= 2 ? x[static_cast<std::int64_t>(t - 2) * C + c] : state[C + c];
    const __half x2 = t >= 1 ? x[static_cast<std::int64_t>(t - 1) * C + c] : state[2 * C + c];
    float acc = 0.0f;
    acc += __half2float(w[c]) * __half2float(x0);
    acc += __half2float(w[C + c]) * __half2float(x1);
    acc += __half2float(w[2 * C + c]) * __half2float(x2);
    acc += __half2float(w[3 * C + c]) * __half2float(x3);
    out[static_cast<std::int64_t>(t) * C + c] = __float2half_rn(silu_f(acc));
}

__global__ void state_f16_kernel(const __half* x, __half* state, int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    if (T >= 3) {
        state[c] = x[static_cast<std::int64_t>(T - 3) * C + c];
        state[C + c] = x[static_cast<std::int64_t>(T - 2) * C + c];
        state[2 * C + c] = x[static_cast<std::int64_t>(T - 1) * C + c];
    } else if (T == 2) {
        state[c] = state[C + c];
        state[C + c] = state[2 * C + c];
        state[2 * C + c] = x[C + c];
    } else {
        state[c] = state[C + c];
        state[C + c] = state[2 * C + c];
        state[2 * C + c] = x[c];
    }
}
} // namespace

void causal_conv1d_silu_f16(const Tensor& x, const Tensor& weight, Tensor& state, Tensor& out,
                            cudaStream_t stream) {
    if (x.dtype != DType::FP16 || weight.dtype != DType::FP16 || state.dtype != DType::FP16 ||
        out.dtype != DType::FP16 || !x.is_contiguous() || !weight.is_contiguous() ||
        !state.is_contiguous() || !out.is_contiguous() || x.data == nullptr ||
        weight.data == nullptr || state.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("causal_conv1d_silu_f16: invalid FP16 tensors");
    }
    const int C = x.ne[0];
    const int T = x.ne[1];
    if (weight.ne[0] != C || weight.ne[1] != 4 || state.ne[0] != 3 || state.ne[1] != C ||
        out.ne[0] != C || out.ne[1] != T) {
        throw std::invalid_argument("causal_conv1d_silu_f16: shape mismatch");
    }
    const int threads = 256;
    const int blocks_c = (C + threads - 1) / threads;
    const dim3 grid(T, blocks_c);
    conv_f16_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __half*>(x.data), static_cast<const __half*>(weight.data),
        static_cast<const __half*>(state.data), static_cast<__half*>(out.data), C, T);
    state_f16_kernel<<<blocks_c, threads, 0, stream>>>(
        static_cast<const __half*>(x.data), static_cast<__half*>(state.data), C, T);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
