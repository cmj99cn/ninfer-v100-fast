#include "ninfer/ops/causal_conv1d_silu_f16.h"

#include "core/device.h"
#include "ops/launcher/causal_conv1d.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ float load_weight(const half* p) { return __half2float(*p); }

__device__ __forceinline__ float load_weight(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <class W>
__global__ void conv_f16_kernel(const __half* x, const W* w, const __half* state, __half* out,
                                int C, int T) {
    const int t = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    const __half x3 = x[static_cast<std::int64_t>(t) * C + c];
    // Tap j reads the concatenated (-3,-2,-1, 0..T-1) stream at position t-3+j; a negative
    // position maps to old state entry (t+j). The offset by t matters when the incoming state is
    // non-zero and T>1 (verification windows), where the old T=1 decode only ever used index 0..2.
    const __half x0 = t >= 3 ? x[static_cast<std::int64_t>(t - 3) * C + c]
                             : state[static_cast<std::int64_t>(t) * C + c];
    const __half x1 = t >= 2 ? x[static_cast<std::int64_t>(t - 2) * C + c]
                             : state[static_cast<std::int64_t>(t + 1) * C + c];
    const __half x2 = t >= 1 ? x[static_cast<std::int64_t>(t - 1) * C + c]
                             : state[static_cast<std::int64_t>(t + 2) * C + c];
    float acc = 0.0f;
    acc += load_weight(w + c) * __half2float(x0);
    acc += load_weight(w + C + c) * __half2float(x1);
    acc += load_weight(w + 2 * C + c) * __half2float(x2);
    acc += load_weight(w + 3 * C + c) * __half2float(x3);
    out[static_cast<std::int64_t>(t) * C + c] = __float2half_rn(acc / (1.0f + expf(-acc)));
}

// The trailing window after consuming T columns is positions [T-3, T-1] of the concatenated
// (-3,-2,-1, 0..T-1) stream, so state_out[s] reads old state[T+s] when T+s < 3 and x[T+s-3]
// otherwise. All three old values are loaded before any store so the in-place form is safe.
__global__ void state_f16_kernel(const __half* x, const __half* state_in, __half* state_out,
                                 int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const __half old0 = state_in[c];
    const __half old1 = state_in[C + c];
    const __half old2 = state_in[2 * C + c];
    const auto pick = [&](int seq_pos) -> __half {
        if (seq_pos == 0) return old0;
        if (seq_pos == 1) return old1;
        if (seq_pos == 2) return old2;
        return x[static_cast<std::int64_t>(seq_pos - 3) * C + c];
    };
    state_out[c]         = pick(T);
    state_out[C + c]     = pick(T + 1);
    state_out[2 * C + c] = pick(T + 2);
}

template <class W>
__global__ void conv_state_f16_decode_kernel(const __half* x, const W* w,
                                            const __half* state_in, __half* state_out,
                                            __half* out, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) { return; }
    const __half old0 = state_in[c];
    const __half old1 = state_in[C + c];
    const __half old2 = state_in[2 * C + c];
    const __half x3   = x[c];
    float acc         = 0.0f;
    acc += load_weight(w + c) * __half2float(old0);
    acc += load_weight(w + C + c) * __half2float(old1);
    acc += load_weight(w + 2 * C + c) * __half2float(old2);
    acc += load_weight(w + 3 * C + c) * __half2float(x3);
    out[c]        = __float2half_rn(acc / (1.0f + expf(-acc)));
    state_out[c]     = old1;
    state_out[C + c] = old2;
    state_out[2 * C + c] = x3;
}

// True batched counterpart of conv_state_f16_decode_kernel. The four convolution weights are
// fetched once per channel and reused across the whole verify span. The convolution state is a
// mathematical chain, so it advances column by column in registers; the three carried values are
// the exact FP16 values the sequential T==1 launches would reload from state_out. The four-tap
// accumulation order, silu rounding and output rounding are unchanged.
template <class W, int Columns>
__global__ void conv_state_f16_decode_multi_kernel(const __half* x, const W* w,
                                                   const __half* state_in, __half* state_out,
                                                   __half* out, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) { return; }
    const float w0 = load_weight(w + c);
    const float w1 = load_weight(w + C + c);
    const float w2 = load_weight(w + 2 * C + c);
    const float w3 = load_weight(w + 3 * C + c);
    __half s0 = state_in[c];
    __half s1 = state_in[C + c];
    __half s2 = state_in[2 * C + c];
#pragma unroll
    for (int t = 0; t < Columns; ++t) {
        const __half x3 = x[static_cast<std::int64_t>(t) * C + c];
        float acc       = 0.0f;
        acc += w0 * __half2float(s0);
        acc += w1 * __half2float(s1);
        acc += w2 * __half2float(s2);
        acc += w3 * __half2float(x3);
        out[static_cast<std::int64_t>(t) * C + c] = __float2half_rn(acc / (1.0f + expf(-acc)));
        s0 = s1;
        s1 = s2;
        s2 = x3;
    }
    state_out[c]         = s0;
    state_out[C + c]     = s1;
    state_out[2 * C + c] = s2;
}

template <class W, int Columns>
void launch_conv_multi(const __half* xp, const W* wp, const __half* sp, __half* sop, __half* op,
                       int C, int threads, int blocks_c, cudaStream_t stream) {
    conv_state_f16_decode_multi_kernel<W, Columns><<<blocks_c, threads, 0, stream>>>(
        xp, wp, sp, sop, op, C);
}

template <class W>
void dispatch_conv_multi(const __half* xp, const W* wp, const __half* sp, __half* sop, __half* op,
                         int C, int T, int threads, int blocks_c, cudaStream_t stream) {
    switch (T) {
    case 2: launch_conv_multi<W, 2>(xp, wp, sp, sop, op, C, threads, blocks_c, stream); break;
    case 3: launch_conv_multi<W, 3>(xp, wp, sp, sop, op, C, threads, blocks_c, stream); break;
    case 4: launch_conv_multi<W, 4>(xp, wp, sp, sop, op, C, threads, blocks_c, stream); break;
    case 5: launch_conv_multi<W, 5>(xp, wp, sp, sop, op, C, threads, blocks_c, stream); break;
    case 6: launch_conv_multi<W, 6>(xp, wp, sp, sop, op, C, threads, blocks_c, stream); break;
    case 7: launch_conv_multi<W, 7>(xp, wp, sp, sop, op, C, threads, blocks_c, stream); break;
    case 8: launch_conv_multi<W, 8>(xp, wp, sp, sop, op, C, threads, blocks_c, stream); break;
    default: throw std::invalid_argument("causal_conv1d_f16 multi: unsupported column count");
    }
}

} // namespace

void causal_conv1d_silu_f16_split_launch(const Tensor& x, const Tensor& weight,
                                         const Tensor& state_in, Tensor& state_out, Tensor& out,
                                         cudaStream_t stream) {
    const int C = x.ne[0];
    const int T = x.ne[1];
    const int threads  = 256;
    const int blocks_c = (C + threads - 1) / threads;
    const dim3 grid(T, blocks_c);
    const auto* xp = static_cast<const __half*>(x.data);
    const auto* sp = static_cast<const __half*>(state_in.data);
    auto* op       = static_cast<__half*>(out.data);
    if (weight.dtype == DType::BF16) {
        conv_f16_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
            xp, static_cast<const __nv_bfloat16*>(weight.data), sp, op, C, T);
    } else {
        conv_f16_kernel<half><<<grid, threads, 0, stream>>>(
            xp, static_cast<const half*>(weight.data), sp, op, C, T);
    }
    state_f16_kernel<<<blocks_c, threads, 0, stream>>>(
        xp, sp, static_cast<__half*>(state_out.data), C, T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_silu_f16_launch(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                                   Tensor& state_out, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::FP16 || state_in.dtype != DType::FP16 ||
        state_out.dtype != DType::FP16 || out.dtype != DType::FP16 ||
        (weight.dtype != DType::FP16 && weight.dtype != DType::BF16) || !x.is_contiguous() ||
        !weight.is_contiguous() || !state_in.is_contiguous() || !state_out.is_contiguous() ||
        !out.is_contiguous() || x.data == nullptr || weight.data == nullptr ||
        state_in.data == nullptr || state_out.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("causal_conv1d_silu_f16: invalid FP16/weight tensors");
    }
    const int C = x.ne[0];
    const int T = x.ne[1];
    if (weight.ne[0] != C || weight.ne[1] != 4 || state_in.ne[0] != 3 ||
        state_in.ne[1] != C || state_out.ne[0] != 3 || state_out.ne[1] != C || out.ne[0] != C ||
        out.ne[1] != T) {
        throw std::invalid_argument("causal_conv1d_silu_f16: shape mismatch");
    }
    const int threads  = 256;
    const int blocks_c = (C + threads - 1) / threads;
    const dim3 grid(T, blocks_c);
    const auto* xp = static_cast<const __half*>(x.data);
    const auto* sp = static_cast<const __half*>(state_in.data);
    auto* op       = static_cast<__half*>(out.data);
    if (T == 1) {
        auto* sop = static_cast<__half*>(state_out.data);
        if (weight.dtype == DType::BF16) {
            conv_state_f16_decode_kernel<__nv_bfloat16><<<blocks_c, threads, 0, stream>>>(
                xp, static_cast<const __nv_bfloat16*>(weight.data), sp, sop, op, C);
        } else {
            conv_state_f16_decode_kernel<half><<<blocks_c, threads, 0, stream>>>(
                xp, static_cast<const half*>(weight.data), sp, sop, op, C);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (T >= 2 && T <= 8) {
        auto* sop = static_cast<__half*>(state_out.data);
        if (weight.dtype == DType::BF16) {
            dispatch_conv_multi<__nv_bfloat16>(xp, static_cast<const __nv_bfloat16*>(weight.data),
                                               sp, sop, op, C, T, threads, blocks_c, stream);
        } else {
            dispatch_conv_multi<half>(xp, static_cast<const half*>(weight.data), sp, sop, op, C, T,
                                      threads, blocks_c, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (weight.dtype == DType::BF16) {
        conv_f16_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
            xp, static_cast<const __nv_bfloat16*>(weight.data), sp, op, C, T);
    } else {
        conv_f16_kernel<half><<<grid, threads, 0, stream>>>(
            xp, static_cast<const half*>(weight.data), sp, op, C, T);
    }
    state_f16_kernel<<<blocks_c, threads, 0, stream>>>(xp, sp,
                                                       static_cast<__half*>(state_out.data), C, T);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

namespace ninfer::ops {

void causal_conv1d_silu_f16(const Tensor& x, const Tensor& weight, Tensor& state, Tensor& out,
                            cudaStream_t stream) {
    detail::causal_conv1d_silu_f16_launch(x, weight, state, state, out, stream);
}

} // namespace ninfer::ops
