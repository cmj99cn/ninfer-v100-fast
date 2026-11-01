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
__global__ void gdn_gating_proj_f16_kernel(const __half* x, const W* a_weight,
                                           const W* b_weight, const float* A_log,
                                           const float* dt_bias, float* g, float* beta,
                                           int heads, int input_rows, int tokens) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= heads * tokens) { return; }
    const int head = index % heads;
    const int token = index / heads;
    float a = 0.0f, b = 0.0f;
    for (int k = 0; k < input_rows; ++k) {
        const float xv = __half2float(x[static_cast<std::int64_t>(token) * input_rows + k]);
        a += ActivationTraits<W>::load(a_weight + static_cast<std::int64_t>(head) * input_rows + k) * xv;
        b += ActivationTraits<W>::load(b_weight + static_cast<std::int64_t>(head) * input_rows + k) * xv;
    }
    g[index] = -expf(A_log[head]) * softplus_f(a + dt_bias[head]);
    beta[index] = 1.0f / (1.0f + expf(-b));
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
    require_f16(x, "x"); require_weight(a_weight, "a_weight"); require_weight(b_weight, "b_weight");
    if (a_weight.dtype != b_weight.dtype) throw std::invalid_argument("gdn_gating_proj_f16: weight dtype mismatch");
    require_f32(A_log, "A_log"); require_f32(dt_bias, "dt_bias");
    require_f32(g, "g"); require_f32(beta, "beta");
    const int input_rows = x.ne[0], tokens = x.ne[1], heads = a_weight.ne[0];
    if (b_weight.ne[0] != heads || a_weight.ne[1] != input_rows || b_weight.ne[1] != input_rows ||
        A_log.ne[0] != heads || dt_bias.ne[0] != heads || g.ne[0] != heads ||
        g.ne[1] != tokens || beta.ne[0] != heads || beta.ne[1] != tokens)
        throw std::invalid_argument("gdn_gating_proj_f16: shape mismatch");
    const int count = heads * tokens, blocks = (count + 255) / 256;
    if (a_weight.dtype == DType::BF16) {
        gdn_gating_proj_f16_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
            static_cast<const __half*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.data),
            static_cast<const __nv_bfloat16*>(b_weight.data),
            static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
            static_cast<float*>(g.data), static_cast<float*>(beta.data), heads, input_rows,
            tokens);
    } else {
        gdn_gating_proj_f16_kernel<__half><<<blocks, 256, 0, stream>>>(
            static_cast<const __half*>(x.data), static_cast<const __half*>(a_weight.data),
            static_cast<const __half*>(b_weight.data), static_cast<const float*>(A_log.data),
            static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
            static_cast<float*>(beta.data), heads, input_rows, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
