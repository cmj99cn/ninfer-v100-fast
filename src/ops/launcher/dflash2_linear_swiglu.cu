// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime: the fused gate/up projection plus SwiGLU is
// expressed over this repository's W8G32_F16S RowSplit weight and an FP32-accumulating sm_70 SIMT
// route, at the real Flash-Next drafter geometry [15360,2560]. The SiLU-multiply epilogue is
// unchanged.
#include "ops/launcher/dflash2_linear_swiglu.h"

#include "core/device.h"
#include "ops/kernel/dflash2_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

__global__ void swiglu_kernel(__nv_bfloat16* __restrict__ up,
                              const __nv_bfloat16* __restrict__ gate, std::int64_t elements) {
    const std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) { return; }
    const float gate_value = __bfloat162float(gate[index]);
    const float silu       = gate_value / (1.0f + __expf(-gate_value));
    up[index] = __float2bfloat16_rn(silu * __bfloat162float(up[index]));
}

} // namespace

void dflash2_linear_swiglu_launch(const Tensor& x, const Weight& parent, Tensor& out,
                                  Tensor& gate, cudaStream_t stream) {
    const std::int32_t columns = x.ne[0];
    const std::int32_t tokens  = x.ne[1];
    const auto* codes_base  = static_cast<const std::uint8_t*>(parent.qdata);
    const auto* scales_base = static_cast<const std::uint8_t*>(parent.scales);
    dflash2_w8_gemm(x.data, codes_base, scales_base, gate.data, 7680, columns, tokens, stream);
    dflash2_w8_gemm(x.data, codes_base + static_cast<std::int64_t>(7680) * columns,
                    scales_base + static_cast<std::int64_t>(7680) * (columns / 32) * 2, out.data,
                    7680, columns, tokens, stream);
    const std::int64_t elements = static_cast<std::int64_t>(7680) * tokens;
    swiglu_kernel<<<static_cast<unsigned>((elements + kBlock - 1) / kBlock), kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), static_cast<const __nv_bfloat16*>(gate.data),
        elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
