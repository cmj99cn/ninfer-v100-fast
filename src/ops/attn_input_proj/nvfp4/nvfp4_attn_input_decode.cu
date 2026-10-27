#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
struct Nvfp4AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        using Sections                     = Nvfp4AttnInputSections<Geometry>;
        constexpr std::int32_t kQueryRows  = Sections::kQueryRows;
        constexpr std::int32_t kKeyRows    = Sections::kKeyRows;
        constexpr std::int32_t kGateRows   = Sections::kQueryRows;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

        const __nv_bfloat16 result_bf16 = __float2bfloat16_rn(result);
        // Section tensors are {sectionRows, tokens}; token is the position inside the current
        // chunk, matching the plan's per-chunk pointer arithmetic. At T=1 token is always 0, so
        // the decode route is bit-identical to the old token-ignoring store.
        if (parent_row < kKeyBegin) {
            query[static_cast<std::int64_t>(token) * kQueryRows + parent_row] = result_bf16;
        } else if (parent_row < kGateBegin) {
            key[static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin] = result_bf16;
        } else if (parent_row < kValueBegin) {
            gate[static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin] =
                result_bf16;
        } else {
            value[static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin] =
                result_bf16;
        }
    }
};

template <class Geometry>
void decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                   Tensor& v, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    using Sections = Nvfp4AttnInputSections<Geometry>;
    static_assert((Sections::kQueryRows % 128) == 0);
    static_assert((Sections::kKeyRows % 128) == 0);

    const Nvfp4AttentionInputOutput<Geometry> output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
#ifdef NINFER_VOLTA_BUILD
    // QPN tensor-core route over checkpoint-native codes/scales for any chunk the schedule
    // supports (t <= 32 = kNvfp4VoltaQpnMaxTokens). The token-indexed output policy makes this
    // valid for prefill chunks, not just T=1 decode; SIMT gemv is the non-Volta/fallback path.
    const auto t = static_cast<std::int32_t>(x.ne[1]);
    if (nvfp4_volta_qpn_supported(Geometry::kOutputRows, Geometry::kInputRows, t)) {
        launch_nvfp4_volta_qpn_with_output(x, weight, output, Geometry::kOutputRows,
                                           inverse_weight_divisor, stream);
        return;
    }
#endif
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    decode_launch<Nvfp4AttnInputGeometry>(x, weight, q, gate, k, v, stream);
}

void nvfp4_attn_input_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    decode_launch<Nvfp4AttnInputTp2ColumnGeometry>(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
