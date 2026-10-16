#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Output>
void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    const Output output{static_cast<__nv_bfloat16*>(qkv.data),
                        static_cast<__nv_bfloat16*>(z.data)};
#ifdef NINFER_VOLTA_BUILD
    // QPN tensor-core route over checkpoint-native codes/scales for any chunk the schedule
    // supports (t <= 32 = kNvfp4VoltaQpnMaxTokens); the plan only invokes this launch for
    // supported shapes. Nvfp4GdnInputOutput is token-indexed, so prefill chunks are valid;
    // SIMT gemv is the non-Volta/fallback path (T=1 only by contract).
    const auto t = static_cast<std::int32_t>(x.ne[1]);
    if (nvfp4_volta_qpn_supported(Geometry::kOutputRows, Geometry::kInputRows, t)) {
        launch_nvfp4_volta_qpn_with_output(x, weight, output, Geometry::kOutputRows, inverse,
                                           stream);
        return;
    }
#endif
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    launch<Nvfp4GdnInputGeometry, Nvfp4GdnInputOutput>(x, weight, qkv, z, stream);
}

// The tp2 column shard, instantiated at Nvfp4GdnInputTp2ColumnGeometry.
void nvfp4_gdn_input_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, cudaStream_t stream) {
    launch<Nvfp4GdnInputTp2ColumnGeometry, Nvfp4GdnInputShardOutput<Nvfp4GdnInputTp2ColumnGeometry>>(
        x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
