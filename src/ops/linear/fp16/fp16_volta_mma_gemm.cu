#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/fp16/fp16_launch.h"
#include "ops/linear/fp16/fp16_volta_mma_gemm.cuh"

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

template <class A, class Sched>
void launch_fp16_volta_mma_scheduled(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(std::uint16_t));

    const auto* wdata = static_cast<const __half*>(w.qdata);
    const dim3 grid(static_cast<unsigned>((n + Sched::kRowsPerCta - 1) / Sched::kRowsPerCta),
                    1u,
                    static_cast<unsigned>((t + Sched::kTTile - 1) / Sched::kTTile));
    fp16_volta_mma_gemm_kernel<A, Sched><<<grid, Sched::kThreads, 0, stream>>>(
        wdata, static_cast<const A*>(x.data), static_cast<A*>(out.data), out_ld, n, k, t);
    CUDA_CHECK(cudaGetLastError());
}

template <class A>
void launch_fp16_volta_mma_typed(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    if (out.ne[0] <= 1024 && x.ne[1] == 1) {
        launch_fp16_volta_mma_scheduled<A, Fp16VoltaMmaDecodeSchedule>(x, w, out, stream);
        return;
    }
    launch_fp16_volta_mma_scheduled<A, Fp16VoltaMmaSchedule>(x, w, out, stream);
}

void launch_fp16_volta_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp16_volta_mma_typed<__nv_bfloat16>(x, w, out, stream);
}

void launch_fp16_volta_mma_f16(const Tensor& x, const Weight& w, Tensor& out,
                               cudaStream_t stream) {
    launch_fp16_volta_mma_typed<__half>(x, w, out, stream);
}

void launch_fp16_volta_mma_small_t(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_fp16_volta_mma_scheduled<__half, Fp16VoltaMmaSmallTSchedule>(x, w, out, stream);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
