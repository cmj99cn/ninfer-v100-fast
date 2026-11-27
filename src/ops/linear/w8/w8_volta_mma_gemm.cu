#include "core/device.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_volta_mma_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = W8VoltaMmaSchedule;
} // namespace

bool w8_volta_mma_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // k % kGroupK keeps whole W8 groups inside a staging step (the scale is per group and read
    // once per step); k % 8 keeps the uint4 activation staging 16-byte aligned; and the loop
    // stages ahead by one step, so K has to be at least one step long.
    if (k % W8RowSplitStorage::kGroupK != 0 || k % 8 != 0) { return false; }
    return k >= S::kKStep;
}

template <class Act>
void launch_typed(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(Act));
    const std::int32_t padded_groups = w.padded_shape[1] / W8RowSplitStorage::kGroupK;

    const dim3 grid(static_cast<unsigned>((n + S::kRowsPerCta - 1) / S::kRowsPerCta),
                    static_cast<unsigned>((t + S::kTTile - 1) / S::kTTile));
    w8_volta_mma_gemm_kernel<Act><<<grid, S::kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        static_cast<const Act*>(x.data), static_cast<Act*>(out.data), n, k, t, padded_groups,
        out_ld);
    CUDA_CHECK(cudaGetLastError());
}

void launch_w8_volta_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_typed<__nv_bfloat16>(x, w, out, stream);
}

void launch_w8_volta_mma_f16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_typed<half>(x, w, out, stream);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
