#include "ops/linear/fp16/fp16_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Fp16Launch select_fp16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        break;
    case LinearPolicy::AllowA4:
        throw std::invalid_argument("fp16 linear: unsupported policy");
    }
    // Native FP16 RowMajor weights. The tensor-core tile loads B in uint4 (8-half) units, so the
    // contraction must be a multiple of 8; N and T are bounds-checked in the kernel. A K that is not
    // a multiple of the 32-wide tile (fc2, k=4304) zero-pads its shared tail, so any k%8==0 works.
    if (n <= 0 || k <= 0 || t <= 0 || k % 8 != 0) {
        throw std::invalid_argument("fp16 linear: unsupported shape or T");
    }
#ifdef NINFER_VOLTA_BUILD
    return launch_fp16_volta_mma;
#else
    throw std::invalid_argument("fp16 linear: requires the sm_70 build");
#endif
}

void fp16_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    // Exact-geometry FP16 decode instances for the Flash-Next output head (Vocab-split) and PLE
    // projections. The 2..T<=8 band uses a true batched kernel: one weight load per row/phase is
    // reused by every token column, while each column keeps the exact T==1 accumulator/chain order,
    // warp reduction and from_float rounding. T==1 remains the original launch_fp16_decode verbatim.
    const bool exact_decode =
        (w.n == 124160 && w.k == 2560) || (w.n == 10240 && w.k == 2560) ||
        (w.n == 2560 && w.k == 2560);
    if (x.dtype == DType::FP16 && exact_decode) {
        if (x.ne[1] == 1) {
            launch_fp16_decode(x, w, out, stream);
            return;
        }
        if (x.ne[1] >= 2 && x.ne[1] <= kFp16SmallTMaxTokens) {
            launch_fp16_decode_multi(x, w, out, x.ne[1], stream);
            return;
        }
    }
    // FP16 geometries with no bit-exact multi decode keep the established per-column replay of the
    // T==1 route. Routing them to the generic T>1 MMA kernel is not merely slower here (e.g. the
    // qsa key/value n=1664 shard), it also changes the FP32 reduction order across token columns.
    if (x.dtype == DType::FP16 && x.ne[1] >= 2 && x.ne[1] <= kFp16SmallTMaxTokens) {
        for (std::int32_t c = 0; c < x.ne[1]; ++c) {
            const Tensor column = x.slice(1, c, 1);
            Tensor out_column   = out.slice(1, c, 1);
            fp16_dispatch(column, w, out_column, policy, stream);
        }
        return;
    }
#endif
    (void)select_fp16_launch(w.n, w.k, x.ne[1], policy);
#ifdef NINFER_VOLTA_BUILD
    if (x.dtype == DType::FP16) {
        launch_fp16_volta_mma_f16(x, w, out, stream);
    } else {
        launch_fp16_volta_mma(x, w, out, stream);
    }
#endif
}

} // namespace ninfer::ops::detail
