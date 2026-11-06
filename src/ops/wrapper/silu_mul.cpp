// ninfer::ops — silu_mul wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/silu_mul.h"

#include "ops/launcher/silu_and_mul.h" // detail::silu_and_mul_launch

#include <stdexcept>

namespace ninfer::ops {

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const bool bf16 = gate.dtype == DType::BF16 && up.dtype == DType::BF16 &&
                      out.dtype == DType::BF16;
    const bool f16 = gate.dtype == DType::FP16 && up.dtype == DType::FP16 &&
                     out.dtype == DType::FP16;
    if (!bf16 && !f16) {
        throw std::invalid_argument("silu_mul: gate/up/out must share BF16 or FP16 dtype");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument("silu_mul: gate/up/out shapes must match");
        }
    }
    if (!out.is_contiguous()) { throw std::invalid_argument("silu_mul: out must be contiguous"); }
    if (out.numel() == 0) { return; }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("silu_mul: gate/up/out data must be non-null");
    }

    if (out.dtype == DType::FP16) {
        detail::silu_and_mul_f16_launch(gate, up, out, stream);
    } else {
        detail::silu_and_mul_launch(gate, up, out, stream);
    }
}

} // namespace ninfer::ops
