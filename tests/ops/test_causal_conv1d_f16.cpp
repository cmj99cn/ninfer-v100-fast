#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/causal_conv1d_silu_f16.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

float load_weight(std::uint16_t bits, DType dtype) {
    if (dtype == DType::FP16) { return __half2float(__ushort_as_half(bits)); }
    __nv_bfloat16 value;
    std::memcpy(&value, &bits, sizeof(bits));
    return __bfloat162float(value);
}

int run_case(int C, int T, std::uint32_t seed, DType weight_dtype, bool separate_state,
             bool public_f16_op) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> xf(static_cast<std::size_t>(C) * T);
    std::vector<float> wf(static_cast<std::size_t>(C) * 4);
    std::vector<float> sf(static_cast<std::size_t>(C) * 3);
    for (auto& v : xf) v = d(rng);
    for (auto& v : wf) v = d(rng);
    for (auto& v : sf) v = d(rng);
    round_to_f16(xf);
    round_to_f16(sf);
    if (weight_dtype == DType::FP16) {
        round_to_f16(wf);
    } else {
        round_to_bf16(wf);
    }

    std::vector<__half> x(xf.size()), st(sf.size()), out(xf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) x[i] = __float2half_rn(xf[i]);
    for (std::size_t i = 0; i < sf.size(); ++i) st[i] = __float2half_rn(sf[i]);
    std::vector<std::uint16_t> w(wf.size());
    for (std::size_t i = 0; i < wf.size(); ++i) {
        w[i] = weight_dtype == DType::FP16 ? f32_to_f16(wf[i]) : f32_to_bf16(wf[i]);
    }

    GuardedDeviceBuffer dx(x.size() * 2), dw(w.size() * 2), ds_in(st.size() * 2);
    GuardedDeviceBuffer ds_out(st.size() * 2), dout(out.size() * 2);
    dx.copy_from_host(x.data(), dx.bytes());
    dw.copy_from_host(w.data(), dw.bytes());
    ds_in.copy_from_host(st.data(), ds_in.bytes());
    ds_out.fill(0xff);
    dout.fill(0xff);

    Tensor xt(dx.data(), DType::FP16, {C, T});
    Tensor wt(dw.data(), weight_dtype, {C, 4});
    Tensor state_in(ds_in.data(), DType::FP16, {3, C});
    Tensor state_out = separate_state ? Tensor(ds_out.data(), DType::FP16, {3, C})
                                      : Tensor(ds_in.data(), DType::FP16, {3, C});
    Tensor ot(dout.data(), DType::FP16, {C, T});
    if (public_f16_op) {
        ops::causal_conv1d_silu_f16(xt, wt, state_in, ot, nullptr);
    } else {
        ops::causal_conv1d_silu(xt, wt, state_in, state_out, ot, nullptr);
    }
    cuda_synchronize();

    const auto ob = from_device<std::uint16_t>(dout.data(), out.size());
    const auto sb = from_device<std::uint16_t>(ds_in.data(), st.size());
    const auto sbo = from_device<std::uint16_t>(ds_out.data(), st.size());
    const auto& state_bits = separate_state ? sbo : sb;

    std::vector<double> oracle_out(static_cast<std::size_t>(C) * T);
    std::vector<float> state(sf);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < C; ++c) {
            const float x0 = t >= 3 ? xf[static_cast<std::size_t>(t - 3) * C + c] : state[c];
            const float x1 = t >= 2 ? xf[static_cast<std::size_t>(t - 2) * C + c] : state[C + c];
            const float x2 = t >= 1 ? xf[static_cast<std::size_t>(t - 1) * C + c] : state[2 * C + c];
            const float x3 = xf[static_cast<std::size_t>(t) * C + c];
            const float a = x0 * load_weight(w[c], weight_dtype) +
                            x1 * load_weight(w[C + c], weight_dtype) +
                            x2 * load_weight(w[2 * C + c], weight_dtype) +
                            x3 * load_weight(w[3 * C + c], weight_dtype);
            oracle_out[static_cast<std::size_t>(t) * C + c] = a / (1.0f + std::exp(-a));
        }
    }
    // state_out[s] = concatenated(-3,-2,-1 | 0..T-1)[T+s].
    std::vector<float> next_state(static_cast<std::size_t>(C) * 3);
    for (int s = 0; s < 3; ++s) {
        for (int c = 0; c < C; ++c) {
            const int pos = T + s;
            next_state[static_cast<std::size_t>(s) * C + c] =
                pos < 3 ? sf[static_cast<std::size_t>(pos) * C + c]
                        : xf[static_cast<std::size_t>(pos - 3) * C + c];
        }
    }

    int failures       = 0;
    const auto& got    = from_device_f16(dout.data(), out.size());
    const auto stats   = compute_reduction_stats(got.data(), oracle_out.data(),
                                                 static_cast<std::int64_t>(got.size()));
    if (!(stats.relative_l2 <= 2.0e-3)) { ++failures; }
    for (int i = 0; i < C * 3; ++i) {
        const std::uint16_t expected = f32_to_f16(next_state[static_cast<std::size_t>(i)]);
        if (state_bits[static_cast<std::size_t>(i)] != expected) { ++failures; }
    }
    if (separate_state) {
        const auto in_after = from_device<std::uint16_t>(ds_in.data(), st.size());
        for (int i = 0; i < C * 3; ++i) {
            if (in_after[static_cast<std::size_t>(i)] != f32_to_f16(sf[static_cast<std::size_t>(i)])) {
                ++failures;
            }
        }
    }
    std::cout << "causal_conv1d_f16 C=" << C << " T=" << T
              << " weight=" << (weight_dtype == DType::FP16 ? "f16" : "bf16")
              << " separate_state=" << separate_state << " rel=" << stats.relative_l2
              << " failures=" << failures << '\n';
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        failures += run_case(16, 1, 9001U, DType::FP16, false, true);
        failures += run_case(16, 2, 9004U, DType::FP16, true, false);
        failures += run_case(16, 5, 9002U, DType::FP16, true, false);
        failures += run_case(17, 7, 9003U, DType::FP16, false, false);
        // The text runtime keeps the persistent convolution weight in BF16 while the activation,
        // history, and output are FP16. Exercise the public wrapper path at the same geometry.
        failures += run_case(32, 1, 9011U, DType::BF16, false, false);
        failures += run_case(32, 2, 9012U, DType::BF16, true, false);
        failures += run_case(32, 3, 9013U, DType::BF16, true, false);
        failures += run_case(33, 7, 9014U, DType::BF16, true, false);
        failures += run_case(64, 17, 9015U, DType::BF16, true, false);
        std::cout << (failures ? "FAIL" : "OK") << " causal_conv1d_f16\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "causal_conv1d_f16: " << e.what() << '\n';
        return 1;
    }
}
