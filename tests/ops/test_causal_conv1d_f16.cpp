#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/causal_conv1d_silu_f16.h"
#include "ops/launcher/causal_conv1d.h"
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

int run_fusion_case(int C, std::uint32_t seed, DType weight_dtype, bool separate_state) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> xf(static_cast<std::size_t>(C)), wf(static_cast<std::size_t>(C) * 4),
        sf(static_cast<std::size_t>(C) * 3);
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
    std::vector<__half> x(xf.size()), st(sf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) x[i] = __float2half_rn(xf[i]);
    for (std::size_t i = 0; i < sf.size(); ++i) st[i] = __float2half_rn(sf[i]);
    std::vector<std::uint16_t> w(wf.size());
    for (std::size_t i = 0; i < wf.size(); ++i) {
        w[i] = weight_dtype == DType::FP16 ? f32_to_f16(wf[i]) : f32_to_bf16(wf[i]);
    }

    GuardedDeviceBuffer ax(x.size() * 2), aw(w.size() * 2), as(st.size() * 2),
        asep(st.size() * 2), aout(x.size() * 2);
    GuardedDeviceBuffer bx(x.size() * 2), bw(w.size() * 2), bs(st.size() * 2),
        bsep(st.size() * 2), bout(x.size() * 2);
    auto seed_buffers = [&](GuardedDeviceBuffer& dx, GuardedDeviceBuffer& dw,
                            GuardedDeviceBuffer& ds, GuardedDeviceBuffer& dsep) {
        dx.copy_from_host(x.data(), dx.bytes());
        dw.copy_from_host(w.data(), dw.bytes());
        ds.copy_from_host(st.data(), ds.bytes());
        dsep.copy_from_host(st.data(), dsep.bytes());
    };
    seed_buffers(ax, aw, as, asep);
    seed_buffers(bx, bw, bs, bsep);
    Tensor ax_t(ax.data(), DType::FP16, {C, 1}), aw_t(aw.data(), weight_dtype, {C, 4});
    Tensor as_t(as.data(), DType::FP16, {3, C}), aout_t(aout.data(), DType::FP16, {C, 1});
    Tensor bx_t(bx.data(), DType::FP16, {C, 1}), bw_t(bw.data(), weight_dtype, {C, 4});
    Tensor bs_t(bs.data(), DType::FP16, {3, C}), bout_t(bout.data(), DType::FP16, {C, 1});
    Tensor as_out = separate_state ? Tensor(asep.data(), DType::FP16, {3, C}) : as_t;
    Tensor bs_out = separate_state ? Tensor(bsep.data(), DType::FP16, {3, C}) : bs_t;
    ninfer::ops::detail::causal_conv1d_silu_f16_launch(ax_t, aw_t, as_t, as_out, aout_t, nullptr);
    ninfer::ops::detail::causal_conv1d_silu_f16_split_launch(bx_t, bw_t, bs_t, bs_out, bout_t,
                                                             nullptr);
    cuda_synchronize();

    int failures = 0;
    const auto ao = from_device<std::uint16_t>(aout.data(), x.size());
    const auto bo = from_device<std::uint16_t>(bout.data(), x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (ao[i] != bo[i]) { ++failures; }
    }
    const auto as_after = from_device<std::uint16_t>(separate_state ? asep.data() : as.data(),
                                                    st.size());
    const auto bs_after = from_device<std::uint16_t>(separate_state ? bsep.data() : bs.data(),
                                                    st.size());
    for (std::size_t i = 0; i < st.size(); ++i) {
        if (as_after[i] != bs_after[i]) { ++failures; }
    }
    std::cout << "conv_f16_fusion C=" << C
              << " weight=" << (weight_dtype == DType::FP16 ? "f16" : "bf16")
              << " separate_state=" << separate_state << " failures=" << failures << '\n';
    return failures;
}

int run_conv_cross_profile_case(int C, int columns, std::uint32_t seed, DType weight_dtype) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> xf(static_cast<std::size_t>(C) * columns);
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

    std::vector<std::uint16_t> x(xf.size()), w(wf.size()), st(sf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) { x[i] = f32_to_f16(xf[i]); }
    for (std::size_t i = 0; i < sf.size(); ++i) { st[i] = f32_to_f16(sf[i]); }
    for (std::size_t i = 0; i < wf.size(); ++i) {
        w[i] = weight_dtype == DType::FP16 ? f32_to_f16(wf[i]) : f32_to_bf16(wf[i]);
    }

    GuardedDeviceBuffer dx(x.size() * 2), dw(w.size() * 2), db_state(st.size() * 2);
    GuardedDeviceBuffer ds_state(st.size() * 2), db_out(x.size() * 2), ds_out(x.size() * 2);
    dx.copy_from_host(x.data(), dx.bytes());
    dw.copy_from_host(w.data(), dw.bytes());
    db_state.copy_from_host(st.data(), db_state.bytes());
    ds_state.copy_from_host(st.data(), ds_state.bytes());
    db_out.fill(0xff);
    ds_out.fill(0xff);

    Tensor weight(dw.data(), weight_dtype, {C, 4});
    Tensor batched_state(db_state.data(), DType::FP16, {3, C});
    Tensor sequential_state(ds_state.data(), DType::FP16, {3, C});
    Tensor batched_out(db_out.data(), DType::FP16, {C, columns});
    Tensor sequential_out(ds_out.data(), DType::FP16, {C, columns});
    const Tensor batched_x(dx.data(), DType::FP16, {C, columns});
    ops::causal_conv1d_silu_f16(batched_x, weight, batched_state, batched_out, nullptr);
    for (int c = 0; c < columns; ++c) {
        Tensor column     = batched_x.slice(1, c, 1);
        Tensor out_column = sequential_out.slice(1, c, 1);
        ops::causal_conv1d_silu_f16(column, weight, sequential_state, out_column, nullptr);
    }
    cuda_synchronize();

    int failures = verify_exact("conv cross-profile output raw bits",
                                from_device<std::uint16_t>(db_out.data(), x.size()),
                                from_device<std::uint16_t>(ds_out.data(), x.size()));
    failures += verify_exact("conv cross-profile state raw bits",
                             from_device<std::uint16_t>(db_state.data(), st.size()),
                             from_device<std::uint16_t>(ds_state.data(), st.size()));
    failures += db_out.verify_guards("conv cross-profile batched out guard");
    failures += ds_out.verify_guards("conv cross-profile sequential out guard");
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
        failures += run_fusion_case(16, 9021U, DType::FP16, false);
        failures += run_fusion_case(17, 9022U, DType::FP16, true);
        failures += run_fusion_case(64, 9023U, DType::BF16, false);
        failures += run_fusion_case(65, 9024U, DType::BF16, true);
        failures += run_conv_cross_profile_case(10240, 2, 9031U, DType::BF16);
        failures += run_conv_cross_profile_case(33, 4, 9032U, DType::FP16);
        failures += run_conv_cross_profile_case(10240, 8, 9033U, DType::BF16);
        std::cout << (failures ? "FAIL" : "OK") << " causal_conv1d_f16\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "causal_conv1d_f16: " << e.what() << '\n';
        return 1;
    }
}
