#include "1cat_flash_attention/gdn_f16.h"
#include "ops/gdn_ref.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kS = 128;
constexpr int kHq = 16;
constexpr int kHv = 32;

void normalize_rows(std::vector<float>& values, int width) {
    for (std::size_t base = 0; base < values.size(); base += width) {
        double sum = 0.0;
        for (int d = 0; d < width; ++d) sum += static_cast<double>(values[base + d]) * values[base + d];
        const float inv = static_cast<float>(1.0 / std::sqrt(sum + 1e-6));
        for (int d = 0; d < width; ++d) values[base + d] *= inv;
    }
}

int run_case(int T, std::uint32_t seed) {
    gdn_ref::Inputs in;
    in.head_dim = kS;
    in.qk_heads = kHq;
    in.value_heads = kHv;
    in.tokens = T;
    const std::size_t qn = static_cast<std::size_t>(T) * kHq * kS;
    const std::size_t vn = static_cast<std::size_t>(T) * kHv * kS;
    const std::size_t sn = static_cast<std::size_t>(kHv) * kS * kS;
    in.q.resize(qn); in.k.resize(qn); in.v.resize(vn);
    in.g.resize(static_cast<std::size_t>(T) * kHv);
    in.beta.resize(static_cast<std::size_t>(T) * kHv);
    in.state.resize(sn);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> qd(-1.0f, 1.0f), vd(-0.5f, 0.5f), gd(-0.10f, -0.005f), bd(0.05f, 0.95f), sd(-0.02f, 0.02f);
    for (auto& x : in.q) x = qd(rng);
    for (auto& x : in.k) x = qd(rng);
    for (auto& x : in.v) x = vd(rng);
    for (auto& x : in.g) x = gd(rng);
    for (auto& x : in.beta) x = bd(rng);
    for (auto& x : in.state) x = sd(rng);
    normalize_rows(in.q, kS);
    normalize_rows(in.k, kS);
    const auto expected = gdn_ref::evaluate(in, 1.0 / std::sqrt(128.0), false);

    std::vector<float> state_t(in.state.size());
    for (int h = 0; h < kHv; ++h) {
        for (int kk = 0; kk < kS; ++kk) {
            for (int vv = 0; vv < kS; ++vv) {
                state_t[static_cast<std::size_t>((h * kS + kk) * kS + vv)] =
                    in.state[static_cast<std::size_t>((h * kS + vv) * kS + kk)];
            }
        }
    }
    std::vector<__half> q(in.q.size()), k(in.k.size()), v(in.v.size()), out(in.v.size());
    for (std::size_t i = 0; i < in.q.size(); ++i) { q[i] = __float2half_rn(in.q[i]); k[i] = __float2half_rn(in.k[i]); }
    for (std::size_t i = 0; i < in.v.size(); ++i) v[i] = __float2half_rn(in.v[i]);
    GuardedDeviceBuffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2), dout(out.size() * 2);
    GuardedDeviceBuffer dg(in.g.size() * 4), db(in.beta.size() * 4), ds(in.state.size() * 4), dfs(in.state.size() * 4);
    dq.copy_from_host(q.data(), dq.bytes()); dk.copy_from_host(k.data(), dk.bytes()); dv.copy_from_host(v.data(), dv.bytes());
    dg.copy_from_host(in.g.data(), dg.bytes()); db.copy_from_host(in.beta.data(), db.bytes());
    ds.copy_from_host(state_t.data(), ds.bytes());
    third_party::onecat::gdn_forward_f16_launch(
        dq.data(), dk.data(), dv.data(), dg.data(), db.data(), ds.data(), dout.data(),
        static_cast<float*>(dfs.data()), 1, T, kHq, kHv,
        static_cast<float>(1.0 / std::sqrt(128.0)), false, 1, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::cerr << "gdn f16 sync failed\n"; return 1; }
    const auto ob = from_device<std::uint16_t>(dout.data(), out.size());
    const auto fs = from_device<float>(dfs.data(), sn);
    int failures = 0;
    double e2 = 0.0, r2 = 0.0;
    for (std::size_t i = 0; i < ob.size(); ++i) {
        const double got = __half2float(__ushort_as_half(ob[i]));
        const double err = got - expected.out[i];
        e2 += err * err; r2 += expected.out[i] * expected.out[i];
    }
    const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
    double se2 = 0.0, sr2 = 0.0;
    for (std::size_t i = 0; i < fs.size(); ++i) {
        const int h = static_cast<int>(i / (kS * kS));
        const int kk = static_cast<int>((i / kS) % kS);
        const int vv = static_cast<int>(i % kS);
        const double err = static_cast<double>(fs[static_cast<std::size_t>((h * kS + vv) * kS + kk)]) -
                           expected.final_state[i];
        se2 += err * err; sr2 += expected.final_state[i] * expected.final_state[i];
    }
    const double srel = std::sqrt(se2 / std::max(sr2, 1e-30));
    std::cout << "gdn_f16 T=" << T << " out_rel=" << rel << " state_rel=" << srel << '\n';
    if (rel > 5e-3 || srel > 5e-3) ++failures;
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return 77; }
    try {
        int failures = 0;
        failures += run_case(1, 5001U);
        failures += run_case(4, 5002U);
        std::cout << (failures ? "FAIL" : "OK") << " gdn_f16\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "gdn_f16: " << e.what() << '\n';
        return 1;
    }
}
