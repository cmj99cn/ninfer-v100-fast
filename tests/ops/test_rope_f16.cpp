#include "ninfer/ops/rope.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int kD = 256;
constexpr int kR = 64;
constexpr int kHq = 24;
constexpr int kHk = 4;

int run_case(int T, std::uint32_t seed) {
    const std::size_t qn = static_cast<std::size_t>(kD) * kHq * T;
    const std::size_t kn = static_cast<std::size_t>(kD) * kHk * T;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> q(qn), k(kn);
    for (auto& x : q) x = __float2half_rn(dist(rng));
    for (auto& x : k) x = __float2half_rn(dist(rng));
    std::vector<std::int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[static_cast<std::size_t>(i)] = i;
    GuardedDeviceBuffer dq(q.size() * 2), dk(k.size() * 2), dp(pos.size() * 4), dq0(q.size() * 2), dk0(k.size() * 2);
    dq.copy_from_host(q.data(), dq.bytes()); dk.copy_from_host(k.data(), dk.bytes());
    dq0.copy_from_host(q.data(), dq0.bytes()); dk0.copy_from_host(k.data(), dk0.bytes());
    dp.copy_from_host(pos.data(), dp.bytes());
    Tensor qt(dq.data(), DType::FP16, {kD, kHq, T});
    Tensor kt(dk.data(), DType::FP16, {kD, kHk, T});
    Tensor pt(dp.data(), DType::I32, {T});
    ops::rope(pt, kR, 1.0e7f, qt, kt, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    const auto got_q = from_device<std::uint16_t>(dq.data(), qn);
    const auto got_k = from_device<std::uint16_t>(dk.data(), kn);
    int failures = 0;
    double e2 = 0.0, r2 = 0.0;
    for (int head = 0; head < kHq; ++head) {
        for (int t = 0; t < T; ++t) {
            for (int p = 0; p < 32; ++p) {
                const float freq = std::pow(1.0e7f, -2.0f * p / 64.0f);
                const float angle = static_cast<float>(t) * freq;
                const float c = std::cos(angle), s = std::sin(angle);
                const std::size_t i0 = static_cast<std::size_t>(p) + static_cast<std::size_t>(head) * kD + static_cast<std::size_t>(t) * kD * kHq;
                const std::size_t i1 = i0 + 32;
                const float a = __half2float(q[i0]), b = __half2float(q[i1]);
                const double ref0 = a * c - b * s;
                const double ref1 = b * c + a * s;
                const double g0 = __half2float(__ushort_as_half(got_q[i0]));
                const double g1 = __half2float(__ushort_as_half(got_q[i1]));
                e2 += (g0-ref0)*(g0-ref0) + (g1-ref1)*(g1-ref1);
                r2 += ref0*ref0 + ref1*ref1;
            }
        }
    }
    for (int head = 0; head < kHk; ++head) {
        for (int t = 0; t < T; ++t) {
            for (int p = 0; p < 32; ++p) {
                const float freq = std::pow(1.0e7f, -2.0f * p / 64.0f);
                const float angle = static_cast<float>(t) * freq;
                const float c = std::cos(angle), s = std::sin(angle);
                const std::size_t i0 = static_cast<std::size_t>(p) + static_cast<std::size_t>(head) * kD + static_cast<std::size_t>(t) * kD * kHk;
                const std::size_t i1 = i0 + 32;
                const float a = __half2float(k[i0]), b = __half2float(k[i1]);
                const double ref0 = a * c - b * s;
                const double ref1 = b * c + a * s;
                const double g0 = __half2float(__ushort_as_half(got_k[i0]));
                const double g1 = __half2float(__ushort_as_half(got_k[i1]));
                e2 += (g0-ref0)*(g0-ref0) + (g1-ref1)*(g1-ref1);
                r2 += ref0*ref0 + ref1*ref1;
            }
        }
    }
    const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
    std::cout << "rope_f16 T=" << T << " rel_l2=" << rel << '\n';
    if (rel > 2e-3) ++failures;
    return failures;
}
} // namespace
int main() {
    if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return 77; }
    try {
        int failures = 0;
        failures += run_case(1, 6001U);
        failures += run_case(32, 6002U);
        std::cout << (failures ? "FAIL" : "OK") << " rope_f16\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << "rope_f16: " << e.what() << '\n'; return 1; }
}
