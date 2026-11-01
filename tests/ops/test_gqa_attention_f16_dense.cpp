#include "1cat_flash_attention/fp16_attention.h"
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

constexpr int kB = 1;
constexpr int kH = 24;
constexpr int kHKV = 4;
constexpr int kD = 256;

std::vector<__half> make_half(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<__half> out(n);
    for (auto& v : out) v = __float2half_rn(dist(rng));
    return out;
}

int run_case(int M) {
    const int N = M;
    const std::size_t qn = static_cast<std::size_t>(kB) * kH * M * kD;
    const std::size_t kvn = static_cast<std::size_t>(kB) * kHKV * N * kD;
    auto q = make_half(qn, 1001U, -1.0f, 1.0f);
    auto k = make_half(kvn, 1002U, -1.0f, 1.0f);
    auto v = make_half(kvn, 1003U, -1.0f, 1.0f);

    GuardedDeviceBuffer dq(q.size() * sizeof(__half));
    GuardedDeviceBuffer dk(k.size() * sizeof(__half));
    GuardedDeviceBuffer dv(v.size() * sizeof(__half));
    GuardedDeviceBuffer dout(q.size() * sizeof(__half));
    GuardedDeviceBuffer dlse(static_cast<std::size_t>(kB) * kH * M * sizeof(float));
    dq.copy_from_host(q.data(), dq.bytes());
    dk.copy_from_host(k.data(), dk.bytes());
    dv.copy_from_host(v.data(), dv.bytes());
    cuda_synchronize();

    const float scale = 1.0f / std::sqrt(static_cast<float>(kD));
    third_party::onecat::flash_attention_forward_f16_dense_launch(
        dq.data(), dk.data(), dv.data(), dout.data(),
        static_cast<float*>(dlse.data()), kB, kH, kHKV, M, N, scale, false, -1, -1,
        nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FP16 dense attention synchronization failed\n";
        return 1;
    }

    const auto out = from_device<std::uint16_t>(dout.data(), q.size());
    int failures = 0;
    double err2 = 0.0, ref2 = 0.0;
    const int sample_m[] = {0, 1, M / 2, M - 1};
    for (int h = 0; h < kH; h += 5) {
        const int hkv = h / (kH / kHKV);
        for (int m : sample_m) {
            std::vector<double> scores(static_cast<std::size_t>(m + 1));
            double max_score = -1e300;
            for (int n = 0; n <= m; ++n) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    const double qv = __half2float(q[(static_cast<std::size_t>(h) * M + m) * kD + d]);
                    const double kv = __half2float(k[(static_cast<std::size_t>(hkv) * N + n) * kD + d]);
                    dot += qv * kv;
                }
                scores[static_cast<std::size_t>(n)] = dot * static_cast<double>(scale);
                max_score = std::max(max_score, scores[static_cast<std::size_t>(n)]);
            }
            double sum = 0.0;
            for (double& s : scores) {
                s = std::exp(s - max_score);
                sum += s;
            }
            for (int d = 0; d < kD; ++d) {
                double ref = 0.0;
                for (int n = 0; n <= m; ++n) {
                    const double vv = __half2float(v[(static_cast<std::size_t>(hkv) * N + n) * kD + d]);
                    ref += scores[static_cast<std::size_t>(n)] / sum * vv;
                }
                const double got = __half2float(__ushort_as_half(
                    out[(static_cast<std::size_t>(h) * M + m) * kD + d]));
                const double err = got - ref;
                err2 += err * err;
                ref2 += ref * ref;
            }
        }
    }
    const double rel = std::sqrt(err2 / std::max(ref2, 1e-30));
    std::cout << "FP16 dense attention M=" << M << " rel_l2=" << rel << '\n';
    if (rel > 3e-3) ++failures;
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
        failures += run_case(128);
        failures += run_case(512);
        std::cout << (failures ? "FAIL" : "OK") << " gqa_attention_f16_dense\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "gqa_attention_f16_dense: " << error.what() << '\n';
        return 1;
    }
}
