#include "1cat_flash_attention/fp16_attention_paged.h"
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

constexpr int kH = 24;
constexpr int kHKV = 4;
constexpr int kD = 256;
constexpr int kPage = 16;

std::vector<__half> make_half(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> out(n);
    for (auto& v : out) v = __float2half_rn(dist(rng));
    return out;
}

int run_case(int M) {
    const int max_blocks = (M + kPage - 1) / kPage;
    const int N = max_blocks * kPage;
    const std::size_t qn = static_cast<std::size_t>(kH) * M * kD;
    const std::size_t kvn = static_cast<std::size_t>(max_blocks) * kPage * kHKV * kD;
    auto q = make_half(qn, 2001U);
    auto k = make_half(kvn, 2002U);
    auto v = make_half(kvn, 2003U);
    std::vector<std::int32_t> block_table(static_cast<std::size_t>(max_blocks));
    for (int i = 0; i < max_blocks; ++i) block_table[static_cast<std::size_t>(i)] = i;
    std::int32_t seq_len = M;

    GuardedDeviceBuffer dq(q.size() * sizeof(__half));
    GuardedDeviceBuffer dk(k.size() * sizeof(__half));
    GuardedDeviceBuffer dv(v.size() * sizeof(__half));
    GuardedDeviceBuffer dout(q.size() * sizeof(__half));
    GuardedDeviceBuffer dlse(static_cast<std::size_t>(kH) * M * sizeof(float));
    GuardedDeviceBuffer dbt(block_table.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dsl(sizeof(std::int32_t));
    dq.copy_from_host(q.data(), dq.bytes());
    dk.copy_from_host(k.data(), dk.bytes());
    dv.copy_from_host(v.data(), dv.bytes());
    dbt.copy_from_host(block_table.data(), dbt.bytes());
    dsl.copy_from_host(&seq_len, sizeof(seq_len));

    const float scale = 1.0f / std::sqrt(static_cast<float>(kD));
    const auto k_block_stride = static_cast<long long>(kPage) * kHKV * kD;
    const auto k_token_stride = static_cast<long long>(kHKV) * kD;
    const auto k_head_stride = static_cast<long long>(kD);
    third_party::onecat::flash_attention_forward_f16_paged_launch(
        dq.data(), dk.data(), dv.data(), dout.data(),
        static_cast<float*>(dlse.data()), static_cast<const int*>(dbt.data()),
        static_cast<const int*>(dsl.data()), 1, kH, M, kPage, kHKV, max_blocks,
        k_block_stride, k_token_stride, k_head_stride, k_block_stride,
        k_token_stride, k_head_stride, scale, true, 1.0f, 1.0f, -1, -1, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FP16 paged attention synchronization failed\n";
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
                const int page = n / kPage;
                const int slot = n % kPage;
                const std::size_t base = ((static_cast<std::size_t>(block_table[page]) * kPage + slot) * kHKV + hkv) * kD;
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    const double qv = __half2float(q[(static_cast<std::size_t>(h) * M + m) * kD + d]);
                    const double kv = __half2float(k[base + d]);
                    dot += qv * kv;
                }
                scores[static_cast<std::size_t>(n)] = dot * static_cast<double>(scale);
                max_score = std::max(max_score, scores[static_cast<std::size_t>(n)]);
            }
            double sum = 0.0;
            for (double& s : scores) { s = std::exp(s - max_score); sum += s; }
            for (int d = 0; d < kD; ++d) {
                double ref = 0.0;
                for (int n = 0; n <= m; ++n) {
                    const int page = n / kPage;
                    const int slot = n % kPage;
                    const std::size_t base = ((static_cast<std::size_t>(block_table[page]) * kPage + slot) * kHKV + hkv) * kD;
                    ref += scores[static_cast<std::size_t>(n)] / sum * __half2float(v[base + d]);
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
    std::cout << "FP16 paged attention M=" << M << " rel_l2=" << rel << '\n';
    if (rel > 3e-3) ++failures;
    return failures;
}

int run_decode_case(int N) {
    const int M = 1;
    const int max_blocks = (N + kPage - 1) / kPage;
    const int padded_N = max_blocks * kPage;
    const std::size_t qn = static_cast<std::size_t>(kH) * M * kD;
    const std::size_t kvn = static_cast<std::size_t>(max_blocks) * kPage * kHKV * kD;
    auto q = make_half(qn, 3001U);
    auto k = make_half(kvn, 3002U);
    auto v = make_half(kvn, 3003U);
    std::vector<std::int32_t> block_table(static_cast<std::size_t>(max_blocks));
    for (int i = 0; i < max_blocks; ++i) block_table[static_cast<std::size_t>(i)] = i;
    std::int32_t seq_len = N;

    GuardedDeviceBuffer dq(q.size() * sizeof(__half));
    GuardedDeviceBuffer dk(k.size() * sizeof(__half));
    GuardedDeviceBuffer dv(v.size() * sizeof(__half));
    GuardedDeviceBuffer dout(q.size() * sizeof(__half));
    GuardedDeviceBuffer dlse(static_cast<std::size_t>(kH) * M * sizeof(float));
    GuardedDeviceBuffer dbt(block_table.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dsl(sizeof(std::int32_t));
    dq.copy_from_host(q.data(), dq.bytes());
    dk.copy_from_host(k.data(), dk.bytes());
    dv.copy_from_host(v.data(), dv.bytes());
    dbt.copy_from_host(block_table.data(), dbt.bytes());
    dsl.copy_from_host(&seq_len, sizeof(seq_len));
    const float scale = 1.0f / std::sqrt(static_cast<float>(kD));
    const auto block_stride = static_cast<long long>(kPage) * kHKV * kD;
    const auto token_stride = static_cast<long long>(kHKV) * kD;
    const auto head_stride = static_cast<long long>(kD);
    third_party::onecat::flash_attention_forward_f16_paged_launch(
        dq.data(), dk.data(), dv.data(), dout.data(),
        static_cast<float*>(dlse.data()), static_cast<const int*>(dbt.data()),
        static_cast<const int*>(dsl.data()), 1, kH, M, kPage, kHKV, max_blocks,
        block_stride, token_stride, head_stride, block_stride, token_stride,
        head_stride, scale, true, 1.0f, 1.0f, -1, -1, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    const auto out = from_device<std::uint16_t>(dout.data(), q.size());
    int failures = 0;
    double err2 = 0.0, ref2 = 0.0;
    for (int h = 0; h < kH; h += 5) {
        const int hkv = h / (kH / kHKV);
        std::vector<double> scores(static_cast<std::size_t>(N));
        double max_score = -1e300;
        for (int n = 0; n < N; ++n) {
            const int page = n / kPage;
            const int slot = n % kPage;
            const std::size_t base = ((static_cast<std::size_t>(block_table[page]) * kPage + slot) * kHKV + hkv) * kD;
            double dot = 0.0;
            for (int d = 0; d < kD; ++d) {
                dot += __half2float(q[static_cast<std::size_t>(h) * kD + d]) *
                       __half2float(k[base + d]);
            }
            scores[static_cast<std::size_t>(n)] = dot * static_cast<double>(scale);
            max_score = std::max(max_score, scores[static_cast<std::size_t>(n)]);
        }
        double sum = 0.0;
        for (double& score : scores) { score = std::exp(score - max_score); sum += score; }
        for (int d = 0; d < kD; ++d) {
            double ref = 0.0;
            for (int n = 0; n < N; ++n) {
                const int page = n / kPage;
                const int slot = n % kPage;
                const std::size_t base = ((static_cast<std::size_t>(block_table[page]) * kPage + slot) * kHKV + hkv) * kD;
                ref += scores[static_cast<std::size_t>(n)] / sum * __half2float(v[base + d]);
            }
            const double got = __half2float(__ushort_as_half(
                out[static_cast<std::size_t>(h) * kD + d]));
            const double err = got - ref;
            err2 += err * err;
            ref2 += ref * ref;
        }
    }
    const double rel = std::sqrt(err2 / std::max(ref2, 1e-30));
    std::cout << "FP16 paged decode N=" << N << " rel_l2=" << rel << '\n';
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
        failures += run_case(64);
        failures += run_case(128);
        failures += run_decode_case(128);
        failures += run_decode_case(512);
        std::cout << (failures ? "FAIL" : "OK") << " gqa_attention_f16_paged\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "gqa_attention_f16_paged: " << error.what() << '\n';
        return 1;
    }
}
