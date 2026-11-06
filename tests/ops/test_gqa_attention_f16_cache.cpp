#include "ninfer/ops/gqa_attention_f16.h"
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
constexpr int kPage = 64;

std::vector<__half> make_half(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> out(n);
    for (auto& v : out) v = __float2half_rn(dist(rng));
    return out;
}

int run_case(int T, int start, int visible) {
    const int max_blocks = (visible + kPage - 1) / kPage;
    const int physical_pages = max_blocks;
    const std::size_t qn = static_cast<std::size_t>(kD) * kH * T;
    const std::size_t kvn = static_cast<std::size_t>(kD) * kPage * kHKV * physical_pages;
    auto q = make_half(qn, 4001U + static_cast<std::uint32_t>(start));
    auto k = make_half(kvn, 4002U + static_cast<std::uint32_t>(start));
    auto v = make_half(kvn, 4003U + static_cast<std::uint32_t>(start));
    std::vector<std::int32_t> block_table(static_cast<std::size_t>(max_blocks));
    for (int i = 0; i < max_blocks; ++i) block_table[static_cast<std::size_t>(i)] = i;
    std::vector<std::int32_t> positions(static_cast<std::size_t>(T));
    for (int i = 0; i < T; ++i) positions[static_cast<std::size_t>(i)] = start + i;

    GuardedDeviceBuffer dq(q.size() * sizeof(__half));
    GuardedDeviceBuffer dk(k.size() * sizeof(__half));
    GuardedDeviceBuffer dv(v.size() * sizeof(__half));
    GuardedDeviceBuffer dout(q.size() * sizeof(__half));
    GuardedDeviceBuffer dpos(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dbt(block_table.size() * sizeof(std::int32_t));
    dq.copy_from_host(q.data(), dq.bytes());
    dk.copy_from_host(k.data(), dk.bytes());
    dv.copy_from_host(v.data(), dv.bytes());
    dpos.copy_from_host(positions.data(), dpos.bytes());
    dbt.copy_from_host(block_table.data(), dbt.bytes());

    Tensor qt(dq.data(), DType::FP16, {kD, kH, T});
    Tensor pos(dpos.data(), DType::I32, {T});
    Tensor out(dout.data(), DType::FP16, {kD, kH, T});
    PagedKVLayerView cache;
    cache.k_pages = Tensor(dk.data(), DType::FP16, {kD, kPage, kHKV, physical_pages});
    cache.v_pages = Tensor(dv.data(), DType::FP16, {kD, kPage, kHKV, physical_pages});
    cache.block_table = Tensor(dbt.data(), DType::I32, {max_blocks});
    cache.head_dim = kD;
    cache.num_kv_heads = kHKV;
    cache.dtype = DType::FP16;

    const ops::GqaExecutionEnvelope envelope{1, static_cast<std::uint32_t>(visible)};
    const std::size_t capacity =
        ops::gqa_attention_f16_workspace_capacity_bytes(kH, envelope, T, T);
    DeviceArena workspace(capacity);
    ops::gqa_attention_f16_cached(qt, pos, 0.0625f, cache, envelope, workspace, out, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "gqa_attention_f16_cached sync failed\n";
        return 1;
    }
    const auto actual = from_device<std::uint16_t>(dout.data(), qn);
    int failures = 0;
    double err2 = 0.0, ref2 = 0.0;
    const int sample_t[] = {0, T / 2, T - 1};
    for (int h = 0; h < kH; h += 5) {
        const int hkv = h / (kH / kHKV);
        for (int t : sample_t) {
            const int query_pos = start + t;
            std::vector<double> scores(static_cast<std::size_t>(query_pos + 1));
            double max_score = -1e300;
            for (int n = 0; n <= query_pos; ++n) {
                const int page = n / kPage;
                const int slot = n % kPage;
                const std::size_t base =
                    static_cast<std::size_t>(slot) * kD +
                    static_cast<std::size_t>(hkv) * kD * kPage +
                    static_cast<std::size_t>(block_table[page]) * kD * kPage * kHKV;
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    dot += __half2float(q[static_cast<std::size_t>(d) +
                                          static_cast<std::size_t>(h) * kD +
                                          static_cast<std::size_t>(t) * kD * kH]) *
                           __half2float(k[base + d]);
                }
                scores[static_cast<std::size_t>(n)] = dot * 0.0625;
                max_score = std::max(max_score, scores[static_cast<std::size_t>(n)]);
            }
            double sum = 0.0;
            for (double& s : scores) { s = std::exp(s - max_score); sum += s; }
            for (int d = 0; d < kD; ++d) {
                double ref = 0.0;
                for (int n = 0; n <= query_pos; ++n) {
                    const int page = n / kPage;
                    const int slot = n % kPage;
                    const std::size_t base =
                        static_cast<std::size_t>(slot) * kD +
                        static_cast<std::size_t>(hkv) * kD * kPage +
                        static_cast<std::size_t>(block_table[page]) * kD * kPage * kHKV;
                    ref += scores[static_cast<std::size_t>(n)] / sum * __half2float(v[base + d]);
                }
                const double got = __half2float(__ushort_as_half(
                    actual[static_cast<std::size_t>(d) + static_cast<std::size_t>(h) * kD +
                           static_cast<std::size_t>(t) * kD * kH]));
                const double err = got - ref;
                err2 += err * err;
                ref2 += ref * ref;
            }
        }
    }
    const double rel = std::sqrt(err2 / std::max(ref2, 1e-30));
    std::cout << "gqa_attention_f16_cached T=" << T << " start=" << start
              << " rel_l2=" << rel << '\n';
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
        failures += run_case(64, 0, 64);
        failures += run_case(1, 127, 128);
        failures += run_case(1, 511, 512);
        std::cout << (failures ? "FAIL" : "OK") << " gqa_attention_f16_cached\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "gqa_attention_f16_cached: " << error.what() << '\n';
        return 1;
    }
}
