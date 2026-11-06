#include "ninfer/ops/gqa_attention_f16.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int kD = 256;
constexpr int kH = 4;
constexpr int kPage = 64;

int run_case(int T, int start, std::uint32_t seed) {
    const int max_blocks = (start + T + kPage - 1) / kPage;
    const std::size_t cn = static_cast<std::size_t>(kD) * kPage * kH * max_blocks;
    const std::size_t kn = static_cast<std::size_t>(kD) * kH * T;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> k(kn), v(kn), cache_k(cn), cache_v(cn);
    for (auto& x : k) x = __float2half_rn(dist(rng));
    for (auto& x : v) x = __float2half_rn(dist(rng));
    for (auto& x : cache_k) x = __float2half_rn(7.0f);
    for (auto& x : cache_v) x = __float2half_rn(8.0f);
    std::vector<std::int32_t> pos(T), bt(max_blocks);
    for (int i = 0; i < T; ++i) pos[i] = start + i;
    for (int i = 0; i < max_blocks; ++i) bt[i] = i;
    GuardedDeviceBuffer dk(kn*2), dv(kn*2), dck(cn*2), dcv(cn*2), dp(T*4), dbt(max_blocks*4);
    dk.copy_from_host(k.data(), dk.bytes()); dv.copy_from_host(v.data(), dv.bytes());
    dck.copy_from_host(cache_k.data(), dck.bytes()); dcv.copy_from_host(cache_v.data(), dcv.bytes());
    dp.copy_from_host(pos.data(), dp.bytes()); dbt.copy_from_host(bt.data(), dbt.bytes());
    Tensor kt(dk.data(), DType::FP16, {kD, kH, T});
    Tensor vt(dv.data(), DType::FP16, {kD, kH, T});
    Tensor pt(dp.data(), DType::I32, {T});
    PagedKVLayerView cache;
    cache.k_pages = Tensor(dck.data(), DType::FP16, {kD, kPage, kH, max_blocks});
    cache.v_pages = Tensor(dcv.data(), DType::FP16, {kD, kPage, kH, max_blocks});
    cache.block_table = Tensor(dbt.data(), DType::I32, {max_blocks});
    cache.head_dim = kD; cache.num_kv_heads = kH; cache.dtype = DType::FP16;
    ops::gqa_kv_append_f16(kt, vt, pt, cache, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    const auto gk = from_device<std::uint16_t>(dck.data(), cn);
    const auto gv = from_device<std::uint16_t>(dcv.data(), cn);
    int failures = 0;
    for (int t = 0; t < T; ++t) {
        const int p = start + t;
        const int page = p / kPage, slot = p % kPage;
        for (int h = 0; h < kH; ++h) {
            for (int d = 0; d < kD; ++d) {
                const std::size_t src = static_cast<std::size_t>(d) + static_cast<std::size_t>(h)*kD + static_cast<std::size_t>(t)*kD*kH;
                const std::size_t dst = static_cast<std::size_t>(d) + static_cast<std::size_t>(slot)*kD + static_cast<std::size_t>(h)*kD*kPage + static_cast<std::size_t>(bt[page])*kD*kPage*kH;
                if (gk[dst] != __half_as_ushort(k[src]) || gv[dst] != __half_as_ushort(v[src])) ++failures;
            }
        }
    }
    std::cout << "gqa_kv_append_f16 T=" << T << " start=" << start << " failures=" << failures << '\n';
    return failures;
}
} // namespace
int main() {
    if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return 77; }
    try {
        int failures = 0;
        failures += run_case(1, 0, 7001U);
        failures += run_case(32, 31, 7002U);
        failures += run_case(64, 0, 7003U);
        std::cout << (failures ? "FAIL" : "OK") << " gqa_kv_append_f16\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
