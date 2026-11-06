#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kD = 256;
constexpr int kPage = 64;
constexpr int kGroup = 64;
constexpr int kGroups = kD / kGroup;

struct HostCache {
    std::vector<std::int8_t> k_code;
    std::vector<std::int8_t> v_code;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
};

std::vector<__half> make_half(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> out(n);
    for (auto& v : out) { v = __float2half_rn(dist(rng)); }
    return out;
}

std::int8_t encode_one(float x, float inv) {
    if (inv == 0.0f) { return 0; }
    int q = static_cast<int>(std::nearbyintf(x * inv));
    q = std::max(-127, std::min(127, q));
    return static_cast<std::int8_t>(q);
}

std::int64_t code_index(int hkv, int head, int position, int d) {
    const int page = position / kPage;
    const int slot = position % kPage;
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kD) *
               (slot + static_cast<std::int64_t>(kPage) *
                           (head + static_cast<std::int64_t>(hkv) * page));
}

std::int64_t scale_index(int hkv, int head, int position, int group) {
    const int page = position / kPage;
    const int slot = position % kPage;
    return static_cast<std::int64_t>(group) +
           static_cast<std::int64_t>(kGroups) *
               (slot + static_cast<std::int64_t>(kPage) *
                           (head + static_cast<std::int64_t>(hkv) * page));
}

HostCache encode_cache(const std::vector<__half>& k, const std::vector<__half>& v, int hkv,
                       int tokens, int pages, int start) {
    HostCache cache;
    const std::size_t elements =
        static_cast<std::size_t>(kD) * kPage * hkv * pages;
    cache.k_code.assign(elements, 0);
    cache.v_code.assign(elements, 0);
    const std::size_t scales =
        static_cast<std::size_t>(kGroups) * kPage * hkv * pages;
    cache.k_scale.assign(scales, 0U);
    cache.v_scale.assign(scales, 0U);
    for (int head = 0; head < hkv; ++head) {
        for (int token = 0; token < tokens; ++token) {
            const int position = start + token;
            for (int group = 0; group < kGroups; ++group) {
                float k_abs = 0.0f;
                float v_abs = 0.0f;
                for (int i = 0; i < kGroup; ++i) {
                    const int d = group * kGroup + i;
                    const std::int64_t src =
                        static_cast<std::int64_t>(d) +
                        static_cast<std::int64_t>(kD) *
                            (head + static_cast<std::int64_t>(hkv) * token);
                    k_abs = std::max(k_abs, std::fabs(__half2float(k[src])));
                    v_abs = std::max(v_abs, std::fabs(__half2float(v[src])));
                }
                const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / 127.0f : 0.0f);
                const __half vsh = __float2half_rn(v_abs > 0.0f ? v_abs / 127.0f : 0.0f);
                const float ks = __half2float(ksh);
                const float vs = __half2float(vsh);
                const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
                const float vinv = vs > 0.0f ? 1.0f / vs : 0.0f;
                cache.k_scale[scale_index(hkv, head, position, group)] =
                    *reinterpret_cast<const std::uint16_t*>(&ksh);
                cache.v_scale[scale_index(hkv, head, position, group)] =
                    *reinterpret_cast<const std::uint16_t*>(&vsh);
                for (int i = 0; i < kGroup; ++i) {
                    const int d = group * kGroup + i;
                    const std::int64_t src =
                        static_cast<std::int64_t>(d) +
                        static_cast<std::int64_t>(kD) *
                            (head + static_cast<std::int64_t>(hkv) * token);
                    const std::int64_t dst = code_index(hkv, head, position, d);
                    cache.k_code[dst] =
                        encode_one(__half2float(k[src]), kinv);
                    cache.v_code[dst] =
                        encode_one(__half2float(v[src]), vinv);
                }
            }
        }
    }
    return cache;
}

double cache_value(const HostCache& cache, bool key, int hkv, int head, int position, int d) {
    const std::int64_t ci = code_index(hkv, head, position, d);
    const std::int64_t si = scale_index(hkv, head, position, d / kGroup);
    return static_cast<double>((key ? cache.k_code : cache.v_code)[ci]) *
           static_cast<double>(__half2float(__ushort_as_half((key ? cache.k_scale : cache.v_scale)[si])));
}

int run_case(int h, int hkv, int tokens, int start) {
    const int capacity = start + tokens;
    const int pages = (capacity + kPage - 1) / kPage;
    const std::size_t qn = static_cast<std::size_t>(kD) * h * tokens;
    const std::size_t kvn = static_cast<std::size_t>(kD) * hkv * tokens;
    auto q = make_half(qn, 5001U + static_cast<std::uint32_t>(start));
    auto k = make_half(kvn, 5002U + static_cast<std::uint32_t>(start));
    auto v = make_half(kvn, 5003U + static_cast<std::uint32_t>(start));

    const HostCache expected = encode_cache(k, v, hkv, tokens, pages, start);

    const std::size_t cache_elements =
        static_cast<std::size_t>(kD) * kPage * hkv * pages;
    const std::size_t scale_elements =
        static_cast<std::size_t>(kGroups) * kPage * hkv * pages;
    std::vector<std::int8_t> zero_codes(cache_elements, 0);
    std::vector<__half> zero_scales(scale_elements, __float2half_rn(0.0f));
    std::vector<std::int32_t> block_table(static_cast<std::size_t>(pages));
    for (int i = 0; i < pages; ++i) { block_table[static_cast<std::size_t>(i)] = i; }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) {
        positions[static_cast<std::size_t>(i)] = start + i;
    }

    GuardedDeviceBuffer dq(qn * sizeof(__half));
    GuardedDeviceBuffer dk(kvn * sizeof(__half));
    GuardedDeviceBuffer dv(kvn * sizeof(__half));
    GuardedDeviceBuffer dout(qn * sizeof(__half));
    GuardedDeviceBuffer dk_code(cache_elements);
    GuardedDeviceBuffer dv_code(cache_elements);
    GuardedDeviceBuffer dk_scale(scale_elements * sizeof(__half));
    GuardedDeviceBuffer dv_scale(scale_elements * sizeof(__half));
    GuardedDeviceBuffer dbt(block_table.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dpos(positions.size() * sizeof(std::int32_t));
    dq.copy_from_host(q.data(), dq.bytes());
    dk.copy_from_host(k.data(), dk.bytes());
    dv.copy_from_host(v.data(), dv.bytes());
    dk_code.copy_from_host(zero_codes.data(), dk_code.bytes());
    dv_code.copy_from_host(zero_codes.data(), dv_code.bytes());
    dk_scale.copy_from_host(zero_scales.data(), dk_scale.bytes());
    dv_scale.copy_from_host(zero_scales.data(), dv_scale.bytes());
    dbt.copy_from_host(block_table.data(), dbt.bytes());
    dpos.copy_from_host(positions.data(), dpos.bytes());

    Tensor tk(dk.data(), DType::FP16, {kD, hkv, tokens});
    Tensor tv(dv.data(), DType::FP16, {kD, hkv, tokens});
    Tensor tpos(dpos.data(), DType::I32, {tokens});
    PagedKVLayerView append_cache;
    append_cache.k_pages = Tensor(dk_code.data(), DType::I8, {kD, kPage, hkv, pages});
    append_cache.v_pages = Tensor(dv_code.data(), DType::I8, {kD, kPage, hkv, pages});
    append_cache.k_scale_pages =
        Tensor(dk_scale.data(), DType::FP16, {kGroups, kPage, hkv, pages});
    append_cache.v_scale_pages =
        Tensor(dv_scale.data(), DType::FP16, {kGroups, kPage, hkv, pages});
    append_cache.block_table = Tensor(dbt.data(), DType::I32, {pages});
    append_cache.head_dim = kD;
    append_cache.num_kv_heads = hkv;
    append_cache.quant_group = kGroup;
    append_cache.dtype = DType::I8;

    ops::gqa_kv_append(tk, tv, tpos, append_cache, nullptr);
    cuda_synchronize();

    const auto actual_k_code = from_device<std::int8_t>(dk_code.data(), cache_elements);
    const auto actual_v_code = from_device<std::int8_t>(dv_code.data(), cache_elements);
    const auto actual_k_scale = from_device<std::uint16_t>(dk_scale.data(), scale_elements);
    const auto actual_v_scale = from_device<std::uint16_t>(dv_scale.data(), scale_elements);
    int failures = 0;
    for (int head = 0; head < hkv; ++head) {
        for (int token = 0; token < tokens; ++token) {
            const int position = start + token;
            for (int group = 0; group < kGroups; ++group) {
                const std::int64_t si = scale_index(hkv, head, position, group);
                if (actual_k_scale[si] != expected.k_scale[si] ||
                    actual_v_scale[si] != expected.v_scale[si]) {
                    ++failures;
                }
                for (int i = 0; i < kGroup; ++i) {
                    const std::int64_t ci = code_index(hkv, head, token, group * kGroup + i);
                    if (actual_k_code[ci] != expected.k_code[ci] ||
                        actual_v_code[ci] != expected.v_code[ci]) {
                        ++failures;
                    }
                }
            }
        }
    }
    if (failures != 0) {
        std::cerr << "gqa_attention_f16_i8: append mismatch H=" << h << " HKV=" << hkv
                  << " T=" << tokens << " start=" << start << " count=" << failures << '\n';
        return 1;
    }

    Tensor tq(dq.data(), DType::FP16, {kD, h, tokens});
    Tensor tout(dout.data(), DType::FP16, {kD, h, tokens});
    PagedKVLayerView cache = append_cache;
    const ops::GqaExecutionEnvelope envelope{1, static_cast<std::uint32_t>(capacity)};
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_capacity_bytes(h, DType::I8, envelope, 1, tokens, tokens);
    DeviceArena workspace(workspace_bytes);
    ops::gqa_attention_cached(tq, tpos, 0.0625f, cache, envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const auto actual = from_device<std::uint16_t>(dout.data(), qn);
    double err2 = 0.0;
    double ref2 = 0.0;
    for (int head = 0; head < h; ++head) {
        const int kv_head = head / (h / hkv);
        for (int token = 0; token < tokens; ++token) {
            const int qpos = start + token;
            std::vector<double> scores(static_cast<std::size_t>(qpos + 1));
            double max_score = -1.0e300;
            for (int key = 0; key <= qpos; ++key) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    const std::int64_t qi =
                        static_cast<std::int64_t>(d) +
                        static_cast<std::int64_t>(kD) *
                            (head + static_cast<std::int64_t>(h) * token);
                    dot += static_cast<double>(__half2float(q[qi])) *
                           cache_value(expected, true, hkv, kv_head, key, d);
                }
                scores[static_cast<std::size_t>(key)] = dot * 0.0625;
                max_score = std::max(max_score, scores[static_cast<std::size_t>(key)]);
            }
            double sum = 0.0;
            for (double& score : scores) {
                score = std::exp(score - max_score);
                sum += score;
            }
            for (int d = 0; d < kD; ++d) {
                double ref = 0.0;
                for (int key = 0; key <= qpos; ++key) {
                    ref += scores[static_cast<std::size_t>(key)] / sum *
                           cache_value(expected, false, hkv, kv_head, key, d);
                }
                const std::int64_t oi =
                    static_cast<std::int64_t>(d) +
                    static_cast<std::int64_t>(kD) *
                        (head + static_cast<std::int64_t>(h) * token);
                const double got = static_cast<double>(
                    __half2float(__ushort_as_half(actual[oi])));
                const double err = got - ref;
                err2 += err * err;
                ref2 += ref * ref;
            }
        }
    }
    const double rel = std::sqrt(err2 / std::max(ref2, 1.0e-30));
    std::cout << "gqa_attention_f16_i8 H=" << h << " HKV=" << hkv << " T=" << tokens
              << " start=" << start << " rel_l2=" << rel << '\n';
    if (!(rel < 5.0e-3)) { ++failures; }
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
        for (const int tokens : {4, 64}) {
            failures += run_case(24, 4, tokens, 0);
            failures += run_case(12, 2, tokens, 0);
            failures += run_case(16, 2, tokens, 0);
        }
        // A long visible window forces the SmallT split-KV route (splits > 1), which is where
        // the FP16 partial accumulator is written and reduced.
        failures += run_case(24, 4, 1, 4096);
        failures += run_case(12, 2, 1, 4096);
        failures += run_case(16, 2, 6, 4096);
        std::cout << (failures ? "FAIL" : "OK") << " gqa_attention_f16_i8\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "gqa_attention_f16_i8: " << error.what() << '\n';
        return 1;
    }
}
