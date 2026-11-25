#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/direct_fp16_weight.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;

std::vector<__half> random_activation(std::int32_t k, std::int32_t columns, std::uint32_t seed) {
    std::vector<__half> values(static_cast<std::size_t>(k) * columns);
    std::mt19937 rng(seed + 17U);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& value : values) { value = __float2half_rn(dist(rng)); }
    return values;
}

// Independent oracle: one batched [K,T] projection must equal T sequential [K,1] projections
// raw-bit for raw-bit. No kernel implementation, staging dtype, or reduction tree is copied; the
// only contract under test is that the small-T unified dispatch replays the exact T==1 route.
int compare_decode_replay(const char* label, const Weight& weight, std::int32_t n, std::int32_t k,
                          const std::vector<__half>& activation, std::int32_t columns,
                          std::size_t workspace_bytes) {
    DeviceBuffer x_bytes(activation.size() * sizeof(__half));
    x_bytes.copy_from_host(activation.data(), x_bytes.bytes);
    Tensor x(x_bytes.p, DType::FP16, {k, columns});

    const std::size_t words = static_cast<std::size_t>(n) * columns;
    GuardedDeviceBuffer batched(words * sizeof(__half));
    GuardedDeviceBuffer sequential(words * sizeof(__half));
    batched.fill(0xff);
    sequential.fill(0xff);
    Tensor batched_out(batched.data(), DType::FP16, {n, columns});
    Tensor sequential_out(sequential.data(), DType::FP16, {n, columns});

    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 16));
    ops::linear(x, weight, batched_out, ops::LinearPolicy::A16Only, workspace, 0);
    for (std::int32_t c = 0; c < columns; ++c) {
        const Tensor column = x.slice(1, c, 1);
        Tensor out_column   = sequential_out.slice(1, c, 1);
        ops::linear(column, weight, out_column, ops::LinearPolicy::A16Only, workspace, 0);
    }
    cuda_synchronize();

    int failures = 0;
    failures += verify_exact((std::string(label) + " raw bits").c_str(),
                             from_device<std::uint16_t>(batched.data(), words),
                             from_device<std::uint16_t>(sequential.data(), words));
    failures += batched.verify_guards(std::string(label) + " batched guard");
    failures += sequential.verify_guards(std::string(label) + " sequential guard");

    std::vector<__half> activation_after(activation.size());
    x_bytes.copy_to_host(activation_after.data(), x_bytes.bytes);
    if (std::memcmp(activation_after.data(), activation.data(), x_bytes.bytes) != 0) {
        std::cerr << label << ": small-T decode modified its activation input\n";
        ++failures;
    }
    return failures;
}

int run_fp16_control_case(const char* label, std::int32_t n, std::int32_t k, std::uint32_t seed,
                          const std::vector<std::int32_t>& columns) {
    int failures = 0;
    direct_fp16_weight::DeviceWeight weight(direct_fp16_weight::make_patterned(n, k, seed));
    for (const std::int32_t t : columns) {
        failures += compare_decode_replay(label, weight.view(), n, k, random_activation(k, t, seed + t),
                                          t, 0);
    }
    return failures;
}

int run_w8_case(const char* label, std::int32_t n, std::int32_t k, std::uint32_t seed,
                const std::vector<std::int32_t>& columns) {
    int failures = 0;
    auto packed = make_w8g32_f16s_weight(n, k, seed);
    DeviceBuffer w_bytes(packed.payload.size());
    w_bytes.copy_from_host(packed.payload.data(), w_bytes.bytes);
    const Weight weight = packed.device_weight(w_bytes.p);
    for (const std::int32_t t : columns) {
        failures += compare_decode_replay(label, weight, n, k, random_activation(k, t, seed + t),
                                          t, 0);
    }
    return failures;
}

int run_nvfp4_case(const char* label, std::int32_t n, std::int32_t k, std::uint32_t seed,
                   const std::vector<std::int32_t>& columns) {
    int failures = 0;
    auto packed = make_nvfp4_weight(n, k, seed);
    DeviceBuffer w_bytes(packed.payload.size());
    w_bytes.copy_from_host(packed.payload.data(), w_bytes.bytes);
    Weight weight = packed.device_weight(w_bytes.p);
    ops::detail::nvfp4_prepack_qpn_sm70(weight);
    const std::size_t capacity = ops::linear_workspace_capacity_bytes(
        weight.qtype, n, k, ops::LinearPolicy::A16Only, 1, 8);
    for (const std::int32_t t : columns) {
        failures += compare_decode_replay(label, weight, n, k, random_activation(k, t, seed + t),
                                          t, capacity);
    }
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const std::vector<std::int32_t> columns{2, 4, 8};
    int failures = 0;
    // qwen4_exp native-FP16 control linears: output head and PLE key/value. The HC low-rank
    // projections use ops::hc_mix (gated_residual), not ops::linear, so they are oracle'd there.
    failures += run_fp16_control_case("head fp16", 124160, 2560, 601U, columns);
    failures += run_fp16_control_case("ple_key fp16", 10240, 2560, 603U, columns);
    failures += run_fp16_control_case("ple_value fp16", 2560, 2560, 605U, columns);
    // qsa key/value shard: no exact multi decode exists, so fp16_dispatch keeps its per-column
    // replay; it must still be raw-bit identical to sequential T==1 calls.
    failures += run_fp16_control_case("qsa_kv fp16", 1664, 2560, 607U, columns);
    // qwen4_exp W8 projections at TP2: attention query/gate shard and a second registered width.
    failures += run_w8_case("attn_qg w8 shard", 6144, 2560, 611U, columns);
    failures += run_w8_case("w8 narrow", 2048, 2560, 613U, columns);
    // Registered NVFP4 geometries used in the FP16-activation profile.
    failures += run_nvfp4_case("attn_in nvfp4", 14336, 5120, 901U, columns);
    failures += run_nvfp4_case("gdn_in nvfp4", 16384, 5120, 903U, columns);
    if (failures == 0) { std::cout << "OK small-T linear unified decode replay bitwise\n"; }
    return failures;
}
