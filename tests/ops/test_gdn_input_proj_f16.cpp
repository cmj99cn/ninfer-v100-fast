#include "ops/linear/linear_test_common.h"
#include "ops/quantized_weight.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/gdn_input_proj.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

int run_case(std::int32_t weight_rows, std::int32_t qkv_rows, std::int32_t z_rows,
             std::int32_t tokens, std::uint32_t seed) {
    constexpr std::int32_t kHidden = 5120;
    auto packed = ninfer::test::linear::make_nvfp4_weight(weight_rows, kHidden, seed);

    std::vector<__half> activation(static_cast<std::size_t>(kHidden) * tokens);
    std::mt19937 rng(seed + 17U);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& value : activation) { value = __float2half_rn(dist(rng)); }

    DeviceBuffer x_bytes(activation.size() * sizeof(__half));
    x_bytes.copy_from_host(activation.data(), x_bytes.bytes);
    DeviceBuffer w_bytes(packed.payload.size());
    w_bytes.copy_from_host(packed.payload.data(), w_bytes.bytes);
    DeviceBuffer qkv_bytes(static_cast<std::size_t>(qkv_rows) * tokens * sizeof(__half));
    DeviceBuffer z_bytes(static_cast<std::size_t>(z_rows) * tokens * sizeof(__half));
    qkv_bytes.fill(0xCD);
    z_bytes.fill(0xCD);

    Weight weight = packed.device_weight(w_bytes.p);
    DeviceArena workspace(std::size_t{16});
    Tensor x(x_bytes.p, DType::FP16, {kHidden, tokens});
    Tensor qkv(qkv_bytes.p, DType::FP16, {qkv_rows, tokens});
    Tensor z(z_bytes.p, DType::FP16, {z_rows, tokens});
    ops::gdn_input_proj(x, weight, qkv, z, ops::LinearPolicy::AllowA4, workspace, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "gdn_input_proj_f16: synchronize failed\n";
        return 1;
    }

    std::vector<__half> actual_qkv(static_cast<std::size_t>(qkv_rows) * tokens);
    std::vector<__half> actual_z(static_cast<std::size_t>(z_rows) * tokens);
    qkv_bytes.copy_to_host(actual_qkv.data(), qkv_bytes.bytes);
    z_bytes.copy_to_host(actual_z.data(), z_bytes.bytes);

    double error_sq = 0.0;
    double reference_sq = 0.0;
    int failures = 0;
    const auto check = [&](const std::vector<__half>& actual, std::int32_t rows,
                           std::int32_t weight_row_offset) {
        const std::int32_t sample_rows[] = {0, 1, rows / 2, rows - 1};
        const std::int32_t sample_tokens[] = {0, tokens > 1 ? 1 : 0, tokens / 2, tokens - 1};
        for (const std::int32_t row : sample_rows) {
            for (const std::int32_t token : sample_tokens) {
                double reference = 0.0;
                for (std::int32_t i = 0; i < kHidden; ++i) {
                    reference +=
                        ninfer::test::quantized_weight::logical_weight_fp64(packed, weight_row_offset + row, i) *
                        static_cast<double>(
                            __half2float(activation[static_cast<std::size_t>(token) * kHidden + i]));
                }
                const double value = static_cast<double>(
                    __half2float(actual[static_cast<std::size_t>(token) * rows + row]));
                if (!std::isfinite(value)) { ++failures; continue; }
                const double error = value - reference;
                error_sq += error * error;
                reference_sq += reference * reference;
            }
        }
    };
    check(actual_qkv, qkv_rows, 0);
    check(actual_z, z_rows, qkv_rows);

    const double relative = std::sqrt(error_sq / std::max(reference_sq, 1e-30));
    std::cout << "GDN_F16 rows=" << weight_rows << " T=" << tokens
              << " relative_l2=" << relative << '\n';
    if (relative > 2e-2) { ++failures; }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        for (const std::int32_t tokens : {1, 4, 8, 33}) {
            failures += run_case(16384, 10240, 6144, tokens, 960U + tokens);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GDN_input_proj F16\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GDN_input_proj F16: " << error.what() << '\n';
        return 1;
    }
}
