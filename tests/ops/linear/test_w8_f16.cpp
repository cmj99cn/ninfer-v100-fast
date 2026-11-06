#include "ops/linear/linear_test_common.h"
#include "ops/quantized_weight.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_pair.h"

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
using namespace ninfer::test::linear;

int run_pair(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    constexpr std::int32_t kRows = 1024;
    auto packed_a = make_w8g32_f16s_weight(kRows, k, seed);
    auto packed_b = make_w8g32_f16s_weight(kRows, k, seed + 1U);
    std::vector<__half> activation(static_cast<std::size_t>(k) * t);
    std::mt19937 rng(seed + 17U);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& value : activation) { value = __float2half_rn(dist(rng)); }

    DeviceBuffer x_bytes(activation.size() * sizeof(__half));
    x_bytes.copy_from_host(activation.data(), x_bytes.bytes);
    DeviceBuffer a_bytes(packed_a.payload.size()), b_bytes(packed_b.payload.size());
    a_bytes.copy_from_host(packed_a.payload.data(), a_bytes.bytes);
    b_bytes.copy_from_host(packed_b.payload.data(), b_bytes.bytes);
    DeviceBuffer out_a(static_cast<std::size_t>(kRows) * t * sizeof(__half));
    DeviceBuffer out_b(static_cast<std::size_t>(kRows) * t * sizeof(__half));
    out_a.fill(0xCD);
    out_b.fill(0xCD);
    Tensor x(x_bytes.p, DType::FP16, {k, t});
    Tensor ta(out_a.p, DType::FP16, {kRows, t});
    Tensor tb(out_b.p, DType::FP16, {kRows, t});
    ops::linear_pair(x, packed_a.device_weight(a_bytes.p), packed_b.device_weight(b_bytes.p), ta, tb,
                     nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { return 1; }
    std::vector<__half> actual_a(static_cast<std::size_t>(kRows) * t);
    std::vector<__half> actual_b(static_cast<std::size_t>(kRows) * t);
    out_a.copy_to_host(actual_a.data(), out_a.bytes);
    out_b.copy_to_host(actual_b.data(), out_b.bytes);
    double error_sq = 0.0, reference_sq = 0.0;
    int failures = 0;
    const auto check = [&](const std::vector<__half>& actual,
                           const test::quantized_weight::PackedWeight& packed) {
        const std::int32_t rows[] = {0, kRows / 2, kRows - 1};
        const std::int32_t cols[] = {0, t > 1 ? 1 : 0, t - 1};
        for (const std::int32_t row : rows) {
            for (const std::int32_t col : cols) {
                double reference = 0.0;
                for (std::int32_t i = 0; i < k; ++i) {
                    reference += test::quantized_weight::logical_weight_fp64(packed, row, i) *
                                 static_cast<double>(__half2float(
                                     activation[static_cast<std::size_t>(col) * k + i]));
                }
                const double value = static_cast<double>(
                    __half2float(actual[static_cast<std::size_t>(col) * kRows + row]));
                if (!std::isfinite(value)) { ++failures; continue; }
                const double error = value - reference;
                error_sq += error * error;
                reference_sq += reference * reference;
            }
        }
    };
    check(actual_a, packed_a);
    check(actual_b, packed_b);
    const double relative = std::sqrt(error_sq / std::max(reference_sq, 1e-30));
    std::cout << "W8_F16_PAIR k=" << k << " t=" << t << " relative_l2=" << relative << '\n';
    if (relative > 2e-2) { ++failures; }
    return failures;
}

int run_shape(std::int32_t n, std::int32_t k, std::int32_t t, std::uint32_t seed) {
    auto packed = make_w8g32_f16s_weight(n, k, seed);
    std::vector<__half> activation(static_cast<std::size_t>(k) * t);
    std::mt19937 rng(seed + 17U);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (std::size_t i = 0; i < activation.size(); ++i) {
        activation[i] = __float2half_rn(dist(rng));
    }

    DeviceBuffer x_bytes(activation.size() * sizeof(__half));
    x_bytes.copy_from_host(activation.data(), x_bytes.bytes);
    DeviceBuffer w_bytes(packed.payload.size());
    w_bytes.copy_from_host(packed.payload.data(), w_bytes.bytes);
    DeviceBuffer out_bytes(static_cast<std::size_t>(n) * t * sizeof(__half));
    out_bytes.fill(0xCD);

    Weight weight = packed.device_weight(w_bytes.p);
    const std::size_t capacity =
        ops::linear_workspace_capacity_bytes(weight.qtype, n, k, ops::LinearPolicy::A16Only, t, t);
    DeviceArena workspace(capacity == 0 ? std::size_t{16} : capacity);
    Tensor x(x_bytes.p, DType::FP16, {k, t});
    Tensor out(out_bytes.p, DType::FP16, {n, t});
    ops::linear(x, weight, out, ops::LinearPolicy::AllowA4, workspace, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "W8_F16: synchronize failed\n";
        return 1;
    }

    std::vector<__half> actual(static_cast<std::size_t>(n) * t);
    out_bytes.copy_to_host(actual.data(), out_bytes.bytes);

    const std::int32_t rows[] = {0, 1, n / 2, n - 1};
    const std::int32_t cols[] = {0, t > 1 ? 1 : 0, t / 2, t - 1};
    double error_sq = 0.0;
    double reference_sq = 0.0;
    int failures = 0;
    for (const std::int32_t row : rows) {
        for (const std::int32_t col : cols) {
            double reference = 0.0;
            for (std::int32_t i = 0; i < k; ++i) {
                reference +=
                    test::quantized_weight::logical_weight_fp64(packed, row, i) *
                    static_cast<double>(
                        __half2float(activation[static_cast<std::size_t>(col) * k + i]));
            }
            const double value = static_cast<double>(
                __half2float(actual[static_cast<std::size_t>(col) * n + row]));
            if (!std::isfinite(value)) { ++failures; continue; }
            const double error = value - reference;
            error_sq += error * error;
            reference_sq += reference * reference;
        }
    }
    const double relative = std::sqrt(error_sq / std::max(reference_sq, 1e-30));
    std::cout << "W8_F16 n=" << n << " k=" << k << " t=" << t
              << " relative_l2=" << relative << '\n';
    if (relative > 2e-2) { ++failures; }
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        constexpr std::int32_t tokens[] = {1, 4, 8};
        constexpr std::int32_t shapes[][2] = {
            {5120, 10240},  // MTP input projection
            {14336, 5120},  // MTP attention projection
            {34816, 5120},  // MTP MLP gate/up
            {5120, 17408},  // MTP MLP down
        };
        std::uint32_t seed = 950U;
        for (const auto& shape : shapes) {
            for (const std::int32_t t : tokens) {
                failures += run_shape(shape[0], shape[1], t, seed++);
            }
        }
        failures += run_pair(5120, 1, 970U);
        failures += run_pair(5120, 4, 971U);
        failures += run_pair(5120, 8, 972U);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " W8_F16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "W8_F16 Linear: " << error.what() << '\n';
        return 1;
    }
}
