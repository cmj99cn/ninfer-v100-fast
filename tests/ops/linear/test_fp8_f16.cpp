#include "ops/linear/linear_test_common.h"
#include "ops/quantized_weight.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

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

int run_shape(std::int32_t n, std::int32_t k, std::int32_t t, std::uint32_t seed) {
    auto packed = make_fp8_weight(n, k, seed);
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
    ops::linear(x, weight, out, ops::LinearPolicy::AllowA8, workspace, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FP8_F16: synchronize failed\n";
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
    std::cout << "FP8_F16 n=" << n << " k=" << k << " t=" << t
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
        failures += run_shape(14336, 5120, 1, 931U);
        failures += run_shape(14336, 5120, 8, 932U);
        failures += run_shape(14336, 5120, 33, 933U);
        failures += run_shape(248320, 5120, 1, 934U);
        failures += run_shape(248320, 5120, 9, 935U);
        failures += run_shape(248320, 5120, 128, 936U);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8_F16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8_F16 Linear: " << error.what() << '\n';
        return 1;
    }
}
