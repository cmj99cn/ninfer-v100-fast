#include "ninfer/ops/sigmoid_mul.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion sigmoid_mul_bf16_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.05e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> sigmoid_mul_oracle(const std::vector<float>& gate,
                                       const std::vector<float>& x) {
    std::vector<double> expected(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i];
        expected[i]    = static_cast<double>(x[i]) / (1.0 + std::exp(-g));
    }
    return expected;
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), x(count);
    fill_uniform(gate, seed, -12.0f, 12.0f);
    fill_uniform(x, seed + 1, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(x);

    const auto expected  = sigmoid_mul_oracle(gate, x);
    const auto gate_bits = encode_bf16(gate);
    const auto x_bits    = encode_bf16(x);
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor x_tensor(device_x.data(), DType::BF16, {rows, columns});
    ops::sigmoid_mul(gate_tensor, x_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_x.data(), count), expected,
                                    sigmoid_mul_bf16_criterion());
    failures += verify_exact("sigmoid_mul gate unchanged",
                             from_device<std::uint16_t>(device_gate.data(), count), gate_bits);
    failures += device_gate.verify_guards("sigmoid_mul gate");
    failures += device_x.verify_guards("sigmoid_mul x");
    return failures;
}

int run_edge_case() {
    std::vector<float> gate{-40.0f, -12.0f, -1.0f, 0.0f, 1.0f, 12.0f, 40.0f};
    std::vector<float> x{7.0f, -6.0f, 5.0f, -4.0f, 3.0f, -2.0f, 1.0f};
    round_to_bf16(gate);
    round_to_bf16(x);

    const auto expected  = sigmoid_mul_oracle(gate, x);
    const auto gate_bits = encode_bf16(gate);
    const auto x_bits    = encode_bf16(x);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {static_cast<int>(gate.size())});
    Tensor x_tensor(device_x.data(), DType::BF16, {static_cast<int>(x.size())});
    ops::sigmoid_mul(gate_tensor, x_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("sigmoid_mul edge values",
                                    from_device_bf16(device_x.data(), expected.size()), expected,
                                    sigmoid_mul_bf16_criterion());
    failures +=
        verify_exact("sigmoid_mul edge gate unchanged",
                     from_device<std::uint16_t>(device_gate.data(), gate_bits.size()), gate_bits);
    failures += device_gate.verify_guards("sigmoid_mul edge gate");
    failures += device_x.verify_guards("sigmoid_mul edge x");
    return failures;
}


std::uint16_t f32_to_f16_bits(float value) {
    const __half converted = __float2half_rn(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

float f16_bits_to_f32(std::uint16_t bits) {
    __half converted;
    std::memcpy(&converted, &bits, sizeof(bits));
    return __half2float(converted);
}

int run_f16_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), x(count);
    fill_uniform(gate, seed, -12.0f, 12.0f);
    fill_uniform(x, seed + 1, -8.0f, 8.0f);
    std::vector<std::uint16_t> gate_bits(count), x_bits(count);
    for (std::size_t i = 0; i < count; ++i) {
        gate_bits[i] = f32_to_f16_bits(gate[i]);
        x_bits[i]    = f32_to_f16_bits(x[i]);
        gate[i]      = f16_bits_to_f32(gate_bits[i]);
        x[i]         = f16_bits_to_f32(x_bits[i]);
    }
    const auto expected = sigmoid_mul_oracle(gate, x);
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    Tensor gate_tensor(device_gate.data(), DType::FP16, {rows, columns});
    Tensor x_tensor(device_x.data(), DType::FP16, {rows, columns});
    ops::sigmoid_mul(gate_tensor, x_tensor, nullptr);
    cuda_synchronize();
    const auto out_bits = from_device<std::uint16_t>(device_x.data(), count);
    std::vector<double> actual(count);
    for (std::size_t i = 0; i < count; ++i) actual[i] = f16_bits_to_f32(out_bits[i]);
    int failures = verify_pointwise(label, actual, expected, {2.0e-3, 1.2e-2});
    failures += verify_exact("sigmoid_mul f16 gate unchanged",
                             from_device<std::uint16_t>(device_gate.data(), count), gate_bits);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("sigmoid_mul [6144,1]", 6144, 1, 101u);
    failures += run_case("sigmoid_mul [6144,48]", 6144, 48, 102u);
    failures += run_case("sigmoid_mul [4096,17]", 4096, 17, 201u);
    failures += run_case("sigmoid_mul [4096,128]", 4096, 128, 301u);
    failures += run_f16_case("sigmoid_mul f16 [6144,1]", 6144, 1, 401u);
    failures += run_f16_case("sigmoid_mul f16 [4096,17]", 4096, 17, 402u);
    failures += run_edge_case();
    std::cout << (failures ? "FAIL" : "OK") << " sigmoid_mul\n";
    return failures ? 1 : 0;
}
