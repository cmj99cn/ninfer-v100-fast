#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/sigmoid_gated_rmsnorm.h"
#include "ops/norm_test_common.h"

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

constexpr ReductionCriterion kBf16Criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 4.5e-5,
            /*gross_relative_to_max_reference*/ 2.8e-3};
}

std::vector<double> oracle(const std::vector<float>& input, const std::vector<float>& weight,
                           const std::vector<float>& gate, const Shape& shape, bool sigmoid) {
    std::vector<double> output(input.size());
    const auto rows = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum             = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum / static_cast<double>(shape.d) + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double z          = gate[base + column];
            const double s          = 1.0 / (1.0 + std::exp(-z));
            const double activation = sigmoid ? s : z * s;
            output[base + column]   = static_cast<double>(input[base + column]) * inverse *
                                    static_cast<double>(weight[column]) * activation;
        }
    }
    return output;
}

int distinctness_failures(const char* label, const std::vector<double>& silu,
                          const std::vector<double>& sigmoid) {
    double sum_difference = 0.0;
    double sum_magnitude  = 0.0;
    std::size_t changed   = 0;
    for (std::size_t i = 0; i < silu.size(); ++i) {
        const double difference = std::abs(silu[i] - sigmoid[i]);
        sum_difference += difference;
        sum_magnitude += std::abs(silu[i]);
        if (difference > 1.0e-3) { ++changed; }
    }
    const double ratio = sum_difference / (sum_magnitude + 1.0e-12);
    if (ratio < 0.1 || changed < silu.size() / 2) {
        std::cerr << label << ": silu and sigmoid gates did not separate (ratio " << ratio
                  << ", changed " << changed << "/" << silu.size() << ")\n";
        return 1;
    }
    return 0;
}

int run_bf16_case(const char* label, const Shape& shape, std::uint32_t seed) {
    const std::size_t count              = shape.elements();
    std::vector<float> input(count), weight(shape.d), gate(count);
    fill_uniform(input, seed, -4.0F, 4.0F);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    fill_uniform(gate, seed + 2U, -5.0F, 5.0F);
    round_to_bf16(input);
    round_to_bf16(weight);
    round_to_bf16(gate);
    const auto silu_reference    = oracle(input, weight, gate, shape, false);
    const auto sigmoid_reference = oracle(input, weight, gate, shape, true);

    DeviceInput device_input  = make_input(input, false);
    DeviceInput device_weight = make_input(weight, false);
    DeviceInput device_gate   = make_input(gate, false);
    GuardedDeviceBuffer silu_out(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer sigmoid_out(count * sizeof(std::uint16_t));
    Tensor input_tensor  = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor gate_tensor   = tensor_for(device_gate.data, shape);
    Tensor silu_tensor   = tensor_for(silu_out.data(), shape);
    Tensor sigmoid_tensor = tensor_for(sigmoid_out.data(), shape);

    ops::gated_rmsnorm(input_tensor, weight_tensor, gate_tensor, kEps, silu_tensor, nullptr);
    ops::sigmoid_gated_rmsnorm(input_tensor, weight_tensor, gate_tensor, kEps, sigmoid_tensor,
                               nullptr);
    cuda_synchronize();

    const auto silu_out_host    = from_device_bf16(silu_out.data(), count);
    const auto sigmoid_out_host = from_device_bf16(sigmoid_out.data(), count);
    int failures = verify_reduction(std::string(label) + " silu oracle", silu_out_host,
                                    silu_reference, kBf16Criterion());
    failures += verify_reduction(std::string(label) + " sigmoid oracle", sigmoid_out_host,
                                 sigmoid_reference, kBf16Criterion());
    failures += distinctness_failures(label, silu_out_host, sigmoid_out_host);
    return failures;
}

int run_f16_case(const char* label, const Shape& shape, std::uint32_t seed) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d), gate(count);
    fill_uniform(input, seed, -4.0F, 4.0F);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    fill_uniform(gate, seed + 2U, -5.0F, 5.0F);
    std::vector<std::uint16_t> ib(count), wb(shape.d), zb(count);
    for (std::size_t i = 0; i < count; ++i) {
        ib[i]   = __half_as_ushort(__float2half_rn(input[i]));
        zb[i]   = __half_as_ushort(__float2half_rn(gate[i]));
        input[i] = __half2float(__ushort_as_half(ib[i]));
        gate[i]  = __half2float(__ushort_as_half(zb[i]));
    }
    for (int i = 0; i < shape.d; ++i) {
        wb[i]    = __half_as_ushort(__float2half_rn(weight[i]));
        weight[i] = __half2float(__ushort_as_half(wb[i]));
    }
    const auto silu_reference    = oracle(input, weight, gate, shape, false);
    const auto sigmoid_reference = oracle(input, weight, gate, shape, true);

    GuardedDeviceBuffer di(count * 2), dw(shape.d * 2), dz(count * 2);
    GuardedDeviceBuffer silu_out(count * 2), sigmoid_out(count * 2);
    di.copy_from_host(ib.data(), di.bytes());
    dw.copy_from_host(wb.data(), dw.bytes());
    dz.copy_from_host(zb.data(), dz.bytes());
    Tensor xt(di.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    Tensor wt(dw.data(), DType::FP16, {shape.d});
    Tensor zt(dz.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    Tensor st(silu_out.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    Tensor gt(sigmoid_out.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});

    ops::gated_rmsnorm(xt, wt, zt, kEps, st, nullptr);
    ops::sigmoid_gated_rmsnorm(xt, wt, zt, kEps, gt, nullptr);
    cuda_synchronize();

    const auto s_bits = from_device<std::uint16_t>(silu_out.data(), count);
    const auto g_bits = from_device<std::uint16_t>(sigmoid_out.data(), count);
    std::vector<double> silu_host(count), sigmoid_host(count);
    for (std::size_t i = 0; i < count; ++i) {
        silu_host[i]    = __half2float(__ushort_as_half(s_bits[i]));
        sigmoid_host[i] = __half2float(__ushort_as_half(g_bits[i]));
    }
    const ReductionCriterion f16_criterion{2e-3, 0.02, 3e-3};
    int failures = verify_reduction(std::string(label) + " silu oracle", silu_host, silu_reference,
                                    f16_criterion);
    failures += verify_reduction(std::string(label) + " sigmoid oracle", sigmoid_host,
                                 sigmoid_reference, f16_criterion);
    failures += distinctness_failures(label, silu_host, sigmoid_host);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_bf16_case("gate separation bf16 [128,48]", {128, 48}, 1701U);
    failures += run_bf16_case("gate separation bf16 [128,32,7]", {128, 32, 7}, 1702U);
    failures += run_f16_case("gate separation f16 [128,48]", {128, 48}, 1703U);
    std::cout << (failures ? "FAIL" : "OK") << " sigmoid_gated_rmsnorm separation\n";
    return failures ? 1 : 0;
}
