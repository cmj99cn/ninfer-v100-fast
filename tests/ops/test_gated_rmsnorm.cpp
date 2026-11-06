#include "ninfer/ops/gated_rmsnorm.h"
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

constexpr ReductionCriterion gated_rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 4.5e-5,
            /*gross_relative_to_max_reference*/ 2.8e-3};
}

std::vector<double> gated_rmsnorm_oracle(const std::vector<float>& input,
                                         const std::vector<float>& weight,
                                         const std::vector<float>& gate, const Shape& shape) {
    std::vector<double> output(input.size());
    const auto row_count = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(shape.d) + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gate_value = gate[base + column];
            const double silu       = gate_value / (1.0 + std::exp(-gate_value));
            output[base + column]   = static_cast<double>(input[base + column]) * inverse *
                                    static_cast<double>(weight[column]) * silu;
        }
    }
    return output;
}

int run_case(const char* label, const Shape& shape, std::uint32_t seed, float input_scale = 4.0F,
             bool bf16x2_unaligned = false) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d), gate(count);
    fill_uniform(input, seed, -input_scale, input_scale);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    fill_uniform(gate, seed + 2U, -5.0F, 5.0F);
    round_to_bf16(input);
    round_to_bf16(weight);
    round_to_bf16(gate);
    const std::vector<double> reference = gated_rmsnorm_oracle(input, weight, gate, shape);

    DeviceInput device_input  = make_input(input, bf16x2_unaligned);
    DeviceInput device_weight = make_input(weight, bf16x2_unaligned);
    DeviceInput device_gate   = make_input(gate, bf16x2_unaligned);
    const std::size_t leading = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor gate_tensor   = tensor_for(device_gate.data, shape);
    Tensor output_tensor = tensor_for(output_data, shape);
    ops::gated_rmsnorm(input_tensor, weight_tensor, gate_tensor, kEps, output_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(output_data, count), reference,
                                    gated_rmsnorm_bf16_criterion());
    failures += verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += verify_preserved(std::string(label) + " preserves input", device_input);
    failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    failures += verify_preserved(std::string(label) + " preserves gate", device_gate);
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
        ib[i] = __half_as_ushort(__float2half_rn(input[i]));
        zb[i] = __half_as_ushort(__float2half_rn(gate[i]));
        input[i] = __half2float(__ushort_as_half(ib[i]));
        gate[i] = __half2float(__ushort_as_half(zb[i]));
    }
    for (int i = 0; i < shape.d; ++i) {
        wb[static_cast<std::size_t>(i)] = __half_as_ushort(__float2half_rn(weight[static_cast<std::size_t>(i)]));
        weight[static_cast<std::size_t>(i)] = __half2float(__ushort_as_half(wb[static_cast<std::size_t>(i)]));
    }
    const auto ref = gated_rmsnorm_oracle(input, weight, gate, shape);
    GuardedDeviceBuffer di(count*2), dw(shape.d*2), dz(count*2), dout(count*2);
    di.copy_from_host(ib.data(), di.bytes()); dw.copy_from_host(wb.data(), dw.bytes());
    dz.copy_from_host(zb.data(), dz.bytes());
    Tensor xt(di.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    Tensor wt(dw.data(), DType::FP16, {shape.d});
    Tensor zt(dz.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    Tensor ot(dout.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    ops::gated_rmsnorm(xt, wt, zt, kEps, ot, nullptr);
    cuda_synchronize();
    const auto bits = from_device<std::uint16_t>(dout.data(), count);
    std::vector<double> got(count);
    for (std::size_t i = 0; i < count; ++i) got[i] = __half2float(__ushort_as_half(bits[i]));
    return verify_reduction(label, got, ref, {2e-3, 0.02, 3e-3});
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("gated_rmsnorm [128,48,1]", {128, 48}, 1401U);
    failures += run_case("gated_rmsnorm [128,48,48]", {128, 48, 48}, 1406U);
    failures += run_case("gated_rmsnorm [128,32,7]", {128, 32, 7}, 1402U);
    failures += run_case("gated_rmsnorm [128,32,128]", {128, 32, 128}, 1403U);
    failures += run_case("gated_rmsnorm near-zero [128,32]", {128, 32}, 1404U, 1.0e-5F);
    failures += run_case("gated_rmsnorm unaligned [128,48]", {128, 48}, 1405U, 4.0F, true);
    failures += run_f16_case("gated_rmsnorm f16 [128,48]", {128, 48}, 1501U);
    failures += run_f16_case("gated_rmsnorm f16 [128,32,7]", {128, 32, 7}, 1502U);
    std::cout << (failures ? "FAIL" : "OK") << " gated_rmsnorm\n";
    return failures ? 1 : 0;
}
