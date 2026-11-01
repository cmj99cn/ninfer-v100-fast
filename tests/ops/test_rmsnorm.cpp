#include "ninfer/ops/rmsnorm.h"
#include "ops/norm_test_common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

constexpr ReductionCriterion rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ 3.4e-3};
}

std::vector<double> rmsnorm_oracle(const std::vector<float>& input,
                                   const std::vector<float>& weight, const Shape& shape,
                                   bool unit_offset) {
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
            const double gain     = static_cast<double>(weight[column]) + (unit_offset ? 1.0 : 0.0);
            output[base + column] = static_cast<double>(input[base + column]) * inverse * gain;
        }
    }
    return output;
}

int run_case(const char* label, const Shape& shape, bool unit_offset, std::uint32_t seed,
             float input_scale = 4.0F, bool bf16x2_unaligned = false) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d);
    fill_uniform(input, seed, -input_scale, input_scale);
    fill_uniform(weight, seed + 1U, unit_offset ? -0.5F : 0.25F, unit_offset ? 0.5F : 1.75F);
    round_to_bf16(input);
    round_to_bf16(weight);
    const std::vector<double> reference = rmsnorm_oracle(input, weight, shape, unit_offset);

    DeviceInput device_input  = make_input(input, bf16x2_unaligned);
    DeviceInput device_weight = make_input(weight, bf16x2_unaligned);
    const std::size_t leading = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor output_tensor = tensor_for(output_data, shape);
    ops::rmsnorm(input_tensor, weight_tensor, kEps, unit_offset, output_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(output_data, count), reference,
                                    rmsnorm_bf16_criterion());
    failures += verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += verify_preserved(std::string(label) + " preserves input", device_input);
    failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    return failures;
}

std::uint16_t f32_to_f16_bits(float value) { return __half_as_ushort(__float2half_rn(value)); }
float f16_bits_to_f32(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }

int run_f16_case(const char* label, const Shape& shape, std::uint32_t seed) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d);
    fill_uniform(input, seed, -4.0F, 4.0F);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    std::vector<std::uint16_t> input_bits(count), weight_bits(shape.d);
    for (std::size_t i = 0; i < count; ++i) {
        input_bits[i] = f32_to_f16_bits(input[i]);
        input[i]      = f16_bits_to_f32(input_bits[i]);
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(shape.d); ++i) {
        weight_bits[i] = f32_to_f16_bits(weight[i]);
        weight[i]      = f16_bits_to_f32(weight_bits[i]);
    }
    const auto reference = rmsnorm_oracle(input, weight, shape, false);
    GuardedDeviceBuffer device_input(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(count * sizeof(std::uint16_t));
    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());
    Tensor input_tensor(device_input.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    Tensor weight_tensor(device_weight.data(), DType::FP16, {shape.d});
    Tensor output_tensor(device_output.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    ops::rmsnorm(input_tensor, weight_tensor, kEps, false, output_tensor, nullptr);
    cuda_synchronize();
    const auto out_bits = from_device<std::uint16_t>(device_output.data(), count);
    std::vector<double> actual(count);
    for (std::size_t i = 0; i < count; ++i) {
        actual[i] = static_cast<double>(f16_bits_to_f32(out_bits[i]));
    }
    return verify_reduction(label, actual, reference,
                            {2e-3, 0.02, 3e-3});
}


std::uint16_t f32_to_bf16_bits(float value) {
    const __nv_bfloat16 converted = __float2bfloat16_rn(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

float bf16_bits_to_f32(std::uint16_t bits) {
    __nv_bfloat16 converted;
    std::memcpy(&converted, &bits, sizeof(bits));
    return __bfloat162float(converted);
}

int run_f16_bf16w_case(const char* label, const Shape& shape, std::uint32_t seed) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d);
    fill_uniform(input, seed, -4.0F, 4.0F);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    std::vector<std::uint16_t> input_bits(count), weight_bits(shape.d);
    for (std::size_t i = 0; i < count; ++i) {
        input_bits[i] = f32_to_f16_bits(input[i]);
        input[i]      = f16_bits_to_f32(input_bits[i]);
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(shape.d); ++i) {
        weight_bits[i] = f32_to_bf16_bits(weight[i]);
        weight[i]      = bf16_bits_to_f32(weight_bits[i]);
    }
    const auto reference = rmsnorm_oracle(input, weight, shape, false);
    GuardedDeviceBuffer device_input(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(count * sizeof(std::uint16_t));
    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());
    Tensor input_tensor(device_input.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    Tensor weight_tensor(device_weight.data(), DType::BF16, {shape.d});
    Tensor output_tensor(device_output.data(), DType::FP16, {shape.d, shape.rows, shape.tokens});
    ops::rmsnorm(input_tensor, weight_tensor, kEps, false, output_tensor, nullptr);
    cuda_synchronize();
    const auto out_bits = from_device<std::uint16_t>(device_output.data(), count);
    std::vector<double> actual(count);
    for (std::size_t i = 0; i < count; ++i) {
        actual[i] = static_cast<double>(f16_bits_to_f32(out_bits[i]));
    }
    return verify_reduction(label, actual, reference, {2e-3, 0.02, 3e-3});
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("rmsnorm offset [5120,1]", {5120, 1}, true, 1101U);
    failures += run_case("rmsnorm offset [5120,128]", {5120, 128}, true, 1102U);
    failures += run_case("rmsnorm offset [2048,7]", {2048, 7}, true, 1103U);
    failures += run_case("rmsnorm offset [256,24,7]", {256, 24, 7}, true, 1104U);
    failures += run_case("rmsnorm offset [256,2,1]", {256, 2}, true, 1105U);
    failures += run_case("rmsnorm offset [256,4,48]", {256, 4, 48}, true, 1106U);
    failures += run_case("rmsnorm plain [2048,1]", {2048, 1}, false, 1201U);
    failures += run_case("rmsnorm plain [2048,128]", {2048, 128}, false, 1202U);
    failures += run_case("rmsnorm plain [128,32,7]", {128, 32, 7}, false, 1203U);
    failures += run_case("rmsnorm plain [128,8,128]", {128, 8, 128}, false, 1204U);
    failures += run_case("rmsnorm offset unaligned [128,32]", {128, 32}, true, 1301U, 4.0F, true);
    failures += run_case("rmsnorm plain unaligned [128,8]", {128, 8}, false, 1302U, 4.0F, true);
    failures += run_case("rmsnorm plain near-zero [128,32]", {128, 32}, false, 1303U, 1.0e-5F);
    failures += run_f16_case("rmsnorm f16 [5120,128]", {5120, 128}, 1401U);
    failures += run_f16_case("rmsnorm f16 [2048,7]", {2048, 7}, 1402U);
    failures += run_f16_bf16w_case("rmsnorm f16/bf16w [5120,128]", {5120, 128}, 1501U);
    failures += run_f16_bf16w_case("rmsnorm f16/bf16w [2048,7]", {2048, 7}, 1502U);
    std::cout << (failures ? "FAIL" : "OK") << " rmsnorm\n";
    return failures ? 1 : 0;
}
