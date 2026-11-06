#include "ninfer/ops/cast.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::uint32_t> fp32_bits(const std::vector<float>& values) {
    std::vector<std::uint32_t> bits(values.size());
    std::memcpy(bits.data(), values.data(), values.size() * sizeof(float));
    return bits;
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> source(count);
    fill_uniform(source, seed, -64.0f, 64.0f);
    if (count >= 8) {
        const float edges[] = {-65504.0f,       -1.00390625f, -0.0f,      0.0f,
                               0.333251953125f, 1.00390625f,  3.1415927f, 65504.0f};
        std::copy(std::begin(edges), std::end(edges), source.begin());
    }

    std::vector<std::uint16_t> expected(count);
    for (std::size_t i = 0; i < count; ++i) expected[i] = f32_to_bf16(source[i]);
    const auto source_bits = fp32_bits(source);
    GuardedDeviceBuffer device_source(count * sizeof(float));
    GuardedDeviceBuffer device_destination(count * sizeof(std::uint16_t));
    device_source.copy_from_host(source.data(), device_source.bytes());

    Tensor source_tensor(device_source.data(), DType::FP32, {rows, columns});
    Tensor destination_tensor(device_destination.data(), DType::BF16, {rows, columns});
    ops::cast_fp32_to_bf16(source_tensor, destination_tensor, nullptr);
    cuda_synchronize();

    int failures =
        verify_exact(label, from_device<std::uint16_t>(device_destination.data(), count), expected);
    const auto source_after = from_device<float>(device_source.data(), count);
    failures += verify_exact("cast source unchanged", fp32_bits(source_after), source_bits);
    failures += device_source.verify_guards("cast source");
    failures += device_destination.verify_guards("cast destination");
    return failures;
}

std::uint16_t f32_to_f16_bits(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

int run_f16_roundtrip(const char* label, std::int32_t rows, std::int32_t columns,
                      std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> source(count);
    fill_uniform(source, seed, -256.0f, 256.0f);
    round_to_bf16(source);
    std::vector<std::uint16_t> bf16(count), expected_f16(count);
    for (std::size_t i = 0; i < count; ++i) {
        bf16[i]        = f32_to_bf16(source[i]);
        const float v  = __bfloat162float(__ushort_as_bfloat16(bf16[i]));
        expected_f16[i] = f32_to_f16_bits(v);
    }

    GuardedDeviceBuffer device_bf16(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_f16(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_back(count * sizeof(std::uint16_t));
    device_bf16.copy_from_host(bf16.data(), device_bf16.bytes());

    Tensor bf16_tensor(device_bf16.data(), DType::BF16, {rows, columns});
    Tensor f16_tensor(device_f16.data(), DType::FP16, {rows, columns});
    Tensor back_tensor(device_back.data(), DType::BF16, {rows, columns});
    ops::cast_bf16_to_fp16(bf16_tensor, f16_tensor, nullptr);
    ops::cast_fp16_to_bf16(f16_tensor, back_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_exact(label, from_device<std::uint16_t>(device_f16.data(), count),
                                expected_f16);
    failures += verify_exact("cast_f16_to_bf16 roundtrip",
                             from_device<std::uint16_t>(device_back.data(), count), bf16);
    failures += device_bf16.verify_guards("cast roundtrip bf16");
    failures += device_f16.verify_guards("cast roundtrip f16");
    failures += device_back.verify_guards("cast roundtrip back");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("cast [1536,1]", 1536, 1, 101u);
    failures += run_case("cast [1536,257]", 1536, 257, 201u);
    failures += run_case("cast edge-length tensor", 13, 1, 301u);
    failures += run_f16_roundtrip("cast bf16<->fp16 [1536,1]", 1536, 1, 401u);
    failures += run_f16_roundtrip("cast bf16<->fp16 [1536,257]", 1536, 257, 402u);
    std::cout << (failures ? "FAIL" : "OK") << " cast_fp32_to_bf16\n";
    return failures ? 1 : 0;
}
