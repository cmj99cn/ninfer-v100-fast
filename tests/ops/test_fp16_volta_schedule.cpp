#include "core/tensor.h"
#include "ops/linear/fp16/fp16_launch.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

#ifdef NINFER_VOLTA_BUILD
namespace {

Weight make_weight(void* data, std::int32_t n, std::int32_t k) {
    Weight weight{};
    weight.qtype         = QType::FP16_CTRL;
    weight.layout        = QuantLayout::Contiguous;
    weight.ndim          = 2;
    weight.n             = n;
    weight.k             = k;
    weight.shape[0]      = n;
    weight.shape[1]      = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    weight.qdata         = data;
    weight.payload_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k) * 2U;
    return weight;
}

int run_case(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    std::vector<std::uint16_t> weight_words(static_cast<std::size_t>(n) * k);
    std::vector<float> x_values(static_cast<std::size_t>(k));
    for (std::uint16_t& word : weight_words) {
        word = f32_to_f16(distribution(generator));
    }
    for (float& value : x_values) { value = bf16_to_f32(f32_to_bf16(distribution(generator))); }

    std::vector<std::uint16_t> x1_words(static_cast<std::size_t>(k));
    for (std::int32_t i = 0; i < k; ++i) { x1_words[static_cast<std::size_t>(i)] = f32_to_bf16(x_values[static_cast<std::size_t>(i)]); }
    std::vector<std::uint16_t> x2_words(static_cast<std::size_t>(k) * 2U);
    for (std::int32_t i = 0; i < k; ++i) {
        x2_words[static_cast<std::size_t>(i)] = x1_words[static_cast<std::size_t>(i)];
        x2_words[static_cast<std::size_t>(i) + static_cast<std::size_t>(k)] =
            x1_words[static_cast<std::size_t>(i)];
    }

    DeviceBuffer device_weight   = to_device(weight_words);
    DeviceBuffer device_x1       = to_device(x1_words);
    DeviceBuffer device_x2       = to_device(x2_words);
    DeviceBuffer device_out1     = DeviceBuffer(static_cast<std::size_t>(n) * 2U);
    DeviceBuffer device_out2     = DeviceBuffer(static_cast<std::size_t>(n) * 2U * 2U);
    const Weight weight = make_weight(device_weight.p, n, k);

    const Tensor x1_tensor{device_x1.p, DType::BF16, {k, 1}};
    Tensor out1_tensor{device_out1.p, DType::BF16, {n, 1}};
    const Tensor x2_tensor{device_x2.p, DType::BF16, {k, 2}};
    Tensor out2_tensor{device_out2.p, DType::BF16, {n, 2}};

    ops::detail::launch_fp16_volta_mma(x1_tensor, weight, out1_tensor, nullptr);
    ops::detail::launch_fp16_volta_mma(x2_tensor, weight, out2_tensor, nullptr);
    cuda_synchronize();

    const std::vector<std::uint16_t> bits1 =
        from_device<std::uint16_t>(device_out1.p, static_cast<std::size_t>(n));
    const std::vector<std::uint16_t> bits2 =
        from_device<std::uint16_t>(device_out2.p, static_cast<std::size_t>(n));

    int failures = 0;
    for (std::int32_t i = 0; i < n; ++i) {
        if (bits1[static_cast<std::size_t>(i)] != bits2[static_cast<std::size_t>(i)]) {
            std::cerr << "fp16 volta schedule n=" << n << " k=" << k << ": decode/tile bit "
                      << i << " differs (" << bits1[static_cast<std::size_t>(i)] << " vs "
                      << bits2[static_cast<std::size_t>(i)] << ")\n";
            ++failures;
            break;
        }
    }
    return failures;
}

} // namespace
#endif

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
#ifdef NINFER_VOLTA_BUILD
    int failures = 0;
    failures += run_case(320, 10240, 41U);
    failures += run_case(1024, 320, 42U);
    failures += run_case(64, 256, 43U);
    std::cout << (failures ? "FAIL" : "OK") << " fp16 volta schedule\n";
    return failures ? 1 : 0;
#else
    std::cout << "SKIP: no Volta route\n";
    return 77;
#endif
}
