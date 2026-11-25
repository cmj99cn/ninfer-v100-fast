#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;

constexpr std::int32_t kInputRows = 2560;
constexpr std::int32_t kQueryGateRows = 6144;
constexpr std::int32_t kKeyValueRows = 1664;

std::vector<__half> random_activation(std::int32_t columns, std::uint32_t seed) {
    std::vector<__half> values(static_cast<std::size_t>(kInputRows) * columns);
    std::mt19937 rng(seed + 17U);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& value : values) { value = __float2half_rn(dist(rng)); }
    return values;
}

int compare_qsa_projection(const char* label, const Weight& weight, QType qtype,
                           std::int32_t parent_rows,
                           std::span<const ops::LinearSplitSegment> segments,
                           const std::vector<__half>& activation, std::int32_t columns) {
    DeviceBuffer x_bytes(activation.size() * sizeof(__half));
    x_bytes.copy_from_host(activation.data(), x_bytes.bytes);
    Tensor x(x_bytes.p, DType::FP16, {kInputRows, columns});

    std::vector<GuardedDeviceBuffer> batched_buffers;
    std::vector<GuardedDeviceBuffer> sequential_buffers;
    std::vector<Tensor> batched_outputs;
    std::vector<Tensor> sequential_outputs;
    for (const auto& segment : segments) {
        const std::size_t words = static_cast<std::size_t>(segment.row_count) * columns;
        batched_buffers.emplace_back(words * sizeof(__half));
        sequential_buffers.emplace_back(words * sizeof(__half));
        batched_buffers.back().fill(0xff);
        sequential_buffers.back().fill(0xff);
        batched_outputs.emplace_back(batched_buffers.back().data(), DType::FP16,
                                     std::initializer_list<std::int32_t>{segment.row_count, columns});
        sequential_outputs.emplace_back(
            sequential_buffers.back().data(), DType::FP16,
            std::initializer_list<std::int32_t>{segment.row_count, columns});
    }

    const std::size_t capacity = ops::linear_split_workspace_capacity_bytes(
        qtype, parent_rows, kInputRows, segments, ops::LinearPolicy::A16Only, 1, columns);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 1));
    ops::linear_split(x, weight, segments, batched_outputs, ops::LinearPolicy::A16Only, workspace,
                      0);
    for (std::int32_t c = 0; c < columns; ++c) {
        const Tensor column = x.slice(1, c, 1);
        std::vector<Tensor> column_outputs;
        for (std::size_t s = 0; s < segments.size(); ++s) {
            column_outputs.push_back(sequential_outputs[s].slice(1, c, 1));
        }
        ops::linear_split(column, weight, segments, column_outputs, ops::LinearPolicy::A16Only,
                          workspace, 0);
    }
    cuda_synchronize();

    int failures = 0;
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const std::size_t words = static_cast<std::size_t>(segments[s].row_count) * columns;
        failures += verify_exact((std::string(label) + " segment " + std::to_string(s) +
                                  " raw bits")
                                     .c_str(),
                                 from_device<std::uint16_t>(batched_buffers[s].data(), words),
                                 from_device<std::uint16_t>(sequential_buffers[s].data(), words));
        failures += batched_buffers[s].verify_guards(std::string(label) + " batched guard");
        failures += sequential_buffers[s].verify_guards(std::string(label) + " sequential guard");
    }
    return failures;
}

int run_qsa_projection_case(std::int32_t columns, std::uint32_t seed) {
    constexpr std::array<ops::LinearSplitSegment, 2> query_gate_segments{{
        {0, 3072},
        {3072, 3072},
    }};
    constexpr std::array<ops::LinearSplitSegment, 4> key_value_segments{{
        {0, 512},
        {512, 512},
        {1024, 512},
        {1536, 128},
    }};

    const auto activation = random_activation(columns, seed);

    auto packed = make_w8g32_f16s_weight(kQueryGateRows, kInputRows, seed + 1U);
    DeviceBuffer w8_bytes(packed.payload.size());
    w8_bytes.copy_from_host(packed.payload.data(), w8_bytes.bytes);
    int failures = compare_qsa_projection("qsa query/gate W8", packed.device_weight(w8_bytes.p),
                                          QType::W8G32_F16S, kQueryGateRows, query_gate_segments,
                                          activation, columns);

    const auto bf16 = direct_bf16_weight::make_patterned(kKeyValueRows, kInputRows, seed + 2U);
    std::vector<std::uint16_t> fp16_bits(bf16.bits.size());
    for (std::size_t i = 0; i < bf16.bits.size(); ++i) {
        fp16_bits[i] = f32_to_f16(bf16_to_f32(bf16.bits[i]));
    }
    DeviceBuffer fp16_bytes(fp16_bits.size() * sizeof(std::uint16_t));
    fp16_bytes.copy_from_host(fp16_bits.data(), fp16_bytes.bytes);
    Weight fp16_weight = bf16.device_weight(fp16_bytes.p);
    fp16_weight.qtype  = QType::FP16_CTRL;
    failures += compare_qsa_projection("qsa key/value FP16", fp16_weight, QType::FP16_CTRL,
                                       kKeyValueRows, key_value_segments, activation, columns);
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_qsa_projection_case(2, 4101U);
    failures += run_qsa_projection_case(4, 4102U);
    failures += run_qsa_projection_case(8, 4103U);
    if (failures == 0) { std::cout << "OK qsa projection cross-profile bitwise\n"; }
    return failures;
}
