#include "ninfer/ops/linear_split.h"

#include "core/arena.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kK = 5120;

std::vector<std::uint16_t> activation_bits(std::size_t count, std::uint32_t seed) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -4.0F, 4.0F);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<std::uint16_t> bf16_rows(std::int32_t row_begin, std::int32_t row_count,
                                     std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(row_count) * kK);
    for (std::int32_t row = 0; row < row_count; ++row) {
        for (std::int32_t column = 0; column < kK; ++column) {
            std::uint32_t hash = static_cast<std::uint32_t>(row + row_begin) * 0x9e3779b9U ^
                                 static_cast<std::uint32_t>(column) * 0x85ebca6bU ^
                                 seed * 0xc2b2ae35U;
            hash ^= hash >> 16;
            hash *= 0x7feb352dU;
            hash ^= hash >> 15;
            const int centered = static_cast<int>((hash >> 8) & 0xffU) - 128;
            bits[static_cast<std::size_t>(row) * kK + column] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 1024.0F));
        }
    }
    return bits;
}

bool throws_before_launch(const std::function<void()>& call) {
    try {
        call();
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

struct SplitCase {
    std::string label;
    QType qtype;
    std::int32_t parent_rows;
    std::vector<LinearSplitSegment> segments;
    std::uint32_t seed;
    std::int32_t T;
    LinearPolicy policy;
};

const PointwiseCriterion kCriterion{1.0e-3, 8.0e-3};

int run_split_case(const SplitCase& c, cudaStream_t stream) {
    int failures = 0;

    quantized_weight::PackedWeight parent;
    std::vector<std::uint16_t> parent_bits;
    Weight parent_weight;
    DeviceBuffer d_parent;
    if (c.qtype == QType::BF16_CTRL) {
        parent_bits = bf16_rows(0, c.parent_rows, c.seed);
        d_parent = to_device(parent_bits);
        parent_weight = direct_bf16_weight::HostWeight{c.parent_rows, kK, parent_bits}
                            .device_weight(d_parent.p);
    } else {
        parent = quantized_weight::make_patterned_weight(c.qtype, c.parent_rows, kK, c.seed);
        d_parent = to_device(parent.payload);
        parent_weight = parent.device_weight(d_parent.p);
    }

    const std::vector<std::uint16_t> x_bits =
        activation_bits(static_cast<std::size_t>(kK) * c.T, c.seed + 7U);
    DeviceBuffer d_x = to_device(x_bits);
    Tensor x = Tensor(d_x.p, DType::BF16, {kK, c.T});

    std::vector<DeviceBuffer> outputs;
    std::vector<Tensor> output_tensors;
    for (const LinearSplitSegment& segment : c.segments) {
        DeviceBuffer out(static_cast<std::size_t>(segment.row_count) * c.T * 2);
        out.fill(0);
        output_tensors.push_back(
            Tensor(out.p, DType::BF16, {segment.row_count, c.T}));
        outputs.push_back(std::move(out));
    }

    const std::size_t capacity = ops::linear_split_workspace_capacity_bytes(
        c.qtype, c.parent_rows, kK, c.segments, c.policy, c.T, c.T);
    std::size_t expected_capacity = 0;
    for (const LinearSplitSegment& segment : c.segments) {
        expected_capacity = std::max(expected_capacity, ops::linear_workspace_capacity_bytes(
                                                            c.qtype, segment.row_count, kK, c.policy,
                                                            c.T, c.T));
    }
    if (capacity != expected_capacity) {
        std::cerr << c.label << ": capacity is not the maximum segment requirement\n";
        ++failures;
    }
    DeviceArena arena(std::max<std::size_t>(capacity, 256));

    std::span<const LinearSplitSegment> segments(c.segments);
    std::span<Tensor> outputs_span(output_tensors);
    ops::linear_split(x, parent_weight, segments, outputs_span, c.policy, arena, stream);
    cuda_synchronize(stream);
    if (arena.peak_used() > capacity) {
        std::cerr << c.label << ": workspace usage exceeded the reported capacity\n";
        ++failures;
    }

    std::vector<double> activation(static_cast<std::size_t>(kK) * c.T);
    for (std::size_t i = 0; i < activation.size(); ++i) {
        activation[i] = static_cast<double>(bf16_to_f32(x_bits[i]));
    }

    for (std::size_t index = 0; index < c.segments.size(); ++index) {
        const LinearSplitSegment& segment = c.segments[index];
        const std::size_t count = static_cast<std::size_t>(segment.row_count) * c.T;

        std::vector<double> oracle;
        oracle.reserve(count);
        for (std::int32_t token = 0; token < c.T; ++token) {
            for (std::int32_t row = 0; row < segment.row_count; ++row) {
                double accumulator = 0.0;
                for (std::int32_t column = 0; column < kK; ++column) {
                    double weight = 0.0;
                    if (c.qtype == QType::BF16_CTRL) {
                        weight = static_cast<double>(bf16_to_f32(
                            parent_bits[static_cast<std::size_t>(segment.row_begin + row) * kK +
                                       static_cast<std::size_t>(column)]));
                    } else {
                        weight = quantized_weight::logical_weight_fp64(
                            parent, segment.row_begin + row, column);
                    }
                    accumulator += weight * activation[static_cast<std::size_t>(token) * kK +
                                                       static_cast<std::size_t>(column)];
                }
                oracle.push_back(static_cast<double>(
                    bf16_to_f32(f32_to_bf16(static_cast<float>(accumulator)))));
            }
        }

        failures += verify_pointwise((c.label + " segment " + std::to_string(index)).c_str(),
                                     from_device_bf16(outputs[index].p, count), oracle, kCriterion);
    }

    return failures;
}

int run_rejections(cudaStream_t stream, WorkspaceArena& arena) {
    int failures = 0;

    quantized_weight::PatternedWeightOptions nvfp4_options;
    nvfp4_options.weight_scale_divisor = 1.5F;
    nvfp4_options.input_scale_divisor  = 2.0F;
    const quantized_weight::PackedWeight nvfp4 =
        quantized_weight::make_patterned_weight(QType::NVFP4, 1024, kK, 101U, nvfp4_options);
    DeviceBuffer d_nvfp4 = to_device(nvfp4.payload);
    const Weight nvfp4_weight = nvfp4.device_weight(d_nvfp4.p);
    const std::vector<std::uint16_t> x_bits = activation_bits(static_cast<std::size_t>(kK) * 1, 102U);
    DeviceBuffer d_x = to_device(x_bits);
    Tensor x = Tensor(d_x.p, DType::BF16, {kK, 1});
    DeviceBuffer d_out(1024 * 2);
    Tensor out = Tensor(d_out.p, DType::BF16, {1024, 1});

    if (!throws_before_launch([&]() {
            const std::vector<LinearSplitSegment> segments{{0, 1024}};
            std::vector<Tensor> outputs{out};
            ops::linear_split(x, nvfp4_weight, segments, outputs, LinearPolicy::A16Only, arena, stream);
        })) {
        std::cerr << "linear_split: NVFP4 parent was not rejected\n";
        ++failures;
    }

    quantized_weight::PackedWeight w8 =
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 1024, kK, 103U);
    DeviceBuffer d_w8 = to_device(w8.payload);
    Weight row_scale = w8.device_weight(d_w8.p);
    row_scale.layout = QuantLayout::RowScale;
    if (!throws_before_launch([&]() {
            const std::vector<LinearSplitSegment> segments{{0, 1024}};
            std::vector<Tensor> outputs{out};
            ops::linear_split(x, row_scale, segments, outputs, LinearPolicy::A16Only, arena, stream);
        })) {
        std::cerr << "linear_split: row-scaled parent was not rejected\n";
        ++failures;
    }

    Weight ggml = w8.device_weight(d_w8.p);
    ggml.qtype = QType::GGML_K;
    if (!throws_before_launch([&]() {
            const std::vector<LinearSplitSegment> segments{{0, 1024}};
            std::vector<Tensor> outputs{out};
            ops::linear_split(x, ggml, segments, outputs, LinearPolicy::A16Only, arena, stream);
        })) {
        std::cerr << "linear_split: GGML_K parent was not rejected\n";
        ++failures;
    }

    const Weight registered_w8 = w8.device_weight(d_w8.p);
    if (!throws_before_launch([&]() {
            const std::vector<LinearSplitSegment> segments{{512, 1024}};
            std::vector<Tensor> outputs{out};
            ops::linear_split(x, registered_w8, segments, outputs, LinearPolicy::A16Only, arena, stream);
        })) {
        std::cerr << "linear_split: out-of-range segment was not rejected\n";
        ++failures;
    }

    if (!throws_before_launch([&]() {
            const std::vector<LinearSplitSegment> segments{{0, 512}};
            DeviceBuffer wrong(1024 * 2);
            std::vector<Tensor> outputs{Tensor(wrong.p, DType::BF16, {1024, 1})};
            ops::linear_split(x, registered_w8, segments, outputs, LinearPolicy::A16Only, arena, stream);
        })) {
        std::cerr << "linear_split: output shape mismatch was not rejected\n";
        ++failures;
    }

    if (!throws_before_launch([&]() {
            const std::vector<LinearSplitSegment> segments{};
            std::vector<Tensor> outputs{};
            ops::linear_split(x, registered_w8, segments, outputs, LinearPolicy::A16Only, arena, stream);
        })) {
        std::cerr << "linear_split: empty segment list was not rejected\n";
        ++failures;
    }

    DeviceBuffer d_fp16(1024 * 2);
    Tensor fp16_x = Tensor(d_fp16.p, DType::FP16, {kK, 1});
    if (!throws_before_launch([&]() {
            const std::vector<LinearSplitSegment> segments{{0, 1024}};
            std::vector<Tensor> outputs{out};
            ops::linear_split(fp16_x, registered_w8, segments, outputs, LinearPolicy::A16Only, arena, stream);
        })) {
        std::cerr << "linear_split: non-BF16 activation was not rejected\n";
        ++failures;
    }

    if (!throws_before_launch([&]() {
            const std::vector<LinearSplitSegment> segments{{0, 1024}};
            (void)ops::linear_split_workspace_capacity_bytes(QType::NVFP4, 1024, kK, segments,
                                                       LinearPolicy::A16Only, 1, 16);
        })) {
        std::cerr << "linear_split workspace: unsupported format was not rejected\n";
        ++failures;
    }

    if (!throws_before_launch([&]() {
            const std::vector<LinearSplitSegment> segments{{0, 1024}};
            (void)ops::linear_split_workspace_capacity_bytes(QType::W8G32_F16S, 1024, kK, segments,
                                                       LinearPolicy::A16Only, 0, 16);
        })) {
        std::cerr << "linear_split workspace: invalid token interval was not rejected\n";
        ++failures;
    }

    cuda_synchronize(stream);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    DeviceArena arena(65536);

    int failures = 0;

    const std::vector<LinearPolicy> policies{LinearPolicy::A16Only, LinearPolicy::AllowA8};
    const std::vector<std::int32_t> token_counts{1, 16};

    const std::vector<std::pair<QType, std::vector<LinearSplitSegment>>> cases{
        {QType::Q4G64_F16S, {{0, 1024}, {1024, 2048}, {3072, 1024}}},
        {QType::Q5G64_F16S, {{0, 1024}, {1024, 3072}}},
        {QType::Q6G64_F16S, {{0, 62080}, {62080, 62080}}},
        {QType::W8G32_F16S, {{0, 1024}, {1024, 3072}}},
        {QType::BF16_CTRL, {{0, 7168}, {7168, 7168}}}};

    for (const auto& entry : cases) {
        std::int32_t parent_rows = 0;
        for (const LinearSplitSegment& segment : entry.second) {
            parent_rows += segment.row_count;
        }
        for (const LinearPolicy policy : policies) {
            if (entry.first == QType::BF16_CTRL && policy != LinearPolicy::A16Only) continue;
            for (std::int32_t t : token_counts) {
                SplitCase c;
                c.label = "linear_split " + std::to_string(static_cast<int>(entry.first)) +
                          " T=" + std::to_string(t);
                c.qtype = entry.first;
                c.parent_rows = parent_rows;
                c.segments = entry.second;
                c.seed = 201U;
                c.T = t;
                c.policy = policy;
                failures += run_split_case(c, stream);
            }
        }
    }

    failures += run_rejections(stream, arena);

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    if (failures == 0) std::cout << "linear_split_test: PASS\n";
    return failures == 0 ? 0 : 1;
}
