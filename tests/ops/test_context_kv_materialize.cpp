#include "ninfer/ops/context_kv_materialize.h"

#include "core/arena.h"
#include "ops/op_tester.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops;

namespace {

constexpr int kHidden = 2560;
constexpr int kRows   = 1024;
constexpr int kHeadDim = 128;
constexpr int kHeads   = 8;

constexpr ReductionCriterion kBf16Criterion{
    .relative_l2                     = 4.0e-3,
    .gross_absolute                  = 4.0e-3,
    .gross_relative_to_max_reference = 5.0e-3,
};

Weight make_w8(std::int32_t rows, std::int32_t columns, DeviceBuffer& storage,
               std::vector<float>& decoded, std::uint32_t seed) {
    const std::size_t code_bytes  = static_cast<std::size_t>(rows) * columns;
    const std::size_t scale_bytes = static_cast<std::size_t>(rows) * (columns / 32) * 2;
    std::vector<std::uint8_t> codes(code_bytes);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(rows) * (columns / 32));
    std::uint32_t state = seed;
    for (auto& code : codes) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        code = static_cast<std::uint8_t>(static_cast<int>(state % 255) - 127);
    }
    for (auto& scale : scales) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        scale = f32_to_f16(static_cast<float>(1 + state % 17) / 512.0f);
    }
    std::vector<std::uint8_t> payload(code_bytes + scale_bytes);
    std::memcpy(payload.data(), codes.data(), code_bytes);
    std::memcpy(payload.data() + code_bytes, scales.data(), scale_bytes);
    storage = to_device(payload);
    decoded.resize(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const auto code = static_cast<std::int8_t>(codes[static_cast<std::size_t>(row) * columns + column]);
            const auto bits = scales[static_cast<std::size_t>(row) * (columns / 32) + column / 32];
            decoded[static_cast<std::size_t>(row) * columns + column] =
                static_cast<float>(code) * f16_to_f32(bits);
        }
    }
    Weight weight{};
    weight.payload = storage.p;
    weight.payload_bytes = payload.size();
    weight.qtype = QType::W8G32_F16S;
    weight.layout = QuantLayout::RowSplit;
    weight.scale_dtype = DType::FP16;
    weight.group_size = 32;
    weight.group = 32;
    weight.ndim = 2;
    weight.n = rows;
    weight.k = columns;
    weight.shape[0] = rows;
    weight.shape[1] = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    weight.qdata = storage.p;
    weight.scales = static_cast<const std::uint8_t*>(storage.p) + code_bytes;
    return weight;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "context_kv_materialize: SKIP (CUDA unavailable)\n";
        return 77;
    }
    constexpr int width = 6;
    constexpr int batch = 2;
    constexpr int capacity = 2048;
    constexpr int lanes = 2;
    constexpr float theta = 10000.0f;
    const std::vector<int> counts{4, 5};
    const std::vector<int> slots{0, 1};
    std::vector<int> positions(static_cast<std::size_t>(width) * batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < width; ++i) {
            positions[static_cast<std::size_t>(b) * width + i] = (b == 0 ? 2046 : 100) + i;
        }
    }
    std::vector<float> context(static_cast<std::size_t>(kHidden) * width * batch);
    std::vector<float> key_norm(kHeadDim);
    fill_uniform(context, 701u, -0.5f, 0.5f);
    fill_uniform(key_norm, 709u, 0.5f, 1.5f);
    round_to_bf16(context);
    round_to_bf16(key_norm);

    std::vector<float> key_decoded, value_decoded;
    DeviceBuffer key_storage, value_storage, norm_storage;
    Weight key_weight = make_w8(kRows, kHidden, key_storage, key_decoded, 811u);
    Weight value_weight = make_w8(kRows, kHidden, value_storage, value_decoded, 821u);
    norm_storage = to_device_bf16(key_norm);
    Tensor norm_t(norm_storage.p, DType::BF16, {kHeadDim});

    const std::size_t cache_elements =
        static_cast<std::size_t>(kHeadDim) * capacity * kHeads * lanes;
    const std::uint16_t sentinel = f32_to_bf16(1.0f);
    std::vector<std::uint16_t> sentinel_bits(cache_elements, sentinel);

    std::vector<DeviceBuffer> cache_k(kContextKVMaterializeLayers);
    std::vector<DeviceBuffer> cache_v(kContextKVMaterializeLayers);
    std::array<ops::ContextKVMaterializeLayerView, kContextKVMaterializeLayers> layers{};
    for (int layer = 0; layer < kContextKVMaterializeLayers; ++layer) {
        cache_k[layer] = to_device(sentinel_bits);
        cache_v[layer] = to_device(sentinel_bits);
        layers[layer] = {
            .key_weight = key_weight,
            .value_weight = value_weight,
            .key_norm_weight = norm_t,
            .cache = {
                .k = Tensor(cache_k[layer].p, DType::BF16, {kHeadDim, capacity, kHeads, lanes}),
                .v = Tensor(cache_v[layer].p, DType::BF16, {kHeadDim, capacity, kHeads, lanes}),
                .capacity = capacity,
                .padded_capacity = capacity,
                .num_kv_heads = kHeads,
                .head_dim = kHeadDim,
                .lane_capacity = lanes,
            }};
    }

    DeviceBuffer d_context = to_device_bf16(context);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_counts = to_device(counts);
    DeviceBuffer d_slots = to_device(slots);
    Tensor context_t(d_context.p, DType::BF16, {kHidden, width, batch});
    Tensor positions_t(d_positions.p, DType::I32, {width, batch});
    Tensor counts_t(d_counts.p, DType::I32, {batch});
    Tensor slots_t(d_slots.p, DType::I32, {batch});
    DeviceArena workspace(ops::context_kv_materialize_workspace_capacity_bytes(batch, 1, width));
    ops::context_kv_materialize(context_t, positions_t, counts_t, slots_t, layers, theta,
                                {4, 5}, workspace, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> expected_k = sentinel_bits;
    std::vector<std::uint16_t> expected_v = sentinel_bits;
    for (int b = 0; b < batch; ++b) {
        for (int local = 0; local < counts[b]; ++local) {
            const std::int64_t column = b * width + local;
            const int position = positions[column];
            for (int head = 0; head < kHeads; ++head) {
                const std::int64_t base =
                    static_cast<std::int64_t>(kHeadDim) *
                    ((position & (capacity - 1)) +
                     static_cast<std::int64_t>(capacity) *
                         (head + static_cast<std::int64_t>(kHeads) * slots[b]));
                std::array<double, kHeadDim> key_raw{}, value_raw{};
                for (int j = 0; j < kHeadDim; ++j) {
                    const int row = head * kHeadDim + j;
                    for (int r = 0; r < kHidden; ++r) {
                        const double x = context[column * kHidden + r];
                        key_raw[j] += static_cast<double>(key_decoded[static_cast<std::size_t>(row) * kHidden + r]) * x;
                        value_raw[j] += static_cast<double>(value_decoded[static_cast<std::size_t>(row) * kHidden + r]) * x;
                    }
                }
                double squares = 0.0;
                for (int j = 0; j < kHeadDim; ++j) { squares += key_raw[j] * key_raw[j]; }
                const double inv = 1.0 / std::sqrt(squares / kHeadDim + 1e-6);
                std::array<double, kHeadDim> rotated{};
                for (int j = 0; j < kHeadDim / 2; ++j) {
                    const double frequency = std::pow(static_cast<double>(theta), -static_cast<double>(j) / 64.0);
                    const double angle = position * frequency;
                    const double cosine = std::cos(angle), sine = std::sin(angle);
                    const double x0 = key_raw[j] * inv * key_norm[j];
                    const double y0 = key_raw[j + kHeadDim / 2] * inv * key_norm[j + kHeadDim / 2];
                    rotated[j] = x0 * cosine - y0 * sine;
                    rotated[j + kHeadDim / 2] = y0 * cosine + x0 * sine;
                }
                for (int j = 0; j < kHeadDim; ++j) {
                    expected_k[static_cast<std::size_t>(base + j)] = f32_to_bf16(static_cast<float>(rotated[j]));
                    expected_v[static_cast<std::size_t>(base + j)] = f32_to_bf16(static_cast<float>(value_raw[j]));
                }
            }
        }
    }

    std::vector<double> reference_k(cache_elements), reference_v(cache_elements);
    for (std::size_t i = 0; i < cache_elements; ++i) {
        reference_k[i] = bf16_to_f32(expected_k[i]);
        reference_v[i] = bf16_to_f32(expected_v[i]);
    }
    int failures = 0;
    for (int layer = 0; layer < kContextKVMaterializeLayers; ++layer) {
        failures += verify_reduction("context_kv_materialize K",
                                     from_device_bf16(cache_k[layer].p, cache_elements),
                                     reference_k, kBf16Criterion);
        failures += verify_reduction("context_kv_materialize V",
                                     from_device_bf16(cache_v[layer].p, cache_elements),
                                     reference_v, kBf16Criterion);
    }
    if (failures != 0) {
        std::cerr << "context_kv_materialize failures=" << failures << '\n';
        return 1;
    }
    std::cout << "context_kv_materialize: PASS\n";
    return 0;
}
