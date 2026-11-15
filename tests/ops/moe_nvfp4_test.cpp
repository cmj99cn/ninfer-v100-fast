#include "core/tensor.h"
#include "ninfer/ops/moe_nvfp4.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops;

namespace {

constexpr std::int32_t kExperts       = 512;
constexpr std::int32_t kHidden        = 2560;
constexpr std::int32_t kIntermediate  = 640;
constexpr std::int32_t kGateUpRows    = 2 * kIntermediate;
constexpr std::int32_t kDownRows      = kHidden;
constexpr std::int32_t kParentRows    = 1793;
constexpr std::int32_t kTopK          = 10;
constexpr std::int32_t kCompared      = 4;
constexpr std::size_t  kTokens        = 9;

constexpr std::int32_t kTieLow  = 100;
constexpr std::int32_t kTieHigh = 200;

const std::vector<std::int32_t> kViewExperts{7, 129, 271, 500};

constexpr PointwiseCriterion kRouteCoefficientCriterion{4.0e-6, 1.5e-5};
constexpr double kCoefficientSumTolerance =
    kRouteCoefficientCriterion.absolute * kTopK + kRouteCoefficientCriterion.relative;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kDestinationCriterion{kBf16UnitRoundoff, kBf16UnitRoundoff,
                                                   2.0 * kBf16UnitRoundoff};
constexpr double kF16UnitRoundoff = 1.0 / 1024.0;
constexpr ReductionCriterion kF16DestinationCriterion{kF16UnitRoundoff, kF16UnitRoundoff,
                                                      2.0 * kF16UnitRoundoff};

std::size_t align_up_256(std::size_t value) { return ((value + 255U) / 256U) * 256U; }

std::vector<double> read_f32(const void* device, std::size_t count) {
    std::vector<double> values;
    for (const float value : from_device<float>(device, count)) {
        values.push_back(static_cast<double>(value));
    }
    return values;
}

struct BatchedFixture {
    std::int32_t rows    = 0;
    std::int32_t columns = 0;
    std::size_t code_plane_bytes   = 0;
    std::size_t scale_plane_offset = 0;
    std::size_t scale_plane_bytes  = 0;
    std::size_t divisor_offset     = 0;
    std::vector<std::uint8_t> payload;
    std::vector<float> divisors;
    std::vector<bool> active;

    BatchedNvfp4Weight view(const void* device_payload) const {
        BatchedNvfp4Weight weight;
        weight.payload        = device_payload;
        weight.payload_bytes  = payload.size();
        weight.matrices       = kExperts;
        weight.rows           = rows;
        weight.columns        = columns;
        weight.codes          = device_payload;
        weight.scales         = static_cast<const std::uint8_t*>(device_payload) + scale_plane_offset;
        weight.divisors = reinterpret_cast<const float*>(
            static_cast<const std::uint8_t*>(device_payload) + divisor_offset);
        return weight;
    }
};

constexpr std::array<std::int32_t, 7> kCheckedRows{0, 1, 63, 64, 127, 128, 255};
constexpr std::array<std::int32_t, 8> kCheckedGroups{0, 1, 2, 3, 4, 5, 6, 7};

const std::array<std::uint8_t, 253>& scale_words() {
    static const std::array<std::uint8_t, 253> words = [] {
        std::array<std::uint8_t, 253> values{};
        std::size_t index = 0;
        values[index++] = 0x00U;
        for (std::uint32_t word = 0x01U; word < 0x7fU; ++word) {
            values[index++] = static_cast<std::uint8_t>(word);
        }
        for (std::uint32_t word = 0x81U; word < 0xffU; ++word) {
            values[index++] = static_cast<std::uint8_t>(word);
        }
        return values;
    }();
    return words;
}

std::size_t checked_rank(std::int32_t matrix, std::int32_t row) {
    const auto matrix_it = std::find(kViewExperts.begin(), kViewExperts.end(), matrix);
    const auto row_it = std::find(kCheckedRows.begin(), kCheckedRows.end(), row);
    if (matrix_it == kViewExperts.end() || row_it == kCheckedRows.end()) {
        return kViewExperts.size() * kCheckedRows.size();
    }
    return static_cast<std::size_t>(matrix_it - kViewExperts.begin()) * kCheckedRows.size() +
           static_cast<std::size_t>(row_it - kCheckedRows.begin());
}

std::size_t scale_word_index(std::int32_t matrix, std::int32_t row, std::int32_t group) {
    const std::size_t rank = checked_rank(matrix, row);
    if (rank < kViewExperts.size() * kCheckedRows.size() &&
        group < static_cast<std::int32_t>(kCheckedGroups.size())) {
        return rank * kCheckedGroups.size() + static_cast<std::size_t>(group);
    }
    return static_cast<std::size_t>(
        (static_cast<std::uint64_t>(matrix) * 0x9e3779b1U +
         static_cast<std::uint64_t>(row) * 0x85ebca6bU +
         static_cast<std::uint64_t>(group) * 0xc2b2ae35U) %
        scale_words().size());
}

int verify_scale_word_domain() {
    std::array<bool, 253> used{};
    int failures = 0;
    for (const std::int32_t matrix : kViewExperts) {
        for (const std::int32_t row : kCheckedRows) {
            for (const std::int32_t group : kCheckedGroups) {
                const std::size_t index = scale_word_index(matrix, row, group);
                if (used[index]) {
                    std::cerr << "scale fixture: colliding word for matrix " << matrix << " row "
                              << row << " group " << group << "\n";
                    ++failures;
                }
                used[index] = true;
                if (group >= 4 &&
                    scale_word_index(matrix, row, group) ==
                        scale_word_index(matrix, row, group % 4)) {
                    std::cerr << "scale fixture: missing-tile offset is not detectable for matrix "
                              << matrix << " row " << row << " group " << group << "\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

BatchedFixture make_batched(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                            const std::vector<std::int32_t>& active_experts,
                            float divisor_scale = 1.0F) {
    BatchedFixture fixture;
    fixture.rows      = rows;
    fixture.columns   = columns;
    fixture.code_plane_bytes   = static_cast<std::size_t>(kExperts) * rows * columns / 2U;
    fixture.scale_plane_offset = align_up_256(fixture.code_plane_bytes);
    fixture.scale_plane_bytes  = static_cast<std::size_t>(kExperts) * rows * columns / 16U;
    fixture.divisor_offset     = fixture.scale_plane_offset + fixture.scale_plane_bytes;
    fixture.payload.assign(fixture.divisor_offset + 4U * kExperts, 0);
    fixture.divisors.assign(kExperts, 0.0F);
    fixture.active.assign(kExperts, false);
    for (const std::int32_t expert : active_experts) { fixture.active[expert] = true; }

    for (std::int32_t matrix = 0; matrix < kExperts; ++matrix) {
        const float divisor = (1.0F + static_cast<float>(matrix % 7) * 0.125F) * divisor_scale;
        fixture.divisors[matrix] = divisor;
        quantized_weight::detail::store_u32_le(fixture.payload,
                                    fixture.divisor_offset + static_cast<std::size_t>(matrix) * 4U,
                                    quantized_weight::detail::float_bits(divisor));
        if (!fixture.active[matrix]) { continue; }

        for (std::int32_t row = 0; row < rows; ++row) {
            for (std::int32_t column = 0; column < columns; column += 2) {
                const auto code = [&](std::int32_t logical_column) {
                    return static_cast<std::uint8_t>(
                        (static_cast<std::uint64_t>(matrix) * 11U +
                         static_cast<std::uint64_t>(row) * 13U +
                         static_cast<std::uint64_t>(logical_column) * 7U + seed) &
                        0x0fU);
                };
                fixture.payload[static_cast<std::size_t>(matrix) * fixture.code_plane_bytes /
                                    kExperts +
                                static_cast<std::size_t>(row) * columns / 2U +
                                static_cast<std::size_t>(column / 2)] =
                    static_cast<std::uint8_t>(code(column) | (code(column + 1) << 4));
            }
        }

        const std::int32_t groups_per_row = columns / 16;
        const std::int32_t k_tiles        = columns / 64;
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int32_t row_tile  = row / 128;
            const std::int32_t row_inner = row % 128;
            for (std::int32_t group = 0; group < groups_per_row; ++group) {
                const std::size_t stored_offset =
                    fixture.scale_plane_offset +
                    static_cast<std::size_t>(matrix) * fixture.scale_plane_bytes / kExperts +
                    static_cast<std::size_t>(row_tile * k_tiles + group / 4) * 512U +
                    static_cast<std::size_t>(row_inner % 32) * 16U +
                    static_cast<std::size_t>(row_inner / 32) * 4U +
                    static_cast<std::size_t>(group % 4);
                fixture.payload[stored_offset] =
                    scale_words()[scale_word_index(matrix, row, group)];
            }
        }
    }
    return fixture;
}

double exact_decode(const BatchedFixture& fixture, std::int32_t matrix, std::int32_t row,
                    std::int32_t column) {
    const std::size_t matrix_code_base =
        static_cast<std::size_t>(matrix) * fixture.code_plane_bytes / kExperts;
    const std::uint8_t packed_codes =
        fixture.payload[matrix_code_base + static_cast<std::size_t>(row) * fixture.columns / 2U +
                        static_cast<std::size_t>(column / 2)];
    const std::uint8_t code = (column & 1) == 0 ? (packed_codes & 0x0fU) : (packed_codes >> 4);

    const std::int32_t group      = column / 16;
    const std::int32_t row_global = matrix * fixture.rows + row;
    const std::size_t scale_offset =
        fixture.scale_plane_offset +
        (static_cast<std::size_t>(row_global / 128) * (fixture.columns / 64) +
         static_cast<std::size_t>(group / 4)) * 512U +
        static_cast<std::size_t>(row_global % 32) * 16U +
        static_cast<std::size_t>((row_global % 128) / 32) * 4U + static_cast<std::size_t>(group % 4);
    const double scale = quantized_weight::detail::decode_e4m3fn(fixture.payload[scale_offset]);
    const double decoded = quantized_weight::detail::decode_e2m1(code) * scale /
                           static_cast<double>(fixture.divisors[matrix]);
    return static_cast<double>(static_cast<float>(decoded));
}

double decode_through_view(const BatchedNvfp4Weight& weight, std::int32_t matrix,
                           std::int32_t row, std::int32_t column) {
    const std::uint8_t* codes = weight.codes_for(matrix);
    const std::uint8_t* scales = weight.scales_for(matrix);
    const std::uint8_t packed_codes =
        codes[static_cast<std::size_t>(row) * weight.columns / 2U + static_cast<std::size_t>(column / 2)];
    const std::uint8_t code = (column & 1) == 0 ? (packed_codes & 0x0fU) : (packed_codes >> 4);
    const double scale =
        quantized_weight::detail::decode_e4m3fn(scales[weight.scale_offset(row, column / 16)]);
    const double decoded =
        quantized_weight::detail::decode_e2m1(code) * scale / static_cast<double>(weight.divisor_for(matrix));
    return static_cast<double>(static_cast<float>(decoded));
}

int verify_view_planes(const BatchedFixture& fixture) {
    const BatchedNvfp4Weight weight = fixture.view(fixture.payload.data());
    int failures = 0;
    for (const std::int32_t matrix : kViewExperts) {
        const std::size_t expected_codes =
            fixture.code_plane_bytes / kExperts * static_cast<std::size_t>(matrix);
        const std::size_t expected_scales =
            fixture.scale_plane_offset +
            fixture.scale_plane_bytes / kExperts * static_cast<std::size_t>(matrix);

        if (static_cast<const std::uint8_t*>(weight.codes) != fixture.payload.data()) {
            std::cerr << "batched view: code plane base must be the payload base\n";
            ++failures;
        }
        if (static_cast<const std::uint8_t*>(weight.scales) !=
            fixture.payload.data() + fixture.scale_plane_offset) {
            std::cerr << "batched view: scale plane base must be the payload scale base\n";
            ++failures;
        }
        if (weight.codes_for(matrix) != fixture.payload.data() + expected_codes) {
            std::cerr << "batched view: code plane base for matrix " << matrix << " is wrong\n";
            ++failures;
        }
        if (weight.scales_for(matrix) != fixture.payload.data() + expected_scales) {
            std::cerr << "batched view: scale plane base for matrix " << matrix << " is wrong\n";
            ++failures;
        }
        if (weight.divisor_for(matrix) != fixture.divisors[matrix]) {
            std::cerr << "batched view: divisor for matrix " << matrix << " is wrong\n";
            ++failures;
        }
        for (const std::int32_t row : {0, 1, 63, 64, 127, 128, 255}) {
            for (const std::int32_t group : {0, 1, 3, 4, 7}) {
                if (group >= fixture.columns / 16) { continue; }
                const std::size_t row_global = static_cast<std::size_t>(matrix) * fixture.rows + row;
                const std::size_t absolute =
                    fixture.scale_plane_offset +
                    (static_cast<std::size_t>(row_global / 128) * (fixture.columns / 64) +
                     static_cast<std::size_t>(group / 4)) * 512U +
                    static_cast<std::size_t>(row_global % 32) * 16U +
                    static_cast<std::size_t>((row_global % 128) / 32) * 4U +
                    static_cast<std::size_t>(group % 4);
                if (weight.scales_for(matrix) + weight.scale_offset(row, group) !=
                    fixture.payload.data() + absolute) {
                    std::cerr << "batched view: swizzled scale position is wrong for matrix " << matrix
                              << " row " << row << " group " << group << "\n";
                    ++failures;
                }
            }
        }
        for (const std::int32_t column : {0, 3, 15, 16, 31, 63, 64, 127}) {
            if (column >= fixture.columns) { continue; }
            for (const std::int32_t row : {0, 127, 128, 255}) {
                if (row >= fixture.rows) { continue; }
                if (decode_through_view(weight, matrix, row, column) !=
                    exact_decode(fixture, matrix, row, column)) {
                    std::cerr << "batched view: decode mismatch for matrix " << matrix << " row "
                              << row << " column " << column << "\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

struct RouteOracle {
    std::vector<double> logits;
    std::vector<std::int32_t> ranked;
    std::array<std::int32_t, kTopK> ids{};
    std::array<double, kTopK> coefficients{};
    double shared_gate = 0.0;
};

std::vector<std::int32_t> ranked_experts(const std::vector<double>& logits) {
    std::vector<std::int32_t> ranked(kExperts);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::sort(ranked.begin(), ranked.end(), [&](std::int32_t a, std::int32_t b) {
        const double left  = logits[static_cast<std::size_t>(a)];
        const double right = logits[static_cast<std::size_t>(b)];
        if (left != right) { return left > right; }
        return a < b;
    });
    return ranked;
}

RouteOracle route_oracle(const direct_bf16_weight::HostWeight& parent,
                         const std::vector<float>& activation) {
    RouteOracle oracle;
    oracle.logits.assign(kParentRows, 0.0);
    for (std::int32_t row = 0; row < kParentRows; ++row) {
        double score = 0.0;
        for (std::int32_t column = 0; column < kHidden; ++column) {
            score += static_cast<double>(bf16_to_f32(parent.bits[static_cast<std::size_t>(row) * kHidden + column])) *
                     static_cast<double>(activation[static_cast<std::size_t>(column)]);
        }
        oracle.logits[static_cast<std::size_t>(row)] = score;
    }

    oracle.ranked = ranked_experts(oracle.logits);
    for (std::int32_t rank = 0; rank < kTopK; ++rank) {
        oracle.ids[rank] = oracle.ranked[static_cast<std::size_t>(rank)];
    }

    double denominator = 0.0;
    std::array<double, kTopK> exponentials{};
    for (std::int32_t rank = 0; rank < kTopK; ++rank) {
        exponentials[rank] = std::exp(oracle.logits[static_cast<std::size_t>(oracle.ids[rank])]);
        denominator += exponentials[rank];
    }
    for (std::int32_t rank = 0; rank < kTopK; ++rank) {
        oracle.coefficients[rank] = exponentials[rank] / denominator;
    }
    oracle.shared_gate = 1.0 / (1.0 + std::exp(-oracle.logits[kExperts]));
    return oracle;
}

std::int32_t tie_boundary_channel(const direct_bf16_weight::HostWeight& parent) {
    for (std::int32_t column = 0; column < kHidden; ++column) {
        std::vector<double> logits(kExperts);
        for (std::int32_t row = 0; row < kExperts; ++row) {
            logits[static_cast<std::size_t>(row)] =
                static_cast<double>(bf16_to_f32(parent.bits[static_cast<std::size_t>(row) * kHidden + column]));
        }
        const std::vector<std::int32_t> ranked = ranked_experts(logits);
        if (ranked[kTopK - 1] == kTieLow && ranked[kTopK] == kTieHigh) { return column; }
    }
    return -1;
}

std::vector<double> expert_contribution(const BatchedFixture& gate_up,
                                        const BatchedFixture& down,
                                        std::int32_t expert, const std::vector<float>& activation,
                                        double coefficient) {
    std::vector<double> gate(kGateUpRows, 0.0);
    std::vector<double> up(kGateUpRows, 0.0);
    for (std::int32_t row = 0; row < kGateUpRows; ++row) {
        for (std::int32_t column = 0; column < kHidden; ++column) {
            const double weight = exact_decode(gate_up, expert, row, column);
            const double value  = static_cast<double>(activation[static_cast<std::size_t>(column)]);
            if (row < kIntermediate) {
                gate[row] += weight * value;
            } else {
                up[row - kIntermediate] += weight * value;
            }
        }
    }
    std::vector<double> activation_values(kIntermediate);
    for (std::int32_t i = 0; i < kIntermediate; ++i) {
        const double s = 1.0 / (1.0 + std::exp(-gate[i]));
        activation_values[i] = gate[i] * s * up[i];
    }
    std::vector<double> contribution(kHidden, 0.0);
    for (std::int32_t row = 0; row < kHidden; ++row) {
        double value = 0.0;
        for (std::int32_t column = 0; column < kIntermediate; ++column) {
            value += exact_decode(down, expert, row, column) * activation_values[static_cast<std::size_t>(column)];
        }
        contribution[static_cast<std::size_t>(row)] = coefficient * value;
    }
    return contribution;
}

std::vector<double> shared_contribution(const direct_bf16_weight::HostWeight& parent,
                                        const direct_bf16_weight::HostWeight& shared_down,
                                        const std::vector<float>& activation, double gate) {
    std::vector<double> gate_row(kIntermediate, 0.0);
    std::vector<double> up_row(kIntermediate, 0.0);
    for (std::int32_t row = 0; row < kIntermediate; ++row) {
        for (std::int32_t column = 0; column < kHidden; ++column) {
            const double weight = static_cast<double>(
                bf16_to_f32(parent.bits[static_cast<std::size_t>(kExperts + 1 + row) * kHidden + column]));
            const double value = static_cast<double>(activation[static_cast<std::size_t>(column)]);
            gate_row[row] += weight * value;
            const double up_weight = static_cast<double>(
                bf16_to_f32(parent.bits[static_cast<std::size_t>(kExperts + 1 + kIntermediate + row) * kHidden + column]));
            up_row[row] += up_weight * value;
        }
    }
    std::vector<double> activation_values(kIntermediate);
    for (std::int32_t i = 0; i < kIntermediate; ++i) {
        const double s = 1.0 / (1.0 + std::exp(-gate_row[i]));
        activation_values[i] = gate_row[i] * s * up_row[i];
    }
    std::vector<double> contribution(kHidden, 0.0);
    for (std::int32_t row = 0; row < kHidden; ++row) {
        double value = 0.0;
        for (std::int32_t column = 0; column < kIntermediate; ++column) {
            value += static_cast<double>(
                bf16_to_f32(shared_down.bits[static_cast<std::size_t>(row) * kIntermediate + column])) *
                     activation_values[static_cast<std::size_t>(column)];
        }
        contribution[static_cast<std::size_t>(row)] = gate * value;
    }
    return contribution;
}

struct LocalBank {
    std::vector<std::uint8_t> payload;
    std::vector<float> divisors;
    std::int32_t rows    = 0;
    std::int32_t columns = 0;
    std::int32_t begin   = 0;
    std::int32_t count   = 0;
    std::size_t scale_offset = 0;

    BatchedNvfp4Weight view(const void* device_payload) const {
        BatchedNvfp4Weight weight;
        weight.payload       = device_payload;
        weight.payload_bytes = payload.size();
        weight.matrices      = count;
        weight.expert_begin  = begin;
        weight.rows          = rows;
        weight.columns       = columns;
        weight.codes         = device_payload;
        weight.scales        = static_cast<const std::uint8_t*>(device_payload) + scale_offset;
        weight.divisors      = reinterpret_cast<const float*>(
            static_cast<const std::uint8_t*>(device_payload) + scale_offset +
            static_cast<std::size_t>(count) * rows * columns / 16U);
        return weight;
    }
};

LocalBank local_bank(const BatchedFixture& full, std::int32_t begin, std::int32_t count) {
    LocalBank bank;
    bank.rows    = full.rows;
    bank.columns = full.columns;
    bank.begin   = begin;
    bank.count   = count;
    const std::size_t matrix_code  = static_cast<std::size_t>(full.rows) * full.columns / 2U;
    const std::size_t matrix_scale = static_cast<std::size_t>(full.rows) * full.columns / 16U;
    const std::size_t code_bytes   = static_cast<std::size_t>(count) * matrix_code;
    bank.scale_offset              = align_up_256(code_bytes);
    const std::size_t scale_bytes  = static_cast<std::size_t>(count) * matrix_scale;
    const std::size_t divisor_offset = bank.scale_offset + scale_bytes;
    bank.payload.assign(divisor_offset + 4U * static_cast<std::size_t>(count), 0);
    bank.divisors.resize(static_cast<std::size_t>(count));
    for (std::int32_t local = 0; local < count; ++local) {
        const std::int32_t global = begin + local;
        const auto code_source = static_cast<std::size_t>(global) * matrix_code;
        std::copy_n(full.payload.begin() + static_cast<std::ptrdiff_t>(code_source), matrix_code,
                    bank.payload.begin() + static_cast<std::ptrdiff_t>(local * matrix_code));
        const auto scale_source = full.scale_plane_offset + static_cast<std::size_t>(global) * matrix_scale;
        std::copy_n(full.payload.begin() + static_cast<std::ptrdiff_t>(scale_source), matrix_scale,
                    bank.payload.begin() +
                        static_cast<std::ptrdiff_t>(bank.scale_offset + local * matrix_scale));
        bank.divisors[static_cast<std::size_t>(local)] = full.divisors[static_cast<std::size_t>(global)];
        quantized_weight::detail::store_u32_le(
            bank.payload, divisor_offset + static_cast<std::size_t>(local) * 4U,
            quantized_weight::detail::float_bits(bank.divisors[static_cast<std::size_t>(local)]));
    }
    return bank;
}

struct Case {
    direct_bf16_weight::HostWeight parent;
    direct_bf16_weight::HostWeight shared_down;
    BatchedFixture gate_up;
    BatchedFixture down;
    std::vector<std::vector<float>> activation;
    std::vector<std::vector<std::int32_t>> compared;
    std::vector<RouteOracle> oracle;
};

int run_route_case(const Case& inputs, std::size_t token, const std::string& label) {
    const RouteOracle& oracle = inputs.oracle[token];

    DeviceBuffer device_parent      = to_device(inputs.parent.bits);
    DeviceBuffer device_shared_down = to_device(inputs.shared_down.bits);
    DeviceBuffer device_gate_up     = to_device(inputs.gate_up.payload);
    DeviceBuffer device_down        = to_device(inputs.down.payload);

    GuardedDeviceBuffer ids_buffer(kTopK * sizeof(std::int32_t));
    GuardedDeviceBuffer coefficients_buffer(kTopK * sizeof(float));
    GuardedDeviceBuffer shared_gate_buffer(sizeof(float));

    DeviceBuffer device_x = to_device_bf16(inputs.activation[token]);
    Tensor x{device_x.p, DType::BF16, {kHidden, 1}};
    Tensor ids{ids_buffer.data(), DType::I32, {kTopK, 1}};
    Tensor coefficients{coefficients_buffer.data(), DType::FP32, {kTopK, 1}};
    Tensor shared_gate{shared_gate_buffer.data(), DType::FP32, {1, 1}};

    MoeNvfp4Weights weights;
    weights.router_shared  = inputs.parent.device_weight(device_parent.p);
    weights.shared_down    = inputs.shared_down.device_weight(device_shared_down.p);
    weights.routed_gate_up = inputs.gate_up.view(device_gate_up.p);
    weights.routed_down    = inputs.down.view(device_down.p);

    ops::moe_nvfp4_route(x, weights, ids, coefficients, shared_gate, 0);
    cuda_synchronize();

    GuardedDeviceBuffer split_ids_buffer(kTopK * sizeof(std::int32_t));
    GuardedDeviceBuffer split_coefficients_buffer(kTopK * sizeof(float));
    GuardedDeviceBuffer split_shared_gate_buffer(sizeof(float));
    DeviceBuffer split_logits((kExperts + 1) * sizeof(float));
    Tensor split_ids{split_ids_buffer.data(), DType::I32, {kTopK, 1}};
    Tensor split_coefficients{split_coefficients_buffer.data(), DType::FP32, {kTopK, 1}};
    Tensor split_shared_gate{split_shared_gate_buffer.data(), DType::FP32, {1, 1}};
    ops::moe_nvfp4_route(x, weights, split_ids, split_coefficients, split_shared_gate, 0,
                         static_cast<float*>(split_logits.p));
    cuda_synchronize();

    int failures = 0;
    if (from_device<std::uint32_t>(ids_buffer.data(), kTopK) !=
        from_device<std::uint32_t>(split_ids_buffer.data(), kTopK)) {
        std::cerr << label << ": the row-split route ids are not bitwise identical\n";
        ++failures;
    }
    if (from_device<std::uint32_t>(coefficients_buffer.data(), kTopK) !=
        from_device<std::uint32_t>(split_coefficients_buffer.data(), kTopK)) {
        std::cerr << label << ": the row-split route coefficients are not bitwise identical\n";
        ++failures;
    }
    if (from_device<std::uint32_t>(shared_gate_buffer.data(), 1) !=
        from_device<std::uint32_t>(split_shared_gate_buffer.data(), 1)) {
        std::cerr << label << ": the row-split shared gate is not bitwise identical\n";
        ++failures;
    }
    failures += split_ids_buffer.verify_guards(label + " split ids guard");
    failures += split_coefficients_buffer.verify_guards(label + " split coefficients guard");
    failures += split_shared_gate_buffer.verify_guards(label + " split shared gate guard");

    const std::vector<std::int32_t> got_ids = from_device<std::int32_t>(ids_buffer.data(), kTopK);
    for (std::int32_t rank = 0; rank < kTopK; ++rank) {
        if (got_ids[static_cast<std::size_t>(rank)] != oracle.ids[static_cast<std::size_t>(rank)]) {
            std::cerr << label << ": top-10 selection mismatch at rank " << rank << " got "
                      << got_ids[static_cast<std::size_t>(rank)] << " expected "
                      << oracle.ids[static_cast<std::size_t>(rank)] << "\n";
            ++failures;
        }
    }

    const std::vector<double> got_coefficients = read_f32(coefficients_buffer.data(), kTopK);
    std::vector<double> expected_coefficients(kTopK);
    for (std::int32_t rank = 0; rank < kTopK; ++rank) {
        expected_coefficients[static_cast<std::size_t>(rank)] = oracle.coefficients[static_cast<std::size_t>(rank)];
    }
    failures += verify_pointwise(label + " coefficients", got_coefficients, expected_coefficients,
                                 kRouteCoefficientCriterion);

    double coefficient_sum = 0.0;
    for (const double value : got_coefficients) coefficient_sum += value;
    if (std::abs(coefficient_sum - 1.0) > kCoefficientSumTolerance) {
        std::cerr << label << ": normalized coefficients sum to " << coefficient_sum << "\n";
        ++failures;
    }

    const std::vector<double> got_gate = read_f32(shared_gate_buffer.data(), 1);
    failures += verify_pointwise(label + " shared gate", got_gate, std::vector<double>{oracle.shared_gate},
                                 kRouteCoefficientCriterion);

    failures += ids_buffer.verify_guards(label + " ids guard");
    failures += coefficients_buffer.verify_guards(label + " coefficients guard");
    failures += shared_gate_buffer.verify_guards(label + " shared gate guard");
    return failures;
}

struct SlotBank {
    std::vector<std::uint8_t> payload;
    std::vector<float> divisors;
    std::vector<std::int32_t> lookup;
    std::int32_t slots = 0;
};

SlotBank make_slot_bank(const BatchedFixture& gate_up, const BatchedFixture& down,
                        const std::vector<std::int32_t>& selected) {
    std::vector<std::int32_t> experts = selected;
    std::sort(experts.begin(), experts.end());
    experts.erase(std::unique(experts.begin(), experts.end()), experts.end());

    constexpr std::size_t kHalfCode  = static_cast<std::size_t>(kIntermediate) * kHidden / 2U;
    constexpr std::size_t kHalfScale = static_cast<std::size_t>(kIntermediate) * kHidden / 16U;
    constexpr std::size_t kSlotBytes = 3U * (kHalfCode + kHalfScale);
    constexpr std::size_t kGateScales = kHalfCode;
    constexpr std::size_t kUpCodes    = kHalfCode + kHalfScale;
    constexpr std::size_t kUpScales   = 2U * kHalfCode + kHalfScale;
    constexpr std::size_t kDownCodes  = 2U * (kHalfCode + kHalfScale);
    const std::size_t gate_up_code  = static_cast<std::size_t>(kGateUpRows) * kHidden / 2U;
    const std::size_t gate_up_scale = static_cast<std::size_t>(kGateUpRows) * kHidden / 16U;
    const std::size_t down_code     = static_cast<std::size_t>(kDownRows) * kIntermediate / 2U;
    const std::size_t down_scale    = static_cast<std::size_t>(kDownRows) * kIntermediate / 16U;

    SlotBank bank;
    bank.slots = static_cast<std::int32_t>(experts.size());
    bank.payload.assign(static_cast<std::size_t>(bank.slots) * kSlotBytes, 0);
    bank.divisors.assign(static_cast<std::size_t>(bank.slots) * 3, 0.0F);
    bank.lookup.assign(static_cast<std::size_t>(kExperts), -1);

    for (std::size_t slot = 0; slot < experts.size(); ++slot) {
        const auto expert = static_cast<std::size_t>(experts[slot]);
        std::uint8_t* base = bank.payload.data() + slot * kSlotBytes;
        const std::uint8_t* source = gate_up.payload.data();
        std::copy_n(source + expert * gate_up_code, kHalfCode, base);
        std::copy_n(source + gate_up.scale_plane_offset + expert * gate_up_scale, kHalfScale,
                    base + kGateScales);
        std::copy_n(source + expert * gate_up_code + kHalfCode, kHalfCode, base + kUpCodes);
        std::copy_n(source + gate_up.scale_plane_offset + expert * gate_up_scale + kHalfScale,
                    kHalfScale, base + kUpScales);
        const std::uint8_t* down_source = down.payload.data();
        std::copy_n(down_source + expert * down_code, down_code, base + kDownCodes);
        std::copy_n(down_source + down.scale_plane_offset + expert * down_scale, down_scale,
                    base + kDownCodes + down_code);

        const float up_divisor = gate_up.divisors[expert];
        bank.divisors[3U * slot + 0U] = up_divisor;
        bank.divisors[3U * slot + 1U] = up_divisor;
        bank.divisors[3U * slot + 2U] = down.divisors[expert];
        bank.lookup[expert] = static_cast<std::int32_t>(slot);
    }
    return bank;
}

void build_token_words(const std::vector<float>& activation_values, DType dtype,
                       std::vector<std::uint16_t>& activation_words,
                       std::vector<std::uint16_t>& residual_words) {
    const bool fp16 = dtype == DType::FP16;
    activation_words.resize(kHidden);
    residual_words.resize(kHidden);
    for (std::int32_t row = 0; row < kHidden; ++row) {
        const float value = activation_values[static_cast<std::size_t>(row)];
        activation_words[static_cast<std::size_t>(row)] = fp16 ? f32_to_f16(value) : f32_to_bf16(value);
        const std::uint32_t hash = static_cast<std::uint32_t>(row) * 0x9e3779b9U ^ 0x85ebca6bU;
        const float residual = static_cast<float>(static_cast<int>((hash >> 9) & 0xffU) - 128) *
                               (1.0F / 2048.0F);
        residual_words[static_cast<std::size_t>(row)] = fp16 ? f32_to_f16(residual) : f32_to_bf16(residual);
    }
}

std::size_t changed_rows(const std::vector<std::uint16_t>& got,
                         const std::vector<std::uint16_t>& residual) {
    std::size_t changed = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != residual[i]) { ++changed; }
    }
    return changed;
}

struct ProbeReference {
    std::vector<std::int32_t> active;
    std::vector<std::int32_t> miss;
};

ProbeReference reference_probe(const std::vector<std::int32_t>& ids,
                               const std::vector<std::int32_t>& lookup,
                               std::int32_t expert_begin, std::int32_t expert_end) {
    ProbeReference reference;
    for (const std::int32_t id : ids) {
        if (id < expert_begin || id >= expert_end) { continue; }
        if (std::find(reference.active.begin(), reference.active.end(), id) !=
            reference.active.end()) {
            continue;
        }
        reference.active.push_back(id);
        if (lookup[static_cast<std::size_t>(id)] < 0) { reference.miss.push_back(id); }
    }
    return reference;
}

struct ProbeBuffers {
    DeviceBuffer payload{3ULL * (static_cast<std::uint64_t>(kIntermediate) * kHidden / 2U +
                                 static_cast<std::uint64_t>(kIntermediate) * kHidden / 16U)};
    DeviceBuffer divisors{3U * sizeof(float)};
    DeviceBuffer lookup{kExperts * sizeof(std::int32_t)};
    DeviceBuffer ids{kTopK * sizeof(std::int32_t)};
    DeviceBuffer coefficients{kTopK * sizeof(float)};
    DeviceBuffer shared_gate{sizeof(float)};
    DeviceBuffer x{static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t)};
    DeviceBuffer activation{static_cast<std::size_t>(kTopK) * kIntermediate * sizeof(float)};
    std::int32_t* host = nullptr;
    cudaEvent_t probe_done = nullptr;

    ProbeBuffers() {
        cuda_check(cudaMemset(payload.p, 0, payload.bytes), "payload memset");
        const float divisors_host[3] = {1.0F, 1.0F, 1.0F};
        cuda_check(cudaMemcpy(divisors.p, divisors_host, sizeof(divisors_host),
                              cudaMemcpyHostToDevice),
                   "divisor copy");
        cuda_check(cudaMemset(x.p, 0, x.bytes), "x memset");
        cuda_check(cudaMemset(coefficients.p, 0, coefficients.bytes), "coefficient memset");
        cuda_check(cudaMemset(shared_gate.p, 0, shared_gate.bytes), "gate memset");
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), (2 + 2 * kExperts) * 4,
                                 cudaHostAllocMapped | cudaHostAllocPortable),
                   "cudaHostAlloc probe");
        host[0] = 0;
        host[1] = 0;
        cuda_check(cudaEventCreateWithFlags(&probe_done, cudaEventDisableTiming),
                   "cudaEventCreate probe");
    }

    ~ProbeBuffers() {
        if (probe_done != nullptr) { (void)cudaEventDestroy(probe_done); }
        if (host != nullptr) { (void)cudaFreeHost(host); }
    }

    [[nodiscard]] std::int32_t* miss_count() const noexcept { return host; }
    [[nodiscard]] std::int32_t* active_count() const noexcept { return host + 1; }
    [[nodiscard]] std::int32_t* miss_ids() const noexcept { return host + 2; }
    [[nodiscard]] std::int32_t* active_ids() const noexcept { return host + 2 + kExperts; }

    [[nodiscard]] std::int32_t* device_base() const {
        void* device = nullptr;
        cuda_check(cudaHostGetDevicePointer(&device, host, 0), "cudaHostGetDevicePointer");
        return static_cast<std::int32_t*>(device);
    }
};

int run_probe_case(const char* label, const std::vector<std::int32_t>& ids,
                   const std::vector<std::int32_t>& lookup, std::int32_t expert_begin,
                   std::int32_t expert_end, bool repair) {
    const ProbeReference reference = reference_probe(ids, lookup, expert_begin, expert_end);
    ProbeBuffers buffers;
    cuda_check(cudaMemcpy(buffers.ids.p, ids.data(), ids.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice),
               "id copy");
    cuda_check(cudaMemcpy(buffers.lookup.p, lookup.data(), lookup.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice),
               "lookup copy");

    const ops::detail::MoeNvfp4RouteOutputs outputs{
        static_cast<std::int32_t*>(buffers.ids.p),
        static_cast<float*>(buffers.coefficients.p),
        static_cast<float*>(buffers.shared_gate.p)};
    const ops::detail::MoeNvfp4Scratch scratch{static_cast<float*>(buffers.activation.p)};
    const Tensor x{buffers.x.p, DType::BF16, {kHidden, 1}};

    auto run = [&](const ops::detail::MoeNvfp4Probe& probe) {
        ops::detail::moe_nvfp4_cached_gate_up_launch(
            x, static_cast<const std::uint8_t*>(buffers.payload.p), buffers.payload.bytes,
            static_cast<const std::int32_t*>(buffers.lookup.p),
            static_cast<const float*>(buffers.divisors.p), scratch, outputs, &probe, nullptr);
        cuda_check(cudaEventRecord(buffers.probe_done, nullptr), "probe event record");
        cuda_check(cudaEventSynchronize(buffers.probe_done), "probe event sync");
    };

    const std::int32_t* probe_device = buffers.device_base();
    const ops::detail::MoeNvfp4Probe probe{
        const_cast<std::int32_t*>(probe_device + 2 + kExperts),
        const_cast<std::int32_t*>(probe_device + 2),
        const_cast<std::int32_t*>(probe_device + 1),
        const_cast<std::int32_t*>(probe_device),
        expert_begin, expert_end};
    run(probe);

    int failures = 0;
    const auto require = [&](bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << label << ": " << message << "\n";
            ++failures;
        }
    };
    require(*buffers.active_count() == static_cast<std::int32_t>(reference.active.size()),
            "probe active count disagrees with the host dedup");
    require(*buffers.miss_count() == static_cast<std::int32_t>(reference.miss.size()),
            "probe miss count disagrees with the host dedup");
    for (std::size_t i = 0; i < reference.active.size(); ++i) {
        require(buffers.active_ids()[i] == reference.active[i], "probe active order disagrees");
    }
    for (std::size_t i = 0; i < reference.miss.size(); ++i) {
        require(buffers.miss_ids()[i] == reference.miss[i], "probe miss set disagrees");
    }

    if (repair) {
        std::vector<std::int32_t> repaired = lookup;
        for (const std::int32_t id : reference.miss) {
            repaired[static_cast<std::size_t>(id)] = 0;
        }
        cuda_check(cudaMemcpy(buffers.lookup.p, repaired.data(),
                              repaired.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                   "lookup repair");
        run(probe);
        require(*buffers.miss_count() == 0, "repaired probe still reports misses");
        require(*buffers.active_count() == static_cast<std::int32_t>(reference.active.size()),
                "repaired probe changed the active set");
    }
    return failures;
}

int run_probe_cases() {
    std::vector<std::int32_t> lookup(kExperts, -1);
    for (std::int32_t expert = 0; expert < kExperts; ++expert) {
        if (expert % 3 == 0) { lookup[static_cast<std::size_t>(expert)] = 0; }
    }
    int failures = 0;
    failures += run_probe_case("probe mixed",
                               {100, 100, 150, 99, 200, 100, 150, 101, 101, 300}, lookup, 100,
                               200, false);
    failures += run_probe_case("probe all miss", {101, 104, 107, 101, 104, 107, 101, 104, 107, 101},
                               lookup, 100, 200, true);
    failures += run_probe_case("probe all hit", {102, 105, 108, 102, 105, 108, 102, 105, 108, 102},
                               lookup, 100, 200, false);
    failures += run_probe_case("probe out of range", {0, 10, 64, 250, 300, 511, 100, 150, 100, 150},
                               lookup, 100, 200, false);
    bool threw = false;
    try {
        ops::detail::moe_nvfp4_route_probe_launch(nullptr, kExperts + 1, nullptr, 0, 0, nullptr,
                                                  nullptr, nullptr, nullptr, nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "probe: entries above the expert count was accepted\n";
        ++failures;
    }
    return failures;
}

int run_cached_bitwise_case(const BatchedFixture& gate_up, const BatchedFixture& down,
                            const MoeNvfp4Weights& full_weights,
                            const std::vector<float>& activation_values, DType dtype,
                            const std::string& label) {
    std::vector<std::uint16_t> activation_words;
    std::vector<std::uint16_t> residual_words;
    build_token_words(activation_values, dtype, activation_words, residual_words);

    DeviceBuffer device_x = to_device(activation_words);
    GuardedDeviceBuffer ids_buffer(kTopK * sizeof(std::int32_t));
    GuardedDeviceBuffer coefficients_buffer(kTopK * sizeof(float));
    GuardedDeviceBuffer shared_gate_buffer(sizeof(float));
    Tensor x{device_x.p, dtype, {kHidden, 1}};
    Tensor ids{ids_buffer.data(), DType::I32, {kTopK, 1}};
    Tensor coefficients{coefficients_buffer.data(), DType::FP32, {kTopK, 1}};
    Tensor shared_gate{shared_gate_buffer.data(), DType::FP32, {1, 1}};
    ops::moe_nvfp4_route(x, full_weights, ids, coefficients, shared_gate, 0);
    cuda_synchronize();

    const MoeNvfp4RouteView route{ids, coefficients, shared_gate};
    const std::vector<std::int32_t> selected = from_device<std::int32_t>(ids_buffer.data(), kTopK);
    const SlotBank slots = make_slot_bank(gate_up, down, selected);

    DeviceBuffer slot_payload = to_device(slots.payload);
    DeviceBuffer slot_divisors = to_device(slots.divisors);
    DeviceBuffer slot_lookup = to_device(slots.lookup);
    const std::size_t slot_bytes = slots.payload.size() / static_cast<std::size_t>(slots.slots);
    const MoeNvfp4ExpertSlots bank{
        .payload    = static_cast<const std::uint8_t*>(slot_payload.p),
        .slot_bytes = slot_bytes,
        .slot_count = slots.slots,
        .lookup     = static_cast<const std::int32_t*>(slot_lookup.p),
        .divisors   = static_cast<const float*>(slot_divisors.p),
    };
    DeviceBuffer scratch(kTopK * kIntermediate * sizeof(float));
    const MoeNvfp4ExpertScratch scratch_view{static_cast<float*>(scratch.p)};

    const std::size_t destination_bytes = static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t);
    GuardedDeviceBuffer parallel(destination_bytes);
    GuardedDeviceBuffer serial(destination_bytes);
    parallel.copy_from_host(residual_words.data(), destination_bytes);
    serial.copy_from_host(residual_words.data(), destination_bytes);
    Tensor parallel_destination{parallel.data(), dtype, {kHidden, 1}};
    Tensor serial_destination{serial.data(), dtype, {kHidden, 1}};

    ops::moe_routed_nvfp4_cached(x, bank, scratch_view, route, parallel_destination, 0);
    ops::moe_routed_nvfp4_cached(x, bank, MoeNvfp4ExpertScratch{}, route, serial_destination, 0);
    cuda_synchronize();

    const std::vector<std::uint16_t> got_parallel =
        from_device<std::uint16_t>(parallel.data(), kHidden);
    const std::vector<std::uint16_t> got_serial =
        from_device<std::uint16_t>(serial.data(), kHidden);

    int failures = 0;
    if (changed_rows(got_serial, residual_words) < kHidden / 2U) {
        std::cerr << label << ": the slot fixture produced almost no routed contribution\n";
        ++failures;
    }
    if (got_parallel != got_serial) {
        std::cerr << label << ": the multi-CTA slot path is not bitwise identical to the "
                             "single-CTA slot path\n";
        ++failures;
    }
    failures += parallel.verify_guards(label + " parallel guard");
    failures += serial.verify_guards(label + " serial guard");
    return failures;
}

int run_shared_bitwise_case(const MoeNvfp4Weights& full_weights,
                            const std::vector<float>& activation_values, DType dtype,
                            const std::string& label) {
    std::vector<std::uint16_t> activation_words;
    std::vector<std::uint16_t> residual_words;
    build_token_words(activation_values, dtype, activation_words, residual_words);

    DeviceBuffer device_x = to_device(activation_words);
    GuardedDeviceBuffer ids_buffer(kTopK * sizeof(std::int32_t));
    GuardedDeviceBuffer coefficients_buffer(kTopK * sizeof(float));
    GuardedDeviceBuffer shared_gate_buffer(sizeof(float));
    Tensor x{device_x.p, dtype, {kHidden, 1}};
    Tensor ids{ids_buffer.data(), DType::I32, {kTopK, 1}};
    Tensor coefficients{coefficients_buffer.data(), DType::FP32, {kTopK, 1}};
    Tensor shared_gate{shared_gate_buffer.data(), DType::FP32, {1, 1}};
    ops::moe_nvfp4_route(x, full_weights, ids, coefficients, shared_gate, 0);
    cuda_synchronize();

    const MoeNvfp4RouteView route{ids, coefficients, shared_gate};
    DeviceBuffer scratch(kTopK * kIntermediate * sizeof(float));
    const MoeNvfp4ExpertScratch scratch_view{static_cast<float*>(scratch.p)};

    const std::size_t destination_bytes = static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t);
    GuardedDeviceBuffer parallel(destination_bytes);
    GuardedDeviceBuffer serial(destination_bytes);
    parallel.copy_from_host(residual_words.data(), destination_bytes);
    serial.copy_from_host(residual_words.data(), destination_bytes);
    Tensor parallel_destination{parallel.data(), dtype, {kHidden, 1}};
    Tensor serial_destination{serial.data(), dtype, {kHidden, 1}};

    ops::moe_shared_nvfp4(x, full_weights, scratch_view, route, parallel_destination, 0);
    ops::moe_shared_nvfp4(x, full_weights, MoeNvfp4ExpertScratch{}, route, serial_destination, 0);
    cuda_synchronize();

    const std::vector<std::uint16_t> got_parallel =
        from_device<std::uint16_t>(parallel.data(), kHidden);
    const std::vector<std::uint16_t> got_serial =
        from_device<std::uint16_t>(serial.data(), kHidden);

    int failures = 0;
    if (changed_rows(got_serial, residual_words) < kHidden / 2U) {
        std::cerr << label << ": the shared fixture produced almost no contribution\n";
        ++failures;
    }
    if (got_parallel != got_serial) {
        std::cerr << label << ": the multi-CTA shared path is not bitwise identical to the "
                             "single-CTA shared path\n";
        ++failures;
    }
    failures += parallel.verify_guards(label + " parallel guard");
    failures += serial.verify_guards(label + " serial guard");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = verify_scale_word_domain();

    failures += verify_view_planes(make_batched(kGateUpRows, kHidden, 0x5eb3U, kViewExperts));
    failures += verify_view_planes(make_batched(kDownRows, kIntermediate, 0xc2b2U, kViewExperts));

    Case inputs;
    inputs.parent      = direct_bf16_weight::make_patterned(kParentRows, kHidden, 0x1234U);
    inputs.shared_down = direct_bf16_weight::make_patterned(kHidden, kIntermediate, 0x5678U);

    for (std::int32_t column = 0; column < kHidden; ++column) {
        inputs.parent.bits[static_cast<std::size_t>(kTieHigh) * kHidden + column] =
            inputs.parent.bits[static_cast<std::size_t>(kTieLow) * kHidden + column];
    }

    const std::int32_t channel = tie_boundary_channel(inputs.parent);
    if (channel < 0) {
        std::cerr << "tie case: no single-channel input puts the tie pair at the top-10 boundary\n";
        return 1;
    }
    std::vector<float> tie_input(kHidden, 0.0F);
    tie_input[static_cast<std::size_t>(channel)] = 1.0F;
    inputs.activation.push_back(tie_input);

    for (std::size_t seed = 1; seed <= 8; ++seed) {
        std::vector<float> candidate(kHidden);
        fill_uniform(candidate, static_cast<std::uint32_t>(seed), -4.0F, 4.0F);
        round_to_bf16(candidate);
        inputs.activation.push_back(candidate);
    }

    for (std::size_t token = 0; token < kTokens; ++token) {
        inputs.oracle.push_back(route_oracle(inputs.parent, inputs.activation[token]));
    }

    const RouteOracle& tie_oracle = inputs.oracle[0];
    if (tie_oracle.logits[kTieLow] != tie_oracle.logits[kTieHigh]) {
        std::cerr << "tie case: identical router rows did not produce an exact tie\n";
        return 1;
    }
    if (tie_oracle.logits[tie_oracle.ranked[kTopK - 2]] <= tie_oracle.logits[kTieLow]) {
        std::cerr << "tie case: the constructed input does not put the tie pair at ranks 10 and 11\n";
        return 1;
    }

    std::vector<std::int32_t> active;
    for (std::size_t token = 0; token < kTokens; ++token) {
        const RouteOracle& oracle = inputs.oracle[token];
        std::vector<std::int32_t> compared(oracle.ranked.begin(),
                                           oracle.ranked.begin() + static_cast<std::ptrdiff_t>(kCompared));
        if (token == 0) compared.push_back(kTieLow);
        inputs.compared.push_back(compared);
        active.insert(active.end(), compared.begin(), compared.end());
    }
    std::sort(active.begin(), active.end());
    active.erase(std::unique(active.begin(), active.end()), active.end());

    inputs.gate_up = make_batched(kGateUpRows, kHidden, 0x9abcU, active);
    inputs.down    = make_batched(kDownRows, kIntermediate, 0xdef0U, active);

    const std::string tie_label = "moe_nvfp4 tie case (channel " + std::to_string(channel) + ")";
    failures += run_route_case(inputs, 0, tie_label);
    if (inputs.oracle[0].ids[kTopK - 1] != kTieLow) {
        std::cerr << tie_label << ": rank 10 is not the lower id of the tie pair\n";
        ++failures;
    }

    DeviceBuffer device_parent      = to_device(inputs.parent.bits);
    DeviceBuffer device_shared_down = to_device(inputs.shared_down.bits);
    DeviceBuffer device_gate_up     = to_device(inputs.gate_up.payload);
    DeviceBuffer device_down        = to_device(inputs.down.payload);

    GuardedDeviceBuffer ids_buffer(kTokens * kTopK * sizeof(std::uint32_t));
    GuardedDeviceBuffer coefficients_buffer(kTokens * kTopK * sizeof(float));
    GuardedDeviceBuffer shared_gate_buffer(kTokens * sizeof(float));
    GuardedDeviceBuffer destination_buffer(kTokens * kHidden * sizeof(std::uint16_t));

    std::vector<std::uint16_t> residual_words(kTokens * kHidden);
    for (std::size_t t = 0; t < kTokens; ++t) {
        for (std::int32_t row = 0; row < kHidden; ++row) {
            const std::uint32_t hash = static_cast<std::uint32_t>(t * kHidden + row) * 0x9e3779b9U ^ 0xc2b2ae35U;
            residual_words[t * kHidden + static_cast<std::size_t>(row)] =
                f32_to_bf16(static_cast<float>(static_cast<int>((hash >> 8) & 0xffU) - 128) * (1.0F / 1024.0F));
        }
    }
    destination_buffer.copy_from_host(residual_words.data(), kTokens * kHidden * sizeof(std::uint16_t));

    std::vector<std::uint16_t> activation_words;
    for (std::size_t t = 0; t < kTokens; ++t) {
        for (std::int32_t column = 0; column < kHidden; ++column) {
            activation_words.push_back(f32_to_bf16(inputs.activation[t][static_cast<std::size_t>(column)]));
        }
    }
    DeviceBuffer device_x = to_device(activation_words);

    Tensor x{device_x.p, DType::BF16, {kHidden, static_cast<std::int32_t>(kTokens)}};
    Tensor ids{ids_buffer.data(), DType::I32, {kTopK, static_cast<std::int32_t>(kTokens)}};
    Tensor coefficients{coefficients_buffer.data(), DType::FP32, {kTopK, static_cast<std::int32_t>(kTokens)}};
    Tensor shared_gate{shared_gate_buffer.data(), DType::FP32, {static_cast<std::int32_t>(kTokens)}};
    Tensor destination{destination_buffer.data(), DType::BF16, {kHidden, static_cast<std::int32_t>(kTokens)}};

    MoeNvfp4Weights weights;
    weights.router_shared  = inputs.parent.device_weight(device_parent.p);
    weights.shared_down    = inputs.shared_down.device_weight(device_shared_down.p);
    weights.routed_gate_up = inputs.gate_up.view(device_gate_up.p);
    weights.routed_down    = inputs.down.view(device_down.p);

    ops::moe_nvfp4_route(x, weights, ids, coefficients, shared_gate, 0);

    const std::size_t capacity = moe_nvfp4_workspace_capacity_bytes(1, static_cast<std::int32_t>(kTokens));
    WorkspaceArena workspace(capacity);
    ops::moe_routed_nvfp4(x, weights, destination, workspace, 0);
    cuda_synchronize();

    const std::vector<std::int32_t> got_ids = from_device<std::int32_t>(ids_buffer.data(), kTokens * kTopK);
    const std::vector<double> got_coefficients = read_f32(coefficients_buffer.data(), kTokens * kTopK);
    const std::vector<double> got_gate = read_f32(shared_gate_buffer.data(), kTokens);
    const std::vector<double> got_destination = from_device_bf16(destination_buffer.data(), kTokens * kHidden);

    std::vector<double> expected_destination(kTokens * kHidden, 0.0);
    for (std::size_t t = 0; t < kTokens; ++t) {
        const RouteOracle& oracle = inputs.oracle[t];
        std::vector<std::int32_t> got_token_ids(got_ids.begin() + static_cast<std::ptrdiff_t>(t * kTopK),
                                                got_ids.begin() + static_cast<std::ptrdiff_t>(t * kTopK + kTopK));
        std::vector<std::int32_t> expected_ids(oracle.ids.begin(), oracle.ids.end());
        if (got_token_ids != expected_ids) {
            std::cerr << "moe_nvfp4 routing: top-10 mismatch at token " << t << "\n";
            ++failures;
        }
        std::vector<double> got_token_coefficients(got_coefficients.begin() + static_cast<std::ptrdiff_t>(t * kTopK),
                                                   got_coefficients.begin() + static_cast<std::ptrdiff_t>(t * kTopK + kTopK));
        std::vector<double> expected_coefficients(oracle.coefficients.begin(), oracle.coefficients.end());
        failures += verify_pointwise("moe_nvfp4 routing token " + std::to_string(t) + " coefficients",
                                     got_token_coefficients, expected_coefficients,
                                     kRouteCoefficientCriterion);
        double sum = 0.0;
        for (const double value : got_token_coefficients) sum += value;
        if (std::abs(sum - 1.0) > kCoefficientSumTolerance) {
            std::cerr << "moe_nvfp4 routing: coefficients at token " << t << " sum to " << sum << "\n";
            ++failures;
        }
        failures += verify_pointwise("moe_nvfp4 routing token " + std::to_string(t) + " shared gate",
                                     std::vector<double>{got_gate[t]},
                                     std::vector<double>{oracle.shared_gate},
                                     kRouteCoefficientCriterion);

        for (std::size_t i = 0; i < kHidden; ++i) {
            expected_destination[t * kHidden + i] = bf16_to_f32(residual_words[t * kHidden + i]);
        }
        for (const std::int32_t expert : inputs.compared[t]) {
            double coefficient = 0.0;
            for (std::int32_t rank = 0; rank < kTopK; ++rank) {
                if (oracle.ids[static_cast<std::size_t>(rank)] == expert) {
                    coefficient = oracle.coefficients[static_cast<std::size_t>(rank)];
                }
            }
            const std::vector<double> contribution =
                expert_contribution(inputs.gate_up, inputs.down, expert, inputs.activation[t], coefficient);
            for (std::size_t i = 0; i < kHidden; ++i) expected_destination[t * kHidden + i] += contribution[i];
        }
        const std::vector<double> shared = shared_contribution(inputs.parent, inputs.shared_down,
                                                                inputs.activation[t], oracle.shared_gate);
        for (std::size_t i = 0; i < kHidden; ++i) expected_destination[t * kHidden + i] += shared[i];
    }

    failures += verify_reduction("moe_nvfp4 destination", got_destination, expected_destination,
                                 kDestinationCriterion);
    if (workspace.used() > capacity) {
        std::cerr << "moe_nvfp4 workspace: used " << workspace.used()
                  << " exceeds the reported capacity " << capacity << "\n";
        ++failures;
    }
    failures += ids_buffer.verify_guards("moe_nvfp4 ids guard");
    failures += coefficients_buffer.verify_guards("moe_nvfp4 coefficients guard");
    failures += shared_gate_buffer.verify_guards("moe_nvfp4 shared gate guard");
    failures += destination_buffer.verify_guards("moe_nvfp4 destination guard");

    constexpr float kF16ExpertDivisorScale = 1024.0F;
    BatchedFixture gate_up_f16 =
        make_batched(kGateUpRows, kHidden, 0x9abcU, active, kF16ExpertDivisorScale);
    BatchedFixture down_f16 =
        make_batched(kDownRows, kIntermediate, 0xdef0U, active, kF16ExpertDivisorScale);
    DeviceBuffer device_gate_up_f16 = to_device(gate_up_f16.payload);
    DeviceBuffer device_down_f16    = to_device(down_f16.payload);

    MoeNvfp4Weights weights_f16;
    weights_f16.router_shared  = weights.router_shared;
    weights_f16.shared_down    = weights.shared_down;
    weights_f16.routed_gate_up = gate_up_f16.view(device_gate_up_f16.p);
    weights_f16.routed_down    = down_f16.view(device_down_f16.p);

    std::vector<float> activation_f16(kTokens * kHidden);
    std::vector<std::uint16_t> activation_f16_bits(kTokens * kHidden);
    for (std::size_t t = 0; t < kTokens; ++t) {
        for (std::int32_t c = 0; c < kHidden; ++c) {
            const std::uint16_t bits = f32_to_f16(inputs.activation[t][static_cast<std::size_t>(c)]);
            activation_f16_bits[t * kHidden + static_cast<std::size_t>(c)] = bits;
            activation_f16[t * kHidden + static_cast<std::size_t>(c)] = f16_to_f32(bits);
        }
    }
    std::vector<std::uint16_t> residual_f16_bits(kTokens * kHidden);
    for (std::size_t i = 0; i < residual_f16_bits.size(); ++i) {
        residual_f16_bits[i] = f32_to_f16(bf16_to_f32(residual_words[i]));
    }

    const DeviceBuffer x_f16 = to_device(activation_f16_bits);
    Tensor x16{x_f16.p, DType::FP16, {kHidden, static_cast<std::int32_t>(kTokens)}};

    GuardedDeviceBuffer ids_f16(kTokens * kTopK * sizeof(std::uint32_t));
    GuardedDeviceBuffer coefficients_f16(kTokens * kTopK * sizeof(float));
    GuardedDeviceBuffer shared_gate_f16(kTokens * sizeof(float));
    Tensor ids16{ids_f16.data(), DType::I32, {kTopK, static_cast<std::int32_t>(kTokens)}};
    Tensor coefficients16{coefficients_f16.data(), DType::FP32,
                          {kTopK, static_cast<std::int32_t>(kTokens)}};
    Tensor shared_gate16{shared_gate_f16.data(), DType::FP32,
                         {static_cast<std::int32_t>(kTokens)}};
    ops::moe_nvfp4_route(x16, weights_f16, ids16, coefficients16, shared_gate16, 0);
    const MoeNvfp4RouteView route16{ids16, coefficients16, shared_gate16};

    GuardedDeviceBuffer destination_f16(kTokens * kHidden * sizeof(std::uint16_t));
    destination_f16.copy_from_host(residual_f16_bits.data(),
                                   kTokens * kHidden * sizeof(std::uint16_t));
    Tensor destination16{destination_f16.data(), DType::FP16,
                         {kHidden, static_cast<std::int32_t>(kTokens)}};
    WorkspaceArena workspace_f16(capacity);
    ops::moe_routed_nvfp4(x16, weights_f16, destination16, workspace_f16, 0);
    cuda_synchronize();

    const std::vector<std::int32_t> got_ids_f16 =
        from_device<std::int32_t>(ids_f16.data(), kTokens * kTopK);
    const std::vector<double> got_coefficients_f16 =
        read_f32(coefficients_f16.data(), kTokens * kTopK);
    const std::vector<double> got_gate_f16 = read_f32(shared_gate_f16.data(), kTokens);
    const std::vector<double> got_destination_f16 =
        from_device_f16(destination_f16.data(), kTokens * kHidden);

    GuardedDeviceBuffer partial_f16(kTokens * kHidden * sizeof(std::uint16_t));
    partial_f16.fill(0);
    Tensor partial16{partial_f16.data(), DType::FP16, {kHidden, static_cast<std::int32_t>(kTokens)}};
    ops::moe_routed_nvfp4_partial(x16, weights_f16, route16, partial16, 0);
    cuda_synchronize();
    const std::vector<double> got_partial_f16 =
        from_device_f16(partial_f16.data(), kTokens * kHidden);

    std::vector<double> expected_destination_f16(kTokens * kHidden, 0.0);
    std::vector<double> expected_partial_f16(kTokens * kHidden, 0.0);
    double max_routed_f16 = 0.0;
    for (std::size_t t = 0; t < kTokens; ++t) {
        std::vector<float> activation(activation_f16.begin() + static_cast<std::ptrdiff_t>(t * kHidden),
                                      activation_f16.begin() + static_cast<std::ptrdiff_t>((t + 1) * kHidden));
        const RouteOracle oracle = route_oracle(inputs.parent, activation);
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            if (got_ids_f16[t * kTopK + static_cast<std::size_t>(rank)] !=
                oracle.ids[static_cast<std::size_t>(rank)]) {
                std::cerr << "moe_nvfp4 f16 routing mismatch at token " << t << " rank " << rank << "\n";
                ++failures;
            }
        }
        for (std::size_t i = 0; i < kHidden; ++i) {
            expected_destination_f16[t * kHidden + i] =
                f16_to_f32(residual_f16_bits[t * kHidden + i]);
        }
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            const std::int32_t expert = oracle.ids[static_cast<std::size_t>(rank)];
            const std::vector<double> contribution =
                expert_contribution(gate_up_f16, down_f16, expert, activation,
                                    oracle.coefficients[static_cast<std::size_t>(rank)]);
            for (std::size_t i = 0; i < kHidden; ++i) {
                expected_destination_f16[t * kHidden + i] += contribution[i];
                expected_partial_f16[t * kHidden + i] += contribution[i];
                max_routed_f16 = std::max(max_routed_f16, std::abs(contribution[i]));
            }
        }
        const std::vector<double> shared =
            shared_contribution(inputs.parent, inputs.shared_down, activation, oracle.shared_gate);
        for (std::size_t i = 0; i < kHidden; ++i) {
            expected_destination_f16[t * kHidden + i] += shared[i];
        }
        failures += verify_pointwise("moe_nvfp4 f16 token " + std::to_string(t) + " coefficients",
                                     std::vector<double>(got_coefficients_f16.begin() +
                                                             static_cast<std::ptrdiff_t>(t * kTopK),
                                                         got_coefficients_f16.begin() +
                                                             static_cast<std::ptrdiff_t>((t + 1) * kTopK)),
                                     std::vector<double>(oracle.coefficients.begin(),
                                                         oracle.coefficients.end()),
                                     kRouteCoefficientCriterion);
        failures += verify_pointwise("moe_nvfp4 f16 token " + std::to_string(t) + " shared gate",
                                     std::vector<double>{got_gate_f16[t]},
                                     std::vector<double>{oracle.shared_gate},
                                     kRouteCoefficientCriterion);
    }
    if (!(max_routed_f16 > 1.0)) {
        std::cerr << "moe_nvfp4 f16 routed fixture is too weak: max |contribution| = "
                  << max_routed_f16 << "\n";
        ++failures;
    }
    for (const double value : expected_partial_f16) {
        if (!std::isfinite(value) || std::abs(value) > 65504.0) {
            std::cerr << "moe_nvfp4 f16 routed-only fixture is not FP16-representable\n";
            ++failures;
            break;
        }
    }
    failures += verify_reduction("moe_nvfp4 f16 routed-only destination", got_partial_f16,
                                 expected_partial_f16, kF16DestinationCriterion);
    failures += partial_f16.verify_guards("moe_nvfp4 f16 routed-only guard");
    for (const double value : expected_destination_f16) {
        if (!std::isfinite(value) || std::abs(value) > 65504.0) {
            std::cerr << "moe_nvfp4 f16 destination: the fixture is not FP16-representable\n";
            ++failures;
            break;
        }
    }
    failures += verify_reduction("moe_nvfp4 f16 destination", got_destination_f16,
                                 expected_destination_f16, kF16DestinationCriterion);
    failures += ids_f16.verify_guards("moe_nvfp4 f16 ids guard");
    failures += coefficients_f16.verify_guards("moe_nvfp4 f16 coefficients guard");
    failures += shared_gate_f16.verify_guards("moe_nvfp4 f16 shared gate guard");
    failures += destination_f16.verify_guards("moe_nvfp4 f16 destination guard");
    if (workspace_f16.used() > capacity) {
        std::cerr << "moe_nvfp4 f16 workspace: used " << workspace_f16.used()
                  << " exceeds the reported capacity " << capacity << "\n";
        ++failures;
    }

    const std::int32_t local_begin = 256;
    const std::int32_t local_count = 256;
    LocalBank local_gate_up = local_bank(inputs.gate_up, local_begin, local_count);
    LocalBank local_down    = local_bank(inputs.down, local_begin, local_count);
    DeviceBuffer local_gate_up_device = to_device(local_gate_up.payload);
    DeviceBuffer local_down_device    = to_device(local_down.payload);

    MoeNvfp4Weights local_weights;
    local_weights.router_shared  = weights.router_shared;
    local_weights.shared_down    = weights.shared_down;
    local_weights.routed_gate_up = local_gate_up.view(local_gate_up_device.p);
    local_weights.routed_down    = local_down.view(local_down_device.p);
    const MoeNvfp4RouteView route{ids, coefficients, shared_gate};

    bool low_selected  = false;
    bool high_selected = false;
    for (const std::int32_t id : got_ids) {
        if (id < local_begin) { low_selected = true; }
        if (id >= local_begin && id < local_begin + local_count) { high_selected = true; }
    }
    if (!low_selected || !high_selected) {
        std::cerr << "moe_nvfp4 local subset: the routing does not straddle the local expert range\n";
        ++failures;
    }

    GuardedDeviceBuffer local_destination_buffer(kTokens * kHidden * sizeof(std::uint16_t));
    local_destination_buffer.copy_from_host(residual_words.data(),
                                            kTokens * kHidden * sizeof(std::uint16_t));
    Tensor local_destination{local_destination_buffer.data(), DType::BF16,
                             {kHidden, static_cast<std::int32_t>(kTokens)}};
    ops::moe_routed_nvfp4_partial(x, local_weights, route, local_destination, 0);
    ops::moe_shared_nvfp4(x, local_weights, MoeNvfp4ExpertScratch{}, route, local_destination, 0);
    cuda_synchronize();
    const std::vector<double> got_local =
        from_device_bf16(local_destination_buffer.data(), kTokens * kHidden);

    std::vector<double> expected_local(kTokens * kHidden, 0.0);
    for (std::size_t t = 0; t < kTokens; ++t) {
        const RouteOracle& oracle = inputs.oracle[t];
        for (std::size_t i = 0; i < kHidden; ++i) {
            expected_local[t * kHidden + i] = bf16_to_f32(residual_words[t * kHidden + i]);
        }
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            const std::int32_t expert = oracle.ids[static_cast<std::size_t>(rank)];
            if (expert < local_begin || expert >= local_begin + local_count) { continue; }
            const std::vector<double> contribution =
                expert_contribution(inputs.gate_up, inputs.down, expert, inputs.activation[t],
                                    oracle.coefficients[static_cast<std::size_t>(rank)]);
            for (std::size_t i = 0; i < kHidden; ++i) {
                expected_local[t * kHidden + i] += contribution[i];
            }
        }
        const std::vector<double> shared = shared_contribution(
            inputs.parent, inputs.shared_down, inputs.activation[t], oracle.shared_gate);
        for (std::size_t i = 0; i < kHidden; ++i) {
            expected_local[t * kHidden + i] += shared[i];
        }
    }
    failures += verify_reduction("moe_nvfp4 local subset", got_local, expected_local,
                                 kDestinationCriterion);
    failures += local_destination_buffer.verify_guards("moe_nvfp4 local destination guard");

    failures += run_cached_bitwise_case(inputs.gate_up, inputs.down, weights,
                                        inputs.activation[1], DType::BF16, "moe_nvfp4 cached bf16");
    failures += run_cached_bitwise_case(gate_up_f16, down_f16, weights_f16,
                                        inputs.activation[2], DType::FP16, "moe_nvfp4 cached f16");
    failures += run_shared_bitwise_case(weights, inputs.activation[1], DType::BF16,
                                        "moe_nvfp4 shared bf16");
    failures += run_shared_bitwise_case(weights_f16, inputs.activation[2], DType::FP16,
                                        "moe_nvfp4 shared f16");
    failures += run_probe_cases();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_nvfp4 correctness\n";
    return failures == 0 ? 0 : 1;
}
