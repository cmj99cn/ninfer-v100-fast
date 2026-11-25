#include "ninfer/ops/linear.h"
#include "ops/direct_fp16_weight.h"
#include "ops/op_tester.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace test = ninfer::test;
namespace fp16w = ninfer::test::direct_fp16_weight;
namespace flash = ninfer::targets::qwen3_8_flash_next::detail;

constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kIntermediate = 640;
constexpr std::int32_t kGateRows = 2 * kIntermediate;
constexpr float kCoefficient = 0.125F;
constexpr test::ReductionCriterion kTolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

std::vector<float> expected_t1_total_local;

std::size_t align256(std::size_t value) { return (value + 255U) / 256U * 256U; }

float decode_e4m3_host(std::uint8_t code) {
    const float sign = (code & 0x80U) != 0 ? -1.0F : 1.0F;
    const std::uint8_t exponent = (code >> 3) & 0x0FU;
    const std::uint8_t fraction = code & 0x07U;
    if (exponent == 0) {
        if (fraction == 0) { return sign * 0.0F; }
        return sign * static_cast<float>(fraction) * (1.0F / 512.0F);
    }
    if (exponent == 0x0FU && fraction == 0x07U) { return std::numeric_limits<float>::quiet_NaN(); }
    return sign * (1.0F + static_cast<float>(fraction) / 8.0F) *
           std::ldexp(1.0F, static_cast<int>(exponent) - 7);
}

struct Bank {
    std::int32_t rows = 0;
    std::int32_t columns = 0;
    std::size_t scale_offset = 0;
    std::size_t blocks = 0;
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;

    std::uint16_t scale(std::int32_t row, std::int32_t column) const {
        const auto index =
            static_cast<std::size_t>(row >> 7) * static_cast<std::size_t>(columns >> 7) +
            static_cast<std::size_t>(column >> 7);
        return scales[index];
    }
};

std::uint8_t code_pattern(std::int32_t row, std::int32_t column, std::uint32_t seed) {
    std::uint32_t hash = static_cast<std::uint32_t>(row) * 0x9e3779b9U ^
                         static_cast<std::uint32_t>(column) * 0x85ebca6bU ^ seed * 0xc2b2ae35U;
    hash ^= hash >> 16;
    hash *= 0x7feb352dU;
    hash ^= hash >> 15;
    return static_cast<std::uint8_t>(hash & 0xfeU); // never the NaN code
}

Bank make_bank(std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    Bank bank;
    bank.rows = rows;
    bank.columns = columns;
    bank.codes.resize(static_cast<std::size_t>(rows) * columns);
    bank.scale_offset = align256(bank.codes.size());
    bank.blocks = static_cast<std::size_t>(rows / 128) * static_cast<std::size_t>(columns / 128);
    bank.scales.resize(bank.blocks);
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            bank.codes[static_cast<std::size_t>(row) * columns + column] =
                code_pattern(row, column, seed);
        }
    }
    for (std::size_t index = 0; index < bank.blocks; ++index) {
        const float value = 0.001953125F +
                            static_cast<float>((index * 37U) % 64U) * (1.0F / 65536.0F);
        bank.scales[index] = test::f32_to_bf16(value);
    }
    // Canaries at block boundaries, where the scale index changes, and at the far corners.
    bank.codes[0] = 0x3f;
    bank.codes[127 * columns + 127] = 0x70;
    bank.codes[128 * columns + 128] = 0x0f;
    bank.codes[static_cast<std::size_t>(rows - 1) * columns + columns - 1] = 0x61;
    bank.scales[0] = test::f32_to_bf16(0.00390625F);
    bank.scales[bank.blocks - 1] = test::f32_to_bf16(0.0078125F);
    return bank;
}

std::vector<std::uint16_t> decode_host(const Bank& bank) {
    std::vector<std::uint16_t> decoded(bank.codes.size());
    for (std::int32_t row = 0; row < bank.rows; ++row) {
        for (std::int32_t column = 0; column < bank.columns; ++column) {
            const auto index = static_cast<std::size_t>(row) * bank.columns + column;
            const float value =
                decode_e4m3_host(bank.codes[index]) * test::bf16_to_f32(bank.scale(row, column));
            decoded[index] = fp16w::f32_to_fp16_bits(value);
        }
    }
    return decoded;
}

fp16w::HostWeight slice_weight(const std::vector<std::uint16_t>& decoded, std::int32_t columns,
                               std::int32_t row_begin, std::int32_t rows) {
    fp16w::HostWeight weight;
    weight.n = rows;
    weight.k = columns;
    weight.bits.assign(decoded.begin() + static_cast<std::ptrdiff_t>(row_begin) * columns,
                       decoded.begin() + static_cast<std::ptrdiff_t>((row_begin + rows) * columns));
    return weight;
}

std::vector<std::uint16_t> make_activation(std::int32_t tokens) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(kHidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < kHidden; ++column) {
            const int centered = ((column * 29 + token * 71 + 17) & 0xff) - 128;
            bits[static_cast<std::size_t>(token) * kHidden + column] =
                fp16w::f32_to_fp16_bits(static_cast<float>(centered) * (1.0F / 512.0F));
        }
    }
    for (std::int32_t token = 0; token < tokens; ++token) {
        const auto base = static_cast<std::size_t>(token) * kHidden;
        bits[base] = fp16w::f32_to_fp16_bits(0.25F);
        bits[base + 127] = fp16w::f32_to_fp16_bits(-0.5F);
        bits[base + 128] = fp16w::f32_to_fp16_bits(0.125F);
        bits[base + kHidden - 1] = fp16w::f32_to_fp16_bits(0.75F);
    }
    return bits;
}

int run_case(std::int32_t tokens, const std::vector<float>& expected_t1_total) {
    const Bank gate_bank = make_bank(kGateRows, kHidden, 0x51U);
    const Bank down_bank = make_bank(kHidden, kIntermediate, 0xa7U);
    const std::vector<std::uint16_t> gate_decoded = decode_host(gate_bank);
    const std::vector<std::uint16_t> down_decoded = decode_host(down_bank);

    DeviceBuffer gate_codes = test::to_device(gate_bank.codes);
    DeviceBuffer gate_scales = test::to_device(gate_bank.scales);
    DeviceBuffer down_codes = test::to_device(down_bank.codes);
    DeviceBuffer down_scales = test::to_device(down_bank.scales);
    DeviceBuffer gate_decoded_device(gate_decoded.size() * 2);
    DeviceBuffer down_decoded_device(down_decoded.size() * 2);
    flash::mtp_decode_fp8_block128(
        static_cast<const std::uint8_t*>(gate_codes.p),
        static_cast<const std::uint16_t*>(gate_scales.p), kGateRows, kHidden,
        gate_decoded_device.p, nullptr);
    flash::mtp_decode_fp8_block128(
        static_cast<const std::uint8_t*>(down_codes.p),
        static_cast<const std::uint16_t*>(down_scales.p), kHidden, kIntermediate,
        down_decoded_device.p, nullptr);
    test::cuda_synchronize();
    int failures = 0;
    const auto gate_actual = test::from_device<std::uint16_t>(gate_decoded_device, gate_decoded.size());
    const auto down_actual = test::from_device<std::uint16_t>(down_decoded_device, down_decoded.size());
    failures += test::verify_exact("draft expert gate decode [1280,2560]", gate_actual, gate_decoded);
    failures += test::verify_exact("draft expert down decode [2560,640]", down_actual, down_decoded);

    fp16w::DeviceWeight gate_weight(slice_weight(gate_decoded, kHidden, 0, kIntermediate));
    fp16w::DeviceWeight up_weight(slice_weight(gate_decoded, kHidden, kIntermediate, kIntermediate));
    fp16w::DeviceWeight down_weight(slice_weight(down_decoded, kIntermediate, 0, kHidden));
    const std::vector<std::uint16_t> activation_bits = make_activation(tokens);
    DeviceBuffer activation_device = test::to_device(activation_bits);
    Tensor x(activation_device.p, DType::FP16, {kHidden, tokens});

    DeviceBuffer gate_part(static_cast<std::size_t>(kIntermediate) * tokens * 2);
    DeviceBuffer up_part(static_cast<std::size_t>(kIntermediate) * tokens * 2);
    Tensor gate_view(gate_part.p, DType::FP16, {kIntermediate, tokens});
    Tensor up_view(up_part.p, DType::FP16, {kIntermediate, tokens});
    DeviceArena workspace(1024 * 1024);
    ops::linear(x, gate_weight.view(), gate_view, ops::LinearPolicy::A16Only, workspace, nullptr);
    ops::linear(x, up_weight.view(), up_view, ops::LinearPolicy::A16Only, workspace, nullptr);

    DeviceBuffer gate_up(static_cast<std::size_t>(kGateRows) * tokens * 2);
    CUDA_CHECK(cudaMemcpy(gate_up.p, gate_part.p, gate_part.bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(gate_up.p) +
                              static_cast<std::size_t>(kIntermediate) * tokens * 2,
                          up_part.p, up_part.bytes, cudaMemcpyDeviceToDevice));

    DeviceBuffer expert_activation(static_cast<std::size_t>(kIntermediate) * tokens * 2);
    flash::mtp_silu_mul_f16(gate_up.p, expert_activation.p, kIntermediate * tokens, nullptr);

    DeviceBuffer expert_output(static_cast<std::size_t>(kHidden) * tokens * 2);
    Tensor activation_view(expert_activation.p, DType::FP16, {kIntermediate, tokens});
    Tensor output_view(expert_output.p, DType::FP16, {kHidden, tokens});
    ops::linear(activation_view, down_weight.view(), output_view, ops::LinearPolicy::A16Only,
                workspace, nullptr);

    std::vector<float> initial_total(static_cast<std::size_t>(kHidden) * tokens);
    for (std::size_t index = 0; index < initial_total.size(); ++index) {
        initial_total[index] = static_cast<float>((index % 17) - 8) * (1.0F / 64.0F);
    }
    DeviceBuffer total_device = test::to_device(initial_total);
    flash::mtp_scale_add_f16(expert_output.p, kCoefficient, static_cast<float*>(total_device.p),
                             kHidden * tokens, nullptr);
    test::cuda_synchronize();

    const auto gate_up_actual = test::from_device<std::uint16_t>(gate_up, gate_up.bytes / 2);
    const auto activation_actual =
        test::from_device<std::uint16_t>(expert_activation, expert_activation.bytes / 2);
    const auto output_actual = test::from_device<std::uint16_t>(expert_output, expert_output.bytes / 2);
    const auto total_actual = test::from_device<float>(total_device, initial_total.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        std::vector<float> activation(static_cast<std::size_t>(kHidden));
        for (std::int32_t column = 0; column < kHidden; ++column) {
            activation[static_cast<std::size_t>(column)] =
                fp16w::fp16_bits_to_f32(activation_bits[static_cast<std::size_t>(token) * kHidden +
                                                       column]);
        }
        std::vector<double> gate_got(static_cast<std::size_t>(kIntermediate));
        std::vector<double> gate_ref(static_cast<std::size_t>(kIntermediate));
        std::vector<double> up_got(static_cast<std::size_t>(kIntermediate));
        std::vector<double> up_ref(static_cast<std::size_t>(kIntermediate));
        std::vector<double> silu_got(static_cast<std::size_t>(kIntermediate));
        std::vector<double> silu_ref(static_cast<std::size_t>(kIntermediate));
        std::vector<double> down_got(static_cast<std::size_t>(kHidden));
        std::vector<double> down_ref(static_cast<std::size_t>(kHidden));
        std::vector<double> total_got(static_cast<std::size_t>(kHidden));
        std::vector<double> total_ref(static_cast<std::size_t>(kHidden));
        for (std::int32_t row = 0; row < kIntermediate; ++row) {
            const auto gate_offset = static_cast<std::size_t>(token) * kIntermediate + row;
            const auto up_offset = static_cast<std::size_t>(tokens) * kIntermediate + gate_offset;
            gate_got[row] = fp16w::fp16_bits_to_f32(gate_up_actual[gate_offset]);
            up_got[row] = fp16w::fp16_bits_to_f32(gate_up_actual[up_offset]);
            gate_ref[row] = fp16w::dot_fp64(gate_weight.host, row, activation);
            up_ref[row] = fp16w::dot_fp64(up_weight.host, row, activation);
            const double gate = gate_got[row];
            const double up = up_got[row];
            silu_got[row] =
                fp16w::fp16_bits_to_f32(activation_actual[static_cast<std::size_t>(token) *
                                                           kIntermediate + row]);
            silu_ref[row] = fp16w::fp16_bits_to_f32(fp16w::f32_to_fp16_bits(
                static_cast<float>((gate / (1.0 + std::exp(-gate))) * up)));
        }
        std::vector<float> silu_activation(static_cast<std::size_t>(kIntermediate));
        for (std::int32_t row = 0; row < kIntermediate; ++row) {
            silu_activation[static_cast<std::size_t>(row)] =
                fp16w::fp16_bits_to_f32(activation_actual[static_cast<std::size_t>(token) *
                                                           kIntermediate + row]);
        }
        for (std::int32_t row = 0; row < kHidden; ++row) {
            down_got[row] = fp16w::fp16_bits_to_f32(
                output_actual[static_cast<std::size_t>(token) * kHidden + row]);
            down_ref[row] = fp16w::dot_fp64(down_weight.host, row, silu_activation);
            total_got[row] = total_actual[static_cast<std::size_t>(token) * kHidden + row];
            total_ref[row] = initial_total[static_cast<std::size_t>(token) * kHidden + row] +
                             static_cast<double>(kCoefficient) * down_got[row];
        }
        const std::string suffix = " T=" + std::to_string(tokens) + " token=" + std::to_string(token);
        failures += test::verify_reduction("draft expert gate GEMV" + suffix, gate_got, gate_ref,
                                            kTolerance);
        failures += test::verify_reduction("draft expert up GEMV" + suffix, up_got, up_ref,
                                            kTolerance);
        failures += test::verify_reduction("draft expert silu" + suffix, silu_got, silu_ref,
                                            kTolerance);
        failures += test::verify_reduction("draft expert down GEMV" + suffix, down_got, down_ref,
                                            kTolerance);
        failures += test::verify_reduction("draft expert scale add" + suffix, total_got, total_ref,
                                            kTolerance);
    }
    if (tokens == 2) {
        for (std::size_t index = 0; index < static_cast<std::size_t>(kHidden); ++index) {
            if (total_actual[index] != expected_t1_total[index]) {
                std::cerr << "draft expert T=2 token0 total differs from T=1 at " << index << '\n';
                ++failures;
                break;
            }
        }
    } else {
        expected_t1_total_local = total_actual;
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (test::cuda_unavailable() || test::cuda_compute_major() != 7) {
            std::cout << "skip: draft expert oracle needs an sm_70 CUDA device\n";
            return 77;
        }
        int failures = run_case(1, std::vector<float>{});
        failures += run_case(2, expected_t1_total_local);
        if (failures != 0) {
            std::cerr << "draft expert oracle: " << failures << " failure(s)\n";
            return 1;
        }
        std::cout << "draft expert oracle: T=1 and T=2 pass with canary, real geometries\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "draft expert oracle failed: " << error.what() << '\n';
        return 1;
    }
}
