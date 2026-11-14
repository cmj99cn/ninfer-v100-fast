#include "ninfer/ops/gated_residual.h"

#include "core/arena.h"
#include "ops/direct_bf16_weight.h"
#include "ops/direct_fp16_weight.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden  = 2560;
constexpr std::int32_t kStreams = 4;
constexpr std::int32_t kRank    = 320;
constexpr float kEpsilon        = 1.0e-6F;

constexpr double kUnitRoundoff = 1.0 / 256.0;

double oracle_sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double e = std::exp(value);
    return e / (1.0 + e);
}

double oracle_silu(double value) { return value * oracle_sigmoid(value); }

float to_bf16(float value) { return bf16_to_f32(f32_to_bf16(value)); }

float to_activation(float value, DType dtype) {
    return dtype == DType::FP16 ? f16_to_f32(f32_to_f16(value))
                                : bf16_to_f32(f32_to_bf16(value));
}

void round_to_activation(std::vector<float>& values, DType dtype) {
    if (dtype == DType::FP16) {
        round_to_f16(values);
    } else {
        round_to_bf16(values);
    }
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<std::uint16_t> encode_activation(const std::vector<float>& values, DType dtype) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        bits[i] = dtype == DType::FP16 ? f32_to_f16(values[i]) : f32_to_bf16(values[i]);
    }
    return bits;
}

Tensor bf16_tensor(void* data, std::int32_t d0, std::int32_t d1) {
    return Tensor(data, DType::BF16, {d0, d1});
}

Tensor bf16_tensor(void* data, std::int32_t d0, std::int32_t d1, std::int32_t d2) {
    return Tensor(data, DType::BF16, {d0, d1, d2});
}

Tensor activation_tensor(void* data, DType dtype, std::int32_t d0, std::int32_t d1) {
    return Tensor(data, dtype, {d0, d1});
}

Tensor activation_tensor(void* data, DType dtype, std::int32_t d0, std::int32_t d1,
                         std::int32_t d2) {
    return Tensor(data, dtype, {d0, d1, d2});
}

std::vector<double> read_activation(const void* data, std::size_t count, DType dtype) {
    return dtype == DType::FP16 ? from_device_f16(data, count) : from_device_bf16(data, count);
}

std::vector<double> expand_norm_oracle(const std::vector<float>& embedding,
                                        const std::vector<float>& norm_weight, std::int32_t columns) {
    std::vector<double> normalized(static_cast<std::size_t>(kHidden) * kStreams * columns);
    for (std::int32_t m = 0; m < columns; ++m) {
        double sum_squares = 0.0;
        for (std::int32_t d = 0; d < kHidden; ++d) {
            const double value = static_cast<double>(embedding[d + static_cast<std::size_t>(kHidden) * m]);
            sum_squares += value * value;
        }
        const double inverse =
            1.0 / std::sqrt(sum_squares / static_cast<double>(kHidden) + static_cast<double>(kEpsilon));
        for (std::int32_t s = 0; s < kStreams; ++s) {
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const double gain = 1.0 + static_cast<double>(norm_weight[d + static_cast<std::size_t>(kHidden) * s]);
                normalized[d + static_cast<std::size_t>(kHidden) * s +
                           static_cast<std::size_t>(kHidden) * kStreams * m] =
                    static_cast<double>(embedding[d + static_cast<std::size_t>(kHidden) * m]) * gain * inverse;
            }
        }
    }
    return normalized;
}

std::vector<double> grouped_rmsnorm_oracle(const std::vector<float>& residual,
                                            const std::vector<float>& norm_weight,
                                            std::int32_t columns) {
    std::vector<double> normalized(static_cast<std::size_t>(kHidden) * kStreams * columns);
    for (std::int32_t m = 0; m < columns; ++m) {
        for (std::int32_t s = 0; s < kStreams; ++s) {
            const std::size_t base = static_cast<std::size_t>(kHidden) * s +
                                     static_cast<std::size_t>(kHidden) * kStreams * m;
            double sum_squares = 0.0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const double value = static_cast<double>(residual[base + d]);
                sum_squares += value * value;
            }
            const double inverse = 1.0 /
                                   std::sqrt(sum_squares / static_cast<double>(kHidden) +
                                             static_cast<double>(kEpsilon));
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const double gain =
                    1.0 + static_cast<double>(norm_weight[d + static_cast<std::size_t>(kHidden) * s]);
                normalized[base + d] = static_cast<double>(residual[base + d]) * gain * inverse;
            }
        }
    }
    return normalized;
}

std::vector<double> combine_oracle(const std::vector<float>& residual,
                                    const std::vector<float>& block_output,
                                    const std::vector<float>& injection, std::int32_t columns) {
    std::vector<double> updated(static_cast<std::size_t>(kHidden) * kStreams * columns);
    for (std::int32_t m = 0; m < columns; ++m) {
        for (std::int32_t s = 0; s < kStreams; ++s) {
            const double coefficient =
                static_cast<double>(injection[s + static_cast<std::size_t>(kStreams) * m]);
            const std::size_t base = static_cast<std::size_t>(kHidden) * s +
                                     static_cast<std::size_t>(kHidden) * kStreams * m;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                updated[base + d] = static_cast<double>(residual[base + d]) +
                                    coefficient * static_cast<double>(
                                                      block_output[d + static_cast<std::size_t>(kHidden) * m]);
            }
        }
    }
    return updated;
}

struct MixReference {
    std::vector<double> block_input;
    std::vector<double> injection;
};

MixReference mix_oracle(const std::vector<float>& normalized, const std::vector<float>& down,
                         const std::vector<float>& up, std::int32_t columns, bool with_injection) {
    const std::int32_t channels = kHidden * kStreams;
    MixReference reference;
    reference.block_input.assign(static_cast<std::size_t>(kHidden) * columns, 0.0);
    reference.injection.assign(static_cast<std::size_t>(kStreams) * columns, 0.0);

    std::vector<double> flat(static_cast<std::size_t>(channels));
    std::vector<double> low(static_cast<std::size_t>(kRank));
    std::vector<double> activated(static_cast<std::size_t>(kRank));
    std::vector<double> gate(static_cast<std::size_t>(channels));

    for (std::int32_t m = 0; m < columns; ++m) {
        for (std::int32_t c = 0; c < channels; ++c) {
            const std::int32_t d = c % kHidden;
            const std::int32_t s = c / kHidden;
            flat[c] = static_cast<double>(
                normalized[d + static_cast<std::size_t>(kHidden) * s +
                           static_cast<std::size_t>(kHidden) * kStreams * m]);
        }
        for (std::int32_t r = 0; r < kRank; ++r) {
            double sum = 0.0;
            for (std::int32_t c = 0; c < channels; ++c) {
                sum += static_cast<double>(down[static_cast<std::size_t>(r) * channels + c]) * flat[c];
            }
            low[r] = sum;
            activated[r] = oracle_silu(sum / static_cast<double>(kStreams));
        }
        if (with_injection) {
            for (std::int32_t s = 0; s < kStreams; ++s) {
                double sum = 0.0;
                for (std::int32_t c = 0; c < channels; ++c) {
                    sum += static_cast<double>(down[static_cast<std::size_t>(kRank + s) * channels + c]) *
                           flat[c];
                }
                reference.injection[s + static_cast<std::size_t>(kStreams) * m] =
                    2.0 * oracle_sigmoid(sum / static_cast<double>(kStreams));
            }
        }
        for (std::int32_t c = 0; c < channels; ++c) {
            double sum = 0.0;
            for (std::int32_t r = 0; r < kRank; ++r) {
                sum += static_cast<double>(up[static_cast<std::size_t>(c) * kRank + r]) * activated[r];
            }
            gate[c] = sum;
        }
        for (std::int32_t d = 0; d < kHidden; ++d) {
            double sum = 0.0;
            for (std::int32_t s = 0; s < kStreams; ++s) {
                const std::int32_t c = s * kHidden + d;
                sum += oracle_sigmoid(gate[c]) * flat[c];
            }
            reference.block_input[d + static_cast<std::size_t>(kHidden) * m] =
                sum / static_cast<double>(kStreams);
        }
    }
    return reference;
}

constexpr ReductionCriterion kNormBf16Criterion{1.85e-3, 1.0e-5, 3.4e-3};
constexpr ReductionCriterion kNormF16Criterion{4.63e-4, 1.0e-5, 8.5e-4};

constexpr PointwiseCriterion kCombineBf16Criterion{0.0, 3.95e-3};
constexpr PointwiseCriterion kCombineF16Criterion{0.0, 9.88e-4};

constexpr ReductionCriterion kMixBlockBf16Criterion{kUnitRoundoff, 1.0e-4,
                                                   2.0 * kUnitRoundoff};
constexpr ReductionCriterion kMixBlockF16Criterion{1.0 / 1024.0, 1.0e-4,
                                                   2.0 / 1024.0};

constexpr ReductionCriterion kMixInjectionFp32Criterion{8.0e-6, 1.0e-5, 2.0e-5};

ReductionCriterion norm_criterion(DType dtype) {
    return dtype == DType::FP16 ? kNormF16Criterion : kNormBf16Criterion;
}

PointwiseCriterion combine_criterion(DType dtype) {
    return dtype == DType::FP16 ? kCombineF16Criterion : kCombineBf16Criterion;
}

ReductionCriterion mix_block_criterion(DType dtype) {
    return dtype == DType::FP16 ? kMixBlockF16Criterion : kMixBlockBf16Criterion;
}

int run_expand_norm_case(const char* label, std::int32_t columns, std::uint32_t seed,
                         DType dtype = DType::BF16) {
    const std::size_t embedding_count = static_cast<std::size_t>(kHidden) * columns;
    const std::size_t weight_count    = static_cast<std::size_t>(kHidden) * kStreams;
    const std::size_t output_count    = static_cast<std::size_t>(kHidden) * kStreams * columns;

    std::vector<float> embedding(embedding_count);
    std::vector<float> norm_weight(weight_count);
    fill_uniform(embedding, seed, -4.0F, 4.0F);
    fill_uniform(norm_weight, seed + 1U, -0.5F, 0.5F);
    round_to_activation(embedding, dtype);
    round_to_bf16(norm_weight);
    embedding[0] = -0.0F;

    const std::vector<double> reference = expand_norm_oracle(embedding, norm_weight, columns);

    GuardedDeviceBuffer device_embedding(embedding_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_weight(weight_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_residual(output_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_normalized(output_count * sizeof(std::uint16_t));
    device_embedding.fill(0xff);
    device_weight.fill(0xff);
    device_residual.fill(0xff);
    device_normalized.fill(0xff);
    const std::vector<std::uint16_t> embedding_bits = encode_activation(embedding, dtype);
    const std::vector<std::uint16_t> weight_bits    = encode_bf16(norm_weight);
    device_embedding.copy_from_host(embedding_bits.data(), device_embedding.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());

    const Tensor embedding_tensor = activation_tensor(device_embedding.data(), dtype, kHidden, columns);
    const Tensor weight_tensor    = bf16_tensor(device_weight.data(), kHidden, kStreams);
    Tensor residual_tensor =
        activation_tensor(device_residual.data(), dtype, kHidden, kStreams, columns);
    Tensor normalized_tensor =
        activation_tensor(device_normalized.data(), dtype, kHidden, kStreams, columns);

    ops::hc_expand_norm(embedding_tensor, weight_tensor, kEpsilon, residual_tensor, normalized_tensor,
                        nullptr);
    cuda_synchronize();

    int failures = 0;
    const std::vector<std::uint16_t> residual_bits =
        from_device<std::uint16_t>(device_residual.data(), output_count);
    std::vector<std::uint16_t> expected_residual(output_count);
    for (std::int32_t m = 0; m < columns; ++m) {
        for (std::int32_t s = 0; s < kStreams; ++s) {
            for (std::int32_t d = 0; d < kHidden; ++d) {
                expected_residual[d + static_cast<std::size_t>(kHidden) * s +
                                 static_cast<std::size_t>(kHidden) * kStreams * m] =
                    embedding_bits[d + static_cast<std::size_t>(kHidden) * m];
            }
        }
    }
    failures += verify_exact((std::string(label) + " residual word copy").c_str(), residual_bits,
                             expected_residual);
    failures += verify_reduction(std::string(label) + " normalized",
                                 read_activation(device_normalized.data(), output_count, dtype),
                                 reference, norm_criterion(dtype));
    failures += device_residual.verify_guards(std::string(label) + " residual");
    failures += device_normalized.verify_guards(std::string(label) + " normalized");
    failures += verify_exact((std::string(label) + " embedding preserved").c_str(),
                             from_device<std::uint16_t>(device_embedding.data(), embedding_count),
                             embedding_bits);
    failures += verify_exact((std::string(label) + " weight preserved").c_str(),
                             from_device<std::uint16_t>(device_weight.data(), weight_count), weight_bits);
    return failures;
}

int run_grouped_rmsnorm_case(const char* label, std::int32_t columns, std::uint32_t seed,
                            DType dtype = DType::BF16) {
    const std::size_t residual_count = static_cast<std::size_t>(kHidden) * kStreams * columns;
    const std::size_t weight_count    = static_cast<std::size_t>(kHidden) * kStreams;

    std::vector<float> residual(residual_count);
    std::vector<float> norm_weight(weight_count);
    fill_uniform(residual, seed, -4.0F, 4.0F);
    fill_uniform(norm_weight, seed + 1U, -0.5F, 0.5F);
    round_to_activation(residual, dtype);
    round_to_bf16(norm_weight);

    const std::vector<double> reference = grouped_rmsnorm_oracle(residual, norm_weight, columns);

    GuardedDeviceBuffer device_residual(residual_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_weight(weight_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_normalized(residual_count * sizeof(std::uint16_t));
    device_normalized.fill(0xff);
    const std::vector<std::uint16_t> residual_bits = encode_activation(residual, dtype);
    const std::vector<std::uint16_t> weight_bits    = encode_bf16(norm_weight);
    device_residual.copy_from_host(residual_bits.data(), device_residual.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());

    const Tensor residual_tensor =
        activation_tensor(device_residual.data(), dtype, kHidden, kStreams, columns);
    const Tensor weight_tensor     = bf16_tensor(device_weight.data(), kHidden, kStreams);
    Tensor normalized_tensor =
        activation_tensor(device_normalized.data(), dtype, kHidden, kStreams, columns);

    ops::hc_grouped_rmsnorm(residual_tensor, weight_tensor, kEpsilon, normalized_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(std::string(label) + " normalized",
                                    read_activation(device_normalized.data(), residual_count, dtype),
                                    reference, norm_criterion(dtype));
    failures += device_normalized.verify_guards(std::string(label) + " normalized");
    failures += verify_exact((std::string(label) + " residual preserved").c_str(),
                             from_device<std::uint16_t>(device_residual.data(), residual_count),
                             residual_bits);
    failures += verify_exact((std::string(label) + " weight preserved").c_str(),
                             from_device<std::uint16_t>(device_weight.data(), weight_count), weight_bits);
    return failures;
}

int run_combine_case(const char* label, std::int32_t columns, std::uint32_t seed,
                     bool in_place, DType dtype = DType::BF16) {
    const std::size_t residual_count = static_cast<std::size_t>(kHidden) * kStreams * columns;
    const std::size_t output_count    = static_cast<std::size_t>(kHidden) * columns;
    const std::size_t injection_count = static_cast<std::size_t>(kStreams) * columns;

    std::vector<float> residual(residual_count);
    std::vector<float> block_output(output_count);
    std::vector<float> injection(injection_count);
    fill_uniform(residual, seed, -4.0F, 4.0F);
    fill_uniform(block_output, seed + 1U, -2.0F, 2.0F);
    fill_uniform(injection, seed + 2U, 0.0F, 2.0F);
    round_to_activation(residual, dtype);
    round_to_activation(block_output, dtype);

    const std::vector<double> reference = combine_oracle(residual, block_output, injection, columns);

    GuardedDeviceBuffer device_residual(residual_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_block_output(output_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_injection(injection_count * sizeof(float));
    GuardedDeviceBuffer device_updated(residual_count * sizeof(std::uint16_t));
    device_updated.fill(0xff);
    const std::vector<std::uint16_t> residual_bits = encode_activation(residual, dtype);
    const std::vector<std::uint16_t> block_bits    = encode_activation(block_output, dtype);
    device_residual.copy_from_host(residual_bits.data(), device_residual.bytes());
    device_block_output.copy_from_host(block_bits.data(), device_block_output.bytes());
    device_injection.copy_from_host(injection.data(), device_injection.bytes());

    const Tensor residual_tensor =
        activation_tensor(device_residual.data(), dtype, kHidden, kStreams, columns);
    const Tensor block_tensor =
        activation_tensor(device_block_output.data(), dtype, kHidden, columns);
    const Tensor injection_tensor(device_injection.data(), DType::FP32, {kStreams, columns});
    Tensor updated_tensor =
        in_place ? activation_tensor(device_residual.data(), dtype, kHidden, kStreams, columns)
                 : activation_tensor(device_updated.data(), dtype, kHidden, kStreams, columns);

    ops::hc_combine(residual_tensor, block_tensor, injection_tensor, updated_tensor, nullptr);
    cuda_synchronize();

    const void* updated_data = in_place ? device_residual.data() : device_updated.data();
    int failures = verify_pointwise(std::string(label) + " updated",
                                    read_activation(updated_data, residual_count, dtype), reference,
                                    combine_criterion(dtype));
    failures += device_updated.verify_guards(std::string(label) + " updated");
    failures += verify_exact((std::string(label) + " block output preserved").c_str(),
                             from_device<std::uint16_t>(device_block_output.data(), output_count),
                             block_bits);
    failures += verify_exact((std::string(label) + " injection preserved").c_str(),
                             from_device<float>(device_injection.data(), injection_count), injection);
    if (!in_place) {
        failures += verify_exact((std::string(label) + " residual preserved").c_str(),
                                 from_device<std::uint16_t>(device_residual.data(), residual_count),
                                 residual_bits);
    }
    return failures;
}

int run_combine_norm_case(const char* label, std::int32_t columns, std::uint32_t seed,
                          bool in_place, DType dtype = DType::BF16) {
    const std::size_t residual_count = static_cast<std::size_t>(kHidden) * kStreams * columns;
    const std::size_t output_count    = static_cast<std::size_t>(kHidden) * columns;
    const std::size_t injection_count = static_cast<std::size_t>(kStreams) * columns;
    const std::size_t weight_count    = static_cast<std::size_t>(kHidden) * kStreams;

    std::vector<float> residual(residual_count);
    std::vector<float> block_output(output_count);
    std::vector<float> injection(injection_count);
    std::vector<float> next_weight(weight_count);
    fill_uniform(residual, seed, -4.0F, 4.0F);
    fill_uniform(block_output, seed + 1U, -2.0F, 2.0F);
    fill_uniform(injection, seed + 2U, 0.0F, 2.0F);
    fill_uniform(next_weight, seed + 3U, -0.5F, 0.5F);
    round_to_activation(residual, dtype);
    round_to_activation(block_output, dtype);
    round_to_bf16(next_weight);

    const std::vector<double> combined = combine_oracle(residual, block_output, injection, columns);
    std::vector<float> combined_rounded(combined.size());
    for (std::size_t i = 0; i < combined.size(); ++i) {
        combined_rounded[i] = to_activation(static_cast<float>(combined[i]), dtype);
    }
    const std::vector<double> reference =
        grouped_rmsnorm_oracle(combined_rounded, next_weight, columns);

    GuardedDeviceBuffer device_residual(residual_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_block_output(output_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_injection(injection_count * sizeof(float));
    GuardedDeviceBuffer device_weight(weight_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_updated(residual_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_normalized(residual_count * sizeof(std::uint16_t));
    device_updated.fill(0xff);
    device_normalized.fill(0xff);
    const std::vector<std::uint16_t> residual_bits = encode_activation(residual, dtype);
    const std::vector<std::uint16_t> block_bits    = encode_activation(block_output, dtype);
    const std::vector<std::uint16_t> weight_bits   = encode_bf16(next_weight);
    device_residual.copy_from_host(residual_bits.data(), device_residual.bytes());
    device_block_output.copy_from_host(block_bits.data(), device_block_output.bytes());
    device_injection.copy_from_host(injection.data(), device_injection.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());

    const Tensor residual_tensor =
        activation_tensor(device_residual.data(), dtype, kHidden, kStreams, columns);
    const Tensor block_tensor =
        activation_tensor(device_block_output.data(), dtype, kHidden, columns);
    const Tensor injection_tensor(device_injection.data(), DType::FP32, {kStreams, columns});
    const Tensor weight_tensor   = bf16_tensor(device_weight.data(), kHidden, kStreams);
    Tensor updated_tensor =
        in_place ? activation_tensor(device_residual.data(), dtype, kHidden, kStreams, columns)
                 : activation_tensor(device_updated.data(), dtype, kHidden, kStreams, columns);
    Tensor normalized_tensor =
        activation_tensor(device_normalized.data(), dtype, kHidden, kStreams, columns);

    ops::hc_combine_norm(residual_tensor, block_tensor, injection_tensor, weight_tensor, kEpsilon,
                         updated_tensor, normalized_tensor, nullptr);
    cuda_synchronize();

    const void* updated_data = in_place ? device_residual.data() : device_updated.data();
    int failures = verify_pointwise(std::string(label) + " updated",
                                    read_activation(updated_data, residual_count, dtype), combined,
                                    combine_criterion(dtype));
    failures += verify_reduction(std::string(label) + " normalized",
                                 read_activation(device_normalized.data(), residual_count, dtype),
                                 reference, norm_criterion(dtype));
    failures += device_normalized.verify_guards(std::string(label) + " normalized");
    if (!in_place) {
        failures += device_updated.verify_guards(std::string(label) + " updated");
        failures += verify_exact((std::string(label) + " residual preserved").c_str(),
                                 from_device<std::uint16_t>(device_residual.data(), residual_count),
                                 residual_bits);
    }
    failures += verify_exact((std::string(label) + " weight preserved").c_str(),
                             from_device<std::uint16_t>(device_weight.data(), weight_count), weight_bits);
    return failures;
}

std::vector<float> weight_floats(const direct_bf16_weight::HostWeight& weight) {
    std::vector<float> values(weight.bits.size());
    for (std::size_t i = 0; i < weight.bits.size(); ++i) { values[i] = bf16_to_f32(weight.bits[i]); }
    return values;
}

std::vector<float> weight_floats(const direct_fp16_weight::HostWeight& weight) {
    std::vector<float> values(weight.bits.size());
    for (std::size_t i = 0; i < weight.bits.size(); ++i) {
        values[i] = direct_fp16_weight::fp16_bits_to_f32(weight.bits[i]);
    }
    return values;
}

struct MixWeights {
    Weight down;
    Weight up;
    std::vector<float> down_values;
    std::vector<float> up_values;
    std::optional<direct_bf16_weight::DeviceWeight> down_bf16;
    std::optional<direct_bf16_weight::DeviceWeight> up_bf16;
    std::optional<direct_fp16_weight::DeviceWeight> down_fp16;
    std::optional<direct_fp16_weight::DeviceWeight> up_fp16;

    int verify_preserved(const std::string& label) const {
        if (down_fp16.has_value()) {
            return down_fp16->verify_preserved(label + " down weight") +
                   up_fp16->verify_preserved(label + " up weight");
        }
        return down_bf16->verify_preserved(label + " down weight") +
               up_bf16->verify_preserved(label + " up weight");
    }
};

MixWeights make_mix_weights(std::int32_t down_rows, std::int32_t channels, std::uint32_t seed,
                            DType dtype) {
    MixWeights weights;
    if (dtype == DType::FP16) {
        weights.down_fp16.emplace(direct_fp16_weight::make_patterned(down_rows, channels, seed));
        weights.up_fp16.emplace(direct_fp16_weight::make_patterned(channels, kRank, seed + 1U));
        weights.down = weights.down_fp16->view();
        weights.up = weights.up_fp16->view();
        weights.down_values = weight_floats(weights.down_fp16->host);
        weights.up_values = weight_floats(weights.up_fp16->host);
    } else {
        weights.down_bf16.emplace(direct_bf16_weight::make_patterned(down_rows, channels, seed));
        weights.up_bf16.emplace(direct_bf16_weight::make_patterned(channels, kRank, seed + 1U));
        weights.down = weights.down_bf16->view();
        weights.up = weights.up_bf16->view();
        weights.down_values = weight_floats(weights.down_bf16->host);
        weights.up_values = weight_floats(weights.up_bf16->host);
    }
    return weights;
}

int run_mix_case(const char* label, std::int32_t columns, std::uint32_t seed, bool final_mix,
                 DType dtype = DType::BF16) {
    const std::int32_t channels  = kHidden * kStreams;
    const std::size_t normal_count = static_cast<std::size_t>(kHidden) * kStreams * columns;
    const std::size_t block_count   = static_cast<std::size_t>(kHidden) * columns;
    const std::size_t injection_count = static_cast<std::size_t>(kStreams) * columns;

    std::vector<float> normalized(normal_count);
    fill_uniform(normalized, seed, -1.0F, 1.0F);
    round_to_activation(normalized, dtype);

    const MixWeights weights =
        make_mix_weights(final_mix ? kRank : kRank + kStreams, channels, seed + 1U, dtype);
    const MixReference reference =
        mix_oracle(normalized, weights.down_values, weights.up_values, columns, !final_mix);

    GuardedDeviceBuffer device_normalized(normal_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_block(block_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_injection(injection_count * sizeof(float));
    device_block.fill(0xff);
    device_injection.fill(0xff);
    const std::vector<std::uint16_t> normalized_bits = encode_activation(normalized, dtype);
    device_normalized.copy_from_host(normalized_bits.data(), device_normalized.bytes());

    const Tensor normalized_tensor =
        activation_tensor(device_normalized.data(), dtype, kHidden, kStreams, columns);
    Tensor block_tensor = activation_tensor(device_block.data(), dtype, kHidden, columns);
    Tensor injection_tensor(device_injection.data(), DType::FP32, {kStreams, columns});

    const std::size_t capacity =
        final_mix
            ? ops::hc_final_mix_workspace_capacity_bytes(
                  ops::HcFinalMixWeights{weights.down, weights.up}, 1, columns)
            : ops::hc_mix_workspace_capacity_bytes(ops::HcMixWeights{weights.down, weights.up}, 1,
                                                   columns);
    WorkspaceArena workspace(capacity);

    if (final_mix) {
        ops::hc_final_mix(normalized_tensor,
                          ops::HcFinalMixWeights{weights.down, weights.up}, workspace,
                          block_tensor, nullptr);
    } else {
        ops::hc_mix(normalized_tensor, ops::HcMixWeights{weights.down, weights.up}, workspace,
                    block_tensor, injection_tensor, nullptr);
    }
    cuda_synchronize();

    int failures = verify_reduction(std::string(label) + " block input",
                                    read_activation(device_block.data(), block_count, dtype),
                                    reference.block_input, mix_block_criterion(dtype));
    if (!final_mix) {
        std::vector<double> injection_actual(injection_count);
        const std::vector<float> injection_raw = from_device<float>(device_injection.data(),
                                                                    injection_count);
        for (std::size_t i = 0; i < injection_count; ++i) {
            injection_actual[i] = static_cast<double>(injection_raw[i]);
        }
        failures += verify_reduction(std::string(label) + " injection", injection_actual,
                                     reference.injection, kMixInjectionFp32Criterion);
        failures += device_injection.verify_guards(std::string(label) + " injection");
    }
    failures += device_block.verify_guards(std::string(label) + " block input");
    failures += verify_exact((std::string(label) + " normalized preserved").c_str(),
                             from_device<std::uint16_t>(device_normalized.data(), normal_count),
                             normalized_bits);
    failures += weights.verify_preserved(label);
    if (workspace.peak_used() != capacity || workspace.used() != 0) {
        std::cerr << label << ": workspace high-water " << workspace.peak_used() << " != " << capacity
                  << '\n';
        ++failures;
    }
    return failures;
}

int verify_graph_capture_case(const char* label, std::uint32_t seed) {
    constexpr std::int32_t columns = 3;
    const std::size_t embedding_count = static_cast<std::size_t>(kHidden) * columns;
    const std::size_t weight_count    = static_cast<std::size_t>(kHidden) * kStreams;
    const std::size_t output_count    = static_cast<std::size_t>(kHidden) * kStreams * columns;

    std::vector<float> embedding(embedding_count);
    std::vector<float> norm_weight(weight_count);
    fill_uniform(embedding, seed, -4.0F, 4.0F);
    fill_uniform(norm_weight, seed + 1U, -0.5F, 0.5F);
    round_to_bf16(embedding);
    round_to_bf16(norm_weight);

    DeviceBuffer device_embedding = to_device_bf16(embedding);
    DeviceBuffer device_weight    = to_device_bf16(norm_weight);
    GuardedDeviceBuffer device_residual(output_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_normalized(output_count * sizeof(std::uint16_t));

    const Tensor embedding_tensor = bf16_tensor(device_embedding.p, kHidden, columns);
    const Tensor weight_tensor    = bf16_tensor(device_weight.p, kHidden, kStreams);
    Tensor residual_tensor         = bf16_tensor(device_residual.data(), kHidden, kStreams, columns);
    Tensor normalized_tensor       = bf16_tensor(device_normalized.data(), kHidden, kStreams, columns);

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    int failures = 0;
    ops::hc_expand_norm(embedding_tensor, weight_tensor, kEpsilon, residual_tensor, normalized_tensor,
                        stream);
    cuda_synchronize(stream);

    cudaGraph_t graph       = nullptr;
    cudaGraphExec_t program = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture begin");
    ops::hc_expand_norm(embedding_tensor, weight_tensor, kEpsilon, residual_tensor, normalized_tensor,
                        stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "capture end");
    cuda_check(cudaGraphInstantiate(&program, graph, nullptr, nullptr, 0), "graph instantiate");

    std::vector<float> replayed(embedding_count);
    fill_uniform(replayed, seed + 7U, -4.0F, 4.0F);
    round_to_bf16(replayed);
    const DeviceBuffer device_replayed = to_device_bf16(replayed);
    cuda_check(cudaMemcpy(device_embedding.p, device_replayed.p, device_embedding.bytes,
                          cudaMemcpyDeviceToDevice),
               "cudaMemcpy device-to-device");
    cuda_check(cudaGraphLaunch(program, stream), "graph launch");
    cuda_synchronize(stream);

    const std::vector<double> reference = expand_norm_oracle(replayed, norm_weight, columns);
    failures += verify_reduction(std::string(label) + " normalized",
                                 read_activation(device_normalized.data(), output_count,
                                                 DType::BF16),
                                 reference, kNormBf16Criterion);
    failures += device_normalized.verify_guards(std::string(label) + " normalized");

    cuda_check(cudaGraphExecDestroy(program), "graph exec destroy");
    cuda_check(cudaGraphDestroy(graph), "graph destroy");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return failures;
}

#ifdef NINFER_VOLTA_BUILD
int verify_graph_capture_mix_case(const char* label, std::uint32_t seed) {
    constexpr std::int32_t columns = 3;
    const std::int32_t channels    = kHidden * kStreams;
    const std::size_t normal_count  = static_cast<std::size_t>(kHidden) * kStreams * columns;
    const std::size_t block_count   = static_cast<std::size_t>(kHidden) * columns;
    const std::size_t injection_count = static_cast<std::size_t>(kStreams) * columns;

    std::vector<float> normalized(normal_count);
    fill_uniform(normalized, seed, -1.0F, 1.0F);
    round_to_bf16(normalized);

    direct_bf16_weight::DeviceWeight down(
        direct_bf16_weight::make_patterned(kRank + kStreams, channels, seed + 1U));
    direct_bf16_weight::DeviceWeight up(direct_bf16_weight::make_patterned(channels, kRank, seed + 2U));

    DeviceBuffer device_normalized = to_device_bf16(normalized);
    GuardedDeviceBuffer device_block(block_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_injection(injection_count * sizeof(float));

    const Tensor normalized_tensor = bf16_tensor(device_normalized.p, kHidden, kStreams, columns);
    Tensor block_tensor             = bf16_tensor(device_block.data(), kHidden, columns);
    Tensor injection_tensor(device_injection.data(), DType::FP32, {kStreams, columns});

    const std::size_t capacity =
        ops::hc_mix_workspace_capacity_bytes(ops::HcMixWeights{down.view(), up.view()}, 1, columns);
    WorkspaceArena workspace(capacity);

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    int failures = 0;
    ops::hc_mix(normalized_tensor, ops::HcMixWeights{down.view(), up.view()}, workspace, block_tensor,
                injection_tensor, stream);
    cuda_synchronize(stream);

    cudaGraph_t graph       = nullptr;
    cudaGraphExec_t program = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture begin");
    ops::hc_mix(normalized_tensor, ops::HcMixWeights{down.view(), up.view()}, workspace, block_tensor,
                injection_tensor, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "capture end");
    cuda_check(cudaGraphInstantiate(&program, graph, nullptr, nullptr, 0), "graph instantiate");

    std::vector<float> replayed(normal_count);
    fill_uniform(replayed, seed + 7U, -1.0F, 1.0F);
    round_to_bf16(replayed);
    const DeviceBuffer device_replayed = to_device_bf16(replayed);
    cuda_check(cudaMemcpy(device_normalized.p, device_replayed.p, device_normalized.bytes,
                          cudaMemcpyDeviceToDevice),
               "cudaMemcpy device-to-device");
    cuda_check(cudaGraphLaunch(program, stream), "graph launch");
    cuda_synchronize(stream);

    const MixReference reference =
        mix_oracle(replayed, weight_floats(down.host), weight_floats(up.host), columns, true);
    failures += verify_reduction(std::string(label) + " block input",
                                 from_device_bf16(device_block.data(), block_count),
                                 reference.block_input, kMixBlockBf16Criterion);
    std::vector<double> injection_actual(injection_count);
    const std::vector<float> injection_raw =
        from_device<float>(device_injection.data(), injection_count);
    for (std::size_t i = 0; i < injection_count; ++i) {
        injection_actual[i] = static_cast<double>(injection_raw[i]);
    }
    failures += verify_reduction(std::string(label) + " injection", injection_actual,
                                 reference.injection, kMixInjectionFp32Criterion);

    cuda_check(cudaGraphExecDestroy(program), "graph exec destroy");
    cuda_check(cudaGraphDestroy(graph), "graph destroy");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return failures;
}
#endif

int verify_workspace_contract() {
    int failures = 0;

    bool threw = false;
    try {
        (void)ops::hc_mix_workspace_capacity_bytes(ops::HcMixWeights{}, 0, 4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "hc mix workspace: an invalid token interval was accepted\n";
        ++failures;
    }

    threw = false;
    try {
        (void)ops::hc_final_mix_workspace_capacity_bytes(ops::HcFinalMixWeights{}, 4, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "hc final mix workspace: an inverted token interval was accepted\n";
        ++failures;
    }

    threw = false;
    try {
        (void)ops::hc_mix_workspace_capacity_bytes(ops::HcMixWeights{}, 1, 4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "hc mix workspace: an empty weight profile was accepted\n";
        ++failures;
    }

    Weight down{};
    down.qtype                = QType::BF16_CTRL;
    down.layout               = QuantLayout::Contiguous;
    down.ndim                 = 2;
    down.n                    = kRank + kStreams;
    down.k                    = kHidden * kStreams;
    down.shape[0]             = down.n;
    down.shape[1]             = down.k;
    down.padded_shape[0]      = down.n;
    down.padded_shape[1]      = down.k;
    Weight up{};
    up.qtype                  = QType::BF16_CTRL;
    up.layout                 = QuantLayout::Contiguous;
    up.ndim                   = 2;
    up.n                      = kHidden * kStreams;
    up.k                      = kRank;
    up.shape[0]               = up.n;
    up.shape[1]               = up.k;
    up.padded_shape[0]        = up.n;
    up.padded_shape[1]        = up.k;
    if (ops::hc_mix_workspace_capacity_bytes(ops::HcMixWeights{down, up}, 1, 7) == 0) {
        std::cerr << "hc mix workspace: metadata-only weights were rejected\n";
        ++failures;
    }
    down.n               = kRank;
    down.shape[0]        = kRank;
    down.padded_shape[0] = kRank;
    if (ops::hc_final_mix_workspace_capacity_bytes(ops::HcFinalMixWeights{down, up}, 1, 7) == 0) {
        std::cerr << "hc final mix workspace: metadata-only weights were rejected\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_expand_norm_case("hc_expand_norm M=1", 1, 2101U);
    failures += run_expand_norm_case("hc_expand_norm M=7", 7, 2102U);
    failures += run_grouped_rmsnorm_case("hc_grouped_rmsnorm M=1", 1, 2201U);
    failures += run_grouped_rmsnorm_case("hc_grouped_rmsnorm M=7", 7, 2202U);
    failures += run_combine_case("hc_combine M=1", 1, 2301U, false);
    failures += run_combine_case("hc_combine M=1 in place", 1, 2302U, true);
    failures += run_combine_case("hc_combine M=7", 7, 2303U, false);
    failures += run_combine_case("hc_combine M=7 in place", 7, 2304U, true);
    failures += run_combine_norm_case("hc_combine_norm M=1", 1, 2401U, false);
    failures += run_combine_norm_case("hc_combine_norm M=1 in place", 1, 2403U, true);
    failures += run_combine_norm_case("hc_combine_norm M=7", 7, 2402U, false);
    failures += run_mix_case("hc_mix M=1", 1, 2501U, false);
    failures += run_mix_case("hc_final_mix M=1", 1, 2601U, true);
#ifdef NINFER_VOLTA_BUILD
    failures += run_mix_case("hc_mix M=7", 7, 2502U, false);
    failures += run_mix_case("hc_final_mix M=7", 7, 2602U, true);
#endif
    failures += run_expand_norm_case("hc_expand_norm f16 M=1", 1, 3101U, DType::FP16);
    failures += run_expand_norm_case("hc_expand_norm f16 M=7", 7, 3102U, DType::FP16);
    failures += run_grouped_rmsnorm_case("hc_grouped_rmsnorm f16 M=1", 1, 3201U, DType::FP16);
    failures += run_grouped_rmsnorm_case("hc_grouped_rmsnorm f16 M=7", 7, 3202U, DType::FP16);
    failures += run_combine_case("hc_combine f16 M=1", 1, 3301U, false, DType::FP16);
    failures += run_combine_case("hc_combine f16 M=1 in place", 1, 3302U, true, DType::FP16);
    failures += run_combine_case("hc_combine f16 M=7", 7, 3303U, false, DType::FP16);
    failures += run_combine_case("hc_combine f16 M=7 in place", 7, 3304U, true, DType::FP16);
    failures += run_combine_norm_case("hc_combine_norm f16 M=1", 1, 3401U, false, DType::FP16);
    failures += run_combine_norm_case("hc_combine_norm f16 M=1 in place", 1, 3402U, true,
                                      DType::FP16);
    failures += run_combine_norm_case("hc_combine_norm f16 M=7", 7, 3403U, false, DType::FP16);
    failures += run_combine_norm_case("hc_combine_norm f16 M=7 in place", 7, 3404U, true,
                                      DType::FP16);
#ifdef NINFER_VOLTA_BUILD
    failures += run_mix_case("hc_mix f16 M=1", 1, 3501U, false, DType::FP16);
    failures += run_mix_case("hc_final_mix f16 M=1", 1, 3601U, true, DType::FP16);
    failures += run_mix_case("hc_mix f16 M=7", 7, 3502U, false, DType::FP16);
    failures += run_mix_case("hc_final_mix f16 M=7", 7, 3602U, true, DType::FP16);
#endif
    failures += verify_graph_capture_case("hc_expand_norm graph", 2701U);
#ifdef NINFER_VOLTA_BUILD
    failures += verify_graph_capture_mix_case("hc_mix graph", 2801U);
#endif
    failures += verify_workspace_contract();

    std::cout << (failures ? "FAIL" : "OK") << " gated_residual\n";
    return failures ? 1 : 0;
}
