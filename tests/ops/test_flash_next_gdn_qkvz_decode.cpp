// Oracle test for the Flash-Next fused GDN qkvz W8 decode kernel
// (ops/gdn_input_proj/w8/w8_gdn_qkvz_decode.cu): one launch over the whole per-rank
// [8192, 2560] W8G32_F16S parent, writing q/k/v/z as four independent [1024|1024|3072|3072, 1]
// tensors in the activation dtype.
// The oracle is an independent naive FP64 evaluation of the logical formula over the exact
// stored codes and scales (tests/ops/quantized_weight.h). The comparison criterion is the one
// the A16 GDN input projection route is held to: the outputs are BF16/FP16 and the reduction
// grouping belongs to the production profile, not the oracle.
#include "ninfer/ops/linear.h"

#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

namespace {

// Same criterion as the A16 GDN input projection route: BF16/FP16 outputs of an
// exact-decode/FP32 production reduction against the FP64 naive oracle.
constexpr ReductionCriterion kFusedQkvzA16Tolerance{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kFusedGatingFp32{1.4e-6, 5.0e-7, 2.5e-6};
constexpr ReductionCriterion kFusedGatingF16{5.0e-4, 1.0e-4, 8.0e-4};

constexpr std::int32_t kHeads  = 48;
constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kQRows  = 1024;
constexpr std::int32_t kKRows  = 1024;
constexpr std::int32_t kVRows  = 3072;
constexpr std::int32_t kZRows  = 3072;
constexpr std::int32_t kRows   = kQRows + kKRows + kVRows + kZRows;  // 8192

template <DType kDtype>
std::vector<double> output_values(const DeviceBuffer& device, std::size_t count);

template <>
std::vector<double> output_values<DType::BF16>(const DeviceBuffer& device, std::size_t count) {
    return from_device_bf16(device, count);
}

template <>
std::vector<double> output_values<DType::FP16>(const DeviceBuffer& device, std::size_t count) {
    return from_device_f16(device, count);
}

template <DType kDtype>
std::vector<std::uint16_t> activation_bits(const std::vector<float>& values);

template <>
std::vector<std::uint16_t> activation_bits<DType::BF16>(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

template <>
std::vector<std::uint16_t> activation_bits<DType::FP16>(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_f16(values[i]);
    return bits;
}

template <DType kDtype>
void round_activation(std::vector<float>& values);

template <>
void round_activation<DType::BF16>(std::vector<float>& values) {
    round_to_bf16(values);
}

template <>
void round_activation<DType::FP16>(std::vector<float>& values) {
    round_to_f16(values);
}

template <DType kDtype>
int run_case(std::uint32_t seed) {
    constexpr const char* kLabel = kDtype == DType::BF16 ? "bf16" : "f16";
    qw::PatternedWeightOptions options;
    options.row_split_scale         = qw::RowSplitScalePattern::Small;
    options.row_split_codes         = qw::RowSplitCodePattern::Coordinate;
    options.decorrelate_coordinates = true;
    const qw::PackedWeight packed =
        qw::make_patterned_weight(QType::W8G32_F16S, kRows, kHidden, seed, options);
    DevicePackedWeight weight(packed);

    std::vector<float> activation     = make_bf16_activation(kHidden, 1, seed + 7U);
    round_activation<kDtype>(activation);
    const std::vector<std::uint16_t> bits = activation_bits<kDtype>(activation);
    DeviceBuffer device_activation    = to_device(bits);

    DeviceBuffer device_query(static_cast<std::size_t>(kQRows) * 2);
    DeviceBuffer device_key(static_cast<std::size_t>(kKRows) * 2);
    DeviceBuffer device_value(static_cast<std::size_t>(kVRows) * 2);
    DeviceBuffer device_z(static_cast<std::size_t>(kZRows) * 2);
    // Poison the outputs: an element the kernel never writes stays non-finite and fails every
    // comparison below.
    device_query.fill(0xff);
    device_key.fill(0xff);
    device_value.fill(0xff);
    device_z.fill(0xff);

    std::vector<float> control_host(static_cast<std::size_t>(2 * kHeads) * kHidden);
    for (std::size_t i = 0; i < control_host.size(); ++i) {
        control_host[i] = std::sin(static_cast<double>(i) * 0.001) * 0.05F;
    }
    DeviceBuffer device_control = to_device_bf16(control_host);
    const std::vector<double> control_values = from_device_bf16(device_control, control_host.size());

    std::vector<float> a_log_host(kHeads), dt_bias_host(kHeads);
    for (int h = 0; h < kHeads; ++h) {
        a_log_host[h]   = -2.0F + static_cast<float>(h) * 0.01F;
        dt_bias_host[h] = -1.0F + static_cast<float>(h) * 0.02F;
    }
    DeviceBuffer device_a_log   = to_device_f32(a_log_host);
    DeviceBuffer device_dt_bias = to_device_f32(dt_bias_host);
    DeviceBuffer device_g(static_cast<std::size_t>(kHeads) * sizeof(float));
    DeviceBuffer device_beta(static_cast<std::size_t>(kHeads) * sizeof(float));
    device_g.fill(0xff);
    device_beta.fill(0xff);

    Weight control{};
    control.qtype           = QType::BF16_CTRL;
    control.layout          = QuantLayout::Contiguous;
    control.payload         = device_control.p;
    control.payload_bytes   = device_control.bytes;
    control.qdata           = device_control.p;
    control.ndim            = 2;
    control.shape[0]        = 2 * kHeads;
    control.shape[1]        = kHidden;
    control.padded_shape[0] = 2 * kHeads;
    control.padded_shape[1] = kHidden;
    control.n               = 2 * kHeads;
    control.k               = kHidden;

    Tensor x(device_activation.p, kDtype, {kHidden, 1});
    Tensor query(device_query.p, kDtype, {kQRows, 1});
    Tensor key(device_key.p, kDtype, {kKRows, 1});
    Tensor value(device_value.p, kDtype, {kVRows, 1});
    Tensor z(device_z.p, kDtype, {kZRows, 1});
    Tensor a_log(device_a_log.p, DType::FP32, {kHeads, 1});
    Tensor dt_bias(device_dt_bias.p, DType::FP32, {kHeads, 1});
    Tensor g(device_g.p, DType::FP32, {kHeads, 1});
    Tensor beta(device_beta.p, DType::FP32, {kHeads, 1});

    ops::detail::w8_gdn_qkvz_decode_launch(x, weight.view(), query, key, value, z, control, a_log,
                                            dt_bias, g, beta, nullptr);
    cuda_synchronize();

    const std::string suffix = std::string(" fused qkvz ") + kLabel + " T=1 seed=" +
                               std::to_string(seed);
    int failures = verify_preserved("qkvz x" + suffix, device_activation, bits);
    failures += weight.verify_preserved("qkvz parent weight" + suffix);

    const auto check_rows = [&](std::string_view label, const DeviceBuffer& device,
                                std::int32_t row_offset, std::int32_t rows) {
        std::vector<double> expected;
        expected.reserve(static_cast<std::size_t>(rows));
        for (std::int32_t row = 0; row < rows; ++row) {
            expected.push_back(qw::dot_fp64(packed, row_offset + row, activation.data(), kHidden));
        }
        const std::vector<double> actual = output_values<kDtype>(device, static_cast<std::size_t>(rows));
        return compare(label, actual, expected, kFusedQkvzA16Tolerance);
    };
    failures += check_rows("qkvz q" + suffix, device_query, 0, kQRows);
    failures += check_rows("qkvz k" + suffix, device_key, kQRows, kKRows);
    failures += check_rows("qkvz v" + suffix, device_value, kQRows + kKRows, kVRows);
    failures += check_rows("qkvz z" + suffix, device_z, kQRows + kKRows + kVRows, kZRows);

    std::vector<double> expected_g(kHeads), expected_beta(kHeads);
    for (int h = 0; h < kHeads; ++h) {
        double a = 0.0, b = 0.0;
        for (int k = 0; k < kHidden; ++k) {
            const double xv = activation[static_cast<std::size_t>(k)];
            a += control_values[static_cast<std::size_t>(h) * kHidden + k] * xv;
            b += control_values[static_cast<std::size_t>(kHeads + h) * kHidden + k] * xv;
        }
        const double sp = std::max(a + dt_bias_host[h], 0.0) +
                          std::log1p(std::exp(-std::abs(a + dt_bias_host[h])));
        expected_g[h]    = -std::exp(a_log_host[h]) * sp;
        const double bv  = b;
        expected_beta[h] = bv >= 0.0 ? 1.0 / (1.0 + std::exp(-bv)) : std::exp(bv) / (1.0 + std::exp(bv));
    }
    const auto gating_criterion = kDtype == DType::BF16 ? kFusedGatingFp32 : kFusedGatingF16;
    failures += compare(("qkvz g" + suffix).c_str(), from_device_f32(device_g, kHeads),
                        expected_g, gating_criterion);
    failures += compare(("qkvz beta" + suffix).c_str(), from_device_f32(device_beta, kHeads),
                        expected_beta, gating_criterion);
    return failures;
}

template <typename T>
std::vector<T> guarded_bits(const GuardedDeviceBuffer& buffer, std::size_t count) {
    return from_device<T>(buffer.data(), count);
}

template <typename T>
std::vector<T> concatenate_columns(
    const std::vector<std::vector<T>>& single, std::int32_t tokens, std::size_t rows) {
    std::vector<T> joined(static_cast<std::size_t>(tokens) * rows);
    for (std::int32_t token = 0; token < tokens; ++token) {
        if (single[static_cast<std::size_t>(token)].size() != rows) {
            throw std::runtime_error("single-token output size mismatch");
        }
        std::copy(single[static_cast<std::size_t>(token)].begin(),
                  single[static_cast<std::size_t>(token)].end(),
                  joined.begin() + static_cast<std::size_t>(token) * rows);
    }
    return joined;
}

int check_finite_bits(std::string_view label, const std::vector<std::uint16_t>& bits) {
    std::size_t poisoned = 0;
    for (std::uint16_t bit : bits) { poisoned += static_cast<std::size_t>(bit == 0xffffu); }
    if (poisoned == 0) { return 0; }
    std::cerr << label << ": " << poisoned << " poisoned 0xffff elements remain\n";
    return 1;
}

int check_finite_f32_bits(std::string_view label, const std::vector<std::uint32_t>& bits) {
    std::size_t poisoned = 0;
    for (std::uint32_t bit : bits) {
        poisoned += static_cast<std::size_t>((bit & 0x7f800000u) == 0x7f800000u);
    }
    if (poisoned == 0) { return 0; }
    std::cerr << label << ": " << poisoned << " non-finite fp32 outputs remain\n";
    return 1;
}

// The multi-column kernel must be bit-identical to sequential T=1 launches for every
// q/k/v/z/g/beta output, at real per-rank geometry, for both activation dtypes. The
// oracle here is the T=1 production path itself: no tolerance, raw stored bits only.
template <DType kDtype>
int run_multi_column_case(std::int32_t tokens, std::uint32_t seed) {
    constexpr const char* kLabel = kDtype == DType::BF16 ? "bf16" : "f16";
    const std::string suffix = std::string(" multi T=") + std::to_string(tokens) + " " + kLabel +
                               " seed=" + std::to_string(seed);

    qw::PatternedWeightOptions options;
    options.row_split_scale         = qw::RowSplitScalePattern::Small;
    options.row_split_codes         = qw::RowSplitCodePattern::Coordinate;
    options.decorrelate_coordinates = true;
    const qw::PackedWeight packed =
        qw::make_patterned_weight(QType::W8G32_F16S, kRows, kHidden, seed, options);
    DevicePackedWeight weight(packed);

    std::vector<float> activation_host = make_bf16_activation(kHidden, tokens, seed + 71U);
    round_activation<kDtype>(activation_host);
    const std::vector<std::uint16_t> activation_u16 = activation_bits<kDtype>(activation_host);
    GuardedDeviceBuffer device_activation(activation_u16.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation_u16.data(), activation_u16.size() * sizeof(std::uint16_t));
    Tensor x(device_activation.data(), kDtype, {kHidden, tokens});

    std::vector<float> control_host(static_cast<std::size_t>(2 * kHeads) * kHidden);
    for (std::size_t i = 0; i < control_host.size(); ++i) {
        control_host[i] = std::sin(static_cast<double>(i) * 0.001) * 0.05F;
    }
    std::vector<std::uint16_t> control_bits(control_host.size());
    for (std::size_t i = 0; i < control_host.size(); ++i) {
        control_bits[i] = f32_to_bf16(control_host[i]);
    }
    GuardedDeviceBuffer device_control(control_bits.size() * sizeof(std::uint16_t));
    device_control.copy_from_host(control_bits.data(), control_bits.size() * sizeof(std::uint16_t));

    std::vector<float> a_log_host(kHeads), dt_bias_host(kHeads);
    for (int h = 0; h < kHeads; ++h) {
        a_log_host[h]   = -2.0F + static_cast<float>(h) * 0.01F;
        dt_bias_host[h] = -1.0F + static_cast<float>(h) * 0.02F;
    }
    DeviceBuffer device_a_log   = to_device_f32(a_log_host);
    DeviceBuffer device_dt_bias = to_device_f32(dt_bias_host);

    Weight control{};
    control.qtype           = QType::BF16_CTRL;
    control.layout          = QuantLayout::Contiguous;
    control.payload         = device_control.data();
    control.payload_bytes   = device_control.bytes();
    control.qdata           = device_control.data();
    control.ndim            = 2;
    control.shape[0]        = 2 * kHeads;
    control.shape[1]        = kHidden;
    control.padded_shape[0] = 2 * kHeads;
    control.padded_shape[1] = kHidden;
    control.n               = 2 * kHeads;
    control.k               = kHidden;

    Tensor a_log(device_a_log.p, DType::FP32, {kHeads, 1});
    Tensor dt_bias(device_dt_bias.p, DType::FP32, {kHeads, 1});

    constexpr std::array<std::int32_t, 4> rows{kQRows, kKRows, kVRows, kZRows};
    constexpr std::array<const char*, 4> names{"q", "k", "v", "z"};

    std::array<GuardedDeviceBuffer, 4> multi(
        {GuardedDeviceBuffer(static_cast<std::size_t>(kQRows) * tokens * 2),
         GuardedDeviceBuffer(static_cast<std::size_t>(kKRows) * tokens * 2),
         GuardedDeviceBuffer(static_cast<std::size_t>(kVRows) * tokens * 2),
         GuardedDeviceBuffer(static_cast<std::size_t>(kZRows) * tokens * 2)});
    GuardedDeviceBuffer multi_g(static_cast<std::size_t>(kHeads) * tokens * sizeof(float));
    GuardedDeviceBuffer multi_beta(static_cast<std::size_t>(kHeads) * tokens * sizeof(float));
    for (auto& buffer : multi) { buffer.fill(0xff); }
    multi_g.fill(0xff);
    multi_beta.fill(0xff);

    Tensor multi_q(multi[0].data(), kDtype, {kQRows, tokens});
    Tensor multi_k(multi[1].data(), kDtype, {kKRows, tokens});
    Tensor multi_v(multi[2].data(), kDtype, {kVRows, tokens});
    Tensor multi_z(multi[3].data(), kDtype, {kZRows, tokens});
    Tensor multi_g_tensor(multi_g.data(), DType::FP32, {kHeads, tokens});
    Tensor multi_beta_tensor(multi_beta.data(), DType::FP32, {kHeads, tokens});

    std::vector<std::array<GuardedDeviceBuffer, 4>> single;
    std::vector<GuardedDeviceBuffer> single_g;
    std::vector<GuardedDeviceBuffer> single_beta;
    single.reserve(static_cast<std::size_t>(tokens));
    single_g.reserve(static_cast<std::size_t>(tokens));
    single_beta.reserve(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        single.emplace_back(std::array<GuardedDeviceBuffer, 4>{
            GuardedDeviceBuffer(static_cast<std::size_t>(kQRows) * 2),
            GuardedDeviceBuffer(static_cast<std::size_t>(kKRows) * 2),
            GuardedDeviceBuffer(static_cast<std::size_t>(kVRows) * 2),
            GuardedDeviceBuffer(static_cast<std::size_t>(kZRows) * 2)});
        single_g.emplace_back(static_cast<std::size_t>(kHeads) * sizeof(float));
        single_beta.emplace_back(static_cast<std::size_t>(kHeads) * sizeof(float));
    }
    for (auto& buffers : single) {
        for (auto& buffer : buffers) { buffer.fill(0xff); }
    }
    for (auto& buffer : single_g) { buffer.fill(0xff); }
    for (auto& buffer : single_beta) { buffer.fill(0xff); }

    ops::detail::w8_gdn_qkvz_decode_launch(x, weight.view(), multi_q, multi_k, multi_v, multi_z,
                                            control, a_log, dt_bias, multi_g_tensor,
                                            multi_beta_tensor, nullptr);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const auto* token_x = static_cast<const std::uint8_t*>(device_activation.data()) +
                              static_cast<std::size_t>(token) * kHidden * sizeof(std::uint16_t);
        Tensor x_single(const_cast<std::uint8_t*>(token_x), kDtype, {kHidden, 1});
        Tensor q(single[token][0].data(), kDtype, {kQRows, 1});
        Tensor k(single[token][1].data(), kDtype, {kKRows, 1});
        Tensor v(single[token][2].data(), kDtype, {kVRows, 1});
        Tensor z(single[token][3].data(), kDtype, {kZRows, 1});
        Tensor g(single_g[token].data(), DType::FP32, {kHeads, 1});
        Tensor beta(single_beta[token].data(), DType::FP32, {kHeads, 1});
        ops::detail::w8_gdn_qkvz_decode_launch(x_single, weight.view(), q, k, v, z, control, a_log,
                                                dt_bias, g, beta, nullptr);
    }
    cuda_synchronize();

    int failures = 0;
    const auto activation_after = guarded_bits<std::uint16_t>(device_activation, activation_u16.size());
    failures += verify_exact(("multi activation bits" + suffix).c_str(), activation_after, activation_u16);
    failures += weight.verify_preserved("multi parent weight" + suffix);
    failures += device_control.verify_guards("multi control" + suffix);
    const auto control_after = guarded_bits<std::uint16_t>(device_control, control_bits.size());
    failures += verify_exact(("multi control bits" + suffix).c_str(), control_after, control_bits);
    failures += verify_exact(("multi a_log" + suffix).c_str(), from_device<float>(device_a_log, kHeads), a_log_host);
    failures += verify_exact(("multi dt_bias" + suffix).c_str(), from_device<float>(device_dt_bias, kHeads), dt_bias_host);
    failures += device_activation.verify_guards("multi activation" + suffix);

    for (std::size_t p = 0; p < 4; ++p) {
        const std::vector<std::uint16_t> actual = guarded_bits<std::uint16_t>(multi[p], rows[p] * tokens);
        std::vector<std::vector<std::uint16_t>> single_bits(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            single_bits[static_cast<std::size_t>(token)] =
                guarded_bits<std::uint16_t>(single[static_cast<std::size_t>(token)][p], rows[p]);
        }
        const std::string label = std::string("multi ") + names[p] + suffix;
        failures += check_finite_bits(label + " poison", actual);
        failures += verify_exact(label.c_str(), actual,
                                 concatenate_columns(single_bits, tokens, rows[p]));
        failures += multi[p].verify_guards(label + " guards");
        for (std::int32_t token = 0; token < tokens; ++token) {
            failures += single[static_cast<std::size_t>(token)][p].verify_guards(
                std::string("single ") + names[p] + " token=" + std::to_string(token) + suffix);
        }
    }

    for (int gate = 0; gate < 2; ++gate) {
        GuardedDeviceBuffer& multi_buffer = gate == 0 ? multi_g : multi_beta;
        const char* gate_name             = gate == 0 ? "g" : "beta";
        const std::vector<std::uint32_t> actual =
            guarded_bits<std::uint32_t>(multi_buffer, static_cast<std::size_t>(kHeads) * tokens);
        std::vector<std::vector<std::uint32_t>> single_bits(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            GuardedDeviceBuffer& single_buffer =
                gate == 0 ? single_g[static_cast<std::size_t>(token)]
                          : single_beta[static_cast<std::size_t>(token)];
            single_bits[static_cast<std::size_t>(token)] =
                guarded_bits<std::uint32_t>(single_buffer, kHeads);
        }
        const std::string label = std::string("multi ") + gate_name + suffix;
        failures += check_finite_f32_bits(label + " poison", actual);
        failures += verify_exact(label.c_str(), actual,
                                 concatenate_columns(single_bits, tokens, kHeads));
        failures += multi_buffer.verify_guards(label + " guards");
        for (std::int32_t token = 0; token < tokens; ++token) {
            GuardedDeviceBuffer& single_buffer =
                gate == 0 ? single_g[static_cast<std::size_t>(token)]
                          : single_beta[static_cast<std::size_t>(token)];
            failures += single_buffer.verify_guards(
                std::string("single ") + gate_name + " token=" + std::to_string(token) + suffix);
        }
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
    failures += run_case<DType::BF16>(503U);
    failures += run_case<DType::BF16>(504U);
    failures += run_case<DType::FP16>(505U);
    failures += run_case<DType::FP16>(506U);
    failures += run_multi_column_case<DType::BF16>(2, 601U);
    failures += run_multi_column_case<DType::BF16>(4, 602U);
    failures += run_multi_column_case<DType::FP16>(2, 603U);
    failures += run_multi_column_case<DType::FP16>(4, 604U);
    std::cout << (failures ? "FAIL" : "OK") << " flash_next gdn qkvz fused decode\n";
    return failures ? 1 : 0;
}
