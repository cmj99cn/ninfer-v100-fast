#include "ninfer/ops/dynamic_grouped_conv.h"

#include "core/arena.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops;

namespace {

constexpr int kHidden = 2560;
constexpr int kGroups = kHidden / 16;

// BF16 outputs of FP32-accumulating projections: two output ULPs at the tensor maximum.
constexpr ReductionCriterion kBf16Criterion{
    .relative_l2                     = 6.0e-3,
    .gross_absolute                  = 1.0e-2,
    .gross_relative_to_max_reference = 2.0e-2,
};

int run_conv(int width, int batch, int side, bool coeff_fp32) {
    const std::size_t cells = static_cast<std::size_t>(kHidden) * width * batch;
    const std::size_t coeffs = static_cast<std::size_t>(4 * kGroups) * width * batch;
    std::vector<float> x(cells), coefficients(coeffs), base(static_cast<std::size_t>(kHidden) * 4),
        residual(cells);
    fill_uniform(x, 11u + width, -0.5f, 0.5f);
    fill_uniform(coefficients, 29u + side, -0.25f, 0.25f);
    fill_uniform(base, 37u, -0.5f, 0.5f);
    fill_uniform(residual, 43u, -0.5f, 0.5f);
    round_to_bf16(x);
    round_to_bf16(base);
    round_to_bf16(residual);
    if (!coeff_fp32) { round_to_bf16(coefficients); }
    std::vector<double> expected(cells);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < width; ++i) {
            const std::int64_t column = b * width + i;
            for (int h = 0; h < kHidden; ++h) {
                const double b0 = base[(side * 2 + 0) * kHidden + h];
                const double b1 = base[(side * 2 + 1) * kHidden + h];
                const std::int64_t row0 = column * 4 * kGroups + (side * 2 + 0) * kGroups + h / 16;
                const std::int64_t row1 = column * 4 * kGroups + (side * 2 + 1) * kGroups + h / 16;
                const std::int64_t e = column * kHidden + h;
                double value = (b0 + coefficients[row0]) * x[e];
                if (i > 0) {
                    value += (b1 + coefficients[row1]) * x[e - kHidden];
                }
                if (side == 1) { value += residual[e]; }
                expected[e] = value;
            }
        }
    }
    DeviceBuffer d_x = to_device_bf16(x);
    DeviceBuffer d_coeff =
        coeff_fp32 ? to_device(coefficients) : to_device_bf16(coefficients);
    DeviceBuffer d_base = to_device_bf16(base);
    GuardedDeviceBuffer d_out(cells * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    if (side == 1) {
        DeviceBuffer d_residual_in = to_device_bf16(residual);
        cuda_check(cudaMemcpy(d_out.data(), d_residual_in.p, cells * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToDevice),
                   "copy residual into output guard");
    }
    Tensor x_t(d_x.p, DType::BF16, {kHidden, width, batch});
    Tensor coeff_t(d_coeff.p, coeff_fp32 ? DType::FP32 : DType::BF16,
                   {4 * kGroups, width, batch});
    Tensor base_t(d_base.p, DType::BF16, {kHidden, 2, 2});
    Tensor out_t(d_out.data(), DType::BF16, {kHidden, width, batch});
    ops::dynamic_grouped_conv(x_t, coeff_t, base_t, side, out_t, nullptr);
    cuda_synchronize();
    return verify_reduction("dynamic_grouped_conv", from_device_bf16(d_out.data(), cells), expected,
                            kBf16Criterion) +
           d_out.verify_guards("dynamic_grouped_conv guards");
}

Weight make_w8(std::int32_t rows, std::int32_t columns, DeviceBuffer& storage,
               std::vector<float>& decoded, std::uint32_t seed) {
    const std::size_t code_bytes = static_cast<std::size_t>(rows) * columns;
    const std::size_t scale_bytes = static_cast<std::size_t>(rows) * (columns / 32) * 2;
    std::vector<std::uint8_t> codes(code_bytes);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(rows) * (columns / 32));
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < code_bytes; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        codes[i] = static_cast<std::uint8_t>(static_cast<int>(state % 255) - 127);
    }
    for (auto& scale : scales) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        scale = f32_to_f16(static_cast<float>(1 + state % 17) / 256.0f);
    }
    std::vector<std::uint8_t> payload(code_bytes + scale_bytes);
    std::memcpy(payload.data(), codes.data(), code_bytes);
    std::memcpy(payload.data() + code_bytes, scales.data(), scale_bytes);
    storage = to_device(payload);
    Weight weight{};
    weight.payload       = storage.p;
    weight.payload_bytes = payload.size();
    weight.qtype         = QType::W8G32_F16S;
    weight.layout        = QuantLayout::RowSplit;
    weight.scale_dtype   = DType::FP16;
    weight.group_size    = 32;
    weight.group         = 32;
    weight.ndim          = 2;
    weight.n             = rows;
    weight.k             = columns;
    weight.shape[0] = rows;
    weight.shape[1] = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    weight.qdata  = storage.p;
    weight.scales = static_cast<const std::uint8_t*>(storage.p) + code_bytes;
    decoded.resize(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const auto code = static_cast<std::int8_t>(
                codes[static_cast<std::size_t>(row) * columns + column]);
            const auto bits =
                scales[static_cast<std::size_t>(row) * (columns / 32) + column / 32];
            decoded[static_cast<std::size_t>(row) * columns + column] =
                static_cast<float>(code) * f16_to_f32(bits);
        }
    }
    return weight;
}


int run_prepare_add() {
    constexpr int width = 2;
    constexpr int batch = 1;
    constexpr int input_rows = 4096;
    const std::size_t cells = static_cast<std::size_t>(kHidden) * width * batch;
    std::vector<float> residual(cells), norm_weight(kHidden), base(static_cast<std::size_t>(kHidden) * 4),
        kernel_proj(static_cast<std::size_t>(4 * kGroups) * kHidden);
    std::vector<float> x(static_cast<std::size_t>(input_rows) * width * batch);
    fill_uniform(residual, 101u, -0.5f, 0.5f);
    fill_uniform(norm_weight, 103u, 0.5f, 1.5f);
    fill_uniform(base, 107u, -0.5f, 0.5f);
    fill_uniform(kernel_proj, 109u, -0.1f, 0.1f);
    fill_uniform(x, 113u, -0.5f, 0.5f);
    round_to_bf16(residual);
    round_to_bf16(norm_weight);
    round_to_bf16(base);
    round_to_bf16(kernel_proj);
    round_to_bf16(x);

    std::vector<float> w8_decoded;
    DeviceBuffer w8_storage;
    Weight projection = make_w8(kHidden, input_rows, w8_storage, w8_decoded, 211u);

    // Oracle: RMSNorm, coefficient projection, prepare, then the W8 projection + finish add.
    std::vector<double> normalized(cells);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < width; ++i) {
            const std::int64_t column = b * width + i;
            double squares = 0.0;
            for (int h = 0; h < kHidden; ++h) {
                squares += static_cast<double>(residual[column * kHidden + h]) *
                           residual[column * kHidden + h];
            }
            const double inv = 1.0 / std::sqrt(squares / kHidden + 1e-6);
            for (int h = 0; h < kHidden; ++h) {
                normalized[column * kHidden + h] =
                    residual[column * kHidden + h] * inv * norm_weight[h];
            }
        }
    }
    std::vector<double> coefficients(static_cast<std::size_t>(4 * kGroups) * width * batch, 0.0);
    for (int n = 0; n < 4 * kGroups; ++n) {
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < width; ++i) {
                const std::int64_t column = b * width + i;
                double sum = 0.0;
                for (int h = 0; h < kHidden; ++h) {
                    sum += static_cast<double>(kernel_proj[static_cast<std::size_t>(n) * kHidden + h]) *
                           normalized[column * kHidden + h];
                }
                coefficients[column * 4 * kGroups + n] = sum;
            }
        }
    }
    std::vector<double> expected_prepared(cells);
    std::vector<double> expected_delta(static_cast<std::size_t>(kGroups) * 2 * width * batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < width; ++i) {
            const std::int64_t column = b * width + i;
            for (int g = 0; g < kGroups; ++g) {
                const std::int64_t dst = column * 2 * kGroups + g;
                expected_delta[dst] = coefficients[column * 4 * kGroups + 2 * kGroups + g];
                expected_delta[dst + kGroups] =
                    coefficients[column * 4 * kGroups + 3 * kGroups + g];
            }
            for (int h = 0; h < kHidden; ++h) {
                const double b0 = base[0 * kHidden + h];
                const double b1 = base[1 * kHidden + h];
                const std::int64_t e = column * kHidden + h;
                double value = (b0 + coefficients[column * 4 * kGroups + h / 16]) *
                               normalized[e];
                if (i > 0) {
                    value += (b1 + coefficients[column * 4 * kGroups + kGroups + h / 16]) *
                             normalized[e - kHidden];
                }
                expected_prepared[e] = value;
            }
        }
    }
    std::vector<double> expected_residual(cells);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < width; ++i) {
            const std::int64_t column = b * width + i;
            for (int h = 0; h < kHidden; ++h) {
                double z = 0.0;
                for (int r = 0; r < input_rows; ++r) {
                    z += static_cast<double>(w8_decoded[static_cast<std::size_t>(h) * input_rows + r]) *
                         x[column * input_rows + r];
                }
                const double b0 = base[2 * kHidden + h];
                const double b1 = base[3 * kHidden + h];
                const std::int64_t di = column * 2 * kGroups + h / 16;
                double value = residual[column * kHidden + h] +
                               (b0 + expected_delta[di]) * z;
                if (i > 0) {
                    double previous = 0.0;
                    for (int r = 0; r < input_rows; ++r) {
                        previous +=
                            static_cast<double>(w8_decoded[static_cast<std::size_t>(h) * input_rows + r]) *
                            x[(column - 1) * input_rows + r];
                    }
                    value += (b1 + expected_delta[di + kGroups]) * previous;
                }
                expected_residual[column * kHidden + h] = value;
            }
        }
    }

    DeviceBuffer d_residual = to_device_bf16(residual);
    DeviceBuffer d_norm = to_device_bf16(norm_weight);
    DeviceBuffer d_base = to_device_bf16(base);
    DeviceBuffer d_kernel_proj = to_device_bf16(kernel_proj);
    Weight kernel_weight{};
    kernel_weight.payload = d_kernel_proj.p;
    kernel_weight.qdata = d_kernel_proj.p;
    kernel_weight.payload_bytes = kernel_proj.size() * 2;
    kernel_weight.qtype = QType::BF16_CTRL;
    kernel_weight.layout = QuantLayout::Contiguous;
    kernel_weight.ndim = 2;
    kernel_weight.n = 4 * kGroups;
    kernel_weight.k = kHidden;
    kernel_weight.shape[0] = 4 * kGroups;
    kernel_weight.shape[1] = kHidden;
    kernel_weight.padded_shape[0] = 4 * kGroups;
    kernel_weight.padded_shape[1] = kHidden;

    DeviceBuffer d_x = to_device_bf16(x);
    GuardedDeviceBuffer d_prepared(cells * sizeof(std::uint16_t));
    d_prepared.fill(0x7f);
    GuardedDeviceBuffer d_delta(expected_delta.size() * sizeof(std::uint16_t));
    d_delta.fill(0x7f);
    GuardedDeviceBuffer d_out(cells * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    cuda_check(cudaMemcpy(d_out.data(), d_residual.p, cells * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice),
               "copy residual into add output");

    Tensor residual_t(d_residual.p, DType::BF16, {kHidden, width, batch});
    Tensor norm_t(d_norm.p, DType::BF16, {kHidden});
    Tensor base_t(d_base.p, DType::BF16, {kHidden, 2, 2});
    Tensor prepared_t(d_prepared.data(), DType::BF16, {kHidden, width, batch});
    Tensor delta_t(d_delta.data(), DType::BF16, {kGroups, 2, width, batch});
    DeviceArena prepare_ws(ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
        kHidden, width, width, batch, batch));
    ops::rmsnorm_dynamic_grouped_conv_prepare(residual_t, norm_t, 1e-6f, base_t, kernel_weight,
                                              prepared_t, delta_t, prepare_ws, nullptr);

    Tensor x_t(d_x.p, DType::BF16, {input_rows, width, batch});
    Tensor out_t(d_out.data(), DType::BF16, {kHidden, width, batch});
    DeviceArena add_ws(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
        kHidden, input_rows, width, width, batch, batch));
    ops::linear_dynamic_grouped_conv_add(x_t, projection, base_t, delta_t, out_t, add_ws, nullptr);
    cuda_synchronize();

    int failures = verify_reduction("prepare prepared",
                                    from_device_bf16(d_prepared.data(), cells), expected_prepared,
                                    kBf16Criterion);
    failures += verify_reduction("prepare finish_delta",
                                 from_device_bf16(d_delta.data(), expected_delta.size()),
                                 expected_delta, kBf16Criterion);
    failures += verify_reduction("linear dynamic conv add",
                                 from_device_bf16(d_out.data(), cells), expected_residual,
                                 kBf16Criterion);
    failures += d_prepared.verify_guards("prepare prepared guards");
    failures += d_delta.verify_guards("prepare delta guards");
    failures += d_out.verify_guards("add residual guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "dynamic_grouped_conv: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = 0;
    failures += run_conv(1, 1, 0, true);
    failures += run_conv(4, 2, 0, true);
    failures += run_conv(4, 2, 1, true);
    failures += run_conv(2, 1, 1, false);
    failures += run_prepare_add();
    if (failures != 0) {
        std::cerr << "dynamic_grouped_conv failures=" << failures << '\n';
        return 1;
    }
    std::cout << "dynamic_grouped_conv: PASS\n";
    return 0;
}
