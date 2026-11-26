#include "ninfer/ops/dflash2_attn_input.h"
#include "ninfer/ops/dflash2_linear_swiglu.h"

#include "core/arena.h"
#include "ops/op_tester.h"

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

// BF16 outputs of an FP32-accumulating 2560-wide projection: allow two output ULPs at the
// tensor's maximum magnitude, which at ~3e3 is ~1.6e1.
constexpr ReductionCriterion kBf16Criterion{
    .relative_l2                     = 6.0e-3,
    .gross_absolute                  = 1.0e-2,
    .gross_relative_to_max_reference = 2.0e-2,
};

struct W8 {
    Weight weight;
    DeviceBuffer storage;
    std::vector<float> decoded;
};

W8 make_w8(std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
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
    W8 out;
    out.storage = to_device(payload);
    out.weight.payload = out.storage.p;
    out.weight.payload_bytes = payload.size();
    out.weight.qtype = QType::W8G32_F16S;
    out.weight.layout = QuantLayout::RowSplit;
    out.weight.scale_dtype = DType::FP16;
    out.weight.group_size = 32;
    out.weight.group = 32;
    out.weight.ndim = 2;
    out.weight.n = rows;
    out.weight.k = columns;
    out.weight.shape[0] = rows;
    out.weight.shape[1] = columns;
    out.weight.padded_shape[0] = rows;
    out.weight.padded_shape[1] = columns;
    out.weight.qdata = out.storage.p;
    out.weight.scales = static_cast<const std::uint8_t*>(out.storage.p) + code_bytes;
    out.decoded.resize(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const auto code = static_cast<std::int8_t>(codes[static_cast<std::size_t>(row) * columns + column]);
            const auto bits = scales[static_cast<std::size_t>(row) * (columns / 32) + column / 32];
            out.decoded[static_cast<std::size_t>(row) * columns + column] =
                static_cast<float>(code) * f16_to_f32(bits);
        }
    }
    return out;
}

int run_attn(int tokens) {
    W8 parent = make_w8(6144, kHidden, 1301u);
    std::vector<float> x(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(x, 1307u, -0.5f, 0.5f);
    round_to_bf16(x);
    const std::size_t qn = static_cast<std::size_t>(4096) * tokens;
    const std::size_t kvn = static_cast<std::size_t>(1024) * tokens;
    std::vector<double> expected_q(qn), expected_k(kvn), expected_v(kvn);
    for (int t = 0; t < tokens; ++t) {
        for (int row = 0; row < 4096; ++row) {
            double sum = 0.0;
            for (int r = 0; r < kHidden; ++r) {
                sum += static_cast<double>(
                           parent.decoded[static_cast<std::size_t>(row) * kHidden + r]) *
                       x[static_cast<std::size_t>(t) * kHidden + r];
            }
            expected_q[static_cast<std::size_t>(t) * 4096 + row] = sum;
        }
        for (int row = 0; row < 1024; ++row) {
            double ksum = 0.0, vsum = 0.0;
            for (int r = 0; r < kHidden; ++r) {
                const double value = x[static_cast<std::size_t>(t) * kHidden + r];
                ksum += static_cast<double>(
                            parent.decoded[static_cast<std::size_t>(4096 + row) * kHidden + r]) *
                        value;
                vsum += static_cast<double>(
                            parent.decoded[static_cast<std::size_t>(5120 + row) * kHidden + r]) *
                        value;
            }
            expected_k[static_cast<std::size_t>(t) * 1024 + row] = ksum;
            expected_v[static_cast<std::size_t>(t) * 1024 + row] = vsum;
        }
    }
    DeviceBuffer d_x = to_device_bf16(x);
    GuardedDeviceBuffer d_q(qn * sizeof(std::uint16_t)), d_k(kvn * sizeof(std::uint16_t)),
        d_v(kvn * sizeof(std::uint16_t));
    d_q.fill(0x7f);
    d_k.fill(0x7f);
    d_v.fill(0x7f);
    Tensor x_t(d_x.p, DType::BF16, {kHidden, tokens});
    Tensor q_t(d_q.data(), DType::BF16, {4096, tokens});
    Tensor k_t(d_k.data(), DType::BF16, {1024, tokens});
    Tensor v_t(d_v.data(), DType::BF16, {1024, tokens});
    ops::dflash2_attn_input_proj(x_t, parent.weight, q_t, k_t, v_t, nullptr);
    cuda_synchronize();
    int failures = verify_reduction("dflash2_attn_input q", from_device_bf16(d_q.data(), qn),
                                    expected_q, kBf16Criterion);
    failures += verify_reduction("dflash2_attn_input k", from_device_bf16(d_k.data(), kvn),
                                 expected_k, kBf16Criterion);
    failures += verify_reduction("dflash2_attn_input v", from_device_bf16(d_v.data(), kvn),
                                 expected_v, kBf16Criterion);
    failures += d_q.verify_guards("attn q guards") + d_k.verify_guards("attn k guards") +
                d_v.verify_guards("attn v guards");
    return failures;
}

int run_swiglu(int tokens) {
    W8 parent = make_w8(15360, kHidden, 1409u);
    std::vector<float> x(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(x, 1423u, -0.5f, 0.5f);
    round_to_bf16(x);
    constexpr int kIntermediate = 7680;
    std::vector<double> expected(static_cast<std::size_t>(kIntermediate) * tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < kIntermediate; ++i) {
            double gate = 0.0, up = 0.0;
            for (int r = 0; r < kHidden; ++r) {
                const double value = x[static_cast<std::size_t>(t) * kHidden + r];
                gate += static_cast<double>(
                            parent.decoded[static_cast<std::size_t>(i) * kHidden + r]) * value;
                up += static_cast<double>(
                          parent.decoded[static_cast<std::size_t>(kIntermediate + i) * kHidden + r]) *
                      value;
            }
            const double silu = gate / (1.0 + std::exp(-gate));
            expected[static_cast<std::size_t>(t) * kIntermediate + i] = silu * up;
        }
    }
    DeviceBuffer d_x = to_device_bf16(x);
    const std::size_t count = static_cast<std::size_t>(kIntermediate) * tokens;
    GuardedDeviceBuffer d_out(count * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    Tensor x_t(d_x.p, DType::BF16, {kHidden, tokens});
    Tensor out_t(d_out.data(), DType::BF16, {kIntermediate, tokens});
    DeviceArena workspace(ops::dflash2_linear_swiglu_workspace_capacity_bytes(tokens, tokens));
    ops::dflash2_linear_swiglu(x_t, parent.weight, out_t, workspace, nullptr);
    cuda_synchronize();
    int failures = verify_reduction("dflash2_linear_swiglu", from_device_bf16(d_out.data(), count),
                                    expected, kBf16Criterion);
    failures += d_out.verify_guards("swiglu guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "dflash2_fused: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = 0;
    failures += run_attn(1);
    failures += run_attn(3);
    failures += run_swiglu(1);
    failures += run_swiglu(2);
    if (failures != 0) {
        std::cerr << "dflash2_fused failures=" << failures << '\n';
        return 1;
    }
    std::cout << "dflash2_fused: PASS\n";
    return 0;
}
