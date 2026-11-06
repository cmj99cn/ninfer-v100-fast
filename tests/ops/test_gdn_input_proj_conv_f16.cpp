#include "ops/linear/linear_test_common.h"
#include "ops/quantized_weight.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/gdn_input_proj.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ninfer;

constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kQueryRows  = 2048;
constexpr std::int32_t kKeyRows    = 2048;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kParentRows = kChannels + kZRows;
constexpr std::int32_t kSlots      = 8;

__half to_half(float value) { return __float2half_rn(value); }
float silu_ref(float x) { return x / (1.0f + std::exp(-x)); }

std::uint16_t bf16_bits(float value) {
    const __nv_bfloat16 converted = __float2bfloat16_rn(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

float bf16_value(std::uint16_t bits) {
    __nv_bfloat16 converted;
    std::memcpy(&converted, &bits, sizeof(bits));
    return __bfloat162float(converted);
}

struct Fixture {
    std::vector<__half> x;
    std::vector<std::uint16_t> conv_bits;
    std::vector<__half> initial_state;
};

Fixture make_fixture(std::int32_t tokens, std::uint32_t seed) {
    Fixture out;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    out.x.resize(static_cast<std::size_t>(kHidden) * tokens);
    for (auto& value : out.x) { value = to_half(dist(rng)); }
    out.conv_bits.resize(static_cast<std::size_t>(kChannels) * 4);
    for (auto& value : out.conv_bits) { value = bf16_bits(dist(rng) * 0.5f); }
    out.initial_state.resize(static_cast<std::size_t>(3) * kChannels);
    for (auto& value : out.initial_state) { value = to_half(dist(rng)); }
    return out;
}

__half eval_channel(const Fixture& fixture, const std::vector<__half>& projected,
                    std::int32_t channel, std::int32_t token, float& s0, float& s1, float& s2) {
    const auto conv = [&](int tap) {
        return bf16_value(fixture.conv_bits[static_cast<std::size_t>(tap) * kChannels + channel]);
    };
    const float p = __half2float(projected[static_cast<std::size_t>(token) * kChannels + channel]);
    const float fused = conv(0) * s0 + conv(1) * s1 + conv(2) * s2 + conv(3) * p;
    s0 = s1;
    s1 = s2;
    s2 = p;
    return to_half(silu_ref(fused));
}

__half output_at(const std::vector<__half>& q, const std::vector<__half>& key,
                 const std::vector<__half>& value, std::int32_t channel, std::int32_t token) {
    if (channel < kQueryRows) {
        return q[static_cast<std::size_t>(token) * kQueryRows + channel];
    }
    if (channel < kQueryRows + kKeyRows) {
        return key[static_cast<std::size_t>(token) * kKeyRows + channel - kQueryRows];
    }
    return value[static_cast<std::size_t>(token) * kValueRows + channel - kQueryRows - kKeyRows];
}

int run_capacity_case(std::int32_t tokens, std::uint32_t seed) {
    auto packed     = test::linear::make_nvfp4_weight(kParentRows, kHidden, seed);
    Fixture fixture = make_fixture(tokens, seed + 1);

    DeviceBuffer x_bytes(fixture.x.size() * sizeof(__half));
    x_bytes.copy_from_host(fixture.x.data(), x_bytes.bytes);
    DeviceBuffer w_bytes(packed.payload.size());
    w_bytes.copy_from_host(packed.payload.data(), w_bytes.bytes);
    DeviceBuffer conv_bytes(fixture.conv_bits.size() * sizeof(std::uint16_t));
    conv_bytes.copy_from_host(fixture.conv_bits.data(), conv_bytes.bytes);
    constexpr std::int32_t kCapacitySlots = 64;
    std::vector<__half> zeros(static_cast<std::size_t>(3) * kChannels * kCapacitySlots,
                              to_half(0.0f));
    std::copy(fixture.initial_state.begin(), fixture.initial_state.end(), zeros.begin());
    DeviceBuffer state_bytes(zeros.size() * sizeof(__half));
    state_bytes.copy_from_host(zeros.data(), state_bytes.bytes);
    DeviceBuffer q_bytes(static_cast<std::size_t>(kQueryRows) * tokens * sizeof(__half));
    DeviceBuffer k_bytes(static_cast<std::size_t>(kKeyRows) * tokens * sizeof(__half));
    DeviceBuffer v_bytes(static_cast<std::size_t>(kValueRows) * tokens * sizeof(__half));
    DeviceBuffer z_bytes(static_cast<std::size_t>(kZRows) * tokens * sizeof(__half));
    DeviceBuffer record_bytes(static_cast<std::size_t>(kChannels) * tokens * sizeof(__half));

    Weight weight = packed.device_weight(w_bytes.p);
    Tensor x(x_bytes.p, DType::FP16, {kHidden, tokens});
    Tensor conv(conv_bytes.p, DType::BF16, {kChannels, 4});
    Tensor state(state_bytes.p, DType::FP16, {3, kChannels, kCapacitySlots});
    Tensor q(q_bytes.p, DType::FP16, {kQueryRows, tokens});
    Tensor k(k_bytes.p, DType::FP16, {kKeyRows, tokens});
    Tensor v(v_bytes.p, DType::FP16, {kValueRows, tokens});
    Tensor z(z_bytes.p, DType::FP16, {kZRows, tokens});
    Tensor record(record_bytes.p, DType::FP16, {kChannels, tokens});

    const std::int32_t initial_slot  = 0;
    const std::int32_t snapshot_base = 1;
    DeviceBuffer initial_slot_bytes(sizeof(std::int32_t));
    DeviceBuffer snapshot_base_bytes(sizeof(std::int32_t));
    initial_slot_bytes.copy_from_host(&initial_slot, sizeof(std::int32_t));
    snapshot_base_bytes.copy_from_host(&snapshot_base, sizeof(std::int32_t));
    Tensor initial(initial_slot_bytes.p, DType::I32, {1});
    Tensor snap_base(snapshot_base_bytes.p, DType::I32, {1});
    Tensor no_valid(nullptr, DType::I32, {1});

    const std::size_t snapshot_capacity =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, kParentRows, kHidden, ops::LinearPolicy::A16Only, 1, tokens, tokens,
            DType::FP16);
    DeviceArena snapshot_workspace(std::max<std::size_t>(snapshot_capacity, 1));
    ops::gdn_input_proj_conv_snapshot(x, weight, conv, state, no_valid, initial, snap_base, q, k,
                                      v, z, ops::LinearPolicy::A16Only, snapshot_workspace,
                                      nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "GDN_CONV_F16: capacity snapshot synchronize failed T=" << tokens << '\n';
        return 1;
    }

    // The record form owns T=2..16; a one-column snapshot is the T=1 case.
    if (tokens < 2) { return 0; }
    const std::size_t record_capacity =
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, kParentRows, kHidden, ops::LinearPolicy::A16Only, 1, tokens, tokens,
            DType::FP16);
    DeviceArena record_workspace(std::max<std::size_t>(record_capacity, 1));
    ops::gdn_input_proj_conv_record(x, weight, conv, state, no_valid, initial, record, q, k, v, z,
                                    ops::LinearPolicy::A16Only, record_workspace, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "GDN_CONV_F16: capacity record synchronize failed T=" << tokens << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (!test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int capacity_failures = 0;
        capacity_failures += run_capacity_case(1, 995U);
        capacity_failures += run_capacity_case(6, 996U);
        capacity_failures += run_capacity_case(16, 997U);
        if (capacity_failures != 0) {
            std::cout << "FAIL GDN conv capacity F16\n";
            return 1;
        }
        constexpr std::int32_t kTokens = 6;
        auto packed = test::linear::make_nvfp4_weight(kParentRows, kHidden, 991U);
        Fixture fixture = make_fixture(kTokens, 992U);

        DeviceBuffer x_bytes(fixture.x.size() * sizeof(__half));
        x_bytes.copy_from_host(fixture.x.data(), x_bytes.bytes);
        DeviceBuffer w_bytes(packed.payload.size());
        w_bytes.copy_from_host(packed.payload.data(), w_bytes.bytes);
        DeviceBuffer conv_bytes(fixture.conv_bits.size() * sizeof(std::uint16_t));
        conv_bytes.copy_from_host(fixture.conv_bits.data(), conv_bytes.bytes);
        DeviceBuffer state_bytes(static_cast<std::size_t>(3) * kChannels * kSlots * sizeof(__half));
        {
            std::vector<__half> zeros(static_cast<std::size_t>(3) * kChannels * kSlots,
                                      to_half(0.0f));
            std::copy(fixture.initial_state.begin(), fixture.initial_state.end(), zeros.begin());
            state_bytes.copy_from_host(zeros.data(), state_bytes.bytes);
        }
        DeviceBuffer q_bytes(static_cast<std::size_t>(kQueryRows) * kTokens * sizeof(__half));
        DeviceBuffer k_bytes(static_cast<std::size_t>(kKeyRows) * kTokens * sizeof(__half));
        DeviceBuffer v_bytes(static_cast<std::size_t>(kValueRows) * kTokens * sizeof(__half));
        DeviceBuffer z_bytes(static_cast<std::size_t>(kZRows) * kTokens * sizeof(__half));
        DeviceBuffer projected_bytes(static_cast<std::size_t>(kChannels) * kTokens * sizeof(__half));
        DeviceBuffer record_bytes(static_cast<std::size_t>(kChannels) * kTokens * sizeof(__half));
        q_bytes.fill(0xCD);
        k_bytes.fill(0xCD);
        v_bytes.fill(0xCD);
        z_bytes.fill(0xCD);
        projected_bytes.fill(0xCD);
        record_bytes.fill(0xCD);

        Weight weight = packed.device_weight(w_bytes.p);
        DeviceArena workspace(std::size_t{1} << 20);
        Tensor x(x_bytes.p, DType::FP16, {kHidden, kTokens});
        Tensor conv(conv_bytes.p, DType::BF16, {kChannels, 4});
        Tensor state(state_bytes.p, DType::FP16, {3, kChannels, kSlots});
        Tensor q(q_bytes.p, DType::FP16, {kQueryRows, kTokens});
        Tensor k(k_bytes.p, DType::FP16, {kKeyRows, kTokens});
        Tensor v(v_bytes.p, DType::FP16, {kValueRows, kTokens});
        Tensor z(z_bytes.p, DType::FP16, {kZRows, kTokens});
        Tensor projected(projected_bytes.p, DType::FP16, {kChannels, kTokens});
        Tensor record(record_bytes.p, DType::FP16, {kChannels, kTokens});

        ops::gdn_input_proj(x, weight, projected, z, ops::LinearPolicy::A16Only, workspace, nullptr);

        const std::int32_t initial_slot  = 0;
        const std::int32_t snapshot_base = 1;
        DeviceBuffer initial_slot_bytes(sizeof(std::int32_t));
        DeviceBuffer snapshot_base_bytes(sizeof(std::int32_t));
        initial_slot_bytes.copy_from_host(&initial_slot, sizeof(std::int32_t));
        snapshot_base_bytes.copy_from_host(&snapshot_base, sizeof(std::int32_t));
        Tensor initial(initial_slot_bytes.p, DType::I32, {1});
        Tensor snap_base(snapshot_base_bytes.p, DType::I32, {1});
        Tensor no_valid(nullptr, DType::I32, {1});

        {
            auto scope = workspace.scope();
            ops::gdn_input_proj_conv_snapshot(x, weight, conv, state, no_valid, initial, snap_base,
                                              q, k, v, z, ops::LinearPolicy::A16Only, workspace,
                                              nullptr);
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << "GDN_CONV_F16: snapshot synchronize failed\n";
            return 1;
        }

        std::vector<__half> projected_host(static_cast<std::size_t>(kChannels) * kTokens);
        std::vector<__half> state_after(static_cast<std::size_t>(3) * kChannels * kSlots);
        std::vector<__half> q_host(static_cast<std::size_t>(kQueryRows) * kTokens);
        std::vector<__half> k_host(static_cast<std::size_t>(kKeyRows) * kTokens);
        std::vector<__half> v_host(static_cast<std::size_t>(kValueRows) * kTokens);
        projected_bytes.copy_to_host(projected_host.data(), projected_bytes.bytes);
        state_bytes.copy_to_host(state_after.data(), state_bytes.bytes);
        q_bytes.copy_to_host(q_host.data(), q_bytes.bytes);
        k_bytes.copy_to_host(k_host.data(), k_bytes.bytes);
        v_bytes.copy_to_host(v_host.data(), v_bytes.bytes);

        int failures = 0;
        int output_failures = 0;
        int state_failures = 0;
        float max_output_diff = 0.0F;
        for (std::int32_t channel = 0; channel < kChannels; ++channel) {
            float s0 = __half2float(fixture.initial_state[channel]);
            float s1 = __half2float(fixture.initial_state[kChannels + channel]);
            float s2 = __half2float(fixture.initial_state[2 * kChannels + channel]);
            for (std::int32_t token = 0; token < kTokens; ++token) {
                const __half want = eval_channel(fixture, projected_host, channel, token, s0, s1, s2);
                const __half got  = output_at(q_host, k_host, v_host, channel, token);
                const float diff = std::fabs(__half2float(got) - __half2float(want));
                max_output_diff = std::max(max_output_diff, diff);
                if (diff > 2.0e-3F) { ++output_failures; }
            }
        }
        for (std::int32_t channel = 0; channel < kChannels; channel += 97) {
            float s0 = __half2float(fixture.initial_state[channel]);
            float s1 = __half2float(fixture.initial_state[kChannels + channel]);
            float s2 = __half2float(fixture.initial_state[2 * kChannels + channel]);
            for (std::int32_t token = 0; token < kTokens; ++token) {
                const float p =
                    __half2float(projected_host[static_cast<std::size_t>(token) * kChannels +
                                                channel]);
                const __half want[3] = {to_half(s1), to_half(s2), to_half(p)};
                const std::size_t slot_base =
                    static_cast<std::size_t>(snapshot_base + token) * 3 * kChannels;
                for (int history = 0; history < 3; ++history) {
                    const __half got =
                        state_after[slot_base + static_cast<std::size_t>(history) * kChannels +
                                    channel];
                    if (std::memcmp(&got, &want[history], sizeof(__half)) != 0) { ++state_failures; }
                }
                s0 = s1;
                s1 = s2;
                s2 = p;
            }
        }

        {
            std::vector<__half> state_before(static_cast<std::size_t>(3) * kChannels * kSlots);
            state_bytes.copy_to_host(state_before.data(), state_bytes.bytes);
            q_bytes.fill(0xCD);
            k_bytes.fill(0xCD);
            v_bytes.fill(0xCD);
            z_bytes.fill(0xCD);
            auto scope = workspace.scope();
            ops::gdn_input_proj_conv_record(x, weight, conv, state, no_valid, initial, record, q, k,
                                            v, z, ops::LinearPolicy::A16Only, workspace, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                std::cerr << "GDN_CONV_F16: record synchronize failed\n";
                return 1;
            }
            std::vector<__half> record_host(static_cast<std::size_t>(kChannels) * kTokens);
            std::vector<__half> state_after_record(
                static_cast<std::size_t>(3) * kChannels * kSlots);
            record_bytes.copy_to_host(record_host.data(), record_bytes.bytes);
            state_bytes.copy_to_host(state_after_record.data(), state_bytes.bytes);
            if (std::memcmp(state_before.data(), state_after_record.data(),
                            state_before.size() * sizeof(__half)) != 0) {
                ++failures;
            }
            q_bytes.copy_to_host(q_host.data(), q_bytes.bytes);
            k_bytes.copy_to_host(k_host.data(), k_bytes.bytes);
            v_bytes.copy_to_host(v_host.data(), v_bytes.bytes);
            for (std::int32_t channel = 0; channel < kChannels; channel += 97) {
                float s0 = __half2float(fixture.initial_state[channel]);
                float s1 = __half2float(fixture.initial_state[kChannels + channel]);
                float s2 = __half2float(fixture.initial_state[2 * kChannels + channel]);
                for (std::int32_t token = 0; token < kTokens; ++token) {
                    const __half want = eval_channel(fixture, record_host, channel, token, s0, s1,
                                                     s2);
                    const __half got = output_at(q_host, k_host, v_host, channel, token);
                    if (std::fabs(__half2float(got) - __half2float(want)) > 2.0e-3F) {
                        ++output_failures;
                    }
                }
            }
        }
        failures = output_failures + state_failures;
        std::cout << "f16 conv: output_failures=" << output_failures
                  << " state_failures=" << state_failures
                  << " max_output_diff=" << max_output_diff << '\n';
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GDN conv snapshot/record F16\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GDN conv F16: " << error.what() << '\n';
        return 1;
    }
}
