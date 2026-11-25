#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStateDim    = 128;
constexpr std::int32_t kQkHeads     = 8;
constexpr std::int32_t kValueHeads  = 24;
constexpr std::int32_t kLayers      = 36;
constexpr std::int32_t kConvChannels = 5120;
constexpr std::int32_t kWidth       = 2;
constexpr std::int32_t kCapacity    = 1;
constexpr float kScale              = 1.0F / std::sqrt(128.0F);

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float pattern(std::uint32_t key, float magnitude) {
    const std::int32_t centered = static_cast<std::int32_t>(mix(key) % 2001U) - 1000;
    return static_cast<float>(centered) * (magnitude / 1000.0F);
}

std::uint16_t activation_word(std::uint32_t key, DType dtype) {
    return dtype == DType::FP16 ? f32_to_f16(pattern(key, 0.08F))
                                : f32_to_bf16(pattern(key, 0.08F));
}

int compare_bits(const std::string& label, const void* got, const void* want, std::size_t words,
                 std::size_t element_bytes) {
    std::vector<std::uint8_t> got_bytes(words * element_bytes);
    std::vector<std::uint8_t> want_bytes(words * element_bytes);
    CUDA_CHECK(cudaMemcpy(got_bytes.data(), got, got_bytes.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(want_bytes.data(), want, want_bytes.size(), cudaMemcpyDeviceToHost));
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < got_bytes.size(); ++index) {
        if (got_bytes[index] != want_bytes[index]) { ++mismatches; }
    }
    if (mismatches != 0) {
        std::cerr << "FAIL " << label.c_str() << " byte_mismatches=" << mismatches << "/"
                  << got_bytes.size()
                  << "\n";
    }
    return mismatches == 0 ? 0 : 1;
}

int run_case(DType dtype, std::uint32_t seed) {
    const std::size_t q_elements     = static_cast<std::size_t>(kStateDim) * kQkHeads;
    const std::size_t v_elements     = static_cast<std::size_t>(kStateDim) * kValueHeads;
    const std::size_t state_elements = v_elements * kStateDim;

    std::vector<std::uint16_t> q_host(q_elements);
    std::vector<std::uint16_t> k_host(q_elements);
    std::vector<std::uint16_t> v_host(v_elements);
    std::vector<float> g_host(kValueHeads);
    std::vector<float> beta_host(kValueHeads);
    std::vector<float> state_host(state_elements);
    for (std::size_t i = 0; i < q_elements; ++i) {
        q_host[i] = activation_word(seed + static_cast<std::uint32_t>(i), dtype);
        k_host[i] = activation_word(seed + 100003U + static_cast<std::uint32_t>(i), dtype);
    }
    for (std::size_t i = 0; i < v_elements; ++i) {
        v_host[i] = activation_word(seed + 200003U + static_cast<std::uint32_t>(i), dtype);
    }
    for (int head = 0; head < kValueHeads; ++head) {
        g_host[static_cast<std::size_t>(head)]    = pattern(seed + 300003U + head, 0.2F);
        beta_host[static_cast<std::size_t>(head)] = pattern(seed + 400003U + head, 0.2F);
    }
    for (std::size_t i = 0; i < state_elements; ++i) {
        state_host[i] = pattern(seed + 500003U + static_cast<std::uint32_t>(i), 0.01F);
    }

    DeviceBuffer q1(q_elements * 2), k1(q_elements * 2), v1(v_elements * 2);
    DeviceBuffer g1(kValueHeads * sizeof(float)), beta1(kValueHeads * sizeof(float));
    q1.copy_from_host(q_host.data(), q1.bytes);
    k1.copy_from_host(k_host.data(), k1.bytes);
    v1.copy_from_host(v_host.data(), v1.bytes);
    g1.copy_from_host(g_host.data(), g1.bytes);
    beta1.copy_from_host(beta_host.data(), beta1.bytes);

    std::vector<std::uint16_t> q2_host(q_elements * kWidth, 0);
    std::vector<std::uint16_t> k2_host(q_elements * kWidth, 0);
    std::vector<std::uint16_t> v2_host(v_elements * kWidth, 0);
    std::vector<float> g2_host(kValueHeads * kWidth, 0.0F);
    std::vector<float> beta2_host(kValueHeads * kWidth, 0.0F);
    std::copy(q_host.begin(), q_host.end(), q2_host.begin());
    std::copy(k_host.begin(), k_host.end(), k2_host.begin());
    std::copy(v_host.begin(), v_host.end(), v2_host.begin());
    std::copy(g_host.begin(), g_host.end(), g2_host.begin());
    std::copy(beta_host.begin(), beta_host.end(), beta2_host.begin());
    DeviceBuffer q2(q2_host.size() * 2), k2(k2_host.size() * 2), v2(v2_host.size() * 2);
    DeviceBuffer g2(g2_host.size() * sizeof(float)), beta2(beta2_host.size() * sizeof(float));
    q2.copy_from_host(q2_host.data(), q2.bytes);
    k2.copy_from_host(k2_host.data(), k2.bytes);
    v2.copy_from_host(v2_host.data(), v2.bytes);
    g2.copy_from_host(g2_host.data(), g2.bytes);
    beta2.copy_from_host(beta2_host.data(), beta2.bytes);

    DeviceBuffer direct_state(state_elements * sizeof(float));
    DeviceBuffer snapshot_state(state_elements * sizeof(float));
    direct_state.copy_from_host(state_host.data(), direct_state.bytes);
    snapshot_state.copy_from_host(state_host.data(), snapshot_state.bytes);
    DeviceBuffer direct_out(v_elements * 2), snapshot_out(v_elements * 2);
    DeviceBuffer record_out(v_elements * kWidth * 2);
    DeviceBuffer zero_slot(sizeof(std::int32_t));
    DeviceBuffer one_valid(sizeof(std::int32_t));
    const std::int32_t zero = 0;
    const std::int32_t one  = 1;
    zero_slot.copy_from_host(&zero, sizeof(zero));
    one_valid.copy_from_host(&one, sizeof(one));

    Tensor q1_tensor(q1.p, dtype, {kStateDim, kQkHeads, 1, 1});
    Tensor k1_tensor(k1.p, dtype, {kStateDim, kQkHeads, 1, 1});
    Tensor v1_tensor(v1.p, dtype, {kStateDim, kValueHeads, 1, 1});
    Tensor g1_tensor(g1.p, DType::FP32, {kValueHeads, 1, 1});
    Tensor beta1_tensor(beta1.p, DType::FP32, {kValueHeads, 1, 1});
    Tensor direct_out_tensor(direct_out.p, dtype, {kStateDim, kValueHeads, 1, 1});
    Tensor snapshot_out_tensor(snapshot_out.p, dtype, {kStateDim, kValueHeads, 1, 1});
    Tensor direct_state_tensor(direct_state.p, DType::FP32, {kStateDim, kStateDim, kValueHeads});
    Tensor snapshot_state_tensor(snapshot_state.p, DType::FP32,
                                 {kStateDim, kStateDim, kValueHeads, 1});
    Tensor slots(zero_slot.p, DType::I32, {1});
    WorkspaceArena scratch(1);
    Tensor direct_q3 = q1_tensor.view({kStateDim, kQkHeads, 1});
    Tensor direct_k3 = k1_tensor.view({kStateDim, kQkHeads, 1});
    Tensor direct_v3 = v1_tensor.view({kStateDim, kValueHeads, 1});
    Tensor direct_g3 = g1_tensor.view({kValueHeads, 1});
    Tensor direct_beta3 = beta1_tensor.view({kValueHeads, 1});
    Tensor direct_out3 = direct_out_tensor.view({kStateDim, kValueHeads, 1});

    ops::gated_delta_net(direct_q3, direct_k3, direct_v3, direct_g3, direct_beta3, kScale, true,
                         scratch, direct_state_tensor, direct_out3, nullptr);
    ops::gated_delta_net_snapshot(q1_tensor, k1_tensor, v1_tensor, g1_tensor, beta1_tensor, kScale,
                                  true, snapshot_state_tensor, Tensor{}, slots, slots,
                                  snapshot_out_tensor, nullptr);

    GdnReplayRecordSpec record_spec{.activation_dtype = dtype,
                                    .layers           = kLayers,
                                    .record_capacity  = kCapacity,
                                    .width            = kWidth,
                                    .conv_channels    = kConvChannels,
                                    .qk_heads         = kQkHeads,
                                    .value_heads      = kValueHeads,
                                    .key_dim          = kStateDim,
                                    .value_dim        = kStateDim};
    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(record_builder, record_spec);
    DeviceBuffer record_storage(record_builder.finish(256));
    record_storage.fill(0xff);
    GdnReplayRecords records({record_storage.p, record_storage.bytes}, record_layout);
    GdnReplayRecordLayer layer_records = records.layer(0, 1);
    std::vector<std::uint16_t> key_record_host(q_elements * kWidth, 0);
    std::vector<std::uint16_t> value_record_host(v_elements * kWidth, 0);
    std::vector<float> gate_record_host(kValueHeads * kWidth * 2, 0.0F);
    std::copy(k_host.begin(), k_host.end(), key_record_host.begin());
    std::copy(v_host.begin(), v_host.end(), value_record_host.begin());
    for (int head = 0; head < kValueHeads; ++head) {
        gate_record_host[static_cast<std::size_t>(head) * 2]     = g_host[static_cast<std::size_t>(head)];
        gate_record_host[static_cast<std::size_t>(head) * 2 + 1] = beta_host[static_cast<std::size_t>(head)];
    }
    CUDA_CHECK(cudaMemset(layer_records.conv.data, 0, layer_records.conv.bytes()));
    CUDA_CHECK(cudaMemcpy(layer_records.key.data, key_record_host.data(),
                          layer_records.key.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer_records.value.data, value_record_host.data(),
                          layer_records.value.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer_records.gate.data, gate_record_host.data(),
                          layer_records.gate.bytes(), cudaMemcpyHostToDevice));

    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout = plan_linear_attention_state_pool(
        state_builder, {.layers         = static_cast<std::uint32_t>(kLayers),
                        .conv_channels  = kConvChannels,
                        .conv_width     = 3,
                        .value_heads    = kValueHeads,
                        .value_head_dim = kStateDim,
                        .key_head_dim   = kStateDim,
                        .slot_count     = 2,
                        .conv_dtype     = dtype});
    DeviceBuffer state_storage(state_builder.finish(256));
    state_storage.fill(0);
    LinearAttentionStatePool pool({state_storage.p, state_storage.bytes}, state_layout);
    CUDA_CHECK(cudaMemcpy(pool.recurrent_slot(0, 0).data, state_host.data(), snapshot_state.bytes,
                          cudaMemcpyHostToDevice));

    Tensor q2_tensor(q2.p, dtype, {kStateDim, kQkHeads, kWidth, 1});
    Tensor k2_tensor(k2.p, dtype, {kStateDim, kQkHeads, kWidth, 1});
    Tensor v2_tensor(v2.p, dtype, {kStateDim, kValueHeads, kWidth, 1});
    Tensor g2_tensor(g2.p, DType::FP32, {kValueHeads, kWidth, 1});
    Tensor beta2_tensor(beta2.p, DType::FP32, {kValueHeads, kWidth, 1});
    Tensor record_out_tensor(record_out.p, dtype, {kStateDim, kValueHeads, kWidth, 1});
    Tensor valid(one_valid.p, DType::I32, {1});
    ops::gated_delta_net_replay_record(
        q2_tensor, k2_tensor, v2_tensor, g2_tensor, beta2_tensor, kScale,
        pool.all_layers_view().recurrent_layer0, valid, slots, layer_records.key,
        layer_records.value, layer_records.gate, record_out_tensor, nullptr);
    const std::array<ops::GdnReplayFoldRow, 1> fold_rows{{{0, 1}}};
    ops::gdn_replay_fold(records, pool.all_layers_view(), fold_rows, nullptr);
    cuda_synchronize();

    const std::string suffix = " dtype=" + std::string(dtype == DType::FP16 ? "fp16" : "bf16");
    int failures = 0;
    failures += compare_bits("snapshot vs direct state" + suffix, snapshot_state.p, direct_state.p,
                             state_elements, sizeof(float));
    failures += compare_bits("snapshot vs direct out" + suffix, snapshot_out.p, direct_out.p,
                             v_elements, 2);
    failures += compare_bits("snapshot vs record out" + suffix, snapshot_out.p, record_out.p,
                             v_elements, 2);
    failures += compare_bits("snapshot vs record+fold state" + suffix, snapshot_state.p,
                             pool.recurrent_slot(0, 0).data, state_elements, sizeof(float));
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case(DType::BF16, 8101U);
    failures += run_case(DType::FP16, 8102U);
    if (failures == 0) {
        std::cout << "OK gdn decode snapshot/record T=1 bit-exact state and output\n";
    }
    return failures;
}
