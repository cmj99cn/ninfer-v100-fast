#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/scatter.h"

#include "core/device.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::uint16_t> bit_pattern(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> values(count);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state     = state * 1664525u + 1013904223u;
        values[i] = static_cast<std::uint16_t>((state >> 16) ^ static_cast<std::uint32_t>(i));
    }
    return values;
}

std::vector<std::uint32_t> bit_pattern32(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint32_t> values(count);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state     = state * 1664525u + 1013904223u;
        values[i] = state ^ static_cast<std::uint32_t>(i << 11);
    }
    return values;
}

int scatter_case(std::int32_t rows, const std::vector<std::int32_t>& indices,
                 std::int32_t destination_columns) {
    const std::int32_t source_columns = static_cast<std::int32_t>(indices.size());
    const auto source = bit_pattern(static_cast<std::size_t>(rows) * source_columns, 0x1324'68acu);
    const auto destination =
        bit_pattern(static_cast<std::size_t>(rows) * destination_columns, 0x9876'4321u);
    auto expected = destination;
    for (std::int32_t source_column = 0; source_column < source_columns; ++source_column) {
        const std::int32_t destination_column = indices[static_cast<std::size_t>(source_column)];
        for (std::int32_t row = 0; row < rows; ++row) {
            expected[static_cast<std::size_t>(destination_column) * rows + row] =
                source[static_cast<std::size_t>(source_column) * rows + row];
        }
    }

    GuardedDeviceBuffer device_source(source.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_indices(indices.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_destination(destination.size() * sizeof(std::uint16_t));
    device_source.copy_from_host(source.data(), source.size() * sizeof(std::uint16_t));
    device_indices.copy_from_host(indices.data(), indices.size() * sizeof(std::int32_t));
    device_destination.copy_from_host(destination.data(),
                                      destination.size() * sizeof(std::uint16_t));

    Tensor source_tensor(device_source.data(), DType::BF16, {rows, source_columns});
    Tensor indices_tensor(device_indices.data(), DType::I32, {source_columns});
    Tensor destination_tensor(device_destination.data(), DType::BF16, {rows, destination_columns});
    ops::scatter(source_tensor, indices_tensor, destination_tensor, nullptr);
    cuda_synchronize();

    const std::string label =
        "scatter D=" + std::to_string(rows) + " V=" + std::to_string(source_columns);
    int failures = 0;
    failures += verify_exact(
        label.c_str(), from_device<std::uint16_t>(device_destination.data(), destination.size()),
        expected);
    failures +=
        verify_exact((label + " preserves source").c_str(),
                     from_device<std::uint16_t>(device_source.data(), source.size()), source);
    failures +=
        verify_exact((label + " preserves indices").c_str(),
                     from_device<std::int32_t>(device_indices.data(), indices.size()), indices);
    failures += device_source.verify_guards((label + " source").c_str());
    failures += device_indices.verify_guards((label + " indices").c_str());
    failures += device_destination.verify_guards((label + " destination").c_str());
    return failures;
}

int extract_case(std::int32_t source_rows, std::int32_t destination_rows,
                 std::int32_t source_offset, std::int32_t tokens) {
    const auto source = bit_pattern(static_cast<std::size_t>(source_rows) * tokens, 0x1357'9bdfu);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(destination_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < destination_rows; ++row) {
            expected[static_cast<std::size_t>(token) * destination_rows + row] =
                source[static_cast<std::size_t>(token) * source_rows + source_offset + row];
        }
    }

    GuardedDeviceBuffer device_source(source.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_destination(expected.size() * sizeof(std::uint16_t));
    device_source.copy_from_host(source.data(), source.size() * sizeof(std::uint16_t));
    device_destination.fill(0xcd);
    Tensor source_tensor(device_source.data(), DType::BF16, {source_rows, tokens});
    Tensor destination_tensor(device_destination.data(), DType::BF16, {destination_rows, tokens});
    ops::extract_bf16_columns(source_tensor, source_offset, destination_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "extract_bf16_columns source=" + std::to_string(source_rows) +
                              " offset=" + std::to_string(source_offset) +
                              " rows=" + std::to_string(destination_rows) +
                              " T=" + std::to_string(tokens);
    int failures = 0;
    failures += verify_exact(label.c_str(),
                             from_device<std::uint16_t>(device_destination.data(), expected.size()),
                             expected);
    failures +=
        verify_exact((label + " preserves source").c_str(),
                     from_device<std::uint16_t>(device_source.data(), source.size()), source);
    failures += device_source.verify_guards((label + " source").c_str());
    failures += device_destination.verify_guards((label + " destination").c_str());
    return failures;
}

int batch_prefix_case() {
    constexpr std::int32_t rows     = 8;
    constexpr std::int32_t width    = 3;
    constexpr std::int32_t batch    = 2;
    constexpr std::int32_t capacity = 3;
    const std::vector<std::int32_t> lanes{2, 0};
    const std::vector<std::int32_t> valid{2, 3};
    const std::vector<std::int32_t> starts{10, 20};
    const std::vector<std::int32_t> ends{12, 23};
    const auto source = bit_pattern(static_cast<std::size_t>(rows * width * batch), 0x2468'ace0u);
    const auto initial =
        bit_pattern(static_cast<std::size_t>(rows * width * capacity), 0x3141'5926u);
    auto expected_pool = initial;
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t column = 0; column < valid[static_cast<std::size_t>(b)]; ++column) {
            for (std::int32_t row = 0; row < rows; ++row) {
                expected_pool[static_cast<std::size_t>(lanes[static_cast<std::size_t>(b)] * width +
                                                       column) *
                                  rows +
                              row] =
                    source[static_cast<std::size_t>(b * width + column) * rows + row];
            }
        }
    }

    DeviceBuffer device_source = to_device(source);
    DeviceBuffer device_lanes  = to_device(lanes);
    DeviceBuffer device_valid  = to_device(valid);
    DeviceBuffer device_starts = to_device(starts);
    DeviceBuffer device_ends   = to_device(ends);
    GuardedDeviceBuffer device_pool(initial.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_gathered(source.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_positions(static_cast<std::size_t>(width * batch) *
                                         sizeof(std::int32_t));
    GuardedDeviceBuffer device_counts(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    device_pool.copy_from_host(initial.data(), initial.size() * sizeof(std::uint16_t));
    device_gathered.fill(0xcd);
    device_positions.fill(0xef);
    device_counts.fill(0xab);

    Tensor source_tensor(device_source.p, DType::BF16, {rows, width, batch});
    Tensor lanes_tensor(device_lanes.p, DType::I32, {batch});
    Tensor valid_tensor(device_valid.p, DType::I32, {batch});
    Tensor pool_tensor(device_pool.data(), DType::BF16, {rows, width, capacity});
    ops::scatter_bf16_batch(source_tensor, lanes_tensor, valid_tensor, pool_tensor, nullptr);

    Tensor starts_tensor(device_starts.p, DType::I32, {batch});
    Tensor ends_tensor(device_ends.p, DType::I32, {batch});
    Tensor gathered_tensor(device_gathered.data(), DType::BF16, {rows, width, batch});
    Tensor positions_tensor(device_positions.data(), DType::I32, {width, batch});
    Tensor counts_tensor(device_counts.data(), DType::I32, {batch});
    ops::prepare_ragged_prefix(pool_tensor, lanes_tensor, starts_tensor, ends_tensor,
                               gathered_tensor, positions_tensor, counts_tensor, nullptr);
    cuda_synchronize();

    auto expected_gathered = source;
    std::fill(expected_gathered.begin() + static_cast<std::ptrdiff_t>(2 * rows),
              expected_gathered.begin() + static_cast<std::ptrdiff_t>(3 * rows), 0);
    const std::vector<std::int32_t> expected_positions{10, 11, 11, 20, 21, 22};

    int failures = verify_exact(
        "scatter_bf16_batch B=2 pool",
        from_device<std::uint16_t>(device_pool.data(), expected_pool.size()), expected_pool);
    failures +=
        verify_exact("prepare_ragged_prefix B=2 values",
                     from_device<std::uint16_t>(device_gathered.data(), expected_gathered.size()),
                     expected_gathered);
    failures +=
        verify_exact("prepare_ragged_prefix B=2 positions",
                     from_device<std::int32_t>(device_positions.data(), expected_positions.size()),
                     expected_positions);
    failures += verify_exact("prepare_ragged_prefix B=2 counts",
                             from_device<std::int32_t>(device_counts.data(), valid.size()), valid);
    failures += device_pool.verify_guards("scatter_bf16_batch B=2 pool guards");
    failures += device_gathered.verify_guards("prepare_ragged_prefix B=2 values guards");
    failures += device_positions.verify_guards("prepare_ragged_prefix B=2 positions guards");
    failures += device_counts.verify_guards("prepare_ragged_prefix B=2 counts guards");
    return failures;
}

int stage_gdn_case() {
    constexpr std::int32_t kKey    = 2048;
    constexpr std::int32_t kValue  = 6144;
    constexpr std::int32_t kChannels = 2 * kKey + kValue;
    constexpr std::int32_t kHeads  = 24;
    constexpr std::int32_t kRank   = 1;
    const auto qb = bit_pattern(kKey, 0x11u);
    const auto kb = bit_pattern(kKey, 0x22u);
    const auto vb = bit_pattern(kValue, 0x33u);
    const auto gb = bit_pattern32(2 * kHeads, 0x44u);
    const auto bb = bit_pattern32(2 * kHeads, 0x55u);

    GuardedDeviceBuffer dq(qb.size() * 2), dk(kb.size() * 2), dv(vb.size() * 2);
    GuardedDeviceBuffer dg(gb.size() * 4), db(bb.size() * 4);
    GuardedDeviceBuffer dci(kChannels * 2), dgr(kHeads * 4), dbr(kHeads * 4);
    dq.copy_from_host(qb.data(), dq.bytes());
    dk.copy_from_host(kb.data(), dk.bytes());
    dv.copy_from_host(vb.data(), dv.bytes());
    dg.copy_from_host(gb.data(), dg.bytes());
    db.copy_from_host(bb.data(), db.bytes());

    Tensor qt(dq.data(), DType::FP16, {kKey, 1});
    Tensor kt(dk.data(), DType::FP16, {kKey, 1});
    Tensor vt(dv.data(), DType::FP16, {kValue, 1});
    Tensor gt(dg.data(), DType::FP32, {2 * kHeads, 1});
    Tensor bt(db.data(), DType::FP32, {2 * kHeads, 1});
    Tensor cit(dci.data(), DType::FP16, {kChannels, 1});
    Tensor grt(dgr.data(), DType::FP32, {kHeads, 1});
    Tensor brt(dbr.data(), DType::FP32, {kHeads, 1});
    ops::stage_gdn_decode(qt, kt, vt, gt, bt, kRank, cit, grt, brt, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> expected_ci(kChannels);
    for (std::int32_t i = 0; i < kKey; ++i) expected_ci[i] = qb[i];
    for (std::int32_t i = 0; i < kKey; ++i) expected_ci[kKey + i] = kb[i];
    for (std::int32_t i = 0; i < kValue; ++i) expected_ci[2 * kKey + i] = vb[i];
    std::vector<std::uint32_t> expected_g(kHeads), expected_b(kHeads);
    for (std::int32_t i = 0; i < kHeads; ++i) {
        expected_g[i] = gb[kRank * kHeads + i];
        expected_b[i] = bb[kRank * kHeads + i];
    }
    int failures = 0;
    failures += verify_exact("stage_gdn_decode conv_in",
                             from_device<std::uint16_t>(dci.data(), expected_ci.size()),
                             expected_ci);
    failures += verify_exact("stage_gdn_decode g_rank",
                             from_device<std::uint32_t>(dgr.data(), expected_g.size()),
                             expected_g);
    failures += verify_exact("stage_gdn_decode beta_rank",
                              from_device<std::uint32_t>(dbr.data(), expected_b.size()),
                              expected_b);
    return failures;
}

// TP2 rank geometry: 8 key heads x 128 + 24 value heads x 128 = 5120 conv channels and 48
// value heads in the full g/beta tensor. The baseline is the five original cudaMemcpy2DAsync
// calls; the fused result must be raw-bit identical to both that baseline and the host oracle.
int stage_gdn_span_case(std::int32_t tokens) {
    constexpr std::int32_t kKey       = 1024;
    constexpr std::int32_t kValue     = 3072;
    constexpr std::int32_t kChannels  = 2 * kKey + kValue;
    constexpr std::int32_t kHeads     = 24;
    constexpr std::int32_t kFullHeads = 48;
    constexpr std::int32_t kRank      = 1;
    const std::size_t q_count = static_cast<std::size_t>(kKey) * tokens;
    const std::size_t v_count = static_cast<std::size_t>(kValue) * tokens;
    const auto qb = bit_pattern(q_count, 0x1111'0101u);
    const auto kb = bit_pattern(q_count, 0x2222'0202u);
    const auto vb = bit_pattern(v_count, 0x3333'0303u);
    const auto gb = bit_pattern32(static_cast<std::size_t>(kFullHeads) * tokens, 0x4444'0404u);
    const auto bb = bit_pattern32(static_cast<std::size_t>(kFullHeads) * tokens, 0x5555'0505u);

    GuardedDeviceBuffer dq(q_count * 2), dk(q_count * 2), dv(v_count * 2);
    GuardedDeviceBuffer dg(gb.size() * 4), db(bb.size() * 4);
    dq.copy_from_host(qb.data(), dq.bytes());
    dk.copy_from_host(kb.data(), dk.bytes());
    dv.copy_from_host(vb.data(), dv.bytes());
    dg.copy_from_host(gb.data(), dg.bytes());
    db.copy_from_host(bb.data(), db.bytes());

    GuardedDeviceBuffer ref_ci(static_cast<std::size_t>(kChannels) * tokens * 2);
    GuardedDeviceBuffer ref_gr(static_cast<std::size_t>(kHeads) * tokens * 4);
    GuardedDeviceBuffer ref_br(static_cast<std::size_t>(kHeads) * tokens * 4);
    ref_ci.fill(0xcd);
    ref_gr.fill(0xcd);
    ref_br.fill(0xcd);
    const auto copy_activation = [&](const void* source, std::int32_t width,
                                     std::int32_t column) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            static_cast<std::byte*>(ref_ci.data()) + static_cast<std::size_t>(column) * 2,
            static_cast<std::size_t>(kChannels) * 2, source,
            static_cast<std::size_t>(width) * 2, static_cast<std::size_t>(width) * 2,
            static_cast<std::size_t>(tokens), cudaMemcpyDeviceToDevice, nullptr));
    };
    copy_activation(dq.data(), kKey, 0);
    copy_activation(dk.data(), kKey, kKey);
    copy_activation(dv.data(), kValue, 2 * kKey);
    CUDA_CHECK(cudaMemcpy2DAsync(
        ref_gr.data(), static_cast<std::size_t>(kHeads) * 4,
        static_cast<const std::byte*>(dg.data()) + static_cast<std::size_t>(kRank * kHeads) * 4,
        static_cast<std::size_t>(kFullHeads) * 4, static_cast<std::size_t>(kHeads) * 4,
        static_cast<std::size_t>(tokens), cudaMemcpyDeviceToDevice, nullptr));
    CUDA_CHECK(cudaMemcpy2DAsync(
        ref_br.data(), static_cast<std::size_t>(kHeads) * 4,
        static_cast<const std::byte*>(db.data()) + static_cast<std::size_t>(kRank * kHeads) * 4,
        static_cast<std::size_t>(kFullHeads) * 4, static_cast<std::size_t>(kHeads) * 4,
        static_cast<std::size_t>(tokens), cudaMemcpyDeviceToDevice, nullptr));

    GuardedDeviceBuffer ci(static_cast<std::size_t>(kChannels) * tokens * 2);
    GuardedDeviceBuffer gr(static_cast<std::size_t>(kHeads) * tokens * 4);
    GuardedDeviceBuffer br(static_cast<std::size_t>(kHeads) * tokens * 4);
    ci.fill(0xcd);
    gr.fill(0xcd);
    br.fill(0xcd);

    Tensor qt(dq.data(), DType::FP16, {kKey, tokens});
    Tensor kt(dk.data(), DType::FP16, {kKey, tokens});
    Tensor vt(dv.data(), DType::FP16, {kValue, tokens});
    Tensor gt(dg.data(), DType::FP32, {kFullHeads, tokens});
    Tensor bt(db.data(), DType::FP32, {kFullHeads, tokens});
    Tensor cit(ci.data(), DType::FP16, {kChannels, tokens});
    Tensor grt(gr.data(), DType::FP32, {kHeads, tokens});
    Tensor brt(br.data(), DType::FP32, {kHeads, tokens});
    ops::stage_gdn_span(qt, kt, vt, gt, bt, kRank, cit, grt, brt, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> expected_ci(static_cast<std::size_t>(kChannels) * tokens);
    std::vector<std::uint32_t> expected_g(static_cast<std::size_t>(kHeads) * tokens);
    std::vector<std::uint32_t> expected_b(static_cast<std::size_t>(kHeads) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t i = 0; i < kChannels; ++i) {
            const auto offset = static_cast<std::size_t>(token) * kChannels + i;
            expected_ci[offset] = i < kKey
                                      ? qb[static_cast<std::size_t>(token) * kKey + i]
                                      : (i < 2 * kKey
                                             ? kb[static_cast<std::size_t>(token) * kKey + i - kKey]
                                             : vb[static_cast<std::size_t>(token) * kValue + i -
                                                  2 * kKey]);
        }
        for (std::int32_t head = 0; head < kHeads; ++head) {
            const auto offset = static_cast<std::size_t>(token) * kHeads + head;
            expected_g[offset] = gb[static_cast<std::size_t>(token) * kFullHeads + kRank * kHeads +
                                    head];
            expected_b[offset] = bb[static_cast<std::size_t>(token) * kFullHeads + kRank * kHeads +
                                    head];
        }
    }

    const std::string label = "stage_gdn_span T=" + std::to_string(tokens);
    const auto fused_ci = from_device<std::uint16_t>(ci.data(), expected_ci.size());
    const auto copied_ci = from_device<std::uint16_t>(ref_ci.data(), expected_ci.size());
    const auto fused_g = from_device<std::uint32_t>(gr.data(), expected_g.size());
    const auto copied_g = from_device<std::uint32_t>(ref_gr.data(), expected_g.size());
    const auto fused_b = from_device<std::uint32_t>(br.data(), expected_b.size());
    const auto copied_b = from_device<std::uint32_t>(ref_br.data(), expected_b.size());
    int failures = 0;
    failures += verify_exact((label + " conv_in vs oracle").c_str(), fused_ci, expected_ci);
    failures += verify_exact((label + " g_rank vs oracle").c_str(), fused_g, expected_g);
    failures += verify_exact((label + " beta_rank vs oracle").c_str(), fused_b, expected_b);
    failures += verify_exact((label + " conv_in fused vs memcpy2D").c_str(), fused_ci, copied_ci);
    failures += verify_exact((label + " g_rank fused vs memcpy2D").c_str(), fused_g, copied_g);
    failures += verify_exact((label + " beta_rank fused vs memcpy2D").c_str(), fused_b, copied_b);
    failures += verify_exact((label + " preserves q").c_str(),
                             from_device<std::uint16_t>(dq.data(), q_count), qb);
    failures += verify_exact((label + " preserves k").c_str(),
                             from_device<std::uint16_t>(dk.data(), q_count), kb);
    failures += verify_exact((label + " preserves v").c_str(),
                             from_device<std::uint16_t>(dv.data(), v_count), vb);
    failures += dq.verify_guards(label + " q guards");
    failures += dk.verify_guards(label + " k guards");
    failures += dv.verify_guards(label + " v guards");
    failures += dg.verify_guards(label + " g guards");
    failures += db.verify_guards(label + " beta guards");
    failures += ci.verify_guards(label + " conv_in guards");
    failures += gr.verify_guards(label + " g_rank guards");
    failures += br.verify_guards(label + " beta_rank guards");
    failures += ref_ci.verify_guards(label + " baseline conv_in guards");
    failures += ref_gr.verify_guards(label + " baseline g_rank guards");
    failures += ref_br.verify_guards(label + " baseline beta_rank guards");
    return failures;
}

int extract_gdn_columns_case(std::int32_t tokens) {
    constexpr std::int32_t kKey      = 1024;
    constexpr std::int32_t kValue    = 3072;
    constexpr std::int32_t kChannels = 2 * kKey + kValue;
    const auto conv = bit_pattern(static_cast<std::size_t>(kChannels) * tokens, 0x7777'0707u);
    GuardedDeviceBuffer dconv(conv.size() * 2);
    dconv.copy_from_host(conv.data(), dconv.bytes());

    GuardedDeviceBuffer ref_q(static_cast<std::size_t>(kKey) * tokens * 2);
    GuardedDeviceBuffer ref_k(static_cast<std::size_t>(kKey) * tokens * 2);
    GuardedDeviceBuffer ref_v(static_cast<std::size_t>(kValue) * tokens * 2);
    GuardedDeviceBuffer q(static_cast<std::size_t>(kKey) * tokens * 2);
    GuardedDeviceBuffer k(static_cast<std::size_t>(kKey) * tokens * 2);
    GuardedDeviceBuffer v(static_cast<std::size_t>(kValue) * tokens * 2);
    ref_q.fill(0xcd);
    ref_k.fill(0xcd);
    ref_v.fill(0xcd);
    q.fill(0xcd);
    k.fill(0xcd);
    v.fill(0xcd);

    Tensor conv_tensor(dconv.data(), DType::FP16, {kChannels, tokens});
    Tensor ref_qt(ref_q.data(), DType::FP16, {kKey, tokens});
    Tensor ref_kt(ref_k.data(), DType::FP16, {kKey, tokens});
    Tensor ref_vt(ref_v.data(), DType::FP16, {kValue, tokens});
    ops::extract_bf16_columns(conv_tensor, 0, ref_qt, nullptr);
    ops::extract_bf16_columns(conv_tensor, kKey, ref_kt, nullptr);
    ops::extract_bf16_columns(conv_tensor, 2 * kKey, ref_vt, nullptr);
    Tensor qt(q.data(), DType::FP16, {kKey, tokens});
    Tensor kt(k.data(), DType::FP16, {kKey, tokens});
    Tensor vt(v.data(), DType::FP16, {kValue, tokens});
    ops::extract_gdn_columns(conv_tensor, qt, kt, vt, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> expected_q(static_cast<std::size_t>(kKey) * tokens);
    std::vector<std::uint16_t> expected_k(static_cast<std::size_t>(kKey) * tokens);
    std::vector<std::uint16_t> expected_v(static_cast<std::size_t>(kValue) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t i = 0; i < kKey; ++i) {
            expected_q[static_cast<std::size_t>(token) * kKey + i] =
                conv[static_cast<std::size_t>(token) * kChannels + i];
            expected_k[static_cast<std::size_t>(token) * kKey + i] =
                conv[static_cast<std::size_t>(token) * kChannels + kKey + i];
        }
        for (std::int32_t i = 0; i < kValue; ++i) {
            expected_v[static_cast<std::size_t>(token) * kValue + i] =
                conv[static_cast<std::size_t>(token) * kChannels + 2 * kKey + i];
        }
    }

    const std::string label = "extract_gdn_columns T=" + std::to_string(tokens);
    const auto fused_q = from_device<std::uint16_t>(q.data(), expected_q.size());
    const auto copied_q = from_device<std::uint16_t>(ref_q.data(), expected_q.size());
    const auto fused_k = from_device<std::uint16_t>(k.data(), expected_k.size());
    const auto copied_k = from_device<std::uint16_t>(ref_k.data(), expected_k.size());
    const auto fused_v = from_device<std::uint16_t>(v.data(), expected_v.size());
    const auto copied_v = from_device<std::uint16_t>(ref_v.data(), expected_v.size());
    int failures = 0;
    failures += verify_exact((label + " q vs oracle").c_str(), fused_q, expected_q);
    failures += verify_exact((label + " k vs oracle").c_str(), fused_k, expected_k);
    failures += verify_exact((label + " v vs oracle").c_str(), fused_v, expected_v);
    failures += verify_exact((label + " q fused vs memcpy2D").c_str(), fused_q, copied_q);
    failures += verify_exact((label + " k fused vs memcpy2D").c_str(), fused_k, copied_k);
    failures += verify_exact((label + " v fused vs memcpy2D").c_str(), fused_v, copied_v);
    failures += verify_exact((label + " preserves source").c_str(),
                             from_device<std::uint16_t>(dconv.data(), conv.size()), conv);
    failures += dconv.verify_guards(label + " source guards");
    failures += q.verify_guards(label + " q guards");
    failures += k.verify_guards(label + " k guards");
    failures += v.verify_guards(label + " v guards");
    failures += ref_q.verify_guards(label + " ref q guards");
    failures += ref_k.verify_guards(label + " ref k guards");
    failures += ref_v.verify_guards(label + " ref v guards");
    return failures;
}

} // namespace


int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += scatter_case(5120, {4, 0, 7, 2}, 9);
    failures += scatter_case(2048, {5, 1, 3}, 7);
    failures += extract_case(10240, 6144, 4096, 6);
    failures += extract_case(8192, 2048, 2048, 1);
    failures += batch_prefix_case();
    failures += stage_gdn_case();
    failures += stage_gdn_span_case(2);
    failures += stage_gdn_span_case(4);
    failures += extract_gdn_columns_case(2);
    failures += extract_gdn_columns_case(4);
    std::cout << (failures ? "FAIL" : "OK")
              << " scatter, extract_bf16_columns, fused GDN span stage/extract\n";
    return failures ? 1 : 0;
}
