#include "ninfer/ops/candidate_selector.h"

#include "core/arena.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops;

namespace {

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

float uniform_at(std::uint64_t seed, int position, int purpose, unsigned sub) {
    std::uint64_t key = seed;
    key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<unsigned>(position)) *
                            0xD1B54A32D192ED03ull));
    key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<unsigned>(purpose)) << 21) ^
                     (static_cast<std::uint64_t>(sub) * 0x2545F4914F6CDD1Dull));
    const unsigned bits = static_cast<unsigned>(key >> 40);
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

int run_topk(int rows, int columns, bool fp32) {
    std::vector<float> logits(static_cast<std::size_t>(rows) * columns);
    for (int c = 0; c < columns; ++c) {
        for (int r = 0; r < rows; ++r) {
            // Deliberate duplicates exercise the lower-id tie-break.
            logits[static_cast<std::size_t>(c) * rows + r] =
                static_cast<float>(((r * 37 + c * 11) % 29) - 14) / 4.0f;
        }
    }
    if (!fp32) { round_to_bf16(logits); }
    std::vector<int> expected_ids(16 * columns);
    std::vector<float> expected_scores(16 * columns);
    for (int c = 0; c < columns; ++c) {
        std::vector<int> ids(rows);
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin() + 16, ids.end(), [&](int lhs, int rhs) {
            const float a = logits[static_cast<std::size_t>(c) * rows + lhs];
            const float b = logits[static_cast<std::size_t>(c) * rows + rhs];
            return a > b || (a == b && lhs < rhs);
        });
        for (int k = 0; k < 16; ++k) {
            expected_ids[static_cast<std::size_t>(c) * 16 + k] = ids[k];
            expected_scores[static_cast<std::size_t>(c) * 16 + k] =
                logits[static_cast<std::size_t>(c) * rows + ids[k]];
        }
    }
    DeviceBuffer d_logits = fp32 ? to_device(logits) : to_device_bf16(logits);
    DeviceBuffer d_ids(static_cast<std::size_t>(16 * columns) * sizeof(std::int32_t));
    DeviceBuffer d_scores(static_cast<std::size_t>(16 * columns) * sizeof(float));
    Tensor logits_t(d_logits.p, fp32 ? DType::FP32 : DType::BF16, {rows, columns});
    Tensor ids_t(d_ids.p, DType::I32, {16, columns});
    Tensor scores_t(d_scores.p, DType::FP32, {16, columns});
    DeviceArena workspace(ops::candidate_topk_workspace_capacity_bytes(columns));
    ops::candidate_topk(logits_t, rows, ids_t, scores_t, workspace, nullptr);
    cuda_synchronize();
    int failures =
        verify_exact("candidate_topk ids", from_device<int>(d_ids, 16 * columns), expected_ids);
    std::vector<double> got = from_device_f32(d_scores, 16 * columns);
    std::vector<double> ref(expected_scores.begin(), expected_scores.end());
    failures += verify_pointwise("candidate_topk scores", got, ref,
                                 PointwiseCriterion{.absolute = 1e-6, .relative = 1e-6});
    return failures;
}

struct SelectorCase {
    int steps;
    int batch;
};

int run_selector(int steps, int batch, float temperature) {
    constexpr int kRank = 256;
    constexpr int kRows = 64;
    std::vector<float> unary(static_cast<std::size_t>(16 * steps * batch));
    std::vector<float> hidden(static_cast<std::size_t>(kRank * steps * batch));
    std::vector<float> predecessor(static_cast<std::size_t>(kRows) * kRank);
    std::vector<float> successor(static_cast<std::size_t>(kRows) * kRank);
    std::vector<int> ids(static_cast<std::size_t>(16 * steps * batch));
    std::vector<int> anchors(batch);
    std::vector<int> positions(batch);
    for (int b = 0; b < batch; ++b) {
        anchors[b]   = (3 + b * 17) % kRows;
        positions[b] = 100 + b * 31;
    }
    std::uint32_t state = 12345u;
    auto next = [&]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < steps; ++k) {
            const int column = b * steps + k;
            for (int c = 0; c < 16; ++c) {
                ids[static_cast<std::size_t>(column) * 16 + c] = static_cast<int>(next() % kRows);
                unary[static_cast<std::size_t>(column) * 16 + c] =
                    static_cast<float>(static_cast<int>(next() % 401) - 200) / 64.0f;
            }
            for (int r = 0; r < kRank; ++r) {
                hidden[static_cast<std::size_t>(column) * kRank + r] =
                    static_cast<float>(static_cast<int>(next() % 101) - 50) / 128.0f;
            }
        }
    }
    for (std::size_t i = 0; i < predecessor.size(); ++i) {
        predecessor[i] = static_cast<float>(static_cast<int>(next() % 61) - 30) / 256.0f;
        successor[i]   = static_cast<float>(static_cast<int>(next() % 73) - 36) / 256.0f;
    }
    round_to_bf16(hidden);
    round_to_bf16(predecessor);
    round_to_bf16(successor);

    std::vector<SamplingConfig> configs(batch);
    for (int b = 0; b < batch; ++b) {
        configs[b].temperature = temperature;
        configs[b].seed        = 0x1234567890abcdefull + b;
        configs[b].top_k       = 0;
    }

    // Independent host oracle over the represented values.
    std::vector<int> expected_drafts(steps * batch);
    std::vector<double> expected_q(16 * steps * batch);
    std::vector<double> expected_confidence(steps * batch);
    for (int b = 0; b < batch; ++b) {
        int parent = anchors[b];
        for (int k = 0; k < steps; ++k) {
            const int column = b * steps + k;
            std::array<double, 16> edge{};
            for (int c = 0; c < 16; ++c) {
                const int token = ids[static_cast<std::size_t>(column) * 16 + c];
                double score    = unary[static_cast<std::size_t>(column) * 16 + c];
                for (int r = 0; r < kRank; ++r) {
                    score += static_cast<double>(
                                 predecessor[static_cast<std::size_t>(parent) * kRank + r]) *
                             static_cast<double>(
                                 hidden[static_cast<std::size_t>(column) * kRank + r]) *
                             static_cast<double>(
                                 successor[static_cast<std::size_t>(token) * kRank + r]);
                }
                edge[c] = score;
            }
            const double maximum = *std::max_element(edge.begin(), edge.end());
            std::array<double, 16> unit_weight{};
            double unit_total = 0.0;
            for (int c = 0; c < 16; ++c) {
                unit_weight[c] = std::exp(edge[c] - maximum);
                unit_total += unit_weight[c];
            }
            int selected = 0;
            if (temperature <= 0.0f) {
                for (int c = 1; c < 16; ++c) {
                    if (edge[c] > edge[selected]) { selected = c; }
                }
                for (int c = 0; c < 16; ++c) {
                    expected_q[column * 16 + c] = c == selected ? 1.0 : 0.0;
                }
            } else {
                double total = 0.0;
                std::array<double, 16> weight{};
                for (int c = 0; c < 16; ++c) {
                    weight[c] = std::exp((edge[c] - maximum) / temperature);
                    total += weight[c];
                }
                double cumulative = 0.0;
                const double u = uniform_at(configs[b].seed, positions[b] + k,
                                            ops::kSamplePurposeDFlash2Proposal, 0);
                selected = 15;
                bool hit = false;
                for (int c = 0; c < 16; ++c) {
                    expected_q[column * 16 + c] = weight[c] / total;
                    cumulative += weight[c] / total;
                    if (!hit && u < cumulative) {
                        selected = c;
                        hit      = true;
                    }
                }
            }
            expected_confidence[column] = unit_weight[selected] / unit_total;
            parent = ids[static_cast<std::size_t>(column) * 16 + selected];
            expected_drafts[column] = parent;
        }
    }

    DeviceBuffer d_unary = to_device(unary);
    DeviceBuffer d_hidden = to_device_bf16(hidden);
    DeviceBuffer d_pred = to_device_bf16(predecessor);
    DeviceBuffer d_succ = to_device_bf16(successor);
    DeviceBuffer d_ids = to_device(ids);
    DeviceBuffer d_anchors = to_device(anchors);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_configs = to_device(configs);
    DeviceBuffer d_drafts(static_cast<std::size_t>(steps * batch) * sizeof(std::int32_t));
    DeviceBuffer d_q(static_cast<std::size_t>(16 * steps * batch) * sizeof(float));
    DeviceBuffer d_confidence(static_cast<std::size_t>(steps * batch) * sizeof(float));

    Tensor unary_t(d_unary.p, DType::FP32, {16, steps, batch});
    Tensor hidden_t(d_hidden.p, DType::BF16, {kRank, steps, batch});
    Tensor pred_t(d_pred.p, DType::BF16, {kRows, kRank});
    Tensor succ_t(d_succ.p, DType::BF16, {kRows, kRank});
    Tensor ids_t(d_ids.p, DType::I32, {16, steps, batch});
    Tensor anchors_t(d_anchors.p, DType::I32, {batch});
    Tensor positions_t(d_positions.p, DType::I32, {batch});
    Tensor drafts_t(d_drafts.p, DType::I32, {steps, batch});
    Tensor q_t(d_q.p, DType::FP32, {16, steps, batch});
    Tensor confidence_t(d_confidence.p, DType::FP32, {steps, batch});
    DeviceArena workspace(std::max<std::size_t>(
        1, ops::candidate_selector_path_workspace_capacity_bytes(steps, steps, batch, batch)));
    ops::candidate_selector_path(ids_t, unary_t, hidden_t, anchors_t, pred_t, succ_t, positions_t,
                                 static_cast<const SamplingConfig*>(d_configs.p), drafts_t, q_t,
                                 workspace, nullptr, &confidence_t);
    cuda_synchronize();
    int failures =
        verify_exact("candidate_selector drafts", from_device<int>(d_drafts, steps * batch),
                     expected_drafts);
    failures += verify_pointwise("candidate_selector proposal_q", from_device_f32(d_q, 16 * steps * batch),
                                 expected_q, PointwiseCriterion{.absolute = 2e-5, .relative = 2e-4});
    failures += verify_pointwise("candidate_selector confidence",
                                 from_device_f32(d_confidence, steps * batch), expected_confidence,
                                 PointwiseCriterion{.absolute = 2e-5, .relative = 2e-4});
    return failures;
}

int run_greedy_tie() {
    // Zero codebooks and hidden make every edge equal to its unary score, so slots 3 and 7 tie.
    constexpr int kRank = 256;
    constexpr int kRows = 32;
    constexpr int steps = 1;
    std::vector<float> unary(16, 0.0f);
    unary[3] = 5.0f;
    unary[7] = 5.0f;
    std::vector<float> zero(static_cast<std::size_t>(kRank) * steps, 0.0f);
    std::vector<float> codebook(static_cast<std::size_t>(kRows) * kRank, 0.0f);
    std::vector<int> ids(16);
    for (int c = 0; c < 16; ++c) { ids[c] = c % kRows; }
    // Slots 3 and 7 both name token 3 (same id) so the tie is a genuine slot tie.
    ids[3] = 3;
    ids[7] = 3;
    std::vector<int> anchors{0};
    std::vector<int> positions{0};
    SamplingConfig config{};
    config.temperature = 0.0f;
    DeviceBuffer d_unary = to_device(unary);
    DeviceBuffer d_hidden = to_device_bf16(zero);
    DeviceBuffer d_codebook = to_device_bf16(codebook);
    DeviceBuffer d_ids = to_device(ids);
    DeviceBuffer d_anchors = to_device(anchors);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_config = to_device(std::vector<SamplingConfig>{config});
    DeviceBuffer d_drafts(sizeof(std::int32_t));
    DeviceBuffer d_q(16 * sizeof(float));
    Tensor unary_t(d_unary.p, DType::FP32, {16, 1, 1});
    Tensor hidden_t(d_hidden.p, DType::BF16, {kRank, 1, 1});
    Tensor codebook_t(d_codebook.p, DType::BF16, {kRows, kRank});
    Tensor ids_t(d_ids.p, DType::I32, {16, 1, 1});
    Tensor anchors_t(d_anchors.p, DType::I32, {1});
    Tensor positions_t(d_positions.p, DType::I32, {1});
    Tensor drafts_t(d_drafts.p, DType::I32, {1, 1});
    Tensor q_t(d_q.p, DType::FP32, {16, 1, 1});
    DeviceArena workspace(
        std::max<std::size_t>(1, ops::candidate_selector_path_workspace_capacity_bytes(1, 1, 1, 1)));
    ops::candidate_selector_path(ids_t, unary_t, hidden_t, anchors_t, codebook_t, codebook_t,
                                 positions_t, static_cast<const SamplingConfig*>(d_config.p),
                                 drafts_t, q_t, workspace, nullptr);
    cuda_synchronize();
    const int draft = from_device<int>(d_drafts, 1)[0];
    return draft == 3 ? 0 : 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "candidate_selector: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = 0;
    failures += run_topk(1024, 3, false);
    failures += run_topk(248320, 1, false);
    failures += run_topk(777, 2, true);
    failures += run_selector(3, 2, 0.0f);
    failures += run_selector(7, 1, 0.0f);
    failures += run_selector(3, 2, 0.7f);
    failures += run_selector(7, 2, 0.7f);
    failures += run_greedy_tie();
    if (failures != 0) {
        std::cerr << "candidate_selector failures=" << failures << '\n';
        return 1;
    }
    std::cout << "candidate_selector: PASS\n";
    return 0;
}
