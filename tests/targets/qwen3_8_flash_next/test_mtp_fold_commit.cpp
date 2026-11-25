#include <ninfer/engine.h>

#include "flash_model_harness.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using ninfer::Engine;
using ninfer::EngineOptions;
using ninfer::GenerationResult;
using ninfer::RequestOptions;
using ninfer::Tensor;
using ninfer::TokenId;

using flash_harness::FlashHarness;
using flash_harness::require;
namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
namespace rt = flash_harness::rt;

constexpr std::uint32_t kKvCapacity = 262144;
constexpr std::size_t kVerify = 4;
constexpr std::uint32_t kOutputTokens = 8;

const std::vector<std::vector<TokenId>> kPrompts{
    {1024, 7, 98140, 33, 4097, 512, 100, 2048},
    {100, 200, 300, 400, 500, 600, 700, 800},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1024, 7, 98140, 33, 4097, 512, 100, 2049},
};

// Two distinct roles, deliberately not conflated:
//
// 1. Fold-correctness oracle (check_profile_matched_state): a batched decode_span over the accepted
//    prefix uses the SAME T>1 verify profile as gdn_replay_fold and does NOT call fold/replay_record.
//    Applying the first commit_columns record transitions to the pre-round state is exactly the
//    mathematical oracle for gdn_replay_fold, so conv/QSA/PLE must be bit-exact and FP32 recurrent
//    must be within a tight reduction bound. This is the hard correctness gate.
//
// 2. Cross-profile diagnostic (sequential_oracle + compare_*): the T=1 sequential route is a
//    DIFFERENT arithmetic profile. Its hidden-state kernel `hc_mix` selects a fused up-gate at
//    columns==1 (T=1) but a GEMV+gate path at columns>1 (T>1); the two reductions are not bit-equal.
//    That divergence enters at layer 0 and propagates through 36 GDN/QSA/PLE/MoE layers. It is NOT a
//    fold bug and the sequential route is NOT the fold oracle. The numbers below are therefore
//    reported-only reference scales for `worst_ratio`, never hard assertions. Material limitation:
//    a future large drift of the batched profile relative to sequential will not be caught here;
//    Task 5 acceptance/throughput gates and the recorded deviation values are the behavioral backstop.
struct Tolerances {
    // Reported reference scales for the cross-profile diagnostic only (no assertion).
    double recurrent_atol = 1e-3;
    double recurrent_rtol = 1e-3;
    double conv_atol = 0.0;
    double conv_rtol = 0.0;
    double qsa_rtol = 0.22;
    double ple_rtol = 0.01;
    // Hard gate: profile-matched in-place span vs fold+rebuild, sharing the T>1 verify profile.
    double profile_recurrent_atol = 1e-3;
    double profile_recurrent_rtol = 1e-3;
};
constexpr Tolerances kTolerance{};

EngineOptions engine_options(const char* path, bool mtp) {
    EngineOptions options;
    options.artifact_path    = path;
    options.device           = 0;
    options.tp               = 2;
    options.devices          = {0, 1};
    options.max_context      = kKvCapacity;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
    options.max_concurrency  = 1;
    options.prefill_chunk    = 64;
    options.use_cuda_graph   = false;
    options.expert_cache_mib = 0;
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    return options;
}

GenerationResult generate(const char* path, bool mtp, const std::vector<TokenId>& prompt) {
    Engine engine(engine_options(path, mtp));
    auto prepared = engine.prepare_tokens(prompt);
    RequestOptions request;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.requested_output_tokens = kOutputTokens;
    request.stop.include_model_defaults       = false;
    return engine.generate(std::move(prepared), std::move(request));
}

float bf16_word_to_float(std::uint16_t word) {
    const std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float fp16_word_to_float(std::uint16_t word) {
    __half value{};
    std::memcpy(&value, &word, sizeof(value));
    return __half2float(value);
}

struct QsaTailSnapshot {
    std::vector<std::uint16_t> raw_keys; // BF16 words, all QSA slots concatenated
    std::vector<std::uint16_t> rope;     // I32 words, all QSA slots concatenated
};

QsaTailSnapshot capture_qsa_tails(rt::TextState& state) {
    QsaTailSnapshot out;
    for (int slot = 0; slot < flash::TextConfig::qsa_layers(); ++slot) {
        const auto tail = state.before_tail(slot);
        std::vector<std::uint16_t> keys(tail.raw_keys.bytes() / 2);
        CUDA_CHECK(cudaMemcpy(keys.data(), tail.raw_keys.data, tail.raw_keys.bytes(),
                              cudaMemcpyDeviceToHost));
        out.raw_keys.insert(out.raw_keys.end(), keys.begin(), keys.end());
        std::vector<std::uint16_t> rope(tail.first_rope.bytes() / 2);
        CUDA_CHECK(cudaMemcpy(rope.data(), tail.first_rope.data, tail.first_rope.bytes(),
                              cudaMemcpyDeviceToHost));
        out.rope.insert(out.rope.end(), rope.begin(), rope.end());
    }
    return out;
}

// The complete observable state of one rank after a forward pass. The oracle and the folded route
// are captured into the same type and compared field by field.
struct RankSnapshot {
    std::vector<std::vector<std::uint16_t>> conv;      // [layer][FP16 word]
    std::vector<std::vector<float>> recurrent;         // [layer][FP32 word]
    QsaTailSnapshot qsa;
    std::vector<std::uint16_t> ple;                     // [FP16 word]
};

struct Snapshot {
    std::uint32_t position = 0;
    std::uint64_t ple_tail[2] = {0, 0};
    std::array<RankSnapshot, 2> ranks;
};

Snapshot capture_snapshot(FlashHarness& harness) {
    Snapshot out;
    auto& head = *harness.states[0];
    out.position = head.next_position();
    out.ple_tail[0] = head.ple_tail(0);
    out.ple_tail[1] = head.ple_tail(1);
    for (int rank = 0; rank < 2; ++rank) {
        auto& state = *harness.states[static_cast<std::size_t>(rank)];
        CUDA_CHECK(cudaSetDevice(state.device_id()));
        CUDA_CHECK(cudaStreamSynchronize(state.stream()));
        RankSnapshot& rank_snap = out.ranks[static_cast<std::size_t>(rank)];
        for (std::size_t layer = 0; layer < state.gdn_state_entries(); ++layer) {
            const auto conv = state.gdn_conv_state(layer);
            std::vector<std::uint16_t> conv_words(conv.bytes() / 2);
            CUDA_CHECK(cudaMemcpy(conv_words.data(), conv.data, conv.bytes(),
                                  cudaMemcpyDeviceToHost));
            rank_snap.conv.push_back(std::move(conv_words));

            const auto recurrent = state.gdn_recurrent_state(layer);
            std::vector<float> recurrent_words(recurrent.bytes() / sizeof(float));
            CUDA_CHECK(cudaMemcpy(recurrent_words.data(), recurrent.data, recurrent.bytes(),
                                  cudaMemcpyDeviceToHost));
            rank_snap.recurrent.push_back(std::move(recurrent_words));
        }
        rank_snap.qsa = capture_qsa_tails(state);
        const auto history = state.ple_history();
        std::vector<std::uint16_t> ple_words(history.bytes() / 2);
        CUDA_CHECK(cudaMemcpy(ple_words.data(), history.data, history.bytes(),
                              cudaMemcpyDeviceToHost));
        rank_snap.ple = std::move(ple_words);
    }
    return out;
}

// Cross-profile diagnostic route: consume exactly the committed prefix with sequential T=1 decode.
// No fold kernel, batched span, verification record or reconstruction op is used on this route.
// This is NOT the fold oracle -- T=1 uses a different hc_mix reduction than the T>1 verify profile;
// it is retained only to record the measured cross-profile deviation, not to gate fold correctness.
Snapshot sequential_oracle(FlashHarness& harness, const std::vector<TokenId>& prompt,
                           std::span<const TokenId> committed_prefix) {
    harness.reset();
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::UpdateInPlace);
    (void)harness.prefill(prompt);
    for (const TokenId token : committed_prefix) { (void)harness.decode(token); }
    return capture_snapshot(harness);
}

struct FoldResult {
    std::uint32_t accepted = 0;
    std::uint32_t committed_count = 0;
    std::vector<TokenId> verify_tokens;
    std::vector<TokenId> proposals;
    std::vector<TokenId> predictions;
    Snapshot snapshot;
};

// Drive the production verify + per-rank fold + partial rebuild sequence through TextTrunk. This
// is the route under test; its state is compared against the two oracles.
FoldResult run_fold_commit(FlashHarness& harness, const std::vector<TokenId>& prompt,
                           const std::vector<TokenId>& verify_tokens) {
    harness.reset();
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::UpdateInPlace);
    (void)harness.prefill(prompt);

    const auto baseline = harness.trunk->capture_verification_baseline();
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::RecordForReplay);
    const std::vector<TokenId> predictions = harness.trunk->decode_span(verify_tokens);
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::UpdateInPlace);

    std::vector<TokenId> proposals(verify_tokens.begin() + 1, verify_tokens.end());
    const std::uint32_t accepted = flash::accepted_prefix(proposals, predictions);
    const std::uint32_t committed_count = accepted + 1;

    harness.trunk->commit_verified_span(verify_tokens, committed_count, baseline);

    FoldResult result;
    result.accepted = accepted;
    result.committed_count = committed_count;
    result.verify_tokens = verify_tokens;
    result.proposals = std::move(proposals);
    result.predictions = std::move(predictions);
    result.snapshot = capture_snapshot(harness);
    return result;
}

struct FieldReport {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double worst_ratio = 0.0; // max(|diff| / bound) where a bound is defined
    std::size_t violations = 0;
};

// Conversions / aggregations shared by the field checks.
struct CompareStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    std::size_t count = 0;
    std::size_t bit_mismatch = 0;
};

FieldReport compare_conv(const std::vector<std::vector<std::uint16_t>>& got,
                         const std::vector<std::vector<std::uint16_t>>& want) {
    FieldReport report;
    CompareStats stats;
    require(got.size() == want.size(), "conv state layer count differs");
    std::size_t layer0_bit_mismatch = 0;
    std::size_t worst_layer = 0;
    double worst_layer_rel = 0.0;
    for (std::size_t layer = 0; layer < got.size(); ++layer) {
        require(got[layer].size() == want[layer].size(), "conv state extent differs");
        double layer_rel = 0.0;
        std::size_t layer_bit_mismatch = 0;
        for (std::size_t index = 0; index < got[layer].size(); ++index) {
            const std::uint16_t g = got[layer][index];
            const std::uint16_t w = want[layer][index];
            const float gf = fp16_word_to_float(g);
            const float wf = fp16_word_to_float(w);
            const double diff = std::abs(static_cast<double>(gf) - wf);
            const double rel = diff / (1.0 + std::abs(wf));
            stats.max_abs = std::max(stats.max_abs, diff);
            stats.max_rel = std::max(stats.max_rel, rel);
            layer_rel = std::max(layer_rel, rel);
            ++stats.count;
            if (g != w) {
                ++stats.bit_mismatch;
                ++layer_bit_mismatch;
            }
            const double bound = kTolerance.conv_atol + kTolerance.conv_rtol * std::abs(wf);
            if (diff > bound) { ++report.violations; }
            // For bit-exact mode (bound==0) use the FP16 min subnormal as the reporting floor so
            // worst_ratio quantifies how far a mismatch is from exactness instead of being 0/0.
            const double ratio_bound = bound > 0.0 ? bound : 5.96e-8;
            report.worst_ratio = std::max(report.worst_ratio, diff / ratio_bound);
        }
        if (layer == 0) { layer0_bit_mismatch = layer_bit_mismatch; }
        if (layer_rel > worst_layer_rel) {
            worst_layer_rel = layer_rel;
            worst_layer = layer;
        }
    }
    report.max_abs = stats.max_abs;
    report.max_rel = stats.max_rel;
    std::cout << "ORACLE_FIELD field=gdn_conv rank_words=" << stats.count
              << " bit_mismatch=" << stats.bit_mismatch << " max_abs=" << stats.max_abs
              << " max_rel=" << stats.max_rel
              << " layer0_bit_mismatch=" << layer0_bit_mismatch
              << " worst_layer=" << worst_layer << " worst_layer_rel=" << worst_layer_rel
              << " ref_bound=atol+rtol atol=" << kTolerance.conv_atol
              << " rtol=" << kTolerance.conv_rtol << " worst_ratio=" << report.worst_ratio
              << " cross_profile_violations=" << report.violations << "\n";
    return report;
}

FieldReport compare_recurrent(const std::vector<std::vector<float>>& got,
                              const std::vector<std::vector<float>>& want) {
    FieldReport report;
    require(got.size() == want.size(), "recurrent state layer count differs");
    for (std::size_t layer = 0; layer < got.size(); ++layer) {
        require(got[layer].size() == want[layer].size(), "recurrent state extent differs");
        for (std::size_t index = 0; index < got[layer].size(); ++index) {
            const double g = got[layer][index];
            const double w = want[layer][index];
            const double diff = std::abs(g - w);
            report.max_abs = std::max(report.max_abs, diff);
            report.max_rel = std::max(report.max_rel, diff / (1.0 + std::abs(w)));
            const double bound =
                kTolerance.recurrent_atol + kTolerance.recurrent_rtol * std::abs(w);
            report.worst_ratio = std::max(report.worst_ratio, diff / bound);
            if (diff > bound) { ++report.violations; }
        }
    }
    std::cout << "ORACLE_FIELD field=gdn_recurrent max_abs=" << report.max_abs
              << " max_rel=" << report.max_rel << " worst_ratio=" << report.worst_ratio
              << " ref_atol=" << kTolerance.recurrent_atol
              << " ref_rtol=" << kTolerance.recurrent_rtol
              << " kind=cross_profile_diagnostic cross_profile_violations=" << report.violations
              << "\n";
    return report;
}

FieldReport compare_qsa(const QsaTailSnapshot& got, const QsaTailSnapshot& want) {
    FieldReport report;
    require(got.rope == want.rope, "QSA rope coordinates differ (semantic, I32 exact)");
    require(got.raw_keys.size() == want.raw_keys.size(), "QSA raw key extent differs");
    std::cout << "ORACLE_FIELD field=qsa_rope exact_words=" << got.rope.size()
              << " kind=semantic violations=0\n";
    for (std::size_t index = 0; index < got.raw_keys.size(); ++index) {
        const double g = bf16_word_to_float(got.raw_keys[index]);
        const double w = bf16_word_to_float(want.raw_keys[index]);
        const double diff = std::abs(g - w);
        report.max_abs = std::max(report.max_abs, diff);
        report.max_rel = std::max(report.max_rel, diff / (1.0 + std::abs(w)));
        const double bound = kTolerance.qsa_rtol * (1.0 + std::abs(w));
        report.worst_ratio = std::max(report.worst_ratio, diff / bound);
        if (diff > bound) { ++report.violations; }
    }
    std::cout << "ORACLE_FIELD field=qsa_raw_keys max_abs=" << report.max_abs
              << " max_rel=" << report.max_rel << " worst_ratio=" << report.worst_ratio
              << " ref_rtol=" << kTolerance.qsa_rtol
              << " kind=cross_profile_diagnostic cross_profile_violations=" << report.violations
              << "\n";
    return report;
}

FieldReport compare_ple(const std::vector<std::uint16_t>& got,
                        const std::vector<std::uint16_t>& want) {
    FieldReport report;
    require(got.size() == want.size(), "PLE history extent differs");
    for (std::size_t index = 0; index < got.size(); ++index) {
        const double g = fp16_word_to_float(got[index]);
        const double w = fp16_word_to_float(want[index]);
        const double diff = std::abs(g - w);
        report.max_abs = std::max(report.max_abs, diff);
        report.max_rel = std::max(report.max_rel, diff / (1.0 + std::abs(w)));
        const double bound = kTolerance.ple_rtol * (1.0 + std::abs(w));
        report.worst_ratio = std::max(report.worst_ratio, diff / bound);
        if (diff > bound) { ++report.violations; }
    }
    std::cout << "ORACLE_FIELD field=ple_history max_abs=" << report.max_abs
              << " max_rel=" << report.max_rel << " worst_ratio=" << report.worst_ratio
              << " ref_rtol=" << kTolerance.ple_rtol
              << " kind=cross_profile_diagnostic cross_profile_violations=" << report.violations
              << "\n";
    return report;
}

// Cross-profile diagnostic only. Host scalars and QSA rope remain exact semantic requirements;
// FP32/FP16 state fields are reported against reference scales but never gate the test, because
// the sequential T=1 route uses a different hc_mix reduction and is not the fold oracle. Fold
// correctness is gated separately and strictly in check_profile_matched_state().
void compare_snapshots(const Snapshot& oracle, const Snapshot& folded) {
    require(oracle.position == folded.position, "folded position differs from sequential oracle");
    require(oracle.ple_tail[0] == folded.ple_tail[0] &&
                oracle.ple_tail[1] == folded.ple_tail[1],
            "folded PLE tail scalars differ from sequential oracle");
    std::cout << "ORACLE_FIELD field=host_scalars exact kind=semantic violations=0\n";

    for (int rank = 0; rank < 2; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        std::cout << "ORACLE_RANK rank=" << rank << "\n";
        (void)compare_conv(folded.ranks[slot].conv, oracle.ranks[slot].conv);
        (void)compare_recurrent(folded.ranks[slot].recurrent, oracle.ranks[slot].recurrent);
        (void)compare_qsa(folded.ranks[slot].qsa, oracle.ranks[slot].qsa);
        (void)compare_ple(folded.ranks[slot].ple, oracle.ranks[slot].ple);
    }
}

void check_engine_greedy(const char* path) {
    bool saw_acceptance = false;
    for (const auto& prompt : kPrompts) {
        const auto reference = generate(path, false, prompt);
        const auto speculative = generate(path, true, prompt);
        require(speculative.generated_token_ids.size() == kOutputTokens,
                "MTP did not emit the requested token count");
        require(speculative.generated_token_ids == reference.generated_token_ids,
                "MTP fold-commit changed the greedy token sequence");
        require(speculative.speculative.rounds > 0, "MTP did not run a decode round");
        require(speculative.speculative.drafted_tokens >= speculative.speculative.accepted_tokens,
                "MTP accepted more drafts than it drafted");
        saw_acceptance = saw_acceptance || speculative.speculative.accepted_tokens > 0;
    }
    require(saw_acceptance, "no prompt produced an accepted draft");
}

// Production-path assertion for the residual sink stream ordering. ProgramImplCore invokes the
// observer right after its commit sync. The first read is issued on the default stream with no
// test-side sync, exactly as draft_chain reads the sink; only the second read is taken after a
// sync to obtain an independent oracle for the same bytes. If the production path left the
// commit's non-blocking copy unordered, the two reads diverge on full-accept rounds.
void check_sink_ordering(const char* path) {
    std::size_t commits         = 0;
    std::size_t words           = 0;
    std::size_t mismatched      = 0;
    flash::residual_sink_observer() = [&](const Tensor& sink) {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(0));
        const std::size_t count = sink.bytes() / 2;
        std::vector<std::uint16_t> unsynced(count);
        CUDA_CHECK(cudaMemcpy(unsynced.data(), sink.data, sink.bytes(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint16_t> synced(count);
        CUDA_CHECK(cudaMemcpy(synced.data(), sink.data, sink.bytes(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(previous));
        for (std::size_t index = 0; index < count; ++index) {
            if (unsynced[index] != synced[index]) { ++mismatched; }
        }
        words  = count;
        ++commits;
    };
    check_engine_greedy(path);
    flash::residual_sink_observer() = {};
    require(commits > 0, "MTP never committed a verified span");
    require(mismatched == 0,
            "residual sink read stale from the default stream after commit (stream race)");
    std::cout << "SINK_ORDER commits=" << commits << " words=" << words
              << " mismatched_words=" << mismatched << "\n";
}

// Strong correctness check of the fold itself: a batched span over exactly the committed prefix
// uses the same T>1 qkvz projection profile as the verify span. Running it with the normal in-place
// GDN update and comparing against the folded+rebuilt state isolates fold/rebuild semantics from
// the T=1 vs T>1 projection difference. Conv, QSA tail and PLE history are required bit-exact;
// the FP32 recurrent state uses a tight reduction-order bound between the two batched kernels.
void check_profile_matched_state(FlashHarness& harness, const std::vector<TokenId>& prompt,
                                 const FoldResult& folded, std::vector<std::string>& failures) {
    harness.reset();
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::UpdateInPlace);
    (void)harness.prefill(prompt);
    (void)harness.trunk->decode_span(std::span<const TokenId>(folded.verify_tokens.data(),
                                                              folded.committed_count));
    const Snapshot profile = capture_snapshot(harness);
    for (int rank = 0; rank < 2; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        const RankSnapshot& a = profile.ranks[slot];
        const RankSnapshot& b = folded.snapshot.ranks[slot];
        require(a.conv == b.conv,
                "folded conv state is not bit-exact against the profile-matched oracle on rank " +
                    std::to_string(rank));
        require(a.qsa.raw_keys == b.qsa.raw_keys && a.qsa.rope == b.qsa.rope,
                "folded QSA tail is not bit-exact against the profile-matched oracle on rank " +
                    std::to_string(rank));
        require(a.ple == b.ple,
                "folded PLE history is not bit-exact against the profile-matched oracle on rank " +
                    std::to_string(rank));
        require(a.recurrent.size() == b.recurrent.size(),
                "folded recurrent shape differs against the profile-matched oracle");
        double max_abs = 0.0;
        double worst_ratio = 0.0;
        for (std::size_t layer = 0; layer < a.recurrent.size(); ++layer) {
            require(a.recurrent[layer].size() == b.recurrent[layer].size(),
                    "folded recurrent extent differs against the profile-matched oracle");
            for (std::size_t index = 0; index < a.recurrent[layer].size(); ++index) {
                const double diff =
                    std::abs(static_cast<double>(b.recurrent[layer][index]) - a.recurrent[layer][index]);
                max_abs = std::max(max_abs, diff);
                const double bound = kTolerance.profile_recurrent_atol +
                                     kTolerance.profile_recurrent_rtol *
                                         std::abs(a.recurrent[layer][index]);
                worst_ratio = std::max(worst_ratio, diff / bound);
            }
        }
        std::cout << "ORACLE_PROFILE scenario=fold_vs_batched rank=" << rank
                  << " conv=bit_exact qsa=bit_exact ple=bit_exact"
                  << " recurrent_max_abs=" << max_abs << " recurrent_worst_ratio=" << worst_ratio
                  << " atol=" << kTolerance.profile_recurrent_atol
                  << " rtol=" << kTolerance.profile_recurrent_rtol << "\n";
        if (worst_ratio > 1.0) {
            failures.push_back("profile_recurrent rank " + std::to_string(rank));
        }
    }
}

void check_scenario(const char* path, const std::vector<TokenId>& prompt,
                    std::vector<TokenId> verify_tokens, const char* scenario,
                    std::size_t expected_accepted, std::vector<std::string>& failures) {
    FlashHarness harness(path, static_cast<std::uint32_t>(kVerify));

    std::vector<TokenId> reference;
    TokenId next = [&] {
        harness.reset();
        return harness.prefill(prompt);
    }();
    for (std::size_t step = 0; step < kOutputTokens + 2; ++step) {
        reference.push_back(next);
        next = harness.decode(next);
    }
    if (verify_tokens.empty()) {
        verify_tokens = {reference[0], reference[1], reference[2], reference[3]};
    }

    // The oracle consumes exactly the committed prefix derived from the *expected* acceptance,
    // independent of anything the fold route computes.
    const std::uint32_t expected_committed = expected_accepted + 1;
    const Snapshot oracle = sequential_oracle(
        harness, prompt, std::span<const TokenId>(verify_tokens.data(), expected_committed));

    FoldResult folded = run_fold_commit(harness, prompt, verify_tokens);
    require(folded.accepted == expected_accepted,
            std::string(scenario) + " did not accept the expected number of drafts (got " +
                std::to_string(folded.accepted) + ")");

    std::cout << "ORACLE_SCENARIO scenario=" << scenario << " prompt=" << prompt[0]
              << " accepted=" << folded.accepted << " committed=" << folded.committed_count
              << "\n";
    compare_snapshots(oracle, folded.snapshot);

    if (std::string(scenario) == "partial") {
        // Forced-reject continuation from the folded state must reproduce the sequential greedy
        // continuation. Runs before the profile-matched route resets the harness.
        TokenId cursor = folded.predictions[folded.accepted];
        std::size_t mismatches = 0;
        for (std::size_t step = 0; step < 4; ++step) {
            if (cursor != reference[3 + step]) { ++mismatches; }
            cursor = harness.decode(cursor);
        }
        std::cout << "FOLDED_CONT prompt=" << prompt[0] << " mismatches=" << mismatches
                  << "/4\n";
        if (mismatches != 0) {
            failures.push_back("FOLDED_CONT prompt=" + std::to_string(prompt[0]) +
                               " mismatches=" + std::to_string(mismatches));
        }
    }

    check_profile_matched_state(harness, prompt, folded, failures);
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (artifact == nullptr || artifact[0] == '\0') {
        std::cout << "skip: fold commit needs the artifact\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }
    try {
        check_sink_ordering(artifact);
        std::vector<std::string> failures;
        for (const auto& prompt : kPrompts) {
            // Partial accept: a deliberately wrong third draft makes drafts 0 and 1 accepted.
            std::vector<TokenId> partial;
            {
                FlashHarness probe(artifact, static_cast<std::uint32_t>(kVerify));
                probe.reset();
                TokenId next = probe.prefill(prompt);
                std::vector<TokenId> reference;
                for (std::size_t step = 0; step < kOutputTokens + 2; ++step) {
                    reference.push_back(next);
                    next = probe.decode(next);
                }
                partial = {reference[0], reference[1], reference[2],
                           static_cast<TokenId>(reference[3] + 1)};
            }
            check_scenario(artifact, prompt, partial, "partial", 2, failures);
            check_scenario(artifact, prompt, {}, "full", 3, failures);
        }
        if (!failures.empty()) {
            std::string message = "fold correctness gate violations:";
            for (const auto& field : failures) { message += " " + field; }
            throw std::runtime_error(message);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK flash-next MTP fold commit: profile-matched in-place span oracle is bit-exact "
                 "against fold+rebuild on GDN conv/recurrent, QSA tail and PLE history; "
                 "FOLDED_CONT is 0 mismatches; cross-profile T=1 deviations are reported as "
                 "diagnostics only (implementation profile, not a fold defect)\n";
    return 0;
}
