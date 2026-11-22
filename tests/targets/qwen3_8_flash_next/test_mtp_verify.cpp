#include "flash_model_harness.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

using flash_harness::FlashHarness;
using flash_harness::require;
namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
using ninfer::TokenId;

constexpr std::size_t kVerify = 4;

const std::vector<std::vector<TokenId>> kPrompts{
    {1024, 7, 98140, 33, 4097, 512, 100, 2048},
    {1024, 7, 98140, 33, 4097, 512, 100, 2049},
    {1024, 7, 98140, 33, 4097, 512, 100, 2050},
    {100, 200, 300, 400, 500, 600, 700, 800},
    {1, 2, 3, 4, 5, 6, 7, 8},
};

void check_greedy_prefix(std::span<const TokenId> drafts, std::span<const TokenId> predictions) {
    const std::uint32_t accepted = flash::accepted_prefix(drafts, predictions);
    std::uint32_t expected = 0;
    while (expected < drafts.size() && drafts[expected] == predictions[expected]) { ++expected; }
    require(accepted == static_cast<std::uint32_t>(expected),
            "greedy acceptance is not the longest common prefix");
    const auto committed = flash::commit_tokens(drafts, predictions);
    require(committed.size() == static_cast<std::size_t>(expected) + 1,
            "commit did not emit the accepted prefix plus the target token");
    for (std::size_t index = 0; index < expected; ++index) {
        require(committed[index] == drafts[index], "commit altered an accepted draft");
    }
    require(committed[expected] == predictions[expected],
            "commit did not emit the target token at the first divergence");
}

void check_budget() {
    ninfer::runtime::RoundBudget budget;
    budget.generated_tokens_remaining = 0;
    require(flash::verify_extent(3, 3, budget, 10, 100) == 0, "an exhausted budget still drafted");
    require(flash::round_finished(budget, 0), "an exhausted budget was not finished");
    budget.generated_tokens_remaining = 1;
    require(flash::verify_extent(3, 3, budget, 10, 100) == 0,
            "a one-token budget still drafted");
    budget.generated_tokens_remaining = 2;
    require(flash::verify_extent(3, 3, budget, 10, 100) == 1,
            "the draft extent exceeded remaining minus one");
    require(!flash::round_finished(budget, 1), "a partially consumed budget was finished");
    budget.generated_tokens_remaining = 8;
    require(flash::verify_extent(3, 3, budget, 10, 100) == 3, "the draft window was not honored");
    require(flash::verify_extent(3, 3, budget, 98, 100) == 1,
            "the capacity bound did not leave room for the target token");
    require(flash::round_finished(budget, 8), "an exactly consumed budget was not finished");
}

std::vector<std::vector<std::uint16_t>> capture_steps(FlashHarness& harness,
                                                       std::span<const TokenId> tokens) {
    flash_harness::capture_buffer().clear();
    flash_harness::g_logits_sink = &flash_harness::capture_buffer();
    (void)flash::verify_sequential(tokens,
                                   [&](TokenId token) { return harness.decode(token); });
    flash_harness::g_logits_sink = nullptr;
    return flash_harness::capture_buffer();
}

void check_prompt(const char* artifact, const std::vector<TokenId>& prompt, std::size_t case_index) {
    FlashHarness harness(artifact);

    harness.reset();
    const TokenId anchor = harness.prefill(std::span<const TokenId>{prompt.data(), prompt.size()});
    std::vector<TokenId> verify_tokens{anchor};
    TokenId cursor = anchor;
    for (std::size_t step = 1; step < kVerify; ++step) {
        cursor = harness.decode(cursor);
        verify_tokens.push_back(cursor);
    }

    harness.reset();
    (void)harness.prefill(std::span<const TokenId>{prompt.data(), prompt.size()});
    const auto reference_logits = capture_steps(harness, verify_tokens);
    const std::uint64_t reference_digest = harness.digest();
    const std::uint64_t reference_kv = harness.kv_digest(
        static_cast<std::uint32_t>(prompt.size()), static_cast<std::uint32_t>(kVerify));

    harness.reset();
    (void)harness.prefill(std::span<const TokenId>{prompt.data(), prompt.size()});
    const auto verify_logits = capture_steps(harness, verify_tokens);
    const std::uint64_t verify_digest = harness.digest();
    const std::uint64_t verify_kv = harness.kv_digest(
        static_cast<std::uint32_t>(prompt.size()), static_cast<std::uint32_t>(kVerify));

    require(verify_logits.size() == kVerify, "the sequential verify did not capture one step per token");
    require(reference_logits.size() == kVerify, "the reference did not capture one step per token");
    require(verify_digest == reference_digest,
            "the sequential verify changed persistent state on prompt " + std::to_string(case_index));
    require(verify_kv == reference_kv,
            "the sequential verify changed KV pages on prompt " + std::to_string(case_index));

    const int vocab = flash::TextConfig::vocab;
    for (std::size_t step = 0; step < kVerify; ++step) {
        const auto& a = verify_logits[step];
        const auto& b = reference_logits[step];
        require(a.size() == static_cast<std::size_t>(vocab) && b.size() == a.size(),
                "a verify logits step has the wrong extent");
        for (std::size_t word = 0; word < a.size(); ++word) {
            require(a[word] == b[word],
                    "the sequential verify changed logits at step " + std::to_string(step) +
                        " word " + std::to_string(word) + " on prompt " +
                        std::to_string(case_index));
        }
    }

    harness.reset();
    (void)harness.prefill(std::span<const TokenId>{prompt.data(), prompt.size()});
    const auto predictions = flash::verify_sequential(verify_tokens, [&](TokenId token) {
        return harness.decode(token);
    });
    require(predictions.size() == verify_tokens.size(), "verify predictions have the wrong extent");
    check_greedy_prefix(std::span<const TokenId>{verify_tokens}.subspan(1), predictions);
}

std::vector<std::pair<const char*, std::uint64_t>> component_digests(FlashHarness& harness) {
    std::vector<std::pair<const char*, std::uint64_t>> out;
    auto& state = *harness.states[0];
    CUDA_CHECK(cudaSetDevice(state.device_id()));
    CUDA_CHECK(cudaStreamSynchronize(state.stream()));
    constexpr std::uint64_t kSeed = 1469598103934665603ULL;
    const std::uint32_t position = state.next_position();
    out.emplace_back("position", flash_harness::fnv(kSeed, &position, sizeof(position)));
    const std::uint64_t tail[2] = {state.ple_tail(0), state.ple_tail(1)};
    out.emplace_back("ple_tail", flash_harness::fnv(kSeed, tail, sizeof(tail)));
    for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
        out.emplace_back("gdn_conv", flash_harness::hash_tensor(kSeed, state.gdn_conv_state(index)));
        out.emplace_back("gdn_recurrent",
                         flash_harness::hash_tensor(kSeed, state.gdn_recurrent_state(index)));
    }
    for (int slot = 0; slot < flash::TextConfig::qsa_layers(); ++slot) {
        out.emplace_back("qsa_before_raw",
                         flash_harness::hash_tensor(kSeed, state.before_tail(slot).raw_keys));
        out.emplace_back("qsa_before_rope",
                         flash_harness::hash_tensor(kSeed, state.before_tail(slot).first_rope));
    }
    out.emplace_back("qsa_tail_raw", flash_harness::hash_tensor(kSeed, state.tail(0).raw_keys));
    out.emplace_back("qsa_tail_rope", flash_harness::hash_tensor(kSeed, state.tail(0).first_rope));
    out.emplace_back("ple_history", flash_harness::hash_tensor(kSeed, state.ple_history()));
    std::int32_t route = 0;
    CUDA_CHECK(cudaMemcpy(&route, state.route_counter(), sizeof(route), cudaMemcpyDeviceToHost));
    out.emplace_back("route_counter", flash_harness::fnv(kSeed, &route, sizeof(route)));
    return out;
}

// Diagnostic only: report how far a batched (T>1) verify span is from T sequential T=1 steps.
// Prints per-prompt logits divergence, argmax disagreement and state/KV digest equality; never fails.
void report_span_divergence(const char* artifact, const std::vector<TokenId>& prompt,
                            std::size_t case_index) {
    FlashHarness harness(artifact);
    harness.reset();
    const TokenId anchor = harness.prefill(std::span<const TokenId>{prompt.data(), prompt.size()});
    std::vector<TokenId> verify_tokens{anchor};
    TokenId cursor = anchor;
    for (std::size_t step = 1; step < kVerify; ++step) {
        cursor = harness.decode(cursor);
        verify_tokens.push_back(cursor);
    }

    harness.reset();
    (void)harness.prefill(std::span<const TokenId>{prompt.data(), prompt.size()});
    flash_harness::capture_buffer().clear();
    flash_harness::g_logits_sink = &flash_harness::capture_buffer();
    std::vector<TokenId> seq_pred;
    seq_pred.reserve(verify_tokens.size());
    for (TokenId token : verify_tokens) { seq_pred.push_back(harness.decode(token)); }
    flash_harness::g_logits_sink = nullptr;
    const auto seq_logits = flash_harness::capture_buffer();
    const std::uint64_t seq_digest = harness.digest();
    const auto seq_components = component_digests(harness);

    harness.reset();
    (void)harness.prefill(std::span<const TokenId>{prompt.data(), prompt.size()});
    flash_harness::capture_buffer().clear();
    flash_harness::g_logits_sink = &flash_harness::capture_buffer();
    const auto span_pred = harness.trunk->decode_span(verify_tokens);
    flash_harness::g_logits_sink = nullptr;
    const auto span_logits = flash_harness::capture_buffer();
    const std::uint64_t span_digest = harness.digest();
    const auto span_components = component_digests(harness);

    const int vocab = flash::TextConfig::vocab;
    if (seq_logits.size() != kVerify || span_logits.size() != 1 ||
        span_logits[0].size() < static_cast<std::size_t>(vocab) * kVerify) {
        std::cout << "case " << case_index << " span_diag: unexpected capture sizes seq="
                  << seq_logits.size() << " span=" << span_logits.size() << '\n';
        return;
    }
    double max_abs = 0.0;
    double sum_abs = 0.0;
    std::uint64_t count = 0;
    int winner_mismatch = 0;
    for (std::size_t step = 0; step < kVerify; ++step) {
        const auto& reference = seq_logits[step];
        for (int word = 0; word < vocab; ++word) {
            __half a;
            __half b;
            std::memcpy(&a, &reference[static_cast<std::size_t>(word)], sizeof(a));
            std::memcpy(&b, &span_logits[0][static_cast<std::size_t>(step) * vocab + word],
                        sizeof(b));
            const double diff = std::fabs(static_cast<double>(__half2float(a)) -
                                          static_cast<double>(__half2float(b)));
            max_abs = std::max(max_abs, diff);
            sum_abs += diff;
            ++count;
        }
        if (static_cast<TokenId>(seq_pred[step]) != span_pred[step]) { ++winner_mismatch; }
    }
    std::string differing;
    const std::size_t component_count =
        std::min(seq_components.size(), span_components.size());
    for (std::size_t index = 0; index < component_count; ++index) {
        if (seq_components[index].second != span_components[index].second) {
            if (!differing.empty()) { differing += ","; }
            differing += seq_components[index].first;
        }
    }
    if (seq_components.size() != span_components.size()) { differing += ",component_count"; }
    std::cout << "case " << case_index << " span_diag: max_abs=" << max_abs
              << " mean_abs=" << (count == 0 ? 0.0 : sum_abs / static_cast<double>(count))
              << " winner_mismatch=" << winner_mismatch
              << " state_digest_equal=" << (seq_digest == span_digest)
              << " diff_state=[" << differing << "]\n";
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (artifact == nullptr || artifact[0] == '\0') {
        check_greedy_prefix(std::vector<TokenId>{5, 6, 7}, std::vector<TokenId>{5, 6, 7, 9});
        check_greedy_prefix(std::vector<TokenId>{5, 99, 7}, std::vector<TokenId>{5, 6, 7, 9});
        check_greedy_prefix(std::vector<TokenId>{99, 6, 7}, std::vector<TokenId>{5, 6, 7, 9});
        check_greedy_prefix(std::vector<TokenId>{5, 6, 7}, std::vector<TokenId>{5, 6, 7, 7});
        check_budget();
        std::cout << "skip: verify-logit parity needs the artifact\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }
    try {
        check_greedy_prefix(std::vector<TokenId>{5, 6, 7}, std::vector<TokenId>{5, 6, 7, 9});
        check_greedy_prefix(std::vector<TokenId>{5, 99, 7}, std::vector<TokenId>{5, 6, 7, 9});
        check_greedy_prefix(std::vector<TokenId>{99, 6, 7}, std::vector<TokenId>{5, 6, 7, 9});
        check_budget();
        const bool span_diag = std::getenv("NINFER_MTP_SPAN_DIAG") != nullptr;
        for (std::size_t index = 0; index < kPrompts.size(); ++index) {
            check_prompt(artifact, kPrompts[index], index);
            if (span_diag) { report_span_divergence(artifact, kPrompts[index], index); }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK flash-next MTP verify: sequential T=1 parity (bit-exact logits, state and KV), "
                 "greedy LCP, commit, and budget across "
              << kPrompts.size() << " prompts\n";
    return 0;
}
