#include "flash_model_harness.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>
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
        for (std::size_t index = 0; index < kPrompts.size(); ++index) {
            check_prompt(artifact, kPrompts[index], index);
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
