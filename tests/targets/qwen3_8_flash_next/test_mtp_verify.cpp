#include "flash_model_harness.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace {

using flash_harness::FlashHarness;
using flash_harness::require;
namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
using ninfer::TokenId;

constexpr std::array<TokenId, 8> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};
constexpr std::size_t kVerify = 4;

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

        FlashHarness harness(artifact);
        harness.reset();
        const TokenId anchor =
            harness.prefill(std::span<const TokenId>{kPrompt.data(), kPrompt.size()});
        const std::array<TokenId, kVerify> greedy{anchor, 359, 81486, 837};
        flash_harness::capture_buffer().clear();
        flash_harness::g_logits_sink = &flash_harness::capture_buffer();
        harness.prefill(std::span<const TokenId>{greedy.data(), greedy.size()});
        flash_harness::g_logits_sink = nullptr;
        const std::vector<std::vector<std::uint16_t>> batched_steps =
            flash_harness::capture_buffer();
        require(batched_steps.size() == 1, "the batched verify did not capture one logits step");

        harness.reset();
        const TokenId sequential_anchor =
            harness.prefill(std::span<const TokenId>{kPrompt.data(), kPrompt.size()});
        require(sequential_anchor == anchor, "the two paths disagree on the first token");
        flash_harness::capture_buffer().clear();
        flash_harness::g_logits_sink = &flash_harness::capture_buffer();
        TokenId cursor = sequential_anchor;
        for (std::size_t step = 1; step < kVerify; ++step) {
            cursor = harness.decode(greedy[step - 1]);
        }
        flash_harness::g_logits_sink = nullptr;
        const std::vector<std::vector<std::uint16_t>> sequential_steps =
            flash_harness::capture_buffer();
        require(sequential_steps.size() == kVerify - 1,
                "the sequential path did not capture one logits step per token");

        const int vocab = flash::TextConfig::vocab;
        for (std::size_t position = 0; position < kVerify - 1; ++position) {
            const auto& a = batched_steps[0];
            const auto& b = sequential_steps[position];
            require(a.size() >= (position + 1) * static_cast<std::size_t>(vocab),
                    "the batched verify logits are short");
            require(b.size() == static_cast<std::size_t>(vocab),
                    "the sequential logits are short");
            const int batched_argmax =
                flash_harness::argmax_at(a, vocab, static_cast<int>(position));
            const int sequential_argmax = flash_harness::argmax_at(b, vocab, 0);
            std::size_t mismatches = 0;
            float max_abs = 0.0F;
            for (int index = 0; index < vocab; ++index) {
                const std::size_t base = position * static_cast<std::size_t>(vocab) +
                                         static_cast<std::size_t>(index);
                __half ah;
                __half bh;
                std::memcpy(&ah, &a[base], sizeof(ah));
                std::memcpy(&bh, &b[static_cast<std::size_t>(index)], sizeof(bh));
                if (a[base] != b[static_cast<std::size_t>(index)]) { ++mismatches; }
                max_abs = std::max(max_abs, std::abs(__half2float(ah) - __half2float(bh)));
            }
            std::cout << "DIAG token " << position << " logit_mismatches " << mismatches
                      << " of " << vocab << " max_abs " << max_abs << " argmax_batched "
                      << batched_argmax << " argmax_sequential " << sequential_argmax << '\n';
            require(batched_argmax == sequential_argmax,
                    "the batched verify changed the greedy decision at token " +
                        std::to_string(position));
        }

        std::vector<TokenId> predictions;
        for (std::size_t position = 0; position < kVerify; ++position) {
            predictions.push_back(static_cast<TokenId>(
                flash_harness::argmax_at(batched_steps[0], vocab, static_cast<int>(position))));
        }
        check_greedy_prefix(std::vector<TokenId>{greedy[1], greedy[2], greedy[3]}, predictions);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK flash-next MTP verify: greedy LCP, commit, and batched logit parity\n";
    return 0;
}
