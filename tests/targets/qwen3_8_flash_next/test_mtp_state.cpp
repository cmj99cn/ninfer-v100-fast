#include "flash_model_harness.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"
#include "targets/qwen3_8_flash_next/impl/state_transaction.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace {

using flash_harness::FlashHarness;
using flash_harness::require;
using ninfer::TokenId;

constexpr std::array<TokenId, 8> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};
constexpr std::size_t kContinue = 8;

void expect_tokens(const std::vector<TokenId>& got, const std::vector<TokenId>& want,
                   const char* label) {
    require(got == want, std::string(label) + ": token sequence differs from the non-speculative path");
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (artifact == nullptr || artifact[0] == '\0') {
        std::cout << "skip: state rollback needs the artifact\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }
    try {
        FlashHarness harness(artifact);

        harness.reset();
        std::vector<TokenId> reference;
        TokenId next = harness.prefill(std::span<const TokenId>{kPrompt.data(), kPrompt.size()});
        std::uint64_t reference_digest = 0;
        for (std::size_t step = 0; step < 3 + kContinue; ++step) {
            reference.push_back(next);
            next = harness.decode(next);
            if (step == 2) { reference_digest = harness.digest(); }
        }

        harness.reset();
        TokenId anchor = harness.prefill(std::span<const TokenId>{kPrompt.data(), kPrompt.size()});

        ninfer::targets::qwen3_8_flash_next::detail::StateTransaction txn(
            harness.state_ptrs, 2);
        txn.begin();

        const std::array<TokenId, 4> verify{reference[0], reference[1], reference[2],
                                            static_cast<TokenId>(reference[3] + 1)};
        flash_harness::g_logits_sink = &flash_harness::capture_buffer();
        flash_harness::capture_buffer().clear();
        harness.prefill(std::span<const TokenId>{verify.data(), verify.size()});
        flash_harness::g_logits_sink = nullptr;

        std::vector<TokenId> predictions;
        const auto& steps = flash_harness::capture_buffer();
        require(steps.size() == 1, "the verify forward did not capture one logits step");
        require(steps[0].size() == static_cast<std::size_t>(ninfer::targets::qwen3_8_flash_next::detail::TextConfig::vocab) * verify.size(),
                "the verify logits have the wrong extent");
        for (std::size_t position = 0; position < verify.size(); ++position) {
            predictions.push_back(static_cast<TokenId>(flash_harness::argmax_at(
                steps[0], ninfer::targets::qwen3_8_flash_next::detail::TextConfig::vocab,
                static_cast<int>(position))));
        }

        const std::vector<TokenId> drafts{verify[1], verify[2], verify[3]};
        const std::uint32_t accepted =
            ninfer::targets::qwen3_8_flash_next::detail::accepted_prefix(drafts, predictions);
        require(accepted == 2, "the forced scenario did not accept exactly two drafts");

        std::vector<TokenId> committed(drafts.begin(), drafts.begin() + accepted);
        committed.push_back(predictions[accepted]);
        const std::vector<TokenId> replay_tokens{anchor, drafts[0], drafts[1]};
        txn.commit(replay_tokens, [&](TokenId token) {
            (void)harness.decode(token);
            return token;
        });

        const std::uint64_t rolled_back = harness.digest();
        require(rolled_back == reference_digest,
                "state after rollback differs bitwise from the non-speculative path");

        std::vector<TokenId> continued;
        TokenId cursor = committed.back();
        for (std::size_t step = 0; step < kContinue; ++step) {
            continued.push_back(cursor);
            cursor = harness.decode(cursor);
        }
        expect_tokens(continued,
                      std::vector<TokenId>(reference.begin() + 3, reference.begin() + 3 + kContinue),
                      "continued decode");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK flash-next MTP state rollback matches the non-speculative path\n";
    return 0;
}
