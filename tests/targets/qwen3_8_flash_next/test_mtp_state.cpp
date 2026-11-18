#include "flash_model_harness.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"
#include "targets/qwen3_8_flash_next/impl/state_transaction.h"

#include <cuda_runtime.h>

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
constexpr std::size_t kContinue = 8;

const std::vector<std::vector<TokenId>> kPrompts{
    {1024, 7, 98140, 33, 4097, 512, 100, 2048},
    {1024, 7, 98140, 33, 4097, 512, 100, 2049},
    {1024, 7, 98140, 33, 4097, 512, 100, 2050},
    {100, 200, 300, 400, 500, 600, 700, 800},
    {1, 2, 3, 4, 5, 6, 7, 8},
};

void expect_tokens(const std::vector<TokenId>& got, const std::vector<TokenId>& want,
                   const std::string& label) {
    require(got == want, label + ": token sequence differs from the non-speculative path");
}

void check_prompt(const char* artifact, const std::vector<TokenId>& prompt, std::size_t case_index) {
    FlashHarness harness(artifact);

    harness.reset();
    std::vector<TokenId> reference;
    TokenId next = harness.prefill(std::span<const TokenId>{prompt.data(), prompt.size()});
    std::uint64_t reference_digest = 0;
    std::uint64_t reference_kv = 0;
    for (std::size_t step = 0; step < 3 + kContinue; ++step) {
        reference.push_back(next);
        next = harness.decode(next);
        if (step == 2) {
            reference_digest = harness.digest();
            reference_kv = harness.kv_digest(static_cast<std::uint32_t>(prompt.size()),
                                             static_cast<std::uint32_t>(kVerify));
        }
    }

    harness.reset();
    TokenId anchor = harness.prefill(std::span<const TokenId>{prompt.data(), prompt.size()});

    flash::StateTransaction txn(harness.state_ptrs, 2, static_cast<std::uint32_t>(kVerify));
    txn.begin();

    const std::vector<TokenId> verify_tokens{reference[0], reference[1], reference[2],
                                             static_cast<TokenId>(reference[3] + 1)};
    const auto predictions = flash::verify_sequential(verify_tokens, [&](TokenId token) {
        return harness.decode(token);
    });

    const std::vector<TokenId> drafts(verify_tokens.begin() + 1, verify_tokens.end());
    const std::uint32_t accepted = flash::accepted_prefix(drafts, predictions);
    require(accepted == 2, "the forced scenario did not accept exactly two drafts on prompt " +
                               std::to_string(case_index));

    std::vector<TokenId> committed(drafts.begin(), drafts.begin() + accepted);
    committed.push_back(predictions[accepted]);
    const std::vector<TokenId> replay_tokens{anchor, drafts[0], drafts[1]};
    txn.commit(replay_tokens, [&](TokenId token) {
        (void)harness.decode(token);
        return token;
    });

    const std::uint64_t rolled_back = harness.digest();
    require(rolled_back == reference_digest,
            "state after rollback differs bitwise from the non-speculative path on prompt " +
                std::to_string(case_index));
    const std::uint64_t rolled_back_kv = harness.kv_digest(
        static_cast<std::uint32_t>(prompt.size()), static_cast<std::uint32_t>(kVerify));
    require(rolled_back_kv == reference_kv,
            "KV pages after rollback retain a rejected draft on prompt " +
                std::to_string(case_index));

    std::vector<TokenId> continued;
    TokenId cursor = committed.back();
    for (std::size_t step = 0; step < kContinue; ++step) {
        continued.push_back(cursor);
        cursor = harness.decode(cursor);
    }
    expect_tokens(continued,
                  std::vector<TokenId>(reference.begin() + 3, reference.begin() + 3 + kContinue),
                  "continued decode on prompt " + std::to_string(case_index));
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
        for (std::size_t index = 0; index < kPrompts.size(); ++index) {
            check_prompt(artifact, kPrompts[index], index);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK flash-next MTP state rollback: bit-exact state and KV, no rejected residue, "
                 "bit-exact continuation across "
              << kPrompts.size() << " prompts\n";
    return 0;
}
