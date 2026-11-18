#include <ninfer/engine.h>

#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::Engine;
using ninfer::EngineOptions;
using ninfer::GenerationResult;
using ninfer::RequestOptions;
using ninfer::TokenId;

namespace flash = ninfer::targets::qwen3_8_flash_next::detail;

constexpr std::uint32_t kKvCapacity = 262144;
constexpr std::array<TokenId, 8> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};
constexpr std::uint32_t kSteps = 3;
constexpr std::uint32_t kGreedyTokens = kSteps + 1;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

EngineOptions engine_options(const char* path) {
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
    return options;
}

std::vector<TokenId> greedy_sequence(const char* path) {
    Engine engine(engine_options(path));
    auto prompt = engine.prepare_tokens(std::vector<TokenId>(kPrompt.begin(), kPrompt.end()));
    RequestOptions request;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.requested_output_tokens = kGreedyTokens;
    request.stop.include_model_defaults       = false;
    GenerationResult result = engine.generate(std::move(prompt), std::move(request));
    require(result.generated_token_ids.size() == kGreedyTokens,
            "the non-MTP Engine did not produce the requested greedy tokens");
    return result.generated_token_ids;
}

std::vector<std::uint16_t> read_hidden(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error(std::string("cannot open hidden fixture: ") + path); }
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    const auto expected = 2U * static_cast<std::size_t>(flash::MtpDraft::hidden_elements);
    if (bytes.size() != expected) {
        throw std::runtime_error("hidden fixture has the wrong element count");
    }
    std::vector<std::uint16_t> data(bytes.size() / 2);
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return data;
}

std::vector<TokenId> commit(const std::vector<TokenId>& drafts,
                            const std::vector<TokenId>& greedy) {
    std::size_t accepted = 0;
    while (accepted < drafts.size() && drafts[accepted] == greedy[accepted]) { ++accepted; }
    std::vector<TokenId> committed(drafts.begin(), drafts.begin() + static_cast<std::ptrdiff_t>(accepted));
    committed.push_back(greedy[accepted]);
    return committed;
}

void check_prefix(const std::vector<TokenId>& committed, const std::vector<TokenId>& greedy,
                  const char* label) {
    require(committed.size() <= greedy.size(), std::string(label) + ": commit is too long");
    for (std::size_t index = 0; index < committed.size(); ++index) {
        require(committed[index] == greedy[index],
                std::string(label) + ": committed token " + std::to_string(index) +
                    " differs from non-MTP greedy");
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    const char* hidden = std::getenv("NINFER_MTP_HIDDEN");
    if (artifact == nullptr || hidden == nullptr) {
        std::cout << "skip: greedy equivalence needs an artifact and a hidden fixture\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }
    try {
        const std::vector<TokenId> greedy = greedy_sequence(artifact);

        const std::vector<TokenId> perfect_drafts(greedy.begin(), greedy.begin() + kSteps);
        const std::vector<TokenId> full_accept = commit(perfect_drafts, greedy);
        require(full_accept == greedy,
                "full acceptance did not commit the non-MTP greedy sequence");
        check_prefix(full_accept, greedy, "full acceptance");

        auto words = read_hidden(hidden);
        ninfer::DeviceBuffer conditioning(words.size() * 2);
        conditioning.copy_from_host(words.data(), words.size() * 2);
        ninfer::Tensor view(conditioning.p, ninfer::DType::FP16, {2560, 4});
        flash::MtpDraft draft(artifact, 0);
        const std::vector<TokenId> drafts =
            flash::draft_chain(draft, view, 504, kSteps);
        const std::vector<TokenId> real_commit = commit(drafts, greedy);
        check_prefix(real_commit, greedy, "real MTP chain");

        std::cout << "greedy";
        for (TokenId token : greedy) { std::cout << ' ' << token; }
        std::cout << "\nfull_accept";
        for (TokenId token : full_accept) { std::cout << ' ' << token; }
        std::cout << "\nmtp_drafts";
        for (TokenId token : drafts) { std::cout << ' ' << token; }
        std::cout << "\nreal_commit";
        for (TokenId token : real_commit) { std::cout << ' ' << token; }
        std::cout << "\n";
    } catch (const std::exception& error) {
        std::cerr << "greedy equivalence failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK flash-next MTP greedy equivalence: full acceptance and real chain\n";
    return 0;
}
