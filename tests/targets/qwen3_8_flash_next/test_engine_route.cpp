#include "options.h"

#include <ninfer/engine.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::Engine;
using ninfer::EngineOptions;
using ninfer::GenerationResult;
using ninfer::KvCacheStorage;
using ninfer::MemorySummary;
using ninfer::RequestOptions;
using ninfer::TokenId;

constexpr std::uint32_t kKvCapacity = 262144;
constexpr std::uint32_t kPageTokens = 64;

constexpr std::array<TokenId, 8> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};
constexpr std::array<TokenId, 3> kDirectPathTokens{4818, 84666, 96991};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::cli::Options parse_cli(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

void check_cli_parse() {
    const auto automatic = parse_cli(
        {"ninfer", "model.ninfer", "--prompt", "x", "--expert-cache-mib", "auto"});
    require(automatic.expert_cache_mib == 0,
            "--expert-cache-mib auto did not resolve to the automatic plan");

    const auto explicit_mib = parse_cli(
        {"ninfer", "model.ninfer", "--prompt", "x", "--expert-cache-mib", "1024"});
    require(explicit_mib.expert_cache_mib == 1024,
            "--expert-cache-mib 1024 did not reach the CLI options");
}

EngineOptions engine_options(const char* path, std::uint32_t expert_cache_mib,
                             std::uint32_t max_concurrency = 1) {
    EngineOptions options;
    options.artifact_path    = path;
    options.device           = 0;
    options.tp               = 2;
    options.devices          = {0, 1};
    options.max_context      = kKvCapacity;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
    options.max_concurrency  = max_concurrency;
    options.prefill_chunk    = 64;
    options.use_cuda_graph   = false;
    options.expert_cache_mib = expert_cache_mib;
    return options;
}

void check_concurrency_limit(const char* path) {
    bool threw = false;
    std::string message;
    try {
        Engine engine(engine_options(path, 0, 2));
    } catch (const std::invalid_argument& error) {
        threw   = true;
        message = error.what();
    }
    require(threw, "a request concurrency above one was accepted");
    require(message.find("one active request") != std::string::npos,
            "the concurrency rejection does not name the single-request contract: " + message);
}

void check_over_budget_cache(const char* path) {
    bool threw = false;
    std::string message;
    try {
        Engine engine(engine_options(path, std::numeric_limits<std::uint32_t>::max()));
    } catch (const std::invalid_argument& error) {
        threw   = true;
        message = error.what();
    }
    require(threw, "an expert cache larger than the device memory was accepted");
    require(message.find("available device memory") != std::string::npos,
            "the over-budget expert cache error does not name the available budget: " + message);
}

void check_engine_route(const char* path) {
    Engine engine(engine_options(path, 0));

    auto prompt = engine.prepare_tokens(std::vector<TokenId>(kPrompt.begin(), kPrompt.end()));
    RequestOptions request;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.requested_output_tokens = kDirectPathTokens.size();
    request.stop.include_model_defaults       = false;
    const GenerationResult result = engine.generate(std::move(prompt), std::move(request));

    require(result.generated_token_ids.size() == kDirectPathTokens.size(),
            "the Engine returned an unexpected number of tokens");
    for (std::size_t index = 0; index < kDirectPathTokens.size(); ++index) {
        require(result.generated_token_ids[index] == kDirectPathTokens[index],
                "the Engine token at position " + std::to_string(index) + " is " +
                    std::to_string(result.generated_token_ids[index]) + " instead of " +
                    std::to_string(kDirectPathTokens[index]));
    }
    require(result.timings.prefill_seconds > 0.0,
            "the Engine did not report a positive prefill duration");
    require(result.timings.decode_seconds > 0.0,
            "the Engine did not report a positive decode duration");

    const ninfer::MetricsSnapshot metrics = engine.metrics();
    require(metrics.scheduler.decode_rounds >= 1,
            "the Engine executed no decode round");
    require(metrics.scheduler.committed_decode_tokens == metrics.scheduler.decode_rounds,
            "the Engine did not commit one decode token per decode round");

    const MemorySummary memory = engine.memory_summary();
    const std::uint32_t pages   = kKvCapacity / kPageTokens + 1;
    require(memory.kv_capacity_page_groups == pages,
            "the Engine reports " + std::to_string(memory.kv_capacity_page_groups) +
                " KV page groups instead of " + std::to_string(pages));
    require(memory.kv_capacity_max_page_groups == pages,
            "the Engine reports " + std::to_string(memory.kv_capacity_max_page_groups) +
                " maximum KV page groups instead of " + std::to_string(pages));
    require(memory.kv_cache == KvCacheStorage::MixedBf16Fp16,
            "the Engine does not report the mixed BF16-K / FP16-V KV layout");
    require(memory.expert_cache_bytes != 0,
            "the Engine reports no resident expert cache");

    std::cout << "flash-next engine route: tokens";
    for (const TokenId token : result.generated_token_ids) { std::cout << ' ' << token; }
    std::cout << " kv-pages " << memory.kv_capacity_page_groups << " expert-cache "
              << memory.expert_cache_bytes << " prefill-s " << result.timings.prefill_seconds
              << " decode-s " << result.timings.decode_seconds << '\n';
}

} // namespace

int main() {
    try {
        check_cli_parse();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    const char* path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (path == nullptr || path[0] == '\0') {
        std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }

    try {
        check_concurrency_limit(path);
        check_engine_route(path);
        check_over_budget_cache(path);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK flash-next engine route: two-rank package through the public Engine\n";
    return 0;
}
