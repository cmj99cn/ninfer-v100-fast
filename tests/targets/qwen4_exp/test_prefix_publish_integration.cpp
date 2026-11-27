#include <ninfer/engine.h>

#include "runtime/prefix_cache/shared_host_tier.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << what << '\n';
}

ninfer::EngineOptions engine_options(const char* artifact, std::uint32_t host_mib) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.tp                        = 2;
    options.devices                   = {0, 1};
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency           = 1;
    options.prefill_chunk             = 128;
    options.use_cuda_graph            = false;
    options.expert_cache_mib          = 0;
    options.prefix_cache.enabled      = true;
    options.prefix_cache.host_cache_mib = host_mib;
    options.create_shared_host_tier   = host_mib > 0;
    return options;
}

std::vector<ninfer::TokenId> make_prompt(int salt) {
    std::vector<ninfer::TokenId> tokens(128);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        tokens[index] = static_cast<ninfer::TokenId>(17 + (index * 37 + salt * 13) % 9000);
    }
    return tokens;
}

bool varied_bytes(const void* data, std::size_t bytes) {
    if (bytes == 0) { return false; }
    const auto* words = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 1; index < bytes; ++index) {
        if (words[index] != words[0]) { return true; }
    }
    return false;
}

void warm_and_publish(ninfer::Engine& engine) {
    auto prepared   = engine.prepare_tokens(make_prompt(0), false);
    const auto hashes = std::vector<std::uint64_t>(prepared.block_hashes().begin(),
                                                   prepared.block_hashes().end());
    check(hashes.size() == 2, "a 128-token prompt yields two chained block hashes");
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    const auto result = engine.generate(std::move(prepared), std::move(request));
    check(result.generated_token_ids.size() == 1, "the prefix-warming request generated one token");
}

} // namespace

int main() {
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
        {
            ninfer::Engine engine(engine_options(path, 2048));
            auto prompt     = make_prompt(0);
            auto prepared   = engine.prepare_tokens(prompt, false);
            const auto hashes = std::vector<std::uint64_t>(prepared.block_hashes().begin(),
                                                           prepared.block_hashes().end());
            check(hashes.size() == 2, "a 128-token prompt yields two chained block hashes");

            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 1;
            request.execution.sampling.temperature    = 0.0F;
            request.execution.allow_prefix_reuse      = false;
            request.stop.include_model_defaults       = false;
            const auto result = engine.generate(std::move(prepared), std::move(request));
            check(result.generated_token_ids.size() == 1,
                  "the prefix-warming request generated one token");

            const auto probe = engine.prefix_probe(hashes);
            check(probe.block_tokens == 64, "the probe granularity is 64 tokens");
            check(probe.device_frontier == 128, "the local tree holds the committed 128-token chain");
            check(probe.manifest_frontier == 128, "the host tier holds a 128-token manifest");
            check(probe.manifest_id != 0, "the probe names a live manifest");

            const auto tier = engine.shared_host_tier();
            check(tier != nullptr, "the Engine owns a process-level host tier");
            if (tier != nullptr) {
                const auto manifest = tier->match(2, hashes, 128);
                check(manifest.has_value(), "the published manifest matches its own hash chain");
                if (manifest.has_value()) {
                    check(manifest->frontier == 128, "the manifest frontier is 128 tokens");
                    check(manifest->hashes == hashes, "the manifest carries the prepared hash chain");
                    check(manifest->block_pages.size() == 2,
                          "the manifest holds one pinned slab per committed block");
                    check(manifest->state_page != ninfer::prefix_cache::kInvalidHostPage,
                          "the manifest holds a pinned state-image slab");
                    bool block_data = false;
                    for (const auto page : manifest->block_pages) {
                        block_data = block_data ||
                                     varied_bytes(tier->data(page), tier->bytes(page));
                    }
                    check(block_data,
                          "published block slabs carry real, non-constant KV bytes");
                    check(varied_bytes(tier->data(manifest->state_page),
                                       tier->bytes(manifest->state_page)),
                          "the published state slab carries real, non-constant state bytes");
                }
            }
        }

        {
            ninfer::Engine device_only(engine_options(path, 0));
            check(device_only.shared_host_tier() == nullptr, "host_cache_mib 0 builds no host tier");
            warm_and_publish(device_only);
            auto prompt   = make_prompt(0);
            auto prepared = device_only.prepare_tokens(prompt, false);
            const auto hashes = std::vector<std::uint64_t>(prepared.block_hashes().begin(),
                                                           prepared.block_hashes().end());
            const auto probe = device_only.prefix_probe(hashes);
            check(probe.manifest_id == 0 && probe.manifest_frontier == 0,
                  "without a host tier no manifest is published or matched");
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    if (failures != 0) {
        std::fprintf(stderr, "qwen4_exp prefix publish integration: %d failure(s)\n", failures);
        return 1;
    }
    std::cout << "OK qwen4_exp prefix publish integration through the public Engine\n";
    return 0;
}
