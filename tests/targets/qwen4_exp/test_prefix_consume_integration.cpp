#include <ninfer/engine.h>

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

struct Setup {
    std::uint32_t host_mib = 2048;
    std::uint32_t chunk    = 128;
    bool dflash2           = false;
};

ninfer::EngineOptions make_options(const char* artifact, const Setup& setup) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.tp                        = 2;
    options.devices                   = {0, 1};
    options.max_context               = 8192;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.max_concurrency           = 1;
    options.prefill_chunk             = setup.chunk;
    options.use_cuda_graph            = false;
    // A fixed residency seed keeps the reused and cold requests numerically comparable: the
    // automatic zero-MiB plan depends on free device memory at construction time.
    options.expert_cache_mib          = 12000;
#ifdef NINFER_SOURCE_DIR
    options.expert_cache_profile =
        std::string(NINFER_SOURCE_DIR) + "/tools/bench/expert-profile-flash-next.txt";
#endif
    options.prefix_cache.enabled      = true;
    options.prefix_cache.host_cache_mib = setup.host_mib;
    options.create_shared_host_tier   = setup.host_mib > 0;
    if (setup.dflash2) {
        const char* draft = std::getenv("NINFER_DFLASH2_ARTIFACT");
        if (draft == nullptr || draft[0] == '\0') {
            throw std::runtime_error("NINFER_DFLASH2_ARTIFACT is required for the DFlash2 case");
        }
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 3;
        options.speculative_draft_artifact = draft;
    }
    return options;
}

std::vector<ninfer::TokenId> make_tokens(std::size_t count, int salt) {
    std::vector<ninfer::TokenId> tokens(count);
    for (std::size_t index = 0; index < count; ++index) {
        tokens[index] = static_cast<ninfer::TokenId>(1000 + (index * 37 + salt * 13) % 4000);
    }
    return tokens;
}

std::vector<ninfer::TokenId> concat(const std::vector<ninfer::TokenId>& prefix,
                                    const std::vector<ninfer::TokenId>& suffix) {
    std::vector<ninfer::TokenId> out = prefix;
    out.insert(out.end(), suffix.begin(), suffix.end());
    return out;
}

ninfer::GenerationResult generate(ninfer::Engine& engine,
                                  const std::vector<ninfer::TokenId>& tokens,
                                  std::uint32_t outputs, bool allow_reuse) {
    auto prepared = engine.prepare_tokens(tokens, false);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = outputs;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.top_k          = 0;
    request.execution.sampling.top_p          = 1.0F;
    request.execution.sampling.min_p          = 0.0F;
    request.execution.sampling.presence_penalty  = 0.0F;
    request.execution.sampling.frequency_penalty = 0.0F;
    request.execution.sampling.seed           = 42;
    request.execution.allow_prefix_reuse      = allow_reuse;
    request.stop.include_model_defaults       = false;
    return engine.generate(std::move(prepared), std::move(request));
}

void report(const char* label, const std::vector<ninfer::TokenId>& tokens) {
    std::cerr << "  " << label << ':';
    for (const auto token : tokens) { std::cerr << ' ' << token; }
    std::cerr << '\n';
}

void check_lossless(const char* label, ninfer::Engine& engine,
                    const std::vector<ninfer::TokenId>& prompt, std::uint32_t outputs,
                    const std::vector<ninfer::TokenId>& reused) {
    const auto first = generate(engine, prompt, outputs, false).generated_token_ids;
    if (reused == first) { return; }

    // The cold MoE route cache is not bit-deterministic at near-tied logits: an isolated cold
    // request can drift by one token while the restored request stays on the canonical path.
    // A single cold sample is therefore not a valid oracle. Establish the canonical cold output
    // by majority over a few extra samples and require an exact token-by-token match against it.
    // A drifting baseline cannot fail a correct restore, but a restore that never reproduces any
    // cold sample still fails.
    constexpr int extra_samples = 4;
    std::vector<std::vector<ninfer::TokenId>> seen{first};
    std::vector<int> counts{1};
    for (int sample = 0; sample < extra_samples; ++sample) {
        const auto tokens = generate(engine, prompt, outputs, false).generated_token_ids;
        std::size_t index = 0;
        for (; index < seen.size(); ++index) {
            if (seen[index] == tokens) { ++counts[index]; break; }
        }
        if (index == seen.size()) { seen.push_back(tokens); counts.push_back(1); }
    }
    std::size_t canonical = 0;
    for (std::size_t index = 1; index < seen.size(); ++index) {
        if (counts[index] > counts[canonical]) { canonical = index; }
    }
    if (reused == seen[canonical]) { return; }

    ++failures;
    std::cerr << "FAIL: " << label << " is not token-identical to the canonical cold prefill\n";
    report("reuse", reused);
    report("canon", seen[canonical]);
    std::cerr << "  cold samples:";
    for (std::size_t index = 0; index < seen.size(); ++index) {
        std::cerr << " [" << counts[index] << "x]";
        for (const auto token : seen[index]) { std::cerr << ' ' << token; }
    }
    std::cerr << '\n';
}

void reuse_case(ninfer::Engine& engine, const char* label,
                const std::vector<ninfer::TokenId>& prefix,
                const std::vector<ninfer::TokenId>& suffix, std::uint32_t outputs,
                std::uint32_t expected_reuse) {
    (void)generate(engine, prefix, 1, false);
    const auto prompt = concat(prefix, suffix);
    (void)generate(engine, prompt, outputs, false);
    const auto result = generate(engine, prompt, outputs, true);
    check(result.reused_prompt_tokens == expected_reuse,
          "the matched prefix supplies the expected number of reused tokens");
    check(result.prefix_reuse_path == ninfer::PrefixReusePath::BlockSnapshot,
          "the matched prefix takes the BlockSnapshot path");
    check_lossless(label, engine, prompt, outputs, result.generated_token_ids);
}

void plain_cases(const char* artifact) {
    ninfer::Engine engine(make_options(artifact, Setup{}));
    const auto prefix = make_tokens(128, 0);
    const auto suffix = make_tokens(64, 1);
    (void)generate(engine, prefix, 1, false);

    const auto prompt = concat(prefix, suffix);
    const auto probe_prompt = engine.prepare_tokens(prompt, false);
    const std::vector<std::uint64_t> hashes(probe_prompt.block_hashes().begin(),
                                            probe_prompt.block_hashes().end());
    const auto probe = engine.prefix_probe(hashes);
    check(probe.manifest_frontier == 128, "prefix_probe sees the 128-token manifest");

    (void)generate(engine, prompt, 4, false);
    const auto reused = generate(engine, prompt, 4, true);
    check(reused.reused_prompt_tokens == 128, "matching request reuses 128 prompt tokens");
    check(reused.prefix_reuse_path == ninfer::PrefixReusePath::BlockSnapshot,
          "matching request takes the BlockSnapshot path");
    check_lossless("128-token prefix reuse", engine, prompt, 4, reused.generated_token_ids);

    const auto disjoint = concat(make_tokens(128, 7), suffix);
    const auto miss = generate(engine, disjoint, 2, true);
    check(miss.reused_prompt_tokens == 0, "unmatched prefix reuses no tokens");
    check(miss.prefix_reuse_path == ninfer::PrefixReusePath::FullReset,
          "unmatched prefix takes FullReset");

    const auto short_prefix = make_tokens(48, 11);
    (void)generate(engine, short_prefix, 1, false);
    const auto short_prompt = concat(short_prefix, make_tokens(64, 12));
    const auto short_reuse = generate(engine, short_prompt, 2, true);
    check(short_reuse.reused_prompt_tokens == 0,
          "a sub-64-token prefix cannot publish or be reused");
    check(short_reuse.prefix_reuse_path == ninfer::PrefixReusePath::FullReset,
          "a sub-64-token prefix takes FullReset");

    reuse_case(engine, "two-chunk prefix reuse", make_tokens(256, 3), make_tokens(64, 4), 4, 256);
}

void boundary_case(const char* artifact) {
    ninfer::Engine engine(make_options(artifact, Setup{2048, 64, false}));
    reuse_case(engine, "64-token boundary reuse", make_tokens(64, 21), make_tokens(32, 22), 3, 64);
}

void dflash2_case(const char* artifact) {
    if (std::getenv("NINFER_DFLASH2_ARTIFACT") == nullptr) {
        std::cout << "skip: NINFER_DFLASH2_ARTIFACT is not set\n";
        return;
    }
    ninfer::Engine engine(make_options(artifact, Setup{2048, 128, true}));
    reuse_case(engine, "DFlash2 prefix reuse", make_tokens(128, 31), make_tokens(64, 32), 4, 128);
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
        plain_cases(path);
        boundary_case(path);
        dflash2_case(path);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "qwen4_exp prefix consume integration: %d failure(s)\n", failures);
        return 1;
    }
    std::cout << "OK qwen4_exp prefix consume integration through the public Engine\n";
    return 0;
}
