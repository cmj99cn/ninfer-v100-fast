#include <ninfer/engine.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
using ninfer::MemorySummary;
using ninfer::RequestOptions;
using ninfer::TokenId;
using Json = nlohmann::json;

constexpr std::uint32_t kKvCapacity     = 262144;
constexpr std::uint32_t kPrefillChunk   = 1024;
constexpr std::uint32_t kMaxConcurrency = 1;
constexpr std::int32_t kVocab           = 248320;
constexpr std::array<TokenId, 8> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};

struct Options {
    std::string artifact;
    std::string mode = "probe";
    std::vector<TokenId> prompt{kPrompt.begin(), kPrompt.end()};
    std::vector<std::uint32_t> cases{512, 3700, 15000, 59000};
    std::uint32_t decode_tokens = 32;
    int reps                    = 2;
    int warmup                  = 1;
    std::uint32_t prefill_chunk = kPrefillChunk;
    std::vector<TokenId> target_tokens;
};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<TokenId> parse_ids(const std::string& text) {
    std::vector<TokenId> ids;
    std::string current;
    for (std::size_t index = 0; index <= text.size(); ++index) {
        const char value = index == text.size() ? ',' : text[index];
        if (value == ',' || value == ' ') {
            if (!current.empty()) {
                ids.push_back(static_cast<TokenId>(std::stol(current)));
                current.clear();
            }
        } else {
            current.push_back(value);
        }
    }
    require(!ids.empty(), "the prompt token list is empty");
    return ids;
}

std::vector<std::uint32_t> parse_u32_list(const std::string& text) {
    std::vector<std::uint32_t> values;
    std::string current;
    for (std::size_t index = 0; index <= text.size(); ++index) {
        const char value = index == text.size() ? ',' : text[index];
        if (value == ',' || value == ' ') {
            if (!current.empty()) {
                values.push_back(static_cast<std::uint32_t>(std::stoul(current)));
                current.clear();
            }
        } else {
            current.push_back(value);
        }
    }
    require(!values.empty(), "the case list is empty");
    return values;
}

const char* environment(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? nullptr : value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (const char* value = environment("NINFER_FLASH_NEXT_ARTIFACT")) { options.artifact = value; }
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        const auto next = [&]() -> std::string {
            require(index + 1 < argc, "missing value for " + arg);
            return argv[++index];
        };
        if (arg == "--artifact") {
            options.artifact = next();
        } else if (arg == "--bench") {
            options.mode = "bench";
        } else if (arg == "--probe") {
            options.mode = "probe";
        } else if (arg == "--prompt-ids") {
            options.prompt = parse_ids(next());
        } else if (arg == "--cases") {
            options.cases = parse_u32_list(next());
        } else if (arg == "--decode-tokens") {
            options.decode_tokens = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (arg == "--reps") {
            options.reps = std::stoi(next());
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(next());
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (arg == "--target-tokens") {
            options.target_tokens = parse_ids(next());
        } else {
            throw std::invalid_argument("unknown argument " + arg);
        }
    }
    require(!options.artifact.empty(), "an artifact path is required");
    require(options.decode_tokens > 0, "the decode token budget must be positive");
    require(options.reps > 0, "the repetition count must be positive");
    require(options.warmup >= 0, "the warmup count must be non-negative");
    return options;
}

EngineOptions engine_options(const Options& options) {
    EngineOptions engine;
    engine.artifact_path    = options.artifact;
    engine.device           = 0;
    engine.tp               = 2;
    engine.devices          = {0, 1};
    engine.max_context      = kKvCapacity;
    engine.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
    engine.max_concurrency  = kMaxConcurrency;
    engine.prefill_chunk    = options.prefill_chunk;
    engine.use_cuda_graph   = false;
    engine.expert_cache_mib = 0;
    return engine;
}

RequestOptions request_options(std::uint32_t tokens) {
    RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.allow_prefix_reuse      = false;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;
    return request;
}

std::vector<TokenId> cycle_prompt(const std::vector<TokenId>& base, std::uint32_t tokens) {
    std::vector<TokenId> prompt(static_cast<std::size_t>(tokens));
    for (std::uint32_t index = 0; index < tokens; ++index) {
        prompt[index] = base[static_cast<std::size_t>(index) % base.size()];
    }
    return prompt;
}

double half_words_to_float(std::uint16_t bits) {
    __half_raw raw{};
    raw.x = bits;
    return static_cast<double>(__half2float(static_cast<__half>(raw)));
}

double log_softmax_nll(const std::vector<std::uint16_t>& words, TokenId token) {
    require(words.size() >= static_cast<std::size_t>(kVocab), "the captured logits are shorter than the vocabulary");
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < kVocab; ++index) {
        maximum = std::max(maximum, half_words_to_float(words[static_cast<std::size_t>(index)]));
    }
    require(std::isfinite(maximum), "the captured logits hold no finite value");
    double sum = 0.0;
    for (std::int32_t index = 0; index < kVocab; ++index) {
        sum += std::exp(half_words_to_float(words[static_cast<std::size_t>(index)]) - maximum);
    }
    require(sum > 0.0 && std::isfinite(sum), "the captured logits have an invalid partition");
    const double log_partition = maximum + std::log(sum);
    require(token >= 0 && token < kVocab, "the sampled token is outside the vocabulary");
    return log_partition - half_words_to_float(words[static_cast<std::size_t>(token)]);
}

Json memory_json(const MemorySummary& memory) {
    return Json{
        {"device", memory.device},
        {"weights_bytes", memory.weights.capacity_bytes},
        {"sequence_bytes", memory.sequence.capacity_bytes},
        {"workspace_bytes", memory.workspace.capacity_bytes},
        {"kv_payload_bytes", memory.kv_payload_bytes},
        {"gdn_state_bytes", memory.gdn_state_bytes},
        {"expert_cache_bytes", memory.expert_cache_bytes},
        {"expert_cache_pairs", memory.expert_cache_pairs},
        {"expert_host_bytes", memory.expert_host_bytes},
        {"host_resident_bytes", memory.host_resident_bytes},
        {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
        {"kv_capacity", memory.kv_capacity},
    };
}

Json device_json(int device) {
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, device);
    return Json{{"name", properties.name}, {"major", properties.major}, {"minor", properties.minor},
                {"memory_bytes", properties.totalGlobalMem}};
}

Json run_probe(const Options& options) {
    Engine engine(engine_options(options));
    engine.debug_enable_logit_capture(true);

    const MemorySummary before = engine.memory_summary();
    const GenerationResult batched =
        engine.generate(engine.prepare_tokens(options.prompt, false), request_options(8));
    const MemorySummary after = engine.memory_summary();
    require(batched.generated_token_ids.size() == 8, "the probe request did not return eight tokens");

    const std::vector<TokenId>& targets =
        options.target_tokens.empty() ? batched.generated_token_ids : options.target_tokens;
    require(targets.size() == 8, "the teacher-forced target sequence must hold eight tokens");
    std::vector<TokenId> sequence = options.prompt;
    std::vector<double> nll;
    Json steps = Json::array();
    nll.reserve(8);
    for (std::size_t step = 0; step < 8; ++step) {
        const GenerationResult result =
            engine.generate(engine.prepare_tokens(sequence, false), request_options(1));
        const std::vector<std::uint16_t> words = engine.debug_last_round_logits_bf16();
        nll.push_back(log_softmax_nll(words, targets[step]));
        std::vector<std::pair<int, float>> top;
        top.reserve(4);
        for (std::int32_t index = 0; index < kVocab; ++index) {
            const float value = static_cast<float>(half_words_to_float(words[static_cast<std::size_t>(index)]));
            top.emplace_back(index, value);
            std::sort(top.begin(), top.end(),
                      [](const auto& left, const auto& right) { return left.second > right.second; });
            if (top.size() > 4) { top.pop_back(); }
        }
        steps.push_back(Json{{"target", targets[step]},
                             {"nll", nll.back()},
                             {"argmax", result.generated_token_ids.front()},
                             {"top", top}});
        sequence.push_back(targets[step]);
    }

    double mean_nll = 0.0;
    for (const double value : nll) { mean_nll += value; }
    mean_nll /= static_cast<double>(nll.size());

    return Json{
        {"mode", "probe"},
        {"artifact", options.artifact},
        {"prompt", options.prompt},
        {"tokens", batched.generated_token_ids},
        {"target_tokens", targets},
        {"nll", nll},
        {"steps", steps},
        {"mean_nll", mean_nll},
        {"sampling", "greedy"},
        {"acceptance", "N/A"},
        {"timings",
         {{"prefill_seconds", batched.timings.prefill_seconds},
          {"first_token_seconds", batched.timings.first_token_seconds},
          {"decode_seconds", batched.timings.decode_seconds},
          {"total_seconds", batched.timings.total_seconds}}},
        {"expert",
         {{"hits", after.expert_cache_hits - before.expert_cache_hits},
          {"misses", after.expert_cache_misses - before.expert_cache_misses},
          {"miss_bytes", after.expert_cache_miss_bytes - before.expert_cache_miss_bytes}}},
        {"memory", memory_json(after)},
        {"device", device_json(0)},
    };
}

Json run_bench(const Options& options) {
    Engine engine(engine_options(options));
    const MemorySummary startup = engine.memory_summary();

    Json cases = Json::array();
    for (const std::uint32_t prompt_tokens : options.cases) {
        const std::vector<TokenId> prompt = cycle_prompt(options.prompt, prompt_tokens);
        for (int index = 0; index < options.warmup; ++index) {
            (void)engine.generate(engine.prepare_tokens(prompt, false),
                                  request_options(options.decode_tokens));
        }

        Json reps = Json::array();
        double decode_seconds_sum = 0.0;
        double committed_sum      = 0.0;
        std::uint64_t hits_sum    = 0;
        std::uint64_t misses_sum  = 0;
        std::uint64_t miss_bytes  = 0;
        double ttft_sum           = 0.0;
        for (int index = 0; index < options.reps; ++index) {
            const MemorySummary before = engine.memory_summary();
            const GenerationResult result =
                engine.generate(engine.prepare_tokens(prompt, false),
                                request_options(options.decode_tokens));
            const MemorySummary after = engine.memory_summary();
            require(result.generated_token_ids.size() == options.decode_tokens,
                    "a benchmark request did not return the requested token count");
            const double committed = static_cast<double>(result.generated_token_ids.size() - 1);
            const double rate =
                result.timings.decode_seconds > 0.0 ? committed / result.timings.decode_seconds : 0.0;
            const std::uint64_t hits = after.expert_cache_hits - before.expert_cache_hits;
            const std::uint64_t misses = after.expert_cache_misses - before.expert_cache_misses;
            const std::uint64_t missed_bytes =
                after.expert_cache_miss_bytes - before.expert_cache_miss_bytes;
            reps.push_back(Json{
                {"committed_tokens", result.generated_token_ids.size() - 1},
                {"generated_tokens", result.generated_token_ids.size()},
                {"prefill_seconds", result.timings.prefill_seconds},
                {"first_token_seconds", result.timings.first_token_seconds},
                {"decode_seconds", result.timings.decode_seconds},
                {"total_seconds", result.timings.total_seconds},
                {"committed_tokens_per_second", rate},
                {"prompt_tokens_per_second",
                 result.timings.prefill_seconds > 0.0
                     ? static_cast<double>(prompt_tokens) / result.timings.prefill_seconds
                     : 0.0},
                {"expert_hits", hits},
                {"expert_misses", misses},
                {"expert_miss_bytes", missed_bytes},
                {"expert_hit_rate",
                 hits + misses == 0 ? 1.0 : static_cast<double>(hits) / static_cast<double>(hits + misses)},
            });
            decode_seconds_sum += result.timings.decode_seconds;
            committed_sum += committed;
            ttft_sum += result.timings.first_token_seconds;
            hits_sum += hits;
            misses_sum += misses;
            miss_bytes += missed_bytes;
        }
        const double reps_count = static_cast<double>(options.reps);
        cases.push_back(Json{
            {"prompt_tokens", prompt_tokens},
            {"decode_tokens", options.decode_tokens},
            {"occupied_context", prompt_tokens + options.decode_tokens},
            {"reps", std::move(reps)},
            {"mean",
             {{"committed_tokens_per_second",
               decode_seconds_sum > 0.0 ? committed_sum / decode_seconds_sum : 0.0},
              {"ttft_seconds", ttft_sum / reps_count},
              {"decode_seconds", decode_seconds_sum / reps_count},
              {"expert_hits", hits_sum},
              {"expert_misses", misses_sum},
              {"expert_miss_bytes", miss_bytes},
              {"expert_hit_rate",
               hits_sum + misses_sum == 0
                   ? 1.0
                   : static_cast<double>(hits_sum) / static_cast<double>(hits_sum + misses_sum)}}},
        });
    }

    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    return Json{
        {"mode", "bench"},
        {"artifact", options.artifact},
        {"prompt_tokens", options.prompt.size()},
        {"decode_tokens", options.decode_tokens},
        {"reps", options.reps},
        {"warmup", options.warmup},
        {"sampling", "greedy"},
        {"acceptance", "N/A"},
        {"engine",
         {{"tp", 2},
          {"devices", {0, 1}},
          {"kv_capacity", kKvCapacity},
          {"prefill_chunk", options.prefill_chunk},
          {"expert_cache_mib", 0},
          {"max_concurrency", kMaxConcurrency},
          {"kv_cache", "MixedBf16Fp16"}}},
        {"cuda_runtime", std::to_string(runtime_version / 1000) + "." +
                             std::to_string((runtime_version % 1000) / 10)},
        {"device", device_json(0)},
        {"startup_memory", memory_json(startup)},
        {"cases", std::move(cases)},
    };
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const Json report = options.mode == "bench" ? run_bench(options) : run_probe(options);
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "flash_next_engine_bench: " << error.what() << '\n';
        return 1;
    }
}
