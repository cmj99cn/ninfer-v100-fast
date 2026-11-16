#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include "targets/qwen4_exp/impl/runtime/activation_dtype.h"

#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen4_exp/runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DeviceArena;
using ninfer::ExecutionContext;
using ninfer::Tensor;
using ninfer::TokenId;

namespace artifact = ninfer::artifact;
namespace detail    = ninfer::targets::qwen3_8_flash_next::detail;
namespace qwen4_exp = ninfer::targets::qwen4_exp;
using Package       = ninfer::targets::qwen3_8_flash_next::Package;

constexpr std::uint32_t kKvCapacity = 262144;

std::vector<std::vector<std::uint16_t>> g_steps;
bool g_capture = false;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

const char* environment(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? nullptr : value;
}

std::vector<TokenId> read_tokens(const std::string& path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot read " + path); }
    std::vector<TokenId> tokens;
    long long value = 0;
    while (input >> value) {
        require(value >= 0 && value < detail::TextConfig::vocab,
                "token id is outside the vocabulary");
        tokens.push_back(static_cast<TokenId>(value));
    }
    require(tokens.size() >= 2, "the token file holds fewer than two ids");
    return tokens;
}

} // namespace

void flash_next_text_nll_capture(const Tensor& logits, cudaStream_t stream, int rank, int) {
    if (!g_capture || rank != 0) { return; }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint16_t> words(logits.bytes() / 2);
    CUDA_CHECK(cudaMemcpy(words.data(), logits.data, logits.bytes(), cudaMemcpyDeviceToHost));
    g_steps.push_back(std::move(words));
}

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_text_nll_runtime
#define NINFER_QWEN4EXP_LAST_STAGE(sample, logits) \
    ::flash_next_text_nll_capture((logits), stream_for(rank), rank, T)
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

namespace {

namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_text_nll_runtime;

double log_softmax_nll(const std::vector<std::uint16_t>& words, std::size_t position,
                       std::int32_t vocab, TokenId token, std::int32_t* argmax) {
    const std::size_t base = position * static_cast<std::size_t>(vocab);
    double maximum = -std::numeric_limits<double>::infinity();
    std::int32_t best = 0;
    for (std::int32_t index = 0; index < vocab; ++index) {
        const std::uint16_t bits = words[base + static_cast<std::size_t>(index)];
        double as_float = 0.0;
        if constexpr (qwen4_exp::detail::kTextActivationDtype == ninfer::DType::FP16) {
            __half value;
            std::memcpy(&value, &bits, sizeof(value));
            as_float = static_cast<double>(__half2float(value));
        } else {
            __nv_bfloat16 value;
            std::memcpy(&value, &bits, sizeof(value));
            as_float = static_cast<double>(__bfloat162float(value));
        }
        if (as_float > maximum) {
            maximum = as_float;
            best    = index;
        }
    }
    if (argmax != nullptr) { *argmax = best; }
    require(std::isfinite(maximum), "the captured logits hold no finite value");
    double sum = 0.0;
    for (std::int32_t index = 0; index < vocab; ++index) {
        const std::uint16_t bits = words[base + static_cast<std::size_t>(index)];
        double as_float = 0.0;
        if constexpr (qwen4_exp::detail::kTextActivationDtype == ninfer::DType::FP16) {
            __half value;
            std::memcpy(&value, &bits, sizeof(value));
            as_float = static_cast<double>(__half2float(value));
        } else {
            __nv_bfloat16 value;
            std::memcpy(&value, &bits, sizeof(value));
            as_float = static_cast<double>(__bfloat162float(value));
        }
        sum += std::exp(as_float - maximum);
    }
    require(sum > 0.0 && std::isfinite(sum), "the captured logits have an invalid partition");
    const std::uint16_t target_bits = words[base + static_cast<std::size_t>(token)];
    double target_value = 0.0;
    if constexpr (qwen4_exp::detail::kTextActivationDtype == ninfer::DType::FP16) {
        __half value;
        std::memcpy(&value, &target_bits, sizeof(value));
        target_value = static_cast<double>(__half2float(value));
    } else {
        __nv_bfloat16 value;
        std::memcpy(&value, &target_bits, sizeof(value));
        target_value = static_cast<double>(__bfloat162float(value));
    }
    return maximum + std::log(sum) - target_value;
}

void run(const char* path, const std::vector<TokenId>& tokens) {
    const std::uint32_t count = static_cast<std::uint32_t>(tokens.size());
    ExecutionContext execution({0, 1});
    artifact::Reader reader(path);
    artifact::Binder binder(reader, 2);

    ninfer::EngineOptions options;
    options.artifact_path    = path;
    options.device           = 0;
    options.tp               = 2;
    options.devices          = {0, 1};
    options.max_context      = kKvCapacity;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
    options.max_concurrency  = 1;
    options.prefill_chunk    = 1024;
    options.expert_cache_mib = 14000;

    auto load_plan = Package::plan_load(binder, options, detail::WeightsProfile::Nvfp4);
    const detail::BindingPlan& bindings = load_plan.bindings();
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, load_plan.materialization(), execution);
    detail::LoadedModelData data(bindings, materialized, detail::TextConfig::layers, 2,
                                 std::filesystem::path(path));
    data.allocate_expert_cache(options, execution);

    const auto recipe = rt::text_workspace_recipe(count, 1);
    std::unique_ptr<ninfer::ops::CollectiveComms> comms =
        std::make_unique<ninfer::ops::CollectiveComms>(execution);
    std::array<std::unique_ptr<rt::TextState>, ninfer::kMaximumDevices> states;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> activations;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> workspaces;
    std::array<rt::TextState*, ninfer::kMaximumDevices> state_ptrs{};
    std::array<const rt::Variant::ModelView*, ninfer::kMaximumDevices> view_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> activation_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> workspace_ptrs{};
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        int previous    = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
        states[slot] = std::make_unique<rt::TextState>(*execution.dev[slot], data.view(rank),
                                                       kKvCapacity, count, 2, rank);
        activations[slot] = std::make_unique<DeviceArena>(rt::text_activation_capacity_bytes(count));
        workspaces[slot]  = std::make_unique<DeviceArena>(recipe.lane_capacity_bytes);
        state_ptrs[slot]      = states[slot].get();
        view_ptrs[slot]       = &data.view(rank);
        activation_ptrs[slot] = activations[slot].get();
        workspace_ptrs[slot]  = workspaces[slot].get();
        CUDA_CHECK(cudaSetDevice(previous));
    }
    rt::TextTrunk trunk(view_ptrs, state_ptrs, activation_ptrs, workspace_ptrs, execution, *comms);

    const auto reset = [&]() {
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            int previous    = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
            states[slot]->reset();
            activations[slot]->reset();
            workspaces[slot]->reset();
            CUDA_CHECK(cudaSetDevice(previous));
        }
    };
    reset();
    g_capture = false;
    (void)trunk.prefill(std::span<const TokenId>{tokens.data(), tokens.size()});
    for (int rank = 0; rank < 2; ++rank) { execution.dev[rank]->synchronize(); }
    reset();

    g_steps.clear();
    g_capture = true;
    (void)trunk.prefill(std::span<const TokenId>{tokens.data(), tokens.size()});
    g_capture = false;
    for (int rank = 0; rank < 2; ++rank) { execution.dev[rank]->synchronize(); }

    require(g_steps.size() == 1, "the prefill did not capture one logits span");
    const std::int32_t vocab = detail::TextConfig::vocab;
    require(g_steps.front().size() == static_cast<std::size_t>(vocab) * count,
            "the captured logits extent does not match the token count");

    const std::vector<std::uint16_t>& words = g_steps.front();
    std::vector<double> values;
    values.reserve(count - 1);
    double total = 0.0;
    std::size_t matches = 0;
    for (std::size_t position = 0; position + 1 < count; ++position) {
        std::int32_t argmax = 0;
        const double value =
            log_softmax_nll(words, position, vocab, tokens[position + 1], &argmax);
        require(std::isfinite(value), "the step NLL is not finite");
        values.push_back(value);
        total += value;
        matches += argmax == static_cast<std::int32_t>(tokens[position + 1]) ? 1 : 0;
    }

    std::cout.precision(12);
    std::cout << "tokens " << count << '\n';
    std::cout << "mean_nll " << (total / static_cast<double>(values.size())) << '\n';
    std::cout << "top1_matches " << matches << '\n';
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::cout << "nll " << (index + 1) << ' ' << tokens[index + 1] << ' ' << values[index]
                  << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact;
        std::string token_path;
        if (const char* value = environment("NINFER_FLASH_NEXT_ARTIFACT")) { artifact = value; }
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            const auto next = [&]() -> std::string {
                require(index + 1 < argc, "missing value for " + argument);
                return argv[++index];
            };
            if (argument == "--artifact") {
                artifact = next();
            } else if (argument == "--tokens") {
                token_path = next();
            } else {
                throw std::invalid_argument("unknown argument " + argument);
            }
        }
        require(!artifact.empty(), "an artifact path is required");
        require(!token_path.empty(), "a token file is required");
        run(artifact.c_str(), read_tokens(token_path));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
