#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include "targets/qwen3_8_flash_next/impl/variant.h"
#include "targets/qwen4_exp/impl/runtime/activation_dtype.h"

#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen4_exp/runtime.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace flash_harness {

using ninfer::DType;
using ninfer::DeviceArena;
using ninfer::ExecutionContext;
using ninfer::Tensor;
using ninfer::TokenId;

namespace artifact = ninfer::artifact;
namespace detail = ninfer::targets::qwen3_8_flash_next::detail;
using Package = ninfer::targets::qwen3_8_flash_next::Package;

inline void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

inline std::vector<std::vector<std::uint16_t>>* g_logits_sink = nullptr;

inline std::vector<std::vector<std::uint16_t>>& capture_buffer() {
    static std::vector<std::vector<std::uint16_t>> buffer;
    return buffer;
}

inline void capture_logits(const Tensor& logits, cudaStream_t stream, int rank, int) {
    if (g_logits_sink == nullptr || rank != 0) { return; }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint16_t> words(logits.bytes() / 2);
    CUDA_CHECK(cudaMemcpy(words.data(), logits.data, logits.bytes(), cudaMemcpyDeviceToHost));
    g_logits_sink->push_back(std::move(words));
}

} // namespace flash_harness

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_runtime
#define NINFER_QWEN4EXP_LAST_STAGE(sample, logits) \
    ::flash_harness::capture_logits((logits), stream_for(rank), rank, T)
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"
#undef NINFER_QWEN4EXP_LAST_STAGE
#undef NINFER_QWEN4EXP_RUNTIME_NS
#undef NINFER_QWEN4EXP_VARIANT

namespace flash_harness {

namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_runtime;

constexpr std::uint32_t kKvCapacity = 262144;
constexpr std::size_t kForwardTokens = 8;

inline std::uint64_t fnv(std::uint64_t hash, const void* data, std::size_t bytes) {
    const auto* bytes_ptr = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= static_cast<std::uint64_t>(bytes_ptr[index]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::uint64_t hash_tensor(std::uint64_t hash, const Tensor& tensor) {
    std::vector<std::uint16_t> words(tensor.bytes() / 2);
    CUDA_CHECK(cudaMemcpy(words.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return fnv(hash, words.data(), words.size() * 2);
}

inline std::uint64_t hash_region(std::uint64_t hash, const Tensor& tensor, std::size_t offset,
                                 std::size_t bytes) {
    std::vector<std::uint16_t> words(bytes / 2);
    CUDA_CHECK(cudaMemcpy(words.data(), static_cast<const std::byte*>(tensor.data) + offset, bytes,
                          cudaMemcpyDeviceToHost));
    return fnv(hash, words.data(), words.size() * 2);
}

inline std::uint64_t kv_region_digest(rt::TextState& state, std::uint32_t position,
                                      std::uint32_t tokens) {
    CUDA_CHECK(cudaSetDevice(state.device_id()));
    CUDA_CHECK(cudaStreamSynchronize(state.stream()));
    std::uint64_t hash = 1469598103934665603ULL;
    if (tokens == 0) { return hash; }
    const std::uint32_t last = position + tokens - 1;
    const int first_page = static_cast<int>(position / 64);
    const int last_page  = static_cast<int>(last / 64);
    for (int slot = 0; slot < detail::TextConfig::qsa_layers(); ++slot) {
        const ninfer::ops::QsaCacheView cache = state.cache(slot);
        const std::size_t token_bytes = static_cast<std::size_t>(cache.k_pages.ne[0]) *
                                        static_cast<std::size_t>(cache.k_pages.ne[1]) *
                                        static_cast<std::size_t>(cache.k_pages.ne[2]) * 2;
        const std::size_t index_bytes = static_cast<std::size_t>(cache.index_k_pages.ne[0]) *
                                        static_cast<std::size_t>(cache.index_k_pages.ne[1]) *
                                        static_cast<std::size_t>(cache.index_k_pages.ne[2]) * 2;
        for (int page = first_page; page <= last_page; ++page) {
            hash = hash_region(hash, cache.k_pages, static_cast<std::size_t>(page) * token_bytes,
                               token_bytes);
            hash = hash_region(hash, cache.v_pages, static_cast<std::size_t>(page) * token_bytes,
                               token_bytes);
            hash = hash_region(hash, cache.index_k_pages,
                               static_cast<std::size_t>(page) * index_bytes, index_bytes);
        }
    }
    return hash;
}

inline std::uint64_t state_digest(rt::TextState& state) {
    CUDA_CHECK(cudaSetDevice(state.device_id()));
    CUDA_CHECK(cudaStreamSynchronize(state.stream()));
    std::uint64_t hash = 1469598103934665603ULL;
    const std::uint32_t position = state.next_position();
    hash = fnv(hash, &position, sizeof(position));
    const std::uint64_t tail[2] = {state.ple_tail(0), state.ple_tail(1)};
    hash = fnv(hash, tail, sizeof(tail));
    for (std::size_t index = 0; index < state.gdn_state_entries(); ++index) {
        hash = hash_tensor(hash, state.gdn_conv_state(index));
        hash = hash_tensor(hash, state.gdn_recurrent_state(index));
    }
    for (int slot = 0; slot < detail::TextConfig::qsa_layers(); ++slot) {
        hash = hash_tensor(hash, state.before_tail(slot).raw_keys);
        hash = hash_tensor(hash, state.before_tail(slot).first_rope);
    }
    hash = hash_tensor(hash, state.tail(0).raw_keys);
    hash = hash_tensor(hash, state.tail(0).first_rope);
    hash = hash_tensor(hash, state.ple_history());
    std::int32_t route = 0;
    CUDA_CHECK(cudaMemcpy(&route, state.route_counter(), sizeof(route), cudaMemcpyDeviceToHost));
    return fnv(hash, &route, sizeof(route));
}

class FlashHarness {
public:
    explicit FlashHarness(const char* path, std::uint32_t replay_width = 0)
        : execution({0, 1}), reader(path), binder(reader, 2) {
        ninfer::EngineOptions options;
        options.artifact_path = path;
        options.device = 0;
        options.tp = 2;
        options.devices = {0, 1};
        options.max_context = kKvCapacity;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
        options.max_concurrency = 1;
        options.prefill_chunk = 64;
        options.expert_cache_mib = 0;

        auto load_plan = Package::plan_load(binder, options, detail::WeightsProfile::Nvfp4);
        materialized = std::make_unique<artifact::MaterializedArtifact>(
            artifact::materialize(reader, load_plan.materialization(), execution));
        data = std::make_unique<detail::LoadedModelData>(
            load_plan.bindings(), *materialized, detail::TextConfig::layers, 2,
            std::filesystem::path(path));
        data->allocate_expert_cache(options, execution);

        const auto recipe = rt::text_workspace_recipe(kForwardTokens, 1);
        comms = std::make_unique<ninfer::ops::CollectiveComms>(execution);
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            int previous = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
            states[slot] = std::make_unique<rt::TextState>(
                *execution.dev[slot], data->view(rank), kKvCapacity, kForwardTokens, 2, rank,
                replay_width == 0 ? kForwardTokens : replay_width);
            activations[slot] = std::make_unique<DeviceArena>(
                rt::text_activation_capacity_bytes(kForwardTokens));
            workspaces[slot] = std::make_unique<DeviceArena>(recipe.lane_capacity_bytes);
            state_ptrs[slot] = states[slot].get();
            view_ptrs[slot] = &data->view(rank);
            activation_ptrs[slot] = activations[slot].get();
            workspace_ptrs[slot] = workspaces[slot].get();
            CUDA_CHECK(cudaSetDevice(previous));
        }
        trunk = std::make_unique<rt::TextTrunk>(
            view_ptrs, state_ptrs, activation_ptrs, workspace_ptrs, execution, *comms);
    }

    void reset() {
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            int previous = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
            states[slot]->reset();
            activations[slot]->reset();
            workspaces[slot]->reset();
            // Test-only determinism: `DeviceArena::reset()` only rewinds the suballocation cursor,
            // so a later T=1 oracle pass could observe stale bytes left by an earlier T>1 batched
            // forward through allocator reuse. Zero the whole reused arena on this device so the
            // oracle is independent of which route ran before it. Production Engine does not zero
            // its arenas; this strictly removes allocator noise from the numerical comparison.
            CUDA_CHECK(cudaMemset(activations[slot]->base(), 0, activations[slot]->capacity()));
            CUDA_CHECK(cudaMemset(workspaces[slot]->base(), 0, workspaces[slot]->capacity()));
            CUDA_CHECK(cudaSetDevice(previous));
        }
    }

    TokenId prefill(std::span<const TokenId> tokens) { return trunk->prefill(tokens); }
    TokenId decode(TokenId token) { return trunk->decode(token); }

    std::uint64_t digest() {
        std::uint64_t hash = 1469598103934665603ULL;
        for (int rank = 0; rank < 2; ++rank) {
            hash = fnv(hash, &rank, sizeof(rank));
            const std::uint64_t rank_digest = state_digest(*states[static_cast<std::size_t>(rank)]);
            hash = fnv(hash, &rank_digest, sizeof(rank_digest));
        }
        return hash;
    }

    std::uint64_t kv_digest(std::uint32_t position, std::uint32_t tokens) {
        std::uint64_t hash = 1469598103934665603ULL;
        for (int rank = 0; rank < 2; ++rank) {
            hash = fnv(hash, &rank, sizeof(rank));
            const std::uint64_t rank_digest =
                kv_region_digest(*states[static_cast<std::size_t>(rank)], position, tokens);
            hash = fnv(hash, &rank_digest, sizeof(rank_digest));
        }
        return hash;
    }

    ExecutionContext execution;
    artifact::Reader reader;
    artifact::Binder binder;
    std::unique_ptr<artifact::MaterializedArtifact> materialized;
    std::unique_ptr<detail::LoadedModelData> data;
    std::unique_ptr<ninfer::ops::CollectiveComms> comms;
    std::array<std::unique_ptr<rt::TextState>, 2> states;
    std::array<std::unique_ptr<DeviceArena>, 2> activations;
    std::array<std::unique_ptr<DeviceArena>, 2> workspaces;
    std::array<rt::TextState*, ninfer::kMaximumDevices> state_ptrs{};
    std::array<const rt::Variant::ModelView*, ninfer::kMaximumDevices> view_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> activation_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> workspace_ptrs{};
    std::unique_ptr<rt::TextTrunk> trunk;
};

inline std::int32_t argmax_at(const std::vector<std::uint16_t>& words, int vocab, int position) {
    std::int32_t best = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    const std::size_t base = static_cast<std::size_t>(position) * static_cast<std::size_t>(vocab);
    for (int index = 0; index < vocab; ++index) {
        __half value;
        std::memcpy(&value, &words[base + static_cast<std::size_t>(index)], sizeof(value));
        const float as_float = __half2float(value);
        if (as_float > best_value) {
            best_value = as_float;
            best = index;
        }
    }
    return best;
}

} // namespace flash_harness
