#include "core/arena.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "core/tensor.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include "targets/qwen4_exp/impl/runtime/activation_dtype.h"

#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen4_exp/runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace artifact = ninfer::artifact;
namespace detail = ninfer::targets::qwen3_8_flash_next::detail;
namespace ops = ninfer::ops;
using ninfer::DecodeGraphDefinition;
using ninfer::DecodeGraphExecutable;
using ninfer::DeviceArena;
using ninfer::ExecutionContext;
using ninfer::PinnedHostBuffer;
using ninfer::Tensor;
using ninfer::TokenId;
using Package = ninfer::targets::qwen3_8_flash_next::Package;

constexpr std::uint32_t kRanks = 2;
std::uint32_t g_rank_count = 2;
constexpr std::uint32_t kKvCapacity = 262144;
constexpr std::uint32_t kForwardTokens = 512;
constexpr std::uint32_t kWarmupTokens = 192;
constexpr std::uint32_t kReplayTokens = 6;

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct GraphInputBlock {
    std::vector<TokenId> ids;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> scalars;
    std::vector<std::int32_t> winner;
};

bool g_capturing = false;
bool g_graph_mode = false;
bool g_skip_ple_prepare = false;
std::array<GraphInputBlock, kRanks> g_graph_inputs;
std::array<std::unique_ptr<PinnedHostBuffer>, kRanks> g_ids_host;
std::array<std::unique_ptr<PinnedHostBuffer>, kRanks> g_positions_host;
std::array<std::unique_ptr<PinnedHostBuffer>, kRanks> g_scalars_host;
std::array<std::unique_ptr<PinnedHostBuffer>, kRanks> g_winner_host;
std::unique_ptr<PinnedHostBuffer> g_logits_host;

inline std::int32_t* graph_scalar(int rank, const char* label) {
    auto* scalars = static_cast<std::int32_t*>(g_scalars_host[static_cast<std::size_t>(rank)]->data());
    if (std::strcmp(label, "ple_offset") == 0) return scalars + 0;
    if (std::strcmp(label, "ple_count") == 0) return scalars + 1;
    if (std::strcmp(label, "live") == 0) return scalars + 2;
    if (std::strcmp(label, "row") == 0) return scalars + 3;
    throw std::runtime_error(std::string("unknown graph scalar label: ") + label);
}

inline void fill_graph_inputs(int rank, TokenId token, std::uint32_t position) {
    auto& block = g_graph_inputs[static_cast<std::size_t>(rank)];
    block.ids.assign(1, token);
    block.positions.assign(1, static_cast<std::int32_t>(position));
    if (block.scalars.empty()) block.scalars.assign(4, 0);
    block.scalars[0] = 0;
    block.scalars[1] = 1;
    block.scalars[2] = 1;
    block.scalars[3] = 0;
    block.winner.assign(1, 0);
    std::memcpy(g_ids_host[static_cast<std::size_t>(rank)]->data(), block.ids.data(), 4);
    std::memcpy(g_positions_host[static_cast<std::size_t>(rank)]->data(), block.positions.data(), 4);
    std::memcpy(g_scalars_host[static_cast<std::size_t>(rank)]->data(), block.scalars.data(),
                block.scalars.size() * 4);
    std::memset(g_winner_host[static_cast<std::size_t>(rank)]->data(), 0, 4);
}

std::vector<TokenId> make_prompt(std::uint32_t prompt_tokens) {
    constexpr std::array<TokenId, 8> base{1024, 7, 98140, 33, 4097, 512, 100, 2048};
    std::vector<TokenId> prompt(prompt_tokens);
    for (std::uint32_t index = 0; index < prompt_tokens; ++index) {
        prompt[index] = base[index % base.size()];
    }
    return prompt;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void write_static_profile(const std::filesystem::path& path) {
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("cannot write the static expert profile"); }
    for (std::uint32_t layer = 0; layer < detail::kTextLayers; ++layer) {
        for (std::uint32_t expert = 0; expert < detail::kRankExperts; ++expert) {
            output << layer << ' ' << expert << " 1\n";
        }
    }
}

} // namespace

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_graph_capture_runtime
#define NINFER_QWEN4EXP_GRAPH_IDS_SRC(rank, tokens)                    \
    (g_capturing ? static_cast<const void*>(g_ids_host[rank]->data()) \
                 : static_cast<const void*>((tokens).data()))
#define NINFER_QWEN4EXP_GRAPH_POSITIONS_SRC(rank, host_positions)                    \
    (g_capturing ? static_cast<const void*>(g_positions_host[rank]->data())         \
                 : static_cast<const void*>((host_positions).data()))
#define NINFER_QWEN4EXP_GRAPH_SCALARS_SRC(rank, value, label)                    \
    (g_capturing ? static_cast<const void*>(graph_scalar((rank), (label)))      \
                 : static_cast<const void*>(&(value)))
#define NINFER_QWEN4EXP_GRAPH_WINNER_PTR()                                  \
    (g_capturing ? static_cast<void*>(g_winner_host[0]->data()) : static_cast<void*>(&token))
#define NINFER_QWEN4EXP_GRAPH_WINNER_COPY(dst, src, bytes, stream) \
    CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes), cudaMemcpyDeviceToHost, (stream)))
#define NINFER_QWEN4EXP_GRAPH_PREPARE_PLE(rank, state, tokens) \
    do { if (!g_capturing && !g_skip_ple_prepare) { (state).prepare_ple_rows(tokens); } } while (0)
#define NINFER_QWEN4EXP_GRAPH_PLE_AWAIT(rank, state) \
    do { if (!g_capturing && !g_skip_ple_prepare) { (state).await_ple_rows(); } } while (0)
#define NINFER_QWEN4EXP_GRAPH_ROUTED_FINISH(weights, stream) \
    do { if (!g_capturing && !g_graph_mode) { Variant::post_mixer_routed_finish((weights), (stream)); } } while (0)
#define NINFER_QWEN4EXP_GRAPH_PREFETCH(weights, stream) \
    do { if (!g_capturing && !g_graph_mode) { Variant::post_mixer_prefetch((weights), (stream)); } } while (0)
#define NINFER_QWEN4EXP_GRAPH_FINISH_SYNC() \
    do { if (!g_capturing) { CUDA_CHECK(cudaStreamSynchronize(states_[0]->stream())); } } while (0)
#define NINFER_QWEN4EXP_LAST_STAGE(sample, logits)                                       \
    do {                                                                                 \
        if (rank == 0) {                                                                  \
            CUDA_CHECK(cudaMemcpyAsync(                                                    \
                g_logits_host->data(),                                                     \
                static_cast<const std::byte*>((logits).data) +                              \
                    static_cast<std::size_t>(static_cast<std::int32_t>(T) - 1) *            \
                        static_cast<std::size_t>(TextConfig::vocab) * 2,                    \
                static_cast<std::size_t>(TextConfig::vocab) * 2,                            \
                cudaMemcpyDeviceToHost, stream_for(rank)));                                 \
        }                                                                                 \
    } while (0)
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

namespace {

namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_graph_capture_runtime;

struct BufferSnapshot {
    void* device = nullptr;
    std::size_t bytes = 0;
    std::vector<std::byte> host;
};

struct RankSnapshot {
    int device = 0;
    std::vector<BufferSnapshot> buffers;
    std::uint32_t next_position = 0;
    std::uint64_t ple_tail[2]{0, 0};
};

struct StateSnapshot {
    std::vector<RankSnapshot> ranks;

    void capture_from(const std::vector<rt::TextState*>& states) {
        ranks.clear();
        for (rt::TextState* state : states) {
            RankSnapshot rank;
            rank.device = state->device_id();
            int previous = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            CUDA_CHECK(cudaSetDevice(rank.device));
            const auto add = [&](const void* device, std::size_t bytes) {
                if (device == nullptr || bytes == 0) { return; }
                BufferSnapshot buffer;
                buffer.device = const_cast<void*>(device);
                buffer.bytes = bytes;
                buffer.host.resize(bytes);
                CUDA_CHECK(cudaMemcpy(buffer.host.data(), device, bytes, cudaMemcpyDeviceToHost));
                rank.buffers.push_back(std::move(buffer));
            };
            for (std::size_t index = 0; index < state->gdn_state_entries(); ++index) {
                const Tensor conv = state->gdn_conv_state(index);
                const Tensor recurrent = state->gdn_recurrent_state(index);
                add(conv.data, conv.bytes());
                add(recurrent.data, recurrent.bytes());
            }
            std::vector<std::int32_t> slots;
            for (std::size_t layer = 0; layer < static_cast<std::size_t>(detail::kTextLayers); ++layer) {
                if (detail::TextConfig::is_qsa_layer(static_cast<int>(layer))) {
                    slots.push_back(state->qsa_slot(layer));
                }
            }
            std::sort(slots.begin(), slots.end());
            slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
            bool page_table_added = false;
            bool after_added = false;
            for (std::int32_t slot : slots) {
                const auto cache = state->cache(slot);
                add(cache.k_pages.data, cache.k_pages.bytes());
                add(cache.v_pages.data, cache.v_pages.bytes());
                add(cache.index_k_pages.data, cache.index_k_pages.bytes());
                if (!page_table_added) {
                    add(cache.page_table.data, cache.page_table.bytes());
                    page_table_added = true;
                }
                const auto before = state->before_tail(slot);
                add(before.raw_keys.data, before.raw_keys.bytes());
                add(before.first_rope.data, before.first_rope.bytes());
                if (!after_added) {
                    const auto after = state->tail(slot);
                    add(after.raw_keys.data, after.raw_keys.bytes());
                    add(after.first_rope.data, after.first_rope.bytes());
                    after_added = true;
                }
            }
            const auto records = state->records(kForwardTokens);
            add(records.raw_keys.data, records.raw_keys.bytes());
            add(records.rope_positions.data, records.rope_positions.bytes());
            add(records.base_positions.data, records.base_positions.bytes());
            const Tensor history = state->ple_history();
            add(history.data, history.bytes());
            add(state->ple_rows(),
                static_cast<std::size_t>(ops::kPleHeads) * kForwardTokens * 4);
            add(state->ple_staged_rows(),
                static_cast<std::size_t>(ops::kPleHeads) * kForwardTokens * 128);
            add(state->route_counter(), 4);
            rank.next_position = state->next_position();
            for (int i = 0; i < 2; ++i) { rank.ple_tail[i] = state->ple_tail(i); }
            ranks.push_back(std::move(rank));
            CUDA_CHECK(cudaSetDevice(previous));
        }
    }

    void restore_to(const std::vector<rt::TextState*>& states) const {
        require(ranks.size() == states.size(), "snapshot rank count mismatch");
        for (std::size_t index = 0; index < states.size(); ++index) {
            const RankSnapshot& rank = ranks[index];
            rt::TextState* state = states[index];
            int previous = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            CUDA_CHECK(cudaSetDevice(rank.device));
            for (const BufferSnapshot& buffer : rank.buffers) {
                CUDA_CHECK(cudaMemcpy(buffer.device, buffer.host.data(), buffer.bytes,
                                      cudaMemcpyHostToDevice));
            }
            state->advance_position(rank.next_position - state->next_position());
            state->advance_ple_tail(rank.ple_tail[0]);
            state->advance_ple_tail(rank.ple_tail[1]);
            CUDA_CHECK(cudaSetDevice(previous));
        }
    }

    std::uint64_t hash() const {
        std::uint64_t value = 1469598103934665603ULL;
        const auto fold = [&](const void* data, std::size_t bytes) {
            const auto* bytes_ptr = static_cast<const std::uint8_t*>(data);
            for (std::size_t i = 0; i < bytes; ++i) {
                value ^= static_cast<std::uint64_t>(bytes_ptr[i]);
                value *= 1099511628211ULL;
            }
        };
        for (const RankSnapshot& rank : ranks) {
            for (const BufferSnapshot& buffer : rank.buffers) { fold(buffer.host.data(), buffer.bytes); }
            fold(&rank.next_position, sizeof(rank.next_position));
            fold(rank.ple_tail, sizeof(rank.ple_tail));
        }
        return value;
    }
};

struct Trunks {
    std::array<std::unique_ptr<rt::TextState>, ninfer::kMaximumDevices> states;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> activations;
    std::array<std::unique_ptr<DeviceArena>, ninfer::kMaximumDevices> workspaces;
    std::array<rt::TextState*, ninfer::kMaximumDevices> state_ptrs{};
    std::array<const rt::Variant::ModelView*, ninfer::kMaximumDevices> view_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> activation_ptrs{};
    std::array<DeviceArena*, ninfer::kMaximumDevices> workspace_ptrs{};
    std::unique_ptr<rt::TextTrunk> trunk;
};

void run(const char* path) {
    if (const char* ranks = std::getenv("NINFER_GRAPH_TP")) {
        g_rank_count = static_cast<std::uint32_t>(std::stoul(ranks));
    }
    require(g_rank_count >= 1 && g_rank_count <= kRanks, "NINFER_GRAPH_TP must be 1 or 2");
    std::vector<int> device_ids;
    for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) {
        device_ids.push_back(static_cast<int>(rank));
    }
    ExecutionContext execution(device_ids);
    artifact::Reader reader(path);
    artifact::Binder binder(reader, static_cast<int>(g_rank_count));

    ninfer::EngineOptions options;
    options.artifact_path = path;
    options.device = 0;
    options.tp = static_cast<int>(g_rank_count);
    options.devices = device_ids;
    options.max_context = kKvCapacity;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
    options.max_concurrency = 1;
    options.prefill_chunk = kForwardTokens;
    options.expert_cache_mib = 20000;
    const std::filesystem::path profile_path = "/tmp/ninfer_flash_next_static.profile";
    write_static_profile(profile_path);
    options.expert_cache_profile = profile_path;

    auto load_plan = Package::plan_load(binder, options, detail::WeightsProfile::Nvfp4);
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, load_plan.materialization(), execution);
    detail::LoadedModelData data(load_plan.bindings(), materialized, detail::TextConfig::layers,
                                 static_cast<int>(g_rank_count), std::filesystem::path(path));
    data.allocate_expert_cache(options, execution);

    const auto recipe = rt::text_workspace_recipe(kForwardTokens, 1);
    auto comms = std::make_unique<ninfer::ops::CollectiveComms>(execution);

    g_logits_host = std::make_unique<PinnedHostBuffer>(
        static_cast<std::size_t>(detail::TextConfig::vocab) * 2);

    Trunks trunks;
    for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
        trunks.states[slot] = std::make_unique<rt::TextState>(
            *execution.dev[slot], data.view(rank), kKvCapacity, kForwardTokens, static_cast<int>(g_rank_count), static_cast<int>(rank));
        trunks.activations[slot] =
            std::make_unique<DeviceArena>(rt::text_activation_capacity_bytes(kForwardTokens));
        trunks.workspaces[slot] = std::make_unique<DeviceArena>(recipe.lane_capacity_bytes);
        trunks.state_ptrs[slot] = trunks.states[slot].get();
        trunks.view_ptrs[slot] = &data.view(rank);
        trunks.activation_ptrs[slot] = trunks.activations[slot].get();
        trunks.workspace_ptrs[slot] = trunks.workspaces[slot].get();
        g_ids_host[slot] = std::make_unique<PinnedHostBuffer>(8192);
        g_positions_host[slot] = std::make_unique<PinnedHostBuffer>(8192);
        g_scalars_host[slot] = std::make_unique<PinnedHostBuffer>(8192);
        g_winner_host[slot] = std::make_unique<PinnedHostBuffer>(8192);
        CUDA_CHECK(cudaSetDevice(previous));
    }
    trunks.trunk = std::make_unique<rt::TextTrunk>(trunks.view_ptrs, trunks.state_ptrs,
                                                    trunks.activation_ptrs, trunks.workspace_ptrs,
                                                    execution, *comms);

    const std::vector<TokenId> prompt = make_prompt(512);
    const auto prefill_and_warm = [&]() {
        TokenId next = 0;
        for (std::size_t offset = 0; offset < prompt.size(); offset += kForwardTokens) {
            const std::size_t count = std::min<std::size_t>(kForwardTokens, prompt.size() - offset);
            next = trunks.trunk->prefill(std::span<const TokenId>{prompt.data() + offset, count});
        }
        for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) { execution.dev[rank]->synchronize(); }
        for (std::uint32_t index = 0; index < kWarmupTokens; ++index) {
            next = trunks.trunk->decode(next);
        }
        for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) { execution.dev[rank]->synchronize(); }
        return next;
    };

    TokenId token = prefill_and_warm();
    const auto warm = data.expert_cache_stats(0);

    std::array<int, kRanks> devices{};
    std::array<cudaStream_t, kRanks> streams{};
    for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) {
        devices[rank] = execution.dev[rank]->device;
        streams[rank] = execution.dev[rank]->stream;
    }
    const std::uint32_t position = trunks.states[0]->next_position();
    const TokenId capture_token = token;
    const std::size_t logits_bytes = static_cast<std::size_t>(detail::TextConfig::vocab) * 2;

    for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) {
        const auto layer_view = data.view(static_cast<int>(rank)).layers[0].post_mixer;
        if (layer_view.cache != nullptr) { layer_view.cache->set_graph_mode(true); }
    }

    std::vector<rt::TextState*> rank_states;
    for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) {
        rank_states.push_back(trunks.states[rank].get());
    }
    StateSnapshot snapshot;
    snapshot.capture_from(rank_states);
    const std::uint64_t snapshot_hash = snapshot.hash();
    StateSnapshot observed_state;
    const auto restore_state = [&]() {
        snapshot.restore_to(rank_states);
        for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) {
            trunks.activations[rank]->reset();
            trunks.workspaces[rank]->reset();
        }
        for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) { execution.dev[rank]->synchronize(); }
        observed_state.capture_from(rank_states);
        return observed_state.hash();
    };

    DecodeGraphDefinition definition;
    std::size_t node_count = 0;
    std::uint64_t capture_state_hash = 0;
    try {
        capture_state_hash = restore_state();
        for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) {
            fill_graph_inputs(rank, capture_token, position);
        }
        g_capturing = true;
        definition.capture(
            std::span<const int>(devices.data(), g_rank_count),
            std::span<cudaStream_t>(streams.data(), g_rank_count),
            [&] { trunks.trunk->decode(capture_token); });
        g_capturing = false;
        g_graph_mode = true;
        for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) { execution.dev[rank]->synchronize(); }
        node_count = definition.node_count();
    } catch (const std::exception& error) {
        g_capturing = false;
        std::cout << "{\"capture_ok\":false,\"error\":\"" << error.what() << "\"}\n";
        return;
    }

    const std::uint64_t eager_state_hash = restore_state();
    g_skip_ple_prepare = true;
    const TokenId eager_token = trunks.trunk->decode(capture_token);
    g_skip_ple_prepare = false;
    for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) { execution.dev[rank]->synchronize(); }
    std::vector<std::uint16_t> reference(logits_bytes / 2);
    std::memcpy(reference.data(), g_logits_host->data(), logits_bytes);

    DecodeGraphExecutable executable;
    executable.instantiate(definition, std::span<const int>(devices.data(), g_rank_count),
                           std::span<cudaStream_t>(streams.data(), g_rank_count));
    executable.upload();

    const auto diff_stats = [](const std::vector<std::uint16_t>& a,
                               const std::vector<std::uint16_t>& b) {
        std::size_t count = 0;
        std::uint16_t max_diff = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const std::uint16_t d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
            if (d != 0) { ++count; }
            if (d > max_diff) { max_diff = d; }
        }
        return std::pair<std::size_t, std::uint16_t>{count, max_diff};
    };

    std::vector<double> walls;
    std::vector<TokenId> replay_tokens;
    std::vector<std::vector<std::uint16_t>> observed_all;
    std::uint64_t replay_state_hash = 0;
    bool replay_states_match = true;
    for (std::uint32_t index = 0; index < kReplayTokens; ++index) {
        const std::uint64_t state_hash = restore_state();
        if (index == 0) { replay_state_hash = state_hash; }
        replay_states_match &= (state_hash == snapshot_hash);
        for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) {
            fill_graph_inputs(rank, capture_token, position);
        }
        const std::int64_t begin = now_ns();
        executable.launch();
        for (std::uint32_t rank = 0; rank < g_rank_count; ++rank) { execution.dev[rank]->synchronize(); }
        walls.push_back(static_cast<double>(now_ns() - begin) / 1.0e6);
        replay_tokens.push_back(
            static_cast<TokenId>(*static_cast<const std::int32_t*>(g_winner_host[0]->data())));
        std::vector<std::uint16_t> observed(logits_bytes / 2);
        std::memcpy(observed.data(), g_logits_host->data(), logits_bytes);
        observed_all.push_back(std::move(observed));
    }
    double sum = 0.0;
    for (double wall : walls) { sum += wall; }
    const auto eager_vs_graph = diff_stats(reference, observed_all[0]);
    const auto replay_rr = diff_stats(observed_all[0], observed_all[1]);
    bool token_match = true;
    for (TokenId result : replay_tokens) { token_match &= (result == eager_token); }
    const bool state_hashes_match =
        capture_state_hash == snapshot_hash && eager_state_hash == snapshot_hash &&
        replay_states_match && replay_state_hash == snapshot_hash;
    const double wall_ms_avg = sum / walls.size();

    std::cout << "{\"capture_ok\":true,\"tp\":" << g_rank_count
              << ",\"node_count\":" << node_count
              << ",\"capture_position\":" << position
              << ",\"capture_token\":" << capture_token
              << ",\"warm_hits\":" << warm.hits << ",\"warm_misses\":" << warm.miss_count
              << ",\"snapshot_hash\":\"" << std::hex << snapshot_hash << std::dec << "\""
              << ",\"state_hash_capture\":\"" << std::hex << capture_state_hash << std::dec << "\""
              << ",\"state_hash_eager\":\"" << std::hex << eager_state_hash << std::dec << "\""
              << ",\"state_hash_replay\":\"" << std::hex << replay_state_hash << std::dec << "\""
              << ",\"state_hashes_match\":" << (state_hashes_match ? "true" : "false")
              << ",\"eager_token\":" << eager_token
              << ",\"token_match\":" << (token_match ? "true" : "false")
              << ",\"replay_tokens\":[";
    for (std::size_t i = 0; i < replay_tokens.size(); ++i) {
        std::cout << (i ? "," : "") << replay_tokens[i];
    }
    std::cout << "],\"replay_wall_ms_avg\":" << wall_ms_avg
              << ",\"replay_tok_per_s\":" << (1000.0 / wall_ms_avg)
              << ",\"eager_vs_graph_logit_diff\":" << eager_vs_graph.first
              << ",\"eager_vs_graph_logit_maxdiff\":" << eager_vs_graph.second
              << ",\"replay_replay_logit_diff\":" << replay_rr.first
              << ",\"replay_replay_logit_maxdiff\":" << replay_rr.second
              << "}\n" << std::flush;

    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
        g_ids_host[rank].reset();
        g_positions_host[rank].reset();
        g_scalars_host[rank].reset();
        g_winner_host[rank].reset();
    }
    g_logits_host.reset();
}

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (path == nullptr || path[0] == '\0') {
        std::cerr << "flash_next_graph_profile: an artifact path is required\n";
        return 1;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < static_cast<int>(g_rank_count)) {
        std::cerr << "flash_next_graph_profile: two devices are required\n";
        return 1;
    }
    try {
        run(path);
    } catch (const std::exception& error) {
        std::cerr << "flash_next_graph_profile: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
