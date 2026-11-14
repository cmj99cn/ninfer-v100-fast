#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/moe_nvfp4.h"
#include "ninfer/ops/qsa.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"

#include <ninfer/targets/qwen4_exp/frontend.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
namespace qwen4_exp = ninfer::targets::qwen4_exp;
namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;

constexpr std::size_t kLoadedLayers = 4;
constexpr std::uint32_t kKvCapacityTokens = 64;
constexpr std::size_t kPromptTokens = 8;
constexpr std::array<std::int32_t, kPromptTokens> kPrompt{1024, 7, 98140, 33, 4097, 512, 100, 2048};
// R44: allreduce_sum is a BF16-in/BF16-out join with one rounding (include/ninfer/ops/allreduce.h),
// so each row-parallel layer adds one BF16 rounding to the residual chain. Allow 2 * sqrt(L) of
// the 2^-8 BF16 unit roundoff across the loaded layers instead of a hand-measured constant.
[[nodiscard]] double relative_l2_bound(std::size_t layers) {
    return 2.0 * std::sqrt(static_cast<double>(layers)) * std::ldexp(1.0, -8);
}
constexpr double kMinCosine = 0.9999;
constexpr std::uint64_t kAllReducePerLayer = 2;

bool g_recording = false;

struct RouteRecord {
    std::vector<std::int32_t> ids;
};

struct QsaRecord {
    std::vector<std::int32_t> counts;
    std::vector<std::int32_t> selection;
};

std::array<std::vector<RouteRecord>, 2> g_routes;
std::array<std::vector<QsaRecord>, 2> g_qsa;
std::array<std::vector<std::uint16_t>, 2> g_sample;
std::array<std::vector<std::uint16_t>, 2> g_logits;
std::array<bool, 2> g_captured{false, false};

void reset_records() {
    for (auto& entry : g_routes) { entry.clear(); }
    for (auto& entry : g_qsa) { entry.clear(); }
    for (auto& entry : g_sample) { entry.clear(); }
    for (auto& entry : g_logits) { entry.clear(); }
    g_captured = {false, false};
}

int current_device() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { throw std::runtime_error("cannot read the device"); }
    return device;
}

float bf16_to_float(std::uint16_t word) {
    const std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void capture_last(const Tensor& sample, const Tensor& logits) {
    if (!g_recording) { return; }
    const int device = current_device();
    if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("cannot synchronize the capture device");
    }
    const std::int32_t hidden = sample.ne[0];
    const std::int32_t tokens = sample.ne[1];
    const std::int32_t vocab  = logits.ne[0];
    if (logits.ne[1] != tokens) { throw std::runtime_error("the stage shapes disagree"); }
    std::vector<std::uint16_t> hidden_words(static_cast<std::size_t>(hidden));
    std::vector<std::uint16_t> logit_words(static_cast<std::size_t>(vocab));
    const auto* sample_base = static_cast<const std::byte*>(sample.data) +
                              static_cast<std::size_t>(tokens - 1) * hidden * 2;
    const auto* logit_base = static_cast<const std::byte*>(logits.data) +
                             static_cast<std::size_t>(tokens - 1) * vocab * 2;
    if (cudaMemcpy(hidden_words.data(), sample_base, hidden_words.size() * 2,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(logit_words.data(), logit_base, logit_words.size() * 2,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("cannot read the captured stage");
    }
    g_sample[static_cast<std::size_t>(device)] = std::move(hidden_words);
    g_logits[static_cast<std::size_t>(device)] = std::move(logit_words);
    g_captured[static_cast<std::size_t>(device)] = true;
}

namespace ninfer::ops {
inline void flash_next_tp2_record_qsa(const Tensor& index_query,
                                      const ConstQsaIndexCacheView& cache,
                                      const QsaBatchView& batch, std::uint32_t max_visible_tokens,
                                      WorkspaceArena& workspace, QsaBlockSelection& selected,
                                      cudaStream_t stream) {
    qsa_indexer(index_query, cache, batch, max_visible_tokens, workspace, selected, stream);
    if (!::g_recording) { return; }
    const int device = ::current_device();
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("cannot synchronize the indexer record");
    }
    const std::int32_t tokens = batch.positions.ne[0];
    const std::int32_t rows   = batch.positions.ne[2];
    std::vector<std::int32_t> counts(static_cast<std::size_t>(tokens) * rows);
    std::vector<std::int32_t> selection(static_cast<std::size_t>(512) * tokens * rows);
    if (cudaMemcpy(counts.data(), selected.counts.data, counts.size() * 4,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(selection.data(), selected.selection.data, selection.size() * 4,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("cannot read the indexer record");
    }
    QsaRecord record;
    record.counts = counts;
    for (std::int32_t t = 0; t < tokens * rows; ++t) {
        const std::int32_t count = counts[static_cast<std::size_t>(t)];
        const auto begin = selection.begin() + static_cast<std::ptrdiff_t>(512) * t;
        record.selection.insert(record.selection.end(), begin,
                                begin + static_cast<std::ptrdiff_t>(count));
    }
    ::g_qsa[static_cast<std::size_t>(device)].push_back(std::move(record));
}

inline void flash_next_tp2_record_route(const Tensor& x, const MoeNvfp4Weights& weights,
                                        Tensor& expert_ids, Tensor& coefficients,
                                        Tensor& shared_gate, cudaStream_t stream) {
    moe_nvfp4_route(x, weights, expert_ids, coefficients, shared_gate, stream);
    if (!::g_recording) { return; }
    const int device = ::current_device();
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("cannot synchronize the route record");
    }
    const std::int32_t tokens = x.ne[1];
    RouteRecord record;
    record.ids.resize(static_cast<std::size_t>(10) * tokens);
    if (cudaMemcpy(record.ids.data(), expert_ids.data, record.ids.size() * 4,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("cannot read the route record");
    }
    ::g_routes[static_cast<std::size_t>(device)].push_back(std::move(record));
}
} // namespace ninfer::ops

#define qsa_indexer flash_next_tp2_record_qsa
#define moe_nvfp4_route flash_next_tp2_record_route
#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_tp2_test_runtime
#define NINFER_QWEN4EXP_LAST_STAGE(sample, logits) ::capture_last((sample), (logits))
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"
#undef qsa_indexer
#undef moe_nvfp4_route
#undef NINFER_QWEN4EXP_LAST_STAGE

namespace rt = ninfer::targets::qwen4_exp::detail::flash_next_tp2_test_runtime;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        if (cudaGetDevice(&previous_) != cudaSuccess ||
            cudaSetDevice(device) != cudaSuccess) {
            throw std::runtime_error("cannot bind the device");
        }
    }
    ~ScopedDevice() { (void)cudaSetDevice(previous_); }
    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = 0;
};

bool needs_prefix(const flash::BoundParameter& parameter) {
    if (parameter.role == roles::Role::PleTable) { return false; }
    if (!roles::is_layer_scoped(parameter.role)) { return true; }
    if (parameter.layer < kLoadedLayers) { return true; }
    return parameter.layer == kLoadedLayers &&
           parameter.role == roles::Role::AttentionHcNorm;
}

void place_prefix(const flash::BindingPlan& plan, artifact::Binder& binder) {
    std::size_t object_slots = 0;
    for (const auto& object : plan.objects) {
        object_slots = std::max(object_slots, object.handle.index + 1);
    }
    std::vector<bool> needed(object_slots, false);
    for (const auto* parameter : plan.parameters()) {
        if (!needs_prefix(*parameter)) { continue; }
        for (const auto& part : parameter->parts) {
            needed[plan.objects.at(part.object).handle.index] = true;
        }
        if (parameter->library.has_value()) {
            needed[parameter->library->handle.index] = true;
        }
    }
    for (const auto& object : plan.objects) {
        if (needed[object.handle.index] &&
            std::holds_alternative<artifact::TensorDescriptor>(binder.descriptor(object.handle))) {
            binder.materialize_on_device(object.handle);
        } else {
            binder.validate_only(object.handle);
        }
    }
}

std::span<const TokenId> prompt_span() {
    return std::span<const TokenId>{kPrompt.data(), kPrompt.size()};
}

struct RunResult {
    TokenId token = 0;
    std::uint64_t allreduce_calls = 0;
};

RunResult run_tp1(const char* path) {
    DeviceContext device(0);
    artifact::Reader reader(path);
    artifact::Binder binder(reader, 1);
    const flash::BindingPlan plan = flash::bind_artifact(binder, flash::WeightsProfile::Nvfp4, 1);
    place_prefix(plan, binder);
    binder.validate_unconsumed_matching();
    const artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, binder.finish(), device);
    const flash::LoadedModelData data(plan, materialized, kLoadedLayers, 1,
                                      std::filesystem::path(path));

    const auto recipe = rt::text_workspace_recipe(kPromptTokens, 1);
    DeviceArena activations(rt::text_activation_capacity_bytes(kPromptTokens));
    DeviceArena workspace(recipe.lane_capacity_bytes);
    rt::TextState state(device, data.view(0), kKvCapacityTokens, kPromptTokens);
    rt::TextTrunk trunk(data.view(0), state, activations, workspace, device.stream);

    reset_records();
    g_recording = true;
    RunResult result;
    result.token = trunk.prefill(prompt_span());
    g_recording = false;
    return result;
}

struct Tp2Model {
    ExecutionContext execution;
    artifact::Reader reader;
    artifact::Binder binder;
    flash::BindingPlan plan;
    artifact::MaterializedArtifact materialized;
    std::unique_ptr<flash::LoadedModelData> data;
    std::unique_ptr<ops::CollectiveComms> comms;

    explicit Tp2Model(const char* path)
        : execution({0, 1}), reader(path), binder(reader, 2),
          plan(flash::bind_artifact(binder, flash::WeightsProfile::Nvfp4, 2)) {
        (void)qwen4_exp::bind_frontend_resources(binder);
        place_prefix(plan, binder);
        binder.validate_unconsumed_matching();
        materialized = artifact::materialize(reader, binder.finish(), execution);
        data = std::make_unique<flash::LoadedModelData>(plan, materialized, kLoadedLayers, 2,
                                                        std::filesystem::path(path));
        comms = std::make_unique<ops::CollectiveComms>(execution);
    }

    RunResult run() {
        std::array<std::unique_ptr<rt::TextState>, kMaximumDevices> states;
        std::array<std::unique_ptr<DeviceArena>, kMaximumDevices> activations;
        std::array<std::unique_ptr<DeviceArena>, kMaximumDevices> workspaces;
        std::array<rt::TextState*, kMaximumDevices> state_ptrs{};
        std::array<const rt::Variant::ModelView*, kMaximumDevices> view_ptrs{};
        std::array<WorkspaceArena*, kMaximumDevices> activation_ptrs{};
        std::array<WorkspaceArena*, kMaximumDevices> workspace_ptrs{};

        const auto recipe = rt::text_workspace_recipe(kPromptTokens, 1);
        for (int rank = 0; rank < 2; ++rank) {
            const ScopedDevice scope(execution.dev[rank]->device);
            const auto slot = static_cast<std::size_t>(rank);
            states[slot] = std::make_unique<rt::TextState>(*execution.dev[slot], data->view(rank),
                                                           kKvCapacityTokens, kPromptTokens, 2,
                                                           rank);
            activations[slot] =
                std::make_unique<DeviceArena>(rt::text_activation_capacity_bytes(kPromptTokens));
            workspaces[slot] = std::make_unique<DeviceArena>(recipe.lane_capacity_bytes);
            state_ptrs[slot] = states[slot].get();
            view_ptrs[slot] = &data->view(rank);
            activation_ptrs[slot] = activations[slot].get();
            workspace_ptrs[slot] = workspaces[slot].get();
        }
        rt::TextTrunk trunk(view_ptrs, state_ptrs, activation_ptrs, workspace_ptrs, execution,
                            *comms);

        comms->reset_stats();
        reset_records();
        g_recording = true;
        RunResult result;
        result.token = trunk.prefill(prompt_span());
        g_recording = false;
        result.allreduce_calls = comms->stats().allreduce_calls;
        for (int rank = 0; rank < 2; ++rank) { execution.dev[rank]->synchronize(); }
        return result;
    }
};

double relative_l2(const std::vector<std::uint16_t>& reference,
                   const std::vector<std::uint16_t>& candidate) {
    require(reference.size() == candidate.size(), "the compared stages differ in size");
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double expected = bf16_to_float(reference[index]);
        const double actual   = bf16_to_float(candidate[index]);
        const double delta    = actual - expected;
        numerator += delta * delta;
        denominator += expected * expected;
    }
    if (denominator == 0.0) { return numerator == 0.0 ? 0.0 : 1.0; }
    return std::sqrt(numerator / denominator);
}

double cosine_of(const std::vector<std::uint16_t>& reference,
                 const std::vector<std::uint16_t>& candidate) {
    require(reference.size() == candidate.size(), "the compared stages differ in size");
    double dot = 0.0;
    double reference_norm = 0.0;
    double candidate_norm = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double expected = bf16_to_float(reference[index]);
        const double actual   = bf16_to_float(candidate[index]);
        dot += expected * actual;
        reference_norm += expected * expected;
        candidate_norm += actual * actual;
    }
    if (reference_norm == 0.0 || candidate_norm == 0.0) {
        return reference_norm == candidate_norm ? 1.0 : 0.0;
    }
    return dot / (std::sqrt(reference_norm) * std::sqrt(candidate_norm));
}

TokenId argmax_of(const std::vector<std::uint16_t>& words) {
    std::size_t best = 0;
    float best_value = bf16_to_float(words[0]);
    for (std::size_t index = 1; index < words.size(); ++index) {
        const float value = bf16_to_float(words[index]);
        if (value > best_value) {
            best_value = value;
            best = index;
        }
    }
    return static_cast<TokenId>(best);
}

bool bitwise_equal(const std::vector<std::uint16_t>& left, const std::vector<std::uint16_t>& right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "SKIP: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
        return 77;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
        std::cout << "SKIP: two CUDA devices are required\n";
        return 77;
    }

    try {
        const RunResult single = run_tp1(path);
        require(g_captured[0], "the tp1 run captured no stage");
        const auto single_sample = g_sample;
        const auto single_logits = g_logits;

        Tp2Model model(path);
        const RunResult first = model.run();
        require(g_captured[0] && g_captured[1], "the tp2 run captured no stage on one rank");
        const auto first_sample = g_sample;
        const auto first_logits = g_logits;
        const auto first_routes = g_routes;
        const auto first_qsa = g_qsa;

        const RunResult second = model.run();
        require(g_captured[0] && g_captured[1], "the second tp2 run captured no stage on one rank");

        require(first.token == second.token, "tp2 is not deterministic across repeated runs");
        require(bitwise_equal(first_logits[0], g_logits[0]) &&
                    bitwise_equal(first_logits[0], g_logits[1]),
                "the repeated tp2 logits are not bitwise identical");
        require(bitwise_equal(first_sample[0], g_sample[0]) &&
                    bitwise_equal(first_sample[0], g_sample[1]),
                "the repeated tp2 sample hidden is not bitwise identical");

        require(bitwise_equal(first_sample[0], first_sample[1]),
                "the two tp2 ranks do not hold the same sample hidden");
        require(bitwise_equal(first_logits[0], first_logits[1]),
                "the two tp2 ranks do not hold the same logits");

        require(single.token == first.token,
                "the tp1 and tp2 greedy tokens differ (" + std::to_string(single.token) + " vs " +
                    std::to_string(first.token) + ")");
        require(argmax_of(first_logits[0]) == first.token,
                "the tp2 greedy token does not match its own logits");
        require(argmax_of(g_logits[0]) == second.token,
                "the repeated tp2 greedy token does not match its own logits");

        const double hidden_error = relative_l2(single_sample[0], first_sample[0]);
        const double logits_error = relative_l2(single_logits[0], first_logits[0]);
        const double hidden_cosine = cosine_of(single_sample[0], first_sample[0]);
        const double logits_cosine = cosine_of(single_logits[0], first_logits[0]);
        const double bound = relative_l2_bound(kLoadedLayers);
        require(hidden_error <= bound,
                "the tp2 sample hidden differs from tp1 by " + std::to_string(hidden_error) +
                    "; the BF16 join bound is " + std::to_string(bound));
        require(logits_error <= bound,
                "the tp2 logits differ from tp1 by " + std::to_string(logits_error) +
                    "; the BF16 join bound is " + std::to_string(bound));
        require(hidden_cosine >= kMinCosine,
                "the tp2 sample hidden cosine is " + std::to_string(hidden_cosine));
        require(logits_cosine >= kMinCosine,
                "the tp2 logits cosine is " + std::to_string(logits_cosine));

        require(first.allreduce_calls == kAllReducePerLayer * kLoadedLayers,
                "the tp2 trunk issued " + std::to_string(first.allreduce_calls) +
                    " all-reduces; two per layer are expected");
        require(second.allreduce_calls == first.allreduce_calls,
                "the repeated tp2 all-reduce count differs");

        require(first_routes[0].size() == kLoadedLayers && first_routes[1].size() == kLoadedLayers,
                "the router records do not cover every layer on both ranks");
        for (std::size_t layer = 0; layer < kLoadedLayers; ++layer) {
            require(first_routes[0][layer].ids == first_routes[1][layer].ids,
                    "the two ranks selected different routed experts at layer " +
                        std::to_string(layer));
        }
        require(first_qsa[0].size() == first_qsa[1].size() && !first_qsa[0].empty(),
                "the QSA records do not cover both ranks");
        for (std::size_t entry = 0; entry < first_qsa[0].size(); ++entry) {
            require(first_qsa[0][entry].counts == first_qsa[1][entry].counts &&
                        first_qsa[0][entry].selection == first_qsa[1][entry].selection,
                    "the two ranks selected different QSA blocks");
        }

        std::cout << "OK flash-next tp2 equivalence: token " << first.token << " hidden_l2 "
                  << hidden_error << " logits_l2 " << logits_error << " hidden_cosine "
                  << hidden_cosine << " logits_cosine " << logits_cosine << " bound " << bound
                  << " allreduce " << first.allreduce_calls << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
