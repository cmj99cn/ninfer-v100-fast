#pragma once

#include "artifact/materializer.h"
#include "core/cyclic_kv_cache.h"
#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "targets/qwen4_exp/impl/runtime/instance.h"
#include "targets/qwen4_exp/impl/runtime/layouts.h"
#include "core/arena.h"
#include "targets/qwen4_exp/impl/runtime/program.h"
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"
#include "targets/qwen3_8_flash_next/impl/dflash2_draft.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <sys/resource.h>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

inline constexpr std::uint32_t kGraphCaptureWarmSteps = 8;

namespace {

inline bool dflash2_phase_diag_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DFLASH2_PHASE_DIAG");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

inline void dflash2_event_create(cudaEvent_t& event) {
    if (event == nullptr) { CUDA_CHECK(cudaEventCreate(&event)); }
}

inline void dflash2_event_destroy(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        (void)cudaEventDestroy(event);
        event = nullptr;
    }
}

inline void dflash2_event_record(cudaEvent_t event, cudaStream_t stream) {
    if (event != nullptr) { CUDA_CHECK(cudaEventRecord(event, stream)); }
}

inline double dflash2_event_ms(cudaEvent_t begin, cudaEvent_t end) {
    if (begin == nullptr || end == nullptr) { return 0.0; }
    float ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
    return static_cast<double>(ms);
}

inline void dflash2_phase_print(const Dflash2PhaseDiag& diag) {
    if (diag.rounds == 0) { return; }
    const double rounds = static_cast<double>(diag.rounds);
    std::fprintf(stderr,
                 "dflash2_phase avg_ms: propose=%.4f propose_host=%.4f verify=%.4f "
                 "commit=%.4f materialize=%.4f total=%.4f rounds=%llu tokens=%llu "
                 "tokens_per_round=%.3f\n",
                 diag.propose_ms / rounds, diag.propose_host_ms / rounds,
                 diag.verify_ms / rounds, diag.commit_ms / rounds,
                 diag.materialize_ms / rounds, diag.total_ms / rounds,
                 static_cast<unsigned long long>(diag.rounds),
                 static_cast<unsigned long long>(diag.tokens),
                 static_cast<double>(diag.tokens) / rounds);
    std::fflush(stderr);
}

inline void dflash2_phase_read_pending(Dflash2PhaseDiag& diag, std::uint64_t print_every) {
    if (!diag.pending) { return; }
    if (diag.pending_drafts) {
        diag.verify_ms += dflash2_event_ms(diag.verify_begin, diag.verify_end);
        diag.commit_ms += dflash2_event_ms(diag.commit_begin, diag.commit_end);
    }
    diag.materialize_ms += dflash2_event_ms(diag.materialize_begin, diag.materialize_end);
    diag.total_ms += dflash2_event_ms(diag.round_begin, diag.materialize_end);
    diag.tokens += diag.pending_tokens;
    diag.rounds += 1;
    diag.pending        = false;
    diag.pending_drafts = false;
    diag.pending_tokens = 0;
    if (print_every != 0 && diag.rounds % print_every == 0) { dflash2_phase_print(diag); }
}

} // namespace

ProgramImplCore::~ProgramImplCore() noexcept {
    if (dflash2_phase_diag_.enabled && dflash2_phase_diag_.rounds > 0) {
        dflash2_phase_print(dflash2_phase_diag_);
    }
    auto& diag = dflash2_phase_diag_;
    dflash2_event_destroy(diag.round_begin);
    dflash2_event_destroy(diag.propose_begin);
    dflash2_event_destroy(diag.propose_end);
    dflash2_event_destroy(diag.verify_begin);
    dflash2_event_destroy(diag.verify_end);
    dflash2_event_destroy(diag.commit_begin);
    dflash2_event_destroy(diag.commit_end);
    dflash2_event_destroy(diag.materialize_begin);
    dflash2_event_destroy(diag.materialize_end);
}

namespace {

class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        if (previous_ != device) { CUDA_CHECK(cudaSetDevice(device)); }
    }
    ~ScopedDevice() { (void)cudaSetDevice(previous_); }
    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = 0;
};

} // namespace

ProgramImplCore::ProgramImplCore(const std::array<const ModelView*, kMaximumDevices>& models,
                                 WeightsProfile weights_profile, std::unique_ptr<SequencePlanImpl> plan,
                                 ExecutionContext& execution, const EngineOptions& options)
    : weights_profile_(weights_profile), models_(models), execution_(execution), options_(options),
      plan_(std::move(plan)) {
    if (plan_ == nullptr) { throw std::logic_error("qwen4_exp program received an empty sequence plan"); }
    if (execution_.tp != plan_->tp) {
        throw std::invalid_argument("qwen4_exp program execution width does not match the sequence plan");
    }
    if (execution_.tp < 1 || execution_.tp > static_cast<int>(kMaximumDevices)) {
        throw std::invalid_argument("qwen4_exp program execution width is invalid");
    }
    for (int rank = 0; rank < execution_.tp; ++rank) {
        if (models_[static_cast<std::size_t>(rank)] == nullptr) {
            throw std::invalid_argument("qwen4_exp program model view is missing for an active rank");
        }
    }

    const bool mtp_enabled = options_.speculative.backend == SpeculativeBackend::Mtp &&
                             options_.speculative.draft_tokens > 0;
    namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
    const bool dflash2_enabled = options_.speculative.backend == SpeculativeBackend::DFlash2 &&
                                 options_.speculative.draft_tokens > 0;
    std::uint32_t replay_width = 1;
    if (mtp_enabled) {
        mtp_draft_window_ = std::min<std::uint32_t>(
            options_.speculative.draft_tokens, flash::MtpConfig::maximum_draft_window);
        replay_width = mtp_draft_window_ + 1U;
    }
    if (dflash2_enabled) {
        dflash2_window_ = std::min<std::uint32_t>(
            options_.speculative.draft_tokens, flash::DFlash2Geometry::maximum_draft_window);
        replay_width = std::max(replay_width, dflash2_window_ + 1U);
    }

    for (int rank = 0; rank < execution_.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const ScopedDevice scope(execution_.dev[slot]->device);
        states_[slot] = std::make_unique<TextState>(
            *execution_.dev[slot], *models_[slot],
            std::max<std::uint32_t>(plan_->kv_capacity, plan_->capacity),
            std::max<std::uint32_t>(plan_->prefill_chunk, 1), execution_.tp, rank, replay_width);
        activation_arenas_[slot] =
            std::make_unique<WorkspaceArena>(std::max<std::size_t>(plan_->activation_capacity_bytes, 1));
        work_arenas_[slot] =
            std::make_unique<WorkspaceArena>(std::max<std::size_t>(plan_->workspace_capacity_bytes, 1));
        view_ptrs_[slot]       = models_[slot];
        state_ptrs_[slot]      = states_[slot].get();
        activation_ptrs_[slot] = activation_arenas_[slot].get();
        workspace_ptrs_[slot]  = work_arenas_[slot].get();
    }
    comms_ = std::make_unique<ops::CollectiveComms>(execution_);
    trunk_ = std::make_unique<TextTrunk>(view_ptrs_, state_ptrs_, activation_ptrs_,
                                         workspace_ptrs_, execution_, *comms_);
    if (mtp_enabled) {
        const std::size_t sink_bytes = static_cast<std::size_t>(plan_->prefill_chunk) *
                                       static_cast<std::size_t>(TextConfig::residual_width) * 2;
        for (int rank = 0; rank < execution_.tp; ++rank) {
            residual_sinks_[static_cast<std::size_t>(rank)] = std::make_unique<DeviceBuffer>(sink_bytes);
            trunk_->set_residual_sink(rank, residual_sinks_[static_cast<std::size_t>(rank)]->p,
                                      sink_bytes);
        }
        mtp_draft_ = std::make_unique<ninfer::targets::qwen3_8_flash_next::detail::MtpDraft>(
            options_.artifact_path, execution_, *comms_);
        mtp_stats_.backend      = SpeculativeBackend::Mtp;
        mtp_stats_.enabled      = true;
        mtp_stats_.draft_window = mtp_draft_window_;
    } else if (options_.use_cuda_graph && !dflash2_enabled) {
        graph_ = std::make_unique<GraphDecodeController>(
            execution_, state_ptrs_, activation_ptrs_, workspace_ptrs_, execution_.tp);
    }
    if (dflash2_enabled) { setup_dflash2(); }
    if (dflash2_enabled && dflash2_phase_diag_enabled()) {
        const ScopedDevice diag_scope(execution_.dev[0]->device);
        auto& diag       = dflash2_phase_diag_;
        diag.enabled     = true;
        dflash2_event_create(diag.round_begin);
        dflash2_event_create(diag.propose_begin);
        dflash2_event_create(diag.propose_end);
        dflash2_event_create(diag.verify_begin);
        dflash2_event_create(diag.verify_end);
        dflash2_event_create(diag.commit_begin);
        dflash2_event_create(diag.commit_end);
        dflash2_event_create(diag.materialize_begin);
        dflash2_event_create(diag.materialize_end);
    }
    CUDA_CHECK(cudaSetDevice(execution_.dev[0]->device));
}

void ProgramImplCore::setup_dflash2() {
    namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
    if (options_.speculative_draft_artifact.empty()) {
        throw std::invalid_argument("--spec dflash2 requires --draft-model <path>");
    }
    if (execution_.tp > 2) {
        throw std::invalid_argument("DFlash2 supports tensor-parallel width one or two only");
    }
    constexpr std::int32_t kHidden = flash::DFlash2Geometry::hidden;
    constexpr std::int32_t kLayers = flash::DFlash2Geometry::layers;

    const ScopedDevice scope(execution_.dev[0]->device);
    dflash2_draft_ = std::make_unique<flash::DFlash2Draft>(
        options_.speculative_draft_artifact, flash::DFlash2Placement::Device);
    dflash2_materialized_ = std::make_unique<artifact::MaterializedArtifact>(
        artifact::materialize(dflash2_draft_->reader(), dflash2_draft_->plan(),
                              *execution_.dev[0]));
    dflash2_operands_ = std::make_unique<flash::DFlash2Operands>(
        flash::bind_dflash2_operands(*dflash2_draft_, *dflash2_materialized_, 0));
    // The target's token rows are replicated on every rank; the target's head is sharded at TP2
    // and reached through the collective in the head callback below.
    dflash2_operands_->embedding = models_[0]->embedding;

    // The companion's context ring: one cyclic layer per drafter layer, one lane per sequence.
    LayoutBuilder builder;
    const CyclicKVCacheLayout ring_layout = plan_cyclic_kv_cache(
        builder, static_cast<std::uint32_t>(kLayers), flash::DFlash2Geometry::window,
        flash::DFlash2Geometry::kv_heads, flash::DFlash2Geometry::head_dim,
        flash::DFlash2Geometry::lane_capacity);
    const std::size_t ring_bytes = builder.finish(256, "DFlash2 ring");
    dflash2_ring_backing_ = DeviceBuffer(ring_bytes);
    dflash2_ring_ = std::make_unique<CyclicKVCache>(
        DeviceSpan{dflash2_ring_backing_.p, dflash2_ring_backing_.bytes}, ring_layout);
    std::vector<CyclicKVCacheLayerView> ring;
    ring.reserve(static_cast<std::size_t>(kLayers));
    for (std::int32_t layer = 0; layer < kLayers; ++layer) {
        ring.push_back(dflash2_ring_->layer_view(static_cast<std::uint32_t>(layer)));
    }

    // Feature capture: the tap writes the mean of the four hyper-connection streams of each
    // tapped target layer; only the most recent ring window of one forward is kept.
    dflash2_tap_layers_.clear();
    for (const std::uint32_t layer : dflash2_draft_->target_layer_ids()) {
        dflash2_tap_layers_.push_back(static_cast<std::int32_t>(layer));
    }
    dflash2_feature_width_ = std::max<std::uint32_t>(
        8U, std::min<std::uint32_t>(plan_->prefill_chunk, flash::DFlash2Geometry::window));
    const std::size_t feature_elements = static_cast<std::size_t>(dflash2_tap_layers_.size()) *
                                         static_cast<std::size_t>(kHidden) *
                                         static_cast<std::size_t>(dflash2_feature_width_);
    dflash2_features_     = DeviceBuffer(feature_elements * sizeof(std::uint16_t));
    dflash2_positions_    = DeviceBuffer(static_cast<std::size_t>(dflash2_feature_width_) * 4);
    dflash2_counts_       = DeviceBuffer(4);
    dflash2_lanes_        = DeviceBuffer(4);
    dflash2_anchors_      = DeviceBuffer(4);
    dflash2_frontiers_    = DeviceBuffer(4);
    dflash2_valid_        = DeviceBuffer(4);
    dflash2_ids_          = DeviceBuffer(8 * 4);
    dflash2_block_positions_ = DeviceBuffer(8 * 4);
    dflash2_drafts_       = DeviceBuffer(7 * 4);
    dflash2_candidates_   = DeviceBuffer(16 * 7 * 4);
    dflash2_proposal_q_   = DeviceBuffer(16 * 7 * 4);
    dflash2_confidence_   = DeviceBuffer(7 * 4);
    dflash2_sampling_     = DeviceBuffer(sizeof(ops::SamplingConfig));

    // The target's head. At TP2 each rank projects its own vocabulary shard from a copy of the
    // block's final hidden and rank 0 gathers the full logits for the selector.
    const Weight& local_head = models_[0]->output_head;
    // The head callback's own planes: the rank-0 linear workspace and, at TP2, its vocabulary
    // shard output. The full-vocabulary logits plane is already counted by the drafter's tail.
    const std::size_t head_planes =
        static_cast<std::size_t>(TextConfig::vocab / execution_.tp) * 8 * sizeof(std::uint16_t);
    const std::size_t head_workspace =
        ops::linear_workspace_capacity_bytes(local_head.qtype, local_head.n, local_head.k,
                                             ops::LinearPolicy::A16Only, 1, 8) +
        head_planes;
    const std::size_t materialize_workspace = flash::DFlash2Drafter::materialize_workspace_bytes(
        1, static_cast<std::int32_t>(dflash2_feature_width_));
    const std::size_t propose_workspace = flash::DFlash2Drafter::propose_workspace_bytes(
        1, static_cast<std::int32_t>(flash::DFlash2Geometry::maximum_draft_window + 1),
        plan_->capacity, head_workspace);
    dflash2_work_ = std::make_unique<WorkspaceArena>(
        std::max(materialize_workspace, propose_workspace) + (1U << 20));
    if (execution_.tp > 1) {
        const Weight& peer_head = models_[1]->output_head;
        const std::size_t peer_head_workspace = ops::linear_workspace_capacity_bytes(
            peer_head.qtype, peer_head.n, peer_head.k, ops::LinearPolicy::A16Only, 1, 8);
        const std::size_t peer_planes =
            (static_cast<std::size_t>(kHidden) + static_cast<std::size_t>(peer_head.n) +
             static_cast<std::size_t>(TextConfig::vocab)) * 8 * 2;
        const ScopedDevice peer_device(execution_.dev[1]->device);
        dflash2_peer_work_ =
            std::make_unique<WorkspaceArena>(peer_head_workspace + peer_planes + (1U << 16));
    }

    flash::DFlash2Head head = [this](const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                                     cudaStream_t stream) {
        const std::int32_t columns = hidden.ne[1];
        if (execution_.tp == 1) {
            ops::linear(hidden, models_[0]->output_head, logits, ops::LinearPolicy::A16Only, work,
                        stream);
            return;
        }
        // Each rank projects its own vocabulary shard from a copy of the block's hidden; rank 0
        // then gathers the shards for the selector. Every join is stream-ordered on the rank
        // streams, so no host synchronization is required.
        const std::int32_t per_rank = TextConfig::vocab / execution_.tp;
        std::array<Tensor, kMaximumDevices> part{};
        part[0] = work.alloc(DType::BF16, {per_rank, columns});
        ops::linear(hidden, models_[0]->output_head, part[0], ops::LinearPolicy::A16Only, work,
                    stream);
        auto peer_scope            = dflash2_peer_work_->scope();
        Tensor peer_hidden         = dflash2_peer_work_->alloc(DType::BF16, {hidden.ne[0], columns});
        part[1]                    = dflash2_peer_work_->alloc(DType::BF16, {per_rank, columns});
        const std::int32_t vocab   = static_cast<std::int32_t>(TextConfig::vocab);
        Tensor peer_destination    = dflash2_peer_work_->alloc(DType::BF16, {1, vocab});
        {
            std::array<Tensor, kMaximumDevices> destination{};
            destination[1] = peer_hidden;
            ops::broadcast_rank0(hidden, destination, execution_, *comms_);
        }
        {
            const ScopedDevice peer_device(execution_.dev[1]->device);
            ops::linear(peer_hidden, models_[1]->output_head, part[1], ops::LinearPolicy::A16Only,
                        *dflash2_peer_work_, states_[1]->stream());
        }
        // The proposal logits are consumed on rank 0 only. Gather one vocabulary column at a time
        // through the established row gather (rank 0's head does the same at TP2).
        const std::size_t element = sizeof(std::uint16_t);
        for (std::int32_t column = 0; column < columns; ++column) {
            std::array<Tensor, kMaximumDevices> destination{};
            std::array<Tensor, kMaximumDevices> source{};
            destination[0] =
                Tensor(static_cast<std::byte*>(logits.data) + static_cast<std::size_t>(column) * vocab * element,
                       DType::BF16, {1, vocab});
            destination[1] = peer_destination;
            source[0] =
                Tensor(static_cast<std::byte*>(part[0].data) + static_cast<std::size_t>(column) * per_rank * element,
                       DType::BF16, {1, per_rank});
            source[1] =
                Tensor(static_cast<std::byte*>(part[1].data) + static_cast<std::size_t>(column) * per_rank * element,
                       DType::BF16, {1, per_rank});
            ops::allgather_rows(destination, source, execution_, *comms_);
        }
    };

    dflash2_drafter_ = std::make_unique<flash::DFlash2Drafter>(
        *dflash2_operands_, static_cast<std::int32_t>(dflash2_draft_->vocab()),
        static_cast<std::int32_t>(dflash2_draft_->vocab()), std::move(ring), plan_->capacity,
        std::move(head));

    ops::SamplingConfig sampling{};
    sampling.temperature = 0.0F;
    dflash2_sampling_.copy_from_host(&sampling, sizeof(sampling));
    const std::int32_t lane = 0;
    const std::int32_t count = 0;
    dflash2_lanes_.copy_from_host(&lane, sizeof(lane));
    dflash2_counts_.copy_from_host(&count, sizeof(count));

    trunk_->set_feature_tap(dflash2_tap_layers_, dflash2_features_.p, dflash2_features_.bytes,
                            flash::DFlash2Geometry::window);
    dflash2_stats_.backend      = SpeculativeBackend::DFlash2;
    dflash2_stats_.enabled      = true;
    dflash2_stats_.draft_window = dflash2_window_;
}

runtime::RequestPlanSummary ProgramImplCore::make_summary(std::uint32_t prompt_tokens,
                                                           std::uint32_t requested_output_tokens,
                                                           std::size_t transient_bytes) const {
    runtime::RequestPlanSummary summary;
    summary.prompt_tokens              = prompt_tokens;
    summary.reusable_prompt_tokens     = 0;
    summary.requested_output_tokens    = requested_output_tokens;
    summary.effective_output_tokens    = requested_output_tokens;
    summary.effective_limit_reason     = FinishReason::OutputLimit;
    summary.transient_bytes            = transient_bytes;
    summary.transient_alignment        = 1;
    summary.admission.active_lanes     = 1;
    summary.admission.main_kv_pages    = page_groups_for(prompt_tokens);
    const std::uint64_t prefill_units =
        1ULL + (static_cast<std::uint64_t>(prompt_tokens) - 1ULL) / plan_->prefill_chunk;
    const std::uint64_t decode_units =
        requested_output_tokens == 0 ? 0ULL : requested_output_tokens - 1ULL;
    summary.service_work_quanta = prefill_units + decode_units;
    return summary;
}

std::uint32_t ProgramImplCore::page_groups_for(std::uint32_t tokens) const noexcept {
    return (tokens + Variant::kv_page_tokens - 1) / Variant::kv_page_tokens;
}

void ProgramImplCore::clear_lane(std::uint32_t lane) noexcept {
    const GenerationTimings timings = lanes_[lane].timings;
    lanes_[lane]                    = LaneState{};
    lanes_[lane].timings            = timings;
}

void ProgramImplCore::prefill_chunk(LaneState& state, std::uint32_t begin, std::uint32_t count) {
    const std::span<const TokenId> chunk{state.context.prompt_tokens.data() + begin, count};
    const std::uint32_t base = states_[0]->next_position();
    last_stage_capture() = &logits_capture_;
    const TokenId sampled = trunk_->prefill(chunk);
    if (begin + count == state.context.prompt_tokens.size()) {
        state.context.next_input = sampled;
    }
    if (dflash2_drafter_ != nullptr) {
        const std::uint32_t captured = std::min<std::uint32_t>(
            count, ninfer::targets::qwen3_8_flash_next::detail::DFlash2Geometry::window);
        dflash2_capture_committed(base + count - captured, static_cast<std::int32_t>(captured));
    }
}

RequestBasePlan ProgramImplCore::plan_request_base(const PreparedPrompt& prompt,
                                                    const runtime::ResolvedExecutionOptions& options) {
    const PreparedPromptData& data = PreparedPromptAccess::view(prompt);
    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(data.token_ids.size());
    if (prompt_tokens == 0) {
        throw std::invalid_argument("qwen4_exp request has no prompt tokens");
    }
    if (prompt_tokens > plan_->capacity) {
        throw std::invalid_argument("qwen4_exp request prompt exceeds the finalized capacity");
    }
    for (const TokenId token : data.token_ids) {
        if (token >= static_cast<TokenId>(TextConfig::vocab)) {
            throw std::invalid_argument("qwen4_exp request contains a token outside the vocabulary");
        }
    }
    if (prompt_tokens + options.requested_output_tokens > plan_->capacity) {
        throw std::invalid_argument("qwen4_exp request reservation exceeds the finalized capacity");
    }

    return RequestBasePlan(std::make_unique<RequestBasePlanImpl>(RequestBasePlanImpl{
        .summary  = make_summary(prompt_tokens, options.requested_output_tokens,
                                 plan_->request_transient_capacity_bytes),
        .sampling = translate_sampling(options.sampling),
    }));
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPrompt& prompt,
                                                    const RequestBasePlan& base) {
    if (lane >= plan_->max_concurrency) {
        throw std::invalid_argument("qwen4_exp request lane is outside the finalized concurrency");
    }
    const PreparedPromptData& data = PreparedPromptAccess::view(prompt);
    if (data.token_ids.size() != base.summary().prompt_tokens) {
        throw std::logic_error("qwen4_exp lane plan does not match its base plan");
    }

    return RequestPlan(std::make_unique<RequestPlanImpl>(
        RequestPlanImpl{.summary = base.summary(), .sampling = base.impl_->sampling}));
}

bool ProgramImplCore::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= plan_->max_concurrency) { return false; }
    if (lanes_[lane].active) { return false; }
    const runtime::RequestPlanSummary& summary = plan.summary();
    if (summary.admission.active_lanes > plan_->max_concurrency) { return false; }
    if (summary.admission.main_kv_pages > plan_->main_page_groups) { return false; }
    return true;
}

bool ProgramImplCore::can_admit_lane_after_retained_eviction(
    std::uint32_t lane, const RequestPlan& plan) const noexcept {
    return can_admit_lane(lane, plan);
}

runtime::AdmissionResources ProgramImplCore::admission_capacity() const noexcept {
    return runtime::AdmissionResources{
        .active_lanes  = plan_->max_concurrency,
        .main_kv_pages = plan_->main_page_groups,
    };
}

runtime::AdmissionResources ProgramImplCore::admission_cache_residue() const noexcept {
    return {};
}

runtime::PrefillStepResult ProgramImplCore::start_prefill_lane(std::uint32_t lane,
                                                                PreparedPrompt&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient) {
    if (lane >= plan_->max_concurrency) {
        throw std::invalid_argument("qwen4_exp prefill lane is outside the finalized concurrency");
    }
    if (lanes_[lane].active) {
        throw std::logic_error("qwen4_exp prefill lane is already active");
    }

    PreparedPromptData data = PreparedPromptAccess::take(std::move(prompt));
    if (data.token_ids.empty()) {
        throw std::logic_error("qwen4_exp prefill received no prompt tokens");
    }

    for (const LaneState& other : lanes_) {
        if (&other != &lanes_[lane] && other.active) {
            throw std::logic_error("qwen4_exp text state serves one active lane");
        }
    }

    for (int rank = 0; rank < execution_.tp; ++rank) {
        states_[static_cast<std::size_t>(rank)]->reset();
    }
    if (graph_ != nullptr && !graph_->ready() &&
        graph_warm_steps_ >= kGraphCaptureWarmSteps) {
        for (int rank = 0; rank < execution_.tp; ++rank) {
            const auto& model = *models_[static_cast<std::size_t>(rank)];
            for (std::size_t layer = 0; layer < model.loaded_layers; ++layer) {
                const auto& moe = model.layers[layer].post_mixer;
                Variant::freeze_expert_residency(moe);
                Variant::set_expert_graph_mode(moe, true);
            }
        }
        const TokenId capture_token = static_cast<TokenId>(data.token_ids.front());
        try {
            graph_->capture(capture_token, [&] { trunk_->decode(capture_token); });
        } catch (const std::exception&) {
            graph_.reset();
        }
    }
    std::vector<TokenId>& tokens = data.token_ids;
    const std::uint32_t chunk = std::min(plan_->prefill_chunk,
                                         static_cast<std::uint32_t>(tokens.size()));
    const auto prefill_start = std::chrono::steady_clock::now();
    const std::uint32_t prefill_base = states_[0]->next_position();
    last_stage_capture() = &logits_capture_;
    const TokenId sampled = trunk_->prefill(std::span<const TokenId>{tokens.data(), chunk});
    const double prefill_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_start).count();
    if (dflash2_drafter_ != nullptr) {
        const std::uint32_t captured = std::min<std::uint32_t>(
            chunk, ninfer::targets::qwen3_8_flash_next::detail::DFlash2Geometry::window);
        dflash2_capture_committed(prefill_base + chunk - captured,
                                  static_cast<std::int32_t>(captured));
    }

    LaneState& state = lanes_[lane];
    state.timings    = GenerationTimings{};
    state.active     = true;
    state.transient  = std::move(transient);
    state.sampling   = plan.impl_->sampling;
    state.context.prompt_tokens = std::move(tokens);
    state.context.processed_prompt_tokens = chunk;
    state.timings.prefill_seconds += prefill_seconds;
    state.context.decode_ready = state.context.processed_prompt_tokens >=
                                 static_cast<std::uint32_t>(state.context.prompt_tokens.size());
    if (state.context.decode_ready) {
        state.context.next_input   = sampled;
        last_prefill_tokens_       = state.context.prompt_tokens.size();
        last_prefill_seconds_      = state.timings.prefill_seconds;
    }

    runtime::PrefillStepResult result;
    result.summary.prompt_tokens          = static_cast<std::uint32_t>(state.context.prompt_tokens.size());
    result.summary.reused_prompt_tokens   = 0;
    result.round.tokens = state.context.decode_ready
                              ? std::span<const TokenId>{&state.context.next_input, 1}
                              : std::span<const TokenId>{};
    result.processed_prompt_tokens        = chunk;
    result.complete                       = state.context.decode_ready;
    return result;
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill_lane(std::uint32_t lane) {
    if (lane >= plan_->max_concurrency) {
        throw std::invalid_argument("qwen4_exp prefill lane is outside the finalized concurrency");
    }
    LaneState& state = lanes_[lane];
    if (!state.active || state.context.decode_ready) {
        throw std::logic_error("qwen4_exp advance prefill lane has invalid state");
    }

    const std::uint32_t remaining =
        static_cast<std::uint32_t>(state.context.prompt_tokens.size()) - state.context.processed_prompt_tokens;
    const std::uint32_t chunk = std::min(plan_->prefill_chunk, remaining);
    const auto prefill_start = std::chrono::steady_clock::now();
    prefill_chunk(state, state.context.processed_prompt_tokens, chunk);
    state.timings.prefill_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_start).count();
    state.context.processed_prompt_tokens += chunk;
    state.context.decode_ready = state.context.processed_prompt_tokens >=
                                 static_cast<std::uint32_t>(state.context.prompt_tokens.size());
    if (state.context.decode_ready) {
        last_prefill_tokens_  = state.context.prompt_tokens.size();
        last_prefill_seconds_ = state.timings.prefill_seconds;
    }

    runtime::PrefillStepResult result;
    result.summary.prompt_tokens          = static_cast<std::uint32_t>(state.context.prompt_tokens.size());
    result.summary.reused_prompt_tokens   = 0;
    result.round.tokens = state.context.decode_ready
                              ? std::span<const TokenId>{&state.context.next_input, 1}
                              : std::span<const TokenId>{};
    result.processed_prompt_tokens        = chunk;
    result.complete                       = state.context.decode_ready;
    return result;
}

void ProgramImplCore::dflash2_capture_committed(std::uint32_t start_position, std::int32_t count) {
    if (count <= 0) { return; }
    namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
    if (static_cast<std::uint32_t>(count) > dflash2_feature_width_) {
        throw std::logic_error("DFlash2 committed feature width exceeds the capture buffer");
    }
    const ScopedDevice scope(execution_.dev[0]->device);
    cudaStream_t stream = states_[0]->stream();
    // The tap and the target forward that produced these features run on this non-blocking
    // stream; drain it before reading the feature buffer through the small host control copies.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::int32_t> positions(static_cast<std::size_t>(count));
    for (std::int32_t index = 0; index < count; ++index) {
        positions[static_cast<std::size_t>(index)] =
            static_cast<std::int32_t>(start_position + static_cast<std::uint32_t>(index));
    }
    dflash2_positions_.copy_from_host(positions.data(),
                                      static_cast<std::size_t>(count) * sizeof(std::int32_t));
    dflash2_counts_.copy_from_host(&count, sizeof(count));
    flash::DFlash2ContextCall call{};
    call.features  = Tensor(dflash2_features_.p, DType::BF16,
                            {flash::DFlash2Geometry::feature_rows, count});
    call.positions = Tensor(dflash2_positions_.p, DType::I32, {count, 1});
    call.counts    = Tensor(dflash2_counts_.p, DType::I32, {1});
    call.lanes     = Tensor(dflash2_lanes_.p, DType::I32, {1});
    call.min_count = static_cast<std::uint32_t>(count);
    call.max_count = static_cast<std::uint32_t>(count);
    auto work_scope = dflash2_work_->scope();
    if (dflash2_phase_diag_.enabled && dflash2_phase_diag_.in_decode) {
        dflash2_event_record(dflash2_phase_diag_.materialize_begin, stream);
    }
    dflash2_drafter_->materialize(call, *dflash2_work_, stream);
    if (dflash2_phase_diag_.enabled && dflash2_phase_diag_.in_decode) {
        dflash2_event_record(dflash2_phase_diag_.materialize_end, stream);
    }
}

TokenId ProgramImplCore::decode_dflash2(LaneState& state, const runtime::RoundBudget& budget,
                                        std::vector<TokenId>& committed) {
    namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
    committed.clear();
    const ScopedDevice scope(execution_.dev[0]->device);
    cudaStream_t stream = states_[0]->stream();
    auto& diag          = dflash2_phase_diag_;
    const bool diag_on  = diag.enabled;
    if (diag_on) {
        if (diag.pending) {
            // The prior round's materialize (and anything before it) is still in flight; drain it
            // once and read the completed phase events before starting the next round's records.
            CUDA_CHECK(cudaStreamSynchronize(stream));
            dflash2_phase_read_pending(diag, /*print_every=*/16);
        }
        dflash2_event_record(diag.round_begin, stream);
        diag.in_decode = true;
    }
    const std::uint32_t frontier = states_[0]->next_position();
    const std::uint32_t drafts =
        flash::verify_extent(dflash2_window_, dflash2_window_, budget, frontier,
                             plan_->kv_capacity);
    std::uint32_t accepted = 0;
    std::uint32_t verified_drafts = drafts;
    TokenId token          = 0;
    if (drafts == 0) {
        last_stage_capture() = &logits_capture_;
        token                = trunk_->decode(state.context.next_input);
        committed.push_back(token);
        dflash2_capture_committed(frontier, 1);
    } else {
        const std::int32_t k          = static_cast<std::int32_t>(drafts);
        const std::int32_t columns    = k + 1;
        const std::int32_t anchor     = static_cast<std::int32_t>(state.context.next_input);
        const std::int32_t base       = static_cast<std::int32_t>(frontier);
        const std::int32_t lane       = 0;
        // The target forward that filled the ring also ran on this non-blocking stream; drain it
        // before the synchronous small control copies.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        dflash2_sampling_.copy_from_host(&state.sampling, sizeof(state.sampling));
        dflash2_anchors_.copy_from_host(&anchor, sizeof(anchor));
        dflash2_frontiers_.copy_from_host(&base, sizeof(base));
        dflash2_valid_.copy_from_host(&columns, sizeof(columns));
        dflash2_lanes_.copy_from_host(&lane, sizeof(lane));

        flash::DFlash2ProposalCall call{};
        call.anchors       = Tensor(dflash2_anchors_.p, DType::I32, {1});
        call.frontiers     = Tensor(dflash2_frontiers_.p, DType::I32, {1});
        call.valid_columns = Tensor(dflash2_valid_.p, DType::I32, {1});
        call.lanes         = Tensor(dflash2_lanes_.p, DType::I32, {1});
        call.configs       = static_cast<const ops::SamplingConfig*>(dflash2_sampling_.p);
        call.batch         = 1;
        call.width         = columns;
        call.ids           = Tensor(dflash2_ids_.p, DType::I32, {columns, 1});
        call.positions     = Tensor(dflash2_block_positions_.p, DType::I32, {columns, 1});
        call.drafts        = Tensor(dflash2_drafts_.p, DType::I32, {k, 1});
        call.candidates    = Tensor(dflash2_candidates_.p, DType::I32, {16, k, 1});
        call.proposal_q    = Tensor(dflash2_proposal_q_.p, DType::FP32, {16, k, 1});
        call.confidence    = Tensor(dflash2_confidence_.p, DType::FP32, {k, 1});
        if (diag_on) { dflash2_event_record(diag.propose_begin, stream); }
        {
            auto work_scope = dflash2_work_->scope();
            dflash2_drafter_->propose(call, *dflash2_work_, stream);
        }
        if (diag_on) { dflash2_event_record(diag.propose_end, stream); }
        const auto propose_host_begin = diag_on ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
        std::vector<TokenId> proposals(static_cast<std::size_t>(k));
        std::vector<float> confidence(static_cast<std::size_t>(k));
        CUDA_CHECK(cudaMemcpyAsync(proposals.data(), dflash2_drafts_.p,
                                   static_cast<std::size_t>(k) * sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(confidence.data(), dflash2_confidence_.p,
                                   static_cast<std::size_t>(k) * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (diag_on) {
            const double propose_gpu =
                dflash2_event_ms(diag.propose_begin, diag.propose_end);
            const double host_wall = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - propose_host_begin).count();
            diag.propose_ms += propose_gpu;
            // The host block's sync waits for the propose GPU work that the propose event already
            // charges; subtract it so the phases stay non-overlapping.
            diag.propose_host_ms += std::max(0.0, host_wall - propose_gpu);
        }

        // The selector marks how far the conditional chain is worth extending. As with the MTP
        // chain, a positive min_p keeps the first step and stops before the first weak one.
        const float min_p = flash::draft_min_p();
        std::size_t keep  = static_cast<std::size_t>(k);
        if (min_p > 0.0F) {
            keep = 1;
            while (keep < static_cast<std::size_t>(k) && confidence[keep - 1] >= min_p) { ++keep; }
        }
        proposals.resize(keep);
        verified_drafts = static_cast<std::uint32_t>(keep);

        std::vector<TokenId> verify_tokens;
        verify_tokens.reserve(static_cast<std::size_t>(columns));
        verify_tokens.push_back(state.context.next_input);
        verify_tokens.insert(verify_tokens.end(), proposals.begin(), proposals.end());

        if (diag_on) { dflash2_event_record(diag.verify_begin, stream); }
        const TextTrunk::VerificationBaseline baseline =
            trunk_->capture_verification_baseline();
        struct ActionGuard {
            TextTrunk* trunk;
            ~ActionGuard() { trunk->set_gdn_state_action(GdnStateAction::UpdateInPlace); }
        } guard{trunk_.get()};
        trunk_->set_gdn_state_action(GdnStateAction::RecordForReplay);
        last_stage_capture() = &logits_capture_;
        const std::vector<TokenId> predictions = trunk_->decode_span(verify_tokens);
        if (diag_on) { dflash2_event_record(diag.verify_end, stream); }
        if (diag_on) { dflash2_event_record(diag.commit_begin, stream); }
        accepted = flash::accepted_prefix(proposals, predictions);
        committed = flash::commit_tokens(proposals, predictions);
        if (committed.empty()) { committed.push_back(predictions.front()); }
        trunk_->commit_verified_span(verify_tokens, accepted + 1, baseline);
        if (diag_on) { dflash2_event_record(diag.commit_end, stream); }
        dflash2_capture_committed(frontier, static_cast<std::int32_t>(accepted + 1));
    }
    dflash2_stats_.rounds += 1;
    dflash2_stats_.drafted_tokens += verified_drafts;
    dflash2_stats_.accepted_tokens += accepted;
    if (dflash2_stats_.accepted_per_position.size() < dflash2_window_) {
        dflash2_stats_.accepted_per_position.resize(dflash2_window_, 0);
    }
    for (std::uint32_t at = 0; at < accepted && at < dflash2_window_; ++at) {
        dflash2_stats_.accepted_per_position[at] += 1;
    }
    if (diag_on) {
        diag.pending        = true;
        diag.pending_drafts = drafts > 0;
        diag.pending_tokens = committed.size();
        diag.in_decode      = false;
    }
    token = committed.back();
    return token;
}

runtime::BatchedGeneratedRound ProgramImplCore::decode_batch(
    std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets) {
    if (lanes.size() != budgets.size()) {
        throw std::logic_error("qwen4_exp decode batch membership and budgets differ");
    }

    for (const std::uint32_t lane : lanes) {
        if (lane >= plan_->max_concurrency) {
            throw std::invalid_argument("qwen4_exp decode batch lane is outside the finalized concurrency");
        }
        const LaneState& state = lanes_[lane];
        if (!state.active || !state.context.decode_ready) {
            throw std::logic_error("qwen4_exp decode batch lane is not decode-ready");
        }
    }

    round_tokens_.clear();
    round_row_counts_.clear();
    for (std::size_t index = 0; index < lanes.size(); ++index) {
        const std::uint32_t lane = lanes[index];
        LaneState& state         = lanes_[lane];
        const runtime::RoundBudget& budget = budgets[index];
        const auto decode_start  = std::chrono::steady_clock::now();
        const std::size_t tokens_before = round_tokens_.size();
        TokenId token = 0;
        if (dflash2_drafter_ != nullptr) {
            std::vector<TokenId> committed;
            token = decode_dflash2(state, budget, committed);
            for (const TokenId emitted : committed) {
                state.last_round_token     = emitted;
                state.has_last_round_token = true;
                round_tokens_.push_back(emitted);
            }
        } else if (mtp_draft_ != nullptr) {
            namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
            const std::uint32_t frontier = states_[0]->next_position();
            const std::uint32_t drafts =
                flash::verify_extent(mtp_draft_window_, mtp_draft_window_, budget, frontier,
                                     plan_->kv_capacity);
            const int32_t position = static_cast<std::int32_t>(frontier);
            flash::MtpDraft::RankTensors conditioning{};
            for (int rank = 0; rank < execution_.tp; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                conditioning[slot] =
                    Tensor(residual_sinks_[slot]->p, kTextActivationDtype,
                           {TextConfig::residual_width, 1});
            }
            const std::vector<TokenId> proposals =
                flash::draft_chain(*mtp_draft_, conditioning, state.context.next_input, drafts,
                                   position);
            last_stage_capture() = &logits_capture_;
            std::uint32_t accepted = 0;
            std::vector<TokenId> committed;
            if (drafts == 0) {
                // A one-token verify span cannot produce replay records; commit it in place.
                token = trunk_->decode(state.context.next_input);
                committed.push_back(token);
            } else {
                std::vector<TokenId> verify_tokens;
                verify_tokens.reserve(proposals.size() + 1);
                verify_tokens.push_back(state.context.next_input);
                verify_tokens.insert(verify_tokens.end(), proposals.begin(), proposals.end());

                const TextTrunk::VerificationBaseline baseline =
                    trunk_->capture_verification_baseline();
                struct ActionGuard {
                    TextTrunk* trunk;
                    ~ActionGuard() { trunk->set_gdn_state_action(GdnStateAction::UpdateInPlace); }
                } guard{trunk_.get()};
                trunk_->set_gdn_state_action(GdnStateAction::RecordForReplay);
                const std::vector<TokenId> predictions = trunk_->decode_span(verify_tokens);
                accepted = flash::accepted_prefix(proposals, predictions);
                committed = flash::commit_tokens(proposals, predictions);
                if (committed.empty()) { committed.push_back(predictions.front()); }
                trunk_->commit_verified_span(verify_tokens, accepted + 1, baseline);
                // The sink copy is staged on each rank's non-blocking stream; make it visible to
                // the next round's draft before returning.
                {
                    const ScopedDevice scope(states_[0]->device_id());
                    CUDA_CHECK(cudaStreamSynchronize(states_[0]->stream()));
                }
                flash::observe_residual_sink(conditioning[0]);
            }
            mtp_stats_.rounds += 1;
            mtp_stats_.drafted_tokens += static_cast<std::uint64_t>(proposals.size());
            mtp_stats_.accepted_tokens += accepted;
            if (mtp_stats_.accepted_per_position.size() < mtp_draft_window_) {
                mtp_stats_.accepted_per_position.resize(mtp_draft_window_, 0);
            }
            for (std::uint32_t at = 0; at < accepted && at < mtp_draft_window_; ++at) {
                mtp_stats_.accepted_per_position[at] += 1;
            }
            for (const TokenId emitted : committed) {
                state.last_round_token     = emitted;
                state.has_last_round_token = true;
                round_tokens_.push_back(emitted);
            }
            token = state.last_round_token;
        } else if (graph_ != nullptr && !graph_->ready()) {
            ++graph_warm_steps_;
            last_stage_capture() = &logits_capture_;
            token = trunk_->decode(state.context.next_input);
            state.last_round_token     = token;
            state.has_last_round_token = true;
            round_tokens_.push_back(token);
        } else if (graph_ != nullptr && graph_->ready()) {
            token = graph_->replay(state.context.next_input, states_[0]->next_position());
            if (logits_capture_.enabled) {
                const std::uint16_t* words = graph_->logits();
                logits_capture_.words.assign(
                    words, words + static_cast<std::size_t>(TextConfig::vocab));
            }
            state.last_round_token     = token;
            state.has_last_round_token = true;
            round_tokens_.push_back(token);
        } else {
            last_stage_capture() = &logits_capture_;
            token = trunk_->decode(state.context.next_input);
            state.last_round_token     = token;
            state.has_last_round_token = true;
            round_tokens_.push_back(token);
        }
        round_row_counts_.push_back(static_cast<std::int32_t>(round_tokens_.size() - tokens_before));
        state.timings.decode_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
    }

    std::uint32_t stride = 0;
    for (const std::int32_t count : round_row_counts_) {
        stride = std::max<std::uint32_t>(stride, static_cast<std::uint32_t>(count));
    }
    if (stride == 0) { stride = 1; }
    padded_tokens_.clear();
    padded_tokens_.reserve(static_cast<std::size_t>(stride) * lanes.size());
    std::size_t cursor = 0;
    for (const std::int32_t count : round_row_counts_) {
        const std::size_t take = static_cast<std::size_t>(count);
        for (std::size_t at = 0; at < take; ++at) { padded_tokens_.push_back(round_tokens_[cursor++]); }
        TokenId fill = take > 0 ? round_tokens_[cursor - 1] : TokenId{0};
        for (std::size_t at = take; at < stride; ++at) { padded_tokens_.push_back(fill); }
    }
    runtime::BatchedGeneratedRound round;
    round.tokens       = padded_tokens_;
    round.row_counts   = round_row_counts_;
    round.row_stride   = stride;
    return round;
}

runtime::MixedBatchResult
ProgramImplCore::decode_mixed_batch(std::span<const std::uint32_t> lanes,
                                    std::span<const runtime::RoundBudget> budgets,
                                    std::uint32_t owner_lane) {
    (void)lanes;
    (void)budgets;
    (void)owner_lane;
    throw std::logic_error(
        "qwen4_exp text state serves one request and does not plan a mixed decode unit");
}

bool ProgramImplCore::supports_mixed_unit() const noexcept { return false; }

bool ProgramImplCore::mixed_owner_eligible(std::uint32_t lane) const noexcept {
    (void)lane;
    return false;
}

void ProgramImplCore::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    if (lane >= plan_->max_concurrency) {
        throw std::invalid_argument("qwen4_exp prefill resolution lane is outside the finalized concurrency");
    }
    if (terminal) { clear_lane(lane); }
}

void ProgramImplCore::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                            std::span<const std::uint32_t> accepted_tokens,
                                            std::span<const std::uint8_t> terminal,
                                            std::span<const std::uint8_t> cancelled) {
    if (lanes.size() != accepted_tokens.size() || lanes.size() != terminal.size() ||
        lanes.size() != cancelled.size()) {
        throw std::logic_error("qwen4_exp pending batch resolution vectors differ");
    }

    for (std::size_t index = 0; index < lanes.size(); ++index) {
        const std::uint32_t lane = lanes[index];
        if (lane >= plan_->max_concurrency) {
            throw std::invalid_argument("qwen4_exp pending batch lane is outside the finalized concurrency");
        }
        const bool finished = terminal[index] != 0 || cancelled[index] != 0;
        if (!finished && accepted_tokens[index] != 0) {
            LaneState& state = lanes_[lane];
            if (!state.has_last_round_token) {
                throw std::logic_error(
                    "qwen4_exp pending batch has no generated token to continue from");
            }
            state.context.next_input = state.last_round_token;
        }
        if (finished) { clear_lane(lane); }
    }
}

void ProgramImplCore::abort_lane(std::uint32_t lane) noexcept {
    if (lane < plan_->max_concurrency) { clear_lane(lane); }
}

bool ProgramImplCore::has_retained_lane(std::uint32_t lane) const noexcept {
    (void)lane;
    return false;
}

void ProgramImplCore::evict_retained_lane(std::uint32_t lane) noexcept { (void)lane; }

GenerationTimings
ProgramImplCore::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < plan_->max_concurrency ? lanes_[lane].timings : GenerationTimings{};
}

SpeculativeStats
ProgramImplCore::speculative_stats_lane(std::uint32_t lane) const noexcept {
    (void)lane;
    return dflash2_drafter_ != nullptr ? dflash2_stats_ : mtp_stats_;
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device                = execution_.dev[0]->device;
    out.max_context           = plan_->capacity;
    out.effective_max_context = Variant::maximum_context;
    out.rope_mode             = RopeMode::Native;
    out.yarn_mscale           = 1.0;
    out.kv_cache              = KvCacheStorage::MixedBf16Fp16;
    out.kv_capacity           = plan_->kv_capacity;
    out.kv_capacity_page_groups = plan_->main_page_groups + 1;
    out.kv_capacity_max_page_groups = plan_->main_page_groups + 1;
    const auto slot = std::size_t{0};
    const TextResidentStats* resident = models_[slot]->resident;
    if (resident != nullptr) {
        out.weights = ArenaMemorySummary{static_cast<std::size_t>(resident->weights_capacity_bytes),
                                         static_cast<std::size_t>(resident->weights_used_bytes),
                                         static_cast<std::size_t>(resident->weights_used_bytes)};
        out.expert_cache_bytes     = static_cast<std::size_t>(resident->expert_cache_bytes);
        out.expert_scratch_bytes   = static_cast<std::size_t>(resident->expert_scratch_bytes);
        out.expert_cache_pairs     = static_cast<std::size_t>(resident->expert_cache_pairs);
        out.expert_cache_hits      = resident->expert_cache_hits;
        out.expert_cache_misses    = resident->expert_cache_misses;
        out.expert_cache_miss_bytes = resident->expert_cache_miss_bytes;
        out.expert_host_bytes      = static_cast<std::size_t>(resident->expert_host_bytes);
    }
    const std::uint64_t state_bytes = states_[slot]->reserved_bytes();
    out.sequence = ArenaMemorySummary{static_cast<std::size_t>(state_bytes),
                                      static_cast<std::size_t>(state_bytes),
                                      static_cast<std::size_t>(state_bytes)};
    out.workspace = ArenaMemorySummary{
        work_arenas_[slot]->capacity() + activation_arenas_[slot]->capacity(),
        work_arenas_[slot]->used() + activation_arenas_[slot]->used(),
        work_arenas_[slot]->peak_used() + activation_arenas_[slot]->peak_used()};
    out.request_transient = ArenaMemorySummary{plan_->request_transient_capacity_bytes, 0, 0};
    out.minimum_runtime_reservation_bytes = plan_->device_reservation_bytes;
    out.runtime_reservation_bytes         = plan_->device_reservation_bytes;
    out.kv_payload_bytes                  = states_[slot]->kv_payload_bytes();
    out.gdn_state_bytes                   = states_[slot]->gdn_state_bytes();
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        out.host_resident_bytes = static_cast<std::size_t>(usage.ru_maxrss) * 1024U;
    }
    return out;
}

MetricsSnapshot ProgramImplCore::metrics() const noexcept {
    MetricsSnapshot out;
    out.devices.reserve(static_cast<std::size_t>(execution_.tp));
    for (int rank = 0; rank < execution_.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        DeviceMemoryMetrics memory;
        memory.device = execution_.dev[slot]->device;
        {
            const ScopedDevice scope(memory.device);
            std::size_t free_bytes  = 0;
            std::size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                memory.free_bytes  = free_bytes;
                memory.total_bytes = total_bytes;
                memory.used_bytes  = total_bytes - free_bytes;
            }
        }
        out.devices.push_back(memory);
    }

    bool active = false;
    for (const LaneState& lane : lanes_) { active = active || lane.active; }
    const std::uint32_t pages = plan_->main_page_groups + 1;
    out.main_kv.capacity_pages = pages;
    out.main_kv.mapped_pages   = active ? pages : 0;
    out.main_kv.free_pages     = pages - out.main_kv.mapped_pages;
    out.main_kv.page_bytes     = Variant::bytes_per_main_page_group;
    return out;
}

PrefixProbe ProgramImplCore::prefix_probe(std::span<const std::uint64_t> hashes) const {
    (void)hashes;
    PrefixProbe probe;
    probe.block_tokens = 64;
    return probe;
}

CapacitySnapshot ProgramImplCore::capacity_snapshot() const noexcept {
    CapacitySnapshot out;
    bool active = false;
    for (const LaneState& lane : lanes_) { active = active || lane.active; }
    out.total_pages     = plan_->main_page_groups + 1;
    out.free_pages      = active ? 0 : out.total_pages;
    out.max_concurrency = plan_->max_concurrency;
    out.max_context     = plan_->capacity;
    out.prefill_chunk   = plan_->prefill_chunk;
    out.recent_prefill_tps =
        last_prefill_seconds_ > 0.0
            ? static_cast<double>(last_prefill_tokens_) / last_prefill_seconds_
            : 0.0;
    return out;
}

void ProgramImplCore::enable_logits_capture(bool enabled) {
    logits_capture_.enabled = enabled;
    if (!enabled) { logits_capture_.words.clear(); }
}

void ProgramImplCore::reset_memory_peaks() noexcept {}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
