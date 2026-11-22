#pragma once

#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "targets/qwen4_exp/impl/runtime/instance.h"
#include "targets/qwen4_exp/impl/runtime/layouts.h"
#include "core/arena.h"
#include "targets/qwen4_exp/impl/runtime/program.h"
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <sys/resource.h>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

inline constexpr std::uint32_t kGraphCaptureWarmSteps = 8;

ProgramImplCore::~ProgramImplCore() noexcept = default;

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

    for (int rank = 0; rank < execution_.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const ScopedDevice scope(execution_.dev[slot]->device);
        states_[slot] = std::make_unique<TextState>(
            *execution_.dev[slot], *models_[slot],
            std::max<std::uint32_t>(plan_->kv_capacity, plan_->capacity),
            std::max<std::uint32_t>(plan_->prefill_chunk, 1), execution_.tp, rank);
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
    const bool mtp_enabled = options_.speculative.backend == SpeculativeBackend::Mtp &&
                             options_.speculative.draft_tokens > 0;
    if (mtp_enabled) {
        mtp_draft_window_ = std::min<std::uint32_t>(
            options_.speculative.draft_tokens,
            ninfer::targets::qwen3_8_flash_next::detail::MtpConfig::maximum_draft_window);
        const std::size_t sink_bytes = static_cast<std::size_t>(plan_->prefill_chunk) *
                                       static_cast<std::size_t>(TextConfig::residual_width) * 2;
        for (int rank = 0; rank < execution_.tp; ++rank) {
            residual_sinks_[static_cast<std::size_t>(rank)] = std::make_unique<DeviceBuffer>(sink_bytes);
            trunk_->set_residual_sink(rank, residual_sinks_[static_cast<std::size_t>(rank)]->p,
                                      sink_bytes);
        }
        mtp_draft_ = std::make_unique<ninfer::targets::qwen3_8_flash_next::detail::MtpDraft>(
            options_.artifact_path, execution_.dev[0]->device);
        mtp_transaction_ =
            std::make_unique<StateTransaction>(state_ptrs_, execution_.tp, mtp_draft_window_ + 1);
        mtp_stats_.backend      = SpeculativeBackend::Mtp;
        mtp_stats_.enabled      = true;
        mtp_stats_.draft_window = mtp_draft_window_;
    } else if (options_.use_cuda_graph) {
        graph_ = std::make_unique<GraphDecodeController>(
            execution_, state_ptrs_, activation_ptrs_, workspace_ptrs_, execution_.tp);
    }
    CUDA_CHECK(cudaSetDevice(execution_.dev[0]->device));
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
    last_stage_capture() = &logits_capture_;
    const TokenId sampled = trunk_->prefill(chunk);
    if (begin + count == state.context.prompt_tokens.size()) {
        state.context.next_input = sampled;
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

    return RequestBasePlan(std::make_unique<RequestBasePlanImpl>(
        RequestBasePlanImpl{make_summary(prompt_tokens, options.requested_output_tokens,
                                        plan_->request_transient_capacity_bytes)}));
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

    return RequestPlan(std::make_unique<RequestPlanImpl>(RequestPlanImpl{base.summary()}));
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
    last_stage_capture() = &logits_capture_;
    const TokenId sampled = trunk_->prefill(std::span<const TokenId>{tokens.data(), chunk});
    const double prefill_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_start).count();

    LaneState& state = lanes_[lane];
    state.timings    = GenerationTimings{};
    state.active     = true;
    state.transient  = std::move(transient);
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
        if (mtp_draft_ != nullptr) {
            namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
            const std::uint32_t frontier = states_[0]->next_position();
            const std::uint32_t drafts =
                flash::verify_extent(mtp_draft_window_, mtp_draft_window_, budget, frontier,
                                     plan_->kv_capacity);
            const int32_t position = static_cast<std::int32_t>(frontier);
            const Tensor conditioning(residual_sinks_[0]->p, kTextActivationDtype,
                                      {TextConfig::residual_width, 1});
            const std::vector<TokenId> proposals =
                flash::draft_chain(*mtp_draft_, conditioning, state.context.next_input, drafts,
                                   position);
            last_stage_capture() = &logits_capture_;
            mtp_transaction_->begin();
            std::vector<TokenId> verify_tokens;
            verify_tokens.reserve(proposals.size() + 1);
            verify_tokens.push_back(state.context.next_input);
            verify_tokens.insert(verify_tokens.end(), proposals.begin(), proposals.end());
            const std::vector<TokenId> predictions = flash::verify_sequential(
                verify_tokens, [&](TokenId next) { return trunk_->decode(next); });
            const std::uint32_t accepted = flash::accepted_prefix(proposals, predictions);
            std::vector<TokenId> committed = flash::commit_tokens(proposals, predictions);
            // On full acceptance the verified sequence equals the committed sequence token for
            // token (verify_tokens == replay, both decoded T=1 in order), so the live state is
            // already the committed state and the rollback+replay pass is redundant. Only a
            // partial acceptance has to rewind and re-apply the accepted prefix.
            if (accepted != proposals.size()) {
                std::vector<TokenId> replay;
                replay.reserve(accepted + 1);
                replay.push_back(state.context.next_input);
                replay.insert(replay.end(), proposals.begin(), proposals.begin() + accepted);
                mtp_transaction_->commit(replay, [&](TokenId next) { (void)trunk_->decode(next); });
            }
            mtp_stats_.rounds += 1;
            mtp_stats_.drafted_tokens += drafts;
            mtp_stats_.accepted_tokens += accepted;
            if (mtp_stats_.accepted_per_position.size() < mtp_draft_window_) {
                mtp_stats_.accepted_per_position.resize(mtp_draft_window_, 0);
            }
            for (std::uint32_t at = 0; at < accepted && at < mtp_draft_window_; ++at) {
                mtp_stats_.accepted_per_position[at] += 1;
            }
            if (committed.empty()) { committed.push_back(predictions.front()); }
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
    return mtp_stats_;
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
