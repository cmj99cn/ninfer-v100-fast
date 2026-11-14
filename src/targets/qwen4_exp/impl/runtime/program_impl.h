#pragma once

#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "targets/qwen4_exp/impl/runtime/instance.h"
#include "targets/qwen4_exp/impl/runtime/layouts.h"
#include "core/arena.h"
#include "targets/qwen4_exp/impl/runtime/program.h"
#include "targets/qwen4_exp/impl/runtime/text_context.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

ProgramImplCore::ProgramImplCore(const std::array<const ModelView*, kMaximumDevices>& models,
                                 WeightsProfile weights_profile, std::unique_ptr<SequencePlanImpl> plan,
                                 ExecutionContext& execution, const EngineOptions& options)
    : weights_profile_(weights_profile), models_(models), execution_(execution), options_(options),
      plan_(std::move(plan)),
      state_(execution_.primary(), *models_[0],
             std::max<std::uint32_t>(plan_->kv_capacity, plan_->capacity),
             std::max<std::uint32_t>(plan_->prefill_chunk, 1)),
      activations_(std::max<std::size_t>(plan_->activation_capacity_bytes, 1)),
      workspace_(std::max<std::size_t>(plan_->workspace_capacity_bytes, 1)),
      trunk_(*models_[0], state_, activations_, workspace_, execution_.primary().stream) {
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
}

runtime::RequestPlanSummary ProgramImplCore::make_summary(std::uint32_t prompt_tokens,
                                                           std::uint32_t requested_output_tokens,
                                                           std::size_t transient_bytes) const {
    runtime::RequestPlanSummary summary;
    summary.prompt_tokens              = prompt_tokens;
    summary.reusable_prompt_tokens     = 0;
    summary.requested_output_tokens    = requested_output_tokens;
    summary.effective_output_tokens    = requested_output_tokens;
    summary.transient_bytes            = transient_bytes;
    summary.transient_alignment        = 1;
    summary.admission.active_lanes     = 1;
    summary.admission.main_kv_pages    = page_groups_for(prompt_tokens);
    summary.service_work_quanta        = 1;
    return summary;
}

std::uint32_t ProgramImplCore::page_groups_for(std::uint32_t tokens) const noexcept {
    return (tokens + Variant::kv_page_tokens - 1) / Variant::kv_page_tokens;
}

void ProgramImplCore::clear_lane(std::uint32_t lane) noexcept {
    lanes_[lane] = LaneState{};
}

void ProgramImplCore::prefill_chunk(LaneState& state, std::uint32_t begin, std::uint32_t count) {
    const std::span<const TokenId> chunk{state.context.prompt_tokens.data() + begin, count};
    const TokenId sampled = trunk_.prefill(chunk);
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

runtime::AdmissionResources ProgramImplCore::admission_capacity() const noexcept {
    return runtime::AdmissionResources{
        .active_lanes  = plan_->max_concurrency,
        .main_kv_pages = plan_->main_page_groups,
    };
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

    state_.reset();
    std::vector<TokenId>& tokens = data.token_ids;
    const std::uint32_t chunk = std::min(plan_->prefill_chunk,
                                         static_cast<std::uint32_t>(tokens.size()));
    const TokenId sampled = trunk_.prefill(std::span<const TokenId>{tokens.data(), chunk});

    LaneState& state = lanes_[lane];
    state.active    = true;
    state.transient = std::move(transient);
    state.context.prompt_tokens = std::move(tokens);
    state.context.processed_prompt_tokens = chunk;
    state.context.decode_ready = state.context.processed_prompt_tokens >=
                                 static_cast<std::uint32_t>(state.context.prompt_tokens.size());
    if (state.context.decode_ready) { state.context.next_input = sampled; }

    runtime::PrefillStepResult result;
    result.summary.prompt_tokens          = static_cast<std::uint32_t>(state.context.prompt_tokens.size());
    result.summary.reused_prompt_tokens   = 0;
    result.round.tokens                   = std::span<const TokenId>{};
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
    prefill_chunk(state, state.context.processed_prompt_tokens, chunk);
    state.context.processed_prompt_tokens += chunk;
    state.context.decode_ready = state.context.processed_prompt_tokens >=
                                 static_cast<std::uint32_t>(state.context.prompt_tokens.size());

    runtime::PrefillStepResult result;
    result.summary.prompt_tokens          = static_cast<std::uint32_t>(state.context.prompt_tokens.size());
    result.summary.reused_prompt_tokens   = 0;
    result.round.tokens                   = std::span<const TokenId>{};
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
    for (const std::uint32_t lane : lanes) {
        const LaneState& state = lanes_[lane];
        round_tokens_.push_back(trunk_.decode(state.context.next_input));
        round_row_counts_.push_back(1);
    }

    runtime::BatchedGeneratedRound round;
    round.tokens       = round_tokens_;
    round.row_counts   = round_row_counts_;
    round.row_stride   = 1;
    return round;
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
        lanes_[lane].context.next_input = accepted_tokens[index];
        if (terminal[index] || cancelled[index]) { clear_lane(lane); }
    }
}

void ProgramImplCore::abort_lane(std::uint32_t lane) noexcept {
    if (lane < plan_->max_concurrency) { clear_lane(lane); }
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    return MemorySummary{};
}

MetricsSnapshot ProgramImplCore::metrics() const noexcept {
    return MetricsSnapshot{};
}

CapacitySnapshot ProgramImplCore::capacity_snapshot() const noexcept {
    return CapacitySnapshot{};
}

void ProgramImplCore::reset_memory_peaks() noexcept {}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
