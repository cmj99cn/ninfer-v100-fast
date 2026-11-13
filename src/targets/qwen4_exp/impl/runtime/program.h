#pragma once

#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "targets/qwen4_exp/impl/runtime/instance.h"
#include "targets/qwen4_exp/impl/runtime/layouts.h"
#include "targets/qwen4_exp/impl/runtime/text_context.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail {

template <>
struct RequestBasePlanImpl<NINFER_QWEN4EXP_VARIANT> {
    runtime::RequestPlanSummary summary;
};

template <>
struct RequestPlanImpl<NINFER_QWEN4EXP_VARIANT> {
    runtime::RequestPlanSummary summary;
};

} // namespace ninfer::targets::qwen4_exp::detail

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

using RequestBasePlanImpl = qwen4_exp::detail::RequestBasePlanImpl<Variant>;
using RequestPlanImpl     = qwen4_exp::detail::RequestPlanImpl<Variant>;

struct LaneState {
    TextLaneContext context;
    runtime::TransientRegion transient;
    bool active = false;
};

class ProgramImplCore {
public:
    ProgramImplCore(const std::array<const ModelView*, kMaximumDevices>& models,
                    WeightsProfile weights_profile, std::unique_ptr<SequencePlanImpl> plan,
                    ExecutionContext& execution, const EngineOptions& options);
    ~ProgramImplCore() noexcept = default;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                     const PreparedPrompt& prompt,
                                                     const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                 PreparedPrompt&& prompt,
                                                                 RequestPlan&& plan,
                                                                 runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    [[nodiscard]] MetricsSnapshot metrics() const noexcept;
    [[nodiscard]] CapacitySnapshot capacity_snapshot() const noexcept;

    void reset_memory_peaks() noexcept;

private:
    [[nodiscard]] runtime::RequestPlanSummary make_summary(std::uint32_t prompt_tokens,
                                                           std::uint32_t requested_output_tokens,
                                                           std::size_t transient_bytes) const;
    [[nodiscard]] std::uint32_t page_groups_for(std::uint32_t tokens) const noexcept;
    void clear_lane(std::uint32_t lane) noexcept;

    WeightsProfile weights_profile_;
    std::array<const ModelView*, kMaximumDevices> models_{};
    ExecutionContext& execution_;
    EngineOptions options_;
    std::unique_ptr<SequencePlanImpl> plan_;
    std::array<LaneState, kMaximumConcurrency> lanes_{};
    TextState state_;
    WorkspaceArena workspace_;
    TextTrunk trunk_;
    std::vector<TokenId> round_tokens_;
    std::vector<std::int32_t> round_row_counts_;
};

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS

namespace ninfer::targets::qwen4_exp::detail {

template <>
class ProgramImpl<NINFER_QWEN4EXP_VARIANT> final : public NINFER_QWEN4EXP_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN4EXP_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen4_exp::detail
