#pragma once

#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "targets/qwen4_exp/impl/runtime/instance.h"
#include "targets/qwen4_exp/impl/runtime/layouts.h"
#include "targets/qwen4_exp/impl/runtime/state_transaction.h"
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

namespace ninfer::targets::qwen3_8_flash_next::detail {
class MtpDraft;
}

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

class GraphDecodeController;

using RequestBasePlanImpl = qwen4_exp::detail::RequestBasePlanImpl<Variant>;
using RequestPlanImpl     = qwen4_exp::detail::RequestPlanImpl<Variant>;

struct LaneState {
    TextLaneContext context;
    runtime::TransientRegion transient;
    GenerationTimings timings;
    TokenId last_round_token = 0;
    bool has_last_round_token = false;
    bool active = false;
};

class ProgramImplCore {
public:
    ProgramImplCore(const std::array<const ModelView*, kMaximumDevices>& models,
                    WeightsProfile weights_profile, std::unique_ptr<SequencePlanImpl> plan,
                    ExecutionContext& execution, const EngineOptions& options);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                     const PreparedPrompt& prompt,
                                                     const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_cache_residue() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                 PreparedPrompt&& prompt,
                                                                 RequestPlan&& plan,
                                                                 runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::MixedBatchResult
    decode_mixed_batch(std::span<const std::uint32_t> lanes,
                       std::span<const runtime::RoundBudget> budgets, std::uint32_t owner_lane);
    [[nodiscard]] bool supports_mixed_unit() const noexcept;
    [[nodiscard]] bool mixed_owner_eligible(std::uint32_t lane) const noexcept;
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    [[nodiscard]] MetricsSnapshot metrics() const noexcept;
    [[nodiscard]] PrefixProbe prefix_probe(std::span<const std::uint64_t> hashes) const;
    [[nodiscard]] CapacitySnapshot capacity_snapshot() const noexcept;

    void reset_memory_peaks() noexcept;

    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept {
        return {logits_capture_.words.data(), logits_capture_.words.size()};
    }
    void enable_logits_capture(bool enabled);
    void enable_peer_egress_check(bool enabled) noexcept { (void)enabled; }
    [[nodiscard]] std::uint64_t peer_egress_check_rounds() const noexcept { return 0; }
    [[nodiscard]] std::uint64_t peer_egress_check_mismatches() const noexcept { return 0; }

    [[nodiscard]] std::shared_ptr<SharedHostTier> shared_host_tier() const noexcept {
        return nullptr;
    }
    void set_shared_tier_owner(std::uint64_t owner) noexcept { (void)owner; }

private:
    [[nodiscard]] runtime::RequestPlanSummary make_summary(std::uint32_t prompt_tokens,
                                                           std::uint32_t requested_output_tokens,
                                                           std::size_t transient_bytes) const;
    [[nodiscard]] std::uint32_t page_groups_for(std::uint32_t tokens) const noexcept;
    void clear_lane(std::uint32_t lane) noexcept;
    void prefill_chunk(LaneState& state, std::uint32_t begin, std::uint32_t count);

    WeightsProfile weights_profile_;
    std::array<const ModelView*, kMaximumDevices> models_{};
    ExecutionContext& execution_;
    EngineOptions options_;
    std::unique_ptr<SequencePlanImpl> plan_;
    std::array<LaneState, kMaximumConcurrency> lanes_{};
    std::array<std::unique_ptr<TextState>, kMaximumDevices> states_;
    std::array<std::unique_ptr<WorkspaceArena>, kMaximumDevices> activation_arenas_;
    std::array<std::unique_ptr<WorkspaceArena>, kMaximumDevices> work_arenas_;
    std::array<const ModelView*, kMaximumDevices> view_ptrs_{};
    std::array<TextState*, kMaximumDevices> state_ptrs_{};
    std::array<WorkspaceArena*, kMaximumDevices> activation_ptrs_{};
    std::array<WorkspaceArena*, kMaximumDevices> workspace_ptrs_{};
    std::unique_ptr<ops::CollectiveComms> comms_;
    std::unique_ptr<TextTrunk> trunk_;
    std::unique_ptr<GraphDecodeController> graph_;
    std::unique_ptr<ninfer::targets::qwen3_8_flash_next::detail::MtpDraft> mtp_draft_;
    std::unique_ptr<StateTransaction> mtp_transaction_;
    std::array<std::unique_ptr<DeviceBuffer>, kMaximumDevices> residual_sinks_;
    std::uint32_t mtp_draft_window_ = 0;
    SpeculativeStats mtp_stats_{};
    LastStageCapture logits_capture_;
    std::vector<TokenId> round_tokens_;
    std::vector<TokenId> padded_tokens_;
    std::vector<std::int32_t> round_row_counts_;
    std::uint64_t last_prefill_tokens_ = 0;
    double last_prefill_seconds_       = 0.0;
    std::uint32_t graph_warm_steps_    = 0;
};

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS

namespace ninfer::targets::qwen4_exp::detail {

template <>
class ProgramImpl<NINFER_QWEN4EXP_VARIANT> final : public NINFER_QWEN4EXP_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN4EXP_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen4_exp::detail
