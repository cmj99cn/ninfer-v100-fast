#pragma once

#include "targets/qwen4_exp/impl/runtime/layouts.h"
#include "targets/qwen4_exp/impl/runtime/workspace_recipe.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

namespace {

inline std::uint32_t groups_for(std::uint32_t tokens, std::uint32_t page_tokens) {
    if (page_tokens == 0) { return 0; }
    return (tokens + page_tokens - 1) / page_tokens;
}

} // namespace

[[nodiscard]] std::unique_ptr<qwen4_exp::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile) {
    if (options.max_context == 0 || options.max_context > kMaximumContext) {
        throw std::invalid_argument("qwen4_exp family cannot plan a context beyond its native capacity");
    }
    if (options.tp < 1 || options.tp > static_cast<int>(kMaximumDevices)) {
        throw std::invalid_argument("qwen4_exp family requires tensor-parallel width in [1, 4]");
    }
    if (options.max_concurrency != 1) {
        throw std::invalid_argument(
            "qwen4_exp family serves one active request; EngineOptions.max_concurrency must be 1");
    }

    auto planner = std::make_unique<qwen4_exp::detail::SequencePlannerImpl<Variant>>();
    planner->inputs = SequencePlanningInputs{
        .weights_profile = weights_profile,
        .capacity        = options.max_context,
        .max_concurrency = options.max_concurrency,
        .prefill_chunk   = static_cast<std::uint32_t>(
            align_up(options.prefill_chunk, kPrefillChunkAlignment)),
        .device          = device.device,
        .tp              = options.tp,
    };

    const std::uint32_t minimum_groups =
        std::max(1U, groups_for(options.max_context, Variant::kv_page_tokens));
    const std::uint32_t maximum_groups =
        std::max(minimum_groups, groups_for(kMaximumContext, Variant::kv_page_tokens));

    planner->curve = runtime::SequenceCapacityCurve{
        .main_page_tokens                   = Variant::kv_page_tokens,
        .minimum_main_page_groups           = minimum_groups,
        .maximum_main_page_groups           = maximum_groups,
        .minimum_device_reservation_bytes   = Variant::minimum_device_reservation_bytes +
                                              static_cast<std::size_t>(minimum_groups - 1) *
                                                  Variant::bytes_per_additional_main_page_group,
        .bytes_per_additional_main_page_group = Variant::bytes_per_additional_main_page_group,
    };
    return planner;
}

[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen4_exp::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr) { throw std::logic_error("qwen4_exp sequence planner is empty"); }
    const runtime::SequenceCapacityCurve& curve = planner->curve;
    if (main_page_groups < curve.minimum_main_page_groups ||
        main_page_groups > curve.maximum_main_page_groups) {
        throw std::invalid_argument("qwen4_exp main page group count is outside the capacity curve");
    }

    auto plan = std::make_unique<SequencePlanImpl>();
    plan->weights_profile = planner->inputs.weights_profile;
    plan->capacity        = curve.resolved_tokens(main_page_groups);
    plan->kv_capacity     = curve.resolved_tokens(main_page_groups);
    plan->main_page_groups = main_page_groups;
    plan->max_concurrency = planner->inputs.max_concurrency;
    plan->prefill_chunk   = planner->inputs.prefill_chunk;
    plan->device          = planner->inputs.device;
    plan->tp              = planner->inputs.tp;
    plan->device_reservation_bytes = curve.reservation_bytes(main_page_groups);
    const WorkspaceRecipe recipe =
        text_workspace_recipe(planner->inputs.prefill_chunk, planner->inputs.max_concurrency);
    plan->workspace_capacity_bytes = recipe.lane_capacity_bytes;
    plan->activation_capacity_bytes = text_activation_capacity_bytes(planner->inputs.prefill_chunk);
    plan->request_transient_capacity_bytes = recipe.request_transient_capacity_bytes;
    return plan;
}

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
