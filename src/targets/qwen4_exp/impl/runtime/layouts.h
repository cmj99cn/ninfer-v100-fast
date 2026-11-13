#pragma once

#include "runtime/contract/types.h"
#include "targets/qwen4_exp/impl/runtime/instance.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

struct SequencePlanningInputs {
    WeightsProfile weights_profile;
    std::uint32_t capacity        = 0;
    std::uint32_t max_concurrency = 1;
    std::uint32_t prefill_chunk   = 0;
    int device                    = 0;
    int tp                        = 1;
};

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS

namespace ninfer::targets::qwen4_exp::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN4EXP_VARIANT> {
    typename NINFER_QWEN4EXP_VARIANT::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t kv_capacity              = 0;
    std::uint32_t main_page_groups         = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    int device                             = 0;
    int tp                                 = 1;
    std::size_t device_reservation_bytes   = 0;
    std::size_t workspace_capacity_bytes   = 0;
    std::size_t activation_capacity_bytes  = 0;
    std::size_t request_transient_capacity_bytes = 0;
};

template <>
struct SequencePlannerImpl<NINFER_QWEN4EXP_VARIANT> {
    NINFER_QWEN4EXP_RUNTIME_NS::SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
};

} // namespace ninfer::targets::qwen4_exp::detail

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

using SequencePlanImpl = qwen4_exp::detail::SequencePlanImpl<Variant>;

[[nodiscard]] std::unique_ptr<qwen4_exp::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen4_exp::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups);

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
