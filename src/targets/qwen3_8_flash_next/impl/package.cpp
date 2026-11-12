#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

#include <memory>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next {
namespace detail {

class LoadPlan::Impl {
public:
    explicit Impl(BindingPlan plan_in) : bindings(std::move(plan_in)) {}

    BindingPlan bindings;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan() = default;

const BindingPlan& LoadPlan::bindings() const { return impl_->bindings; }

} // namespace detail

std::optional<Package::WeightsProfile> Package::resolve_weights(std::string_view model,
                                                                std::string_view weights) {
    if (model == model_id && weights == weights_id) { return WeightsProfile::Nvfp4; }
    return std::nullopt;
}

ModelSamplingDefaults Package::sampling_defaults() {
    return ModelSamplingDefaults{
        .thinking =
            {.temperature       = 1.0F,
             .top_k             = 20,
             .top_p             = 0.95F,
             .min_p             = 0.0F,
             .presence_penalty  = 0.0F,
             .frequency_penalty = 0.0F},
        .non_thinking =
            {.temperature       = 1.0F,
             .top_k             = 20,
             .top_p             = 0.95F,
             .min_p             = 0.0F,
             .presence_penalty  = 0.0F,
             .frequency_penalty = 0.0F}};
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, WeightsProfile profile, int tp) {
    return LoadPlan(
        std::make_unique<detail::LoadPlan::Impl>(detail::bind_artifact(binder, profile, tp)));
}

} // namespace ninfer::targets::qwen3_8_flash_next
