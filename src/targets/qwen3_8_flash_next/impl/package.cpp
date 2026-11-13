#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/variant.h"

#include <ninfer/targets/qwen4_exp/frontend.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next {
namespace detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan binding_in,
         artifact::MaterializationPlan materialization_in,
         qwen4_exp::FrontendResourcePlan resources_in, int tensor_parallel)
        : weights_profile(weights_profile_in), bindings(std::move(binding_in)),
          materialization(std::move(materialization_in)), resources(resources_in), tp(tensor_parallel) {}

    WeightsProfile weights_profile;
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
    qwen4_exp::FrontendResourcePlan resources;
    int tp = 1;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan() = default;

const BindingPlan& LoadPlan::bindings() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->bindings;
}

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->materialization;
}

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, RuntimeModelView view_in,
         qwen4_exp::FrontendResources resources_in, int tensor_parallel)
        : weights_profile(weights_profile_in), view(std::move(view_in)),
          resources(std::move(resources_in)), tp(tensor_parallel) {}

    WeightsProfile weights_profile;
    RuntimeModelView view;
    qwen4_exp::FrontendResources resources;
    int tp = 1;
};

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;

} // namespace detail

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == weights_id) {
        return WeightsProfile::Nvfp4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model != model_id) {
        throw std::runtime_error("model '" + std::string(model) +
                                 "' has no sampling defaults in target package '" +
                                 std::string(target_key) + "'");
    }
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

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    if (binder.device_count() != options.tp) {
        throw std::invalid_argument("artifact binder device count does not match EngineOptions.tp");
    }

    detail::BindingPlan bindings = detail::bind_artifact(binder, weights_profile, options.tp);
    qwen4_exp::FrontendResourcePlan resources = qwen4_exp::bind_frontend_resources(binder);

    for (const auto& object : bindings.objects) {
        binder.materialize_on_device(object.handle);
    }
    binder.validate_unconsumed_matching("");

    return LoadPlan(std::make_unique<LoadPlan::Impl>(weights_profile, std::move(bindings),
                                                      binder.finish(), resources, options.tp));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }

    qwen4_exp::FrontendResources resources;
    if (materialized.stats().resource_count != 0) {
        resources = qwen4_exp::take_frontend_resources(materialized, plan.impl_->resources);
    }

    auto impl = std::make_unique<LoadedModel::Impl>(plan.impl_->weights_profile,
                                                     detail::RuntimeModelView{.tp = plan.impl_->tp},
                                                     std::move(resources), plan.impl_->tp);
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen4_exp::make_frontend(model.impl_->resources,
                                    qwen4_exp::FrontendOptions{
                                        .vision_enabled           = false,
                                        .max_context              = options.max_context,
                                        .media_cache_bytes        = options.media_cache_bytes,
                                        .media_live_bytes         = options.media_live_bytes,
                                        .media_preprocess_threads = options.media_preprocess_threads,
                                    });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen4_exp::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan,
                        ExecutionContext& execution, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (model.impl_->tp != execution.tp) {
        throw std::invalid_argument("loaded model shard width does not match the execution context");
    }

    std::array<const detail::RuntimeModelView*, artifact::kMaximumDevices> models{};
    for (int rank = 0; rank < execution.tp; ++rank) {
        models[static_cast<std::size_t>(rank)] = &model.impl_->view;
    }
    return qwen4_exp::create_program<detail::Variant>(models, model.impl_->weights_profile,
                                                       std::move(plan), execution, options);
}

} // namespace ninfer::targets::qwen3_8_flash_next
