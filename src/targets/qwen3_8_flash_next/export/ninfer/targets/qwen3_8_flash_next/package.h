#pragma once

#include "artifact/materializer.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen4_exp/frontend.h>
#include <ninfer/targets/qwen4_exp/runtime.h>
#include <ninfer/types.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen3_8_flash_next {

struct Package;

namespace detail {

struct Variant;

struct BindingPlan;

// The weights recipe of this package. The package registers one recipe: the NVFP4 text
// checkpoint.
enum class WeightsProfile : std::uint8_t {
    Nvfp4,
};

// The binding plan of one load: which artifact objects hold which weight roles and in which geometry.
//
// The plan is a binding fact, not a placement: it names the object of every weight role, the
// rectangle the role occupies inside that object and the object's format. The Engine wiring
// turns the plan into device residency.
class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    // The binding plan of the load.
    [[nodiscard]] const BindingPlan& bindings() const;

    // The device placement produced by the binder.
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_flash_next::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_flash_next::Package;
};

} // namespace detail

// The qwen3.8-flash-next package: the identity, the weights recipe and the binding plan of the
// Qwen3.8-Flash-Next / Qwen4Exp checkpoint.
struct Package {
    static constexpr std::string_view model_id   = "qwen3.8-flash-next";
    static constexpr std::string_view weights_id = "nvfp4";
    static constexpr std::string_view target_key = "qwen3_8_flash_next";

    using WeightsProfile  = detail::WeightsProfile;
    using LoadPlan        = detail::LoadPlan;
    using LoadedModel     = detail::LoadedModel;
    using Frontend        = qwen4_exp::Frontend;
    using PreparedPrompt     = qwen4_exp::PreparedPrompt;
    using PreparedPromptData = qwen4_exp::PreparedPromptData;
    using OutputSession      = qwen4_exp::OutputSession;
    using SequencePlanner = qwen4_exp::SequencePlanner<detail::Variant>;
    using SequencePlan    = qwen4_exp::SequencePlan<detail::Variant>;
    using RequestBasePlan = qwen4_exp::RequestBasePlan<detail::Variant>;
    using RequestPlan     = qwen4_exp::RequestPlan<detail::Variant>;
    using Program         = qwen4_exp::Program<detail::Variant>;

    // The weights recipe of an artifact identity. The package owns exactly one identity.
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);

    // The model's own sampling defaults. The checkpoint publishes one preset -- temperature
    // 1.0, top-k 20, top-p 0.95 -- and no separate thinking and non-thinking presets, so both
    // modes carry it.
    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);

    // The binding and materialization plan of a load. `tp` is the tensor-parallel width; the
    // full-model execution path plans two ranks, each holding its own shard, KV/state planes and
    // expert cache.
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model,
                                                const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                                const EngineOptions& options,
                                                                WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, ExecutionContext& execution,
                   const EngineOptions& options);
};

} // namespace targets::qwen3_8_flash_next
} // namespace ninfer
