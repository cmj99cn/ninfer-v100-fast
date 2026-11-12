#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace ninfer {

namespace artifact {
class Binder;
} // namespace artifact

namespace targets::qwen3_8_flash_next {

struct Package;

namespace detail {

struct BindingPlan;

// The weights recipe of this package. The package registers one recipe: the NVFP4 text
// checkpoint.
enum class WeightsProfile : std::uint8_t {
    Nvfp4,
};

// The plan of one load: which artifact objects hold which weight roles and in which geometry.
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

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
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

    using WeightsProfile = detail::WeightsProfile;
    using LoadPlan       = detail::LoadPlan;

    // The weights recipe of an artifact identity, or no recipe when the identity is not this
    // package's.
    [[nodiscard]] static std::optional<WeightsProfile> resolve_weights(std::string_view model,
                                                                      std::string_view weights);

    // The model's own sampling defaults. The checkpoint publishes one preset -- temperature
    // 1.0, top-k 20, top-p 0.95 -- and no separate thinking and non-thinking presets, so both
    // modes carry it.
    [[nodiscard]] static ModelSamplingDefaults sampling_defaults();

    // The binding plan of a load. `tp` is the tensor-parallel width; only one device is
    // planned at this milestone.
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, WeightsProfile profile,
                                            int tp);
};

} // namespace targets::qwen3_8_flash_next
} // namespace ninfer
