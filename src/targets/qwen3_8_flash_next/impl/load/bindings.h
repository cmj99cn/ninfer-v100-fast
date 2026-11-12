#pragma once

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/config.h"
#include "targets/qwen3_8_flash_next/impl/load/geometry.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// One artifact object the package bound, with the artifact's own descriptor.
struct BoundObject {
    std::string name;
    artifact::ObjectHandle handle;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
    artifact::StorageLayout layout = artifact::StorageLayout::ContiguousLeV1;
    std::vector<std::uint64_t> shape;
};

// The logical rectangle one artifact parameter occupies inside one object.
struct BoundPart {
    std::uint32_t object = 0;
    artifact::ElementIntervalView view;
};

// One artifact parameter of the text namespace: the role it denotes, the layer it lives in,
// the artifact's own name for it, the rectangles it occupies and the auxiliary inputs its
// operator use edge carries.
struct BoundParameter {
    roles::Role role = roles::Role::TokenEmbedding;
    std::uint32_t layer = 0;
    std::string parameter;
    std::vector<BoundPart> parts;
    // The auxiliary inputs of the parameter's use edge, by the auxiliary's role name.
    std::map<std::string, std::vector<BoundPart>> auxiliaries;
};

// The four-stream gated residual block of one layer or of the final mix.
struct HcPlan {
    BoundParameter norm;
    BoundParameter down;
    BoundParameter inject;
    BoundParameter up;
};

// The linear-attention (GDN) mixer of one layer.
struct GdnPlan {
    BoundParameter query;
    BoundParameter key;
    BoundParameter value;
    BoundParameter z;
    BoundParameter a_projection;
    BoundParameter b_projection;
    BoundParameter a_log;
    BoundParameter dt_bias;
    BoundParameter convolution;
    BoundParameter norm;
    BoundParameter output;
};

// The indexed-attention (QSA) mixer of one layer.
struct QsaPlan {
    BoundParameter query;
    BoundParameter key;
    BoundParameter value;
    BoundParameter index_query;
    BoundParameter index_key;
    BoundParameter output;
    BoundParameter output_gate;
    BoundParameter query_norm;
    BoundParameter key_norm;
    BoundParameter index_query_norm;
    BoundParameter index_key_norm;
};

// The mixture of experts of one layer.
struct MoePlan {
    BoundParameter router;
    BoundParameter shared_score;
    BoundParameter shared_gate;
    BoundParameter shared_up;
    BoundParameter shared_down;
    BoundParameter routed_gate;
    BoundParameter routed_up;
    BoundParameter routed_down;
};

// One text layer: its two four-stream blocks, its mixer and its mixture of experts.
struct LayerPlan {
    HcPlan attention_hc;
    HcPlan moe_hc;
    std::variant<GdnPlan, QsaPlan> mixer;
    MoePlan moe;
};

// The n-gram prediction layer of the trunk.
struct PlePlan {
    BoundParameter table;
    BoundParameter key;
    BoundParameter value;
    BoundParameter norm_key;
    BoundParameter norm_query;
    BoundParameter norm_conv;
    BoundParameter convolution;
};

// The artifact parameters this milestone does not bind.
//
// The text namespace belongs to this milestone, so an unclaimed text parameter is an error
// rather than a diagnostic. The other namespaces belong to later milestones: their parameters
// are counted here, so that a milestone that owns one of them finds every parameter of that
// namespace already listed.
struct UnclaimedParameters {
    std::map<std::string, std::vector<std::string>> by_namespace;
};

// The binding plan: which artifact object holds which weight role, which rectangle of that
// object the role occupies, and in which format the object is stored.
struct BindingPlan {
    std::vector<BoundObject> objects;
    BoundParameter token_embedding;
    BoundParameter output_head;
    HcPlan final_hc;
    std::array<LayerPlan, TextConfig::layers> layers;
    PlePlan ple;
    UnclaimedParameters unclaimed;

    // Every parameter the plan binds, in the order the plan holds them: the global
    // parameters, then every layer, then the n-gram prediction layer.
    [[nodiscard]] std::vector<const BoundParameter*> parameters() const;
};

// Binds every artifact object the text milestone of this package needs.
//
// The binder is consumed: every bound object is claimed exactly once, and the objects of the
// namespaces this milestone does not own stay unclaimed. The plan therefore describes the
// binding and the geometry of the text namespace only; the placement of the bound objects and
// the materialization of the artifact belong to the Engine wiring.
//
// `tp` is the tensor-parallel width: only one device is planned at this milestone, so a wider
// plan is rejected instead of silently binding the whole model on one device.
[[nodiscard]] BindingPlan bind_artifact(artifact::Binder& binder, WeightsProfile profile, int tp);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
