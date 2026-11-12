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

struct BoundObject {
    std::string name;
    artifact::ObjectHandle handle;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
    artifact::StorageLayout layout = artifact::StorageLayout::ContiguousLeV1;
    std::vector<std::uint64_t> shape;
};

struct BoundPart {
    std::uint32_t object = 0;
    artifact::ElementIntervalView view;
};

struct ExpertLibrary {
    std::string object;
    artifact::ObjectHandle handle;
    std::uint64_t matrices = 0;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_count = 0;
};

struct BoundParameter {
    roles::Role role = roles::Role::TokenEmbedding;
    std::uint32_t layer = 0;
    std::string parameter;
    std::vector<BoundPart> parts;
    std::map<std::string, std::vector<BoundPart>> auxiliaries;
    std::optional<std::string> activation_policy;
    std::optional<ExpertLibrary> library;
};

struct HcPlan {
    BoundParameter norm;
    BoundParameter down;
    BoundParameter inject;
    BoundParameter up;
};

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

struct LayerPlan {
    HcPlan attention_hc;
    HcPlan moe_hc;
    std::variant<GdnPlan, QsaPlan> mixer;
    MoePlan moe;
};

struct PlePlan {
    BoundParameter table;
    BoundParameter key;
    BoundParameter value;
    BoundParameter norm_key;
    BoundParameter norm_query;
    BoundParameter norm_conv;
    BoundParameter convolution;
};

struct UnclaimedParameters {
    std::map<std::string, std::vector<std::string>> by_namespace;
};

struct BindingPlan {
    std::vector<BoundObject> objects;
    BoundParameter token_embedding;
    BoundParameter output_head;
    HcPlan final_hc;
    std::array<LayerPlan, TextConfig::layers> layers;
    PlePlan ple;
    UnclaimedParameters unclaimed;

    [[nodiscard]] std::vector<const BoundParameter*> parameters() const;
};

[[nodiscard]] BindingPlan bind_artifact(artifact::Binder& binder, WeightsProfile profile, int tp);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
