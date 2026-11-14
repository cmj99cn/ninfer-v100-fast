#pragma once

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "ninfer/ops/ple.h"
#include "targets/qwen3_8_flash_next/impl/config.h"
#include "targets/qwen3_8_flash_next/impl/load/geometry.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"
#include "targets/qwen3_8_flash_next/impl/load/sharding.h"
#include "targets/qwen3_8_flash_next/impl/variant.h"
#include <ninfer/targets/qwen4_exp/model_view.h>
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <cstdint>
#include <functional>
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
    std::vector<std::string> object_names;
};

struct BindingPlan {
    std::vector<BoundObject> objects;
    BoundParameter token_embedding;
    BoundParameter output_head;
    HcPlan final_hc;
    std::array<LayerPlan, TextConfig::layers> layers;
    PlePlan ple;
    artifact::ObjectHandle ple_controls;
    UnclaimedParameters unclaimed;

    [[nodiscard]] std::vector<const BoundParameter*> parameters() const;
};

[[nodiscard]] BindingPlan bind_artifact(artifact::Binder& binder, WeightsProfile profile, int tp);

// The shard map of every text role at one tensor-parallel width, keyed by role name.
using ShardPlanByName = std::map<std::string, sharding::ShardMapping, std::less<>>;

// What one object's shard holds on each rank: the ranges the role map gives, in the order they
// are concatenated, as parent-payload plane copies. Every object the plan binds appears once.
struct RoleBlock {
    roles::Role role;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

struct ObjectShardPlan {
    std::string object;
    sharding::SplitAxis split = sharding::SplitAxis::Output;
    std::vector<roles::Role> roles;
    std::array<std::vector<artifact::SliceRange>, kMaximumDevices> ranges;
    std::array<std::vector<artifact::PlaneCopy>, kMaximumDevices> copies;
    std::array<std::vector<std::byte>, kMaximumDevices> prefix;
    std::array<std::vector<RoleBlock>, kMaximumDevices> blocks;
    std::array<std::uint64_t, kMaximumDevices> encoded_bytes{};
    std::array<std::uint64_t, kMaximumDevices> scale_plane_offset{};
    std::array<std::uint64_t, kMaximumDevices> divisor_plane_offset{};
};

[[nodiscard]] ShardPlanByName shard_plan_by_name(int tp);

[[nodiscard]] std::vector<ObjectShardPlan> shard_byte_plan(const BindingPlan& plan,
                                                           const ShardPlanByName& shards, int tp);

// The per-object placement the binder resolves from the shard map.
[[nodiscard]] std::map<std::string, artifact::ShardPlacement> shard_placements(
    const BindingPlan& plan, const ShardPlanByName& shards, int tp);

// The per-rank weight views. Ranks outside [0, tp) stay empty.
struct RankViews {
    std::array<Variant::ModelView, kMaximumDevices> views;
};

[[nodiscard]] RankViews shard_views(const BindingPlan& plan,
                                    const artifact::MaterializedArtifact& materialized,
                                    const ShardPlanByName& shards, int tp);

class PleTableSource {
public:
    PleTableSource(const std::filesystem::path& path, const BindingPlan& plan);

    void fetch(const ops::PleRowAddress* addresses, std::size_t count,
               std::byte* destination) const;

    [[nodiscard]] const float* divisors() const noexcept { return divisors_.data(); }
    [[nodiscard]] const ops::PleHashControls& controls() const noexcept { return controls_; }

private:
    artifact::Reader reader_;
    std::array<std::uint64_t, ops::kPleShards> shard_offsets_{};
    artifact::RowPagedNf4Geometry geometry_{};
    std::array<float, ops::kPleShards> divisors_{};
    ops::PleHashControls controls_{};
};

void fetch_ple_rows(void* context, const ops::PleRowAddress* addresses, std::size_t count,
                    std::byte* destination);

class LoadedModelData {
public:
    LoadedModelData(const BindingPlan& plan, const artifact::MaterializedArtifact& materialized,
                    std::size_t loaded_layers, int tp, const std::filesystem::path& artifact_path);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = default;
    LoadedModelData& operator=(LoadedModelData&&)      = default;

    [[nodiscard]] const Variant::ModelView& view(int rank) const noexcept { return views_.views[static_cast<std::size_t>(rank)]; }

private:
    std::unique_ptr<PleTableSource> ple_source_;
    RankViews views_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
