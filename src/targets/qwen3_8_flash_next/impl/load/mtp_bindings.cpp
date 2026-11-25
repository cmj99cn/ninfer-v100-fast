#include "targets/qwen3_8_flash_next/impl/load/mtp_bindings.h"

#include "targets/qwen3_8_flash_next/impl/config.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail::mtp {
namespace {

using artifact::ArtifactError;
using artifact::ElementIntervalView;
using artifact::NumericFormat;
using artifact::StorageLayout;

constexpr std::string_view kNamespace = "mtp/layers/0/";
constexpr std::string_view kBankGate = "weight/000096";
constexpr std::string_view kBankDown = "weight/000097";

constexpr std::uint32_t kQsaRows =
    2 * TextConfig::kv_width + TextConfig::index_query_width + TextConfig::index_key_width +
    2 * TextConfig::query_width;

constexpr std::array<std::uint64_t, 3> shape2(std::uint64_t rows, std::uint64_t columns) {
    return {rows, columns, 0};
}

constexpr std::array<std::uint64_t, 3> shape3(std::uint64_t matrices, std::uint64_t rows,
                                              std::uint64_t columns) {
    return {matrices, rows, columns};
}

constexpr RoleGeometry dense(std::array<std::uint64_t, 3> shape, std::uint64_t row_begin,
                             std::uint64_t row_count, std::uint64_t column_count) {
    return RoleGeometry{NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                        shape[2] != 0 ? 3u : (shape[1] != 0 ? 2u : 1u), shape, row_begin, row_count,
                        0, column_count, 1, 1};
}

constexpr RoleGeometry vector_of(std::array<std::uint64_t, 3> shape) {
    return dense(shape, 0, shape[0], 1);
}

constexpr RoleGeometry bank(std::array<std::uint64_t, 3> shape, std::uint64_t row_begin,
                             std::uint64_t row_count) {
    return RoleGeometry{NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
                        StorageLayout::BlockScaleM128K128V1, 3, shape, row_begin, row_count, 0,
                        shape[2], shape[0], shape[0]};
}

constexpr RoleGeometry row_split_q4(std::array<std::uint64_t, 3> shape) {
    return RoleGeometry{NumericFormat::Q4G64_F16S, StorageLayout::RowSplitK128V1, 2, shape, 0,
                        shape[0], 0, shape[1], 1, 1};
}


class PlanBuilder {
public:
    PlanBuilder(artifact::Binder& binder, const artifact::V3Directory& directory, MtpPlan& plan)
        : binder_(binder), directory_(directory), plan_(plan) {}

    void adopt(const std::vector<BoundObject>& objects) {
        for (std::size_t index = 0; index < objects.size(); ++index) {
            objects_.emplace(objects[index].name, static_cast<std::uint32_t>(index));
        }
    }

    BoundParameter bind_shared(const BoundParameter& shared) {
        ++binding_count_;
        return shared;
    }

    BoundParameter bind(std::string_view name, const RoleGeometry& geometry) {
        return bind_at(std::string(kNamespace) + std::string(name), geometry);
    }

    BoundParameter bind_global(std::string_view name, const RoleGeometry& geometry) {
        return bind_at(std::string("mtp/") + std::string(name), geometry);
    }

    BoundParameter bind_artifact_parameter(std::string_view name, const RoleGeometry& geometry) {
        return bind_at(std::string(name), geometry);
    }

    BoundParameter bind_at(const std::string& parameter, const RoleGeometry& geometry) {
        const auto single = directory_.bindings.find(parameter);
        const auto parts = directory_.part_bindings.find(parameter);
        if (single == directory_.bindings.end() && parts == directory_.part_bindings.end()) {
            throw ArtifactError("qwen3.8-flash-next: the MTP artifact does not bind " + parameter);
        }
        BoundParameter result;
        result.parameter = parameter;
        if (single != directory_.bindings.end()) {
            const auto handle = bind_object(single->second, geometry);
            const BoundObject& object = plan_.objects.at(static_cast<std::size_t>(handle));
            const auto count = artifact::tensor_logical_element_count(
                object.layout, object.format, object.shape);
            BoundPart part;
            part.object = handle;
            part.view = artifact::tensor_element_interval_view(object.layout, object.format,
                                                               object.shape, 0, count);
            result.parts.push_back(part);
        } else {
            for (const auto& part : parts->second) {
                const auto handle = bind_object(part.object, geometry);
                const BoundObject& object = plan_.objects.at(static_cast<std::size_t>(handle));
                BoundPart bound;
                bound.object = handle;
                bound.view = artifact::tensor_element_interval_view(
                    object.layout, object.format, object.shape, part.begin, part.end - part.begin);
                result.parts.push_back(bound);
            }
        }
        check(result, geometry, parameter);
        bind_library(result, geometry);
        ++binding_count_;
        return result;
    }

    [[nodiscard]] std::size_t binding_count() const noexcept { return binding_count_; }

private:
    std::uint32_t bind_object(const std::string& name, const RoleGeometry& geometry) {
        const auto found = objects_.find(name);
        if (found != objects_.end()) {
            const BoundObject& object = plan_.objects.at(static_cast<std::size_t>(found->second));
            const bool same = object.format == geometry.format && object.layout == geometry.layout &&
                              object.shape.size() == geometry.rank &&
                              std::equal(object.shape.begin(), object.shape.end(),
                                         geometry.object_shape().begin());
            if (!same) {
                throw ArtifactError("qwen3.8-flash-next: the MTP object " + name +
                                    " carries two different geometries");
            }
            return found->second;
        }
        const auto handle = binder_.require_tensor(name, geometry.format, geometry.layout,
                                                   geometry.object_shape());
        BoundObject object;
        object.name = name;
        object.handle = handle;
        object.format = geometry.format;
        object.layout = geometry.layout;
        object.shape.assign(geometry.object_shape().begin(), geometry.object_shape().end());
        plan_.objects.push_back(std::move(object));
        const auto index = static_cast<std::uint32_t>(plan_.objects.size() - 1);
        objects_.emplace(name, index);
        return index;
    }

    static void check(const BoundParameter& parameter, const RoleGeometry& geometry,
                      const std::string& name) {
        if (parameter.parts.size() != geometry.part_count) {
            throw ArtifactError("qwen3.8-flash-next: the MTP parameter " + name + " holds " +
                                std::to_string(parameter.parts.size()) + " parts; it expects " +
                                std::to_string(geometry.part_count));
        }
        std::map<std::uint32_t, std::set<std::uint64_t>> matrices;
        for (const auto& part : parameter.parts) {
            const ElementIntervalView& view = part.view;
            if (view.matrix_count != 1 || view.matrix_row_begin != geometry.row_begin ||
                view.matrix_row_count != geometry.row_count ||
                view.column_begin != geometry.column_begin ||
                view.column_count != geometry.column_count) {
                throw ArtifactError("qwen3.8-flash-next: the MTP parameter " + name +
                                    " does not occupy the rectangle this package expects");
            }
            matrices[part.object].insert(view.matrix_begin);
        }
        for (const auto& [object, indices] : matrices) {
            if (indices.size() != geometry.matrix_count) {
                throw ArtifactError("qwen3.8-flash-next: the MTP parameter " + name +
                                    " covers " + std::to_string(indices.size()) +
                                    " matrices; it expects " + std::to_string(geometry.matrix_count));
            }
            std::uint64_t expected = 0;
            for (const auto index : indices) {
                if (index != expected++) {
                    throw ArtifactError("qwen3.8-flash-next: the MTP parameter " + name +
                                        " does not cover its matrices in order");
                }
            }
        }
    }

    void bind_library(BoundParameter& parameter, const RoleGeometry& geometry) {
        if (geometry.matrix_count <= 1) { return; }
        if (geometry.rank != 3 || geometry.format != NumericFormat::FP8_E4M3FN_BLOCK128_BF16S) {
            throw ArtifactError("qwen3.8-flash-next: the MTP expert bank has an unexpected layout");
        }
        const auto& first = plan_.objects.at(static_cast<std::size_t>(parameter.parts.front().object));
        ExpertLibrary library;
        library.object = first.name;
        library.handle = first.handle;
        library.matrices = geometry.shape[0];
        library.rows = geometry.shape[1];
        library.columns = geometry.shape[2];
        library.row_begin = geometry.row_begin;
        library.row_count = geometry.row_count;
        parameter.library = std::move(library);
    }

    artifact::Binder& binder_;
    const artifact::V3Directory& directory_;
    MtpPlan& plan_;
    std::map<std::string, std::uint32_t> objects_;
    std::size_t binding_count_ = 0;
};

} // namespace

std::vector<const BoundParameter*> MtpPlan::parameters() const {
    std::vector<const BoundParameter*> out;
    out.push_back(&token_embedding);
    out.push_back(&output_head);
    out.push_back(&proposal_head);
    out.push_back(&fc_embedding);
    out.push_back(&fc_hidden);
    out.push_back(&pre_fc_embedding_norm);
    out.push_back(&pre_fc_hidden_norm);
    out.push_back(&final_hc.norm);
    out.push_back(&final_hc.down);
    out.push_back(&final_hc.up);
    out.push_back(&layer.attention_hc.norm);
    out.push_back(&layer.attention_hc.down);
    out.push_back(&layer.attention_hc.inject);
    out.push_back(&layer.attention_hc.up);
    out.push_back(&layer.moe_hc.norm);
    out.push_back(&layer.moe_hc.down);
    out.push_back(&layer.moe_hc.inject);
    out.push_back(&layer.moe_hc.up);
    const auto& qsa = std::get<QsaPlan>(layer.mixer);
    out.push_back(&qsa.query);
    out.push_back(&qsa.key);
    out.push_back(&qsa.value);
    out.push_back(&qsa.index_query);
    out.push_back(&qsa.index_key);
    out.push_back(&qsa.output);
    out.push_back(&qsa.output_gate);
    out.push_back(&qsa.query_norm);
    out.push_back(&qsa.key_norm);
    out.push_back(&qsa.index_query_norm);
    out.push_back(&qsa.index_key_norm);
    out.push_back(&layer.moe.router);
    out.push_back(&layer.moe.shared_score);
    out.push_back(&layer.moe.shared_gate);
    out.push_back(&layer.moe.shared_up);
    out.push_back(&layer.moe.shared_down);
    out.push_back(&layer.moe.routed_gate);
    out.push_back(&layer.moe.routed_up);
    out.push_back(&layer.moe.routed_down);
    return out;
}

MtpPlan bind_mtp_artifact(artifact::Binder& binder, const BindingPlan& text) {
    const artifact::V3Directory& directory = binder.directory();
    const auto mtp_config = directory.component_config.find("mtp");
    if (mtp_config == directory.component_config.end()) {
        throw ArtifactError("qwen3.8-flash-next: the artifact carries no MTP configuration");
    }
    const auto layer_types = mtp_config->second.find("layer_types");
    if (layer_types == mtp_config->second.end() || layer_types->second != "indexed_attention") {
        throw ArtifactError("qwen3.8-flash-next: the MTP layer is not an indexed-attention layer");
    }

    MtpPlan plan;
    plan.objects = text.objects;
    PlanBuilder builder(binder, directory, plan);
    builder.adopt(text.objects);

    plan.token_embedding = builder.bind_shared(text.token_embedding);
    plan.output_head = builder.bind_shared(text.output_head);
    plan.proposal_head = builder.bind_artifact_parameter(
        "proposal/head", row_split_q4(shape2(TextConfig::vocab, TextConfig::hidden)));
    plan.fc_embedding = builder.bind_global("fc_embedding",
                                     dense(shape2(TextConfig::hidden, TextConfig::hidden), 0,
                                           TextConfig::hidden, TextConfig::hidden));
    plan.fc_hidden = builder.bind_global("fc_hidden",
                                  dense(shape2(TextConfig::hidden, TextConfig::hidden), 0,
                                        TextConfig::hidden, TextConfig::hidden));
    plan.pre_fc_embedding_norm = builder.bind_global(
        "pre_fc_embedding_norm", vector_of(shape2(TextConfig::hidden, 0)));
    plan.pre_fc_hidden_norm = builder.bind_global(
        "pre_fc_hidden_norm", vector_of(shape2(TextConfig::residual_width, 0)));
    plan.final_hc.norm = builder.bind_global(
        "final_hc/norm", vector_of(shape2(TextConfig::residual_width, 0)));
    plan.final_hc.down = builder.bind_global(
        "final_hc/down", dense(shape2(TextConfig::hc_lowrank, TextConfig::residual_width), 0,
                               TextConfig::hc_lowrank, TextConfig::residual_width));
    plan.final_hc.up = builder.bind_global(
        "final_hc/up", dense(shape2(TextConfig::residual_width, TextConfig::hc_lowrank), 0,
                             TextConfig::residual_width, TextConfig::hc_lowrank));

    LayerPlan& layer = plan.layer;
    layer.attention_hc.norm = builder.bind(
        "attention_hc/norm", vector_of(shape2(TextConfig::residual_width, 0)));
    layer.attention_hc.down = builder.bind(
        "attention_hc/down",
        dense(shape2(TextConfig::hc_lowrank + TextConfig::hc_streams, TextConfig::residual_width),
              0, TextConfig::hc_lowrank, TextConfig::residual_width));
    layer.attention_hc.inject = builder.bind(
        "attention_hc/inject",
        dense(shape2(TextConfig::hc_lowrank + TextConfig::hc_streams, TextConfig::residual_width),
              TextConfig::hc_lowrank, TextConfig::hc_streams, TextConfig::residual_width));
    layer.attention_hc.up = builder.bind(
        "attention_hc/up",
        dense(shape2(TextConfig::residual_width, TextConfig::hc_lowrank), 0,
              TextConfig::residual_width, TextConfig::hc_lowrank));
    layer.moe_hc.norm = builder.bind("moe_hc/norm",
                                     vector_of(shape2(TextConfig::residual_width, 0)));
    layer.moe_hc.down = builder.bind(
        "moe_hc/down",
        dense(shape2(TextConfig::hc_lowrank + TextConfig::hc_streams, TextConfig::residual_width),
              0, TextConfig::hc_lowrank, TextConfig::residual_width));
    layer.moe_hc.inject = builder.bind(
        "moe_hc/inject",
        dense(shape2(TextConfig::hc_lowrank + TextConfig::hc_streams, TextConfig::residual_width),
              TextConfig::hc_lowrank, TextConfig::hc_streams, TextConfig::residual_width));
    layer.moe_hc.up = builder.bind(
        "moe_hc/up",
        dense(shape2(TextConfig::residual_width, TextConfig::hc_lowrank), 0,
              TextConfig::residual_width, TextConfig::hc_lowrank));

    QsaPlan qsa;
    qsa.key = builder.bind("attention/key",
                           dense(shape2(kQsaRows, TextConfig::hidden), 0, TextConfig::kv_width,
                                 TextConfig::hidden));
    qsa.value = builder.bind("attention/value",
                             dense(shape2(kQsaRows, TextConfig::hidden), TextConfig::kv_width,
                                   TextConfig::kv_width, TextConfig::hidden));
    qsa.index_key = builder.bind(
        "attention/index_key",
        dense(shape2(kQsaRows, TextConfig::hidden), 2 * TextConfig::kv_width,
              TextConfig::index_key_width, TextConfig::hidden));
    qsa.query = builder.bind(
        "attention/query",
        dense(shape2(kQsaRows, TextConfig::hidden), 2 * TextConfig::kv_width + TextConfig::index_key_width,
              TextConfig::query_width, TextConfig::hidden));
    qsa.output_gate = builder.bind(
        "attention/output_gate",
        dense(shape2(kQsaRows, TextConfig::hidden),
              2 * TextConfig::kv_width + TextConfig::index_key_width + TextConfig::query_width,
              TextConfig::query_width, TextConfig::hidden));
    qsa.index_query = builder.bind(
        "attention/index_query",
        dense(shape2(kQsaRows, TextConfig::hidden),
              2 * TextConfig::kv_width + TextConfig::index_key_width + 2 * TextConfig::query_width,
              TextConfig::index_query_width, TextConfig::hidden));
    qsa.output = builder.bind("attention/output",
                              dense(shape2(TextConfig::hidden, TextConfig::query_width), 0,
                                    TextConfig::hidden, TextConfig::query_width));
    qsa.query_norm = builder.bind("attention/query_norm",
                                  vector_of(shape2(TextConfig::head_dim, 0)));
    qsa.key_norm = builder.bind("attention/key_norm", vector_of(shape2(TextConfig::head_dim, 0)));
    qsa.index_query_norm = builder.bind(
        "attention/index_query_norm", vector_of(shape2(TextConfig::index_head_dim, 0)));
    qsa.index_key_norm = builder.bind(
        "attention/index_key_norm", vector_of(shape2(TextConfig::index_head_dim, 0)));
    layer.mixer = std::move(qsa);

    constexpr std::uint32_t kMoeControlRows =
        TextConfig::experts + 1 + 2 * TextConfig::shared_intermediate;
    layer.moe.router = builder.bind("moe/router",
                                    dense(shape2(kMoeControlRows, TextConfig::hidden), 0,
                                          TextConfig::experts, TextConfig::hidden));
    layer.moe.shared_score = builder.bind(
        "moe/shared_score",
        dense(shape2(kMoeControlRows, TextConfig::hidden), TextConfig::experts, 1,
              TextConfig::hidden));
    layer.moe.shared_gate = builder.bind(
        "moe/shared/gate",
        dense(shape2(kMoeControlRows, TextConfig::hidden), TextConfig::experts + 1,
              TextConfig::shared_intermediate, TextConfig::hidden));
    layer.moe.shared_up = builder.bind(
        "moe/shared/up",
        dense(shape2(kMoeControlRows, TextConfig::hidden),
              TextConfig::experts + 1 + TextConfig::shared_intermediate,
              TextConfig::shared_intermediate, TextConfig::hidden));
    layer.moe.shared_down = builder.bind(
        "moe/shared/down",
        dense(shape2(TextConfig::hidden, TextConfig::shared_intermediate), 0, TextConfig::hidden,
              TextConfig::shared_intermediate));
    layer.moe.routed_gate = builder.bind(
        "moe/experts/gate",
        bank(shape3(TextConfig::experts, 2 * TextConfig::expert_intermediate, TextConfig::hidden), 0,
             TextConfig::expert_intermediate));
    layer.moe.routed_up = builder.bind(
        "moe/experts/up",
        bank(shape3(TextConfig::experts, 2 * TextConfig::expert_intermediate, TextConfig::hidden),
             TextConfig::expert_intermediate, TextConfig::expert_intermediate));
    layer.moe.routed_down = builder.bind(
        "moe/experts/down",
        bank(shape3(TextConfig::experts, TextConfig::hidden, TextConfig::expert_intermediate), 0,
             TextConfig::hidden));

    plan.binding_count = builder.binding_count();
    if (plan.binding_count != 37) {
        throw ArtifactError("qwen3.8-flash-next: the MTP plan parsed " +
                            std::to_string(plan.binding_count) + " bindings; it expects 37");
    }
    if (layer.moe.routed_gate.library->object != kBankGate ||
        layer.moe.routed_up.library->object != kBankGate ||
        layer.moe.routed_down.library->object != kBankDown) {
        throw ArtifactError("qwen3.8-flash-next: the MTP expert bank objects are not 000096/000097");
    }
    return plan;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail::mtp
