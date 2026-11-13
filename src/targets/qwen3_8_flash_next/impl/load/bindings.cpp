#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <cstddef>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

using artifact::ArtifactError;
using artifact::NumericFormat;
using artifact::StorageLayout;

constexpr std::string_view kTextNamespace   = "text";
constexpr std::string_view kActivationInput = "activation_input_divisor";

const std::map<std::string, std::string>& text_config(const artifact::V3Directory& directory) {
    const auto found = directory.component_config.find(std::string(kTextNamespace));
    if (found == directory.component_config.end()) {
        throw ArtifactError("qwen3.8-flash-next: the artifact carries no text configuration");
    }
    return found->second;
}

std::string_view config_value(const std::map<std::string, std::string>& config,
                               std::string_view key) {
    const auto found = config.find(std::string(key));
    if (found == config.end()) {
        throw ArtifactError("qwen3.8-flash-next: the artifact's text configuration has no '" +
                            std::string(key) + "'");
    }
    return found->second;
}

long long parse_integer(std::string_view text, std::string_view key) {
    long long value       = 0;
    const auto* const end = text.data() + text.size();
    const auto result     = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw ArtifactError("qwen3.8-flash-next: the artifact's text configuration key '" +
                            std::string(key) + "' is not an integer: " + std::string(text));
    }
    return value;
}

double parse_number(std::string_view text, std::string_view key) {
    const std::string owned(text);
    char* end         = nullptr;
    const double value = std::strtod(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size()) {
        throw ArtifactError("qwen3.8-flash-next: the artifact's text configuration key '" +
                            std::string(key) + "' is not a number: " + owned);
    }
    return value;
}

void require_integer(const std::map<std::string, std::string>& config, std::string_view key,
                      long long expected) {
    const auto value = parse_integer(config_value(config, key), key);
    if (value != expected) {
        throw ArtifactError("qwen3.8-flash-next: the artifact's text configuration key '" +
                            std::string(key) + "' is " + std::to_string(value) + "; this package "
                            "expects " + std::to_string(expected));
    }
}

void require_number(const std::map<std::string, std::string>& config, std::string_view key,
                    double expected) {
    const auto value = parse_number(config_value(config, key), key);
    const double scale = std::max(1.0, std::abs(expected));
    if (std::abs(value - expected) > 1.0e-9 * scale) {
        throw ArtifactError("qwen3.8-flash-next: the artifact's text configuration key '" +
                            std::string(key) + "' is " + std::to_string(value) + "; this package "
                            "expects " + std::to_string(expected));
    }
}

void require_text(const std::map<std::string, std::string>& config, std::string_view key,
                  std::string_view expected) {
    const auto value = config_value(config, key);
    if (value != expected) {
        throw ArtifactError("qwen3.8-flash-next: the artifact's text configuration key '" +
                            std::string(key) + "' is '" + std::string(value) + "'; this package "
                            "expects '" + std::string(expected) + "'");
    }
}

std::vector<std::string_view> split(std::string_view text, char separator) {
    std::vector<std::string_view> parts;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = text.find(separator, begin);
        if (end == std::string_view::npos) {
            parts.push_back(text.substr(begin));
            break;
        }
        parts.push_back(text.substr(begin, end - begin));
        begin = end + 1;
    }
    return parts;
}

void validate_layer_types(const std::map<std::string, std::string>& config) {
    const auto kinds = split(config_value(config, "layer_types"), ',');
    if (kinds.size() != static_cast<std::size_t>(TextConfig::layers)) {
        throw ArtifactError("qwen3.8-flash-next: the artifact declares " +
                            std::to_string(kinds.size()) + " layer kinds; this package expects " +
                            std::to_string(TextConfig::layers));
    }
    for (std::size_t layer = 0; layer < kinds.size(); ++layer) {
        const std::string_view expected =
            TextConfig::is_qsa_layer(static_cast<int>(layer)) ? "indexed_attention"
                                                               : "linear_attention";
        if (kinds[layer] != expected) {
            throw ArtifactError("qwen3.8-flash-next: the artifact's layer " +
                                std::to_string(layer) + " is '" + std::string(kinds[layer]) +
                                "'; this package expects '" + std::string(expected) + "'");
        }
    }
}

void validate_ple_shard_shape(const std::map<std::string, std::string>& config) {
    const auto shape = split(config_value(config, "ple_shard_shape"), ',');
    if (shape.size() != 2) {
        throw ArtifactError("qwen3.8-flash-next: the artifact declares " +
                            std::to_string(shape.size()) + " PLE shard dimensions; it declares 2");
    }
    const auto rows = parse_integer(shape[0], "ple_shard_shape");
    const auto columns = parse_integer(shape[1], "ple_shard_shape");
    if (rows != TextConfig::ple_shard_rows || columns != TextConfig::ple_head_width) {
        throw ArtifactError("qwen3.8-flash-next: the artifact's PLE shard is " +
                            std::to_string(rows) + "x" + std::to_string(columns) +
                            "; this package expects " + std::to_string(TextConfig::ple_shard_rows) +
                            "x" + std::to_string(TextConfig::ple_head_width));
    }
}

void validate_mrope_sections(const std::map<std::string, std::string>& config) {
    const auto sections = split(config_value(config, "rope_parameters.mrope_section"), ',');
    if (sections.size() != TextConfig::rope_sections.size()) {
        throw ArtifactError("qwen3.8-flash-next: the artifact declares " +
                            std::to_string(sections.size()) +
                            " rotary sections; this package expects " +
                            std::to_string(TextConfig::rope_sections.size()));
    }
    for (std::size_t index = 0; index < sections.size(); ++index) {
        const auto value = parse_integer(sections[index], "rope_parameters.mrope_section");
        if (value != TextConfig::rope_sections[index]) {
            throw ArtifactError("qwen3.8-flash-next: the artifact's rotary section " +
                                std::to_string(index) + " is " + std::to_string(value) +
                                "; this package expects " +
                                std::to_string(TextConfig::rope_sections[index]));
        }
    }
}

void validate_text_config(const artifact::V3Directory& directory) {
    const auto& config = text_config(directory);
    require_integer(config, "hidden_size", TextConfig::hidden);
    require_integer(config, "vocab_size", TextConfig::vocab);
    require_integer(config, "num_hidden_layers", TextConfig::layers);
    require_integer(config, "max_position_embeddings", TextConfig::max_position_embeddings);
    require_integer(config, "hc_count", TextConfig::hc_streams);
    require_integer(config, "hc_lowrank", TextConfig::hc_lowrank);
    require_integer(config, "num_attention_heads", TextConfig::query_heads);
    require_integer(config, "num_key_value_heads", TextConfig::kv_heads);
    require_integer(config, "head_dim", TextConfig::head_dim);
    require_integer(config, "linear_num_key_heads", TextConfig::gdn_key_heads);
    require_integer(config, "linear_num_value_heads", TextConfig::gdn_value_heads);
    require_integer(config, "linear_key_head_dim", TextConfig::gdn_head_dim);
    require_integer(config, "linear_value_head_dim", TextConfig::gdn_head_dim);
    require_integer(config, "linear_conv_kernel_dim", TextConfig::gdn_conv_kernel);
    require_integer(config, "indexer_n_heads", TextConfig::index_heads);
    require_integer(config, "indexer_kv_heads", TextConfig::index_kv_heads);
    require_integer(config, "indexer_head_dim", TextConfig::index_head_dim);
    require_integer(config, "indexer_compress_ratio", TextConfig::index_compress);
    require_integer(config, "indexer_budget", TextConfig::index_budget);
    require_integer(config, "num_experts", TextConfig::experts);
    require_integer(config, "num_experts_per_tok", TextConfig::experts_per_token);
    require_integer(config, "moe_intermediate_size", TextConfig::expert_intermediate);
    require_integer(config, "shared_expert_intermediate_size", TextConfig::shared_intermediate);
    require_integer(config, "ple_embed_dim", TextConfig::ple_embed_dim);
    require_integer(config, "ngram_size", TextConfig::ple_ngram);
    require_integer(config, "heads_per_ngram", TextConfig::ple_heads_per_ngram);
    require_integer(config, "split_ngram_parts", TextConfig::ple_split_parts);
    require_integer(config, "ple_conv_kernel_size", TextConfig::ple_conv_kernel);
    require_integer(config, "ple_conv_dilation", TextConfig::ple_conv_dilation);
    require_integer(config, "eos_token_id", TextConfig::eos_token_id);
    require_integer(config, "image_token_id", TextConfig::image_token_id);
    require_integer(config, "video_token_id", TextConfig::video_token_id);
    require_integer(config, "vision_start_token_id", TextConfig::vision_start_token_id);
    require_integer(config, "vision_end_token_id", TextConfig::vision_end_token_id);
    require_integer(config, "ple_layer_ids", roles::kPleLayer);
    require_number(config, "rms_norm_eps", TextConfig::rms_epsilon);
    require_number(config, "rope_parameters.rope_theta", TextConfig::rope_theta);
    require_number(config, "rope_parameters.partial_rotary_factor", TextConfig::partial_rotary);
    require_text(config, "norm_topk_prob", "true");
    require_text(config, "hidden_act", "silu");
    require_text(config, "output_gate_type", "sigmoid");
    require_text(config, "tie_word_embeddings", "false");
    require_text(config, "attention_bias", "false");
    require_text(config, "mamba_ssm_dtype", "float32");
    require_text(config, "architectures", "Qwen4ExpForCausalLM");
    require_text(config, "model_type", "qwen4_exp_text");
    require_text(config, "rope_parameters.mrope_interleaved", "true");
    require_text(config, "rope_parameters.rope_type", "default");
    validate_ple_shard_shape(config);
    validate_mrope_sections(config);
    validate_layer_types(config);
}

std::optional<std::uint64_t> activation_divisor_column(roles::Role role) {
    switch (role) {
    case roles::Role::MoeRoutedGate:
    case roles::Role::MoeRoutedUp:
        return 0;
    case roles::Role::MoeRoutedDown:
        return 1;
    default:
        return std::nullopt;
    }
}

class PlanBuilder {
public:
    PlanBuilder(artifact::Binder& binder, const artifact::V3Directory& directory, BindingPlan& plan)
        : binder_(binder), directory_(directory), plan_(plan) {
        for (const auto& use : directory.uses) { uses_.emplace(use.parameter, &use); }
    }

    BoundParameter bind(roles::Role role, std::uint32_t layer) {
        BoundParameter parameter;
        parameter.role      = role;
        parameter.layer     = layer;
        parameter.parameter = roles::parameter_name(role, layer);
        const RoleGeometry geometry = role_geometry(role);

        const auto single = directory_.bindings.find(parameter.parameter);
        const auto parts  = directory_.part_bindings.find(parameter.parameter);
        if (single == directory_.bindings.end() && parts == directory_.part_bindings.end()) {
            throw ArtifactError("qwen3.8-flash-next: the artifact does not bind the parameter " +
                                parameter.parameter);
        }
        if (single != directory_.bindings.end()) {
            const auto object = bind_object(single->second, geometry);
            const BoundObject& descriptor = plan_.objects[object];
            const auto count = artifact::tensor_logical_element_count(
                descriptor.layout, descriptor.format, descriptor.shape);
            BoundPart part;
            part.object = object;
            part.view   = artifact::tensor_element_interval_view(descriptor.layout, descriptor.format,
                                                               descriptor.shape, 0, count);
            parameter.parts.push_back(part);
        } else {
            for (const auto& part : parts->second) {
                const auto object = bind_object(part.object, geometry);
                const BoundObject& descriptor = plan_.objects[object];
                BoundPart bound;
                bound.object = object;
                bound.view   = artifact::tensor_element_interval_view(
                    descriptor.layout, descriptor.format, descriptor.shape, part.begin,
                    part.end - part.begin);
                parameter.parts.push_back(bound);
            }
        }
        check_rectangle(parameter, geometry);
        check_matrix_coverage(parameter, geometry);
        bind_use(parameter);
        bind_library(parameter, geometry);
        claimed_.insert(parameter.parameter);
        return parameter;
    }

    [[nodiscard]] const std::set<std::string>& claimed() const noexcept { return claimed_; }

private:
    std::uint32_t bind_object(const std::string& name, const RoleGeometry& geometry) {
        const auto found = objects_.find(name);
        if (found != objects_.end()) {
            const BoundObject& object = plan_.objects[found->second];
            const bool same = object.format == geometry.format && object.layout == geometry.layout &&
                              object.shape.size() == geometry.rank &&
                              std::equal(object.shape.begin(), object.shape.end(),
                                         geometry.object_shape().begin());
            if (!same) {
                throw ArtifactError("qwen3.8-flash-next: the object " + name +
                                    " carries two different geometries");
            }
            return found->second;
        }
        const auto handle = binder_.require_tensor(name, geometry.format, geometry.layout,
                                                   geometry.object_shape());
        const auto* tensor = std::get_if<artifact::TensorDescriptor>(&binder_.descriptor(handle));
        if (tensor == nullptr || tensor->shape.size() != geometry.rank) {
            throw ArtifactError("qwen3.8-flash-next: the object " + name +
                                " does not have the rank this package expects");
        }
        BoundObject object;
        object.name   = name;
        object.handle = handle;
        object.format = geometry.format;
        object.layout = geometry.layout;
        object.shape.assign(geometry.object_shape().begin(), geometry.object_shape().end());
        plan_.objects.push_back(std::move(object));
        const auto index = static_cast<std::uint32_t>(plan_.objects.size() - 1);
        objects_.emplace(name, index);
        return index;
    }

    void check_rectangle(const BoundParameter& parameter, const RoleGeometry& geometry) const {
        if (parameter.parts.size() != geometry.part_count) {
            throw ArtifactError("qwen3.8-flash-next: the parameter " + parameter.parameter +
                                " holds " + std::to_string(parameter.parts.size()) +
                                " parts; this package expects " +
                                std::to_string(geometry.part_count));
        }
        for (const auto& part : parameter.parts) {
            if (part.view.matrix_count != 1 ||
                part.view.matrix_row_begin != geometry.row_begin ||
                part.view.matrix_row_count != geometry.row_count ||
                part.view.column_begin != geometry.column_begin ||
                part.view.column_count != geometry.column_count) {
                throw ArtifactError("qwen3.8-flash-next: the parameter " + parameter.parameter +
                                    " does not occupy the rows of its parent this package expects");
            }
        }
    }

    void check_matrix_coverage(const BoundParameter& parameter, const RoleGeometry& geometry) const {
        std::map<std::uint32_t, std::set<std::uint64_t>> matrices;
        for (const auto& part : parameter.parts) { matrices[part.object].insert(part.view.matrix_begin); }
        for (const auto& [object, indices] : matrices) {
            if (indices.size() != geometry.matrix_count) {
                throw ArtifactError("qwen3.8-flash-next: the parameter " + parameter.parameter +
                                    " covers " + std::to_string(indices.size()) +
                                    " matrices of one object; this package expects " +
                                    std::to_string(geometry.matrix_count));
            }
            std::uint64_t expected = 0;
            for (const auto index : indices) {
                if (index != expected) {
                    throw ArtifactError("qwen3.8-flash-next: the parameter " + parameter.parameter +
                                        " does not cover the matrices of its object in order");
                }
                ++expected;
            }
        }
    }

    void bind_use(BoundParameter& parameter) {
        const auto use = uses_.find(parameter.parameter);
        if (use == uses_.end()) { return; }
        parameter.activation_policy = use->second->activation_policy;
        for (const auto& [name, parts] : use->second->auxiliaries) {
            if (name != kActivationInput) {
                throw ArtifactError("qwen3.8-flash-next: the use edge of " + parameter.parameter +
                                    " carries the unknown auxiliary " + name);
            }
            const auto column = activation_divisor_column(parameter.role);
            if (!column.has_value()) {
                throw ArtifactError("qwen3.8-flash-next: the use edge of " + parameter.parameter +
                                    " carries an activation input divisor this package does not "
                                    "expect");
            }
            std::vector<BoundPart> bound;
            for (const auto& part : parts) {
                const RoleGeometry geometry{NumericFormat::FP32,
                                            StorageLayout::ContiguousLeV1,
                                            2,
                                            {static_cast<std::uint64_t>(TextConfig::layers), 2, 0},
                                            parameter.layer,
                                            1,
                                            *column,
                                            1,
                                            1};
                const auto object = bind_object(part.object, geometry);
                const BoundObject& descriptor = plan_.objects[object];
                BoundPart auxiliary;
                auxiliary.object = object;
                auxiliary.view   = artifact::tensor_element_interval_view(
                    descriptor.layout, descriptor.format, descriptor.shape, part.begin,
                    part.end - part.begin);
                if (auxiliary.view.matrix_row_begin != parameter.layer ||
                    auxiliary.view.matrix_row_count != 1 ||
                    auxiliary.view.column_begin != *column ||
                    auxiliary.view.column_count != 1) {
                    throw ArtifactError("qwen3.8-flash-next: the activation input divisor of " +
                                        parameter.parameter +
                                        " is not the divisor this package expects");
                }
                bound.push_back(auxiliary);
            }
            parameter.auxiliaries.emplace(name, std::move(bound));
        }
    }

    void bind_library(BoundParameter& parameter, const RoleGeometry& geometry) {
        if (geometry.matrix_count <= 1) { return; }
        const auto& part = parameter.parts.front();
        const BoundObject& object = plan_.objects.at(part.object);
        ExpertLibrary library;
        library.object     = object.name;
        library.handle     = object.handle;
        library.matrices   = geometry.shape[0];
        library.rows       = geometry.shape[1];
        library.columns    = geometry.shape[2];
        library.row_begin  = geometry.row_begin;
        library.row_count  = geometry.row_count;
        parameter.library  = std::move(library);
    }

    artifact::Binder& binder_;
    const artifact::V3Directory& directory_;
    BindingPlan& plan_;
    std::map<std::string, const artifact::V3Use*> uses_;
    std::map<std::string, std::uint32_t> objects_;
    std::set<std::string> claimed_;
};

template <typename Bindings>
void collect_unclaimed(const Bindings& bindings, const std::set<std::string>& claimed,
                       UnclaimedParameters& unclaimed) {
    for (const auto& entry : bindings) {
        const std::string& name = entry.first;
        if (claimed.count(name) != 0) { continue; }
        const auto slash = name.find('/');
        const std::string space = slash == std::string::npos ? name : name.substr(0, slash);
        if (space == kTextNamespace) {
            throw ArtifactError("qwen3.8-flash-next: the artifact holds the text parameter " + name +
                                " that this package does not claim");
        }
        unclaimed.by_namespace[space].push_back(name);
    }
}

template <typename Bindings>
void collect_unclaimed_objects(const Bindings& bindings, const std::set<std::string>& claimed,
                               UnclaimedParameters& unclaimed) {
    for (const auto& entry : bindings) {
        const std::string& name = entry.first;
        if (claimed.count(name) != 0) { continue; }
        unclaimed.object_names.push_back(entry.second);
    }
}

template <typename Bindings>
void collect_unclaimed_part_objects(const Bindings& bindings, const std::set<std::string>& claimed,
                                    UnclaimedParameters& unclaimed) {
    for (const auto& entry : bindings) {
        const std::string& name = entry.first;
        if (claimed.count(name) != 0) { continue; }
        for (const auto& part : entry.second) {
            unclaimed.object_names.push_back(part.object);
        }
    }
}

} // namespace

std::vector<const BoundParameter*> BindingPlan::parameters() const {
    std::vector<const BoundParameter*> out;
    out.reserve(roles::kTextNameCount);
    out.push_back(&token_embedding);
    out.push_back(&output_head);
    out.push_back(&final_hc.norm);
    out.push_back(&final_hc.down);
    out.push_back(&final_hc.up);
    for (const auto& layer : layers) {
        out.push_back(&layer.attention_hc.norm);
        out.push_back(&layer.attention_hc.down);
        out.push_back(&layer.attention_hc.inject);
        out.push_back(&layer.attention_hc.up);
        out.push_back(&layer.moe_hc.norm);
        out.push_back(&layer.moe_hc.down);
        out.push_back(&layer.moe_hc.inject);
        out.push_back(&layer.moe_hc.up);
        if (const auto* gdn = std::get_if<GdnPlan>(&layer.mixer)) {
            out.push_back(&gdn->query);
            out.push_back(&gdn->key);
            out.push_back(&gdn->value);
            out.push_back(&gdn->z);
            out.push_back(&gdn->a_projection);
            out.push_back(&gdn->b_projection);
            out.push_back(&gdn->a_log);
            out.push_back(&gdn->dt_bias);
            out.push_back(&gdn->convolution);
            out.push_back(&gdn->norm);
            out.push_back(&gdn->output);
        } else {
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
        }
        out.push_back(&layer.moe.router);
        out.push_back(&layer.moe.shared_score);
        out.push_back(&layer.moe.shared_gate);
        out.push_back(&layer.moe.shared_up);
        out.push_back(&layer.moe.shared_down);
        out.push_back(&layer.moe.routed_gate);
        out.push_back(&layer.moe.routed_up);
        out.push_back(&layer.moe.routed_down);
    }
    out.push_back(&ple.table);
    out.push_back(&ple.key);
    out.push_back(&ple.value);
    out.push_back(&ple.norm_key);
    out.push_back(&ple.norm_query);
    out.push_back(&ple.norm_conv);
    out.push_back(&ple.convolution);
    return out;
}

BindingPlan bind_artifact(artifact::Binder& binder, WeightsProfile profile, int tp) {
    if (profile != WeightsProfile::Nvfp4) {
        throw std::invalid_argument("qwen3.8-flash-next binds the nvfp4 weights profile only");
    }
    if (tp != 1) {
        throw std::invalid_argument("qwen3.8-flash-next binds one device at this milestone; the "
                                    "tensor-parallel plan belongs to the TP2 milestone");
    }
    const artifact::V3Directory& directory = binder.directory();
    validate_text_config(directory);

    BindingPlan plan;
    PlanBuilder builder(binder, directory, plan);

    plan.token_embedding      = builder.bind(roles::Role::TokenEmbedding, 0);
    plan.output_head          = builder.bind(roles::Role::OutputHead, 0);
    plan.final_hc.norm        = builder.bind(roles::Role::FinalHcNorm, 0);
    plan.final_hc.down        = builder.bind(roles::Role::FinalHcDown, 0);
    plan.final_hc.up          = builder.bind(roles::Role::FinalHcUp, 0);

    for (std::uint32_t layer = 0; layer < static_cast<std::uint32_t>(TextConfig::layers); ++layer) {
        LayerPlan& layer_plan = plan.layers[layer];
        layer_plan.attention_hc.norm   = builder.bind(roles::Role::AttentionHcNorm, layer);
        layer_plan.attention_hc.down   = builder.bind(roles::Role::AttentionHcDown, layer);
        layer_plan.attention_hc.inject = builder.bind(roles::Role::AttentionHcInject, layer);
        layer_plan.attention_hc.up     = builder.bind(roles::Role::AttentionHcUp, layer);
        layer_plan.moe_hc.norm         = builder.bind(roles::Role::MoeHcNorm, layer);
        layer_plan.moe_hc.down         = builder.bind(roles::Role::MoeHcDown, layer);
        layer_plan.moe_hc.inject       = builder.bind(roles::Role::MoeHcInject, layer);
        layer_plan.moe_hc.up           = builder.bind(roles::Role::MoeHcUp, layer);
        layer_plan.moe.router          = builder.bind(roles::Role::MoeRouter, layer);
        layer_plan.moe.shared_score    = builder.bind(roles::Role::MoeSharedScore, layer);
        layer_plan.moe.shared_gate     = builder.bind(roles::Role::MoeSharedGate, layer);
        layer_plan.moe.shared_up       = builder.bind(roles::Role::MoeSharedUp, layer);
        layer_plan.moe.shared_down     = builder.bind(roles::Role::MoeSharedDown, layer);
        layer_plan.moe.routed_gate     = builder.bind(roles::Role::MoeRoutedGate, layer);
        layer_plan.moe.routed_up       = builder.bind(roles::Role::MoeRoutedUp, layer);
        layer_plan.moe.routed_down     = builder.bind(roles::Role::MoeRoutedDown, layer);
        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            QsaPlan qsa;
            qsa.query           = builder.bind(roles::Role::QsaQuery, layer);
            qsa.key             = builder.bind(roles::Role::QsaKey, layer);
            qsa.value           = builder.bind(roles::Role::QsaValue, layer);
            qsa.index_query     = builder.bind(roles::Role::QsaIndexQuery, layer);
            qsa.index_key       = builder.bind(roles::Role::QsaIndexKey, layer);
            qsa.output          = builder.bind(roles::Role::QsaOutput, layer);
            qsa.output_gate     = builder.bind(roles::Role::QsaOutputGate, layer);
            qsa.query_norm      = builder.bind(roles::Role::QsaQueryNorm, layer);
            qsa.key_norm        = builder.bind(roles::Role::QsaKeyNorm, layer);
            qsa.index_query_norm = builder.bind(roles::Role::QsaIndexQueryNorm, layer);
            qsa.index_key_norm  = builder.bind(roles::Role::QsaIndexKeyNorm, layer);
            layer_plan.mixer    = std::move(qsa);
        } else {
            GdnPlan gdn;
            gdn.query         = builder.bind(roles::Role::GdnQuery, layer);
            gdn.key           = builder.bind(roles::Role::GdnKey, layer);
            gdn.value         = builder.bind(roles::Role::GdnValue, layer);
            gdn.z             = builder.bind(roles::Role::GdnZ, layer);
            gdn.a_projection  = builder.bind(roles::Role::GdnAProjection, layer);
            gdn.b_projection  = builder.bind(roles::Role::GdnBProjection, layer);
            gdn.a_log         = builder.bind(roles::Role::GdnALog, layer);
            gdn.dt_bias       = builder.bind(roles::Role::GdnDtBias, layer);
            gdn.convolution   = builder.bind(roles::Role::GdnConvolution, layer);
            gdn.norm          = builder.bind(roles::Role::GdnNorm, layer);
            gdn.output        = builder.bind(roles::Role::GdnOutput, layer);
            layer_plan.mixer   = std::move(gdn);
        }
    }

    plan.ple.table       = builder.bind(roles::Role::PleTable, roles::kPleLayer);
    plan.ple.key         = builder.bind(roles::Role::PleKey, roles::kPleLayer);
    plan.ple.value       = builder.bind(roles::Role::PleValue, roles::kPleLayer);
    plan.ple.norm_key    = builder.bind(roles::Role::PleNormKey, roles::kPleLayer);
    plan.ple.norm_query  = builder.bind(roles::Role::PleNormQuery, roles::kPleLayer);
    plan.ple.norm_conv   = builder.bind(roles::Role::PleNormConv, roles::kPleLayer);
    plan.ple.convolution = builder.bind(roles::Role::PleConvolution, roles::kPleLayer);

    collect_unclaimed(directory.bindings, builder.claimed(), plan.unclaimed);
    collect_unclaimed(directory.part_bindings, builder.claimed(), plan.unclaimed);
    collect_unclaimed_objects(directory.bindings, builder.claimed(), plan.unclaimed);
    collect_unclaimed_part_objects(directory.part_bindings, builder.claimed(), plan.unclaimed);
    return plan;
}


namespace {

std::int64_t align_up_256(std::int64_t value) noexcept {
    return (value + 255) / 256 * 256;
}

Weight part_weight(const artifact::MaterializedArtifact& materialized, const BoundParameter& parameter,
                   std::int32_t rows, std::int32_t columns) {
    const auto handle = artifact::ObjectHandle{static_cast<std::size_t>(parameter.parts.front().object)};
    if (!materialized.is_materialized(handle)) { return Weight{}; }
    return artifact::materialized_weight(materialized, handle, role_geometry(parameter.role).format,
                                         rows, columns, 0);
}

Tensor part_tensor(const artifact::MaterializedArtifact& materialized, const BoundParameter& parameter,
                   std::initializer_list<std::int32_t> shape) {
    const auto handle = artifact::ObjectHandle{static_cast<std::size_t>(parameter.parts.front().object)};
    if (!materialized.is_materialized(handle)) { return Tensor{}; }
    return artifact::materialized_tensor(materialized, handle, role_geometry(parameter.role).format,
                                         shape, 0);
}

BatchedNvfp4Weight part_library(const artifact::MaterializedArtifact& materialized,
                                const BoundParameter& parameter) {
    const ExpertLibrary& library = parameter.library.value();
    BatchedNvfp4Weight weight;
    if (!materialized.is_materialized(library.handle)) { return weight; }
    weight.payload      = materialized.device_data(library.handle);
    weight.payload_bytes = materialized.device_bytes(library.handle, 0);
    weight.qtype        = QType::NVFP4;
    weight.layout       = QuantLayout::BatchedBlockScaleK16M128x4;
    weight.group_size   = 16U;
    weight.matrices     = static_cast<std::int32_t>(library.matrices);
    weight.rows         = static_cast<std::int32_t>(library.rows);
    weight.columns      = static_cast<std::int32_t>(library.columns);
    const std::int64_t code_bytes  = static_cast<std::int64_t>(library.matrices) * library.rows *
                                     library.columns / 2;
    const std::int64_t scale_bytes = static_cast<std::int64_t>(library.matrices) * library.rows *
                                     library.columns / 16;
    const std::int64_t scale_offset = align_up_256(code_bytes);
    const auto* payload = static_cast<const std::uint8_t*>(weight.payload);
    weight.codes    = payload;
    weight.scales   = payload + scale_offset;
    weight.divisors = reinterpret_cast<const float*>(payload + scale_offset + scale_bytes);
    return weight;
}

qwen4_exp::HcWeights hc_view(const artifact::MaterializedArtifact& materialized, const HcPlan& plan) {
    return qwen4_exp::HcWeights{
        .down_injection = part_weight(materialized, plan.down, TextConfig::hc_lowrank + TextConfig::hc_streams,
                                      TextConfig::residual_width),
        .up = part_weight(materialized, plan.up, TextConfig::residual_width, TextConfig::hc_lowrank),
        .norm = part_tensor(materialized, plan.norm, {TextConfig::residual_width}),
    };
}

} // namespace

LoadedModelData::LoadedModelData(const BindingPlan& plan,
                                 const artifact::MaterializedArtifact& materialized,
                                 std::size_t loaded_layers) {
    model_.tp = 1;
    model_.loaded_layers = loaded_layers;

    model_.embedding = part_weight(materialized, plan.token_embedding, TextConfig::vocab,
                                   TextConfig::hidden);
    model_.output_head = part_weight(materialized, plan.output_head, TextConfig::vocab,
                                     TextConfig::hidden);
    model_.final_hc = hc_view(materialized, plan.final_hc);

    model_.ple = qwen4_exp::PleWeights{
        .key        = part_weight(materialized, plan.ple.key, TextConfig::residual_width,
                                  TextConfig::hidden),
        .value      = part_weight(materialized, plan.ple.value, TextConfig::hidden,
                                  TextConfig::hidden),
        .norm_key   = part_tensor(materialized, plan.ple.norm_key, {TextConfig::residual_width}),
        .norm_query = part_tensor(materialized, plan.ple.norm_query, {TextConfig::residual_width}),
        .norm_conv  = part_tensor(materialized, plan.ple.norm_conv, {TextConfig::residual_width}),
        .kernel     = part_tensor(materialized, plan.ple.convolution,
                                  {TextConfig::ple_conv_kernel, TextConfig::residual_width}),
    };

    for (std::size_t layer = 0; layer < loaded_layers; ++layer) {
        const LayerPlan& source = plan.layers[layer];
        auto& target = model_.layers[layer];
        target.attention_hc = hc_view(materialized, source.attention_hc);
        target.moe_hc       = hc_view(materialized, source.moe_hc);

        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            const QsaPlan& qsa = std::get<QsaPlan>(source.mixer);
            target.attention_projection = AttentionProjectionWeights{
                .query_gate        = part_weight(materialized, qsa.query, 2 * TextConfig::query_width,
                                                 TextConfig::hidden),
                .key_value         = part_weight(materialized, qsa.key, 2 * TextConfig::kv_width +
                                                 TextConfig::index_query_width +
                                                 TextConfig::index_key_width, TextConfig::hidden),
                .query_norm        = part_tensor(materialized, qsa.query_norm, {TextConfig::head_dim}),
                .key_norm          = part_tensor(materialized, qsa.key_norm, {TextConfig::head_dim}),
                .index_query_norm  = part_tensor(materialized, qsa.index_query_norm,
                                                 {TextConfig::index_head_dim}),
                .index_key_norm    = part_tensor(materialized, qsa.index_key_norm,
                                                 {TextConfig::index_head_dim}),
                .output            = part_weight(materialized, qsa.output, TextConfig::hidden,
                                                 TextConfig::query_width),
            };
        } else {
            const GdnPlan& gdn = std::get<GdnPlan>(source.mixer);
            target.gdn_projection = GdnProjectionWeights{
                .qkvz       = part_weight(materialized, gdn.query,
                                          TextConfig::gdn_conv_channels + TextConfig::gdn_value_width,
                                          TextConfig::hidden),
                .control    = part_weight(materialized, gdn.a_projection, 2 * TextConfig::gdn_value_heads,
                                          TextConfig::hidden),
                .a_log      = part_tensor(materialized, gdn.a_log, {TextConfig::gdn_value_heads}),
                .dt_bias    = part_tensor(materialized, gdn.dt_bias, {TextConfig::gdn_value_heads}),
                .convolution = part_tensor(materialized, gdn.convolution,
                                           {TextConfig::gdn_conv_kernel, TextConfig::gdn_conv_channels}),
                .norm       = part_tensor(materialized, gdn.norm, {128}),
                .output     = part_weight(materialized, gdn.output, TextConfig::hidden,
                                          TextConfig::gdn_value_width),
            };
        }

        target.post_mixer = ops::MoeNvfp4Weights{
            .router_shared    = part_weight(materialized, source.moe.router,
                                            TextConfig::experts + 1, TextConfig::hidden),
            .shared_down      = part_weight(materialized, source.moe.shared_down, TextConfig::hidden,
                                            TextConfig::shared_intermediate),
            .routed_gate_up   = part_library(materialized, source.moe.routed_gate),
            .routed_down      = part_library(materialized, source.moe.routed_down),
        };
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
