#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

#include "targets/qwen4_exp/impl/runtime/activation_dtype.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>

#include "artifact/typed_binding.h"

#include <cstddef>

#include <algorithm>
#include <cstring>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
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
    if (tp < 1 || tp > static_cast<int>(kMaximumDevices)) {
        throw std::invalid_argument("qwen3.8-flash-next binds between one and " +
                                    std::to_string(kMaximumDevices) + " devices");
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
    plan.ple_controls = binder.require_resource("resource/text/ple_hash_controls",
                                                artifact::ResourceEncoding::RawBytesV1);
    binder.retain_on_host(plan.ple_controls);

    collect_unclaimed(directory.bindings, builder.claimed(), plan.unclaimed);
    collect_unclaimed(directory.part_bindings, builder.claimed(), plan.unclaimed);
    collect_unclaimed_objects(directory.bindings, builder.claimed(), plan.unclaimed);
    collect_unclaimed_part_objects(directory.part_bindings, builder.claimed(), plan.unclaimed);

    if (tp > 1) {
        const auto shards = shard_plan_by_name(tp);
        auto placements = shard_placements(plan, shards, tp);
        binder.set_shard_resolver([placements = std::move(placements)](std::string_view name) {
            const auto found = placements.find(std::string(name));
            if (found == placements.end()) {
                throw std::invalid_argument("the shard map has no placement for object " + std::string(name));
            }
            return found->second;
        });
    }
    return plan;
}


bool is_fp16_control_object(const BindingPlan& plan, const BoundObject& object) {
    bool found = false;
    for (const auto* parameter : plan.parameters()) {
        for (const auto& part : parameter->parts) {
            if (plan.objects.at(part.object).name != object.name) { continue; }
            found = true;
            switch (parameter->role) {
            case roles::Role::OutputHead:
            case roles::Role::FinalHcDown:
            case roles::Role::FinalHcUp:
            case roles::Role::AttentionHcDown:
            case roles::Role::AttentionHcInject:
            case roles::Role::AttentionHcUp:
            case roles::Role::MoeHcDown:
            case roles::Role::MoeHcInject:
            case roles::Role::MoeHcUp:
            case roles::Role::PleKey:
            case roles::Role::PleValue:
            case roles::Role::QsaKey:
            case roles::Role::QsaValue:
            case roles::Role::QsaIndexQuery:
            case roles::Role::QsaIndexKey:
                break;
            default:
                return false;
            }
        }
    }
    return found;
}

namespace {

std::int64_t align_up_256(std::int64_t value) noexcept {
    return (value + 255) / 256 * 256;
}

artifact::ObjectHandle object_handle(const BindingPlan& plan, const BoundParameter& parameter) {
    return plan.objects[static_cast<std::size_t>(parameter.parts.front().object)].handle;
}

const sharding::ShardMapping& shard_of(const ShardPlanByName& shards, std::string_view role) {
    const auto found = shards.find(role);
    if (found == shards.end()) {
        throw std::runtime_error("the shard map has no role " + std::string(role));
    }
    return found->second;
}

std::uint64_t range_total(const std::vector<artifact::SliceRange>& ranges) {
    return std::accumulate(ranges.begin(), ranges.end(), std::uint64_t{0},
                           [](std::uint64_t value, const artifact::SliceRange& range) {
                               return value + range.count;
                           });
}

std::vector<std::uint64_t> narrowed_object_shape(std::span<const std::uint64_t> shape,
                                                 sharding::SplitAxis split, std::uint64_t selected) {
    std::vector<std::uint64_t> out(shape.begin(), shape.end());
    if (split == sharding::SplitAxis::Input) { out[1] = selected; } else { out[0] = selected; }
    return out;
}

artifact::SliceRange role_range(const ShardPlanByName& shards, roles::Role role,
                                sharding::SplitAxis split, int rank) {
    const auto& mapping = shard_of(shards, roles::role_name(role));
    const auto geometry = role_geometry(role);
    if (!mapping.ranks.empty()) {
        const auto& part = mapping.ranks[static_cast<std::size_t>(rank)];
        return {static_cast<std::uint64_t>(part.begin),
                static_cast<std::uint64_t>(part.end - part.begin)};
    }
    const auto begin = split == sharding::SplitAxis::Input ? geometry.column_begin : geometry.row_begin;
    const auto count = split == sharding::SplitAxis::Input ? geometry.column_count : geometry.row_count;
    return {begin, count};
}

const RoleBlock& role_block(const ObjectShardPlan& entry, roles::Role role, int rank) {
    for (const auto& block : entry.blocks[static_cast<std::size_t>(rank)]) {
        if (block.role == role) { return block; }
    }
    throw std::runtime_error("the shard byte plan has no role block for " +
                             std::string(roles::role_name(role)));
}

std::uint64_t block_bytes(const ShardPlanByName& shards, const ObjectShardPlan& entry, roles::Role role,
                          int rank) {
    const auto geometry = role_geometry(role);
    const auto own = role_range(shards, role, entry.split, rank);
    return artifact::tensor_encoded_size(geometry.layout, geometry.format,
                                         narrowed_object_shape(geometry.object_shape(), entry.split, own.count));
}

std::int64_t rank_extent(const ShardPlanByName& shards, roles::Role role, int rank) {
    const auto& mapping = shard_of(shards, roles::role_name(role));
    if (mapping.ranks.empty()) { return -1; }
    const auto& part = mapping.ranks[static_cast<std::size_t>(rank)];
    return part.end - part.begin;
}

const ObjectShardPlan& object_entry(const std::vector<ObjectShardPlan>& byte_plan,
                                    const std::string& name) {
    for (const auto& entry : byte_plan) {
        if (entry.object == name) { return entry; }
    }
    throw std::runtime_error("no shard byte plan for object " + name);
}

std::uint64_t block_offset(const ShardPlanByName& shards, const ObjectShardPlan& entry,
                           roles::Role role, int rank) {
    const std::vector<artifact::SliceRange> ranges{role_range(shards, role, entry.split, rank)};
    const auto geometry = role_geometry(role);
    const auto axis = entry.split == sharding::SplitAxis::Input ? geometry.object_shape()[1]
                                                                : geometry.object_shape()[0];
    if (ranges.size() == 1 && ranges.front().begin == 0 && ranges.front().count == axis) { return 0; }
    const auto own = entry.split == sharding::SplitAxis::Input
        ? artifact::tensor_column_slice(geometry.layout, geometry.format, geometry.object_shape(), ranges)
        : artifact::tensor_row_slice(geometry.layout, geometry.format, geometry.object_shape(), ranges);
    if (own.copies.empty()) { return 0; }
    const auto source = own.copies.front().source_offset;
    for (const auto& copy : entry.copies[static_cast<std::size_t>(rank)]) {
        if (copy.source_offset == source) { return copy.dest_offset; }
    }
    throw std::runtime_error("the shard of " + entry.object + " on rank " + std::to_string(rank) +
                             " does not hold the role block " + std::string(roles::role_name(role)));
}

} // namespace

ShardPlanByName shard_plan_by_name(int tp) {
    ShardPlanByName shards;
    for (const auto role : roles::all_roles()) {
        shards.emplace(std::string(roles::role_name(role)), sharding::shard_mapping(role, tp));
    }
    return shards;
}

std::vector<ObjectShardPlan> shard_byte_plan(const BindingPlan& plan, const ShardPlanByName& shards,
                                             int tp) {
    std::map<std::string, std::vector<const BoundParameter*>> owners;
    for (const auto* parameter : plan.parameters()) {
        for (const auto& part : parameter->parts) {
            owners[plan.objects.at(part.object).name].push_back(parameter);
        }
    }

    std::vector<ObjectShardPlan> byte_plan;
    byte_plan.reserve(plan.objects.size());
    for (const auto& object : plan.objects) {
        const auto owner = owners.find(object.name);
        const std::vector<const BoundParameter*> none;
        const auto& bound = owner == owners.end() ? none : owner->second;
        ObjectShardPlan entry;
        entry.object = object.name;
        for (const auto* parameter : bound) {
            if (std::find(entry.roles.begin(), entry.roles.end(), parameter->role) == entry.roles.end()) {
                entry.roles.push_back(parameter->role);
            }
        }
        entry.split = bound.empty() ? sharding::SplitAxis::Output
                                    : shard_of(shards, roles::role_name(entry.roles.front())).split;

        for (int rank = 0; rank < tp; ++rank) {
            std::vector<artifact::SliceRange> ranges;
            if (bound.empty()) {
                const auto extent = entry.split == sharding::SplitAxis::Input ? object.shape[1]
                                                                             : object.shape[0];
                ranges.push_back({0, extent});
            }
            for (const auto* parameter : bound) {
                ranges.push_back(role_range(shards, parameter->role, entry.split, rank));
            }
            std::sort(ranges.begin(), ranges.end(), [](const artifact::SliceRange& left,
                                                       const artifact::SliceRange& right) {
                return left.begin < right.begin;
            });
            ranges.erase(std::unique(ranges.begin(), ranges.end(),
                                     [](const artifact::SliceRange& left, const artifact::SliceRange& right) {
                                         return left.begin == right.begin && left.count == right.count;
                                     }), ranges.end());
            entry.ranges[static_cast<std::size_t>(rank)] = ranges;

            const auto shape = narrowed_object_shape(object.shape, entry.split, range_total(ranges));
            entry.encoded_bytes[static_cast<std::size_t>(rank)] =
                artifact::tensor_encoded_size(object.layout, object.format, shape);

            const auto extent = entry.split == sharding::SplitAxis::Input ? object.shape[1]
                                                                         : object.shape[0];
            if (ranges.size() == 1 && ranges.front().begin == 0 && ranges.front().count == extent) {
                entry.copies[static_cast<std::size_t>(rank)].push_back(
                    {0, 0, entry.encoded_bytes[static_cast<std::size_t>(rank)]});
            } else {
                const auto slice = entry.split == sharding::SplitAxis::Input
                    ? artifact::tensor_column_slice(object.layout, object.format, object.shape, ranges)
                    : artifact::tensor_row_slice(object.layout, object.format, object.shape, ranges);
                entry.copies[static_cast<std::size_t>(rank)] = slice.copies;
                entry.prefix[static_cast<std::size_t>(rank)] = slice.prefix;
            }

            for (const auto role : entry.roles) {
                RoleBlock block;
                block.role   = role;
                block.offset = block_offset(shards, entry, role, rank);
                block.bytes  = block_bytes(shards, entry, role, rank);
                entry.blocks[static_cast<std::size_t>(rank)].push_back(block);
            }

            if (object.layout == StorageLayout::BatchedBlockScaleK16M128x4V1) {
                const auto geometry = artifact::batched_block_scale_geometry(object.format, shape);
                entry.scale_plane_offset[static_cast<std::size_t>(rank)] = geometry.scale_plane_offset;
                entry.divisor_plane_offset[static_cast<std::size_t>(rank)] =
                    geometry.weight_divisor_offset;
            } else if (object.layout == StorageLayout::RowSplitK128V1) {
                entry.scale_plane_offset[static_cast<std::size_t>(rank)] =
                    artifact::row_split_geometry(object.format, shape).scale_plane_offset;
            }
        }
        byte_plan.push_back(std::move(entry));
    }
    return byte_plan;
}

std::map<std::string, artifact::ShardPlacement> shard_placements(const BindingPlan& plan,
                                                                  const ShardPlanByName& shards,
                                                                  int tp) {
    const auto byte_plan = shard_byte_plan(plan, shards, tp);
    std::map<std::string, artifact::ShardPlacement> placements;
    for (const auto& entry : byte_plan) {
        if (entry.roles.empty()) {
            artifact::ShardPlacement placement;
            placement.axis = artifact::ShardAxis::Replicated;
            placements.emplace(entry.object, placement);
            continue;
        }
        bool split = false;
        for (const auto role : entry.roles) {
            if (shard_of(shards, roles::role_name(role)).axis != sharding::ShardAxis::Replicated) {
                split = true;
            }
        }
        artifact::ShardPlacement placement;
        if (!split) {
            placement.axis = artifact::ShardAxis::Replicated;
        } else {
            placement.axis = entry.split == sharding::SplitAxis::Input
                ? artifact::ShardAxis::Columns
                : artifact::ShardAxis::Rows;
            for (int rank = 0; rank < tp; ++rank) {
                for (const auto& range : entry.ranges[static_cast<std::size_t>(rank)]) {
                    placement.device_ranges[static_cast<std::size_t>(rank)].push_back(range);
                }
            }
        }
        placements.emplace(entry.object, placement);
    }
    return placements;
}

namespace {

Weight part_weight(const BindingPlan& plan, const artifact::MaterializedArtifact& materialized,
                   const ShardPlanByName& shards, const std::vector<ObjectShardPlan>& byte_plan,
                   const BoundParameter& parameter, std::int32_t rows, std::int32_t columns,
                   int rank) {
    const auto handle = object_handle(plan, parameter);
    if (!materialized.is_materialized(handle, rank)) { return Weight{}; }
    const auto& mapping = shard_of(shards, roles::role_name(parameter.role));
    const auto geometry = role_geometry(parameter.role);
    const auto& entry = object_entry(byte_plan, plan.objects.at(parameter.parts.front().object).name);
    const auto extent = rank_extent(shards, parameter.role, rank);
    std::int32_t entry_rows = rows;
    std::int32_t entry_columns = columns;
    if (extent >= 0) {
        if (mapping.split == sharding::SplitAxis::Output) {
            entry_rows = rows == static_cast<std::int32_t>(geometry.row_count)
                ? static_cast<std::int32_t>(extent)
                : static_cast<std::int32_t>(range_total(entry.ranges[static_cast<std::size_t>(rank)]));
        } else if (mapping.split == sharding::SplitAxis::Input) {
            entry_columns = columns == static_cast<std::int32_t>(geometry.column_count)
                ? static_cast<std::int32_t>(extent)
                : static_cast<std::int32_t>(range_total(entry.ranges[static_cast<std::size_t>(rank)]));
        }
    }

    Weight weight = artifact::materialized_weight(materialized, handle, geometry.format,
                                                  entry_rows, entry_columns, rank);
    const auto& block = role_block(entry, parameter.role, rank);
    if (block.offset == 0) { return weight; }
    const auto* base = static_cast<const std::byte*>(weight.payload) + block.offset;
    weight.payload = base;
    weight.qdata   = base;
    weight.payload_bytes = block.bytes;
    return weight;
}

Tensor part_tensor(const BindingPlan& plan, const artifact::MaterializedArtifact& materialized,
                   const ShardPlanByName& shards, const BoundParameter& parameter,
                   std::initializer_list<std::int32_t> shape, int rank) {
    const auto handle = object_handle(plan, parameter);
    if (!materialized.is_materialized(handle, rank)) { return Tensor{}; }
    const auto& mapping = shard_of(shards, roles::role_name(parameter.role));
    return artifact::materialized_tensor(materialized, handle, role_geometry(parameter.role).format,
                                         shape, rank);
}

BatchedNvfp4Weight part_library(const BindingPlan& plan, const artifact::MaterializedArtifact& materialized,
                                const ShardPlanByName& shards,
                                const std::vector<ObjectShardPlan>& byte_plan,
                                const BoundParameter& parameter, int rank) {
    const ExpertLibrary& library = parameter.library.value();
    const auto handle = library.handle;
    BatchedNvfp4Weight weight;
    if (!materialized.is_materialized(handle, rank)) { return weight; }
    const auto& entry = object_entry(byte_plan, library.object);
    const auto& ranges = entry.ranges[static_cast<std::size_t>(rank)];
    const auto* payload = static_cast<const std::uint8_t*>(materialized.device_data(handle, rank));
    weight.payload       = payload;
    weight.payload_bytes = entry.encoded_bytes[static_cast<std::size_t>(rank)];
    weight.qtype         = QType::NVFP4;
    weight.layout        = QuantLayout::BatchedBlockScaleK16M128x4;
    weight.group_size    = 16U;
    weight.matrices      = static_cast<std::int32_t>(range_total(ranges));
    const auto& mapping = shard_of(shards, roles::role_name(parameter.role));
    weight.expert_begin = static_cast<std::int32_t>(
        mapping.ranks.empty() ? 0 : mapping.ranks[static_cast<std::size_t>(rank)].begin);
    weight.rows          = static_cast<std::int32_t>(library.rows);
    weight.columns       = static_cast<std::int32_t>(library.columns);
    weight.codes         = payload;
    weight.scales        = payload + entry.scale_plane_offset[static_cast<std::size_t>(rank)];
    weight.divisors      = reinterpret_cast<const float*>(
        payload + entry.divisor_plane_offset[static_cast<std::size_t>(rank)]);
    return weight;
}

qwen4_exp::HcWeights hc_view(const BindingPlan& bindings, const artifact::MaterializedArtifact& materialized,
                             const ShardPlanByName& shards, const std::vector<ObjectShardPlan>& byte_plan,
                             const HcPlan& plan, std::int32_t down_rows, int rank) {
    return qwen4_exp::HcWeights{
        .down_injection = part_weight(bindings, materialized, shards, byte_plan, plan.down, down_rows,
                                      TextConfig::residual_width, rank),
        .up = part_weight(bindings, materialized, shards, byte_plan, plan.up, TextConfig::residual_width,
                          TextConfig::hc_lowrank, rank),
        .norm = part_tensor(bindings, materialized, shards, plan.norm,
                            {TextConfig::residual_width / TextConfig::hc_streams,
                             TextConfig::hc_streams}, rank),
    };
}

} // namespace

namespace {

std::uint64_t descriptor_offset(const artifact::ObjectDescriptor& descriptor) {
    return std::visit([](const auto& value) { return value.offset; }, descriptor);
}

} // namespace

PleTableSource::PleTableSource(const std::filesystem::path& path, const BindingPlan& plan)
    : reader_(path) {
    if (plan.ple.table.parts.size() != ops::kPleShards) {
        throw std::runtime_error("the PLE table does not carry one part per shard");
    }
    const std::array<std::uint64_t, 2> table_shape{TextConfig::ple_shard_rows,
                                                   TextConfig::ple_head_width};
    geometry_ = artifact::row_paged_nf4_geometry(artifact::NumericFormat::NF4_G16_E4M3S,
                                                 table_shape);
    const artifact::ObjectDescriptor* controls =
        reader_.find("resource/text/ple_hash_controls");
    if (controls == nullptr) {
        throw std::runtime_error("the artifact does not carry the PLE hash controls");
    }
    std::array<std::byte, ops::kPleHashControlsBytes> controls_payload{};
    const std::uint64_t controls_offset =
        reader_.payload_offset() + descriptor_offset(*controls);
    if (reader_.read_direct(controls_offset, controls_payload) != controls_payload.size()) {
        throw std::runtime_error("the PLE hash controls were read short");
    }
    controls_ = ops::parse_ple_hash_controls(controls_payload);
    for (std::size_t shard = 0; shard < ops::kPleShards; ++shard) {
        const auto& part = plan.ple.table.parts[shard];
        const auto& object = plan.objects[static_cast<std::size_t>(part.object)];
        const artifact::ObjectDescriptor* descriptor = reader_.find(object.name);
        if (descriptor == nullptr) {
            throw std::runtime_error("the artifact does not carry the PLE shard " + object.name);
        }
        shard_offsets_[shard] = descriptor_offset(*descriptor);
        std::array<std::byte, 4> word{};
        const std::uint64_t offset = reader_.payload_offset() + shard_offsets_[shard] +
                                     geometry_.matrix_scale_offset;
        if (reader_.read_direct(offset, word) != word.size()) {
            throw std::runtime_error("the PLE shard " + object.name + " has no divisor word");
        }
        std::memcpy(&divisors_[shard], word.data(), sizeof(float));
    }
}

void PleTableSource::fetch(const ops::PleRowAddress* addresses, std::size_t count,
                           std::byte* destination) const {
    const std::uint64_t row_bytes = geometry_.row_bytes;
    for (std::size_t index = 0; index < count; ++index) {
        const ops::PleRowAddress& address = addresses[index];
        const std::uint64_t offset = reader_.payload_offset() + shard_offsets_.at(address.shard) +
                                     artifact::row_paged_nf4_row_offset(geometry_, address.row);
        std::span<std::byte> row{destination + index * row_bytes,
                                 static_cast<std::size_t>(row_bytes)};
        if (reader_.read_direct(offset, row) != row.size()) {
            throw std::runtime_error("a PLE row read was short");
        }
    }
}

void fetch_ple_rows(void* context, const ops::PleRowAddress* addresses, std::size_t count,
                    std::byte* destination) {
    static_cast<PleTableSource*>(context)->fetch(addresses, count, destination);
}

namespace {

Variant::ModelView build_rank_view(const BindingPlan& plan,
                                   const artifact::MaterializedArtifact& materialized,
                                   const ShardPlanByName& shards,
                                   const std::vector<ObjectShardPlan>& byte_plan, int rank,
                                   std::size_t loaded_layers) {
    Variant::ModelView view;

    view.embedding = part_weight(plan, materialized, shards, byte_plan, plan.token_embedding,
                                 TextConfig::vocab, TextConfig::hidden, rank);
    view.output_head = part_weight(plan, materialized, shards, byte_plan, plan.output_head,
                                   TextConfig::vocab, TextConfig::hidden, rank);
    view.final_hc = hc_view(plan, materialized, shards, byte_plan, plan.final_hc,
                            TextConfig::hc_lowrank, rank);

    view.ple = qwen4_exp::PleWeights{
        .key        = part_weight(plan, materialized, shards, byte_plan, plan.ple.key,
                                  TextConfig::residual_width, TextConfig::hidden, rank),
        .value      = part_weight(plan, materialized, shards, byte_plan, plan.ple.value,
                                  TextConfig::hidden, TextConfig::hidden, rank),
        .norm_key   = part_tensor(plan, materialized, shards, plan.ple.norm_key,
                                  {TextConfig::residual_width}, rank),
        .norm_query = part_tensor(plan, materialized, shards, plan.ple.norm_query,
                                  {TextConfig::residual_width}, rank),
        .norm_conv  = part_tensor(plan, materialized, shards, plan.ple.norm_conv,
                                  {TextConfig::residual_width}, rank),
        .kernel     = part_tensor(plan, materialized, shards, plan.ple.convolution,
                                  {TextConfig::ple_conv_kernel, TextConfig::residual_width}, rank),
    };

    for (std::size_t layer = 0; layer < loaded_layers; ++layer) {
        const LayerPlan& source = plan.layers[layer];
        auto& target = view.layers[layer];
        target.attention_hc = hc_view(plan, materialized, shards, byte_plan, source.attention_hc,
                                      TextConfig::hc_lowrank + TextConfig::hc_streams, rank);
        target.moe_hc = hc_view(plan, materialized, shards, byte_plan, source.moe_hc,
                                TextConfig::hc_lowrank + TextConfig::hc_streams, rank);

        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            const QsaPlan& qsa = std::get<QsaPlan>(source.mixer);
            target.attention_projection = AttentionProjectionWeights{
                .query_gate        = part_weight(plan, materialized, shards, byte_plan, qsa.query,
                                                 2 * TextConfig::query_width, TextConfig::hidden, rank),
                .key_value         = part_weight(plan, materialized, shards, byte_plan, qsa.key,
                                                 2 * TextConfig::kv_width + TextConfig::index_query_width +
                                                 TextConfig::index_key_width, TextConfig::hidden, rank),
                .query_norm        = part_tensor(plan, materialized, shards, qsa.query_norm,
                                                 {TextConfig::head_dim}, rank),
                .key_norm          = part_tensor(plan, materialized, shards, qsa.key_norm,
                                                 {TextConfig::head_dim}, rank),
                .index_query_norm  = part_tensor(plan, materialized, shards, qsa.index_query_norm,
                                                 {TextConfig::index_head_dim}, rank),
                .index_key_norm    = part_tensor(plan, materialized, shards, qsa.index_key_norm,
                                                 {TextConfig::index_head_dim}, rank),
                .output            = part_weight(plan, materialized, shards, byte_plan, qsa.output,
                                                 TextConfig::hidden, TextConfig::query_width, rank),
            };
        } else {
            const GdnPlan& gdn = std::get<GdnPlan>(source.mixer);
            target.gdn_projection = GdnProjectionWeights{
                .qkvz       = part_weight(plan, materialized, shards, byte_plan, gdn.query,
                                          TextConfig::gdn_conv_channels + TextConfig::gdn_value_width,
                                          TextConfig::hidden, rank),
                .control    = part_weight(plan, materialized, shards, byte_plan, gdn.a_projection,
                                          2 * TextConfig::gdn_value_heads, TextConfig::hidden, rank),
                .a_log      = part_tensor(plan, materialized, shards, gdn.a_log,
                                          {TextConfig::gdn_value_heads}, rank),
                .dt_bias    = part_tensor(plan, materialized, shards, gdn.dt_bias,
                                          {TextConfig::gdn_value_heads}, rank),
                .convolution = part_tensor(plan, materialized, shards, gdn.convolution,
                                           {TextConfig::gdn_conv_kernel, TextConfig::gdn_conv_channels}, rank),
                .norm       = part_tensor(plan, materialized, shards, gdn.norm, {128}, rank),
                .output     = part_weight(plan, materialized, shards, byte_plan, gdn.output,
                                          TextConfig::hidden, TextConfig::gdn_value_width, rank),
            };
        }

        target.post_mixer = MoeWeights{};
        target.post_mixer.router_shared =
            part_weight(plan, materialized, shards, byte_plan, source.moe.router,
                        TextConfig::experts + 1 + 2 * TextConfig::shared_intermediate,
                        TextConfig::hidden, rank);
        target.post_mixer.shared_down =
            part_weight(plan, materialized, shards, byte_plan, source.moe.shared_down,
                        TextConfig::hidden, TextConfig::shared_intermediate, rank);
        target.post_mixer.routed_gate_up =
            part_library(plan, materialized, shards, byte_plan, source.moe.routed_gate, rank);
        target.post_mixer.routed_down =
            part_library(plan, materialized, shards, byte_plan, source.moe.routed_down, rank);
        target.post_mixer.cache = nullptr;
        target.post_mixer.layer = static_cast<std::uint32_t>(layer);
    }

    return view;
}

} // namespace

RankViews shard_views(const BindingPlan& plan, const artifact::MaterializedArtifact& materialized,
                      const ShardPlanByName& shards, int tp, std::size_t loaded_layers) {
    const auto byte_plan = shard_byte_plan(plan, shards, tp);
    RankViews views;
    for (int rank = 0; rank < tp; ++rank) {
        auto& view = views.views[static_cast<std::size_t>(rank)];
        view = build_rank_view(plan, materialized, shards, byte_plan, rank, loaded_layers);
        view.tp = tp;
    }
    return views;
}

namespace {

class BoundDevice {
public:
    explicit BoundDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        if (previous_ != device) { CUDA_CHECK(cudaSetDevice(device)); }
    }
    ~BoundDevice() { (void)cudaSetDevice(previous_); }
    BoundDevice(const BoundDevice&)            = delete;
    BoundDevice& operator=(const BoundDevice&) = delete;

private:
    int previous_ = 0;
};

std::vector<std::byte> read_object_bytes(artifact::Reader& reader, const std::string& name) {
    const auto* descriptor = reader.find(name);
    if (descriptor == nullptr) {
        throw std::runtime_error("the artifact has no object " + name);
    }
    if (std::holds_alternative<artifact::ResourceDescriptor>(*descriptor)) {
        throw std::runtime_error("the object " + name + " is a resource, not a tensor");
    }
    const std::uint64_t bytes = artifact::object_bytes(*descriptor);
    const std::uint64_t absolute = reader.payload_offset() + artifact::object_offset(*descriptor);
    std::vector<std::byte> out(bytes);
    std::size_t copied = 0;
    for (const auto& view : reader.payload_views(absolute, bytes)) {
        std::copy(view.data.begin(), view.data.end(),
                  out.begin() + static_cast<std::ptrdiff_t>(copied));
        copied += view.data.size();
    }
    if (copied != bytes) {
        throw std::runtime_error("the object " + name + " was read short");
    }
    return out;
}

Weight fp16_host_weight(const BindingPlan& plan, artifact::Reader& reader,
                        const ShardPlanByName& shards,
                        const std::vector<ObjectShardPlan>& byte_plan,
                        const BoundParameter& parameter, std::int32_t rows, std::int32_t columns,
                        int rank, int device, DeviceArena& arena,
                        std::map<std::pair<std::string, int>, const std::byte*>& cache) {
    const std::string& object_name = plan.objects.at(parameter.parts.front().object).name;
    const auto key = std::make_pair(object_name, rank);
    auto found = cache.find(key);
    if (found == cache.end()) {
        const auto& entry = object_entry(byte_plan, object_name);
        std::vector<std::byte> shard(
            entry.encoded_bytes[static_cast<std::size_t>(rank)], std::byte{0});
        const auto parent = read_object_bytes(reader, object_name);
        for (const auto& copy : entry.copies[static_cast<std::size_t>(rank)]) {
            if (copy.source_offset + copy.bytes > parent.size() ||
                copy.dest_offset + copy.bytes > shard.size()) {
                throw std::runtime_error("the shard copy of " + object_name +
                                         " is outside its payload");
            }
            std::memcpy(shard.data() + copy.dest_offset, parent.data() + copy.source_offset,
                        static_cast<std::size_t>(copy.bytes));
        }
        const auto& prefix = entry.prefix[static_cast<std::size_t>(rank)];
        if (!prefix.empty()) {
            if (prefix.size() > shard.size()) {
                throw std::runtime_error("the shard prefix of " + object_name + " is too large");
            }
            std::memcpy(shard.data(), prefix.data(), prefix.size());
        }
        float max_abs = 0.0F;
        const std::string label = roles::parameter_name(parameter.role, parameter.layer);
        for (std::size_t word = 0; word + 1 < shard.size(); word += 2) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, shard.data() + word, sizeof(bits));
            __nv_bfloat16 raw;
            std::memcpy(&raw, &bits, sizeof(raw));
            const float value = __bfloat162float(raw);
            const float magnitude = std::abs(value);
            if (magnitude > max_abs) { max_abs = magnitude; }
            if (!(magnitude <= 65504.0F)) {
                throw std::runtime_error("the fp16 materialization of " + label +
                                         " exceeds the FP16 range; max |value| = " +
                                         std::to_string(max_abs));
            }
            const __half converted = __float2half_rn(value);
            std::memcpy(shard.data() + word, &converted, sizeof(converted));
        }
        const BoundDevice bound(device);
        const DeviceSpan span = arena.alloc_bytes(shard.size());
        CUDA_CHECK(cudaMemcpy(span.data, shard.data(), shard.size(), cudaMemcpyHostToDevice));
        found = cache.emplace(key, static_cast<const std::byte*>(span.data)).first;
    }
    const auto geometry = role_geometry(parameter.role);
    const auto mapping = shard_of(shards, roles::role_name(parameter.role));
    const auto extent = rank_extent(shards, parameter.role, rank);
    std::int32_t entry_rows = rows;
    std::int32_t entry_columns = columns;
    const auto& entry = object_entry(byte_plan, object_name);
    if (extent >= 0) {
        if (mapping.split == sharding::SplitAxis::Output) {
            entry_rows = rows == static_cast<std::int32_t>(geometry.row_count)
                ? static_cast<std::int32_t>(extent)
                : static_cast<std::int32_t>(range_total(entry.ranges[static_cast<std::size_t>(rank)]));
        } else if (mapping.split == sharding::SplitAxis::Input) {
            entry_columns = columns == static_cast<std::int32_t>(geometry.column_count)
                ? static_cast<std::int32_t>(extent)
                : static_cast<std::int32_t>(range_total(entry.ranges[static_cast<std::size_t>(rank)]));
        }
    }
    const auto& block = role_block(entry, parameter.role, rank);
    Weight weight;
    const auto* base = found->second + block.offset;
    weight.payload = base;
    weight.qdata = base;
    weight.payload_bytes = static_cast<std::uint64_t>(entry_rows) * entry_columns * 2;
    weight.qtype = QType::FP16_CTRL;
    weight.layout = QuantLayout::Contiguous;
    weight.n = entry_rows;
    weight.k = entry_columns;
    weight.shape[0] = entry_rows;
    weight.shape[1] = entry_columns;
    weight.padded_shape[0] = entry_rows;
    weight.padded_shape[1] = entry_columns;
    weight.ndim = 2;
    return weight;
}

} // namespace

std::array<std::uint64_t, kMaximumDevices>
fp16_control_capacity_bytes(const BindingPlan& plan, int tp, std::size_t loaded_layers) {
    std::array<std::uint64_t, kMaximumDevices> out{};
    if constexpr (qwen4_exp::detail::kTextActivationDtype != DType::FP16) { return out; }

    const auto shards = shard_plan_by_name(tp);
    const auto byte_plan = shard_byte_plan(plan, shards, tp);
    std::array<std::set<std::string>, kMaximumDevices> seen;
    const auto add = [&](const BoundParameter& parameter) {
        for (int rank = 0; rank < tp; ++rank) {
            const auto& object = plan.objects.at(parameter.parts.front().object);
            if (!is_fp16_control_object(plan, object)) { continue; }
            if (!seen[static_cast<std::size_t>(rank)].insert(object.name).second) { continue; }
            const auto& entry = object_entry(byte_plan, object.name);
            out[static_cast<std::size_t>(rank)] +=
                align_up_256(static_cast<std::int64_t>(entry.encoded_bytes[static_cast<std::size_t>(rank)]));
        }
    };

    add(plan.output_head);
    add(plan.final_hc.down);
    add(plan.final_hc.up);
    add(plan.ple.key);
    add(plan.ple.value);
    for (std::size_t layer = 0; layer < loaded_layers; ++layer) {
        const LayerPlan& source = plan.layers[layer];
        add(source.attention_hc.down);
        add(source.attention_hc.up);
        add(source.moe_hc.down);
        add(source.moe_hc.up);
        if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
            add(std::get<QsaPlan>(source.mixer).key);
        }
    }
    return out;
}

LoadedModelData::LoadedModelData(const BindingPlan& plan,
                                 artifact::MaterializedArtifact& materialized,
                                 std::size_t loaded_layers, int tp,
                                 const std::filesystem::path& artifact_path)
    : backing_(&materialized) {
    ple_source_ = std::make_unique<PleTableSource>(artifact_path, plan);

    const auto shards = shard_plan_by_name(tp);
    views_ = shard_views(plan, materialized, shards, tp, loaded_layers);

    if constexpr (qwen4_exp::detail::kTextActivationDtype == DType::FP16) {
        const auto probe_handle =
            plan.objects.at(plan.token_embedding.parts.front().object).handle;
        if (materialized.stats().tensor_count != 0) {
            if (!materialized.is_materialized(probe_handle, 0)) {
                throw std::runtime_error(
                    "the fp16 control materialization probe object is not resident");
            }
            artifact::Reader reader(artifact_path);
            const auto byte_plan = shard_byte_plan(plan, shards, tp);
            for (int rank = 0; rank < tp; ++rank) {
                const int device = materialized.physical_device(rank);
                auto& view = views_.views[static_cast<std::size_t>(rank)];
                DeviceArena& arena = materialized.device_arena(rank);
                std::map<std::pair<std::string, int>, const std::byte*> cache;
                const auto build = [&](const BoundParameter& parameter, std::int32_t rows,
                                       std::int32_t columns) {
                    return fp16_host_weight(plan, reader, shards, byte_plan, parameter, rows, columns,
                                            rank, device, arena, cache);
                };
                view.output_head = build(plan.output_head, TextConfig::vocab, TextConfig::hidden);
                view.final_hc.down_injection =
                    build(plan.final_hc.down, TextConfig::hc_lowrank, TextConfig::residual_width);
                view.final_hc.up =
                    build(plan.final_hc.up, TextConfig::residual_width, TextConfig::hc_lowrank);
                view.ple.key =
                    build(plan.ple.key, TextConfig::residual_width, TextConfig::hidden);
                view.ple.value = build(plan.ple.value, TextConfig::hidden, TextConfig::hidden);
                for (std::size_t layer = 0; layer < loaded_layers; ++layer) {
                    const LayerPlan& source = plan.layers[layer];
                    view.layers[layer].attention_hc.down_injection =
                        build(source.attention_hc.down, TextConfig::hc_lowrank + TextConfig::hc_streams,
                              TextConfig::residual_width);
                    view.layers[layer].attention_hc.up =
                        build(source.attention_hc.up, TextConfig::residual_width, TextConfig::hc_lowrank);
                    view.layers[layer].moe_hc.down_injection =
                        build(source.moe_hc.down, TextConfig::hc_lowrank + TextConfig::hc_streams,
                              TextConfig::residual_width);
                    view.layers[layer].moe_hc.up =
                        build(source.moe_hc.up, TextConfig::residual_width, TextConfig::hc_lowrank);
                    if (TextConfig::is_qsa_layer(static_cast<int>(layer))) {
                        const QsaPlan& qsa = std::get<QsaPlan>(source.mixer);
                        view.layers[layer].attention_projection.key_value =
                            build(qsa.key,
                                  2 * TextConfig::kv_width + TextConfig::index_query_width +
                                      TextConfig::index_key_width,
                                  TextConfig::hidden);
                    }
                }
            }
        }
    }

    for (int rank = 0; rank < tp; ++rank) {
        auto& view = views_.views[static_cast<std::size_t>(rank)];
        view.loaded_layers = loaded_layers;
        const BoundParameter& tail_norm = loaded_layers < static_cast<std::size_t>(TextConfig::layers)
                                              ? plan.layers[loaded_layers].attention_hc.norm
                                              : plan.final_hc.norm;
        view.tail_norm = part_tensor(plan, materialized, shards, tail_norm,
                                     {TextConfig::residual_width / TextConfig::hc_streams,
                                      TextConfig::hc_streams}, rank);
        view.ple.controls = ple_source_->controls();
        view.ple.rows = qwen4_exp::PleRowSource{ple_source_.get(), &fetch_ple_rows,
                                                 ple_source_->divisors()};
        if (materialized.stats().tensor_count != 0) {
            resident_[static_cast<std::size_t>(rank)].weights_capacity_bytes =
                materialized.device_arena(rank).capacity();
            resident_[static_cast<std::size_t>(rank)].weights_used_bytes =
                materialized.device_arena(rank).used();
        }
        view.resident = &resident_[static_cast<std::size_t>(rank)];
    }
}

void LoadedModelData::allocate_expert_cache(const EngineOptions& options,
                                            ExecutionContext& execution) {
    if (backing_ == nullptr || backing_->stats().tensor_count == 0) { return; }
    if (execution.tp < 1 || execution.tp > static_cast<int>(kMaximumDevices)) {
        throw std::invalid_argument("the expert cache needs one to four ranks");
    }
    if (experts_[0] != nullptr) { return; }

    constexpr std::uint64_t kVramReserveBytes = 1ULL << 30;
    for (int rank = 0; rank < execution.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        int previous    = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
        std::size_t free_bytes  = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        CUDA_CHECK(cudaSetDevice(previous));

        ExpertCacheOptions cache_options;
        if (options.expert_cache_mib != 0) {
            cache_options.capacity_bytes =
                static_cast<std::uint64_t>(options.expert_cache_mib) * 1024ULL * 1024ULL;
        }
        cache_options.vram_reserve_bytes = kVramReserveBytes;
        experts_[slot] = std::make_unique<ExpertRuntime>(
            options.artifact_path, execution.dev[slot]->device,
            static_cast<std::uint32_t>(rank) * kRankExperts, cache_options, free_bytes,
            &resident_[slot]);
        auto& view = views_.views[slot];
        for (std::size_t layer = 0; layer < TextConfig::layers; ++layer) {
            view.layers[layer].post_mixer.cache = experts_[slot].get();
        }
    }
}

CacheStats LoadedModelData::expert_cache_stats(int rank) const {
    if (rank < 0 || rank >= static_cast<int>(kMaximumDevices) || experts_[static_cast<std::size_t>(rank)] == nullptr) {
        throw std::invalid_argument("the expert cache is not allocated for this rank");
    }
    return experts_[static_cast<std::size_t>(rank)]->cache().stats();
}

std::uint64_t LoadedModelData::expert_cache_bytes(int rank) const {
    if (rank < 0 || rank >= static_cast<int>(kMaximumDevices) || experts_[static_cast<std::size_t>(rank)] == nullptr) {
        return 0;
    }
    return experts_[static_cast<std::size_t>(rank)]->device_bytes();
}

std::uint64_t LoadedModelData::expert_host_bytes(int rank) const {
    if (rank < 0 || rank >= static_cast<int>(kMaximumDevices) || experts_[static_cast<std::size_t>(rank)] == nullptr) {
        return 0;
    }
    return experts_[static_cast<std::size_t>(rank)]->host_bytes();
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
