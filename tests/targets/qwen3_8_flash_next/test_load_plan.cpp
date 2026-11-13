#include "fixture.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer_flash_next_fixture::FixtureArtifact;
using ninfer_flash_next_fixture::FixtureBinding;
using ninfer_flash_next_fixture::TemporaryArtifact;

using ninfer::artifact::ArtifactError;
using ninfer::artifact::ArtifactIdentity;
using ninfer::artifact::Binder;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::Reader;
using ninfer::artifact::StorageLayout;

namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;
using Package = ninfer::targets::qwen3_8_flash_next::Package;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool has_frontend_resources(const Binder& binder) {
    return ninfer_flash_next_fixture::has_frontend_resources(binder);
}

const flash::BoundObject& object_of(const flash::BindingPlan& plan, const flash::BoundPart& part) {
    return plan.objects.at(part.object);
}

void require_rectangle(const flash::BoundParameter& parameter, std::uint64_t row_begin,
                       std::uint64_t row_count, std::uint64_t column_count,
                       const std::string& what) {
    require(!parameter.parts.empty(), what + " has no part");
    for (const auto& part : parameter.parts) {
        require(part.view.matrix_count == 1, what + " spans more than one matrix");
        require(part.view.matrix_row_begin == row_begin,
                what + " starts at row " + std::to_string(part.view.matrix_row_begin) +
                    "; it starts at " + std::to_string(row_begin));
        require(part.view.matrix_row_count == row_count,
                what + " spans " + std::to_string(part.view.matrix_row_count) + " rows; it spans " +
                    std::to_string(row_count));
        require(part.view.column_begin == 0,
                what + " starts at column " + std::to_string(part.view.column_begin) +
                    "; it starts at column 0");
        require(part.view.column_count == column_count,
                what + " spans " + std::to_string(part.view.column_count) +
                    " columns; it spans " + std::to_string(column_count));
    }
}

void test_identity() {
    require(Package::model_id == "qwen3.8-flash-next", "the package's model id is wrong");
    require(Package::weights_id == "nvfp4", "the package's weights id is wrong");
    require(Package::target_key == "qwen3_8_flash_next", "the package's target key is wrong");

    require(Package::resolve_weights(
                ArtifactIdentity{std::string(Package::model_id), std::string(Package::weights_id)}) ==
            Package::WeightsProfile::Nvfp4,
            "the package does not resolve its own identity");

    bool rejected = false;
    try {
        (void)Package::resolve_weights(ArtifactIdentity{"qwen3.8-27b", "nvfp4"});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "the package resolves another target's identity");

    rejected = false;
    try {
        (void)Package::resolve_weights(
            ArtifactIdentity{std::string(Package::model_id), "gguf-q4-k-m"});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "the package resolves another weights recipe");

    rejected = false;
    try {
        (void)Package::resolve_weights(ArtifactIdentity{std::string(Package::model_id), ""});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "the package resolves an empty weights id");

    const auto defaults = Package::sampling_defaults(Package::model_id);
    require(defaults.thinking.temperature == 1.0F, "the thinking temperature is wrong");
    require(defaults.thinking.top_k == 20, "the thinking top-k is wrong");
    require(defaults.thinking.top_p == 0.95F, "the thinking top-p is wrong");
    require(defaults.non_thinking.temperature == 1.0F, "the non-thinking temperature is wrong");
    require(defaults.non_thinking.top_k == 20, "the non-thinking top-k is wrong");
    require(defaults.non_thinking.top_p == 0.95F, "the non-thinking top-p is wrong");
}

void check_coverage(const flash::BindingPlan& plan, const Reader& reader) {
    const auto& directory = reader.v3_directory();
    std::set<std::string> artifact_text;
    for (const auto& entry : directory.bindings) {
        if (entry.first.starts_with("text/")) { artifact_text.insert(entry.first); }
    }
    for (const auto& entry : directory.part_bindings) {
        if (entry.first.starts_with("text/")) { artifact_text.insert(entry.first); }
    }

    const auto parameters = plan.parameters();
    require(parameters.size() == roles::kTextNameCount,
            "the plan binds " + std::to_string(parameters.size()) + " text parameters; it binds " +
                std::to_string(roles::kTextNameCount));
    require(artifact_text.size() == roles::kTextNameCount,
            "the artifact holds " + std::to_string(artifact_text.size()) +
                " text parameters; the role table expects " +
                std::to_string(roles::kTextNameCount));

    std::set<std::string> bound;
    for (const auto* parameter : parameters) {
        require(!parameter->parts.empty(), "the parameter " + parameter->parameter + " has no part");
        require(artifact_text.count(parameter->parameter) == 1,
                "the plan binds " + parameter->parameter + ", which the artifact does not hold");
        bound.insert(parameter->parameter);
        const auto role = roles::role_of(parameter->parameter);
        require(role.has_value() && *role == parameter->role,
                "the parameter " + parameter->parameter + " does not denote its role");
        if (roles::is_layer_scoped(parameter->role)) {
            const auto layer = roles::layer_of(parameter->parameter);
            require(layer.has_value() && *layer == parameter->layer,
                    "the parameter " + parameter->parameter + " does not denote its layer");
        }
    }
    require(bound == artifact_text,
            "the plan binds a different set of text parameters than the artifact holds");
}

void check_unclaimed(const flash::BindingPlan& plan, const std::map<std::string, std::size_t>& expected) {
    require(plan.unclaimed.by_namespace.size() == expected.size(),
            "the plan reports " + std::to_string(plan.unclaimed.by_namespace.size()) +
                " unclaimed namespaces; it reports " + std::to_string(expected.size()));
    for (const auto& [space, count] : expected) {
        const auto found = plan.unclaimed.by_namespace.find(space);
        require(found != plan.unclaimed.by_namespace.end(),
                "the plan does not report the unclaimed namespace " + space);
        require(found->second.size() == count,
                "the plan reports " + std::to_string(found->second.size()) +
                    " unclaimed " + space + " parameters; the artifact holds " +
                    std::to_string(count));
    }
}

void check_layers(const flash::BindingPlan& plan) {
    for (std::uint32_t layer = 0; layer < 48; ++layer) {
        const auto& layer_plan = plan.layers[layer];
        require(layer_plan.attention_hc.norm.role == roles::Role::AttentionHcNorm,
                "layer " + std::to_string(layer) + " has no attention HC norm");
        require(layer_plan.moe_hc.norm.role == roles::Role::MoeHcNorm,
                "layer " + std::to_string(layer) + " has no MoE HC norm");
        require(layer_plan.moe.router.role == roles::Role::MoeRouter,
                "layer " + std::to_string(layer) + " has no router");
        require(layer_plan.moe.routed_gate.role == roles::Role::MoeRoutedGate,
                "layer " + std::to_string(layer) + " has no routed gate");
        if (layer % 4 == 3) {
            require(std::holds_alternative<flash::QsaPlan>(layer_plan.mixer),
                    "layer " + std::to_string(layer) + " has no QSA mixer");
            const auto& qsa = std::get<flash::QsaPlan>(layer_plan.mixer);
            require(qsa.query.role == roles::Role::QsaQuery,
                    "layer " + std::to_string(layer) + " has no QSA query");
            require(qsa.index_key.role == roles::Role::QsaIndexKey,
                    "layer " + std::to_string(layer) + " has no QSA index key");
        } else {
            require(std::holds_alternative<flash::GdnPlan>(layer_plan.mixer),
                    "layer " + std::to_string(layer) + " has no GDN mixer");
            const auto& gdn = std::get<flash::GdnPlan>(layer_plan.mixer);
            require(gdn.query.role == roles::Role::GdnQuery,
                    "layer " + std::to_string(layer) + " has no GDN query");
            require(gdn.z.role == roles::Role::GdnZ,
                    "layer " + std::to_string(layer) + " has no GDN z");
        }
    }
}

void check_routed_bank(const flash::BindingPlan& plan, std::uint32_t layer) {
    const auto& layer_plan = plan.layers[layer];
    for (const auto* parameter : {&layer_plan.moe.routed_gate, &layer_plan.moe.routed_up,
                                  &layer_plan.moe.routed_down}) {
        require(parameter->parts.size() == 512,
                "the routed bank of layer " + std::to_string(layer) + " holds " +
                    std::to_string(parameter->parts.size()) + " parts; it holds 512");
        const auto& object = object_of(plan, parameter->parts.front());
        require(object.format == NumericFormat::NVFP4,
                "the routed bank of layer " + std::to_string(layer) + " is not NVFP4");
        require(object.layout == StorageLayout::BatchedBlockScaleK16M128x4V1,
                "the routed bank of layer " + std::to_string(layer) +
                    " is not a batched block scale library");
        for (const auto& part : parameter->parts) {
            require(part.object == parameter->parts.front().object,
                    "the routed bank of layer " + std::to_string(layer) +
                        " is spread over several objects");
            require(part.view.matrix_count == 1,
                    "the routed bank of layer " + std::to_string(layer) +
                        " has a part that spans several matrices");
        }
        require(parameter->parts.front().view.matrix_begin == 0,
                "the routed bank of layer " + std::to_string(layer) + " starts at matrix 1");
        require(parameter->parts.back().view.matrix_begin == 511,
                "the routed bank of layer " + std::to_string(layer) + " ends before matrix 512");
    }
    require(layer_plan.moe.routed_gate.parts.front().view.matrix_row_count == 640,
            "the routed gate of layer " + std::to_string(layer) + " does not span 640 rows");
    require(layer_plan.moe.routed_up.parts.front().view.matrix_row_begin == 640,
            "the routed up projection of layer " + std::to_string(layer) +
                " does not start at row 640");
    require(layer_plan.moe.routed_up.parts.front().view.matrix_row_count == 640,
            "the routed up projection of layer " + std::to_string(layer) + " does not span 640 rows");
    require(layer_plan.moe.routed_down.parts.front().view.matrix_row_count == 2560,
            "the routed down projection of layer " + std::to_string(layer) +
                " does not span 2560 rows");

    const auto library_of = [&](const flash::BoundParameter& parameter) {
        require(parameter.library.has_value(),
                "the routed bank of layer " + std::to_string(layer) +
                    " has no expert library descriptor");
        return *parameter.library;
    };
    const auto& gate = library_of(layer_plan.moe.routed_gate);
    require(gate.object == object_of(plan, layer_plan.moe.routed_gate.parts.front()).name,
            "the expert library of the routed gate names another object");
    require(gate.matrices == 512, "the routed gate's expert library does not hold 512 experts");
    require(gate.rows == 1280, "the routed gate's expert library does not hold 1280 rows");
    require(gate.columns == 2560,
            "the routed gate's expert library does not hold 2560 columns");
    require(gate.row_begin == 0, "the routed gate's expert library does not start at row 0");
    require(gate.row_count == 640, "the routed gate's expert library does not span 640 rows");

    const auto& up = library_of(layer_plan.moe.routed_up);
    require(up.object == gate.object,
            "the routed up projection's expert library is another object");
    require(up.matrices == 512 && up.rows == 1280 && up.columns == 2560,
            "the routed up projection's expert library has another geometry");
    require(up.row_begin == 640, "the routed up projection's expert library does not start at 640");
    require(up.row_count == 640, "the routed up projection's expert library does not span 640 rows");

    const auto& down = library_of(layer_plan.moe.routed_down);
    require(down.object != gate.object,
            "the routed down projection's expert library is the gate's object");
    require(down.matrices == 512, "the routed down projection's expert library holds 512 experts");
    require(down.rows == 2560, "the routed down projection's expert library holds 2560 rows");
    require(down.columns == 640,
            "the routed down projection's expert library holds 640 columns");
    require(down.row_begin == 0, "the routed down projection's expert library starts at row 0");
    require(down.row_count == 2560,
            "the routed down projection's expert library does not span 2560 rows");
}

void check_activation_policy(const flash::BindingPlan& plan, std::uint32_t layer) {
    const auto& layer_plan = plan.layers[layer];
    const auto check = [&](const flash::BoundParameter& parameter, const std::string& policy) {
        require(parameter.activation_policy.has_value(),
                "the parameter " + parameter.parameter + " has no activation policy");
        require(*parameter.activation_policy == policy,
                "the parameter " + parameter.parameter + " has the activation policy '" +
                    *parameter.activation_policy + "'; it has '" + policy + "'");
    };
    check(layer_plan.moe.routed_gate, "AllowA4");
    check(layer_plan.moe.routed_up, "AllowA4");
    check(layer_plan.moe.routed_down, "AllowA4");
    check(layer_plan.moe.router, "A16Only");
    check(layer_plan.attention_hc.down, "A16Only");
    const auto& gdn = std::get<flash::GdnPlan>(layer_plan.mixer);
    check(gdn.query, "A16Only");
    require(!layer_plan.attention_hc.norm.activation_policy.has_value(),
            "the attention HC norm has an activation policy; the artifact declares no use edge "
            "for it");
}

void check_activation_divisor(const flash::BindingPlan& plan, std::uint32_t layer) {
    const auto& layer_plan = plan.layers[layer];
    const auto check = [&](const flash::BoundParameter& parameter, std::uint64_t column) {
        const auto found = parameter.auxiliaries.find("activation_input_divisor");
        require(found != parameter.auxiliaries.end(),
                "the parameter " + parameter.parameter + " has no activation input divisor");
        require(found->second.size() == 1,
                "the parameter " + parameter.parameter + " has " +
                    std::to_string(found->second.size()) + " activation input divisors");
        const auto& part = found->second.front();
        require(part.view.matrix_row_begin == layer,
                "the activation input divisor of " + parameter.parameter + " is not the divisor "
                "of layer " + std::to_string(layer));
        require(part.view.column_begin == column,
                "the activation input divisor of " + parameter.parameter + " is the wrong "
                "divisor of its layer");
    };
    check(layer_plan.moe.routed_gate, 0);
    check(layer_plan.moe.routed_up, 0);
    check(layer_plan.moe.routed_down, 1);
}

void check_geometry(const flash::BindingPlan& plan) {
    require_rectangle(plan.token_embedding, 0, 248320, 2560, "the token embedding");
    require_rectangle(plan.output_head, 0, 248320, 2560, "the output head");
    require_rectangle(plan.final_hc.norm, 0, 10240, 1, "the final HC norm");
    require_rectangle(plan.final_hc.down, 0, 320, 10240, "the final HC down projection");
    require_rectangle(plan.final_hc.up, 0, 10240, 320, "the final HC up projection");

    const auto& layer = plan.layers[0];
    require_rectangle(layer.attention_hc.norm, 0, 10240, 1, "the layer 0 attention HC norm");
    require_rectangle(layer.attention_hc.down, 0, 320, 10240,
                      "the layer 0 attention HC down projection");
    require_rectangle(layer.attention_hc.inject, 320, 4, 10240,
                      "the layer 0 attention HC injection projection");
    require_rectangle(layer.attention_hc.up, 0, 10240, 320, "the layer 0 attention HC up projection");
    require_rectangle(layer.moe_hc.down, 0, 320, 10240, "the layer 0 MoE HC down projection");
    require_rectangle(layer.moe_hc.inject, 320, 4, 10240,
                      "the layer 0 MoE HC injection projection");
    require_rectangle(layer.moe.router, 0, 512, 2560, "the layer 0 router");
    require_rectangle(layer.moe.shared_score, 512, 1, 2560, "the layer 0 shared logit");
    require_rectangle(layer.moe.shared_gate, 513, 640, 2560, "the layer 0 shared gate");
    require_rectangle(layer.moe.shared_up, 1153, 640, 2560, "the layer 0 shared up projection");
    require_rectangle(layer.moe.shared_down, 0, 2560, 640, "the layer 0 shared down projection");

    const auto& gdn = std::get<flash::GdnPlan>(layer.mixer);
    require_rectangle(gdn.query, 0, 2048, 2560, "the layer 0 GDN query");
    require_rectangle(gdn.key, 2048, 2048, 2560, "the layer 0 GDN key");
    require_rectangle(gdn.value, 4096, 6144, 2560, "the layer 0 GDN value");
    require_rectangle(gdn.z, 10240, 6144, 2560, "the layer 0 GDN z");
    require_rectangle(gdn.a_projection, 0, 48, 2560, "the layer 0 GDN a projection");
    require_rectangle(gdn.b_projection, 48, 48, 2560, "the layer 0 GDN b projection");
    require_rectangle(gdn.a_log, 0, 48, 1, "the layer 0 GDN a log");
    require_rectangle(gdn.dt_bias, 0, 48, 1, "the layer 0 GDN dt bias");
    require_rectangle(gdn.convolution, 0, 4, 10240, "the layer 0 GDN convolution");
    require_rectangle(gdn.norm, 0, 128, 1, "the layer 0 GDN norm");
    require_rectangle(gdn.output, 0, 2560, 6144, "the layer 0 GDN output projection");

    const auto& qsa = std::get<flash::QsaPlan>(plan.layers[3].mixer);
    require_rectangle(qsa.query, 0, 6144, 2560, "the layer 3 QSA query");
    require_rectangle(qsa.output_gate, 6144, 6144, 2560, "the layer 3 QSA output gate");
    require_rectangle(qsa.key, 0, 512, 2560, "the layer 3 QSA key");
    require_rectangle(qsa.value, 512, 512, 2560, "the layer 3 QSA value");
    require_rectangle(qsa.index_query, 1024, 512, 2560, "the layer 3 QSA index query");
    require_rectangle(qsa.index_key, 1536, 128, 2560, "the layer 3 QSA index key");
    require_rectangle(qsa.output, 0, 2560, 6144, "the layer 3 QSA output projection");
    require_rectangle(qsa.query_norm, 0, 256, 1, "the layer 3 QSA query norm");
    require_rectangle(qsa.key_norm, 0, 256, 1, "the layer 3 QSA key norm");
    require_rectangle(qsa.index_query_norm, 0, 128, 1, "the layer 3 QSA index query norm");
    require_rectangle(qsa.index_key_norm, 0, 128, 1, "the layer 3 QSA index key norm");

    require_rectangle(plan.ple.table, 0, 2500012, 160, "the PLE table");
    require_rectangle(plan.ple.key, 0, 10240, 2560, "the PLE key projection");
    require_rectangle(plan.ple.value, 10240, 2560, 2560, "the PLE value projection");
    require_rectangle(plan.ple.norm_key, 0, 10240, 1, "the PLE key norm");
    require_rectangle(plan.ple.norm_query, 0, 10240, 1, "the PLE query norm");
    require_rectangle(plan.ple.norm_conv, 0, 10240, 1, "the PLE convolution norm");
    require_rectangle(plan.ple.convolution, 0, 4, 10240, "the PLE convolution");
    require(plan.ple.table.parts.size() == 128,
            "the PLE table holds " + std::to_string(plan.ple.table.parts.size()) +
                " shards; it holds 128");
}

template <typename Function>
void expect_rejection(Function&& function, const std::string& what) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("the package accepted " + what);
}

void test_synthetic_artifact() {
    auto artifact = ninfer_flash_next_fixture::build_flash_next_fixture();
    const auto path = ninfer_flash_next_fixture::write_flash_next_fixture(artifact);
    const TemporaryArtifact temporary(path);

    Reader reader(path);
    Binder binder(reader, 1);
    ninfer::EngineOptions options;
    options.tp = 1;
    const auto plan = Package::plan_load(binder, options, Package::WeightsProfile::Nvfp4);
    const auto& bindings = plan.bindings();

    check_coverage(bindings, reader);
    check_layers(bindings);
    check_geometry(bindings);
    check_routed_bank(bindings, 0);
    check_activation_policy(bindings, 0);
    check_activation_divisor(bindings, 0);
    check_unclaimed(bindings, {{"mtp", 2}, {"vision", 1}, {"proposal", 1}});

    require(bindings.objects.size() == 955,
            "the plan binds " + std::to_string(bindings.objects.size()) +
                " objects; it binds 955");

    bool rejected = false;
    try {
        Binder wider(reader, 1);
        ninfer::EngineOptions wider_options;
        wider_options.tp = 2;
        (void)Package::plan_load(wider, wider_options, Package::WeightsProfile::Nvfp4);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "the package accepts a tensor-parallel plan it does not own");

    std::cout << "flash-next load plan: " << bindings.parameters().size()
              << " text parameters bound to " << bindings.objects.size() << " objects\n";
}

void test_unclaimed_text_parameter_is_rejected() {
    auto artifact = ninfer_flash_next_fixture::build_flash_next_fixture();
    FixtureBinding extra;
    extra.name   = "text/layers/0/gdn/unknown";
    extra.whole  = true;
    extra.object = artifact.objects.front().id;
    artifact.bindings.push_back(std::move(extra));

    const auto path = ninfer_flash_next_fixture::write_flash_next_fixture(artifact);
    const TemporaryArtifact temporary(path);
    Reader reader(path);
    ninfer::EngineOptions options;
    options.tp = 1;
    expect_rejection(
        [&] {
            Binder binder(reader, 1);
            (void)Package::plan_load(binder, options, Package::WeightsProfile::Nvfp4);
        },
        "a text parameter it does not claim");
}

void test_foreign_configuration_is_rejected() {
    auto artifact = ninfer_flash_next_fixture::build_flash_next_fixture();
    artifact.config["hidden_size"] = 2561;

    const auto path = ninfer_flash_next_fixture::write_flash_next_fixture(artifact);
    const TemporaryArtifact temporary(path);
    Reader reader(path);
    ninfer::EngineOptions options;
    options.tp = 1;
    expect_rejection(
        [&] {
            Binder binder(reader, 1);
            (void)Package::plan_load(binder, options, Package::WeightsProfile::Nvfp4);
        },
        "a configuration this package does not expect");
}

int test_real_artifact(const char* path) {
    Reader reader(path);
    Binder binder(reader, 1);
    if (!has_frontend_resources(binder)) {
        std::cout << "skip: the real artifact does not carry the frontend resources\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.tp = 1;
    const auto plan = Package::plan_load(binder, options, Package::WeightsProfile::Nvfp4);
    const auto& bindings = plan.bindings();

    check_coverage(bindings, reader);
    check_layers(bindings);
    check_geometry(bindings);
    check_routed_bank(bindings, 0);
    check_activation_policy(bindings, 0);
    check_activation_divisor(bindings, 0);
    check_unclaimed(bindings, {{"mtp", 36}, {"vision", 441}, {"proposal", 1}});

    const auto object_name = [&](const flash::BoundParameter& parameter) {
        return bindings.objects.at(parameter.parts.front().object).name;
    };
    require(object_name(bindings.token_embedding) == "weight/000525",
            "the token embedding is not weight/000525");
    require(object_name(bindings.output_head) == "weight/000526",
            "the output head is not weight/000526");
    require(object_name(bindings.final_hc.norm) == "weight/000527",
            "the final HC norm is not weight/000527");
    require(object_name(bindings.final_hc.down) == "weight/000528",
            "the final HC down projection is not weight/000528");
    require(object_name(bindings.final_hc.up) == "weight/000529",
            "the final HC up projection is not weight/000529");

    const auto& layer0 = bindings.layers[0];
    require(object_name(layer0.attention_hc.norm) == "weight/000530",
            "the layer 0 attention HC norm is not weight/000530");
    require(object_name(layer0.attention_hc.down) == "weight/000226",
            "the layer 0 attention HC down projection is not weight/000226");
    require(object_name(layer0.moe_hc.down) == "weight/000227",
            "the layer 0 MoE HC down projection is not weight/000227");
    require(object_name(layer0.moe.router) == "weight/000228",
            "the layer 0 router is not weight/000228");
    require(object_name(layer0.moe.shared_down) == "weight/000534",
            "the layer 0 shared down projection is not weight/000534");
    const auto& gdn0 = std::get<flash::GdnPlan>(layer0.mixer);
    require(object_name(gdn0.query) == "weight/000229",
            "the layer 0 GDN query is not weight/000229");
    require(object_name(gdn0.key) == "weight/000229",
            "the layer 0 GDN key is not weight/000229");
    require(object_name(gdn0.a_projection) == "weight/000230",
            "the layer 0 GDN a projection is not weight/000230");
    require(object_name(gdn0.output) == "weight/000535",
            "the layer 0 GDN output projection is not weight/000535");
    require(object_name(gdn0.a_log) == "weight/000537",
            "the layer 0 GDN a log is not weight/000537");
    require(object_name(gdn0.dt_bias) == "weight/000538",
            "the layer 0 GDN dt bias is not weight/000538");
    require(object_name(gdn0.convolution) == "weight/000536",
            "the layer 0 GDN convolution is not weight/000536");
    require(object_name(gdn0.norm) == "weight/000539",
            "the layer 0 GDN norm is not weight/000539");

    const auto& qsa3 = std::get<flash::QsaPlan>(bindings.layers[3].mixer);
    require(object_name(qsa3.query) == "weight/000245",
            "the layer 3 QSA query is not weight/000245");
    require(object_name(qsa3.output_gate) == "weight/000245",
            "the layer 3 QSA output gate is not weight/000245");
    require(object_name(qsa3.key) == "weight/000246",
            "the layer 3 QSA key is not weight/000246");
    require(object_name(qsa3.index_key) == "weight/000246",
            "the layer 3 QSA index key is not weight/000246");
    require(object_name(qsa3.output) == "weight/000569",
            "the layer 3 QSA output projection is not weight/000569");
    require(object_name(qsa3.query_norm) == "weight/000570",
            "the layer 3 QSA query norm is not weight/000570");
    require(object_name(qsa3.key_norm) == "weight/000571",
            "the layer 3 QSA key norm is not weight/000571");
    require(object_name(qsa3.index_query_norm) == "weight/000572",
            "the layer 3 QSA index query norm is not weight/000572");
    require(object_name(qsa3.index_key_norm) == "weight/000573",
            "the layer 3 QSA index key norm is not weight/000573");

    require(object_name(bindings.layers[0].moe.routed_gate) == "weight/000000",
            "the layer 0 routed gate is not weight/000000");
    require(object_name(bindings.layers[0].moe.routed_down) == "weight/000001",
            "the layer 0 routed down projection is not weight/000001");
    require(object_name(bindings.layers[1].moe.routed_gate) == "weight/000002",
            "the layer 1 routed gate is not weight/000002");
    require(object_name(bindings.ple.key) == "weight/000236",
            "the PLE key projection is not weight/000236");
    require(bindings.objects.at(bindings.ple.table.parts.front().object).name == "weight/000098",
            "the PLE table does not start at weight/000098");
    require(bindings.objects.at(bindings.ple.table.parts.back().object).name == "weight/000225",
            "the PLE table does not end at weight/000225");

    std::cout << "flash-next load plan: " << bindings.parameters().size()
              << " text parameters bound to " << bindings.objects.size() << " objects, "
              << bindings.unclaimed.by_namespace.at("mtp").size() << " mtp / "
              << bindings.unclaimed.by_namespace.at("vision").size() << " vision / "
              << bindings.unclaimed.by_namespace.at("proposal").size() << " proposal unclaimed\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const bool artifact_mode = argc > 1 && std::string_view(argv[1]) == "artifact";
    try {
        test_identity();
        if (!artifact_mode) {
            test_synthetic_artifact();
            test_unclaimed_text_parameter_is_rejected();
            test_foreign_configuration_is_rejected();
            return 0;
        }
        const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        if (artifact == nullptr || *artifact == '\0') {
            std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
            return 77;
        }
        return test_real_artifact(artifact);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
