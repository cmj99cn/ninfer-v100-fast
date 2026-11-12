#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"

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

using ninfer::artifact::Reader;
namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;

struct ExpectedRole {
    roles::Role role;
    std::string_view pattern;
    std::uint32_t count;
};

constexpr ExpectedRole kExpectedRoles[] = {
    {roles::Role::AttentionHcNorm, "text/layers/N/attention_hc/norm", 48},
    {roles::Role::AttentionHcDown, "text/layers/N/attention_hc/down", 48},
    {roles::Role::AttentionHcInject, "text/layers/N/attention_hc/inject", 48},
    {roles::Role::AttentionHcUp, "text/layers/N/attention_hc/up", 48},
    {roles::Role::MoeHcNorm, "text/layers/N/moe_hc/norm", 48},
    {roles::Role::MoeHcDown, "text/layers/N/moe_hc/down", 48},
    {roles::Role::MoeHcInject, "text/layers/N/moe_hc/inject", 48},
    {roles::Role::MoeHcUp, "text/layers/N/moe_hc/up", 48},
    {roles::Role::GdnQuery, "text/layers/N/gdn/query", 36},
    {roles::Role::GdnKey, "text/layers/N/gdn/key", 36},
    {roles::Role::GdnValue, "text/layers/N/gdn/value", 36},
    {roles::Role::GdnZ, "text/layers/N/gdn/z", 36},
    {roles::Role::GdnAProjection, "text/layers/N/gdn/a_projection", 36},
    {roles::Role::GdnBProjection, "text/layers/N/gdn/b_projection", 36},
    {roles::Role::GdnALog, "text/layers/N/gdn/a_log", 36},
    {roles::Role::GdnDtBias, "text/layers/N/gdn/dt_bias", 36},
    {roles::Role::GdnConvolution, "text/layers/N/gdn/convolution", 36},
    {roles::Role::GdnNorm, "text/layers/N/gdn/norm", 36},
    {roles::Role::GdnOutput, "text/layers/N/gdn/output", 36},
    {roles::Role::QsaQuery, "text/layers/N/attention/query", 12},
    {roles::Role::QsaKey, "text/layers/N/attention/key", 12},
    {roles::Role::QsaValue, "text/layers/N/attention/value", 12},
    {roles::Role::QsaIndexQuery, "text/layers/N/attention/index_query", 12},
    {roles::Role::QsaIndexKey, "text/layers/N/attention/index_key", 12},
    {roles::Role::QsaOutput, "text/layers/N/attention/output", 12},
    {roles::Role::QsaOutputGate, "text/layers/N/attention/output_gate", 12},
    {roles::Role::QsaQueryNorm, "text/layers/N/attention/query_norm", 12},
    {roles::Role::QsaIndexQueryNorm, "text/layers/N/attention/index_query_norm", 12},
    {roles::Role::QsaKeyNorm, "text/layers/N/attention/key_norm", 12},
    {roles::Role::QsaIndexKeyNorm, "text/layers/N/attention/index_key_norm", 12},
    {roles::Role::MoeRouter, "text/layers/N/moe/router", 48},
    {roles::Role::MoeSharedScore, "text/layers/N/moe/shared_score", 48},
    {roles::Role::MoeSharedGate, "text/layers/N/moe/shared/gate", 48},
    {roles::Role::MoeSharedUp, "text/layers/N/moe/shared/up", 48},
    {roles::Role::MoeSharedDown, "text/layers/N/moe/shared/down", 48},
    {roles::Role::MoeRoutedGate, "text/layers/N/moe/experts/gate", 48},
    {roles::Role::MoeRoutedUp, "text/layers/N/moe/experts/up", 48},
    {roles::Role::MoeRoutedDown, "text/layers/N/moe/experts/down", 48},
    {roles::Role::PleTable, "text/layers/N/ple/table", 1},
    {roles::Role::PleKey, "text/layers/N/ple/key", 1},
    {roles::Role::PleValue, "text/layers/N/ple/value", 1},
    {roles::Role::PleNormKey, "text/layers/N/ple/norm_key", 1},
    {roles::Role::PleNormQuery, "text/layers/N/ple/norm_query", 1},
    {roles::Role::PleNormConv, "text/layers/N/ple/norm_conv", 1},
    {roles::Role::PleConvolution, "text/layers/N/ple/convolution", 1},
    {roles::Role::TokenEmbedding, "text/token_embedding", 1},
    {roles::Role::OutputHead, "text/output_head", 1},
    {roles::Role::FinalHcNorm, "text/final_hc/norm", 1},
    {roles::Role::FinalHcDown, "text/final_hc/down", 1},
    {roles::Role::FinalHcUp, "text/final_hc/up", 1},
};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string pattern_of(std::string_view name) {
    constexpr std::string_view kPrefix = "text/layers/";
    if (!name.starts_with(kPrefix)) { return std::string(name); }
    const auto rest = name.substr(kPrefix.size());
    const auto slash = rest.find('/');
    if (slash == std::string_view::npos) { return std::string(name); }
    std::string out(kPrefix);
    out += "N";
    out += rest.substr(slash);
    return out;
}

void test_role_set_matches_the_pattern_table() {
    require(roles::all_roles().size() == std::size(kExpectedRoles),
            "the role enumeration holds " + std::to_string(roles::all_roles().size()) +
                " roles; the artifact pattern table holds " +
                std::to_string(std::size(kExpectedRoles)));

    std::set<std::string> expected;
    std::set<roles::Role> expected_roles;
    std::uint64_t expected_bindings = 0;
    for (const auto& entry : kExpectedRoles) {
        require(expected.insert(std::string(entry.pattern)).second,
                "the pattern table repeats " + std::string(entry.pattern));
        require(expected_roles.insert(entry.role).second,
                "the pattern table repeats a role");
        expected_bindings += entry.count;
    }
    require(expected_bindings == roles::kTextNameCount,
            "the pattern table counts " + std::to_string(expected_bindings) +
                " text parameters; the artifact declares " +
                std::to_string(roles::kTextNameCount));

    std::set<std::string> produced;
    for (const auto role : roles::all_roles()) {
        produced.insert(pattern_of(roles::parameter_name(role, 17)));
    }
    require(produced == expected, "the role enumeration does not match the artifact pattern table");

    for (const auto& entry : kExpectedRoles) {
        const auto produced_pattern = pattern_of(roles::parameter_name(entry.role, 17));
        require(produced_pattern == entry.pattern,
                "the role for " + std::string(entry.pattern) + " spells " + produced_pattern);
    }
}

void test_parameter_names_of_a_layer() {
    require(roles::parameter_name(roles::Role::GdnQuery, 0) == "text/layers/0/gdn/query",
            "the layer 0 GDN query name is wrong");
    require(roles::parameter_name(roles::Role::AttentionHcInject, 0) ==
                "text/layers/0/attention_hc/inject",
            "the layer 0 attention HC injection name is wrong");
    require(roles::parameter_name(roles::Role::QsaQuery, 3) == "text/layers/3/attention/query",
            "the layer 3 QSA query name is wrong");
    require(roles::parameter_name(roles::Role::MoeHcInject, 3) == "text/layers/3/moe_hc/inject",
            "the layer 3 MoE HC injection name is wrong");
    require(roles::parameter_name(roles::Role::MoeRoutedGate, 47) ==
                "text/layers/47/moe/experts/gate",
            "the layer 47 routed gate name is wrong");
    require(roles::parameter_name(roles::Role::PleConvolution, 1) ==
                "text/layers/1/ple/convolution",
            "the PLE convolution name is wrong");
    require(roles::parameter_name(roles::Role::TokenEmbedding) == "text/token_embedding",
            "the embedding name is wrong");
    require(roles::parameter_name(roles::Role::OutputHead) == "text/output_head",
            "the output head name is wrong");
    require(roles::parameter_name(roles::Role::FinalHcNorm) == "text/final_hc/norm",
            "the final HC norm name is wrong");
    require(roles::parameter_name(roles::Role::FinalHcDown) == "text/final_hc/down",
            "the final HC down name is wrong");
    require(roles::parameter_name(roles::Role::FinalHcUp) == "text/final_hc/up",
            "the final HC up name is wrong");
}

void test_layer_of_and_role_of() {
    require(roles::layer_of("text/layers/17/gdn/query") == 17, "layer 17 is not recovered");
    require(roles::layer_of("text/layers/0/attention/query") == 0, "layer 0 is not recovered");
    require(roles::layer_of("text/layers/47/moe/experts/down") == 47, "layer 47 is not recovered");
    require(roles::layer_of("text/layers/1/ple/table") == 1, "the PLE layer is not recovered");
    require(!roles::layer_of("text/token_embedding").has_value(),
            "the embedding is not a layer parameter");
    require(!roles::layer_of("text/final_hc/norm").has_value(),
            "the final HC norm is not a layer parameter");
    require(!roles::layer_of("text/output_head").has_value(),
            "the output head is not a layer parameter");
    require(!roles::layer_of("mtp/output_head").has_value(),
            "an MTP parameter is not a layer parameter");
    require(!roles::layer_of("vision/patch_embedding").has_value(),
            "a vision parameter is not a layer parameter");
    require(!roles::layer_of("text/layers/17/attention/none").has_value(),
            "an unknown parameter is not a layer parameter");
    require(!roles::layer_of("text/layers/x/gdn/query").has_value(),
            "a malformed layer index is not a layer parameter");
    require(!roles::layer_of("text/layers/99999999999999999999/gdn/query").has_value(),
            "an overflowing layer index is not a layer parameter");

    require(roles::role_of("text/layers/17/moe/experts/down") == roles::Role::MoeRoutedDown,
            "the routed down role is not recovered");
    require(roles::role_of("text/layers/3/attention/output_gate") == roles::Role::QsaOutputGate,
            "the QSA output gate role is not recovered");
    require(roles::role_of("text/final_hc/up") == roles::Role::FinalHcUp,
            "the final HC up role is not recovered");
    require(!roles::role_of("text/layers/17/moe/experts/none").has_value(),
            "an unknown parameter has no role");
    require(!roles::role_of("mtp/layers/0/attention/query").has_value(),
            "an MTP parameter has no role in this target");
    require(!roles::role_of("text").has_value(), "a prefix has no role");
}

std::string layers_text(const std::set<std::uint32_t>& layers) {
    std::string out = "[";
    for (const auto layer : layers) {
        if (out.size() > 1) { out += ","; }
        out += std::to_string(layer);
    }
    out += "]";
    return out;
}

std::set<std::uint32_t> expected_layers_of(std::string_view pattern,
                                            const std::set<std::uint32_t>& every_layer) {
    constexpr std::string_view kLayerPrefix = "text/layers/N/";
    if (!pattern.starts_with(kLayerPrefix)) { return {}; }
    const auto rest     = pattern.substr(kLayerPrefix.size());
    const auto slash    = rest.find('/');
    const auto family   = rest.substr(0, slash);
    std::set<std::uint32_t> layers;
    if (family == "gdn") {
        for (const auto layer : every_layer) {
            if (layer % 4 != 3) { layers.insert(layer); }
        }
    } else if (family == "attention") {
        for (const auto layer : every_layer) {
            if (layer % 4 == 3) { layers.insert(layer); }
        }
    } else if (family == "ple") {
        layers.insert(roles::kPleLayer);
    } else {
        layers = every_layer;
    }
    return layers;
}

void test_real_artifact_binding_coverage(const char* artifact) {
    Reader reader(artifact);
    const auto& directory = reader.v3_directory();

    std::set<std::string> text_parameters;
    for (const auto& entry : directory.bindings) {
        if (entry.first.starts_with("text/")) { text_parameters.insert(entry.first); }
    }
    for (const auto& entry : directory.part_bindings) {
        if (entry.first.starts_with("text/")) { text_parameters.insert(entry.first); }
    }
    require(text_parameters.size() == roles::kTextNameCount,
            "the artifact holds " + std::to_string(text_parameters.size()) +
                " text parameters; the role table expects " +
                std::to_string(roles::kTextNameCount));

    std::map<roles::Role, std::uint32_t> claimed;
    std::map<roles::Role, std::set<std::uint32_t>> claimed_layers;
    std::set<std::uint32_t> artifact_layers;
    for (const auto& name : text_parameters) {
        const auto role = roles::role_of(name);
        require(role.has_value(), "the artifact parameter " + name + " has no role");
        ++claimed[*role];
        const auto layer = roles::layer_of(name);
        if (roles::is_layer_scoped(*role)) {
            require(layer.has_value(), "the artifact parameter " + name + " has no layer");
            claimed_layers[*role].insert(*layer);
            artifact_layers.insert(*layer);
        } else {
            require(!layer.has_value(),
                    "the global artifact parameter " + name + " names a layer");
        }
    }
    for (const auto& entry : kExpectedRoles) {
        const auto found = claimed.find(entry.role);
        require(found != claimed.end(),
                "the artifact holds no parameter for the role of " + std::string(entry.pattern));
        require(found->second == entry.count,
                "the role of " + std::string(entry.pattern) + " claims " +
                    std::to_string(found->second) + " artifact parameters; it should claim " +
                    std::to_string(entry.count));
    }
    require(claimed.size() == std::size(kExpectedRoles),
            "the artifact parameters do not cover every role");

    const std::uint32_t layer_count = *artifact_layers.rbegin() + 1;
    require(layer_count == 48,
            "the artifact holds " + std::to_string(layer_count) + " layers; it holds 48");
    std::set<std::uint32_t> every_layer;
    for (std::uint32_t layer = 0; layer < layer_count; ++layer) { every_layer.insert(layer); }
    require(artifact_layers == every_layer, "the artifact layers are not a contiguous range");

    for (const auto& entry : kExpectedRoles) {
        const auto expected_layers = expected_layers_of(entry.pattern, every_layer);
        const auto found = claimed_layers.find(entry.role);
        const auto actual = found == claimed_layers.end() ? std::set<std::uint32_t>{}
                                                          : found->second;
        require(actual == expected_layers,
                "the role of " + std::string(entry.pattern) + " occupies " +
                    std::to_string(actual.size()) + " layers " + layers_text(actual) +
                    "; it should occupy " + std::to_string(expected_layers.size()) + " layers " +
                    layers_text(expected_layers));
    }

    std::set<std::string> generated;
    for (const auto role : roles::all_roles()) {
        for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
            generated.insert(roles::parameter_name(role, layer));
        }
    }
    for (const auto& name : text_parameters) {
        require(generated.count(name) == 1, "the role table does not spell the parameter " + name);
    }
    std::size_t present = 0;
    for (const auto& name : generated) {
        if (text_parameters.count(name) != 0) { ++present; }
    }
    require(present == roles::kTextNameCount,
            "the role table spells " + std::to_string(present) +
                " artifact parameters; the artifact holds " +
                std::to_string(roles::kTextNameCount));

    std::cout << "flash-next roles: " << text_parameters.size() << " text parameters claim "
              << claimed.size() << " roles exactly, over " << layer_count << " layers\n";
}

} // namespace

int main(int argc, char** argv) {
    const bool artifact_mode = argc > 1 && std::string_view(argv[1]) == "artifact";
    try {
        test_role_set_matches_the_pattern_table();
        test_parameter_names_of_a_layer();
        test_layer_of_and_role_of();
        if (!artifact_mode) { return 0; }
        const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        if (artifact == nullptr || *artifact == '\0') {
            std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
            return 77;
        }
        test_real_artifact_binding_coverage(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
