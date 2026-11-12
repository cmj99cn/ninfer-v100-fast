#include "targets/qwen3_8_flash_next/impl/load/roles.h"

#include <array>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next::detail::roles {
namespace {

struct RoleSpec {
    Role role;
    std::string_view suffix;
    bool layer_scoped;
};

constexpr std::array<RoleSpec, kRoleCount> kRoleSpecs{{
    {Role::TokenEmbedding, "token_embedding", false},
    {Role::OutputHead, "output_head", false},
    {Role::FinalHcNorm, "final_hc/norm", false},
    {Role::FinalHcDown, "final_hc/down", false},
    {Role::FinalHcUp, "final_hc/up", false},
    {Role::AttentionHcNorm, "attention_hc/norm", true},
    {Role::AttentionHcDown, "attention_hc/down", true},
    {Role::AttentionHcInject, "attention_hc/inject", true},
    {Role::AttentionHcUp, "attention_hc/up", true},
    {Role::MoeHcNorm, "moe_hc/norm", true},
    {Role::MoeHcDown, "moe_hc/down", true},
    {Role::MoeHcInject, "moe_hc/inject", true},
    {Role::MoeHcUp, "moe_hc/up", true},
    {Role::GdnQuery, "gdn/query", true},
    {Role::GdnKey, "gdn/key", true},
    {Role::GdnValue, "gdn/value", true},
    {Role::GdnZ, "gdn/z", true},
    {Role::GdnAProjection, "gdn/a_projection", true},
    {Role::GdnBProjection, "gdn/b_projection", true},
    {Role::GdnALog, "gdn/a_log", true},
    {Role::GdnDtBias, "gdn/dt_bias", true},
    {Role::GdnConvolution, "gdn/convolution", true},
    {Role::GdnNorm, "gdn/norm", true},
    {Role::GdnOutput, "gdn/output", true},
    {Role::QsaQuery, "attention/query", true},
    {Role::QsaKey, "attention/key", true},
    {Role::QsaValue, "attention/value", true},
    {Role::QsaIndexQuery, "attention/index_query", true},
    {Role::QsaIndexKey, "attention/index_key", true},
    {Role::QsaOutput, "attention/output", true},
    {Role::QsaOutputGate, "attention/output_gate", true},
    {Role::QsaQueryNorm, "attention/query_norm", true},
    {Role::QsaIndexQueryNorm, "attention/index_query_norm", true},
    {Role::QsaKeyNorm, "attention/key_norm", true},
    {Role::QsaIndexKeyNorm, "attention/index_key_norm", true},
    {Role::MoeRouter, "moe/router", true},
    {Role::MoeSharedScore, "moe/shared_score", true},
    {Role::MoeSharedGate, "moe/shared/gate", true},
    {Role::MoeSharedUp, "moe/shared/up", true},
    {Role::MoeSharedDown, "moe/shared/down", true},
    {Role::MoeRoutedGate, "moe/experts/gate", true},
    {Role::MoeRoutedUp, "moe/experts/up", true},
    {Role::MoeRoutedDown, "moe/experts/down", true},
    {Role::PleTable, "ple/table", true},
    {Role::PleKey, "ple/key", true},
    {Role::PleValue, "ple/value", true},
    {Role::PleNormKey, "ple/norm_key", true},
    {Role::PleNormQuery, "ple/norm_query", true},
    {Role::PleNormConv, "ple/norm_conv", true},
    {Role::PleConvolution, "ple/convolution", true},
}};

constexpr bool table_matches_the_enumeration() {
    for (std::size_t i = 0; i < kRoleSpecs.size(); ++i) {
        if (static_cast<std::size_t>(kRoleSpecs[i].role) != i) { return false; }
    }
    return true;
}

constexpr bool suffixes_are_unique() {
    for (std::size_t i = 0; i < kRoleSpecs.size(); ++i) {
        for (std::size_t j = i + 1; j < kRoleSpecs.size(); ++j) {
            if (kRoleSpecs[i].suffix == kRoleSpecs[j].suffix) { return false; }
        }
    }
    return true;
}

static_assert(table_matches_the_enumeration());
static_assert(suffixes_are_unique());
static_assert(kRoleCount == 50);
static_assert(static_cast<std::size_t>(Role::PleConvolution) + 1 == kRoleCount);

constexpr std::array<Role, kRoleCount> kRoleOrder = [] {
    std::array<Role, kRoleCount> out{};
    for (std::size_t i = 0; i < kRoleSpecs.size(); ++i) { out[i] = kRoleSpecs[i].role; }
    return out;
}();

struct ParsedName {
    Role role;
    std::optional<std::uint32_t> layer;
};

std::optional<ParsedName> parse(std::string_view name) noexcept {
    constexpr std::string_view kTextPrefix   = "text/";
    constexpr std::string_view kLayerPrefix  = "text/layers/";
    std::string_view suffix;
    std::optional<std::uint32_t> layer;
    if (name.starts_with(kLayerPrefix)) {
        const auto rest  = name.substr(kLayerPrefix.size());
        const auto slash = rest.find('/');
        if (slash == std::string_view::npos || slash == 0) { return std::nullopt; }
        std::uint64_t value = 0;
        for (const char digit : rest.substr(0, slash)) {
            if (digit < '0' || digit > '9') { return std::nullopt; }
            value = value * 10 + static_cast<std::uint64_t>(digit - '0');
            if (value > UINT32_MAX) { return std::nullopt; }
        }
        layer  = static_cast<std::uint32_t>(value);
        suffix = rest.substr(slash + 1);
    } else if (name.starts_with(kTextPrefix)) {
        suffix = name.substr(kTextPrefix.size());
    } else {
        return std::nullopt;
    }
    for (const auto& spec : kRoleSpecs) {
        if (spec.suffix != suffix) { continue; }
        if (spec.layer_scoped != layer.has_value()) { continue; }
        return ParsedName{spec.role, layer};
    }
    return std::nullopt;
}

} // namespace

std::span<const Role> all_roles() noexcept { return kRoleOrder; }

std::string parameter_name(Role role, std::uint32_t layer) {
    const auto& spec = kRoleSpecs[static_cast<std::size_t>(role)];
    if (!spec.layer_scoped) { return std::string("text/") + std::string(spec.suffix); }
    return "text/layers/" + std::to_string(layer) + "/" + std::string(spec.suffix);
}

std::optional<Role> role_of(std::string_view name) noexcept {
    const auto parsed = parse(name);
    if (!parsed) { return std::nullopt; }
    return parsed->role;
}

std::optional<std::uint32_t> layer_of(std::string_view name) noexcept {
    const auto parsed = parse(name);
    if (!parsed || !parsed->layer) { return std::nullopt; }
    return parsed->layer;
}

bool is_layer_scoped(Role role) noexcept {
    return kRoleSpecs[static_cast<std::size_t>(role)].layer_scoped;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail::roles
