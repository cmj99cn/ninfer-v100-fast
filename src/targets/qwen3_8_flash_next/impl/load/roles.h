#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

// Weight roles of the qwen3.8-flash-next artifact.
//
// A role is the semantic identity of one artifact parameter. The role table is the exact
// inverse of the artifact's `text/*` parameter names: every `text/*` binding of the artifact
// denotes exactly one role, and every role denotes exactly one parameter name per layer. The
// names are the artifact's own; this module owns no geometry, no binding decision, no layer
// kind and no milestone beyond the `text/*` namespace.
//
// A layer-scoped role embeds the layer index in its parameter name ("text/layers/17/gdn/query");
// the three global roles -- the token embedding, the output head and the final HC -- do not
// carry a layer. The role table is layer-index agnostic: it spells a well formed name for any
// layer index, and the caller owns the knowledge of which layer owns a role (the PLE roles
// exist in one layer only). The `mtp/*`, `vision/*` and `proposal/*` namespaces of the same
// artifact belong to later milestones and have no role here.
namespace ninfer::targets::qwen3_8_flash_next::detail::roles {

enum class Role : std::uint8_t {
    TokenEmbedding,
    OutputHead,
    FinalHcNorm,
    FinalHcDown,
    FinalHcUp,
    AttentionHcNorm,
    AttentionHcDown,
    AttentionHcInject,
    AttentionHcUp,
    MoeHcNorm,
    MoeHcDown,
    MoeHcInject,
    MoeHcUp,
    GdnQuery,
    GdnKey,
    GdnValue,
    GdnZ,
    GdnAProjection,
    GdnBProjection,
    GdnALog,
    GdnDtBias,
    GdnConvolution,
    GdnNorm,
    GdnOutput,
    QsaQuery,
    QsaKey,
    QsaValue,
    QsaIndexQuery,
    QsaIndexKey,
    QsaOutput,
    QsaOutputGate,
    QsaQueryNorm,
    QsaIndexQueryNorm,
    QsaKeyNorm,
    QsaIndexKeyNorm,
    MoeRouter,
    MoeSharedScore,
    MoeSharedGate,
    MoeSharedUp,
    MoeSharedDown,
    MoeRoutedGate,
    MoeRoutedUp,
    MoeRoutedDown,
    PleTable,
    PleKey,
    PleValue,
    PleNormKey,
    PleNormQuery,
    PleNormConv,
    PleConvolution,
};

// The number of roles. One role per `text/*` parameter name pattern of the artifact.
inline constexpr std::size_t kRoleCount = 50;

// The `text/*` parameter count of the artifact, and the per-pattern counts of its name table.
inline constexpr std::uint32_t kTextBindingCount = 1308;
inline constexpr std::uint32_t kAttentionHcBindingCount = 48;
inline constexpr std::uint32_t kMoeHcBindingCount = 48;
inline constexpr std::uint32_t kGdnBindingCount = 36;
inline constexpr std::uint32_t kQsaBindingCount = 12;
inline constexpr std::uint32_t kMoeBindingCount = 48;
inline constexpr std::uint32_t kPleBindingCount = 1;
inline constexpr std::uint32_t kGlobalBindingCount = 1;

// The Text layers of the artifact and the layer kinds among them.
inline constexpr std::uint32_t kTextLayers = 48;
inline constexpr std::uint32_t kGdnLayers = 36;
inline constexpr std::uint32_t kQsaLayers = 12;

// The layer that owns the PLE parameters.
inline constexpr std::uint32_t kPleLayer = 1;

// Every role, in enumeration order.
[[nodiscard]] std::span<const Role> all_roles() noexcept;

// The artifact parameter name of a role. A layer-scoped role embeds `layer`; the three global
// roles ignore it.
[[nodiscard]] std::string parameter_name(Role role, std::uint32_t layer = 0);

// The role a parameter name denotes, or no role when the name is outside this target's
// `text/*` namespace.
[[nodiscard]] std::optional<Role> role_of(std::string_view name) noexcept;

// The layer a parameter name belongs to, or no layer when the name denotes a global parameter
// or no parameter of this target at all.
[[nodiscard]] std::optional<std::uint32_t> layer_of(std::string_view name) noexcept;

// Whether a role names a per-layer parameter.
[[nodiscard]] bool is_layer_scoped(Role role) noexcept;

} // namespace ninfer::targets::qwen3_8_flash_next::detail::roles
