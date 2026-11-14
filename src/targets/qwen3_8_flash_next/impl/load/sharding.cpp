#include "targets/qwen3_8_flash_next/impl/load/sharding.h"

#include "targets/qwen3_8_flash_next/impl/config.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next::detail::sharding {
namespace {

using Axis  = ShardAxis;
using Split = SplitAxis;
using Role  = roles::Role;

constexpr std::int64_t kHidden      = TextConfig::hidden;
constexpr std::int64_t kVocab       = TextConfig::vocab;
constexpr std::int64_t kResidual    = TextConfig::residual_width;
constexpr std::int64_t kHcLowrank   = TextConfig::hc_lowrank;
constexpr std::int64_t kHcStreams   = TextConfig::hc_streams;
constexpr std::int64_t kHead        = TextConfig::head_dim;
constexpr std::int64_t kIndexHead   = TextConfig::index_head_dim;
constexpr std::int64_t kQuery       = TextConfig::query_width;
constexpr std::int64_t kKv          = TextConfig::kv_width;
constexpr std::int64_t kIndexQuery  = TextConfig::index_query_width;
constexpr std::int64_t kIndexKey    = TextConfig::index_key_width;
constexpr std::int64_t kGdnKey      = TextConfig::gdn_key_width;
constexpr std::int64_t kGdnValue    = TextConfig::gdn_value_width;
constexpr std::int64_t kGdnHead     = TextConfig::gdn_head_dim;
constexpr std::int64_t kGdnControl  = TextConfig::gdn_value_heads;
constexpr std::int64_t kGdnKernel   = TextConfig::gdn_conv_kernel;
constexpr std::int64_t kExperts     = TextConfig::experts;
constexpr std::int64_t kSharedWidth = TextConfig::shared_intermediate;
constexpr std::int64_t kSharedBegin = kExperts + 1;
constexpr std::int64_t kPleKernel   = TextConfig::ple_conv_kernel;
constexpr std::int64_t kPleRows     = TextConfig::ple_shard_rows;
constexpr std::int64_t kBand        = 128;

struct ShardSpec {
    Role role;
    Axis axis;
    Split split;
    std::int64_t begin;
    std::int64_t extent;
    std::int64_t unit;
};

constexpr std::array<ShardSpec, roles::kRoleCount> kShardSpecs{{
    {Role::TokenEmbedding, Axis::Replicated, Split::Output, 0, kVocab, 1},
    {Role::OutputHead, Axis::Vocab, Split::Output, 0, kVocab, 1},
    {Role::FinalHcNorm, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::FinalHcDown, Axis::Replicated, Split::Output, 0, kHcLowrank, 1},
    {Role::FinalHcUp, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::AttentionHcNorm, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::AttentionHcDown, Axis::Replicated, Split::Output, 0, kHcLowrank, 1},
    {Role::AttentionHcInject, Axis::Replicated, Split::Output, 0, kHcStreams, 1},
    {Role::AttentionHcUp, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::MoeHcNorm, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::MoeHcDown, Axis::Replicated, Split::Output, 0, kHcLowrank, 1},
    {Role::MoeHcInject, Axis::Replicated, Split::Output, 0, kHcStreams, 1},
    {Role::MoeHcUp, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::GdnQuery, Axis::Head, Split::Output, 0, kGdnKey, kGdnHead},
    {Role::GdnKey, Axis::Head, Split::Output, kGdnKey, kGdnKey, kGdnHead},
    {Role::GdnValue, Axis::Head, Split::Output, 2 * kGdnKey, kGdnValue, kGdnHead},
    {Role::GdnZ, Axis::Head, Split::Output, 2 * kGdnKey + kGdnValue, kGdnValue, kGdnHead},
    {Role::GdnAProjection, Axis::Replicated, Split::Output, 0, kGdnControl, 1},
    {Role::GdnBProjection, Axis::Replicated, Split::Output, 0, kGdnControl, 1},
    {Role::GdnALog, Axis::Replicated, Split::Output, 0, kGdnControl, 1},
    {Role::GdnDtBias, Axis::Replicated, Split::Output, 0, kGdnControl, 1},
    {Role::GdnConvolution, Axis::Replicated, Split::Output, 0, kGdnKernel, 1},
    {Role::GdnNorm, Axis::Replicated, Split::Output, 0, kGdnHead, 1},
    {Role::GdnOutput, Axis::Head, Split::Input, 0, kGdnValue, kGdnHead},
    {Role::QsaQuery, Axis::Head, Split::Output, 0, kQuery, kHead},
    {Role::QsaKey, Axis::Head, Split::Output, 0, kKv, kHead},
    {Role::QsaValue, Axis::Head, Split::Output, kKv, kKv, kHead},
    {Role::QsaIndexQuery, Axis::Replicated, Split::Output, 2 * kKv, kIndexQuery, kIndexHead},
    {Role::QsaIndexKey, Axis::Replicated, Split::Output, 0, kIndexKey, 1},
    {Role::QsaOutput, Axis::Head, Split::Input, 0, kQuery, kHead},
    {Role::QsaOutputGate, Axis::Head, Split::Output, kQuery, kQuery, kHead},
    {Role::QsaQueryNorm, Axis::Replicated, Split::Output, 0, kHead, 1},
    {Role::QsaIndexQueryNorm, Axis::Replicated, Split::Output, 0, kIndexHead, 1},
    {Role::QsaKeyNorm, Axis::Replicated, Split::Output, 0, kHead, 1},
    {Role::QsaIndexKeyNorm, Axis::Replicated, Split::Output, 0, kIndexHead, 1},
    {Role::MoeRouter, Axis::Replicated, Split::Output, 0, kExperts, 1},
    {Role::MoeSharedScore, Axis::Replicated, Split::Output, 0, 1, 1},
    {Role::MoeSharedGate, Axis::Replicated, Split::Output, kSharedBegin, kSharedWidth, 1},
    {Role::MoeSharedUp, Axis::Replicated, Split::Output, kSharedBegin + kSharedWidth,
     kSharedWidth, 1},
    {Role::MoeSharedDown, Axis::Replicated, Split::Input, 0, kSharedWidth, 1},
    {Role::MoeRoutedGate, Axis::Expert, Split::Batch, 0, kExperts, 1},
    {Role::MoeRoutedUp, Axis::Expert, Split::Batch, 0, kExperts, 1},
    {Role::MoeRoutedDown, Axis::Expert, Split::Batch, 0, kExperts, 1},
    {Role::PleTable, Axis::Replicated, Split::Output, 0, kPleRows, 1},
    {Role::PleKey, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::PleValue, Axis::Replicated, Split::Output, 0, kHidden, 1},
    {Role::PleNormKey, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::PleNormQuery, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::PleNormConv, Axis::Replicated, Split::Output, 0, kResidual, 1},
    {Role::PleConvolution, Axis::Replicated, Split::Output, 0, kPleKernel, 1},
}};

constexpr bool table_matches_the_enumeration() {
    for (std::size_t i = 0; i < kShardSpecs.size(); ++i) {
        if (static_cast<std::size_t>(kShardSpecs[i].role) != i) { return false; }
    }
    return true;
}

static_assert(table_matches_the_enumeration());
static_assert(kShardSpecs.size() == roles::kRoleCount);

} // namespace

ShardMapping shard_mapping(roles::Role role, int tp) {
    if (tp < 1) {
        throw std::invalid_argument("shard_mapping: tp must be >= 1 for " +
                                    roles::parameter_name(role, 0));
    }

    const auto& spec = kShardSpecs[static_cast<std::size_t>(role)];
    ShardMapping mapping{spec.axis, spec.split, {}};
    if (spec.axis == Axis::Replicated) { return mapping; }

    const std::string name = roles::parameter_name(role, 0);
    if (spec.extent % tp != 0) {
        throw std::invalid_argument("shard_mapping: " + name + " spans " +
                                    std::to_string(spec.extent) + " units on the split axis, which "
                                    "is not divisible by tp=" + std::to_string(tp));
    }
    if (spec.begin % kBand != 0) {
        throw std::invalid_argument("shard_mapping: " + name + " starts at " +
                                    std::to_string(spec.begin) + " on the split axis, which is "
                                    "not a " + std::to_string(kBand) + "-row band boundary");
    }

    const std::int64_t per_rank = spec.extent / tp;
    if (per_rank % kBand != 0) {
        throw std::invalid_argument("shard_mapping: " + name + " rank covers " +
                                    std::to_string(per_rank) + " units on the split axis, which "
                                    "is not a whole number of " + std::to_string(kBand) +
                                    "-row bands");
    }

    for (int rank = 0; rank < tp; ++rank) {
        const std::int64_t offset = rank * per_rank;
        if (offset % spec.unit != 0) {
            throw std::invalid_argument("shard_mapping: " + name + " rank " +
                                        std::to_string(rank) + " starts at offset " +
                                        std::to_string(offset) + " on the split axis, which is not "
                                        "a " + std::to_string(spec.unit) + " boundary");
        }
        mapping.ranks.push_back(Shard{spec.begin + offset, spec.begin + offset + per_rank});
    }
    return mapping;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail::sharding
