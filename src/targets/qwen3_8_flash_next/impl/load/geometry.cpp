#include "targets/qwen3_8_flash_next/impl/load/geometry.h"

#include "targets/qwen3_8_flash_next/impl/config.h"

#include <array>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

using artifact::NumericFormat;
using artifact::StorageLayout;

constexpr int kHidden            = TextConfig::hidden;
constexpr int kVocab             = TextConfig::vocab;
constexpr int kResidual          = TextConfig::residual_width;
constexpr int kHcLowrank         = TextConfig::hc_lowrank;
constexpr int kHcStreams         = TextConfig::hc_streams;
constexpr int kQueryWidth        = TextConfig::query_width;
constexpr int kKvWidth           = TextConfig::kv_width;
constexpr int kIndexQueryWidth   = TextConfig::index_query_width;
constexpr int kIndexKeyWidth     = TextConfig::index_key_width;
constexpr int kGdnKeyWidth       = TextConfig::gdn_key_width;
constexpr int kGdnValueWidth     = TextConfig::gdn_value_width;
constexpr int kGdnControlRows    = TextConfig::gdn_value_heads;
constexpr int kExperts           = TextConfig::experts;
constexpr int kExpertWidth       = TextConfig::expert_intermediate;
constexpr int kSharedWidth       = TextConfig::shared_intermediate;
constexpr int kPleShardRows      = TextConfig::ple_shard_rows;
constexpr int kPleShardColumns   = TextConfig::ple_head_width;
constexpr int kPleShards         = TextConfig::ple_shards;
constexpr int kConvKernel        = TextConfig::gdn_conv_kernel;
constexpr int kPleConvKernel     = TextConfig::ple_conv_kernel;

constexpr std::uint32_t rank_of(std::array<std::uint64_t, 3> shape) {
    return shape[2] != 0 ? 3u : (shape[1] != 0 ? 2u : 1u);
}

constexpr RoleGeometry make(NumericFormat format, StorageLayout layout,
                            std::array<std::uint64_t, 3> shape, std::uint64_t row_begin,
                            std::uint64_t row_count, std::uint64_t column_begin,
                            std::uint64_t column_count, std::uint64_t matrix_count = 1,
                            std::uint64_t part_count = 1) {
    return RoleGeometry{format, layout, rank_of(shape), shape, row_begin, row_count, column_begin,
                        column_count, matrix_count, part_count};
}

constexpr RoleGeometry dense(std::array<std::uint64_t, 3> shape, std::uint64_t row_begin,
                             std::uint64_t row_count, std::uint64_t column_count) {
    return make(NumericFormat::BF16, StorageLayout::ContiguousLeV1, shape, row_begin, row_count, 0,
                column_count);
}

constexpr RoleGeometry vector_of(std::array<std::uint64_t, 3> shape) {
    return dense(shape, 0, shape[0], 1);
}

constexpr RoleGeometry fp32_vector(std::array<std::uint64_t, 3> shape) {
    return make(NumericFormat::FP32, StorageLayout::ContiguousLeV1, shape, 0, shape[0], 0, 1);
}

constexpr RoleGeometry dense_rows(std::array<std::uint64_t, 3> shape, std::uint64_t row_begin,
                                  std::uint64_t row_count) {
    return dense(shape, row_begin, row_count, shape[1]);
}

constexpr RoleGeometry w8_rows(std::array<std::uint64_t, 3> shape, std::uint64_t row_begin,
                                std::uint64_t row_count) {
    return make(NumericFormat::W8G32_F16S, StorageLayout::RowSplitK128V1, shape, row_begin,
                row_count, 0, shape[1]);
}

constexpr RoleGeometry bank(std::array<std::uint64_t, 3> shape, std::uint64_t row_begin,
                             std::uint64_t row_count, std::uint64_t column_count) {
    return make(NumericFormat::NVFP4, StorageLayout::BatchedBlockScaleK16M128x4V1, shape,
                row_begin, row_count, 0, column_count, shape[0], shape[0]);
}

constexpr RoleGeometry paged(std::array<std::uint64_t, 3> shape, std::uint64_t row_begin,
                             std::uint64_t row_count, std::uint64_t part_count) {
    return make(NumericFormat::NF4_G16_E4M3S, StorageLayout::RowPagedG16x4KV1, shape, row_begin,
                row_count, 0, shape[1], 1, part_count);
}

} // namespace

RoleGeometry role_geometry(roles::Role role) noexcept {
    switch (role) {
    case roles::Role::TokenEmbedding:
        return dense({kVocab, kHidden}, 0, kVocab, kHidden);
    case roles::Role::OutputHead:
        return dense({kVocab, kHidden}, 0, kVocab, kHidden);
    case roles::Role::FinalHcNorm:
        return vector_of({kResidual});
    case roles::Role::FinalHcDown:
        return dense({kHcLowrank, kResidual}, 0, kHcLowrank, kResidual);
    case roles::Role::FinalHcUp:
        return dense({kResidual, kHcLowrank}, 0, kResidual, kHcLowrank);
    case roles::Role::AttentionHcNorm:
        return vector_of({kResidual});
    case roles::Role::AttentionHcDown:
        return dense_rows({kHcLowrank + kHcStreams, kResidual}, 0, kHcLowrank);
    case roles::Role::AttentionHcInject:
        return dense_rows({kHcLowrank + kHcStreams, kResidual}, kHcLowrank, kHcStreams);
    case roles::Role::AttentionHcUp:
        return dense({kResidual, kHcLowrank}, 0, kResidual, kHcLowrank);
    case roles::Role::MoeHcNorm:
        return vector_of({kResidual});
    case roles::Role::MoeHcDown:
        return dense_rows({kHcLowrank + kHcStreams, kResidual}, 0, kHcLowrank);
    case roles::Role::MoeHcInject:
        return dense_rows({kHcLowrank + kHcStreams, kResidual}, kHcLowrank, kHcStreams);
    case roles::Role::MoeHcUp:
        return dense({kResidual, kHcLowrank}, 0, kResidual, kHcLowrank);
    case roles::Role::GdnQuery:
        return w8_rows({2 * (kGdnKeyWidth + kGdnValueWidth), kHidden}, 0, kGdnKeyWidth);
    case roles::Role::GdnKey:
        return w8_rows({2 * (kGdnKeyWidth + kGdnValueWidth), kHidden}, kGdnKeyWidth, kGdnKeyWidth);
    case roles::Role::GdnValue:
        return w8_rows({2 * (kGdnKeyWidth + kGdnValueWidth), kHidden}, 2 * kGdnKeyWidth,
                       kGdnValueWidth);
    case roles::Role::GdnZ:
        return w8_rows({2 * (kGdnKeyWidth + kGdnValueWidth), kHidden},
                       2 * kGdnKeyWidth + kGdnValueWidth, kGdnValueWidth);
    case roles::Role::GdnAProjection:
        return dense_rows({2 * kGdnControlRows, kHidden}, 0, kGdnControlRows);
    case roles::Role::GdnBProjection:
        return dense_rows({2 * kGdnControlRows, kHidden}, kGdnControlRows, kGdnControlRows);
    case roles::Role::GdnALog:
        return fp32_vector({kGdnControlRows});
    case roles::Role::GdnDtBias:
        return fp32_vector({kGdnControlRows});
    case roles::Role::GdnConvolution:
        return dense({kConvKernel, kResidual}, 0, kConvKernel, kResidual);
    case roles::Role::GdnNorm:
        return vector_of({TextConfig::gdn_head_dim});
    case roles::Role::GdnOutput:
        return w8_rows({kHidden, kGdnValueWidth}, 0, kHidden);
    case roles::Role::QsaQuery:
        return w8_rows({2 * kQueryWidth, kHidden}, 0, kQueryWidth);
    case roles::Role::QsaKey:
        return dense_rows({2 * kKvWidth + kIndexQueryWidth + kIndexKeyWidth, kHidden}, 0,
                          kKvWidth);
    case roles::Role::QsaValue:
        return dense_rows({2 * kKvWidth + kIndexQueryWidth + kIndexKeyWidth, kHidden}, kKvWidth,
                          kKvWidth);
    case roles::Role::QsaIndexQuery:
        return dense_rows({2 * kKvWidth + kIndexQueryWidth + kIndexKeyWidth, kHidden},
                          2 * kKvWidth, kIndexQueryWidth);
    case roles::Role::QsaIndexKey:
        return dense_rows({2 * kKvWidth + kIndexQueryWidth + kIndexKeyWidth, kHidden},
                          2 * kKvWidth + kIndexQueryWidth, kIndexKeyWidth);
    case roles::Role::QsaOutput:
        return w8_rows({kHidden, kQueryWidth}, 0, kHidden);
    case roles::Role::QsaOutputGate:
        return w8_rows({2 * kQueryWidth, kHidden}, kQueryWidth, kQueryWidth);
    case roles::Role::QsaQueryNorm:
        return vector_of({TextConfig::head_dim});
    case roles::Role::QsaIndexQueryNorm:
        return vector_of({TextConfig::index_head_dim});
    case roles::Role::QsaKeyNorm:
        return vector_of({TextConfig::head_dim});
    case roles::Role::QsaIndexKeyNorm:
        return vector_of({TextConfig::index_head_dim});
    case roles::Role::MoeRouter:
        return dense_rows({kExperts + 1 + 2 * kSharedWidth, kHidden}, 0, kExperts);
    case roles::Role::MoeSharedScore:
        return dense_rows({kExperts + 1 + 2 * kSharedWidth, kHidden}, kExperts, 1);
    case roles::Role::MoeSharedGate:
        return dense_rows({kExperts + 1 + 2 * kSharedWidth, kHidden}, kExperts + 1, kSharedWidth);
    case roles::Role::MoeSharedUp:
        return dense_rows({kExperts + 1 + 2 * kSharedWidth, kHidden}, kExperts + 1 + kSharedWidth,
                          kSharedWidth);
    case roles::Role::MoeSharedDown:
        return dense({kHidden, kSharedWidth}, 0, kHidden, kSharedWidth);
    case roles::Role::MoeRoutedGate:
        return bank({kExperts, 2 * kExpertWidth, kHidden}, 0, kExpertWidth, kHidden);
    case roles::Role::MoeRoutedUp:
        return bank({kExperts, 2 * kExpertWidth, kHidden}, kExpertWidth, kExpertWidth, kHidden);
    case roles::Role::MoeRoutedDown:
        return bank({kExperts, kHidden, kExpertWidth}, 0, kHidden, kExpertWidth);
    case roles::Role::PleTable:
        return paged({kPleShardRows, kPleShardColumns}, 0, kPleShardRows, kPleShards);
    case roles::Role::PleKey:
        return dense_rows({kResidual + kHidden, kHidden}, 0, kResidual);
    case roles::Role::PleValue:
        return dense_rows({kResidual + kHidden, kHidden}, kResidual, kHidden);
    case roles::Role::PleNormKey:
        return vector_of({kResidual});
    case roles::Role::PleNormQuery:
        return vector_of({kResidual});
    case roles::Role::PleNormConv:
        return vector_of({kResidual});
    case roles::Role::PleConvolution:
        return dense({kPleConvKernel, kResidual}, 0, kPleConvKernel, kResidual);
    }
    return RoleGeometry{};
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
