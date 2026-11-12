#include "artifact/reader.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace ninfer::artifact {
namespace {

constexpr std::uint64_t kTensorAlignment = 256;
constexpr std::uint64_t kKAlignment      = 128;

// blockscale-k16-m128x4-v1 physical tiling constants (storage-layouts.md section 4, and
// tools/artifact/layouts.py swizzle_nvfp4_scales, which is the encoder of record).
constexpr std::uint64_t kNvfp4RowTile      = 128; // rows per scale-plane tile row
constexpr std::uint64_t kNvfp4TileColumns  = 64;  // logical columns per scale tile (4 groups of 16)
constexpr std::uint64_t kNvfp4TileBytes    = 512; // 128 rows x 4 lanes of one scale tile
constexpr std::uint64_t kNvfp4DivisorBytes = 4;

constexpr std::uint64_t kNf4GroupSize        = 16;
constexpr std::uint64_t kNf4PageBytes        = 4096;
constexpr std::uint64_t kNf4MatrixScaleBytes = 4;

constexpr std::uint64_t kBlock128Tile       = 128;
constexpr std::uint64_t kBlock128ScaleBytes = 2;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

struct QuantGeometry {
    std::uint64_t group_size;
    std::uint64_t base_bytes_per_group;
    std::uint64_t high_bytes_per_group;
};

QuantGeometry quant_geometry(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q4G64_F16S:
        return {64, 32, 0};
    case NumericFormat::Q5G64_F16S:
        return {64, 32, 8};
    case NumericFormat::Q6G64_F16S:
        return {64, 32, 16};
    case NumericFormat::W8G32_F16S:
        return {32, 32, 0};
    default:
        throw ArtifactError("row-split-k128-v1 requires a grouped quantized format");
    }
}

std::uint64_t direct_word_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
    case NumericFormat::FP16:
        return 2;
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return 4;
    default:
        throw ArtifactError("contiguous-le-v1 requires BF16, FP16, FP32, or I32");
    }
}

} // namespace

std::uint64_t ggml_k_code_offset(std::uint64_t rows) {
    return align_up(checked_mul(rows, 8, "GGML row descriptors"), 256, "GGML code offset");
}

namespace {
std::uint64_t ggml_row(std::span<const std::byte> payload, std::uint64_t row) {
    std::uint64_t value = 0;
    for (unsigned byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned>(payload[row * 8 + byte])) << (byte * 8);
    }
    return value;
}

void set_ggml_row(std::vector<std::byte>& prefix, std::uint64_t row, std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte) {
        prefix[row * 8 + byte] = static_cast<std::byte>((value >> (byte * 8)) & 255);
    }
}
}

void validate_ggml_k_payload(std::span<const std::uint64_t> shape,
                             std::span<const std::byte> payload) {
    tensor_encoded_size(StorageLayout::GgmlK256V1, NumericFormat::GGML_K, shape, payload.size());
    std::uint64_t cursor = 0;
    for (std::uint64_t row = 0; row < shape[0]; ++row) {
        const auto descriptor = ggml_row(payload, row);
        if ((descriptor >> 1) != cursor) {
            throw ArtifactError("GGML_K row offsets are not canonical contiguous rows");
        }
        cursor = checked_add(cursor, (shape[1] / 256) * ((descriptor & 1) ? 210 : 144), "GGML row bytes");
    }
    const auto code_offset = ggml_k_code_offset(shape[0]);
    if (checked_add(code_offset, cursor, "GGML payload bytes") != payload.size()) {
        throw ArtifactError("GGML_K descriptors disagree with payload byte size");
    }
    for (auto offset = shape[0] * 8; offset < code_offset; ++offset) {
        if (payload[offset] != std::byte{0}) { throw ArtifactError("GGML_K descriptor padding is not zero"); }
    }
}

std::string_view format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16:
        return "BF16";
    case NumericFormat::FP16:
        return "FP16";
    case NumericFormat::FP32:
        return "FP32";
    case NumericFormat::I32:
        return "I32";
    case NumericFormat::Q4G64_F16S:
        return "Q4G64_F16S";
    case NumericFormat::Q5G64_F16S:
        return "Q5G64_F16S";
    case NumericFormat::Q6G64_F16S:
        return "Q6G64_F16S";
    case NumericFormat::W8G32_F16S:
        return "W8G32_F16S";
    case NumericFormat::NVFP4:
        return "NVFP4";
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return "FP8_E4M3FN_ROW_BF16S";
    case NumericFormat::GGML_K:
        return "GGML_K";
    case NumericFormat::NF4_G16_E4M3S:
        return "nf4_g16_e4m3fn";
    case NumericFormat::FP8_E4M3FN_BLOCK128_BF16S:
        return "fp8_e4m3fn_block128x128_bf16";
    }
    return {};
}

std::string_view layout_name(StorageLayout layout) noexcept {
    switch (layout) {
    case StorageLayout::ContiguousLeV1:
        return "contiguous-le-v1";
    case StorageLayout::RowSplitK128V1:
        return "row-split-k128-v1";
    case StorageLayout::BlockScaleK16M128x4V1:
        return "blockscale-k16-m128x4-v1";
    case StorageLayout::RowScaleV1:
        return "row-scale-v1";
    case StorageLayout::GgmlK256V1:
        return "ggml-k256-v1";
    case StorageLayout::RowPagedG16x4KV1:
        return "row_paged_g16_4k_v1";
    case StorageLayout::BlockScaleM128K128V1:
        return "block_scale_m128_k128_v1";
    case StorageLayout::BatchedBlockScaleK16M128x4V1:
        return "batched_block_scale_k16_m128x4_v1";
    }
    return {};
}

std::string_view encoding_name(ResourceEncoding encoding) noexcept {
    switch (encoding) {
    case ResourceEncoding::RawBytesV1:
        return "raw-bytes-v1";
    }
    return {};
}

std::uint64_t tensor_alignment(StorageLayout) noexcept { return kTensorAlignment; }

std::uint64_t resource_alignment(ResourceEncoding) noexcept { return 1; }

std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape,
                                  std::uint64_t stored_bytes) {
    if (layout == StorageLayout::GgmlK256V1) {
        if (format != NumericFormat::GGML_K || shape.size() != 2 || shape[0] == 0 ||
            shape[1] == 0 || shape[1] % 256 != 0) {
            throw ArtifactError("ggml-k256-v1 requires GGML_K [N,K] with K divisible by 256");
        }
        const auto blocks = checked_mul(shape[0], shape[1] / 256, "GGML K blocks");
        const auto prefix = ggml_k_code_offset(shape[0]);
        const auto minimum = checked_add(prefix, checked_mul(blocks, 144, "GGML Q4 bytes"), "GGML bytes");
        const auto maximum = checked_add(prefix, checked_mul(blocks, 210, "GGML Q6 bytes"), "GGML bytes");
        if (stored_bytes < minimum || stored_bytes > maximum ||
            (stored_bytes - minimum) % ((shape[1] / 256) * 66) != 0) {
            throw ArtifactError("ggml-k256-v1 byte size does not describe whole Q4_K/Q6_K rows");
        }
        return stored_bytes;
    }
    if (layout == StorageLayout::ContiguousLeV1) {
        if (shape.size() > 16) {
            throw ArtifactError("contiguous-le-v1 supports rank 0 through 16");
        }
        std::uint64_t elements = 1;
        for (const auto dim : shape) {
            if (dim == 0) { throw ArtifactError("tensor shape dimensions must be positive"); }
            elements = checked_mul(elements, dim, "tensor element count");
        }
        return checked_mul(elements, direct_word_bytes(format), "tensor encoded size");
    }

    if (layout == StorageLayout::RowSplitK128V1) {
        if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
            throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
        }
        return row_split_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        return block_scale_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::RowScaleV1) {
        return row_scale_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::RowPagedG16x4KV1) {
        return row_paged_nf4_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::BlockScaleM128K128V1) {
        return block_scale_m128_k128_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::BatchedBlockScaleK16M128x4V1) {
        return batched_block_scale_geometry(format, shape).encoded_bytes;
    }
    throw ArtifactError("unknown tensor layout");
}

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
    }
    const auto format_geometry = quant_geometry(format);
    RowSplitGeometry out;
    out.rows                 = shape[0];
    out.columns              = shape[1];
    out.padded_columns       = align_up(shape[1], kKAlignment, "padded K");
    out.group_size           = format_geometry.group_size;
    out.groups_per_row       = out.padded_columns / out.group_size;
    out.low_bytes_per_group  = format_geometry.base_bytes_per_group;
    out.high_bytes_per_group = format_geometry.high_bytes_per_group;
    const auto groups        = checked_mul(out.rows, out.groups_per_row, "physical group count");
    out.low_plane_bytes      = checked_mul(groups, out.low_bytes_per_group, "base plane bytes");
    out.high_plane_bytes     = checked_mul(groups, out.high_bytes_per_group, "high plane bytes");
    out.scale_plane_bytes    = checked_mul(groups, 2, "scale plane bytes");
    out.high_plane_offset    = align_up(out.low_plane_bytes, kTensorAlignment, "high plane offset");
    const auto aligned_high =
        align_up(out.high_plane_bytes, kTensorAlignment, "scale plane alignment");
    out.scale_plane_offset = checked_add(out.high_plane_offset, aligned_high, "scale plane offset");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "tensor encoded size");
    return out;
}

BlockScaleGeometry block_scale_geometry(NumericFormat format,
                                        std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::NVFP4) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires NVFP4");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires a positive rank-two shape");
    }
    if (shape[0] % 128 != 0 || shape[1] % 64 != 0) {
        throw ArtifactError(
            "blockscale-k16-m128x4-v1 requires N divisible by 128 and K divisible by 64");
    }

    BlockScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.groups_per_row   = shape[1] / 16;
    out.k_tiles          = shape[1] / 64;
    const auto elements  = checked_mul(out.rows, out.columns, "NVFP4 element count");
    out.code_plane_bytes = elements / 2;
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "NVFP4 scale plane offset");
    out.scale_plane_bytes = elements / 16;
    out.weight_divisor_offset =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "NVFP4 weight divisor offset");
    out.encoded_bytes = checked_add(out.weight_divisor_offset, 4, "NVFP4 tensor encoded size");
    return out;
}

namespace {

// One plane of a multi-plane layout, described in the units the slice arithmetic needs.
struct SlicePlane {
    std::uint64_t source_base = 0; // plane offset inside the parent payload
    std::uint64_t dest_base   = 0; // plane offset inside the shard payload
    std::uint64_t unit_bytes  = 0; // bytes per source unit (row, row tile, ...)
};

void require_slice(bool condition, std::string_view reason) {
    if (!condition) { throw ArtifactError("tensor slice: " + std::string(reason)); }
}

std::uint64_t validate_row_ranges(std::span<const SliceRange> rows, std::uint64_t tensor_rows,
                                  std::uint64_t alignment) {
    require_slice(!rows.empty(), "at least one row range is required");
    std::uint64_t total        = 0;
    std::uint64_t previous_end = 0;
    for (const SliceRange& range : rows) {
        require_slice(range.count != 0, "a row range must be nonempty");
        require_slice(range.begin >= previous_end, "row ranges must be ascending and disjoint");
        require_slice(range.begin <= tensor_rows && tensor_rows - range.begin >= range.count,
                      "a row range reaches past the last row");
        require_slice(range.begin % alignment == 0 && range.count % alignment == 0,
                      "a row range does not land on this layout's row-tile boundary");
        previous_end = range.begin + range.count;
        total        = checked_add(total, range.count, "row slice row count");
    }
    return total;
}

std::uint64_t row_element_stride(std::span<const std::uint64_t> shape) {
    std::uint64_t elements = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        require_slice(shape[i] != 0, "tensor shape dimensions must be positive");
        elements = checked_mul(elements, shape[i], "row element count");
    }
    return elements;
}

// Appends, for every selected row range, one contiguous copy per nonempty plane. Row-addressable
// planes only: `unit_bytes` is the plane's bytes-per-row (or bytes-per-row-tile, with `rows`
// pre-divided by the tile height).
void append_row_plane_copies(TensorSlice& out, std::span<const SlicePlane> planes,
                             std::span<const SliceRange> rows, std::uint64_t unit_rows) {
    for (const SlicePlane& plane : planes) {
        if (plane.unit_bytes == 0) { continue; }
        std::uint64_t destination = 0;
        for (const SliceRange& range : rows) {
            const std::uint64_t units = range.count / unit_rows;
            out.copies.push_back(PlaneCopy{
                .source_offset = checked_add(plane.source_base,
                                             checked_mul(range.begin / unit_rows, plane.unit_bytes,
                                                         "row slice source offset"),
                                             "row slice source offset"),
                .dest_offset   = checked_add(plane.dest_base,
                                             checked_mul(destination, plane.unit_bytes,
                                                         "row slice destination offset"),
                                             "row slice destination offset"),
                .bytes         = checked_mul(units, plane.unit_bytes, "row slice copy bytes"),
            });
            destination += units;
        }
    }
}

// Appends one copy per row for every plane whose rows are strided in the parent (a column slice
// keeps every row but narrows it, so no plane of any layout stays contiguous across rows).
//
// `select_units` is how many units of each row this call copies and `dest_skip_units` where they
// land inside the shard's own (possibly wider) row; a single-range slice passes
// `select_units == dest_units_per_row` and `dest_skip_units == 0`, which is what every quantized
// layout does. A multi-range contiguous-le-v1 slice calls this once per range with the same
// `dest_units_per_row` and an advancing `dest_skip_units`, so the ranges concatenate into one
// narrowed row in the order given.
void append_column_plane_copies(TensorSlice& out, std::span<const SlicePlane> planes,
                                std::uint64_t rows, std::uint64_t source_skip_units,
                                std::uint64_t source_units_per_row,
                                std::uint64_t dest_units_per_row, std::uint64_t select_units,
                                std::uint64_t dest_skip_units = 0) {
    for (const SlicePlane& plane : planes) {
        if (plane.unit_bytes == 0) { continue; }
        const std::uint64_t source_stride =
            checked_mul(source_units_per_row, plane.unit_bytes, "column slice source stride");
        const std::uint64_t dest_stride =
            checked_mul(dest_units_per_row, plane.unit_bytes, "column slice destination stride");
        const std::uint64_t skip =
            checked_mul(source_skip_units, plane.unit_bytes, "column slice source skip");
        const std::uint64_t dest_skip =
            checked_mul(dest_skip_units, plane.unit_bytes, "column slice destination skip");
        const std::uint64_t bytes =
            checked_mul(select_units, plane.unit_bytes, "column slice copy bytes");
        for (std::uint64_t row = 0; row < rows; ++row) {
            out.copies.push_back(PlaneCopy{
                .source_offset = checked_add(
                    plane.source_base,
                    checked_add(checked_mul(row, source_stride, "column slice source offset"), skip,
                                "column slice source offset"),
                    "column slice source offset"),
                .dest_offset   = checked_add(
                    plane.dest_base,
                    checked_add(checked_mul(row, dest_stride, "column slice destination offset"),
                                dest_skip, "column slice destination offset"),
                    "column slice destination offset"),
                .bytes         = bytes,
            });
        }
    }
}

// Same contract as validate_row_ranges, on the column axis. Kept separate so the diagnostics name
// the axis the caller actually asked for.
std::uint64_t validate_column_ranges(std::span<const SliceRange> columns,
                                     std::uint64_t tensor_columns) {
    require_slice(!columns.empty(), "at least one column range is required");
    std::uint64_t total        = 0;
    std::uint64_t previous_end = 0;
    for (const SliceRange& range : columns) {
        require_slice(range.count != 0, "a column range must be nonempty");
        require_slice(range.begin >= previous_end, "column ranges must be ascending and disjoint");
        require_slice(range.begin <= tensor_columns &&
                          tensor_columns - range.begin >= range.count,
                      "the column range reaches past the last column");
        previous_end = range.begin + range.count;
        total        = checked_add(total, range.count, "column slice column count");
    }
    return total;
}

} // namespace

TensorSlice tensor_row_slice(StorageLayout layout, NumericFormat format,
                             std::span<const std::uint64_t> shape,
                             std::span<const SliceRange> rows,
                             std::span<const std::byte> payload) {
    require_slice(!shape.empty(), "a row slice needs a tensor of rank one or higher");
    TensorSlice out;

    if (layout == StorageLayout::GgmlK256V1) {
        validate_ggml_k_payload(shape, payload);
        const auto total_rows = validate_row_ranges(rows, shape[0], 1);
        const auto source_base = ggml_k_code_offset(shape[0]);
        const auto dest_base = ggml_k_code_offset(total_rows);
        out.prefix.resize(dest_base);
        std::uint64_t dest_row = 0;
        std::uint64_t cursor = 0;
        for (const auto& range : rows) {
            for (auto row = range.begin; row < range.begin + range.count; ++row) {
                const auto descriptor = ggml_row(payload, row);
                const auto bytes = (shape[1] / 256) * ((descriptor & 1) ? 210 : 144);
                set_ggml_row(out.prefix, dest_row++, (cursor << 1) | (descriptor & 1));
                out.copies.push_back({source_base + (descriptor >> 1), dest_base + cursor, bytes});
                cursor += bytes;
            }
        }
        out.encoded_bytes = dest_base + cursor;
        return out;
    }

    if (layout == StorageLayout::ContiguousLeV1) {
        // Row-major with no internal structure: one contiguous range per row range.
        const std::uint64_t total_rows = validate_row_ranges(rows, shape[0], 1);
        const std::uint64_t row_bytes  = checked_mul(
            row_element_stride(shape), direct_word_bytes(format), "row slice row bytes");
        out.encoded_bytes = checked_mul(total_rows, row_bytes, "row slice encoded size");
        const std::array<SlicePlane, 1> planes = {SlicePlane{0, 0, row_bytes}};
        append_row_plane_copies(out, planes, rows, 1);
        return out;
    }

    if (layout == StorageLayout::RowSplitK128V1) {
        // Three row-addressable planes (storage-layouts.md 3.7). Any row boundary is legal; the
        // shard keeps the parent's K, hence its groups_per_row, so only the plane offsets move.
        require_slice(shape.size() == 2, "row-split-k128-v1 requires a rank-two shape");
        const std::uint64_t total_rows           = validate_row_ranges(rows, shape[0], 1);
        const RowSplitGeometry parent            = row_split_geometry(format, shape);
        const std::array<std::uint64_t, 2> shard_shape = {total_rows, shape[1]};
        const RowSplitGeometry shard             = row_split_geometry(format, shard_shape);
        out.encoded_bytes                        = shard.encoded_bytes;
        const std::array<SlicePlane, 3> planes   = {
            SlicePlane{0, 0, parent.groups_per_row * parent.low_bytes_per_group},
            SlicePlane{parent.high_plane_offset, shard.high_plane_offset,
                         parent.groups_per_row * parent.high_bytes_per_group},
            SlicePlane{parent.scale_plane_offset, shard.scale_plane_offset,
                         parent.groups_per_row * 2},
        };
        append_row_plane_copies(out, planes, rows, 1);
        return out;
    }

    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        // The code plane is row-major [N, K/2]. The swizzled scale plane's outermost axis is the
        // 128-row tile (offset (row_tile * K_tiles + scale_tile) * 512), so a 128-row-aligned row
        // range is one contiguous scale-plane range too. The FP32 weight divisor is matrix-level
        // and is therefore copied to every shard.
        require_slice(shape.size() == 2, "blockscale-k16-m128x4-v1 requires a rank-two shape");
        const std::uint64_t total_rows = validate_row_ranges(rows, shape[0], kNvfp4RowTile);
        const BlockScaleGeometry parent = block_scale_geometry(format, shape);
        const std::array<std::uint64_t, 2> shard_shape = {total_rows, shape[1]};
        const BlockScaleGeometry shard  = block_scale_geometry(format, shard_shape);
        out.encoded_bytes               = shard.encoded_bytes;
        const std::array<SlicePlane, 1> code = {SlicePlane{0, 0, shape[1] / 2}};
        append_row_plane_copies(out, code, rows, 1);
        const std::array<SlicePlane, 1> scales = {
            SlicePlane{parent.scale_plane_offset, shard.scale_plane_offset,
                       checked_mul(parent.k_tiles, kNvfp4TileBytes, "NVFP4 row tile bytes")}};
        append_row_plane_copies(out, scales, rows, kNvfp4RowTile);
        out.copies.push_back(PlaneCopy{parent.weight_divisor_offset, shard.weight_divisor_offset,
                                       kNvfp4DivisorBytes});
        return out;
    }

    if (layout == StorageLayout::RowScaleV1) {
        // Row-major code plane plus one BF16 scale word per row; both are row-addressable.
        require_slice(shape.size() == 2, "row-scale-v1 requires a rank-two shape");
        const std::uint64_t total_rows           = validate_row_ranges(rows, shape[0], 1);
        const RowScaleGeometry parent            = row_scale_geometry(format, shape);
        const std::array<std::uint64_t, 2> shard_shape = {total_rows, shape[1]};
        const RowScaleGeometry shard             = row_scale_geometry(format, shard_shape);
        out.encoded_bytes                        = shard.encoded_bytes;
        const std::array<SlicePlane, 2> planes   = {
            SlicePlane{0, 0, shape[1]},
            SlicePlane{parent.scale_plane_offset, shard.scale_plane_offset, 2},
        };
        append_row_plane_copies(out, planes, rows, 1);
        return out;
    }

    if (layout == StorageLayout::BatchedBlockScaleK16M128x4V1) {
        // The row axis of a batched object is its expert axis: a row slice selects whole
        // experts, and every selected expert contributes its code block, its 128-row scale
        // tiles and its FP32 divisor, exactly as the single-matrix rule does.
        require_slice(shape.size() == 3,
                      "batched_block_scale_k16_m128x4_v1 requires a rank-three shape");
        const std::uint64_t matrices = shape[0];
        const std::uint64_t selected = validate_row_ranges(rows, matrices, 1);
        const auto parent = batched_block_scale_geometry(format, shape);
        const std::array<std::uint64_t, 3> shard_shape = {selected, shape[1], shape[2]};
        const auto shard = batched_block_scale_geometry(format, shard_shape);
        out.encoded_bytes = shard.encoded_bytes;
        const auto matrix =
            block_scale_geometry(format, std::array<std::uint64_t, 2>{shape[1], shape[2]});
        const std::uint64_t row_tiles  = shape[1] / kNvfp4RowTile;
        const std::uint64_t tile_bytes =
            checked_mul(matrix.k_tiles, kNvfp4TileBytes, "NVFP4 scale tile bytes");
        std::uint64_t index = 0;
        for (const SliceRange& range : rows) {
            for (std::uint64_t expert = range.begin; expert < range.begin + range.count; ++expert) {
                const auto source_code =
                    checked_mul(expert, parent.code_plane_bytes, "row slice source offset");
                const auto dest_code =
                    checked_mul(index, shard.code_plane_bytes, "row slice destination offset");
                const auto source_scale = checked_add(
                    parent.scale_plane_offset,
                    checked_mul(expert, parent.scale_plane_bytes, "row slice source offset"),
                    "row slice source offset");
                const auto dest_scale = checked_add(
                    shard.scale_plane_offset,
                    checked_mul(index, shard.scale_plane_bytes, "row slice destination offset"),
                    "row slice destination offset");
                const auto source_divisor = checked_add(
                    parent.weight_divisor_offset,
                    checked_mul(expert, kNvfp4DivisorBytes, "row slice source offset"),
                    "row slice source offset");
                const auto dest_divisor = checked_add(
                    shard.weight_divisor_offset,
                    checked_mul(index, kNvfp4DivisorBytes, "row slice destination offset"),
                    "row slice destination offset");
                out.copies.push_back(PlaneCopy{
                    .source_offset = source_code,
                    .dest_offset   = dest_code,
                    .bytes         = parent.code_plane_bytes,
                });
                for (std::uint64_t tile = 0; tile < row_tiles; ++tile) {
                    out.copies.push_back(PlaneCopy{
                        .source_offset = checked_add(source_scale,
                                                     checked_mul(tile, tile_bytes,
                                                                 "row slice source offset"),
                                                     "row slice source offset"),
                        .dest_offset = checked_add(dest_scale,
                                                   checked_mul(tile, tile_bytes,
                                                               "row slice destination offset"),
                                                   "row slice destination offset"),
                        .bytes = tile_bytes,
                    });
                }
                out.copies.push_back(PlaneCopy{
                    .source_offset = source_divisor,
                    .dest_offset   = dest_divisor,
                    .bytes         = kNvfp4DivisorBytes,
                });
                ++index;
            }
        }
        return out;
    }

    if (layout == StorageLayout::BlockScaleM128K128V1) {
        const auto parent = block_scale_m128_k128_geometry(format, shape);
        if (shape.size() == 2) {
            // One matrix: the row axis is the matrix's row axis, a row range covers whole
            // 128-row bands, and each band contributes one code copy and one scale copy.
            const std::uint64_t selected = validate_row_ranges(rows, shape[0], kBlock128Tile);
            const std::array<std::uint64_t, 2> shard_shape = {selected, shape[1]};
            const auto shard = block_scale_m128_k128_geometry(format, shard_shape);
            out.encoded_bytes = shard.encoded_bytes;
            const std::uint64_t code_band_bytes =
                checked_mul(kBlock128Tile, shape[1], "FP8 band code bytes");
            const std::uint64_t scale_band_bytes =
                checked_mul(shape[1] / kBlock128Tile, kBlock128ScaleBytes, "FP8 band scale bytes");
            std::uint64_t destination_band = 0;
            for (const SliceRange& range : rows) {
                const auto first = range.begin / kBlock128Tile;
                const auto last  = (range.begin + range.count) / kBlock128Tile;
                for (std::uint64_t band = first; band < last; ++band) {
                    out.copies.push_back(PlaneCopy{
                        .source_offset = checked_mul(band, code_band_bytes,
                                                     "row slice source offset"),
                        .dest_offset = checked_mul(destination_band, code_band_bytes,
                                                   "row slice destination offset"),
                        .bytes = code_band_bytes,
                    });
                    out.copies.push_back(PlaneCopy{
                        .source_offset = checked_add(
                            parent.scale_plane_offset,
                            checked_mul(band, scale_band_bytes, "row slice source offset"),
                            "row slice source offset"),
                        .dest_offset = checked_add(
                            shard.scale_plane_offset,
                            checked_mul(destination_band, scale_band_bytes,
                                        "row slice destination offset"),
                            "row slice destination offset"),
                        .bytes = scale_band_bytes,
                    });
                    ++destination_band;
                }
            }
            return out;
        }
        // A stack of matrices: the row axis is its matrix axis, so a row slice selects whole
        // matrices, and each selected matrix contributes its code block and its scale block.
        require_slice(shape.size() == 3,
                      "block_scale_m128_k128_v1 row slicing requires a rank-two or rank-three "
                      "shape");
        const std::uint64_t matrices = shape[0];
        const std::uint64_t selected = validate_row_ranges(rows, matrices, 1);
        const std::array<std::uint64_t, 3> shard_shape = {selected, shape[1], shape[2]};
        const auto shard = block_scale_m128_k128_geometry(format, shard_shape);
        out.encoded_bytes = shard.encoded_bytes;
        std::uint64_t index = 0;
        for (const SliceRange& range : rows) {
            for (std::uint64_t matrix_index = range.begin; matrix_index < range.begin + range.count;
                 ++matrix_index) {
                out.copies.push_back(PlaneCopy{
                    .source_offset = checked_mul(matrix_index, parent.code_plane_bytes,
                                                 "row slice source offset"),
                    .dest_offset = checked_mul(index, shard.code_plane_bytes,
                                               "row slice destination offset"),
                    .bytes = parent.code_plane_bytes,
                });
                out.copies.push_back(PlaneCopy{
                    .source_offset = checked_add(
                        parent.scale_plane_offset,
                        checked_mul(matrix_index, parent.scale_plane_bytes,
                                    "row slice source offset"),
                        "row slice source offset"),
                    .dest_offset = checked_add(
                        shard.scale_plane_offset,
                        checked_mul(index, shard.scale_plane_bytes,
                                    "row slice destination offset"),
                        "row slice destination offset"),
                    .bytes = parent.scale_plane_bytes,
                });
                ++index;
            }
        }
        return out;
    }

    if (layout == StorageLayout::RowPagedG16x4KV1) {
        throw ArtifactError(
            "tensor slice: row_paged_g16_4k_v1 has no slice form; the paged object is "
            "replicated whole");
    }
    throw ArtifactError("unknown tensor layout");
}

TensorSlice tensor_column_slice(StorageLayout layout, NumericFormat format,
                                std::span<const std::uint64_t> shape,
                                std::span<const SliceRange> column_ranges,
                                std::span<const std::byte> payload) {
    if (layout == StorageLayout::BatchedBlockScaleK16M128x4V1 ||
        layout == StorageLayout::BlockScaleM128K128V1) {
        require_slice(shape.size() == 2 || shape.size() == 3,
                      "a block-scaled column slice requires a rank-two or rank-three shape");
    } else {
        require_slice(shape.size() == 2, "a column slice requires a rank-two shape");
    }
    TensorSlice out;

    if (layout == StorageLayout::BatchedBlockScaleK16M128x4V1) {
        // The column axis of a batched object is the K axis of every expert: one column range
        // narrows each matrix with the single-matrix blockscale rule.
        require_slice(shape.size() == 3,
                      "batched_block_scale_k16_m128x4_v1 requires a rank-three shape");
        const std::uint64_t matrices       = shape[0];
        const std::uint64_t matrix_rows    = shape[1];
        const std::uint64_t matrix_columns = shape[2];
        validate_column_ranges(column_ranges, matrix_columns);
        require_slice(column_ranges.size() == 1,
                      "this storage layout requires one contiguous column range");
        const SliceRange columns = column_ranges.front();
        require_slice(columns.begin % kNvfp4TileColumns == 0 &&
                          columns.count % kNvfp4TileColumns == 0,
                      "batched_block_scale_k16_m128x4_v1 column ranges must be multiples of 64");
        const auto parent = batched_block_scale_geometry(format, shape);
        const std::array<std::uint64_t, 3> shard_shape = {matrices, matrix_rows, columns.count};
        const auto shard = batched_block_scale_geometry(format, shard_shape);
        out.encoded_bytes = shard.encoded_bytes;
        const auto parent_matrix = block_scale_geometry(
            format, std::array<std::uint64_t, 2>{matrix_rows, matrix_columns});
        const auto shard_matrix = block_scale_geometry(
            format, std::array<std::uint64_t, 2>{matrix_rows, columns.count});
        const std::uint64_t row_tiles = matrix_rows / kNvfp4RowTile;
        const std::uint64_t parent_tile_bytes =
            checked_mul(parent_matrix.k_tiles, kNvfp4TileBytes, "NVFP4 scale tile bytes");
        const std::uint64_t shard_tile_bytes =
            checked_mul(shard_matrix.k_tiles, kNvfp4TileBytes, "NVFP4 scale tile bytes");
        const std::uint64_t source_row_bytes = matrix_columns / 2;
        const std::uint64_t shard_row_bytes  = columns.count / 2;
        for (std::uint64_t expert = 0; expert < matrices; ++expert) {
            const auto source_code =
                checked_mul(expert, parent.code_plane_bytes, "column slice source offset");
            const auto dest_code =
                checked_mul(expert, shard.code_plane_bytes, "column slice destination offset");
            const auto source_scale = checked_add(
                parent.scale_plane_offset,
                checked_mul(expert, parent.scale_plane_bytes, "column slice source offset"),
                "column slice source offset");
            const auto dest_scale = checked_add(
                shard.scale_plane_offset,
                checked_mul(expert, shard.scale_plane_bytes, "column slice destination offset"),
                "column slice destination offset");
            for (std::uint64_t row = 0; row < matrix_rows; ++row) {
                out.copies.push_back(PlaneCopy{
                    .source_offset = checked_add(
                        source_code,
                        checked_add(checked_mul(row, source_row_bytes,
                                                "column slice source offset"),
                                    columns.begin / 2, "column slice source offset"),
                        "column slice source offset"),
                    .dest_offset = checked_add(dest_code,
                                               checked_mul(row, shard_row_bytes,
                                                           "column slice destination offset"),
                                               "column slice destination offset"),
                    .bytes = shard_row_bytes,
                });
            }
            for (std::uint64_t tile = 0; tile < row_tiles; ++tile) {
                out.copies.push_back(PlaneCopy{
                    .source_offset = checked_add(
                        source_scale,
                        checked_add(checked_mul(tile, parent_tile_bytes,
                                                "column slice source offset"),
                                    (columns.begin / kNvfp4TileColumns) * kNvfp4TileBytes,
                                    "column slice source offset"),
                        "column slice source offset"),
                    .dest_offset = checked_add(dest_scale,
                                               checked_mul(tile, shard_tile_bytes,
                                                           "column slice destination offset"),
                                               "column slice destination offset"),
                    .bytes = shard_tile_bytes,
                });
            }
            out.copies.push_back(PlaneCopy{
                .source_offset = checked_add(
                    parent.weight_divisor_offset,
                    checked_mul(expert, kNvfp4DivisorBytes, "column slice source offset"),
                    "column slice source offset"),
                .dest_offset = checked_add(
                    shard.weight_divisor_offset,
                    checked_mul(expert, kNvfp4DivisorBytes, "column slice destination offset"),
                    "column slice destination offset"),
                .bytes = kNvfp4DivisorBytes,
            });
        }
        return out;
    }

    if (layout == StorageLayout::RowPagedG16x4KV1) {
        throw ArtifactError(
            "tensor slice: row_paged_g16_4k_v1 has no slice form; the paged object is "
            "replicated whole");
    }

    if (layout == StorageLayout::BlockScaleM128K128V1) {
        // The column axis is the K axis: one column range narrows every matrix of the stack,
        // and the scale plane is addressed by 128-row bands of block-column words.
        const std::uint64_t matrices = shape.size() == 3 ? shape[0] : 1;
        const std::uint64_t rows     = shape[shape.size() - 2];
        const std::uint64_t columns  = shape.back();
        validate_column_ranges(column_ranges, columns);
        require_slice(column_ranges.size() == 1,
                      "this storage layout requires one contiguous column range");
        const SliceRange selected = column_ranges.front();
        require_slice(selected.begin % kBlock128Tile == 0 && selected.count % kBlock128Tile == 0,
                      "block_scale_m128_k128_v1 column ranges must be multiples of 128");
        const auto parent = block_scale_m128_k128_geometry(format, shape);
        Block128Geometry shard;
        if (shape.size() == 2) {
            shard = block_scale_m128_k128_geometry(
                format, std::array<std::uint64_t, 2>{rows, selected.count});
        } else {
            shard = block_scale_m128_k128_geometry(
                format, std::array<std::uint64_t, 3>{matrices, rows, selected.count});
        }
        out.encoded_bytes = shard.encoded_bytes;
        const std::uint64_t bands = rows / kBlock128Tile;
        const std::uint64_t parent_band_bytes =
            checked_mul(columns / kBlock128Tile, kBlock128ScaleBytes, "FP8 band scale bytes");
        const std::uint64_t shard_band_bytes = checked_mul(selected.count / kBlock128Tile,
                                                           kBlock128ScaleBytes,
                                                           "FP8 band scale bytes");
        for (std::uint64_t matrix_index = 0; matrix_index < matrices; ++matrix_index) {
            const auto source_code = checked_mul(matrix_index, parent.code_plane_bytes,
                                                 "column slice source offset");
            const auto dest_code = checked_mul(matrix_index, shard.code_plane_bytes,
                                               "column slice destination offset");
            const auto source_scale = checked_add(
                parent.scale_plane_offset,
                checked_mul(matrix_index, parent.scale_plane_bytes, "column slice source offset"),
                "column slice source offset");
            const auto dest_scale = checked_add(
                shard.scale_plane_offset,
                checked_mul(matrix_index, shard.scale_plane_bytes,
                            "column slice destination offset"),
                "column slice destination offset");
            for (std::uint64_t row = 0; row < rows; ++row) {
                out.copies.push_back(PlaneCopy{
                    .source_offset = checked_add(
                        source_code,
                        checked_add(checked_mul(row, columns, "column slice source offset"),
                                    selected.begin, "column slice source offset"),
                        "column slice source offset"),
                    .dest_offset = checked_add(dest_code,
                                               checked_mul(row, selected.count,
                                                           "column slice destination offset"),
                                               "column slice destination offset"),
                    .bytes = selected.count,
                });
            }
            for (std::uint64_t band = 0; band < bands; ++band) {
                out.copies.push_back(PlaneCopy{
                    .source_offset = checked_add(
                        source_scale,
                        checked_add(checked_mul(band, parent_band_bytes,
                                                "column slice source offset"),
                                    (selected.begin / kBlock128Tile) * kBlock128ScaleBytes,
                                    "column slice source offset"),
                        "column slice source offset"),
                    .dest_offset = checked_add(dest_scale,
                                               checked_mul(band, shard_band_bytes,
                                                           "column slice destination offset"),
                                               "column slice destination offset"),
                    .bytes = shard_band_bytes,
                });
            }
        }
        return out;
    }

    const std::uint64_t rows        = shape[0];
    // Row-split-k128 shards may extend into the parent's padded K (align_up(K, 128)): when the
    // logical K is not itself 128-aligned, the final rank's column range legitimately covers the
    // trailing pad groups as well. No other layout physically pads the column axis.
    const std::uint64_t span_limit =
        layout == StorageLayout::RowSplitK128V1 ? align_up(shape[1], kKAlignment, "padded K") : shape[1];
    const std::uint64_t total_count = validate_column_ranges(column_ranges, span_limit);

    if (layout == StorageLayout::GgmlK256V1) {
        validate_ggml_k_payload(shape, payload);
        for (const auto& columns : column_ranges) {
            require_slice(columns.begin % 256 == 0 && columns.count % 256 == 0,
                          "ggml-k256-v1 column boundaries must be multiples of 256");
        }
        const auto code_base = ggml_k_code_offset(rows);
        out.prefix.resize(code_base);
        std::uint64_t cursor = 0;
        for (std::uint64_t row = 0; row < rows; ++row) {
            const auto descriptor = ggml_row(payload, row);
            const auto block_bytes = (descriptor & 1) ? 210 : 144;
            set_ggml_row(out.prefix, row, (cursor << 1) | (descriptor & 1));
            for (const auto& columns : column_ranges) {
                const auto bytes = (columns.count / 256) * block_bytes;
                out.copies.push_back({code_base + (descriptor >> 1) + (columns.begin / 256) * block_bytes,
                                       code_base + cursor, bytes});
                cursor += bytes;
            }
        }
        out.encoded_bytes = code_base + cursor;
        return out;
    }

    if (layout == StorageLayout::ContiguousLeV1) {
        // Row-major with no internal structure: every range of every row is one contiguous copy,
        // so any number of ranges concatenates cleanly. This is the only layout that admits more
        // than one range; see the guard below.
        const std::uint64_t word                       = direct_word_bytes(format);
        const std::array<std::uint64_t, 2> shard_shape = {rows, total_count};
        out.encoded_bytes = tensor_encoded_size(layout, format, shard_shape);
        const std::array<SlicePlane, 1> planes = {SlicePlane{0, 0, word}};
        std::uint64_t destination              = 0;
        for (const SliceRange& range : column_ranges) {
            append_column_plane_copies(out, planes, rows, range.begin, shape[1], total_count,
                                       range.count, destination);
            destination += range.count;
        }
        return out;
    }

    // Every remaining layout groups, tiles, or swizzles along the column axis, so a shard made of
    // several disjoint column ranges would not reconstruct into that layout's own geometry. Those
    // families are all row-parallel GEMM weights, whose shard genuinely is one contiguous input
    // range, so the restriction costs nothing and is rejected loudly rather than mis-encoded.
    require_slice(column_ranges.size() == 1,
                  "this storage layout requires one contiguous column range");
    const SliceRange columns = column_ranges.front();

    if (layout == StorageLayout::RowSplitK128V1) {
        // Columns are grouped; a column range must therefore start and end on a group boundary.
        // The unit is the format's own group size (not the legacy 128): a 128-multiple count is a
        // special case where the shard's own K_pad equals its K, but a non-128-aligned total K
        // forces some shards to carry the parent's trailing pad groups (the caller's ranges tile
        // the padded span [0, K_pad) exactly). Pad groups decode per the layout's padding rule, so
        // their contribution to a GEMM over the shard's logical K_r is zero whenever the
        // corresponding activation lanes are zero -- the row-parallel consumers pair this with a
        // zeroed tail segment of their up-projection.
        const RowSplitGeometry parent = row_split_geometry(format, shape);
        require_slice(columns.begin % parent.group_size == 0 &&
                          columns.count % parent.group_size == 0,
                      "row-split-k128-v1 column ranges must be multiples of the format group size");
        require_slice(columns.begin + columns.count <= parent.padded_columns,
                      "row-split-k128-v1 column range exceeds the padded span");
        const std::array<std::uint64_t, 2> shard_shape = {rows, columns.count};
        const RowSplitGeometry shard = row_split_geometry(format, shard_shape);
        out.encoded_bytes            = shard.encoded_bytes;
        const std::uint64_t skip     = columns.begin / parent.group_size;
        // A group-aligned range whose count is not itself 128-aligned (e.g. 576 of a 1152-wide K)
        // still needs align_up(count, 128) columns of row storage, so the shard owns more groups
        // than it copies: the trailing pad groups have no parent group to point at. They are
        // row-stride padding the consumer never reads, and the destination is zero-filled before
        // the copies land, so they are simply left uncopied. Real groups always fit -- begin +
        // count is bounded by the parent's padded span, so every logical group has a source.
        const std::uint64_t available = parent.groups_per_row - skip;
        const std::uint64_t copied    = std::min(shard.groups_per_row, available);
        const std::array<SlicePlane, 3> planes = {
            SlicePlane{0, 0, parent.low_bytes_per_group},
            SlicePlane{parent.high_plane_offset, shard.high_plane_offset,
                       parent.high_bytes_per_group},
            SlicePlane{parent.scale_plane_offset, shard.scale_plane_offset, 2},
        };
        append_column_plane_copies(out, planes, rows, skip, parent.groups_per_row,
                                   shard.groups_per_row, copied);
        return out;
    }

    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        // A column range of whole 64-column scale tiles is contiguous *within* one 128-row tile
        // but strided across row tiles, and the code plane is strided per row. Both are copied as
        // regular strides; the matrix-level divisor is replicated.
        const BlockScaleGeometry parent = block_scale_geometry(format, shape);
        require_slice(columns.begin % kNvfp4TileColumns == 0 &&
                          columns.count % kNvfp4TileColumns == 0,
                      "blockscale-k16-m128x4-v1 column ranges must be multiples of 64");
        const std::array<std::uint64_t, 2> shard_shape = {rows, columns.count};
        const BlockScaleGeometry shard = block_scale_geometry(format, shard_shape);
        out.encoded_bytes              = shard.encoded_bytes;
        const std::array<SlicePlane, 1> code = {SlicePlane{0, 0, 1}};
        append_column_plane_copies(out, code, rows, columns.begin / 2, shape[1] / 2,
                                   columns.count / 2, columns.count / 2);
        const std::array<SlicePlane, 1> scales = {SlicePlane{
            parent.scale_plane_offset, shard.scale_plane_offset, kNvfp4TileBytes}};
        append_column_plane_copies(out, scales, rows / kNvfp4RowTile,
                                   columns.begin / kNvfp4TileColumns, parent.k_tiles,
                                   shard.k_tiles, shard.k_tiles);
        out.copies.push_back(PlaneCopy{parent.weight_divisor_offset, shard.weight_divisor_offset,
                                       kNvfp4DivisorBytes});
        return out;
    }

    if (layout == StorageLayout::RowScaleV1) {
        // The BF16 multiplier is per output row, and a column slice keeps every row, so the whole
        // scale plane is replicated: each shard scales its partial product by the same factor,
        // which is exactly what summing the partials across devices requires.
        const RowScaleGeometry parent                  = row_scale_geometry(format, shape);
        const std::array<std::uint64_t, 2> shard_shape = {rows, columns.count};
        const RowScaleGeometry shard = row_scale_geometry(format, shard_shape);
        out.encoded_bytes            = shard.encoded_bytes;
        const std::array<SlicePlane, 1> code = {SlicePlane{0, 0, 1}};
        append_column_plane_copies(out, code, rows, columns.begin, shape[1], columns.count,
                                   columns.count);
        out.copies.push_back(PlaneCopy{parent.scale_plane_offset, shard.scale_plane_offset,
                                       shard.scale_plane_bytes});
        return out;
    }
    throw ArtifactError("unknown tensor layout");
}

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
        throw ArtifactError("row-scale-v1 requires FP8_E4M3FN_ROW_BF16S");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-scale-v1 requires a positive rank-two shape");
    }

    RowScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.code_plane_bytes = checked_mul(out.rows, out.columns, "FP8 element count");
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "FP8 scale plane offset");
    out.scale_plane_bytes = checked_mul(out.rows, 2, "FP8 scale plane bytes");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "FP8 tensor encoded size");
    return out;
}

RowPagedNf4Geometry row_paged_nf4_geometry(NumericFormat format,
                                           std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::NF4_G16_E4M3S) {
        throw ArtifactError("row_paged_g16_4k_v1 requires NF4_G16_E4M3S");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row_paged_g16_4k_v1 requires a positive rank-two shape");
    }
    if (shape[1] % kNf4GroupSize != 0) {
        throw ArtifactError("row_paged_g16_4k_v1 requires K divisible by 16");
    }

    RowPagedNf4Geometry out;
    out.rows                = shape[0];
    out.columns             = shape[1];
    out.group_size          = kNf4GroupSize;
    out.page_bytes          = kNf4PageBytes;
    out.code_bytes_per_row  = shape[1] / 2;
    out.scale_bytes_per_row = shape[1] / kNf4GroupSize;
    out.row_bytes           = checked_add(out.code_bytes_per_row, out.scale_bytes_per_row,
                                          "NF4 row bytes");
    out.rows_per_page       = kNf4PageBytes / out.row_bytes;
    if (out.rows_per_page == 0) {
        throw ArtifactError("row_paged_g16_4k_v1 requires a row that fits in one 4096-byte page");
    }
    const auto rows_ceiling = checked_add(shape[0], out.rows_per_page - 1, "NF4 row count");
    out.page_count          = rows_ceiling / out.rows_per_page;
    out.code_bytes          = checked_mul(shape[0], out.code_bytes_per_row, "NF4 code bytes");
    out.scale_bytes         = checked_mul(shape[0], out.scale_bytes_per_row, "NF4 scale bytes");
    out.matrix_scale_offset = checked_mul(out.page_count, kNf4PageBytes, "NF4 matrix scale offset");
    out.encoded_bytes =
        checked_add(out.matrix_scale_offset, kNf4MatrixScaleBytes, "NF4 tensor encoded size");
    return out;
}

std::uint64_t row_paged_nf4_row_offset(const RowPagedNf4Geometry& geometry, std::uint64_t row) {
    if (geometry.rows_per_page == 0 || row >= geometry.rows) {
        throw ArtifactError("row_paged_g16_4k_v1 row is outside its parent");
    }
    const auto page = row / geometry.rows_per_page;
    const auto index = row % geometry.rows_per_page;
    return checked_add(checked_mul(page, geometry.page_bytes, "NF4 page offset"),
                       checked_mul(index, geometry.row_bytes, "NF4 row offset"),
                       "NF4 row offset");
}

std::uint64_t row_paged_nf4_scale_offset(const RowPagedNf4Geometry& geometry, std::uint64_t row,
                                         std::uint64_t group) {
    if (group >= geometry.columns / geometry.group_size) {
        throw ArtifactError("row_paged_g16_4k_v1 group is outside its row");
    }
    return checked_add(row_paged_nf4_row_offset(geometry, row),
                       checked_add(geometry.code_bytes_per_row, group, "NF4 group scale offset"),
                       "NF4 group scale offset");
}

Block128Geometry block_scale_m128_k128_geometry(NumericFormat format,
                                               std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::FP8_E4M3FN_BLOCK128_BF16S) {
        throw ArtifactError("block_scale_m128_k128_v1 requires FP8_E4M3FN_BLOCK128_BF16S");
    }

    Block128Geometry out;
    if (shape.size() == 2) {
        out.rows    = shape[0];
        out.columns = shape[1];
    } else if (shape.size() == 3) {
        out.matrices = shape[0];
        out.rows     = shape[1];
        out.columns  = shape[2];
    } else {
        throw ArtifactError("block_scale_m128_k128_v1 requires a rank-two or rank-three shape");
    }
    if (out.matrices == 0 || out.rows == 0 || out.columns == 0) {
        throw ArtifactError("block_scale_m128_k128_v1 requires positive dimensions");
    }
    if (out.rows % kBlock128Tile != 0 || out.columns % kBlock128Tile != 0) {
        throw ArtifactError("block_scale_m128_k128_v1 requires N and K divisible by 128");
    }

    out.code_plane_bytes = checked_mul(out.rows, out.columns, "FP8 block code plane bytes");
    const auto blocks =
        checked_mul(out.rows / kBlock128Tile, out.columns / kBlock128Tile, "FP8 block count");
    out.scale_plane_bytes = checked_mul(blocks, kBlock128ScaleBytes, "FP8 block scale bytes");
    const auto codes = checked_mul(out.matrices, out.code_plane_bytes, "FP8 block code bytes");
    out.scale_plane_offset = align_up(codes, kTensorAlignment, "FP8 block scale plane offset");
    out.encoded_bytes      = checked_add(
        out.scale_plane_offset,
        checked_mul(out.matrices, out.scale_plane_bytes, "FP8 block scale plane bytes"),
        "FP8 block tensor encoded size");
    return out;
}

BatchedBlockScaleGeometry batched_block_scale_geometry(NumericFormat format,
                                                       std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::NVFP4) {
        throw ArtifactError("batched_block_scale_k16_m128x4_v1 requires NVFP4");
    }
    if (shape.size() != 3) {
        throw ArtifactError("batched_block_scale_k16_m128x4_v1 requires a rank-three shape");
    }
    const std::uint64_t matrices = shape[0];
    const std::uint64_t rows     = shape[1];
    const std::uint64_t columns  = shape[2];
    if (matrices == 0 || rows == 0 || columns == 0) {
        throw ArtifactError("batched_block_scale_k16_m128x4_v1 requires positive dimensions");
    }
    if (rows % kNvfp4RowTile != 0 || columns % kNvfp4TileColumns != 0) {
        throw ArtifactError(
            "batched_block_scale_k16_m128x4_v1 requires N divisible by 128 and K divisible by 64");
    }

    BatchedBlockScaleGeometry out;
    out.matrices          = matrices;
    out.rows              = rows;
    out.columns           = columns;
    out.code_plane_bytes  = checked_mul(rows, columns / 2, "batched NVFP4 code plane bytes");
    out.scale_plane_bytes = checked_mul(rows, columns / 16, "batched NVFP4 scale plane bytes");
    const auto codes = checked_mul(matrices, out.code_plane_bytes, "batched NVFP4 code bytes");
    out.scale_plane_offset = align_up(codes, kTensorAlignment, "batched NVFP4 scale plane offset");
    const auto scales =
        checked_mul(matrices, out.scale_plane_bytes, "batched NVFP4 scale plane bytes");
    out.weight_divisor_offset =
        checked_add(out.scale_plane_offset, scales, "batched NVFP4 weight divisor offset");
    out.encoded_bytes = checked_add(
        out.weight_divisor_offset,
        checked_mul(matrices, kNvfp4DivisorBytes, "batched NVFP4 weight divisor bytes"),
        "batched NVFP4 tensor encoded size");
    return out;
}

namespace {

// The logical axes an element interval is expressed in: a stack of matrices, and inside one matrix
// the rows and the columns.
struct ViewAxes {
    std::uint64_t matrices = 1;
    std::uint64_t rows     = 1;
    std::uint64_t columns  = 1;
};

// The axes of a contiguous tensor: its shape is row-major, so the leading axis is the matrix
// stack, the trailing axis is the column axis, and what lies between them is the row axis.
ViewAxes contiguous_view_axes(std::span<const std::uint64_t> shape) {
    for (const auto dimension : shape) {
        if (dimension == 0) { throw ArtifactError("tensor shape dimensions must be positive"); }
    }
    ViewAxes axes;
    if (shape.size() >= 3) {
        axes.matrices = shape[0];
        for (std::size_t index = 1; index + 1 < shape.size(); ++index) {
            axes.rows = checked_mul(axes.rows, shape[index], "tensor row count");
        }
        axes.columns = shape.back();
    } else if (shape.size() == 2) {
        axes.rows    = shape[0];
        axes.columns = shape[1];
    } else if (shape.size() == 1) {
        axes.rows = shape[0];
    }
    return axes;
}

// The axes of a layout, with its format and rank validated the way the layout's own geometry
// validates them.
ViewAxes element_view_axes(StorageLayout layout, NumericFormat format,
                           std::span<const std::uint64_t> shape) {
    switch (layout) {
    case StorageLayout::ContiguousLeV1:
        (void)direct_word_bytes(format);
        return contiguous_view_axes(shape);
    case StorageLayout::RowSplitK128V1: {
        const auto geometry = row_split_geometry(format, shape);
        return ViewAxes{1, geometry.rows, geometry.columns};
    }
    case StorageLayout::RowScaleV1: {
        const auto geometry = row_scale_geometry(format, shape);
        return ViewAxes{1, geometry.rows, geometry.columns};
    }
    case StorageLayout::GgmlK256V1:
        if (format != NumericFormat::GGML_K || shape.size() != 2 || shape[0] == 0 ||
            shape[1] == 0 || shape[1] % 256 != 0) {
            throw ArtifactError("ggml-k256-v1 requires GGML_K [N,K] with K divisible by 256");
        }
        return ViewAxes{1, shape[0], shape[1]};
    case StorageLayout::BlockScaleK16M128x4V1: {
        const auto geometry = block_scale_geometry(format, shape);
        return ViewAxes{1, geometry.rows, geometry.columns};
    }
    case StorageLayout::RowPagedG16x4KV1: {
        const auto geometry = row_paged_nf4_geometry(format, shape);
        return ViewAxes{1, geometry.rows, geometry.columns};
    }
    case StorageLayout::BlockScaleM128K128V1: {
        const auto geometry = block_scale_m128_k128_geometry(format, shape);
        return ViewAxes{geometry.matrices, geometry.rows, geometry.columns};
    }
    case StorageLayout::BatchedBlockScaleK16M128x4V1: {
        const auto geometry = batched_block_scale_geometry(format, shape);
        return ViewAxes{geometry.matrices, geometry.rows, geometry.columns};
    }
    }
    throw ArtifactError("unknown tensor layout");
}

} // namespace

ElementIntervalView tensor_element_interval_view(StorageLayout layout, NumericFormat format,
                                                 std::span<const std::uint64_t> shape,
                                                 std::uint64_t begin, std::uint64_t count) {
    if (count == 0) {
        throw ArtifactError("element interval view: an empty interval has no rectangle");
    }
    const ViewAxes axes   = element_view_axes(layout, format, shape);
    const auto per_matrix = checked_mul(axes.rows, axes.columns, "matrix element count");
    const auto total      = checked_mul(axes.matrices, per_matrix, "tensor element count");
    const auto end        = checked_add(begin, count, "element interval end");
    if (end > total) {
        throw ArtifactError("element interval view: the interval reaches past the last element");
    }

    const auto matrix       = begin / per_matrix;
    const auto offset       = begin % per_matrix;
    const auto row          = offset / axes.columns;
    const auto column       = offset % axes.columns;
    const auto matrix_first = checked_mul(matrix, per_matrix, "element interval offset");
    const auto row_first    = checked_add(
        matrix_first, checked_mul(row, axes.columns, "element interval offset"),
        "element interval offset");

    ElementIntervalView view;
    view.layout        = layout;
    view.linear_begin  = begin;
    view.element_count = count;

    if (column != 0) {
        // The interval starts inside a row, so it is a run of columns of that row, and it must not
        // reach into the next one.
        if (end > checked_add(row_first, axes.columns, "element interval row end")) {
            throw ArtifactError("element interval view: the interval is not a rectangle");
        }
        view.matrix_begin = matrix;
        view.matrix_count = 1;
        view.row_begin    = row;
        view.row_count    = 1;
        view.column_begin = column;
        view.column_count = count;
        return view;
    }

    if (count % axes.columns != 0) {
        throw ArtifactError("element interval view: the interval is not a rectangle");
    }
    const auto rows_covered = count / axes.columns;
    if (rows_covered <= axes.rows - row) {
        view.matrix_begin = matrix;
        view.matrix_count = 1;
        view.row_begin    = row;
        view.row_count    = rows_covered;
        view.column_begin = 0;
        view.column_count = axes.columns;
        return view;
    }
    // The interval reaches past the matrix it starts in, so it must cover whole matrices.
    if (row != 0 || rows_covered % axes.rows != 0) {
        throw ArtifactError("element interval view: the interval is not a rectangle");
    }
    view.matrix_begin = matrix;
    view.matrix_count = rows_covered / axes.rows;
    view.row_begin    = 0;
    view.row_count    = axes.rows;
    view.column_begin = 0;
    view.column_count = axes.columns;
    return view;
}

} // namespace ninfer::artifact
