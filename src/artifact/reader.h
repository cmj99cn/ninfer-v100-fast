#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::artifact {

class ArtifactError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class NumericFormat {
    BF16,
    FP16,
    FP32,
    I32,
    Q4G64_F16S,
    Q5G64_F16S,
    Q6G64_F16S,
    W8G32_F16S,
    NVFP4,
    FP8_E4M3FN_ROW_BF16S,
    GGML_K,
    NF4_G16_E4M3S,
    FP8_E4M3FN_BLOCK128_BF16S,
};

enum class StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
    GgmlK256V1,
    RowPagedG16x4KV1,
    BlockScaleM128K128V1,
    BatchedBlockScaleK16M128x4V1,
};

enum class ResourceEncoding {
    RawBytesV1,
};

std::string_view format_name(NumericFormat format) noexcept;
std::string_view layout_name(StorageLayout layout) noexcept;
std::string_view encoding_name(ResourceEncoding encoding) noexcept;

// The v3 container spells its numeric formats, storage layouts and resource encodings in lower
// case snake case; the v2 container spells the same concepts differently. The two spellings are
// separate contracts that denote the same enum values, so the v3 table is the authority for
// reading an upstream v3 artifact and never widens the v2 one. It accepts every spelling an
// upstream v3 artifact uses.
NumericFormat parse_v3_format(std::string_view name);
StorageLayout parse_v3_layout(std::string_view name);
ResourceEncoding parse_v3_encoding(std::string_view name);

std::uint64_t tensor_alignment(StorageLayout layout) noexcept;
std::uint64_t resource_alignment(ResourceEncoding encoding) noexcept;
std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape,
                                  std::uint64_t stored_bytes = 0);
std::uint64_t ggml_k_code_offset(std::uint64_t rows);
void validate_ggml_k_payload(std::span<const std::uint64_t> shape,
                             std::span<const std::byte> payload);

struct RowSplitGeometry {
    std::uint64_t rows                 = 0;
    std::uint64_t columns              = 0;
    std::uint64_t padded_columns       = 0;
    std::uint64_t group_size           = 0;
    std::uint64_t groups_per_row       = 0;
    std::uint64_t low_bytes_per_group  = 0;
    std::uint64_t high_bytes_per_group = 0;
    std::uint64_t low_plane_bytes      = 0;
    std::uint64_t high_plane_offset    = 0;
    std::uint64_t high_plane_bytes     = 0;
    std::uint64_t scale_plane_offset   = 0;
    std::uint64_t scale_plane_bytes    = 0;
    std::uint64_t encoded_bytes        = 0;
};

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct BlockScaleGeometry {
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowScaleGeometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t encoded_bytes      = 0;
};

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowPagedNf4Geometry {
    std::uint64_t rows                = 0;
    std::uint64_t columns             = 0;
    std::uint64_t group_size          = 16;
    std::uint64_t code_bytes_per_row  = 0;
    std::uint64_t scale_bytes_per_row = 0;
    std::uint64_t row_bytes           = 0;
    std::uint64_t rows_per_page       = 0;
    std::uint64_t page_bytes          = 0;
    std::uint64_t page_count          = 0;
    std::uint64_t code_bytes          = 0;
    std::uint64_t scale_bytes         = 0;
    std::uint64_t matrix_scale_offset = 0;
    std::uint64_t encoded_bytes       = 0;
};

RowPagedNf4Geometry row_paged_nf4_geometry(NumericFormat format,
                                           std::span<const std::uint64_t> shape);

// Byte offset of one encoded NF4 row inside its paged parent.
std::uint64_t row_paged_nf4_row_offset(const RowPagedNf4Geometry& geometry, std::uint64_t row);

// Byte offset of one group scale word inside its paged parent.
std::uint64_t row_paged_nf4_scale_offset(const RowPagedNf4Geometry& geometry, std::uint64_t row,
                                         std::uint64_t group);

struct Block128Geometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t matrices           = 1;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t encoded_bytes      = 0;
};

Block128Geometry block_scale_m128_k128_geometry(NumericFormat format,
                                               std::span<const std::uint64_t> shape);

struct BatchedBlockScaleGeometry {
    std::uint64_t matrices              = 0;
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BatchedBlockScaleGeometry batched_block_scale_geometry(NumericFormat format,
                                                       std::span<const std::uint64_t> shape);

// --- Element intervals (v3 binding parts) -----------------------------------------------------
//
// A v3 binding part names the elements a parameter occupies inside one object, counted in the
// object's own logical element order. `tensor_element_interval_view` turns such an interval into
// the logical rectangle it covers: which matrices, which rows of each matrix, and which columns
// of each row.
//
// The logical axes of a view follow the object's rank: a rank-three object is a stack of matrices,
// a rank-two object is one matrix, and a rank-one object is one axis of rows. The matrix axis is
// the object's shape[0] when it has one, the matrix-internal row axis is the matrix's
// output-feature axis, and the column axis is its input-feature axis. A contiguous tensor of rank
// four or more keeps shape[0] as the matrix axis and its last dimension as the column axis, and
// takes the product of the dimensions between them as the matrix-internal row axis.
//
// The matrix axis and the matrix-internal row axis are not the same axis. `tensor_row_slice`
// splits axis 0, which is the matrix axis of a rank-three object -- the expert axis of a batched
// expert library -- and the matrix-internal row axis of a rank-two object. A view of a batched
// object therefore reports an expert number in `matrix_begin` and a row number inside that
// expert's own matrix in `matrix_row_begin`.
//
// Every layout the reader accepts counts its elements matrix-major and, inside one matrix,
// row-major over the logical columns. The row-split layout counts the logical columns only: the
// padded K of its physical groups is not part of the logical order, so an interval that reaches
// into the pad is outside the tensor.
//
// An interval that no rectangle describes is an error: the view never approximates one. An empty
// interval has no rectangle either, so it is an error as well. A rejected interval reports the
// layout, the interval and the object's element count.
//
// The view reports the layout it was asked about, the first element index of the interval and the
// number of elements in it; all three follow from the arguments. Whether the rectangle covers the
// whole tensor is decided by the three extents `matrix_count`, `matrix_row_count` and
// `column_count`.
struct ElementIntervalView {
    StorageLayout layout = StorageLayout::ContiguousLeV1;
    std::uint64_t matrix_begin = 0;
    std::uint64_t matrix_count = 0;
    std::uint64_t matrix_row_begin = 0;
    std::uint64_t matrix_row_count = 0;
    std::uint64_t column_begin = 0;
    std::uint64_t column_count = 0;
    std::uint64_t linear_begin = 0;
    std::uint64_t element_count = 0;
};

// The logical rectangle the half open element interval [begin, begin + count) covers inside a
// tensor of the given layout, format and shape. The interval must lie inside the tensor and must
// be a rectangle of the tensor's logical axes; anything else throws ArtifactError.
ElementIntervalView tensor_element_interval_view(StorageLayout layout, NumericFormat format,
                                                 std::span<const std::uint64_t> shape,
                                                 std::uint64_t begin, std::uint64_t count);

// The number of logical elements of a tensor: the elements in the tensor's own logical order,
// without the padding a layout adds for its groups or tiles. A rank-three object counts the
// elements of every matrix it stacks, a rank-two object its rows and columns, a rank-one object
// its rows. It is the element count of the whole tensor, so
// `tensor_element_interval_view(layout, format, shape, 0, count)` is the rectangle of the whole
// tensor.
std::uint64_t tensor_logical_element_count(StorageLayout layout, NumericFormat format,
                                           std::span<const std::uint64_t> shape);

// --- Tensor slicing (TP2 sharded materialization) ---------------------------------------------
//
// A shard of a tensor is a *standalone tensor of the same layout and format* whose logical shape
// is the parent's with one axis narrowed. Producing one is a pure byte operation: no layout of
// ours interleaves logical values with anything that has to be recomputed, so a shard is always
// some list of contiguous parent byte ranges copied into the shard's own plane offsets. Those
// ranges are what `TensorSlice` carries; the geometry that decides them is documented in
// docs/maintainer/storage-layouts.md and re-derived per layout in storage_layouts.cpp.

// One contiguous byte range copied from a parent payload into a shard.
struct PlaneCopy {
    std::uint64_t source_offset = 0; // relative to the parent object's payload begin
    std::uint64_t dest_offset   = 0; // relative to the shard's own payload begin
    std::uint64_t bytes         = 0;
};

// A half-open [begin, begin + count) range along the sliced axis, in logical coordinates.
struct SliceRange {
    std::uint64_t begin = 0;
    std::uint64_t count = 0;
};

struct TensorSlice {
    // The shard's own encoded size, i.e. tensor_encoded_size() of the narrowed logical shape.
    std::uint64_t encoded_bytes = 0;
    std::vector<PlaneCopy> copies;
    std::vector<std::byte> prefix;
};

// Row (axis 0) slice. The shard is the concatenation of `rows` in the order given; the ranges
// must be nonempty, ascending, disjoint, and inside the tensor. Multiple ranges exist because a
// fused object (attention query|key|gate|value, MLP gate|up) contributes one row block per fused
// sub-matrix to each device.
TensorSlice tensor_row_slice(StorageLayout layout, NumericFormat format,
                             std::span<const std::uint64_t> shape,
                             std::span<const SliceRange> rows,
                             std::span<const std::byte> payload = {});

// Column (axis 1) slice, rank two only. The shard is the concatenation of `columns` within every
// row, in the order given; the ranges must be nonempty, ascending, disjoint, and inside the
// tensor. Only contiguous-le-v1 accepts more than one range -- every other layout groups, tiles,
// or swizzles along the column axis, so a multi-range shard would not reconstruct into its own
// geometry, and >1 range is rejected there. Multiple ranges exist because a depthwise object whose
// channel axis is the column axis (GDN `gdn/convolution`, 10240 = Q|K|V) contributes one channel
// block per fused section to each device, exactly as `tensor_row_slice` does for fused GEMMs.
TensorSlice tensor_column_slice(StorageLayout layout, NumericFormat format,
                                std::span<const std::uint64_t> shape,
                                std::span<const SliceRange> columns,
                                std::span<const std::byte> payload = {});

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
    std::uint64_t offset;
    std::uint64_t bytes;
};

struct ResourceDescriptor {
    std::string name;
    ResourceEncoding encoding;
    std::uint64_t offset;
    std::uint64_t bytes;
};

using ObjectDescriptor = std::variant<TensorDescriptor, ResourceDescriptor>;

std::string_view object_name(const ObjectDescriptor& object) noexcept;
std::uint64_t object_offset(const ObjectDescriptor& object) noexcept;
std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept;

struct PayloadSpan {
    std::uint64_t absolute_offset;
    std::span<const std::byte> data;
};

// One volume-contiguous piece of a logical payload range. A multi-volume artifact splits a
// range at its volume boundaries; a single-file artifact reports the range itself.
struct PayloadSegment {
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

struct ArtifactIdentity {
    std::string model_id;
    std::string weights_id;

    bool operator==(const ArtifactIdentity&) const = default;
};

// One part of a parameter binding: the element range the parameter occupies inside one object.
// The range is a half open [begin, end) interval of elements, counted in the object's own
// element order.
struct V3BindingPart {
    std::string object;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

// One operator use edge of a v3 directory: the parameter that consumes the input, and the
// activation policy the consuming operator runs under. The auxiliaries of an edge are the
// auxiliary inputs it reads: the role name, such as "activation_input_divisor", to the parts of
// the objects that carry the role.
struct V3Use {
    std::string parameter;
    std::string input;
    std::string activation_policy;
    std::map<std::string, std::vector<V3BindingPart>> auxiliaries;
};

// The directory metadata of a v3 container: the per component configuration, the parameter
// binding table and the operator use edges.
//
// A configuration value is always text, so a consumer never has to know whether the directory
// spelled a number, a boolean or a string: a number is its shortest round trip form, a boolean
// is "true" or "false", and a string is itself. A nested configuration object is flattened into
// dot joined key paths ("rope_parameters.rope_theta"). An array is not a scalar: an array whose
// elements are all scalars is stored as its elements joined by commas
// ("layer_types" -> "linear_attention,indexed_attention"), and a key whose array holds an
// element that is not a scalar, or holds no element at all, is skipped. A joined array is not
// reversible when one of its elements contains a comma itself.
//
// A binding names either one object or a list of parts. The bindings that name one object are in
// `bindings`; the bindings that name parts are in `part_bindings`, so that a binding is never
// dropped silently. A binding that names neither is not reported.
//
// `component_resources` names the resources a component owns: the resource role, such as
// "tokenizer.json", to the id of the object that carries it.
struct V3Directory {
    std::map<std::string, std::map<std::string, std::string>> component_config;
    std::map<std::string, std::map<std::string, std::string>> component_resources;
    std::map<std::string, std::string> bindings;
    std::map<std::string, std::vector<V3BindingPart>> part_bindings;
    std::vector<V3Use> uses;
};

class Reader {
public:
    static constexpr std::size_t direct_io_alignment = 4096;

    explicit Reader(const std::filesystem::path& path);
    ~Reader();

    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    const ArtifactIdentity& identity() const noexcept;
    const std::vector<ObjectDescriptor>& objects() const noexcept;
    const ObjectDescriptor* find(std::string_view name) const noexcept;
    // The v3 directory metadata. A v2 artifact carries none, and the 27B narrow adapter projects
    // the v3 directory away, so both report an empty directory.
    const V3Directory& v3_directory() const noexcept;

    std::uint64_t file_bytes() const noexcept;
    std::uint64_t payload_offset() const noexcept;
    PayloadSpan payload(const ObjectDescriptor& object) const;
    PayloadSpan payload(std::string_view name) const;
    // The volume-contiguous pieces of a logical payload range, in order. A range that spans
    // volumes is split at the volume boundaries; a range inside one volume is one piece.
    std::vector<PayloadSegment> payload_segments(std::uint64_t offset, std::uint64_t bytes) const;
    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
