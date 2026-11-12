#include "artifact/binder.h"
#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::ArtifactError;
using ninfer::artifact::Binder;
using ninfer::artifact::DeviceMaterialization;
using ninfer::artifact::MaterializationPlan;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::ShardAxis;
using ninfer::artifact::ShardPlacement;
using ninfer::artifact::SliceRange;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using ninfer::artifact::TensorPlacement;

constexpr std::uint64_t kEntryPayloadStart = 4808704;
constexpr std::uint64_t kArtifactBytes      = 108445200388;
constexpr std::uint64_t kPartPayloadStart   = 4096;

void fail(const std::string& message) { throw std::runtime_error(message); }

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct Volume {
    std::filesystem::path path;
    std::uint64_t file_base     = 0;
    std::uint64_t logical_begin = 0;
    std::uint64_t logical_bytes = 0;
};

std::vector<std::byte> read_file_range(const std::filesystem::path& path, std::uint64_t offset,
                                       std::uint64_t bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { fail("cannot open " + path.string()); }
    in.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> out(bytes);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (in.gcount() != static_cast<std::streamsize>(bytes)) {
        fail("short read from " + path.string());
    }
    return out;
}

std::vector<Volume> collect_volumes(const std::filesystem::path& entry) {
    std::vector<Volume> volumes;
    const auto entry_base = kEntryPayloadStart;
    volumes.push_back({entry, entry_base, 0,
                       std::filesystem::file_size(entry) - entry_base});
    for (int part = 1;; ++part) {
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), ".part-%04d", part);
        const auto path = std::filesystem::path(entry.string() + suffix);
        if (!std::filesystem::exists(path)) { break; }
        const auto bytes = std::filesystem::file_size(path) - kPartPayloadStart;
        volumes.push_back({path, kPartPayloadStart, volumes.back().logical_begin +
                                                       volumes.back().logical_bytes,
                           bytes});
    }
    return volumes;
}

std::vector<std::byte> assemble_from_volumes(const std::vector<Volume>& volumes,
                                             std::uint64_t offset, std::uint64_t bytes) {
    std::vector<std::byte> out;
    out.reserve(bytes);
    for (const auto& volume : volumes) {
        const auto begin = std::max(offset, volume.logical_begin);
        const auto end   = std::min(offset + bytes, volume.logical_begin + volume.logical_bytes);
        if (begin >= end) { continue; }
        const auto piece = read_file_range(volume.path,
                                           volume.file_base + (begin - volume.logical_begin),
                                           end - begin);
        out.insert(out.end(), piece.begin(), piece.end());
    }
    if (out.size() != bytes) { fail("volume assembly produced the wrong byte count"); }
    return out;
}

float half_to_float(std::uint16_t bits) {
    const auto exponent = static_cast<std::uint32_t>(bits >> 10) & 0x1Fu;
    const auto mantissa = static_cast<float>(bits & 0x3FFu);
    float value;
    if (exponent == 0) {
        value = std::ldexp(mantissa, -24);
    } else if (exponent == 31) {
        value = std::numeric_limits<float>::infinity();
    } else {
        value = std::ldexp(mantissa + 1024.0F, static_cast<int>(exponent) - 25);
    }
    return (bits & 0x8000u) != 0 ? -value : value;
}

const std::filesystem::path& artifact_path() {
    static const std::filesystem::path path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    return path;
}

void expect_artifact_error(const std::function<void()>& body, std::string_view needle) {
    try {
        body();
    } catch (const ArtifactError& error) {
        if (std::string_view(error.what()).find(needle) != std::string_view::npos) { return; }
        fail(std::string("wrong artifact error: ") + error.what());
    }
    fail("expected an ArtifactError");
}

struct Inventory {
    std::map<NumericFormat, std::uint64_t> tensors;
    std::uint64_t resources = 0;
};

Inventory inventory_of(const Reader& reader) {
    Inventory out;
    for (const auto& object : reader.objects()) {
        if (const auto* tensor = std::get_if<TensorDescriptor>(&object)) {
            ++out.tensors[tensor->format];
        } else {
            ++out.resources;
        }
    }
    return out;
}

void test_identity_and_inventory() {
    Reader reader(artifact_path());

    if (reader.identity().model_id != "qwen3.8-flash-next" ||
        reader.identity().weights_id != "nvfp4") {
        fail("unexpected artifact identity");
    }
    if (reader.file_bytes() != kArtifactBytes) { fail("unexpected artifact byte count"); }
    if (reader.objects().size() != 1319) { fail("unexpected object count"); }

    const auto inventory = inventory_of(reader);
    if (inventory.resources != 7) { fail("unexpected resource count"); }
    const std::map<NumericFormat, std::uint64_t> expected = {
        {NumericFormat::BF16, 916},
        {NumericFormat::NF4_G16_E4M3S, 128},
        {NumericFormat::W8G32_F16S, 96},
        {NumericFormat::NVFP4, 96},
        {NumericFormat::FP32, 73},
        {NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, 2},
        {NumericFormat::Q4G64_F16S, 1},
    };
    if (inventory.tensors != expected) {
        fail("artifact format inventory does not match the recorded census");
    }

    const auto report_path = std::filesystem::path(artifact_path().string() + ".conversion.json");
    if (std::filesystem::exists(report_path)) {
        std::ifstream in(report_path);
        const auto report = nlohmann::json::parse(in);
        if (report.at("objects").get<std::uint64_t>() != reader.objects().size()) {
            fail("conversion report object count differs from the artifact directory");
        }
        const std::map<NumericFormat, std::string_view> spellings = {
            {NumericFormat::BF16, "bf16"},
            {NumericFormat::NF4_G16_E4M3S, "nf4_g16_e4m3fn"},
            {NumericFormat::W8G32_F16S, "q8_g32_fp16"},
            {NumericFormat::NVFP4, "nvfp4"},
            {NumericFormat::FP32, "fp32"},
            {NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, "fp8_e4m3fn_block128x128_bf16"},
            {NumericFormat::Q4G64_F16S, "q4_g64_fp16"},
        };
        std::size_t reported  = 0;
        std::size_t shortfall = 0;
        for (const auto& [format, spelling] : spellings) {
            const auto count = inventory.tensors.at(format);
            const auto declared = report.at("formats").at(std::string(spelling)).get<std::uint64_t>();
            reported += declared;
            if (declared == count) { continue; }
            if (format == NumericFormat::FP32 && declared + 1 == count) {
                ++shortfall;
                continue;
            }
            fail("conversion report count differs from the artifact directory: " +
                 std::string(spelling) + " report " + std::to_string(declared) + " vs " +
                 std::to_string(count));
        }
        if (reported + shortfall != reader.objects().size() - inventory.resources) {
            fail("conversion report formats do not cover the artifact tensors");
        }
    }
}

void test_every_object_matches_its_geometry() {
    Reader reader(artifact_path());
    for (const auto& object : reader.objects()) {
        const auto offset = ninfer::artifact::object_offset(object);
        const auto bytes  = ninfer::artifact::object_bytes(object);
        const auto* tensor = std::get_if<TensorDescriptor>(&object);
        if (tensor == nullptr) {
            const auto* resource = std::get_if<ResourceDescriptor>(&object);
            if (offset % ninfer::artifact::resource_alignment(resource->encoding) != 0) {
                fail("resource offset is not aligned");
            }
            continue;
        }
        if (offset % ninfer::artifact::tensor_alignment(tensor->layout) != 0) {
            fail("tensor offset is not aligned: " + tensor->name);
        }
        const auto encoded =
            ninfer::artifact::tensor_encoded_size(tensor->layout, tensor->format, tensor->shape,
                                                 bytes);
        if (encoded != bytes) {
            fail("object byte count differs from its storage geometry: " + tensor->name);
        }
        const StorageLayout expected_layout =
            tensor->format == NumericFormat::NF4_G16_E4M3S      ? StorageLayout::RowPagedG16x4KV1
            : tensor->format == NumericFormat::FP8_E4M3FN_BLOCK128_BF16S
                ? StorageLayout::BlockScaleM128K128V1
            : tensor->format == NumericFormat::NVFP4            ? StorageLayout::BatchedBlockScaleK16M128x4V1
            : tensor->format == NumericFormat::W8G32_F16S       ? StorageLayout::RowSplitK128V1
            : tensor->format == NumericFormat::Q4G64_F16S       ? StorageLayout::RowSplitK128V1
                                                                : StorageLayout::ContiguousLeV1;
        if (tensor->layout != expected_layout) {
            fail("object layout does not match its format: " + tensor->name);
        }
    }
}

void test_cross_volume_objects() {
    Reader reader(artifact_path());
    const auto volumes = collect_volumes(artifact_path());
    if (volumes.size() != 4) { fail("unexpected volume count"); }

    std::vector<std::string> crossing;
    for (const auto& object : reader.objects()) {
        if (reader.payload_segments(ninfer::artifact::object_offset(object),
                                    ninfer::artifact::object_bytes(object))
                .size() > 1) {
            crossing.push_back(std::string(ninfer::artifact::object_name(object)));
        }
    }
    const std::vector<std::string> expected = {"weight/000028", "weight/000074", "weight/000171"};
    if (crossing != expected) { fail("crossing objects do not match the recorded set"); }

    for (const auto& object : reader.objects()) {
        const auto name = std::string(ninfer::artifact::object_name(object));
        if (std::find(expected.begin(), expected.end(), name) == expected.end()) { continue; }
        const auto offset = ninfer::artifact::object_offset(object);
        const auto bytes  = ninfer::artifact::object_bytes(object);
        expect_artifact_error([&] { (void)reader.payload(object); }, "spans volumes");

        const auto aligned_bytes = align_up(bytes, 4096);
        void* buffer_raw         = nullptr;
        if (posix_memalign(&buffer_raw, 4096, aligned_bytes) != 0) {
            fail("aligned read buffer allocation failed");
        }
        std::memset(buffer_raw, 0xEE, aligned_bytes);
        std::span<std::byte> buffer(static_cast<std::byte*>(buffer_raw), aligned_bytes);
        const auto read = reader.read_direct(offset, buffer);
        if (read != aligned_bytes) { fail("crossing object read returned a short count: " + name); }
        const auto expected_bytes = assemble_from_volumes(volumes, offset, bytes);
        const auto matches =
            std::equal(expected_bytes.begin(), expected_bytes.end(), buffer.begin());
        std::free(buffer_raw);
        if (!matches) { fail("crossing object bytes differ from the volume files: " + name); }
        const auto pieces = reader.payload_segments(offset, bytes);
        if (pieces.size() != 2 || pieces.front().offset != offset ||
            pieces.back().offset + pieces.back().bytes != offset + bytes) {
            fail("crossing object segments do not tile its range: " + name);
        }
    }
}

void test_q8_g32_geometry_and_scales() {
    Reader reader(artifact_path());
    const auto* sample = std::get_if<TensorDescriptor>(reader.find("weight/000229"));
    if (sample == nullptr || sample->format != NumericFormat::W8G32_F16S) {
        fail("the q8_g32_fp16 sample is missing");
    }
    if (ninfer::artifact::parse_v3_format("q8_g32_fp16") != NumericFormat::W8G32_F16S) {
        fail("q8_g32_fp16 does not denote the shared W8G32 format");
    }

    std::vector<std::string> names;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<TensorDescriptor>(&object);
        if (tensor == nullptr || tensor->format != NumericFormat::W8G32_F16S) { continue; }
        const auto rows     = tensor->shape.at(0);
        const auto columns  = tensor->shape.at(1);
        const auto padded   = align_up(columns, 128);
        const auto low_bytes   = rows * padded;
        const auto scale_bytes = rows * (padded / 32) * 2;
        const auto scale_start = align_up(low_bytes, 256);
        if (scale_start + scale_bytes != tensor->bytes) {
            fail("independent q8_g32 geometry disagrees with the object: " + tensor->name);
        }
        const auto geometry = ninfer::artifact::row_split_geometry(tensor->format, tensor->shape);
        if (geometry.low_plane_bytes != low_bytes || geometry.high_plane_bytes != 0 ||
            geometry.scale_plane_offset != scale_start ||
            geometry.scale_plane_bytes != scale_bytes || geometry.encoded_bytes != tensor->bytes) {
            fail("row_split geometry disagrees with the independent formula: " + tensor->name);
        }
        if (names.size() < 4) { names.push_back(tensor->name); }
    }
    if (names.size() != 4) { fail("unexpected q8_g32_fp16 sample count"); }

    for (const auto& name : names) {
        const auto* tensor = std::get_if<TensorDescriptor>(reader.find(name));
        const auto payload = reader.payload(*tensor);
        const auto geometry = ninfer::artifact::row_split_geometry(tensor->format, tensor->shape);
        if (geometry.scale_plane_offset + 64 > payload.data.size()) {
            fail("q8_g32_fp16 scale plane is truncated: " + name);
        }
        for (std::size_t index = 0; index < 32; ++index) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, payload.data.data() + geometry.scale_plane_offset + index * 2, 2);
            const auto exponent = static_cast<unsigned>(bits >> 10) & 0x1Fu;
            const auto scale    = half_to_float(bits);
            if ((bits & 0x8000u) != 0 || exponent == 0 || exponent > 24 ||
                !std::isfinite(scale) || scale <= 1e-7F || scale >= 1e2F) {
                fail("q8_g32_fp16 scale plane does not decode as FP16: " + name);
            }
        }
    }
}

ShardPlacement expert_halves() {
    ShardPlacement placement;
    placement.axis             = ShardAxis::Rows;
    placement.device_ranges[0] = {SliceRange{0, 256}};
    placement.device_ranges[1] = {SliceRange{256, 256}};
    return placement;
}

void require_in_bounds(const std::vector<ninfer::artifact::PlaneCopy>& copies,
                       std::uint64_t parent_bytes, std::uint64_t shard_bytes) {
    std::uint64_t copied = 0;
    for (const auto& copy : copies) {
        if (copy.source_offset + copy.bytes > parent_bytes ||
            copy.dest_offset + copy.bytes > shard_bytes) {
            fail("shard copy runs past its parent or shard");
        }
        copied += copy.bytes;
    }
    if (copied != shard_bytes) { fail("shard copies do not fill the shard"); }
}

void test_nvfp4_expert_shard_smoke() {
    Reader reader(artifact_path());
    const std::array<std::uint64_t, 3> shape = {512, 1280, 2560};
    const auto* tensor = std::get_if<TensorDescriptor>(reader.find("weight/000000"));
    if (tensor == nullptr || tensor->format != NumericFormat::NVFP4 ||
        tensor->shape != std::vector<std::uint64_t>(shape.begin(), shape.end())) {
        fail("the nvfp4 gate_up sample is missing");
    }
    const auto parent = ninfer::artifact::batched_block_scale_geometry(tensor->format, shape);
    const auto half = ninfer::artifact::batched_block_scale_geometry(
        tensor->format, std::array<std::uint64_t, 3>{256, 1280, 2560});

    Binder binder(reader, 2);
    binder.set_shard_resolver([&](std::string_view name) {
        if (name != "weight/000000") { fail("unexpected shard request: " + std::string(name)); }
        return expert_halves();
    });
    const auto handle = binder.require_tensor("weight/000000", NumericFormat::NVFP4,
                                              StorageLayout::BatchedBlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(handle);
    binder.validate_unconsumed_matching();
    const auto plan = binder.finish();

    if (plan.device_count != 2 || plan.device_objects.size() != 2) {
        fail("expert shard plan did not split across two devices");
    }
    for (std::size_t index = 0; index < plan.device_objects.size(); ++index) {
        const DeviceMaterialization& shard = plan.device_objects[index];
        if (shard.device != static_cast<int>(index) || shard.bytes != half.encoded_bytes) {
            fail("expert shard size does not match the half-expert geometry");
        }
        require_in_bounds(shard.copies, parent.encoded_bytes, half.encoded_bytes);
    }
}

void test_nvfp4_column_shard_smoke() {
    Reader reader(artifact_path());
    const std::array<std::uint64_t, 3> shape = {512, 1280, 2560};
    const auto* tensor = std::get_if<TensorDescriptor>(reader.find("weight/000000"));
    if (tensor == nullptr) { fail("the nvfp4 gate_up sample is missing"); }
    const auto parent = ninfer::artifact::batched_block_scale_geometry(tensor->format, shape);
    const auto half = ninfer::artifact::batched_block_scale_geometry(
        tensor->format, std::array<std::uint64_t, 3>{512, 1280, 1280});

    for (const std::uint64_t begin : {std::uint64_t{0}, std::uint64_t{1280}}) {
        const std::array<SliceRange, 1> ranges = {SliceRange{begin, 1280}};
        const auto slice = ninfer::artifact::tensor_column_slice(
            tensor->layout, tensor->format, shape, ranges);
        if (slice.encoded_bytes != half.encoded_bytes) {
            fail("column shard size does not match the half-width geometry");
        }
        require_in_bounds(slice.copies, parent.encoded_bytes, half.encoded_bytes);
    }
}

// The logical axes of an object, as the artifact's own shapes spell them: a stack of matrices, and
// inside one matrix the rows and the columns.
struct ViewAxes {
    std::uint64_t matrices;
    std::uint64_t rows;
    std::uint64_t columns;
};

ViewAxes view_axes(const TensorDescriptor& tensor) {
    if (tensor.shape.size() >= 3) {
        return ViewAxes{tensor.shape[0], tensor.shape[1], tensor.shape[2]};
    }
    if (tensor.shape.size() == 2) { return ViewAxes{1, tensor.shape[0], tensor.shape[1]}; }
    if (tensor.shape.size() == 1) { return ViewAxes{1, tensor.shape[0], 1}; }
    return ViewAxes{1, 1, 1};
}

// A view must cover exactly the elements of the interval it was asked about. The element order is
// matrix-major and row-major inside one matrix, so the first and the last element of the reported
// rectangle fix the interval, and the rectangle must be contiguous in that order.
void check_view_covers(const TensorDescriptor& tensor,
                       const ninfer::artifact::ElementIntervalView& view, std::uint64_t begin,
                       std::uint64_t end, const std::string& parameter) {
    const auto axes       = view_axes(tensor);
    const auto per_matrix = axes.rows * axes.columns;
    if (view.element_count != end - begin) {
        fail("the view of a part does not count the part's elements: " + parameter);
    }
    const auto first = view.matrix_begin * per_matrix + view.row_begin * axes.columns +
                       view.column_begin;
    if (first != begin) {
        fail("the view of a part does not start at the part's first element: " + parameter);
    }
    const auto last = (view.matrix_begin + view.matrix_count - 1) * per_matrix +
                      (view.row_begin + view.row_count - 1) * axes.columns + view.column_begin +
                      view.column_count - 1;
    if (last + 1 != end) {
        fail("the view of a part does not end at the part's last element: " + parameter);
    }
    const bool contiguous =
        view.matrix_count == 1
            ? (view.row_count == 1 ||
               (view.column_begin == 0 && view.column_count == axes.columns))
            : (view.row_begin == 0 && view.row_count == axes.rows && view.column_begin == 0 &&
               view.column_count == axes.columns);
    if (!contiguous) {
        fail("the view of a part is not a rectangle of the logical axes: " + parameter);
    }
}

// Every part binding of the artifact is an element interval of one object, and every one of them
// must map to a rectangle of that object's logical axes. The parts of an object also tile it
// exactly, which is the artifact's own statement that the element order is the one the views
// assume.
void test_part_bindings_map_to_logical_views() {
    Reader reader(artifact_path());
    const auto& directory = reader.v3_directory();

    std::map<StorageLayout, std::uint64_t> parts_per_layout;
    std::map<std::string, std::vector<std::pair<std::uint64_t, std::uint64_t>>> parts_per_object;
    std::uint64_t parts = 0;
    for (const auto& [parameter, part_list] : directory.part_bindings) {
        for (const auto& part : part_list) {
            const auto* tensor = std::get_if<TensorDescriptor>(reader.find(part.object));
            if (tensor == nullptr) {
                fail("a part binding names an object that is not a tensor: " + part.object);
            }
            if (part.end <= part.begin) {
                fail("a part binding names an empty interval: " + parameter);
            }
            const auto view = ninfer::artifact::tensor_element_interval_view(
                tensor->layout, tensor->format, tensor->shape, part.begin, part.end - part.begin);
            if (view.layout != tensor->layout) {
                fail("the view of a part does not report the object's layout: " + parameter);
            }
            check_view_covers(*tensor, view, part.begin, part.end, parameter);
            ++parts_per_layout[tensor->layout];
            parts_per_object[part.object].emplace_back(part.begin, part.end);
            ++parts;
        }
    }

    if (parts != 76242) { fail("the artifact part census changed"); }
    const std::map<StorageLayout, std::uint64_t> expected = {
        {StorageLayout::BatchedBlockScaleK16M128x4V1, 73728},
        {StorageLayout::BlockScaleM128K128V1, 1536},
        {StorageLayout::ContiguousLeV1, 682},
        {StorageLayout::RowPagedG16x4KV1, 128},
        {StorageLayout::RowSplitK128V1, 168},
    };
    if (parts_per_layout != expected) {
        fail("the artifact part census per layout changed");
    }

    for (const auto& [object, intervals] : parts_per_object) {
        const auto* tensor = std::get_if<TensorDescriptor>(reader.find(object));
        const auto axes   = view_axes(*tensor);
        const auto total  = axes.matrices * axes.rows * axes.columns;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted = intervals;
        std::sort(sorted.begin(), sorted.end());
        std::uint64_t cursor = 0;
        for (const auto& [begin, end] : sorted) {
            if (begin != cursor) { fail("the parts of an object do not tile it: " + object); }
            cursor = end;
        }
        if (cursor != total) { fail("the parts of an object do not cover it: " + object); }
    }

    std::cout << "flash-next part bindings: " << parts << " intervals over "
              << parts_per_object.size()
              << " objects, every one a rectangle of its object's logical axes\n";
}

} // namespace

int main() {
    const auto path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
        return 77;
    }
    try {
        test_identity_and_inventory();
        test_every_object_matches_its_geometry();
        test_cross_volume_objects();
        test_q8_g32_geometry_and_scales();
        test_nvfp4_expert_shard_smoke();
        test_nvfp4_column_shard_smoke();
        test_part_bindings_map_to_logical_views();
    } catch (const std::exception& error) {
        std::cerr << "flash-next real artifact test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "flash-next real artifact: identity, inventory, geometry, cross-volume reads, "
                 "q8_g32 scales and TP2 shard smoke all match\n";
    return 0;
}
