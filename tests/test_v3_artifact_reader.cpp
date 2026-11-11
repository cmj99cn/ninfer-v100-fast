#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

using ninfer::artifact::ArtifactError;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::ResourceEncoding;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;

constexpr std::array<std::uint8_t, 8> kEntryMagic = {'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
constexpr std::array<std::uint8_t, 8> kPartMagic  = {'N', 'I', 'N', 'P', 'R', 'T', 0, 3};
constexpr std::uint64_t kHeaderBytes      = 32;
constexpr std::uint64_t kPayloadAlignment = 4096;
constexpr std::uint64_t kPartPayloadBytes = 4096;

constexpr std::array<std::byte, 16> kArtifactId = {
    std::byte{0x41}, std::byte{0x42}, std::byte{0x43}, std::byte{0x44},
    std::byte{0x45}, std::byte{0x46}, std::byte{0x47}, std::byte{0x48},
    std::byte{0x49}, std::byte{0x4A}, std::byte{0x4B}, std::byte{0x4C},
    std::byte{0x4D}, std::byte{0x4E}, std::byte{0x4F}, std::byte{0x50},
};

struct FixtureOptions {
    std::uint64_t part1_index    = 1;
    std::uint64_t part2_index    = 2;
    bool part2_foreign_id        = false;
    bool swap_part_contents      = false;
};

struct PlacedObject {
    Json spec;
    std::size_t file = 0;
    std::uint64_t file_offset = 0;
};

struct V3Fixture {
    std::filesystem::path entry;
    std::filesystem::path part1;
    std::filesystem::path part2;
    std::uint64_t total_file_bytes = 0;

    ~V3Fixture() {
        std::error_code ignored;
        std::filesystem::remove(entry, ignored);
        std::filesystem::remove(part1, ignored);
        std::filesystem::remove(part2, ignored);
    }
};

std::filesystem::path fixture_base() {
    return std::filesystem::temp_directory_path() /
           ("ninfer-v3-artifact-test-" + std::to_string(static_cast<long long>(::getpid())));
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void store_u64_le(std::byte* output, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        output[i] = std::byte(static_cast<std::uint8_t>((value >> (i * 8)) & 0xff));
    }
}

void write_file(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) { throw std::runtime_error("failed to write a v3 fixture file"); }
}

std::vector<std::byte> marker_payload(std::uint64_t bytes, std::uint8_t marker) {
    return std::vector<std::byte>(bytes, std::byte{marker});
}

bool is_marker(std::byte value, std::byte marker) { return value == marker; }

std::vector<PlacedObject> canonical_objects() {
    return {
        {Json{{"id", "alpha"},
              {"kind", "tensor"},
              {"format", "bf16"},
              {"layout", "contiguous_le_v1"},
              {"shape", Json::array({2, 2})},
              {"bytes", 8}},
         0, 0},
        {Json{{"id", "beta"},
              {"kind", "tensor"},
              {"format", "fp32"},
              {"layout", "contiguous_le_v1"},
              {"shape", Json::array({1, 2})},
              {"bytes", 8}},
         1, 0},
        {Json{{"id", "gamma"},
              {"kind", "resource"},
              {"encoding", "raw_bytes_v1"},
              {"bytes", 5}},
         2, 0},
    };
}

V3Fixture write_v3_multi_part(const std::vector<PlacedObject>& objects,
                               const std::vector<std::vector<std::byte>>& payloads,
                               const FixtureOptions& options = {}) {
    const auto base = fixture_base();
    V3Fixture fixture{base.string() + ".ninfer",
                       base.string() + ".ninfer.part-0001",
                       base.string() + ".ninfer.part-0002",
                       0};
    const std::array<std::filesystem::path, 2> part_paths = {fixture.part1, fixture.part2};

    std::vector<std::uint64_t> logical_begin(payloads.size(), 0);
    for (std::size_t index = 1; index < payloads.size(); ++index) {
        logical_begin[index] = logical_begin[index - 1] + payloads[index - 1].size();
    }

    Json files = Json::array();
    for (std::size_t index = 0; index < payloads.size(); ++index) {
        files.push_back(Json{
            {"path", index == 0 ? Json(nullptr) : Json(part_paths[index - 1].filename().string())},
            {"payload_bytes", payloads[index].size()},
        });
    }

    Json directory{
        {"metadata", Json{{"name", "qwen3.8-flash-next"}}},
        {"files", std::move(files)},
        {"objects", Json::array()},
        {"bindings", Json::object()},
        {"uses", Json::array()},
        {"components", Json::object()},
    };
    for (const PlacedObject& placed : objects) {
        Json object = placed.spec;
        object["offset"] = logical_begin[placed.file] + placed.file_offset;
        directory["objects"].push_back(std::move(object));
    }

    std::string text = directory.dump();
    const auto region    = align_up(kHeaderBytes + text.size(), kPayloadAlignment);
    const auto json_bytes = region - kHeaderBytes;
    text.append(json_bytes - text.size(), ' ');

    std::vector<std::byte> entry_file(region + payloads[0].size(), std::byte{0});
    for (std::size_t i = 0; i < kEntryMagic.size(); ++i) {
        entry_file[i] = std::byte{kEntryMagic[i]};
    }
    store_u64_le(entry_file.data() + 8, json_bytes);
    std::copy(kArtifactId.begin(), kArtifactId.end(), entry_file.data() + 16);
    std::memcpy(entry_file.data() + kHeaderBytes, text.data(), text.size());
    std::copy(payloads[0].begin(), payloads[0].end(),
              entry_file.data() + static_cast<std::ptrdiff_t>(region));

    const auto part_file = [](const std::vector<std::byte>& payload, std::uint64_t index,
                              bool foreign_id) {
        std::vector<std::byte> file(kPayloadAlignment + payload.size(), std::byte{0});
        for (std::size_t i = 0; i < kPartMagic.size(); ++i) {
            file[i] = std::byte{kPartMagic[i]};
        }
        store_u64_le(file.data() + 8, index);
        std::copy(kArtifactId.begin(), kArtifactId.end(), file.data() + 16);
        if (foreign_id) { file[31] = std::byte{0xFF}; }
        std::copy(payload.begin(), payload.end(), file.data() + kPayloadAlignment);
        return file;
    };
    std::vector<std::vector<std::byte>> part_bytes;
    for (std::size_t index = 1; index < payloads.size(); ++index) {
        const auto declared = index == 1 ? options.part1_index : options.part2_index;
        part_bytes.push_back(
            part_file(payloads[index], declared, index == 2 && options.part2_foreign_id));
    }

    write_file(fixture.entry, entry_file);
    if (part_bytes.size() > 0) {
        write_file(fixture.part1, options.swap_part_contents ? part_bytes.at(1) : part_bytes.at(0));
    }
    if (part_bytes.size() > 1) {
        write_file(fixture.part2, options.swap_part_contents ? part_bytes.at(0) : part_bytes.at(1));
    }
    fixture.total_file_bytes = entry_file.size();
    for (const auto& bytes : part_bytes) { fixture.total_file_bytes += bytes.size(); }
    return fixture;
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view needle) {
    try {
        function();
    } catch (const ArtifactError& error) {
        if (std::string_view(error.what()).find(needle) == std::string_view::npos) {
            throw std::runtime_error(std::string("unexpected rejection: ") + error.what());
        }
        return;
    }
    throw std::runtime_error("invalid v3 multi-part artifact was accepted");
}

// The v3 container spells its formats, layouts and encodings in lower case snake case. Its
// spelling table is upstream's legal set, and every spelling denotes a public enum value. A
// value's canonical name is the spelling of the container that introduced it, so the values the
// v3 container introduced round trip through the v3 table.
void test_v3_spelling_round_trip() {
    struct FormatCase {
        std::string_view spelling;
        NumericFormat value;
        std::string_view canonical_name;
        bool v3_native;
    };
    struct LayoutCase {
        std::string_view spelling;
        StorageLayout value;
        std::string_view canonical_name;
        bool v3_native;
    };

    const std::array<FormatCase, 9> format_cases = {{
        {"bf16", NumericFormat::BF16, "BF16", false},
        {"fp32", NumericFormat::FP32, "FP32", false},
        {"int32", NumericFormat::I32, "I32", false},
        {"q4_g64_fp16", NumericFormat::Q4G64_F16S, "Q4G64_F16S", false},
        {"q8_g32_fp16", NumericFormat::W8G32_F16S, "W8G32_F16S", false},
        {"nvfp4", NumericFormat::NVFP4, "NVFP4", false},
        {"nf4_g16_e4m3fn", NumericFormat::NF4_G16_E4M3S, "nf4_g16_e4m3fn", true},
        {"fp8_e4m3fn_block128x128_bf16", NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
         "fp8_e4m3fn_block128x128_bf16", true},
        {"fp8_e4m3fn_row_bf16", NumericFormat::FP8_E4M3FN_ROW_BF16S, "FP8_E4M3FN_ROW_BF16S", false},
    }};
    const std::array<LayoutCase, 6> layout_cases = {{
        {"contiguous_le_v1", StorageLayout::ContiguousLeV1, "contiguous-le-v1", false},
        {"row_split_k128_v1", StorageLayout::RowSplitK128V1, "row-split-k128-v1", false},
        {"row_paged_g16_4k_v1", StorageLayout::RowPagedG16x4KV1, "row_paged_g16_4k_v1", true},
        {"block_scale_k16_m128x4_v1", StorageLayout::BlockScaleK16M128x4V1,
         "blockscale-k16-m128x4-v1", false},
        {"batched_block_scale_k16_m128x4_v1", StorageLayout::BatchedBlockScaleK16M128x4V1,
         "batched_block_scale_k16_m128x4_v1", true},
        {"block_scale_m128_k128_v1", StorageLayout::BlockScaleM128K128V1,
         "block_scale_m128_k128_v1", true},
    }};

    for (const auto& entry : format_cases) {
        if (ninfer::artifact::parse_v3_format(entry.spelling) != entry.value) {
            throw std::runtime_error("v3 format spelling " + std::string(entry.spelling) +
                                     " does not denote its public enum value");
        }
        if (ninfer::artifact::format_name(entry.value) != entry.canonical_name) {
            throw std::runtime_error("v3 format value " + std::string(entry.canonical_name) +
                                     " carries a different canonical name");
        }
        if (entry.v3_native &&
            ninfer::artifact::parse_v3_format(entry.canonical_name) != entry.value) {
            throw std::runtime_error("v3 format spelling " + std::string(entry.spelling) +
                                     " does not round trip through its canonical name");
        }
    }
    for (const auto& entry : layout_cases) {
        if (ninfer::artifact::parse_v3_layout(entry.spelling) != entry.value) {
            throw std::runtime_error("v3 layout spelling " + std::string(entry.spelling) +
                                     " does not denote its public enum value");
        }
        if (ninfer::artifact::layout_name(entry.value) != entry.canonical_name) {
            throw std::runtime_error("v3 layout value " + std::string(entry.canonical_name) +
                                     " carries a different canonical name");
        }
        if (entry.v3_native &&
            ninfer::artifact::parse_v3_layout(entry.canonical_name) != entry.value) {
            throw std::runtime_error("v3 layout spelling " + std::string(entry.spelling) +
                                     " does not round trip through its canonical name");
        }
    }

    expect_artifact_error([] { ninfer::artifact::parse_v3_format("q9_g64_fp16"); },
                          "unknown v3 tensor format");
    expect_artifact_error([] { ninfer::artifact::parse_v3_layout("row_paged_g16_4k_v2"); },
                          "unknown v3 tensor layout");
    expect_artifact_error([] { ninfer::artifact::parse_v3_encoding("raw_bytes_v2"); },
                          "unknown v3 resource encoding");
}

// A v3 object reaches the reader as the same public enum value the v2 container's table denotes.
void test_v3_object_spelling_reaches_the_reader() {
    const std::array<std::uint64_t, 2> shape = {4, 256};
    const auto bytes =
        ninfer::artifact::row_split_geometry(NumericFormat::W8G32_F16S, shape).encoded_bytes;
    const std::vector<PlacedObject> objects = {
        {Json{{"id", "alpha"},
              {"kind", "tensor"},
              {"format", "bf16"},
              {"layout", "contiguous_le_v1"},
              {"shape", Json::array({2, 2})},
              {"bytes", 8}},
         0, 0},
        {Json{{"id", "q8"},
              {"kind", "tensor"},
              {"format", "q8_g32_fp16"},
              {"layout", "row_split_k128_v1"},
              {"shape", Json::array({4, 256})},
              {"bytes", bytes}},
         0, 256},
    };
    const auto fixture = write_v3_multi_part(objects, {marker_payload(kPartPayloadBytes, 0x11)});
    Reader reader(fixture.entry);

    const auto* q8 = std::get_if<TensorDescriptor>(reader.find("q8"));
    if (q8 == nullptr || q8->format != NumericFormat::W8G32_F16S ||
        q8->layout != StorageLayout::RowSplitK128V1 || q8->bytes != bytes) {
        throw std::runtime_error("v3 object spelling did not reach the reader as its enum value");
    }
}

void test_v3_unknown_spelling_rejected() {
    const std::vector<PlacedObject> objects = {
        {Json{{"id", "alpha"},
              {"kind", "tensor"},
              {"format", "q8_g33_fp16"},
              {"layout", "contiguous_le_v1"},
              {"shape", Json::array({2, 2})},
              {"bytes", 8}},
         0, 0},
    };
    const auto fixture = write_v3_multi_part(objects, {marker_payload(kPartPayloadBytes, 0x11)});
    expect_artifact_error([&] { Reader reader(fixture.entry); }, "unknown v3 tensor format");
}

void test_identity_synthesized() {
    const auto fixture =
        write_v3_multi_part(canonical_objects(),
                            {marker_payload(kPartPayloadBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)});
    Reader reader(fixture.entry);
    if (reader.identity().model_id != "qwen3.8-flash-next" ||
        reader.identity().weights_id != "nvfp4") {
        throw std::runtime_error("v3 multi-part identity mismatch");
    }
}

void test_objects_and_payloads_across_parts() {
    const auto fixture =
        write_v3_multi_part(canonical_objects(),
                            {marker_payload(kPartPayloadBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)});
    Reader reader(fixture.entry);

    if (reader.objects().size() != 3) {
        throw std::runtime_error("v3 multi-part object count mismatch");
    }
    const std::array<std::string_view, 3> names = {"alpha", "beta", "gamma"};
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (ninfer::artifact::object_name(reader.objects()[i]) != names[i] ||
            reader.find(names[i]) != &reader.objects()[i]) {
            throw std::runtime_error("v3 multi-part object index mismatch");
        }
    }

    const auto* alpha = std::get_if<TensorDescriptor>(reader.find("alpha"));
    const auto* beta  = std::get_if<TensorDescriptor>(reader.find("beta"));
    const auto* gamma = std::get_if<ResourceDescriptor>(reader.find("gamma"));
    if (alpha == nullptr || alpha->shape != std::vector<std::uint64_t>({2, 2}) ||
        alpha->format != NumericFormat::BF16 || alpha->layout != StorageLayout::ContiguousLeV1 ||
        alpha->bytes != 8 || beta == nullptr || beta->shape != std::vector<std::uint64_t>({1, 2}) ||
        beta->format != NumericFormat::FP32 || beta->layout != StorageLayout::ContiguousLeV1 ||
        beta->bytes != 8 || gamma == nullptr || gamma->encoding != ResourceEncoding::RawBytesV1 ||
        gamma->bytes != 5) {
        throw std::runtime_error("v3 multi-part object signature mismatch");
    }

    const auto alpha_payload = reader.payload(*alpha);
    const auto beta_payload  = reader.payload("beta");
    const auto gamma_payload = reader.payload(*gamma);
    if (alpha_payload.absolute_offset != 0 || beta_payload.absolute_offset != kPartPayloadBytes ||
        gamma_payload.absolute_offset != 2 * kPartPayloadBytes ||
        alpha_payload.data.size() != 8 || beta_payload.data.size() != 8 ||
        gamma_payload.data.size() != 5 ||
        !std::all_of(alpha_payload.data.begin(), alpha_payload.data.end(),
                     [](std::byte value) { return is_marker(value, std::byte{0x11}); }) ||
        !std::all_of(beta_payload.data.begin(), beta_payload.data.end(),
                     [](std::byte value) { return is_marker(value, std::byte{0x22}); }) ||
        !std::all_of(gamma_payload.data.begin(), gamma_payload.data.end(),
                     [](std::byte value) { return is_marker(value, std::byte{0x33}); }) ||
        reader.file_bytes() != fixture.total_file_bytes) {
        throw std::runtime_error("v3 multi-part payload mapping mismatch");
    }
}

void test_wrong_part_index_rejected() {
    FixtureOptions options;
    options.part2_index = 3;
    const auto fixture =
        write_v3_multi_part(canonical_objects(),
                            {marker_payload(kPartPayloadBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)},
                            options);
    expect_artifact_error([&] { Reader reader(fixture.entry); }, "declares part index");
}

void test_swapped_parts_rejected() {
    FixtureOptions options;
    options.swap_part_contents = true;
    const auto fixture =
        write_v3_multi_part(canonical_objects(),
                            {marker_payload(kPartPayloadBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)},
                            options);
    expect_artifact_error([&] { Reader reader(fixture.entry); }, "declares part index");
}

void test_foreign_part_id_rejected() {
    FixtureOptions options;
    options.part2_foreign_id = true;
    const auto fixture =
        write_v3_multi_part(canonical_objects(),
                            {marker_payload(kPartPayloadBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)},
                            options);
    expect_artifact_error([&] { Reader reader(fixture.entry); }, "belongs to a different artifact");
}

// An object that straddles a volume boundary is read in segments: one piece per volume, each
// piece's bytes equal to the bytes written into that volume.
void test_crossing_object_read_in_segments() {
    constexpr std::uint64_t kFirstVolumeBytes = 4096;
    constexpr std::uint64_t kCrossingBytes    = 512;
    constexpr std::uint64_t kCrossingOffset   = kFirstVolumeBytes - 256;
    auto objects = canonical_objects();
    objects.resize(1);
    objects.push_back({Json{{"id", "delta"},
                            {"kind", "tensor"},
                            {"format", "bf16"},
                            {"layout", "contiguous_le_v1"},
                            {"shape", Json::array({256, 1})},
                            {"bytes", kCrossingBytes}},
                       0, kCrossingOffset});
    const auto fixture =
        write_v3_multi_part(objects,
                            {marker_payload(kFirstVolumeBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)});
    Reader reader(fixture.entry);

    const auto* delta = std::get_if<TensorDescriptor>(reader.find("delta"));
    if (delta == nullptr) { throw std::runtime_error("crossing object is missing"); }
    const auto offset = ninfer::artifact::object_offset(*delta);
    const auto bytes  = ninfer::artifact::object_bytes(*delta);

    const auto segments = reader.payload_segments(offset, bytes);
    if (segments.size() != 2 || segments[0].offset != kCrossingOffset ||
        segments[0].bytes != 256 || segments[1].offset != kFirstVolumeBytes ||
        segments[1].bytes != kCrossingBytes - 256) {
        throw std::runtime_error("crossing object was not split at the volume boundary");
    }

    std::vector<std::byte> buffer(bytes, std::byte{0});
    const auto read = reader.read_direct(offset, buffer);
    if (read != bytes) { throw std::runtime_error("crossing object read returned a short count"); }
    const auto first = std::all_of(
        buffer.begin(), buffer.begin() + 256,
        [](std::byte value) { return is_marker(value, std::byte{0x11}); });
    const auto second = std::all_of(
        buffer.begin() + 256, buffer.end(),
        [](std::byte value) { return is_marker(value, std::byte{0x22}); });
    if (!first || !second) {
        throw std::runtime_error("crossing object bytes differ from the volumes");
    }

    expect_artifact_error([&] { reader.payload(*delta); }, "spans volumes");
}

void test_single_file_container_accepted() {
    const auto fixture = write_v3_multi_part(
        {canonical_objects().front()}, {marker_payload(kPartPayloadBytes, 0x11)});
    Reader reader(fixture.entry);
    const auto payload = reader.payload("alpha");
    if (reader.identity().model_id != "qwen3.8-flash-next" || reader.objects().size() != 1 ||
        payload.absolute_offset != 0 || payload.data.size() != 8 ||
        !std::all_of(payload.data.begin(), payload.data.end(),
                     [](std::byte value) { return is_marker(value, std::byte{0x11}); })) {
        throw std::runtime_error("single-file v3 container artifact mismatch");
    }
}

void test_direct_reads_resolve_volumes() {
    const auto fixture =
        write_v3_multi_part(canonical_objects(),
                            {marker_payload(kPartPayloadBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)});
    Reader reader(fixture.entry);
    const std::array<std::uint8_t, 3> markers = {0x11, 0x22, 0x33};
    for (std::size_t volume = 0; volume < markers.size(); ++volume) {
        auto* buffer =
            static_cast<std::byte*>(std::aligned_alloc(kPayloadAlignment, kPayloadAlignment));
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate a direct read buffer");
        }
        const auto read = reader.read_direct(volume * kPartPayloadBytes,
                                             std::span<std::byte>(buffer, kPayloadAlignment));
        const bool matched =
            read == kPayloadAlignment &&
            std::all_of(buffer, buffer + kPayloadAlignment, [&](std::byte value) {
                return is_marker(value, std::byte{markers[volume]});
            });
        std::free(buffer);
        if (!matched) {
            throw std::runtime_error("v3 multi-part direct read resolved the wrong volume");
        }
    }
}

// The materializer must serve a multi-volume artifact: an object that ends near a volume's end
// (its 4096-byte read request reaches past that end) and an object that straddles the boundary.
// Returns 0 when the case ran, 77 when it skipped. The skip path returns instead of exiting so
// that the fixture's destructor removes the files it wrote.
int test_multi_volume_materialization() {
    constexpr std::uint64_t kFirstVolumeBytes = 3000;
    constexpr std::uint64_t kTailBytes        = 256;
    constexpr std::uint64_t kTailOffset       = 2560;
    constexpr std::uint64_t kCrossingBytes    = 512;
    constexpr std::uint64_t kCrossingOffset   = kTailOffset + kTailBytes;
    const auto objects = std::vector<PlacedObject>{
        {Json{{"id", "tail"},
              {"kind", "tensor"},
              {"format", "bf16"},
              {"layout", "contiguous_le_v1"},
              {"shape", Json::array({128, 1})},
              {"bytes", kTailBytes}},
         0, kTailOffset},
        {Json{{"id", "crossing"},
              {"kind", "tensor"},
              {"format", "bf16"},
              {"layout", "contiguous_le_v1"},
              {"shape", Json::array({256, 1})},
              {"bytes", kCrossingBytes}},
         0, kCrossingOffset},
    };
    const auto fixture =
        write_v3_multi_part(objects,
                            {marker_payload(kFirstVolumeBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)});
    Reader reader(fixture.entry);

    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);
    if (device_count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    ninfer::artifact::Binder binder(reader);
    constexpr std::array<std::uint64_t, 2> tail_shape = {128, 1};
    const auto tail = binder.require_tensor("tail", NumericFormat::BF16,
                                            StorageLayout::ContiguousLeV1, tail_shape);
    binder.materialize_on_device(tail);
    constexpr std::array<std::uint64_t, 2> crossing_shape = {256, 1};
    const auto crossing =
        binder.require_tensor("crossing", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                              crossing_shape);
    binder.materialize_on_device(crossing);
    const auto plan = binder.finish();

    ninfer::DeviceContext device(0);
    auto materialized = ninfer::artifact::materialize(reader, plan, device);

    std::vector<std::byte> tail_bytes(kTailBytes);
    CUDA_CHECK(cudaMemcpy(tail_bytes.data(), materialized.device_data(tail), tail_bytes.size(),
                          cudaMemcpyDeviceToHost));
    std::vector<std::byte> crossing_bytes(kCrossingBytes);
    CUDA_CHECK(cudaMemcpy(crossing_bytes.data(), materialized.device_data(crossing),
                          crossing_bytes.size(), cudaMemcpyDeviceToHost));

    const auto tail_ok = std::all_of(
        tail_bytes.begin(), tail_bytes.end(),
        [](std::byte value) { return is_marker(value, std::byte{0x11}); });
    const auto split = kFirstVolumeBytes - kCrossingOffset;
    const auto crossing_first = std::all_of(
        crossing_bytes.begin(), crossing_bytes.begin() + split,
        [](std::byte value) { return is_marker(value, std::byte{0x11}); });
    const auto crossing_second = std::all_of(
        crossing_bytes.begin() + split, crossing_bytes.end(),
        [](std::byte value) { return is_marker(value, std::byte{0x22}); });
    if (!tail_ok || !crossing_first || !crossing_second) {
        throw std::runtime_error("multi-volume materialization produced wrong bytes");
    }
    return 0;
}

} // namespace

int main() {
    try {
        test_v3_spelling_round_trip();
        test_v3_object_spelling_reaches_the_reader();
        test_v3_unknown_spelling_rejected();
        test_identity_synthesized();
        test_objects_and_payloads_across_parts();
        test_wrong_part_index_rejected();
        test_swapped_parts_rejected();
        test_foreign_part_id_rejected();
        test_crossing_object_read_in_segments();
        test_single_file_container_accepted();
        test_direct_reads_resolve_volumes();
        const int skipped = test_multi_volume_materialization();
        if (skipped != 0) { return skipped; }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
