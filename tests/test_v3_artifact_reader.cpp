#include "artifact/reader.h"

#include <nlohmann/json.hpp>

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
    const auto directory_path = std::filesystem::temp_directory_path();
    V3Fixture fixture{directory_path / "v3_multi_entry.ninfer",
                       directory_path / "v3_multi_entry.ninfer.part-0001",
                       directory_path / "v3_multi_entry.ninfer.part-0002",
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
    const auto region   = align_up(kHeaderBytes + text.size(), kPayloadAlignment);
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

void test_crossing_object_rejected() {
    auto objects = canonical_objects();
    objects.pop_back();
    objects.push_back({Json{{"id", "delta"},
                            {"kind", "tensor"},
                            {"format", "bf16"},
                            {"layout", "contiguous_le_v1"},
                            {"shape", Json::array({256, 1})},
                            {"bytes", 512}},
                       1, kPartPayloadBytes - 256});
    const auto fixture =
        write_v3_multi_part(objects,
                            {marker_payload(kPartPayloadBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)});
    expect_artifact_error([&] { Reader reader(fixture.entry); }, "crosses a part boundary");
}

void test_single_file_container_accepted() {
    const auto fixture =
        write_v3_multi_part({canonical_objects().front()}, {marker_payload(kPartPayloadBytes, 0x11)});
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
        if (buffer == nullptr) { throw std::runtime_error("failed to allocate a direct read buffer"); }
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

} // namespace

int main() {
    try {
        test_identity_synthesized();
        test_objects_and_payloads_across_parts();
        test_wrong_part_index_rejected();
        test_swapped_parts_rejected();
        test_foreign_part_id_rejected();
        test_crossing_object_rejected();
        test_single_file_container_accepted();
        test_direct_reads_resolve_volumes();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
