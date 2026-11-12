#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
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
#include <numeric>
#include <set>
#include <span>
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
using ninfer::artifact::SliceRange;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;

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
    Json components              = Json::object();
    Json bindings                = Json::object();
    Json uses                    = Json::array();
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
        {"bindings", options.bindings},
        {"uses", options.uses},
        {"components", options.components},
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

// A rejection has to name the layout, the interval and the object's element count, so that the
// failing part of an artifact can be located.
template <typename Function>
void expect_artifact_error_mentions(Function&& function,
                                    std::initializer_list<std::string_view> needles) {
    try {
        function();
    } catch (const ArtifactError& error) {
        const std::string_view message = error.what();
        for (const auto needle : needles) {
            if (message.find(needle) == std::string_view::npos) {
                throw std::runtime_error(std::string("the rejection does not mention ") +
                                         std::string(needle) + ": " + std::string(message));
            }
        }
        return;
    }
    throw std::runtime_error("an invalid element interval was accepted");
}

// The v3 container spells its formats, layouts and encodings in lower case snake case. The table
// accepts every spelling an upstream v3 artifact uses; it is a superset of the narrow 27B adapter,
// not a claim about upstream's complete legal set. A value's canonical name is the spelling of the
// container that introduced it, so the values the v3 container introduced round trip through the
// v3 table.
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

    const std::array<FormatCase, 12> format_cases = {{
        {"bf16", NumericFormat::BF16, "BF16", false},
        {"fp16", NumericFormat::FP16, "FP16", false},
        {"fp32", NumericFormat::FP32, "FP32", false},
        {"int32", NumericFormat::I32, "I32", false},
        {"q4_g64_fp16", NumericFormat::Q4G64_F16S, "Q4G64_F16S", false},
        {"q5_g64_fp16", NumericFormat::Q5G64_F16S, "Q5G64_F16S", false},
        {"q6_g64_fp16", NumericFormat::Q6G64_F16S, "Q6G64_F16S", false},
        {"q8_g32_fp16", NumericFormat::W8G32_F16S, "W8G32_F16S", false},
        {"nvfp4", NumericFormat::NVFP4, "NVFP4", false},
        {"nf4_g16_e4m3fn", NumericFormat::NF4_G16_E4M3S, "nf4_g16_e4m3fn", true},
        {"fp8_e4m3fn_block128x128_bf16", NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
         "fp8_e4m3fn_block128x128_bf16", true},
        {"fp8_e4m3fn_row_bf16", NumericFormat::FP8_E4M3FN_ROW_BF16S, "FP8_E4M3FN_ROW_BF16S", false},
    }};
    const std::array<LayoutCase, 7> layout_cases = {{
        {"contiguous_le_v1", StorageLayout::ContiguousLeV1, "contiguous-le-v1", false},
        {"row_split_k128_v1", StorageLayout::RowSplitK128V1, "row-split-k128-v1", false},
        {"row_scale_v1", StorageLayout::RowScaleV1, "row-scale-v1", false},
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

// The v3 directory carries the component configuration, the parameter binding table and the
// operator use edges. The reader reports all three without a JSON type in its public interface.
void test_v3_directory_metadata() {
    FixtureOptions options;
    options.components = Json{
        {"text",
         Json{{"target", "text"},
              {"resources", Json{{"tokenizer.json", "resource/text/tokenizer.json"},
                                 {"chat_template.jinja", "resource/text/chat_template.jinja"}}},
              {"config",
               Json{{"hidden_size", 2560},
                    {"num_hidden_layers", 48},
                    {"tie_word_embeddings", false},
                    {"rms_norm_eps", 1e-06},
                    {"layer_types", Json::array({"linear_attention", "indexed_attention"})},
                    {"rope_parameters",
                     Json{{"rope_theta", 10000000},
                          {"partial_rotary_factor", 0.25},
                          {"mrope_section", Json::array({11, 11, 10})}}}}}}},
        {"vision", Json{{"target", "text"},
                        {"resources",
                         Json{{"preprocessor_config.json",
                               "resource/vision/preprocessor_config.json"}}},
                        {"config", Json{{"depth", 27}}}}},
    };
    options.bindings = Json{
        {"text/layers/0/attention_hc/down", Json{{"object", "weight/000001"}}},
        {"text/layers/0/attention_hc/up",
         Json{{"parts", Json::array({Json{{"object", "weight/000002"},
                                         {"range", Json::array({0, 1024})}}})}}},
        {"text/unbound", Json{{"kind", "unresolved"}}},
    };
    options.uses = Json::array({
        Json{{"parameter", "text/layers/0/attention_hc/down"},
             {"input", "text/layers/0/attention_hc/normalized_streams"},
             {"activation_policy", "A16Only"}},
        Json{{"parameter", "text/layers/0/moe/experts/gate"},
             {"input", "text/layers/0/moe_input"},
             {"activation_policy", "AllowA4"},
             {"auxiliaries",
              Json{{"activation_input_divisor",
                    Json{{"parts",
                          Json::array({Json{
                                        {"object", "auxiliary/shared/expert_activation_divisors"},
                                        {"range", Json::array({0, 1})}}})}}}}}},
    });

    const auto fixture =
        write_v3_multi_part(canonical_objects(),
                            {marker_payload(kPartPayloadBytes, 0x11),
                             marker_payload(kPartPayloadBytes, 0x22),
                             marker_payload(kPartPayloadBytes, 0x33)},
                            options);
    Reader reader(fixture.entry);
    const auto& directory = reader.v3_directory();

    if (directory.component_config.size() != 2) {
        throw std::runtime_error("the v3 directory component count mismatch");
    }
    const auto text = directory.component_config.find("text");
    if (text == directory.component_config.end()) {
        throw std::runtime_error("the v3 directory does not report the text component");
    }
    const auto& config = text->second;
    if (config.at("hidden_size") != "2560" || config.at("num_hidden_layers") != "48" ||
        config.at("tie_word_embeddings") != "false" || config.at("rms_norm_eps") != "1e-06" ||
        config.at("layer_types") != "linear_attention,indexed_attention" ||
        config.at("rope_parameters.rope_theta") != "10000000" ||
        config.at("rope_parameters.partial_rotary_factor") != "0.25" ||
        config.at("rope_parameters.mrope_section") != "11,11,10") {
        throw std::runtime_error("the v3 directory text configuration mismatch");
    }
    const auto vision = directory.component_config.find("vision");
    if (vision == directory.component_config.end() || vision->second.at("depth") != "27") {
        throw std::runtime_error("the v3 directory vision configuration mismatch");
    }

    if (directory.component_resources.size() != 2 ||
        directory.component_resources.at("text").at("tokenizer.json") !=
            "resource/text/tokenizer.json" ||
        directory.component_resources.at("text").at("chat_template.jinja") !=
            "resource/text/chat_template.jinja" ||
        directory.component_resources.at("vision").at("preprocessor_config.json") !=
            "resource/vision/preprocessor_config.json") {
        throw std::runtime_error("the v3 directory component resource mismatch");
    }

    if (directory.bindings.size() != 1 ||
        directory.bindings.at("text/layers/0/attention_hc/down") != "weight/000001") {
        throw std::runtime_error("the v3 directory binding table mismatch");
    }
    const auto parts = directory.part_bindings.find("text/layers/0/attention_hc/up");
    if (parts == directory.part_bindings.end() || parts->second.size() != 1 ||
        parts->second.front().object != "weight/000002" || parts->second.front().begin != 0 ||
        parts->second.front().end != 1024) {
        throw std::runtime_error("the v3 directory part binding mismatch");
    }
    if (directory.bindings.count("text/unbound") != 0 ||
        directory.part_bindings.count("text/unbound") != 0) {
        throw std::runtime_error("a binding that names no object was reported");
    }

    if (directory.uses.size() != 2) {
        throw std::runtime_error("the v3 directory use count mismatch");
    }
    const auto& use = directory.uses.front();
    if (use.parameter != "text/layers/0/attention_hc/down" ||
        use.input != "text/layers/0/attention_hc/normalized_streams" ||
        use.activation_policy != "A16Only") {
        throw std::runtime_error("the v3 directory use edge mismatch");
    }
    if (!use.auxiliaries.empty()) {
        throw std::runtime_error("a use edge without auxiliaries reported one");
    }
    const auto& moe = directory.uses.back();
    if (moe.parameter != "text/layers/0/moe/experts/gate" ||
        moe.input != "text/layers/0/moe_input" || moe.activation_policy != "AllowA4") {
        throw std::runtime_error("the v3 directory expert use edge mismatch");
    }
    const auto divisor = moe.auxiliaries.find("activation_input_divisor");
    if (divisor == moe.auxiliaries.end() || divisor->second.size() != 1 ||
        divisor->second.front().object != "auxiliary/shared/expert_activation_divisors" ||
        divisor->second.front().begin != 0 || divisor->second.front().end != 1) {
        throw std::runtime_error("the v3 directory use auxiliary mismatch");
    }
}

// A v2 artifact carries no v3 directory, so the reader reports an empty one instead of failing.
void test_v2_directory_is_empty() {
    const Json directory{
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({
                        {{"name", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "BF16"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 0},
                         {"bytes", 12}},
                    })},
    };
    const auto fixture = write_fixture(directory, "v2_directory");
    Reader reader(fixture.path);
    const auto& v3 = reader.v3_directory();
    if (!v3.component_config.empty() || !v3.bindings.empty() || !v3.part_bindings.empty() ||
        !v3.uses.empty()) {
        throw std::runtime_error("a v2 artifact reported v3 directory metadata");
    }
}

// The v3 directory an upstream artifact carries, read from the container header.
Json read_v3_directory(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("failed to open the v3 artifact"); }
    std::array<std::byte, 32> header{};
    input.read(reinterpret_cast<char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
    const std::array<std::uint8_t, 8> magic = {'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    for (std::size_t i = 0; i < magic.size(); ++i) {
        if (std::to_integer<unsigned char>(header[i]) != magic[i]) {
            throw std::runtime_error("the artifact is not a v3 container");
        }
    }
    std::uint64_t json_bytes = 0;
    for (unsigned i = 0; i < 8; ++i) {
        json_bytes |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(header[8 + i]))
                      << (i * 8);
    }
    std::vector<char> text(json_bytes);
    input.read(text.data(), static_cast<std::streamsize>(json_bytes));
    if (!input) { throw std::runtime_error("the v3 directory is truncated"); }
    return Json::parse(text.begin(), text.end());
}

// The shared v3 spelling table must accept every spelling an upstream v3 artifact uses. The 27B
// narrow adapter reads exactly those spellings, so one spelling must not be accepted in one place
// and rejected in the other.
// NINFER_27B_V3_ARTIFACT=/home/cmj/models/ninfer-artifacts/qwen3_8_27b_nvfp4.ninfer
void test_public_table_covers_upstream_v3_artifact() {
    const char* artifact = std::getenv("NINFER_27B_V3_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_27B_V3_ARTIFACT is not set\n";
        return;
    }
    const Json directory = read_v3_directory(artifact);
    if (directory.at("metadata").at("name").get<std::string>() != "qwen3.8-27b") {
        throw std::runtime_error("NINFER_27B_V3_ARTIFACT is not the upstream 27B v3 artifact");
    }
    std::size_t tensors   = 0;
    std::size_t resources = 0;
    for (const auto& object : directory.at("objects")) {
        if (object.at("kind").get<std::string>() == "tensor") {
            ++tensors;
            const auto format = object.at("format").get<std::string>();
            const auto layout = object.at("layout").get<std::string>();
            try {
                ninfer::artifact::parse_v3_format(format);
                ninfer::artifact::parse_v3_layout(layout);
            } catch (const ArtifactError& error) {
                throw std::runtime_error("the shared v3 table rejects " + format + "/" + layout +
                                         ": " + error.what());
            }
        } else {
            ++resources;
            const auto encoding = object.at("encoding").get<std::string>();
            try {
                ninfer::artifact::parse_v3_encoding(encoding);
            } catch (const ArtifactError& error) {
                throw std::runtime_error("the shared v3 table rejects " + encoding + ": " +
                                         error.what());
            }
        }
    }
    if (tensors == 0 || resources == 0) {
        throw std::runtime_error("the upstream v3 artifact has no objects");
    }
    Reader reader(artifact);
    if (reader.identity().model_id != "qwen3.8-27b" || reader.identity().weights_id != "nvfp4" ||
        reader.objects().empty()) {
        throw std::runtime_error("the 27B narrow adapter did not open the upstream v3 artifact");
    }
    const auto& v3 = reader.v3_directory();
    if (!v3.component_config.empty() || !v3.bindings.empty() || !v3.part_bindings.empty() ||
        !v3.uses.empty()) {
        throw std::runtime_error("the 27B narrow adapter reported v3 directory metadata");
    }
    std::cout << "upstream v3 artifact: " << tensors << " tensors, " << resources
              << " resources, every spelling accepted by the shared table\n";
}

// row_paged_g16_4k_v1: NF4 codes with one E4M3FN scale per 16 elements, packed into 4096-byte
// pages, with an FP32 matrix scale after the last page.
void test_nf4_paged_geometry() {
    const std::array<std::uint64_t, 2> shape = {2500012, 160};
    const auto geometry =
        ninfer::artifact::row_paged_nf4_geometry(NumericFormat::NF4_G16_E4M3S, shape);
    if (geometry.row_bytes != 90 || geometry.rows_per_page != 45 || geometry.page_count != 55556 ||
        geometry.matrix_scale_offset != 227557376 || geometry.encoded_bytes != 227557380) {
        throw std::runtime_error("row_paged_g16_4k_v1 geometry is not the upstream geometry");
    }
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> ragged = {16, 24};
            (void)ninfer::artifact::row_paged_nf4_geometry(NumericFormat::NF4_G16_E4M3S, ragged);
        },
        "divisible by 16");
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> shape = {16, 32};
            (void)ninfer::artifact::row_paged_nf4_geometry(NumericFormat::NVFP4, shape);
        },
        "NF4_G16_E4M3S");
}

// block_scale_m128_k128_v1: an E4M3FN code plane plus one BF16 scale per 128x128 block.
void test_block128_geometry() {
    const std::array<std::uint64_t, 2> shape = {2560, 640};
    const auto geometry = ninfer::artifact::block_scale_m128_k128_geometry(
        NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, shape);
    if (geometry.code_plane_bytes != 1638400 || geometry.scale_plane_offset != 1638400 ||
        geometry.scale_plane_bytes != 200 || geometry.encoded_bytes != 1638600) {
        throw std::runtime_error("block_scale_m128_k128_v1 geometry is not the upstream geometry");
    }
    {
        const std::array<std::uint64_t, 3> stack = {512, 1280, 2560};
        const auto stacked = ninfer::artifact::block_scale_m128_k128_geometry(
            NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, stack);
        if (stacked.matrices != 512 || stacked.code_plane_bytes != 3276800 ||
            stacked.scale_plane_offset != 1677721600 || stacked.scale_plane_bytes != 400 ||
            stacked.encoded_bytes != 1677926400) {
            throw std::runtime_error(
                "block_scale_m128_k128_v1 stack geometry is not the upstream geometry");
        }
    }
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> ragged = {256, 192};
            (void)ninfer::artifact::block_scale_m128_k128_geometry(
                NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, ragged);
        },
        "divisible by 128");
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> shape = {256, 128};
            (void)ninfer::artifact::block_scale_m128_k128_geometry(NumericFormat::NVFP4, shape);
        },
        "FP8_E4M3FN_BLOCK128_BF16S");
}

// The paged NF4 rows live in 4096-byte pages, and a row's group scales follow its codes.
void test_nf4_paged_addressing() {
    const std::array<std::uint64_t, 2> shape = {2500012, 160};
    const auto geometry =
        ninfer::artifact::row_paged_nf4_geometry(NumericFormat::NF4_G16_E4M3S, shape);
    if (geometry.code_bytes_per_row != 80 || geometry.scale_bytes_per_row != 10 ||
        geometry.code_bytes != 2500012ULL * 80 || geometry.scale_bytes != 2500012ULL * 10) {
        throw std::runtime_error("row_paged_g16_4k_v1 does not expose its code and scale bytes");
    }
    if (ninfer::artifact::row_paged_nf4_row_offset(geometry, 0) != 0 ||
        ninfer::artifact::row_paged_nf4_row_offset(geometry, 44) != 44 * 90 ||
        ninfer::artifact::row_paged_nf4_row_offset(geometry, 45) != 4096) {
        throw std::runtime_error("row_paged_g16_4k_v1 row addressing is not the upstream one");
    }
    if (ninfer::artifact::row_paged_nf4_scale_offset(geometry, 45, 0) != 4096 + 80 ||
        ninfer::artifact::row_paged_nf4_scale_offset(geometry, 45, 9) != 4096 + 89) {
        throw std::runtime_error("row_paged_g16_4k_v1 scale addressing is not the upstream one");
    }
}

// batched_block_scale_k16_m128x4_v1: expert matrices in the NVFP4 blockscale layout.
void test_batched_blockscale_geometry() {
    {
        const std::array<std::uint64_t, 3> shape = {512, 1280, 2560};
        const auto geometry =
            ninfer::artifact::batched_block_scale_geometry(NumericFormat::NVFP4, shape);
        if (geometry.matrices != 512 || geometry.code_plane_bytes != 1638400 ||
            geometry.scale_plane_bytes != 204800 || geometry.scale_plane_offset != 838860800 ||
            geometry.weight_divisor_offset != 943718400 ||
            geometry.encoded_bytes != 943720448) {
            throw std::runtime_error(
                "batched_block_scale_k16_m128x4_v1 geometry is not the upstream geometry");
        }
    }
    {
        const std::array<std::uint64_t, 3> shape = {512, 2560, 640};
        const auto geometry =
            ninfer::artifact::batched_block_scale_geometry(NumericFormat::NVFP4, shape);
        if (geometry.code_plane_bytes != 819200 || geometry.scale_plane_bytes != 102400 ||
            geometry.scale_plane_offset != 419430400 ||
            geometry.weight_divisor_offset != 471859200 ||
            geometry.encoded_bytes != 471861248) {
            throw std::runtime_error(
                "batched_block_scale_k16_m128x4_v1 geometry is not the upstream geometry");
        }
    }
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 3> shape = {512, 1280, 200};
            (void)ninfer::artifact::batched_block_scale_geometry(NumericFormat::NVFP4, shape);
        },
        "batched_block_scale_k16_m128x4_v1 requires N divisible by 128 and K divisible by 64");
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 3> shape = {512, 192, 2560};
            (void)ninfer::artifact::batched_block_scale_geometry(NumericFormat::NVFP4, shape);
        },
        "batched_block_scale_k16_m128x4_v1 requires N divisible by 128 and K divisible by 64");
}

// The batched layout slices per matrix: a row slice selects whole experts, a column slice
// narrows the K axis of every expert with the single-matrix blockscale rule.
void test_batched_slices() {
    const std::array<std::uint64_t, 3> shape = {512, 1280, 2560};
    const std::array<SliceRange, 1> experts = {SliceRange{0, 256}};
    const auto row_slice = ninfer::artifact::tensor_row_slice(
        StorageLayout::BatchedBlockScaleK16M128x4V1, NumericFormat::NVFP4, shape, experts);
    // one code block, ten 128-row scale tiles and one FP32 divisor per selected expert
    if (row_slice.encoded_bytes != 471860224 || row_slice.copies.size() != 256 * 12) {
        throw std::runtime_error("batched expert row slice is not the single-matrix rule");
    }
    const std::array<SliceRange, 1> halves = {SliceRange{0, 1280}};
    const auto column_slice = ninfer::artifact::tensor_column_slice(
        StorageLayout::BatchedBlockScaleK16M128x4V1, NumericFormat::NVFP4, shape, halves);
    // one code copy per row, ten scale tiles and one divisor per expert
    if (column_slice.encoded_bytes != 471861248 || column_slice.copies.size() != 512 * 1291) {
        throw std::runtime_error("batched expert column slice is not the single-matrix rule");
    }
}

// The batched slice arithmetic must copy the exact bytes of the selected experts and columns.
// The parent payload and the expected shard are both built from the layout's logical
// coordinates, so a wrong plane offset or a wrong stride shows up as a byte mismatch.
using Bytes = std::vector<std::uint8_t>;

std::uint8_t slice_pattern(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    return static_cast<std::uint8_t>((a * 131 + b * 17 + c * 7 + 11) & 0xFFU);
}

Bytes apply_copies(const Bytes& parent, const ninfer::artifact::TensorSlice& slice) {
    Bytes out(static_cast<std::size_t>(slice.encoded_bytes), 0);
    for (const auto& copy : slice.copies) {
        if (copy.source_offset + copy.bytes > parent.size() ||
            copy.dest_offset + copy.bytes > out.size()) {
            throw std::runtime_error("slice copy range escapes its buffer");
        }
        for (std::uint64_t i = 0; i < copy.bytes; ++i) {
            out[static_cast<std::size_t>(copy.dest_offset + i)] =
                parent[static_cast<std::size_t>(copy.source_offset + i)];
        }
    }
    return out;
}

Bytes build_batched_nvfp4(std::uint64_t experts, std::uint64_t rows, std::uint64_t columns,
                          std::uint64_t expert_origin, std::uint64_t column_origin,
                          std::uint64_t group_origin) {
    const std::array<std::uint64_t, 3> shape = {experts, rows, columns};
    const auto geometry =
        ninfer::artifact::batched_block_scale_geometry(NumericFormat::NVFP4, shape);
    Bytes out(static_cast<std::size_t>(geometry.encoded_bytes), 0);
    const std::uint64_t k_tiles = columns / 64;
    for (std::uint64_t expert = 0; expert < experts; ++expert) {
        const std::uint64_t code_base  = expert * geometry.code_plane_bytes;
        const std::uint64_t scale_base = geometry.scale_plane_offset +
                                        expert * geometry.scale_plane_bytes;
        for (std::uint64_t row = 0; row < rows; ++row) {
            for (std::uint64_t byte = 0; byte < columns / 2; ++byte) {
                out[static_cast<std::size_t>(code_base + row * (columns / 2) + byte)] =
                    slice_pattern(expert_origin + expert, row, column_origin / 2 + byte);
            }
            for (std::uint64_t group = 0; group < columns / 16; ++group) {
                const std::uint64_t row_inner = row % 128;
                const std::uint64_t offset = ((row / 128) * k_tiles + group / 4) * 512 +
                                             (row_inner % 32) * 16 + (row_inner / 32) * 4 +
                                             (group % 4);
                out[static_cast<std::size_t>(scale_base + offset)] =
                    slice_pattern(expert_origin + expert, group_origin + group, 200);
            }
        }
        for (std::uint64_t byte = 0; byte < 4; ++byte) {
            out[static_cast<std::size_t>(geometry.weight_divisor_offset + 4 * expert + byte)] =
                slice_pattern(expert_origin + expert, byte, 240);
        }
    }
    return out;
}

// block_scale_m128_k128_v1: one E4M3FN code byte per weight, one BF16 scale word per 128x128
// block, the words addressed by 128-row band and block column.
Bytes build_block128(std::uint64_t matrices, std::uint64_t rows, std::uint64_t columns,
                      std::uint64_t matrix_origin, std::uint64_t row_origin,
                      std::uint64_t column_origin, std::uint64_t block_origin) {
    const std::array<std::uint64_t, 3> shape = {matrices, rows, columns};
    const auto geometry = ninfer::artifact::block_scale_m128_k128_geometry(
        NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, shape);
    Bytes out(static_cast<std::size_t>(geometry.encoded_bytes), 0);
    const std::uint64_t band_bytes = (columns / 128) * 2;
    for (std::uint64_t matrix = 0; matrix < matrices; ++matrix) {
        const std::uint64_t code_base  = matrix * geometry.code_plane_bytes;
        const std::uint64_t scale_base = geometry.scale_plane_offset +
                                        matrix * geometry.scale_plane_bytes;
        for (std::uint64_t row = 0; row < rows; ++row) {
            for (std::uint64_t column = 0; column < columns; ++column) {
                out[static_cast<std::size_t>(code_base + row * columns + column)] =
                    slice_pattern(matrix_origin + matrix, row_origin + row,
                                  column_origin + column);
            }
            for (std::uint64_t block = 0; block < columns / 128; ++block) {
                const std::uint64_t at = scale_base + (row / 128) * band_bytes + block * 2;
                out[static_cast<std::size_t>(at)] =
                    slice_pattern(matrix_origin + matrix, block_origin + block, 200);
                out[static_cast<std::size_t>(at + 1)] =
                    slice_pattern(matrix_origin + matrix, block_origin + block, 201);
            }
        }
    }
    return out;
}

void test_batched_slice_bytes() {
    constexpr std::uint64_t kExperts = 4;
    constexpr std::uint64_t kRows    = 256;
    constexpr std::uint64_t kColumns = 256;
    const std::array<std::uint64_t, 3> shape = {kExperts, kRows, kColumns};
    const Bytes parent = build_batched_nvfp4(kExperts, kRows, kColumns, 0, 0, 0);

    // Expert interval [1, 3): the shard holds experts 1 and 2 with their own coordinates.
    const std::array<SliceRange, 1> experts = {SliceRange{1, 2}};
    const auto row_slice = ninfer::artifact::tensor_row_slice(
        StorageLayout::BatchedBlockScaleK16M128x4V1, NumericFormat::NVFP4, shape, experts);
    if (apply_copies(parent, row_slice) != build_batched_nvfp4(2, kRows, kColumns, 1, 0, 0)) {
        throw std::runtime_error("batched expert row slice did not copy the expert bytes");
    }

    // K interval [64, 192): the shard holds the parent's columns 64..191 of every expert.
    const std::array<SliceRange, 1> columns = {SliceRange{64, 128}};
    const auto column_slice = ninfer::artifact::tensor_column_slice(
        StorageLayout::BatchedBlockScaleK16M128x4V1, NumericFormat::NVFP4, shape, columns);
    if (apply_copies(parent, column_slice) !=
        build_batched_nvfp4(kExperts, kRows, 128, 0, 64, 4)) {
        throw std::runtime_error("batched expert column slice did not copy the column bytes");
    }
}

// The block-FP8 row axis is the 128-row band of one matrix, the row axis of a stack is its
// matrix axis, and the column axis is the K axis.
void test_block128_slice_bytes() {
    constexpr std::uint64_t kRows    = 256;
    constexpr std::uint64_t kColumns = 256;

    {
        const std::array<std::uint64_t, 2> shape = {kRows, kColumns};
        const Bytes parent = build_block128(1, kRows, kColumns, 0, 0, 0, 0);
        const std::array<SliceRange, 1> bands = {SliceRange{128, 128}};
        const auto row_slice = ninfer::artifact::tensor_row_slice(
            StorageLayout::BlockScaleM128K128V1, NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
            shape, bands);
        if (apply_copies(parent, row_slice) != build_block128(1, 128, kColumns, 0, 128, 0, 0)) {
            throw std::runtime_error("block128 row band slice did not copy the band bytes");
        }
        const std::array<SliceRange, 1> columns = {SliceRange{128, 128}};
        const auto column_slice = ninfer::artifact::tensor_column_slice(
            StorageLayout::BlockScaleM128K128V1, NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
            shape, columns);
        if (apply_copies(parent, column_slice) !=
            build_block128(1, kRows, 128, 0, 0, 128, 1)) {
            throw std::runtime_error("block128 column slice did not copy the column bytes");
        }
    }

    {
        const std::array<std::uint64_t, 3> shape = {2, kRows, kColumns};
        const Bytes parent = build_block128(2, kRows, kColumns, 0, 0, 0, 0);
        const std::array<SliceRange, 1> matrices = {SliceRange{1, 1}};
        const auto row_slice = ninfer::artifact::tensor_row_slice(
            StorageLayout::BlockScaleM128K128V1, NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
            shape, matrices);
        if (apply_copies(parent, row_slice) != build_block128(1, kRows, kColumns, 1, 0, 0, 0)) {
            throw std::runtime_error("block128 matrix slice did not copy the matrix bytes");
        }
        const std::array<SliceRange, 1> columns = {SliceRange{128, 128}};
        const auto column_slice = ninfer::artifact::tensor_column_slice(
            StorageLayout::BlockScaleM128K128V1, NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
            shape, columns);
        if (apply_copies(parent, column_slice) !=
            build_block128(2, kRows, 128, 0, 0, 128, 1)) {
            throw std::runtime_error("block128 stack column slice did not copy the column bytes");
        }
    }
}

// The paged expert layout is replicated whole: it has no slice form yet.
void test_paged_layout_has_no_slice() {
    const std::array<std::uint64_t, 2> shape = {64, 160};
    const std::array<SliceRange, 1> rows = {SliceRange{0, 32}};
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::tensor_row_slice(StorageLayout::RowPagedG16x4KV1,
                                                      NumericFormat::NF4_G16_E4M3S, shape, rows);
        },
        "row_paged_g16_4k_v1");
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::tensor_column_slice(StorageLayout::RowPagedG16x4KV1,
                                                         NumericFormat::NF4_G16_E4M3S, shape, rows);
        },
        "row_paged_g16_4k_v1");
}

// The v2 column slice contract keeps its own rejection wording.
void test_column_slice_rank_rejection_unchanged() {
    const std::array<std::uint64_t, 1> shape = {8};
    const std::array<SliceRange, 1> columns = {SliceRange{0, 4}};
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::tensor_column_slice(StorageLayout::ContiguousLeV1,
                                                         NumericFormat::BF16, shape, columns);
        },
        "a column slice requires a rank-two shape");
}

// The Flash-Next artifact must open through the public Reader, and every tensor's declared byte
// count must be the geometry of its own layout and format.
// NINFER_FLASH_NEXT_V3_ARTIFACT=
//   /home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer
void test_flash_next_artifact_geometries() {
    const char* artifact = std::getenv("NINFER_FLASH_NEXT_V3_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_FLASH_NEXT_V3_ARTIFACT is not set\n";
        return;
    }
    Reader reader(artifact);
    std::size_t tensors           = 0;
    std::size_t paged             = 0;
    std::size_t block128          = 0;
    std::size_t batched_blockscale = 0;
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        ++tensors;
        const auto expected =
            ninfer::artifact::tensor_encoded_size(tensor->layout, tensor->format, tensor->shape);
        if (tensor->bytes != expected) {
            throw std::runtime_error("artifact tensor " + tensor->name + " stores " +
                                     std::to_string(tensor->bytes) + " bytes; its geometry needs " +
                                     std::to_string(expected));
        }
        switch (tensor->layout) {
        case StorageLayout::RowPagedG16x4KV1: ++paged; break;
        case StorageLayout::BlockScaleM128K128V1: ++block128; break;
        case StorageLayout::BatchedBlockScaleK16M128x4V1: ++batched_blockscale; break;
        default: break;
        }
    }
    if (tensors != 1312 || paged != 128 || block128 != 2 || batched_blockscale != 96) {
        throw std::runtime_error("the Flash-Next artifact does not carry the expected objects");
    }
    std::cout << "flash-next artifact: " << tensors << " tensors, every byte count matches its "
              << "geometry (" << paged << " paged, " << batched_blockscale << " batched, "
              << block128 << " block128)\n";
}

// The upstream Flash-Next directory carries the component configuration, the parameter binding
// table and the operator use edges. The reader reports the census the artifact ships.
// NINFER_FLASH_NEXT_V3_ARTIFACT=
//   /home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer
void test_flash_next_directory_metadata() {
    const char* artifact = std::getenv("NINFER_FLASH_NEXT_V3_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_FLASH_NEXT_V3_ARTIFACT is not set\n";
        return;
    }
    Reader reader(artifact);
    const auto& directory = reader.v3_directory();

    if (directory.component_config.size() != 3 ||
        directory.component_config.count("text") == 0 ||
        directory.component_config.count("mtp") == 0 ||
        directory.component_config.count("vision") == 0) {
        throw std::runtime_error("the Flash-Next directory component census mismatch");
    }
    const auto& config = directory.component_config.at("text");
    if (config.at("hidden_size") != "2560" || config.at("num_hidden_layers") != "48" ||
        config.at("hc_count") != "4" || config.at("hc_lowrank") != "320" ||
        config.at("indexer_budget") != "2048" || config.at("num_experts") != "512" ||
        config.at("num_experts_per_tok") != "10" || config.at("moe_intermediate_size") != "640" ||
        config.at("ple_shard_shape") != "2500012,160" ||
        config.at("rope_parameters.mrope_section") != "11,11,10") {
        throw std::runtime_error("the Flash-Next text configuration mismatch");
    }
    if (directory.component_config.at("vision").at("depth") != "27" ||
        directory.component_config.at("mtp").at("num_hidden_layers") != "1") {
        throw std::runtime_error("the Flash-Next vision or mtp configuration mismatch");
    }

    const auto& text_resources = directory.component_resources.at("text");
    if (text_resources.size() != 5 ||
        text_resources.at("tokenizer.json") != "resource/text/tokenizer.json" ||
        text_resources.at("tokenizer_config.json") != "resource/text/tokenizer_config.json" ||
        text_resources.at("chat_template.jinja") != "resource/text/chat_template.jinja" ||
        text_resources.at("generation_config.json") != "resource/text/generation_config.json" ||
        text_resources.at("ple_hash_controls") != "resource/text/ple_hash_controls") {
        throw std::runtime_error("the Flash-Next text resource census mismatch");
    }
    const auto& vision_resources = directory.component_resources.at("vision");
    if (vision_resources.size() != 2 ||
        vision_resources.at("preprocessor_config.json") !=
            "resource/vision/preprocessor_config.json" ||
        vision_resources.at("video_preprocessor_config.json") !=
            "resource/vision/video_preprocessor_config.json") {
        throw std::runtime_error("the Flash-Next vision resource census mismatch");
    }

    std::size_t with_auxiliaries = 0;
    std::set<std::uint64_t> covered;
    for (const auto& use : directory.uses) {
        if (use.auxiliaries.empty()) { continue; }
        ++with_auxiliaries;
        if (use.auxiliaries.size() != 1) {
            throw std::runtime_error("the Flash-Next use auxiliary role count mismatch");
        }
        const auto divisor = use.auxiliaries.find("activation_input_divisor");
        if (divisor == use.auxiliaries.end() || divisor->second.size() != 1) {
            throw std::runtime_error("the Flash-Next use auxiliary divisor mismatch");
        }
        const auto& part = divisor->second.front();
        if (part.object != "auxiliary/shared/expert_activation_divisors" ||
            part.end <= part.begin) {
            throw std::runtime_error("the Flash-Next use auxiliary part mismatch");
        }
        for (auto element = part.begin; element < part.end; ++element) { covered.insert(element); }
    }
    if (with_auxiliaries != 144) {
        throw std::runtime_error("the Flash-Next use auxiliary census mismatch");
    }
    if (covered.size() != 96 || *covered.begin() != 0 || *covered.rbegin() != 95) {
        throw std::runtime_error("the Flash-Next use auxiliary coverage mismatch");
    }

    if (directory.bindings.size() + directory.part_bindings.size() != 1786) {
        throw std::runtime_error("the Flash-Next binding census mismatch");
    }
    if (directory.uses.size() != 1205) {
        throw std::runtime_error("the Flash-Next use census mismatch");
    }
    std::cout << "flash-next directory: " << directory.component_config.size() << " components, "
              << directory.bindings.size() << " object bindings, "
              << directory.part_bindings.size() << " part bindings, " << directory.uses.size()
              << " uses (" << with_auxiliaries << " with auxiliaries), "
              << directory.component_resources.size() << " component resource maps\n";
}

// The logical axes of a layout, as the test understands them: a stack of matrices, and inside one
// matrix the rows and the columns.
struct ViewAxes {
    std::uint64_t matrices = 1;
    std::uint64_t rows     = 1;
    std::uint64_t columns  = 1;
};

// The rectangle a view is expected to report.
struct ViewExpectation {
    ViewAxes axes;
    std::uint64_t matrix_begin = 0;
    std::uint64_t matrix_count = 0;
    std::uint64_t matrix_row_begin = 0;
    std::uint64_t matrix_row_count = 0;
    std::uint64_t column_begin = 0;
    std::uint64_t column_count = 0;
};

// The element indices a view covers, recomputed from the element order: matrix-major, row-major
// inside one matrix. The axes come from the caller, so the recomputation does not borrow the
// implementation's own idea of them.
std::vector<std::uint64_t> view_element_indices(const ninfer::artifact::ElementIntervalView& view,
                                                 const ViewAxes& axes) {
    const std::uint64_t per_matrix = axes.rows * axes.columns;
    std::vector<std::uint64_t> indices;
    indices.reserve(static_cast<std::size_t>(view.element_count));
    for (std::uint64_t matrix = view.matrix_begin;
         matrix < view.matrix_begin + view.matrix_count; ++matrix) {
        for (std::uint64_t row = view.matrix_row_begin;
             row < view.matrix_row_begin + view.matrix_row_count; ++row) {
            for (std::uint64_t column = view.column_begin;
                 column < view.column_begin + view.column_count; ++column) {
                indices.push_back(matrix * per_matrix + row * axes.columns + column);
            }
        }
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

// A view must report the requested interval and the expected rectangle, and the elements of that
// rectangle must be exactly the elements of the interval.
void expect_element_view(StorageLayout layout, NumericFormat format,
                         std::span<const std::uint64_t> shape, std::uint64_t begin,
                         std::uint64_t count, const ViewExpectation& expected) {
    const auto view = ninfer::artifact::tensor_element_interval_view(layout, format, shape, begin,
                                                                      count);
    if (view.layout != layout || view.linear_begin != begin || view.element_count != count) {
        throw std::runtime_error(
            "the element interval view does not report the requested interval");
    }
    if (view.matrix_begin != expected.matrix_begin || view.matrix_count != expected.matrix_count ||
        view.matrix_row_begin != expected.matrix_row_begin ||
        view.matrix_row_count != expected.matrix_row_count ||
        view.column_begin != expected.column_begin ||
        view.column_count != expected.column_count) {
        throw std::runtime_error(
            "the element interval view does not report the expected rectangle");
    }
    const auto indices = view_element_indices(view, expected.axes);
    std::vector<std::uint64_t> wanted(count);
    std::iota(wanted.begin(), wanted.end(), begin);
    if (indices != wanted) {
        throw std::runtime_error(
            "the element interval view does not describe the requested elements");
    }
}

// contiguous-le-v1 counts its elements in one linear order: matrix-major, row-major inside one
// matrix.
void test_element_view_contiguous() {
    const std::array<std::uint64_t, 2> shape = {4, 6};
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 0, 24,
                        ViewExpectation{{1, 4, 6}, 0, 1, 0, 4, 0, 6});
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 23, 1,
                        ViewExpectation{{1, 4, 6}, 0, 1, 3, 1, 5, 1});

    // A run of columns inside one row is a rectangle, whichever end of the row it starts from.
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 6, 3,
                        ViewExpectation{{1, 4, 6}, 0, 1, 1, 1, 0, 3});
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 7, 5,
                        ViewExpectation{{1, 4, 6}, 0, 1, 1, 1, 1, 5});
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 8, 4,
                        ViewExpectation{{1, 4, 6}, 0, 1, 1, 1, 2, 4});

    // A run of whole rows is a rectangle as well.
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 6, 12,
                        ViewExpectation{{1, 4, 6}, 0, 1, 1, 2, 0, 6});
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 0, 18,
                        ViewExpectation{{1, 4, 6}, 0, 1, 0, 3, 0, 6});

    const std::array<std::uint64_t, 1> vector = {5};
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, vector, 2, 3,
                        ViewExpectation{{1, 5, 1}, 0, 1, 2, 3, 0, 1});

    // A rank-three contiguous tensor is a stack of matrices, and a rank-four one keeps its first
    // dimension as the matrix axis and the product of the dimensions between the first and the
    // last as the matrix-internal row axis.
    const std::array<std::uint64_t, 3> stack = {2, 3, 4};
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, stack, 0, 24,
                        ViewExpectation{{2, 3, 4}, 0, 2, 0, 3, 0, 4});
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, stack, 12, 12,
                        ViewExpectation{{2, 3, 4}, 1, 1, 0, 3, 0, 4});
    const std::array<std::uint64_t, 4> deep = {2, 3, 4, 5};
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, deep, 0, 120,
                        ViewExpectation{{2, 12, 5}, 0, 2, 0, 12, 0, 5});
    expect_element_view(StorageLayout::ContiguousLeV1, NumericFormat::BF16, deep, 70, 2,
                        ViewExpectation{{2, 12, 5}, 1, 1, 2, 1, 0, 2});

    // An interval that crosses a row boundary is not a rectangle of the logical axes.
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> shape = {4, 6};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 7, 6);
        },
        "rectangle");
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> shape = {4, 6};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 0, 20);
        },
        "rectangle");
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> shape = {4, 6};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 0, 25);
        },
        "last element");
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> shape = {4, 6};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 0, 0);
        },
        "empty");

    // A rejection names the layout, the interval and the object's element count.
    expect_artifact_error_mentions(
        [] {
            const std::array<std::uint64_t, 2> shape = {4, 6};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 7, 6);
        },
        {"contiguous-le-v1", "begin 7", "count 6", "object elements 24"});
    expect_artifact_error_mentions(
        [] {
            const std::array<std::uint64_t, 2> shape = {4, 6};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, 0, 25);
        },
        {"contiguous-le-v1", "begin 0", "count 25", "object elements 24"});
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> shape = {4, 6};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::ContiguousLeV1, NumericFormat::NVFP4, shape, 0, 24);
        },
        "contiguous-le-v1");
}

// row-split-k128-v1 counts its elements row-major over the logical columns: the padded K of its
// physical groups is not part of the logical order.
void test_element_view_row_split() {
    const std::array<std::uint64_t, 2> shape = {4, 256};
    expect_element_view(StorageLayout::RowSplitK128V1, NumericFormat::W8G32_F16S, shape, 0, 512,
                        ViewExpectation{{1, 4, 256}, 0, 1, 0, 2, 0, 256});
    expect_element_view(StorageLayout::RowSplitK128V1, NumericFormat::W8G32_F16S, shape, 512, 256,
                        ViewExpectation{{1, 4, 256}, 0, 1, 2, 1, 0, 256});
    expect_element_view(StorageLayout::RowSplitK128V1, NumericFormat::W8G32_F16S, shape, 100, 6,
                        ViewExpectation{{1, 4, 256}, 0, 1, 0, 1, 100, 6});

    // The logical order stops at the logical K: the pad groups of the physical layout are not
    // elements of the tensor's logical order.
    const std::array<std::uint64_t, 2> padded = {4, 100};
    expect_element_view(StorageLayout::RowSplitK128V1, NumericFormat::W8G32_F16S, padded, 0, 400,
                        ViewExpectation{{1, 4, 100}, 0, 1, 0, 4, 0, 100});
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> padded = {4, 100};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::RowSplitK128V1, NumericFormat::W8G32_F16S, padded, 0, 401);
        },
        "last element");
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> shape = {4, 256};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::RowSplitK128V1, NumericFormat::BF16, shape, 0, 1024);
        },
        "grouped quantized format");
}

// row_paged_g16_4k_v1 counts its elements row-major over the rows of the table.
void test_element_view_row_paged() {
    const std::array<std::uint64_t, 2> shape = {10, 160};
    expect_element_view(StorageLayout::RowPagedG16x4KV1, NumericFormat::NF4_G16_E4M3S, shape, 0,
                        320, ViewExpectation{{1, 10, 160}, 0, 1, 0, 2, 0, 160});
    expect_element_view(StorageLayout::RowPagedG16x4KV1, NumericFormat::NF4_G16_E4M3S, shape, 160,
                        480, ViewExpectation{{1, 10, 160}, 0, 1, 1, 3, 0, 160});
    expect_element_view(StorageLayout::RowPagedG16x4KV1, NumericFormat::NF4_G16_E4M3S, shape, 1590,
                        10, ViewExpectation{{1, 10, 160}, 0, 1, 9, 1, 150, 10});
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> shape = {10, 160};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::RowPagedG16x4KV1, NumericFormat::NF4_G16_E4M3S, shape, 0, 1601);
        },
        "last element");
}

// batched_block_scale_k16_m128x4_v1 counts its elements matrix-major, row-major inside one
// matrix: the expert gate is a row range of one expert matrix, not a range of the expert axis.
void test_element_view_batched_block_scale() {
    const std::array<std::uint64_t, 3> shape = {4, 128, 256};
    const std::uint64_t matrix_elements = 128 * 256;
    expect_element_view(StorageLayout::BatchedBlockScaleK16M128x4V1, NumericFormat::NVFP4, shape,
                        0, 4 * matrix_elements,
                        ViewExpectation{{4, 128, 256}, 0, 4, 0, 128, 0, 256});
    expect_element_view(StorageLayout::BatchedBlockScaleK16M128x4V1, NumericFormat::NVFP4, shape,
                        matrix_elements, 64 * 256,
                        ViewExpectation{{4, 128, 256}, 1, 1, 0, 64, 0, 256});
    expect_element_view(StorageLayout::BatchedBlockScaleK16M128x4V1, NumericFormat::NVFP4, shape,
                        2 * matrix_elements + 64 * 256, 64 * 256,
                        ViewExpectation{{4, 128, 256}, 2, 1, 64, 64, 0, 256});
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 3> shape = {4, 128, 256};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::BatchedBlockScaleK16M128x4V1, NumericFormat::NVFP4, shape,
                128 * 256 + 100, 200);
        },
        "rectangle");
}

// block_scale_m128_k128_v1 counts its elements matrix-major, row-major inside one matrix, for a
// single matrix as well as for a stack of matrices.
void test_element_view_block_scale_m128_k128() {
    const std::array<std::uint64_t, 2> single = {256, 256};
    expect_element_view(StorageLayout::BlockScaleM128K128V1,
                        NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, single, 0, 256 * 256,
                        ViewExpectation{{1, 256, 256}, 0, 1, 0, 256, 0, 256});
    expect_element_view(StorageLayout::BlockScaleM128K128V1,
                        NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, single, 128 * 256, 128 * 256,
                        ViewExpectation{{1, 256, 256}, 0, 1, 128, 128, 0, 256});

    const std::array<std::uint64_t, 3> stack = {4, 256, 256};
    expect_element_view(StorageLayout::BlockScaleM128K128V1,
                        NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, stack, 0, 4 * 256 * 256,
                        ViewExpectation{{4, 256, 256}, 0, 4, 0, 256, 0, 256});
    expect_element_view(StorageLayout::BlockScaleM128K128V1,
                        NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, stack, 256 * 256, 128 * 256,
                        ViewExpectation{{4, 256, 256}, 1, 1, 0, 128, 0, 256});
}

// The remaining layouts count their elements the same way: the single-matrix NVFP4 blockscale,
// the row-scaled FP8, and the GGML blocks.
void test_element_view_remaining_layouts() {
    const std::array<std::uint64_t, 2> nvfp4 = {128, 64};
    expect_element_view(StorageLayout::BlockScaleK16M128x4V1, NumericFormat::NVFP4, nvfp4, 0,
                        128 * 64, ViewExpectation{{1, 128, 64}, 0, 1, 0, 128, 0, 64});
    expect_element_view(StorageLayout::BlockScaleK16M128x4V1, NumericFormat::NVFP4, nvfp4,
                        64 * 64, 64 * 64,
                        ViewExpectation{{1, 128, 64}, 0, 1, 64, 64, 0, 64});

    const std::array<std::uint64_t, 2> row_scale = {4, 8};
    expect_element_view(StorageLayout::RowScaleV1, NumericFormat::FP8_E4M3FN_ROW_BF16S, row_scale,
                        8, 16, ViewExpectation{{1, 4, 8}, 0, 1, 1, 2, 0, 8});

    const std::array<std::uint64_t, 2> ggml = {2, 256};
    expect_element_view(StorageLayout::GgmlK256V1, NumericFormat::GGML_K, ggml, 256, 256,
                        ViewExpectation{{1, 2, 256}, 0, 1, 1, 1, 0, 256});
    expect_artifact_error(
        [] {
            const std::array<std::uint64_t, 2> ggml = {2, 128};
            (void)ninfer::artifact::tensor_element_interval_view(
                StorageLayout::GgmlK256V1, NumericFormat::GGML_K, ggml, 0, 256);
        },
        "divisible by 256");
}

std::vector<PlacedObject> new_format_objects(std::uint64_t& fp8_offset,
                                             std::uint64_t& payload_bytes) {
    const std::array<std::uint64_t, 2> nf4_shape = {16, 160};
    const std::array<std::uint64_t, 2> fp8_shape = {128, 128};
    const auto nf4_bytes = ninfer::artifact::row_paged_nf4_geometry(NumericFormat::NF4_G16_E4M3S,
                                                                     nf4_shape)
                               .encoded_bytes;
    const auto fp8_bytes = ninfer::artifact::block_scale_m128_k128_geometry(
                               NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, fp8_shape)
                               .encoded_bytes;
    fp8_offset    = align_up(nf4_bytes, 256);
    payload_bytes = fp8_offset + fp8_bytes;
    return {
        {Json{{"id", "nf4"},
              {"kind", "tensor"},
              {"format", "nf4_g16_e4m3fn"},
              {"layout", "row_paged_g16_4k_v1"},
              {"shape", Json::array({16, 160})},
              {"bytes", nf4_bytes}},
         0, 0},
        {Json{{"id", "fp8_block"},
              {"kind", "tensor"},
              {"format", "fp8_e4m3fn_block128x128_bf16"},
              {"layout", "block_scale_m128_k128_v1"},
              {"shape", Json::array({128, 128})},
              {"bytes", fp8_bytes}},
         0, fp8_offset},
    };
}

void test_validate_only_accepts_new_quantized_formats() {
    std::uint64_t fp8_offset    = 0;
    std::uint64_t payload_bytes = 0;
    const auto objects          = new_format_objects(fp8_offset, payload_bytes);
    const auto fixture = write_v3_multi_part(objects, {marker_payload(payload_bytes, 0x11)});
    Reader reader(fixture.entry);
    ninfer::artifact::Binder binder(reader);

    const auto nf4 = ninfer::artifact::bind_tensor(
        binder, "nf4", NumericFormat::NF4_G16_E4M3S, {16, 160},
        ninfer::artifact::TensorPlacement::ValidateOnly);
    const auto fp8_block = ninfer::artifact::bind_tensor(
        binder, "fp8_block", NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, {128, 128},
        ninfer::artifact::TensorPlacement::ValidateOnly);
    const auto plan = binder.finish();

    const auto* nf4_tensor = std::get_if<TensorDescriptor>(&binder.descriptor(nf4));
    const auto* fp8_tensor = std::get_if<TensorDescriptor>(&binder.descriptor(fp8_block));
    if (nf4_tensor == nullptr || nf4_tensor->format != NumericFormat::NF4_G16_E4M3S ||
        nf4_tensor->layout != StorageLayout::RowPagedG16x4KV1 || fp8_tensor == nullptr ||
        fp8_tensor->format != NumericFormat::FP8_E4M3FN_BLOCK128_BF16S ||
        fp8_tensor->layout != StorageLayout::BlockScaleM128K128V1) {
        throw std::runtime_error("the binder did not resolve the new quantized formats");
    }
    if (plan.object_count != 2 || !plan.device_objects.empty() || !plan.host_objects.empty() ||
        plan.device_capacity_bytes[0] != 0 || plan.host_capacity_bytes[0] != 0) {
        throw std::runtime_error("validate-only placed bytes for a new quantized format");
    }
}

int test_new_quantized_formats_reject_decode() {
    std::uint64_t fp8_offset    = 0;
    std::uint64_t payload_bytes = 0;
    const auto objects          = new_format_objects(fp8_offset, payload_bytes);
    const auto fixture = write_v3_multi_part(objects, {marker_payload(payload_bytes, 0x22)});
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
    const auto nf4 = ninfer::artifact::bind_tensor(
        binder, "nf4", NumericFormat::NF4_G16_E4M3S, {16, 160},
        ninfer::artifact::TensorPlacement::Device);
    const auto fp8_block = ninfer::artifact::bind_tensor(
        binder, "fp8_block", NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, {128, 128},
        ninfer::artifact::TensorPlacement::Device);
    const auto plan = binder.finish();

    ninfer::DeviceContext device(0);
    auto materialized = ninfer::artifact::materialize(reader, plan, device);

    constexpr std::string_view needle = "decode lands in the qwen3.8-flash-next package";
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::materialized_weight(
                materialized, nf4, NumericFormat::NF4_G16_E4M3S, 16, 160);
        },
        needle);
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::materialized_weight(
                materialized, fp8_block, NumericFormat::FP8_E4M3FN_BLOCK128_BF16S, 128, 128);
        },
        needle);
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::materialized_tensor(
                materialized, nf4, NumericFormat::NF4_G16_E4M3S, {16, 160});
        },
        needle);
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::staged_weight_view(NumericFormat::NF4_G16_E4M3S, nullptr, 16,
                                                       160);
        },
        needle);
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::staged_tensor_view(ninfer::DType::NF4_G16_E4M3S, nullptr,
                                                       {16, 160});
        },
        needle);
    return 0;
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
        test_v3_directory_metadata();
        test_v2_directory_is_empty();
        test_public_table_covers_upstream_v3_artifact();
        test_nf4_paged_geometry();
        test_nf4_paged_addressing();
        test_block128_geometry();
        test_batched_blockscale_geometry();
        test_batched_slices();
        test_batched_slice_bytes();
        test_block128_slice_bytes();
        test_paged_layout_has_no_slice();
        test_column_slice_rank_rejection_unchanged();
        test_element_view_contiguous();
        test_element_view_row_split();
        test_element_view_row_paged();
        test_element_view_batched_block_scale();
        test_element_view_block_scale_m128_k128();
        test_element_view_remaining_layouts();
        test_flash_next_artifact_geometries();
        test_flash_next_directory_metadata();
        test_validate_only_accepts_new_quantized_formats();
        const int decode_status = test_new_quantized_formats_reject_decode();
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
        return decode_status;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
