#include "ninfer/ops/ple.h"

#include "artifact/reader.h"
#include "ops/op_tester.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"

#include <nlohmann/json.hpp>

#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

using ninfer::artifact::NumericFormat;
using ninfer::artifact::Reader;
using ninfer::artifact::RowPagedNf4Geometry;
using ninfer::artifact::object_name;
using ninfer::artifact::object_offset;
using ninfer::artifact::row_paged_nf4_geometry;
using ninfer::artifact::row_paged_nf4_row_offset;
using ninfer::ops::PleHashControls;
using ninfer::ops::PleNf4Table;
using ninfer::ops::PleRowAddress;

namespace {

constexpr double kBf16UlpRelative = 1.0 / 128.0;
constexpr double kGateFloor = 1.0e-6;

const ReductionCriterion kNormCriterion{1.85e-3, 1.0e-5, 3.4e-3};

void fail(const std::string& message) { throw std::runtime_error(message); }

std::uint16_t double_to_bf16(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint64_t sign = (bits >> 63) << 15;
    const std::uint64_t magnitude = bits & 0x7fffffffffffffffULL;
    if (magnitude == 0) return static_cast<std::uint16_t>(sign);
    if (magnitude >= 0x7ff0000000000000ULL) {
        return static_cast<std::uint16_t>(sign | 0x7f80U | ((magnitude & 0xfffffffffffffULL) ? 0x20U : 0));
    }
    int exponent = static_cast<int>((magnitude >> 52) & 0x7ffULL) - 1023 + 127;
    std::uint64_t mantissa = magnitude & 0xfffffffffffffULL;
    if (exponent >= 255) return static_cast<std::uint16_t>(sign | 0x7f80U);
    if (exponent <= 0) {
        if (exponent < -7) return static_cast<std::uint16_t>(sign);
        mantissa |= 1ULL << 52;
        const int shift = 45 - exponent;
        const std::uint64_t half = 1ULL << (shift - 1);
        const std::uint64_t base = mantissa >> shift;
        const std::uint64_t rest = mantissa & ((1ULL << shift) - 1);
        const std::uint64_t rounded = base + ((rest > half || (rest == half && (base & 1))) ? 1 : 0);
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint64_t half = 1ULL << 44;
    const std::uint64_t base = mantissa >> 45;
    const std::uint64_t rest = mantissa & ((1ULL << 45) - 1);
    std::uint64_t rounded = base + ((rest > half || (rest == half && (base & 1))) ? 1 : 0);
    if (rounded == 128) {
        rounded = 0;
        ++exponent;
        if (exponent >= 255) return static_cast<std::uint16_t>(sign | 0x7f80U);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint64_t>(exponent) << 7) | rounded);
}

double bf16_bits_to_double(std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); }

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}


constexpr double kE2M1Magnitude[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

double oracle_e2m1(std::uint8_t code) {
    const double magnitude = kE2M1Magnitude[code & 7U];
    const bool negative = (code & 8U) != 0;
    if (magnitude == 0.0) return negative ? -0.0 : 0.0;
    return negative ? -magnitude : magnitude;
}

double oracle_e4m3(std::uint8_t byte) {
    const bool negative = (byte & 0x80U) != 0;
    const int exponent = (byte >> 3) & 0xF;
    const int mantissa = byte & 0x7;
    double magnitude;
    if (exponent == 0) magnitude = std::ldexp(static_cast<double>(mantissa), -9);
    else magnitude = std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0, exponent - 7);
    return negative ? -magnitude : magnitude;
}

std::vector<std::uint16_t> decode_row_oracle(const std::vector<std::byte>& row,
                                             const RowPagedNf4Geometry& geometry, double divisor) {
    std::vector<std::uint16_t> out(geometry.columns);
    for (std::uint64_t column = 0; column < geometry.columns; ++column) {
        const std::uint8_t byte = static_cast<std::uint8_t>(row[column / 2]);
        const std::uint8_t code = (column % 2 == 0) ? (byte & 0xFU) : (byte >> 4U);
        const std::uint8_t scale_byte =
            static_cast<std::uint8_t>(row[geometry.code_bytes_per_row + column / geometry.group_size]);
        const double product = oracle_e2m1(code) * oracle_e4m3(scale_byte);
        out[column] = double_to_bf16(product / divisor);
    }
    return out;
}


const std::filesystem::path& artifact_path() {
    static const std::filesystem::path path = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        if (value == nullptr || *value == '\0') fail("NINFER_FLASH_NEXT_ARTIFACT is not set");
        return std::filesystem::path(value);
    }();
    return path;
}

std::vector<std::byte> read_object_range(const Reader& reader, const std::string& object,
                                         std::uint64_t offset, std::uint64_t bytes) {
    const auto* descriptor = reader.find(object);
    if (descriptor == nullptr) fail("the artifact does not carry " + object);
    std::vector<std::byte> out(bytes);
    const auto read = reader.read_direct(reader.payload_offset() + object_offset(*descriptor) + offset,
                                         out);
    if (read != bytes) fail("short read for " + object);
    return out;
}

std::array<std::string, ninfer::ops::kPleShards> report_shard_objects() {
    const auto path = artifact_path().parent_path() / (artifact_path().filename().string() + ".conversion.json");
    std::ifstream in(path);
    if (!in) fail("cannot open the conversion report " + path.string());
    const nlohmann::json report = nlohmann::json::parse(in);

    std::array<std::string, ninfer::ops::kPleShards> objects;
    std::array<bool, ninfer::ops::kPleShards> seen{};
    for (const auto& method : report.at("methods")) {
        if (method.at("format").get<std::string>() != "nf4_g16_e4m3fn") continue;
        const std::string source = method.at("sources")[0].get<std::string>();
        const auto start = source.find("shard_");
        if (start == std::string::npos) fail("report source is not an n-gram shard: " + source);
        std::size_t end = start + 6;
        while (end < source.size() && std::isdigit(static_cast<unsigned char>(source[end]))) ++end;
        const auto index = std::stoul(source.substr(start + 6, end - start - 6));
        if (index >= ninfer::ops::kPleShards) fail("report shard index is out of range");
        if (seen[index]) fail("report shard is claimed twice");
        seen[index] = true;
        objects[index] = method.at("object").get<std::string>();
        const auto shape = method.at("shape").get<std::vector<std::uint64_t>>();
        if (shape.size() != 2 || shape[0] != ninfer::ops::kPleShardRows || shape[1] != ninfer::ops::kPleFeatures) {
            fail("report shard shape is not [2500012, 160]");
        }
        if (method.at("layout").get<std::string>() != "row_paged_g16_4k_v1") {
            fail("report shard layout is not row_paged_g16_4k_v1");
        }
        const std::string role = "text/layers/" +
            std::to_string(ninfer::targets::qwen3_8_flash_next::detail::roles::kPleLayer) + "/ple/table";
        const auto parameters = method.at("parameters").get<std::vector<std::string>>();
        if (parameters.size() != 1 || parameters[0] != role) {
            fail("report shard is not bound to the PLE layer role");
        }
    }
    for (bool value : seen) {
        if (!value) fail("the conversion report does not cover every shard");
    }
    return objects;
}


std::vector<std::uint32_t> hash_oracle(const PleHashControls& controls,
                                       std::span<const std::uint64_t> ids,
                                       std::array<std::uint64_t, 2> tail, std::uint64_t eos,
                                       std::size_t& negative_words) {
    std::vector<std::uint32_t> rows;
    std::array<std::uint64_t, 2> previous = tail;
    for (const std::uint64_t id : ids) {
        const std::uint64_t older = previous[1] == eos ? eos : previous[0];
        const std::uint64_t pair = id * controls.multipliers[0] ^ previous[1] * controls.multipliers[1];
        const std::uint64_t triple = pair ^ older * controls.multipliers[2];
        for (std::size_t head = 0; head < controls.vocab_sizes.size(); ++head) {
            const std::uint64_t word_bits = head < 8 ? pair : triple;
            const std::int64_t word = static_cast<std::int64_t>(word_bits);
            if (word < 0) ++negative_words;
            std::int64_t index = word % static_cast<std::int64_t>(controls.vocab_sizes[head]);
            if (index < 0) index += static_cast<std::int64_t>(controls.vocab_sizes[head]);
            rows.push_back(static_cast<std::uint32_t>(index + controls.offsets[head]));
        }
        previous = {previous[1], id};
    }
    return rows;
}


double oracle_sigmoid(double value) {
    if (value >= 0.0) return 1.0 / (1.0 + std::exp(-value));
    const double e = std::exp(value);
    return e / (1.0 + e);
}

double oracle_silu(double value) { return value * oracle_sigmoid(value); }

double oracle_inverse_rms(std::span<const double> values, double epsilon) {
    double sum = 0.0;
    for (const double value : values) sum += value * value;
    return 1.0 / std::sqrt(sum / static_cast<double>(values.size()) + epsilon);
}


std::vector<std::uint16_t> read_bits(const void* device, std::size_t count) {
    return from_device<std::uint16_t>(device, count);
}

int verify_decode(std::string_view label, const std::vector<std::uint16_t>& got,
                  const std::vector<std::uint16_t>& expected, bool exact) {
    if (got.size() != expected.size()) fail(std::string(label) + ": size mismatch");
    std::vector<double> got_values(got.size());
    std::vector<double> expected_values(expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        got_values[i] = bf16_bits_to_double(got[i]);
        expected_values[i] = bf16_bits_to_double(expected[i]);
    }
    if (exact) return verify_exact(std::string(label).c_str(), got, expected);
    return verify_pointwise(label, got_values, expected_values, {0.0, kBf16UlpRelative});
}


int test_hash_and_shard_mapping() {
    Reader reader(artifact_path());
    const auto payload = reader.payload("resource/text/ple_hash_controls");
    if (payload.data.size() != ninfer::ops::kPleHashControlsBytes) {
        fail("the hash controls resource is not 280 bytes");
    }
    const PleHashControls controls = ninfer::ops::parse_ple_hash_controls(payload.data);

    const std::array<std::uint64_t, 3> documented_multipliers = {23703573157769ULL, 20109073645365ULL,
                                                                  8052911324071ULL};
    if (controls.multipliers != documented_multipliers) fail("the parsed multipliers are wrong");
    const std::array<std::uint64_t, 16> documented_vocab = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    if (controls.vocab_sizes != documented_vocab) fail("the parsed vocab sizes are wrong");
    const std::array<std::uint64_t, 16> documented_offsets = {
        0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
    if (controls.offsets != documented_offsets) fail("the parsed offsets are wrong");

    const auto report_objects = report_shard_objects();
    const std::vector<std::uint64_t> shard_shape = {ninfer::ops::kPleShardRows, ninfer::ops::kPleFeatures};
    const RowPagedNf4Geometry geometry =
        row_paged_nf4_geometry(NumericFormat::NF4_G16_E4M3S, shard_shape);
    if (geometry.rows_per_page != 45 || geometry.code_bytes_per_row != 80 ||
        geometry.scale_bytes_per_row != 10) {
        fail("the paged NF4 geometry is not the registered one");
    }

    PleNf4Table table{};
    table.geometry = geometry;
    table.shard_count = ninfer::ops::kPleShards;
    std::vector<double> divisors(ninfer::ops::kPleShards, 1.0);
    for (std::size_t shard = 0; shard < ninfer::ops::kPleShards; ++shard) {
        const auto word = read_object_range(reader, report_objects[shard], geometry.matrix_scale_offset, 4);
        float value = 0.0F;
        std::memcpy(&value, word.data(), sizeof(value));
        divisors[shard] = static_cast<double>(value);
    }
    const DeviceBuffer device_divisors = to_device_f32(std::vector<float>(divisors.begin(), divisors.end()));
    table.divisors = static_cast<const float*>(device_divisors.p);

    const std::uint64_t eos = 151645;
    const std::vector<std::uint64_t> ids = {1024, 7, 98140, 151645, 33, 4097, 20000160, 512};
    const std::array<std::uint64_t, 2> tail = {42, eos};

    std::size_t negative_words = 0;
    const std::vector<std::uint32_t> expected_rows = hash_oracle(controls, ids, tail, eos, negative_words);
    if (negative_words == 0) fail("the hash fixture never produced a negative word");

    std::vector<std::uint32_t> rows(ids.size() * ninfer::ops::kPleHeads, 0);
    ninfer::ops::ple_hash_tokens(ids, tail, eos, controls, rows);
    if (rows != expected_rows) fail("the hash rows do not match the independent oracle");

    std::vector<std::uint32_t> boundary_rows;
    for (std::size_t head = 0; head < 12; ++head) {
        boundary_rows.push_back(static_cast<std::uint32_t>(controls.offsets[head]));
    }
    boundary_rows.push_back(static_cast<std::uint32_t>(controls.offsets[0] + controls.vocab_sizes[0] - 1));
    boundary_rows.push_back(static_cast<std::uint32_t>(controls.offsets[8] + controls.vocab_sizes[8] / 2));
    boundary_rows.push_back(static_cast<std::uint32_t>(controls.offsets[15] + controls.vocab_sizes[15] - 1));
    boundary_rows.push_back(static_cast<std::uint32_t>(ninfer::ops::kPleShards * ninfer::ops::kPleShardRows - 1));

    std::vector<std::uint32_t> tested_rows = rows;
    for (const std::uint32_t row : boundary_rows) tested_rows.push_back(row);

    std::vector<std::byte> staged;
    std::vector<std::vector<std::uint16_t>> expected;
    std::vector<bool> exact;
    std::vector<std::size_t> tested_shards;
    for (const std::uint32_t row : tested_rows) {
        const PleRowAddress address = ninfer::ops::ple_row_address(table, row);
        const std::size_t oracle_shard = static_cast<std::size_t>(row) / ninfer::ops::kPleShardRows;
        if (address.shard != oracle_shard || address.row != row % ninfer::ops::kPleShardRows) {
            fail("the row address does not match the shard-order assumption for global row " +
                 std::to_string(row));
        }
        if (object_name(*reader.find(report_objects[address.shard])) != report_objects[address.shard]) {
            fail("the resolved shard is not an independent artifact object");
        }
        tested_shards.push_back(address.shard);
        const auto row_bytes =
            read_object_range(reader, report_objects[address.shard],
                              row_paged_nf4_row_offset(geometry, address.row), geometry.row_bytes);
        staged.insert(staged.end(), row_bytes.begin(), row_bytes.end());
        expected.push_back(decode_row_oracle(row_bytes, geometry, divisors[address.shard]));
        exact.push_back(std::popcount(static_cast<std::uint64_t>(divisors[address.shard])) == 1);
    }

    std::vector<std::size_t> distinct_shards(tested_shards);
    std::sort(distinct_shards.begin(), distinct_shards.end());
    distinct_shards.erase(std::unique(distinct_shards.begin(), distinct_shards.end()), distinct_shards.end());
    const std::size_t covered_shards = distinct_shards.size();
    for (std::size_t shard : {0ULL, 7ULL, 127ULL}) {
        bool seen_shard = false;
        for (std::size_t shard_index : distinct_shards) seen_shard = seen_shard || shard_index == shard;
        if (!seen_shard) fail("the mapping test does not cover shard " + std::to_string(shard));
    }
    if (covered_shards < 4) fail("the mapping test covers too few shards");

    const DeviceBuffer staged_block = to_device(staged);
    const DeviceBuffer device_rows = to_device(tested_rows);
    const std::size_t tokens = tested_rows.size() / ninfer::ops::kPleHeads;
    const std::size_t embedding_words = tokens * ninfer::ops::kPleFeatures * ninfer::ops::kPleHeads;
    DeviceBuffer embedding(embedding_words * 2);
    Tensor embedding_view(embedding.p, DType::BF16,
                          {static_cast<std::int32_t>(ninfer::ops::kPleFeatures * ninfer::ops::kPleHeads),
                           static_cast<std::int32_t>(tokens), 1});
    ninfer::ops::ple_lookup_rows(table, static_cast<const std::uint32_t*>(device_rows.p),
                                 tested_rows.size(), static_cast<const std::uint8_t*>(staged_block.p),
                                 embedding_view, 0);
    cuda_synchronize();

    const auto written = read_bits(embedding.p, embedding_words);
    for (std::size_t i = 0; i < tested_rows.size(); ++i) {
        const std::size_t slot_base = (i / ninfer::ops::kPleHeads) *
                                          (ninfer::ops::kPleFeatures * ninfer::ops::kPleHeads) +
                                      (i % ninfer::ops::kPleHeads) * ninfer::ops::kPleFeatures;
        std::vector<std::uint16_t> slot(written.begin() + static_cast<std::ptrdiff_t>(slot_base),
                                        written.begin() + static_cast<std::ptrdiff_t>(slot_base + ninfer::ops::kPleFeatures));
        if (verify_decode("ple_lookup_rows row", slot, expected[i], exact[i])) return 1;
    }

    const std::vector<std::uint16_t> zeros(ninfer::ops::kPleFeatures * ninfer::ops::kPleHeads, 0);
    const std::uint32_t uncovered = ninfer::ops::kPleShards * ninfer::ops::kPleShardRows;
    std::vector<std::uint32_t> single(ninfer::ops::kPleHeads, uncovered);
    DeviceBuffer uncovered_embedding(single.size() * ninfer::ops::kPleFeatures * 2);
    Tensor uncovered_view(uncovered_embedding.p, DType::BF16,
                          {static_cast<std::int32_t>(ninfer::ops::kPleFeatures * ninfer::ops::kPleHeads), 1, 1});
    const std::vector<std::byte> filler(single.size() * geometry.row_bytes, std::byte{0x33});
    const DeviceBuffer filler_block = to_device(filler);
    const DeviceBuffer single_row = to_device(single);
    ninfer::ops::ple_lookup_rows(table, static_cast<const std::uint32_t*>(single_row.p), single.size(),
                                 static_cast<const std::uint8_t*>(filler_block.p), uncovered_view, 0);
    cuda_synchronize();
    if (read_bits(uncovered_embedding.p, zeros.size()) != zeros) {
        fail("a row no shard covers must contribute exact +0 words");
    }
    bool outside_throws = false;
    try {
        const PleRowAddress outside = ninfer::ops::ple_row_address(table, uncovered);
        (void)outside;
    } catch (const std::invalid_argument&) {
        outside_throws = true;
    }
    if (!outside_throws) fail("a row outside the table must have no address");

    std::cout << "OK ple hash + shard mapping: " << covered_shards << " shards, "
              << negative_words << " negative words\n";
    return 0;
}


int test_gate() {
    const std::int32_t tokens = 4;
    const std::int32_t batch = 2;
    const std::size_t columns = static_cast<std::size_t>(tokens) * batch;
    const std::size_t channels = ninfer::ops::kPleChannels;
    const float epsilon = 1.0e-6F;

    std::vector<float> key(channels * columns);
    std::vector<float> value(static_cast<std::size_t>(ninfer::ops::kPleHidden) * columns);
    std::vector<float> residual(channels * columns);
    std::vector<float> norm_key(channels);
    std::vector<float> norm_query(channels);
    std::vector<float> norm_conv(channels);

    fill_uniform(key, 9001U, -1.0F, 1.0F);
    fill_uniform(value, 9002U, -1.0F, 1.0F);
    fill_uniform(residual, 9003U, -1.0F, 1.0F);
    fill_uniform(norm_key, 9004U, -0.5F, 0.5F);
    fill_uniform(norm_query, 9005U, -0.5F, 0.5F);
    fill_uniform(norm_conv, 9006U, -0.5F, 0.5F);

    for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) key[d] = 1.0e-13F;
    for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) key[ninfer::ops::kPleHidden + d] = -1.0e-13F;
    for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) key[2 * ninfer::ops::kPleHidden + d] = 0.0F;

    const auto key_bits = encode_bf16(key);
    const auto value_bits = encode_bf16(value);
    const auto residual_bits = encode_bf16(residual);
    const auto norm_key_bits = encode_bf16(norm_key);
    const auto norm_query_bits = encode_bf16(norm_query);
    const auto norm_conv_bits = encode_bf16(norm_conv);

    const DeviceBuffer key_device = to_device(key_bits);
    const DeviceBuffer value_device = to_device(value_bits);
    const DeviceBuffer residual_device = to_device(residual_bits);
    const DeviceBuffer norm_key_device = to_device(norm_key_bits);
    const DeviceBuffer norm_query_device = to_device(norm_query_bits);
    const DeviceBuffer norm_conv_device = to_device(norm_conv_bits);
    GuardedDeviceBuffer gated(channels * columns * 2);
    GuardedDeviceBuffer normalized(channels * columns * 2);

    Tensor key_view(key_device.p, DType::BF16, {static_cast<std::int32_t>(channels), tokens, batch});
    Tensor value_view(value_device.p, DType::BF16, {static_cast<std::int32_t>(ninfer::ops::kPleHidden), tokens, batch});
    Tensor residual_view(residual_device.p, DType::BF16,
                         {static_cast<std::int32_t>(ninfer::ops::kPleHidden), static_cast<std::int32_t>(ninfer::ops::kPleStreams),
                          tokens, batch});
    Tensor norm_key_view(norm_key_device.p, DType::BF16, {static_cast<std::int32_t>(channels)});
    Tensor norm_query_view(norm_query_device.p, DType::BF16, {static_cast<std::int32_t>(channels)});
    Tensor norm_conv_view(norm_conv_device.p, DType::BF16, {static_cast<std::int32_t>(channels)});
    Tensor gated_view(gated.data(), DType::BF16, {static_cast<std::int32_t>(channels), tokens, batch});
    Tensor normalized_view(normalized.data(), DType::BF16, {static_cast<std::int32_t>(channels), tokens, batch});

    ninfer::ops::ple_gate(key_view, value_view, residual_view, norm_key_view, norm_query_view,
                          norm_conv_view, epsilon, gated_view, normalized_view, 0);
    cuda_synchronize();

    std::size_t floor_branches = 0;
    std::size_t negative_branches = 0;
    std::size_t zero_branches = 0;

    std::vector<double> expected_gated(channels * columns);
    std::vector<double> expected_normalized(channels * columns);
    for (std::size_t b = 0; b < static_cast<std::size_t>(batch); ++b) {
        for (std::size_t t = 0; t < static_cast<std::size_t>(tokens); ++t) {
            const std::size_t column = t + static_cast<std::size_t>(tokens) * b;
            for (std::size_t s = 0; s < ninfer::ops::kPleStreams; ++s) {
                const std::size_t base = s * ninfer::ops::kPleHidden;
                std::span<const double> key_stream;
                std::span<const double> query_stream;
                std::vector<double> key_values(base + ninfer::ops::kPleHidden);
                std::vector<double> query_values(base + ninfer::ops::kPleHidden);
                for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) {
                    key_values[base + d] = bf16_bits_to_double(key_bits[base + d + column * channels]);
                    query_values[base + d] =
                        bf16_bits_to_double(residual_bits[base + d + column * channels]);
                }
                key_stream = std::span<const double>(key_values.data() + base, ninfer::ops::kPleHidden);
                query_stream = std::span<const double>(query_values.data() + base, ninfer::ops::kPleHidden);

                const double inverse_key = oracle_inverse_rms(key_stream, epsilon);
                const double inverse_query = oracle_inverse_rms(query_stream, epsilon);
                double dot = 0.0;
                for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) {
                    const double normalized_key =
                        key_stream[d] * (1.0 + bf16_bits_to_double(norm_key_bits[base + d]));
                    const double normalized_query =
                        query_stream[d] * (1.0 + bf16_bits_to_double(norm_query_bits[base + d]));
                    dot += normalized_key * normalized_query;
                }
                const double a = dot * inverse_key * inverse_query / std::sqrt(static_cast<double>(ninfer::ops::kPleHidden));
                if (std::abs(a) < kGateFloor) ++floor_branches;
                if (a < 0.0) ++negative_branches;
                if (a == 0.0) ++zero_branches;
                const double gate = oracle_sigmoid((a < 0.0 ? -1.0 : (a == 0.0 ? 0.0 : 1.0)) *
                                                   std::sqrt(std::max(std::abs(a), kGateFloor)));

                std::vector<std::uint16_t> stored(ninfer::ops::kPleHidden);
                std::vector<double> stored_values(ninfer::ops::kPleHidden);
                for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) {
                    const double product = gate * bf16_bits_to_double(value_bits[d + column * ninfer::ops::kPleHidden]);
                    stored[d] = double_to_bf16(product);
                    stored_values[d] = bf16_bits_to_double(stored[d]);
                    expected_gated[base + d + column * channels] = stored_values[d];
                }
                const double inverse_conv = oracle_inverse_rms(stored_values, epsilon);
                for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) {
                    expected_normalized[base + d + column * channels] =
                        stored_values[d] * (1.0 + bf16_bits_to_double(norm_conv_bits[base + d])) * inverse_conv;
                }
            }
        }
    }

    if (floor_branches == 0 || negative_branches == 0 || zero_branches == 0) {
        fail("the gate fixture does not cover the floor, negative and zero branches");
    }

    int failures = gated.verify_guards("ple_gate gated_value");
    const auto gated_bits = read_bits(gated.data(), channels * columns);
    std::vector<std::uint16_t> exact_zero_gate;
    for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) {
        exact_zero_gate.push_back(f32_to_bf16(0.5F * static_cast<float>(bf16_bits_to_double(value_bits[d]))));
    }
    std::vector<std::uint16_t> zero_stream(gated_bits.begin() + 2 * ninfer::ops::kPleHidden,
                                           gated_bits.begin() + 3 * ninfer::ops::kPleHidden);
    failures += verify_exact("ple_gate gate at zero", zero_stream, exact_zero_gate);
    failures += normalized.verify_guards("ple_gate normalized_conv");
    failures += verify_reduction("ple_gate gated_value", from_device_bf16(gated.data(), channels * columns),
                                 expected_gated, kNormCriterion);
    failures += verify_reduction("ple_gate normalized_conv",
                                 from_device_bf16(normalized.data(), channels * columns), expected_normalized,
                                 kNormCriterion);
    if (failures == 0) std::cout << "OK ple_gate\n";
    return failures;
}


int test_conv(bool normalized_exit) {
    const std::int32_t tokens = 4;
    const std::int32_t batch = 2;
    const std::size_t columns = static_cast<std::size_t>(tokens) * batch;
    const std::size_t channels = ninfer::ops::kPleChannels;
    const float epsilon = 1.0e-6F;

    std::vector<float> history(channels * ninfer::ops::kPleConvWindow * batch);
    std::vector<float> values(channels * columns);
    std::vector<float> residual(channels * columns);
    std::vector<float> gated(channels * columns);
    std::vector<float> kernel(ninfer::ops::kPleConvTaps * channels);
    std::vector<float> next_norm(channels);

    fill_uniform(history, 7001U, -1.0F, 1.0F);
    fill_uniform(values, 7002U, -1.0F, 1.0F);
    fill_uniform(residual, 7003U, -1.0F, 1.0F);
    fill_uniform(gated, 7004U, -1.0F, 1.0F);
    fill_uniform(kernel, 7005U, -0.5F, 0.5F);
    fill_uniform(next_norm, 7006U, -0.5F, 0.5F);

    const auto history_bits = encode_bf16(history);
    const auto values_bits = encode_bf16(values);
    const auto residual_bits = encode_bf16(residual);
    const auto gated_bits = encode_bf16(gated);
    const auto kernel_bits = encode_bf16(kernel);
    const auto next_norm_bits = encode_bf16(next_norm);

    const std::vector<std::int32_t> offsets = {0, 2};
    const std::vector<std::int32_t> counts = {tokens, 2};

    const DeviceBuffer history_device = to_device(history_bits);
    const DeviceBuffer values_device = to_device(values_bits);
    const DeviceBuffer residual_device = to_device(residual_bits);
    const DeviceBuffer gated_device = to_device(gated_bits);
    const DeviceBuffer kernel_device = to_device(kernel_bits);
    const DeviceBuffer next_norm_device = to_device(next_norm_bits);
    const DeviceBuffer offsets_device = to_device_i32(offsets);
    const DeviceBuffer counts_device = to_device_i32(counts);
    GuardedDeviceBuffer updated(channels * columns * 2);
    GuardedDeviceBuffer normalized(normalized_exit ? channels * columns * 2 : 0);

    Tensor history_view(history_device.p, DType::BF16,
                        {static_cast<std::int32_t>(channels), static_cast<std::int32_t>(ninfer::ops::kPleConvWindow), batch});
    Tensor values_view(values_device.p, DType::BF16, {static_cast<std::int32_t>(channels), tokens, batch});
    Tensor residual_view(residual_device.p, DType::BF16,
                         {static_cast<std::int32_t>(ninfer::ops::kPleHidden), static_cast<std::int32_t>(ninfer::ops::kPleStreams),
                          tokens, batch});
    Tensor gated_view(gated_device.p, DType::BF16, {static_cast<std::int32_t>(channels), tokens, batch});
    Tensor kernel_view(kernel_device.p, DType::BF16, {static_cast<std::int32_t>(ninfer::ops::kPleConvTaps), static_cast<std::int32_t>(channels)});
    Tensor next_norm_view(next_norm_device.p, DType::BF16, {static_cast<std::int32_t>(channels)});
    Tensor offsets_view(offsets_device.p, DType::I32, {batch});
    Tensor counts_view(counts_device.p, DType::I32, {batch});
    Tensor updated_view(updated.data(), DType::BF16,
                        {static_cast<std::int32_t>(ninfer::ops::kPleHidden), static_cast<std::int32_t>(ninfer::ops::kPleStreams),
                         tokens, batch});
    Tensor normalized_view;
    if (normalized_exit) {
        normalized_view = Tensor(normalized.data(), DType::BF16,
                                 {static_cast<std::int32_t>(ninfer::ops::kPleHidden), static_cast<std::int32_t>(ninfer::ops::kPleStreams),
                                  tokens, batch});
    }

    if (normalized_exit) {
        ninfer::ops::ple_conv_add_norm(history_view, values_view, residual_view, gated_view, kernel_view,
                                       offsets_view, counts_view, next_norm_view, epsilon, updated_view,
                                       normalized_view, 0);
    } else {
        ninfer::ops::ple_conv_add(history_view, values_view, residual_view, gated_view, kernel_view,
                                  offsets_view, counts_view, updated_view, 0);
    }
    cuda_synchronize();

    std::size_t history_taps = 0;
    std::vector<double> expected_updated(channels * columns);
    std::vector<double> expected_normalized(channels * columns);
    std::vector<std::uint16_t> stored(channels * columns);

    for (std::size_t b = 0; b < static_cast<std::size_t>(batch); ++b) {
        for (std::int32_t t = 0; t < counts[b]; ++t) {
            for (std::size_t c = 0; c < channels; ++c) {
                double sum = 0.0;
                for (std::size_t tap = 0; tap < ninfer::ops::kPleConvTaps; ++tap) {
                    const std::int32_t k = t - static_cast<std::int32_t>(ninfer::ops::kPleConvWindow) +
                                           static_cast<std::int32_t>(tap) * static_cast<std::int32_t>(ninfer::ops::kPleConvDilation);
                    double source = 0.0;
                    if (k < 0) {
                        source = bf16_bits_to_double(
                            history_bits[c + channels * static_cast<std::size_t>(9 + k) +
                                        channels * ninfer::ops::kPleConvWindow * b]);
                        ++history_taps;
                    } else {
                        const std::size_t column = static_cast<std::size_t>(offsets[b] + k) +
                                                   static_cast<std::size_t>(tokens) * b;
                        source = bf16_bits_to_double(values_bits[c + channels * column]);
                    }
                    sum += bf16_bits_to_double(kernel_bits[tap * channels + c]) * source;
                }
                const std::size_t s = c / ninfer::ops::kPleHidden;
                const std::size_t d = c % ninfer::ops::kPleHidden;
                const std::size_t column = static_cast<std::size_t>(t) + static_cast<std::size_t>(tokens) * b;
                const std::size_t index = d + ninfer::ops::kPleHidden * s + channels * column;
                const double ideal = bf16_bits_to_double(residual_bits[index]) +
                                     bf16_bits_to_double(gated_bits[c + channels * column]) + oracle_silu(sum);
                stored[index] = double_to_bf16(ideal);
                expected_updated[index] = bf16_bits_to_double(stored[index]);
            }
        }
    }

    for (std::size_t b = 0; b < static_cast<std::size_t>(batch); ++b) {
        for (std::int32_t t = 0; t < counts[b]; ++t) {
            for (std::size_t s = 0; s < ninfer::ops::kPleStreams; ++s) {
                const std::size_t column = static_cast<std::size_t>(t) + static_cast<std::size_t>(tokens) * b;
                std::vector<double> stream(ninfer::ops::kPleHidden);
                for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) {
                    stream[d] = expected_updated[d + ninfer::ops::kPleHidden * s + channels * column];
                }
                const double inverse = oracle_inverse_rms(stream, epsilon);
                for (std::size_t d = 0; d < ninfer::ops::kPleHidden; ++d) {
                    expected_normalized[d + ninfer::ops::kPleHidden * s + channels * column] =
                        stream[d] * (1.0 + bf16_bits_to_double(next_norm_bits[d + ninfer::ops::kPleHidden * s])) * inverse;
                }
            }
        }
    }

    if (history_taps == 0) fail("the conv fixture never read the history window");

    int failures = updated.verify_guards("ple_conv_add updated_residual");
    failures += verify_reduction("ple_conv_add updated_residual",
                                 from_device_bf16(updated.data(), channels * columns), expected_updated,
                                 kNormCriterion);
    if (normalized_exit) {
        failures += normalized.verify_guards("ple_conv_add_norm normalized");
        failures += verify_reduction("ple_conv_add_norm normalized",
                                     from_device_bf16(normalized.data(), channels * columns),
                                     expected_normalized, kNormCriterion);
    }
    if (failures == 0) std::cout << "OK ple_conv_add" << (normalized_exit ? "_norm" : "") << "\n";
    return failures;
}


int test_state_extract() {
    const std::int32_t tokens = 4;
    const std::int32_t batch = 2;
    const std::size_t channels = ninfer::ops::kPleChannels;
    const std::size_t window = ninfer::ops::kPleConvWindow;

    std::vector<float> before(channels * window * batch);
    std::vector<float> records(channels * tokens * batch);
    fill_uniform(before, 6001U, -1.0F, 1.0F);
    fill_uniform(records, 6002U, -1.0F, 1.0F);

    const auto before_bits = encode_bf16(before);
    const auto records_bits = encode_bf16(records);
    const std::vector<std::int32_t> accepted = {0, 4};

    const DeviceBuffer before_device = to_device(before_bits);
    const DeviceBuffer records_device = to_device(records_bits);
    const DeviceBuffer accepted_device = to_device_i32(accepted);
    GuardedDeviceBuffer after(channels * window * batch * 2);

    Tensor before_view(before_device.p, DType::BF16, {static_cast<std::int32_t>(channels), static_cast<std::int32_t>(window), batch});
    Tensor records_view(records_device.p, DType::BF16, {static_cast<std::int32_t>(channels), tokens, batch});
    Tensor accepted_view(accepted_device.p, DType::I32, {batch});
    Tensor after_view(after.data(), DType::BF16, {static_cast<std::int32_t>(channels), static_cast<std::int32_t>(window), batch});

    ninfer::ops::ple_state_extract(before_view, records_view, accepted_view, after_view, 0);
    cuda_synchronize();

    std::vector<std::uint16_t> expected(channels * window * batch);
    for (std::size_t b = 0; b < static_cast<std::size_t>(batch); ++b) {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t j = 0; j < window; ++j) {
                const std::int32_t k = accepted[b] - static_cast<std::int32_t>(window) + static_cast<std::int32_t>(j);
                const std::size_t index = c + channels * j + channels * window * b;
                if (k < 0) {
                    expected[index] = before_bits[c + channels * static_cast<std::size_t>(9 + k) + channels * window * b];
                } else {
                    expected[index] = records_bits[c + channels * static_cast<std::size_t>(k) + channels * tokens * b];
                }
            }
        }
    }

    int failures = after.verify_guards("ple_state_extract");
    failures += verify_exact("ple_state_extract", read_bits(after.data(), expected.size()), expected);

    const DeviceBuffer alias_device = to_device(before_bits);
    Tensor before_alias(alias_device.p, DType::BF16,
                        {static_cast<std::int32_t>(channels), static_cast<std::int32_t>(window), batch});
    Tensor after_alias(alias_device.p, DType::BF16,
                       {static_cast<std::int32_t>(channels), static_cast<std::int32_t>(window), batch});
    ninfer::ops::ple_state_extract(before_alias, records_view, accepted_view, after_alias, 0);
    cuda_synchronize();

    std::vector<std::uint16_t> alias = before_bits;
    for (std::size_t b = 0; b < static_cast<std::size_t>(batch); ++b) {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t j = 0; j < window; ++j) {
                const std::int32_t k = accepted[b] - static_cast<std::int32_t>(window) + static_cast<std::int32_t>(j);
                const std::size_t index = c + channels * j + channels * window * b;
                if (k < 0) {
                    alias[index] = alias[c + channels * static_cast<std::size_t>(9 + k) + channels * window * b];
                } else {
                    alias[index] = records_bits[c + channels * static_cast<std::size_t>(k) + channels * tokens * b];
                }
            }
        }
    }

    failures += verify_exact("ple_state_extract alias", read_bits(alias_device.p, alias.size()), alias);

    if (failures == 0) std::cout << "OK ple_state_extract\n";
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (std::getenv("NINFER_FLASH_NEXT_ARTIFACT") == nullptr) {
        std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
        return 77;
    }

    int failures = 0;
    failures += test_hash_and_shard_mapping();
    failures += test_gate();
    failures += test_conv(false);
    failures += test_conv(true);
    failures += test_state_extract();
    return failures;
}
