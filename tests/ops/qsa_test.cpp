#include "ninfer/ops/qsa.h"

#include "core/arena.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads = 24;
constexpr std::int32_t kKvHeads = 2;
constexpr std::int32_t kIndexDim = 128;
constexpr std::int32_t kIndexHeads = 4;
constexpr std::int32_t kCompress = 4;
constexpr std::int32_t kCapacity = 512;
constexpr std::int32_t kRotary = 64;
constexpr float kTheta = 1000000.0F;
constexpr float kEpsilon = 1.0e-6F;
constexpr std::uint16_t kSentinelBf16 = 0x07FFU;
constexpr std::int32_t kSentinelI32 = 0x7FF;

const PointwiseCriterion kCriterion{1.0e-3, 8.0e-3};

double bf16_double(std::uint16_t bits) { return double(bf16_to_f32(bits)); }
double fp16_double(std::uint16_t bits) { return double(f16_to_f32(bits)); }
std::uint16_t double_bf16(double value) { return f32_to_bf16(static_cast<float>(value)); }
std::uint16_t double_fp16(double value) { return f32_to_f16(static_cast<float>(value)); }

double act_double(std::uint16_t bits, DType dtype) {
    return dtype == DType::FP16 ? fp16_double(bits) : bf16_double(bits);
}
std::uint16_t act_bits(double value, DType dtype) {
    return dtype == DType::FP16 ? double_fp16(value) : double_bf16(value);
}
std::vector<std::uint16_t> to_activation(std::vector<std::uint16_t> bits, DType dtype) {
    if (dtype == DType::FP16) {
        for (std::uint16_t& value : bits) { value = f32_to_f16(bf16_to_f32(value)); }
    }
    return bits;
}
std::vector<double> read_activation(const void* device, std::size_t count, DType dtype) {
    return dtype == DType::FP16 ? from_device_f16(device, count)
                                : from_device_bf16(device, count);
}

std::vector<std::uint16_t> bf16_inputs(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<std::uint16_t> integer_inputs(std::size_t count, std::uint32_t seed) {
    static const double kValues[]{0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0};
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) {
        bits[i] = double_bf16(kValues[(i * 2654435761U + seed) % 8U]);
    }
    return bits;
}

std::vector<double> rmsnorm(const std::vector<double>& x, const std::vector<double>& gain,
                            std::int32_t dim) {
    double sum = 0.0;
    for (double value : x) sum += value * value;
    const double inverse = 1.0 / std::sqrt(sum / static_cast<double>(dim) + double(kEpsilon));
    std::vector<double> out(static_cast<std::size_t>(dim));
    for (std::int32_t d = 0; d < dim; ++d) {
        out[static_cast<std::size_t>(d)] =
            x[static_cast<std::size_t>(d)] * (1.0 + gain[static_cast<std::size_t>(d)]) * inverse;
    }
    return out;
}

std::vector<double> rotate(const std::vector<double>& z, const std::array<double, 3>& rho,
                           std::int32_t dim) {
    const std::int32_t half = kRotary / 2;
    std::vector<double> out(static_cast<std::size_t>(dim));
    for (std::int32_t i = 0; i < half; ++i) {
        const double frequency = std::pow(double(kTheta), -2.0 * double(i) / double(kRotary));
        const double angle = rho[static_cast<std::size_t>(i % 3)] * frequency;
        const double sine = std::sin(angle);
        const double cosine = std::cos(angle);
        out[static_cast<std::size_t>(i)] = z[static_cast<std::size_t>(i)] * cosine -
                                           z[static_cast<std::size_t>(i + half)] * sine;
        out[static_cast<std::size_t>(i + half)] = z[static_cast<std::size_t>(i + half)] * cosine +
                                                   z[static_cast<std::size_t>(i)] * sine;
    }
    for (std::int32_t d = kRotary; d < dim; ++d) out[static_cast<std::size_t>(d)] = z[static_cast<std::size_t>(d)];
    return out;
}

Tensor tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape) {
    return Tensor(data, dtype, shape);
}

DeviceBuffer filled_device(std::size_t bytes, std::uint8_t byte) {
    DeviceBuffer buffer(bytes);
    buffer.fill(byte);
    return buffer;
}

std::vector<double> gain_vector(const std::vector<std::uint16_t>& bits) {
    std::vector<double> gain(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) gain[i] = bf16_double(bits[i]);
    return gain;
}

std::size_t d2i(std::int32_t d, std::int32_t h, std::int32_t t, std::int32_t b, std::int32_t dim,
                std::int32_t heads, std::int32_t T, std::int32_t B) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(dim) *
               (static_cast<std::size_t>(h) +
                static_cast<std::size_t>(heads) *
                    (static_cast<std::size_t>(t) + static_cast<std::size_t>(T) * b));
}

struct PrepareCase {
    std::string label;
    DType activation = DType::BF16;
    std::int32_t T;
    std::int32_t B;
    std::int32_t axes;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> valid;
    std::vector<std::int32_t> table_rows;
    std::int32_t logical_pages;
    std::vector<std::int32_t> page_table;
    std::vector<std::uint16_t> proj_query;
    std::vector<std::uint16_t> proj_key;
    std::vector<std::uint16_t> proj_value;
    std::vector<std::uint16_t> proj_index_query;
    std::vector<std::uint16_t> proj_index_key;
    std::vector<std::uint16_t> norm_query;
    std::vector<std::uint16_t> norm_key;
    std::vector<std::uint16_t> norm_index_query;
    std::vector<std::uint16_t> norm_index_key;
    std::vector<std::uint16_t> tail_keys;
    std::vector<std::int32_t> tail_rope;
    std::int32_t record_width;
    std::int32_t physical_pages;
    std::int32_t index_pages;
};

int run_prepare(const PrepareCase& c, cudaStream_t stream, WorkspaceArena& arena) {
    const std::size_t query_count = static_cast<std::size_t>(kHeadDim) * kQHeads * c.T * c.B;
    const std::size_t index_query_count = static_cast<std::size_t>(kIndexDim) * kIndexHeads * c.T * c.B;
    const std::size_t k_count = static_cast<std::size_t>(kHeadDim) * kQsaTokenPageSize * kKvHeads *
                                c.physical_pages;
    const std::size_t index_count =
        static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * c.index_pages;
    const std::size_t record_count = static_cast<std::size_t>(kIndexDim) * c.record_width * c.B;

    std::vector<double> ref_query(query_count, act_double(kSentinelBf16, c.activation));
    std::vector<double> ref_index_query(index_query_count,
                                      act_double(kSentinelBf16, c.activation));
    std::vector<double> ref_k(k_count, bf16_double(kSentinelBf16));
    std::vector<std::uint16_t> ref_v(k_count, kSentinelBf16);
    std::vector<double> ref_index(index_count, bf16_double(kSentinelBf16));
    std::vector<std::uint16_t> ref_records(record_count, kSentinelBf16);
    std::vector<std::int32_t> ref_rope(3 * c.record_width * c.B, kSentinelI32);
    std::vector<std::int32_t> ref_base(c.B, kSentinelI32);

    for (std::int32_t b = 0; b < c.B; ++b) {
        const std::int32_t F = c.positions[static_cast<std::size_t>(b)];
        const std::int32_t n = c.valid[static_cast<std::size_t>(b)];
        const std::int32_t table_row = c.table_rows[static_cast<std::size_t>(b)];

        for (std::int32_t t = 0; t < n; ++t) {
            std::array<double, 3> rho{};
            for (std::int32_t axis = 0; axis < 3; ++axis) {
                const std::int32_t a = c.axes == 1 ? 0 : axis;
                rho[static_cast<std::size_t>(axis)] =
                    c.positions[static_cast<std::size_t>((t * c.axes + a) * c.B + b)];
            }
            for (std::int32_t h = 0; h < kQHeads; ++h) {
                std::vector<double> head(static_cast<std::size_t>(kHeadDim));
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    head[static_cast<std::size_t>(d)] =
                        act_double(c.proj_query[static_cast<std::size_t>(
                                       d + static_cast<std::size_t>(kHeadDim) *
                                               (h + static_cast<std::size_t>(kQHeads) *
                                                        (t + c.T * b)))],
                                   c.activation);
                }
                const std::vector<double> rotated =
                    rotate(rmsnorm(head, gain_vector(c.norm_query), kHeadDim), rho, kHeadDim);
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    ref_query[static_cast<std::size_t>(d2i(d, h, t, b, kHeadDim, kQHeads, c.T, c.B))] =
                        act_double(act_bits(rotated[static_cast<std::size_t>(d)], c.activation),
                                   c.activation);
                }
            }
            for (std::int32_t h = 0; h < kIndexHeads; ++h) {
                std::vector<double> head(static_cast<std::size_t>(kIndexDim));
                for (std::int32_t d = 0; d < kIndexDim; ++d) {
                    head[static_cast<std::size_t>(d)] =
                        act_double(c.proj_index_query[static_cast<std::size_t>(
                                       d + static_cast<std::size_t>(kIndexDim) *
                                               (h + static_cast<std::size_t>(kIndexHeads) *
                                                        (t + c.T * b)))],
                                   c.activation);
                }
                const std::vector<double> rotated = rotate(
                    rmsnorm(head, gain_vector(c.norm_index_query), kIndexDim), rho, kIndexDim);
                for (std::int32_t d = 0; d < kIndexDim; ++d) {
                    ref_index_query[static_cast<std::size_t>(
                        d2i(d, h, t, b, kIndexDim, kIndexHeads, c.T, c.B))] =
                        act_double(act_bits(rotated[static_cast<std::size_t>(d)], c.activation),
                                   c.activation);
                }
            }
            for (std::int32_t h = 0; h < kKvHeads; ++h) {
                std::vector<double> head(static_cast<std::size_t>(kHeadDim));
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    head[static_cast<std::size_t>(d)] =
                        act_double(c.proj_key[static_cast<std::size_t>(
                                       d + static_cast<std::size_t>(kHeadDim) *
                                               (h + static_cast<std::size_t>(kKvHeads) *
                                                        (t + c.T * b)))],
                                   c.activation);
                }
                const std::vector<double> rotated =
                    rotate(rmsnorm(head, gain_vector(c.norm_key), kHeadDim), rho, kHeadDim);
                const std::int32_t position = F + t;
                const std::int32_t page = position / kQsaTokenPageSize;
                const std::int32_t physical =
                    c.page_table[static_cast<std::size_t>(page + c.logical_pages * table_row)];
                const std::int32_t row = position % kQsaTokenPageSize;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t cache_base = static_cast<std::size_t>(d) +
                        static_cast<std::size_t>(kHeadDim) *
                            (static_cast<std::size_t>(row) +
                             static_cast<std::size_t>(kQsaTokenPageSize) *
                                 (static_cast<std::size_t>(h) +
                                  static_cast<std::size_t>(kKvHeads) *
                                      static_cast<std::size_t>(physical)));
                    ref_k[cache_base] = bf16_double(double_bf16(rotated[static_cast<std::size_t>(d)]));
                    const std::uint16_t value_bits = c.proj_value[static_cast<std::size_t>(
                        d + static_cast<std::size_t>(kHeadDim) *
                            (h + static_cast<std::size_t>(kKvHeads) * (t + c.T * b)))];
                    ref_v[cache_base] = double_fp16(act_double(value_bits, c.activation));
                }
            }
            for (std::int32_t d = 0; d < kIndexDim; ++d) {
                ref_records[static_cast<std::size_t>(
                    d + static_cast<std::size_t>(kIndexDim) * (static_cast<std::size_t>(t) +
                                                                static_cast<std::size_t>(c.record_width) * b))] =
                    double_bf16(act_double(
                        c.proj_index_key[static_cast<std::size_t>(
                            d + static_cast<std::size_t>(kIndexDim) *
                                (static_cast<std::size_t>(t) + static_cast<std::size_t>(c.T) * b))],
                        c.activation));
            }
            for (std::int32_t axis = 0; axis < 3; ++axis) {
                const std::int32_t a = c.axes == 1 ? 0 : axis;
                ref_rope[static_cast<std::size_t>(
                    axis + 3 * (static_cast<std::size_t>(t) + static_cast<std::size_t>(c.record_width) * b))] =
                    c.positions[static_cast<std::size_t>((t * c.axes + a) * c.B + b)];
            }
            ref_base[static_cast<std::size_t>(b)] = F;
        }

        const std::int32_t first_block = F / kCompress;
        const std::int32_t end_block = (F + n) / kCompress;
        for (std::int32_t g = first_block; g < end_block; ++g) {
            const std::int32_t block_start = g * kCompress;
            std::vector<double> u(static_cast<std::size_t>(kIndexDim));
            for (std::int32_t d = 0; d < kIndexDim; ++d) {
                double sum = 0.0;
                for (std::int32_t c2 = 0; c2 < kCompress; ++c2) {
                    const std::int32_t token = block_start + c2;
                    if (token < F) {
                        const std::int32_t column = token - (F - F % kCompress);
                        sum += bf16_double(c.tail_keys[static_cast<std::size_t>(
                            d + static_cast<std::size_t>(kIndexDim) *
                                (static_cast<std::size_t>(column) + 4 * b))]);
                    } else {
                        const std::int32_t local = token - F;
                        sum += act_double(c.proj_index_key[static_cast<std::size_t>(
                            d + static_cast<std::size_t>(kIndexDim) *
                                (static_cast<std::size_t>(local) + static_cast<std::size_t>(c.T) * b))],
                            c.activation);
                    }
                }
                u[static_cast<std::size_t>(d)] = bf16_double(double_bf16(sum / 4.0));
            }
            std::array<double, 3> rho_block{};
            if (block_start < F) {
                for (std::int32_t axis = 0; axis < 3; ++axis) {
                    rho_block[static_cast<std::size_t>(axis)] =
                        c.tail_rope[static_cast<std::size_t>(axis + 3 * b)];
                }
            } else {
                const std::int32_t local = block_start - F;
                for (std::int32_t axis = 0; axis < 3; ++axis) {
                    rho_block[static_cast<std::size_t>(axis)] =
                        c.positions[static_cast<std::size_t>((local * c.axes + axis) * c.B + b)];
                }
            }
            const std::vector<double> rotated =
                rotate(rmsnorm(u, gain_vector(c.norm_index_key), kIndexDim), rho_block, kIndexDim);
            const std::int32_t logical_page = g / kQsaIndexBlockPageSize;
            const std::int32_t physical =
                c.page_table[static_cast<std::size_t>(logical_page + c.logical_pages * table_row)];
            const std::int32_t row = g % kQsaIndexBlockPageSize;
            for (std::int32_t d = 0; d < kIndexDim; ++d) {
                ref_index[static_cast<std::size_t>(
                    d + static_cast<std::size_t>(kIndexDim) *
                        (static_cast<std::size_t>(row) +
                         static_cast<std::size_t>(kQsaIndexBlockPageSize) *
                             static_cast<std::size_t>(physical)))] =
                    bf16_double(double_bf16(rotated[static_cast<std::size_t>(d)]));
            }
        }
    }

    DeviceBuffer d_proj_query = to_device(c.proj_query);
    DeviceBuffer d_proj_key = to_device(c.proj_key);
    DeviceBuffer d_proj_value = to_device(c.proj_value);
    DeviceBuffer d_proj_index_query = to_device(c.proj_index_query);
    DeviceBuffer d_proj_index_key = to_device(c.proj_index_key);
    DeviceBuffer d_norm_query = to_device(c.norm_query);
    DeviceBuffer d_norm_key = to_device(c.norm_key);
    DeviceBuffer d_norm_index_query = to_device(c.norm_index_query);
    DeviceBuffer d_norm_index_key = to_device(c.norm_index_key);
    DeviceBuffer d_tail_keys = to_device(c.tail_keys);
    DeviceBuffer d_tail_rope = to_device_i32(c.tail_rope);
    DeviceBuffer d_positions = to_device_i32(c.positions);
    DeviceBuffer d_valid = to_device_i32(c.valid);
    DeviceBuffer d_table_rows = to_device_i32(c.table_rows);
    DeviceBuffer d_page_table = to_device_i32(c.page_table);
    DeviceBuffer d_records(static_cast<std::size_t>(record_count) * 2);
    d_records.copy_from_host(ref_records.data(), record_count * 2);
    DeviceBuffer d_rope(static_cast<std::size_t>(3) * c.record_width * c.B * 4);
    d_rope.copy_from_host(ref_rope.data(), static_cast<std::size_t>(3) * c.record_width * c.B * 4);
    DeviceBuffer d_base(static_cast<std::size_t>(c.B) * 4);
    d_base.copy_from_host(ref_base.data(), static_cast<std::size_t>(c.B) * 4);

    DeviceBuffer d_query(query_count * 2);
    d_query.copy_from_host(std::vector<std::uint16_t>(query_count, kSentinelBf16).data(), query_count * 2);
    DeviceBuffer d_index_query(index_query_count * 2);
    d_index_query.copy_from_host(
        std::vector<std::uint16_t>(index_query_count, kSentinelBf16).data(), index_query_count * 2);
    DeviceBuffer d_k(k_count * 2);
    d_k.copy_from_host(std::vector<std::uint16_t>(k_count, kSentinelBf16).data(), k_count * 2);
    DeviceBuffer d_v(k_count * 2);
    d_v.copy_from_host(std::vector<std::uint16_t>(k_count, kSentinelBf16).data(), k_count * 2);
    DeviceBuffer d_index(index_count * 2);
    d_index.copy_from_host(std::vector<std::uint16_t>(index_count, kSentinelBf16).data(), index_count * 2);

    QsaProjectionView projected{
        tensor(d_proj_query.p, c.activation, {kHeadDim, kQHeads, c.T, c.B}),
        tensor(d_proj_key.p, c.activation, {kHeadDim, kKvHeads, c.T, c.B}),
        tensor(d_proj_value.p, c.activation, {kHeadDim, kKvHeads, c.T, c.B}),
        tensor(d_proj_index_query.p, c.activation, {kIndexDim, kIndexHeads, c.T, c.B}),
        tensor(d_proj_index_key.p, c.activation, {kIndexDim, 1, c.T, c.B})};
    QsaPrepareParameters parameters{
        tensor(d_norm_query.p, DType::BF16, {kHeadDim}),
        tensor(d_norm_key.p, DType::BF16, {kHeadDim}),
        tensor(d_norm_index_query.p, DType::BF16, {kIndexDim}),
        tensor(d_norm_index_key.p, DType::BF16, {kIndexDim}),
        kEpsilon, kRotary, kTheta};
    QsaBatchView batch{
        tensor(d_positions.p, DType::I32, {c.T, c.axes, c.B}),
        tensor(d_valid.p, DType::I32, {c.B}),
        tensor(d_table_rows.p, DType::I32, {c.B})};
    QsaTailStateView before{
        tensor(d_tail_keys.p, DType::BF16, {kIndexDim, kCompress, c.B}),
        tensor(d_tail_rope.p, DType::I32, {3, c.B})};
    QsaCacheView cache{
        tensor(d_k.p, DType::BF16, {kHeadDim, kQsaTokenPageSize, kKvHeads, c.physical_pages}),
        tensor(d_v.p, DType::FP16, {kHeadDim, kQsaTokenPageSize, kKvHeads, c.physical_pages}),
        tensor(d_index.p, DType::BF16, {kIndexDim, kQsaIndexBlockPageSize, 1, c.index_pages}),
        tensor(d_page_table.p, DType::I32, {c.logical_pages, static_cast<std::int32_t>(c.table_rows.size())})};
    QsaRawRecordWriteView records{
        tensor(d_records.p, DType::BF16, {kIndexDim, c.record_width, c.B}),
        tensor(d_rope.p, DType::I32, {3, c.record_width, c.B}),
        tensor(d_base.p, DType::I32, {c.B})};
    Tensor query = tensor(d_query.p, c.activation, {kHeadDim, kQHeads, c.T, c.B});
    Tensor index_query = tensor(d_index_query.p, c.activation, {kIndexDim, kIndexHeads, c.T, c.B});

    ops::qsa_prepare(projected, parameters, batch, before, cache, arena, records, query, index_query,
                     stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_pointwise(c.label + " query",
                                 read_activation(d_query.p, query_count, c.activation), ref_query,
                                 kCriterion);
    failures += verify_pointwise(c.label + " index_query",
                                 read_activation(d_index_query.p, index_query_count, c.activation),
                                 ref_index_query, kCriterion);
    failures += verify_pointwise(c.label + " key cache", from_device_bf16(d_k.p, k_count), ref_k,
                                 kCriterion);
    failures += verify_pointwise(c.label + " index cache", from_device_bf16(d_index.p, index_count),
                                 ref_index, kCriterion);
    failures += verify_exact((c.label + " value cache").c_str(), from_device<std::uint16_t>(d_v.p, k_count), ref_v);
    failures += verify_exact((c.label + " record keys").c_str(), from_device<std::uint16_t>(d_records.p, record_count),
                             ref_records);
    failures += verify_exact((c.label + " record rope").c_str(), from_device<std::int32_t>(d_rope.p, 3 * c.record_width * c.B),
                             ref_rope);
    failures += verify_exact((c.label + " record base").c_str(), from_device<std::int32_t>(d_base.p, static_cast<std::size_t>(c.B)),
                             ref_base);
    return failures;
}

struct IndexerCase {
    std::string label;
    DType activation = DType::BF16;
    std::int32_t T;
    std::int32_t B;
    std::int32_t axes;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> valid;
    std::vector<std::int32_t> table_rows;
    std::int32_t logical_pages;
    std::vector<std::int32_t> page_table;
    std::vector<std::uint16_t> index_query;
    std::vector<std::uint16_t> index_keys;
    std::int32_t index_pages;
};

int run_indexer(const IndexerCase& c, cudaStream_t stream, WorkspaceArena& arena) {
    const std::size_t counts_count = static_cast<std::size_t>(c.T) * c.B;
    const std::size_t selection_count = static_cast<std::size_t>(kCapacity) * c.T * c.B;

    std::vector<std::int32_t> ref_counts(counts_count, 0);
    std::vector<std::int32_t> ref_selection(selection_count, -1);

    for (std::int32_t b = 0; b < c.B; ++b) {
        for (std::int32_t t = 0; t < c.T; ++t) {
            if (t >= c.valid[static_cast<std::size_t>(b)]) continue;
            const std::int32_t position =
                c.positions[static_cast<std::size_t>(t * c.axes * c.B + b)];
            const std::int32_t complete = (position + 1) / kCompress;
            ref_counts[static_cast<std::size_t>(t + c.T * b)] = std::min(complete, kCapacity);
            if (complete <= kCapacity) {
                for (std::int32_t g = 0; g < complete; ++g) {
                    ref_selection[static_cast<std::size_t>(kCapacity * (t + c.T * b) + g)] = g;
                }
                continue;
            }
            std::vector<std::pair<double, std::int32_t>> scored;
            for (std::int32_t g = 0; g < complete; ++g) {
                const std::int32_t logical_page = g / kQsaIndexBlockPageSize;
                const std::int32_t row = g % kQsaIndexBlockPageSize;
                const std::int32_t physical =
                    c.page_table[static_cast<std::size_t>(
                        logical_page + c.logical_pages * c.table_rows[static_cast<std::size_t>(b)])];
                double score = 0.0;
                for (std::int32_t h = 0; h < kIndexHeads; ++h) {
                    double dot = 0.0;
                    for (std::int32_t d = 0; d < kIndexDim; ++d) {
                        dot += act_double(c.index_query[static_cast<std::size_t>(d2i(d, h, t, b, kIndexDim, kIndexHeads, c.T, c.B))], c.activation) *
                               bf16_double(c.index_keys[static_cast<std::size_t>(
                                   d + static_cast<std::size_t>(kIndexDim) *
                                       (static_cast<std::size_t>(row) +
                                        static_cast<std::size_t>(kQsaIndexBlockPageSize) *
                                            static_cast<std::size_t>(physical)))]);
                    }
                    score += std::max(0.0, dot);
                }
                scored.emplace_back(score, g);
            }
            std::sort(scored.begin(), scored.end(),
                     [](const std::pair<double, std::int32_t>& a,
                        const std::pair<double, std::int32_t>& b2) {
                         if (a.first != b2.first) return a.first > b2.first;
                         return a.second < b2.second;
                     });
            scored.resize(static_cast<std::size_t>(kCapacity));
            std::sort(scored.begin(), scored.end(),
                     [](const std::pair<double, std::int32_t>& a,
                        const std::pair<double, std::int32_t>& b2) { return a.second < b2.second; });
            for (std::size_t i = 0; i < scored.size(); ++i) {
                ref_selection[static_cast<std::size_t>(kCapacity * (t + c.T * b) + i)] =
                    scored[i].second;
            }
        }
    }

    DeviceBuffer d_index_query = to_device(c.index_query);
    DeviceBuffer d_index_keys = to_device(c.index_keys);
    DeviceBuffer d_positions = to_device_i32(c.positions);
    DeviceBuffer d_valid = to_device_i32(c.valid);
    DeviceBuffer d_table_rows = to_device_i32(c.table_rows);
    DeviceBuffer d_page_table = to_device_i32(c.page_table);
    DeviceBuffer d_selection(selection_count * 4);
    d_selection.fill(0xFF);
    DeviceBuffer d_counts(counts_count * 4);
    d_counts.fill(0xFF);

    ConstQsaIndexCacheView cache{
        tensor(d_index_keys.p, DType::BF16, {kIndexDim, kQsaIndexBlockPageSize, 1, c.index_pages}),
        tensor(d_page_table.p, DType::I32, {c.logical_pages, static_cast<std::int32_t>(c.table_rows.size())})};
    QsaBatchView batch{
        tensor(d_positions.p, DType::I32, {c.T, c.axes, c.B}),
        tensor(d_valid.p, DType::I32, {c.B}),
        tensor(d_table_rows.p, DType::I32, {c.B})};
    QsaBlockSelection selected{
        tensor(d_selection.p, DType::I32, {kCapacity, c.T, c.B}),
        tensor(d_counts.p, DType::I32, {c.T, c.B})};

    ops::qsa_indexer(tensor(d_index_query.p, c.activation, {kIndexDim, kIndexHeads, c.T, c.B}), cache,
                     batch, 4096, arena, selected, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact((c.label + " counts").c_str(), from_device<std::int32_t>(d_counts.p, counts_count),
                             ref_counts);
    failures += verify_exact((c.label + " selection").c_str(),
                             from_device<std::int32_t>(d_selection.p, selection_count), ref_selection);
    return failures;
}

struct GqaCase {
    std::string label;
    DType activation = DType::BF16;
    std::int32_t T;
    std::int32_t B;
    std::int32_t axes;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> valid;
    std::vector<std::int32_t> table_rows;
    std::int32_t logical_pages;
    std::vector<std::int32_t> page_table;
    std::vector<std::uint16_t> query;
    std::vector<std::uint16_t> gate;
    std::vector<std::uint16_t> k_pages;
    std::vector<std::uint16_t> v_pages;
    std::vector<std::int32_t> selection;
    std::vector<std::int32_t> counts;
    std::int32_t physical_pages;
    std::int32_t index_pages;
};

int run_sparse_gqa(const GqaCase& c, cudaStream_t stream, WorkspaceArena& arena) {
    const std::size_t out_count = static_cast<std::size_t>(kHeadDim) * kQHeads * c.T * c.B;
    std::vector<double> ref(out_count, 0.0);

    for (std::int32_t b = 0; b < c.B; ++b) {
        for (std::int32_t t = 0; t < c.T; ++t) {
            if (t >= c.valid[static_cast<std::size_t>(b)]) continue;
            const std::int32_t position = c.positions[static_cast<std::size_t>(t * c.axes * c.B + b)];
            const std::int32_t complete = (position + 1) / kCompress;
            const std::int32_t count = c.counts[static_cast<std::size_t>(t + c.T * b)];
            std::vector<std::int32_t> j;
            for (std::int32_t i = 0; i < count; ++i) {
                const std::int32_t g = c.selection[static_cast<std::size_t>(
                    kCapacity * (t + c.T * b) + i)];
                for (std::int32_t k2 = 0; k2 < kCompress; ++k2) j.push_back(4 * g + k2);
            }
            for (std::int32_t token = 4 * complete; token <= position; ++token) j.push_back(token);
            if (j.empty()) continue;
            const std::int32_t table_row = c.table_rows[static_cast<std::size_t>(b)];
            std::vector<double> scores(j.size());
            for (std::int32_t h = 0; h < kQHeads; ++h) {
                const std::int32_t kv_head = h / (kQHeads / kKvHeads);
                for (std::size_t i = 0; i < j.size(); ++i) {
                    const std::int32_t token = j[i];
                    const std::int32_t physical =
                        c.page_table[static_cast<std::size_t>(
                            token / kQsaTokenPageSize + c.logical_pages * table_row)];
                    const std::int32_t row = token % kQsaTokenPageSize;
                    double dot = 0.0;
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        dot += act_double(c.query[static_cast<std::size_t>(d2i(d, h, t, b, kHeadDim, kQHeads, c.T, c.B))], c.activation) *
                               bf16_double(c.k_pages[static_cast<std::size_t>(
                                   d + static_cast<std::size_t>(kHeadDim) *
                                       (static_cast<std::size_t>(row) +
                                        static_cast<std::size_t>(kQsaTokenPageSize) *
                                            (static_cast<std::size_t>(kv_head) +
                                             static_cast<std::size_t>(kKvHeads) *
                                                 static_cast<std::size_t>(physical))))]);
                    }
                    scores[i] = dot / 16.0;
                }
                double maximum = -std::numeric_limits<double>::infinity();
                for (double s : scores) maximum = std::max(maximum, s);
                double denominator = 0.0;
                std::vector<double> probability(scores.size());
                for (std::size_t i = 0; i < scores.size(); ++i) {
                    probability[i] = std::exp(scores[i] - maximum);
                    denominator += probability[i];
                }
                for (double& pr : probability) pr /= denominator;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    double attention = 0.0;
                    for (std::size_t i = 0; i < j.size(); ++i) {
                        const std::int32_t token = j[i];
                        const std::int32_t physical =
                            c.page_table[static_cast<std::size_t>(
                                token / kQsaTokenPageSize + c.logical_pages * table_row)];
                        const std::int32_t row = token % kQsaTokenPageSize;
                        attention += probability[i] *
                                     fp16_double(c.v_pages[static_cast<std::size_t>(
                                         d + static_cast<std::size_t>(kHeadDim) *
                                             (static_cast<std::size_t>(row) +
                                              static_cast<std::size_t>(kQsaTokenPageSize) *
                                                  (static_cast<std::size_t>(kv_head) +
                                                   static_cast<std::size_t>(kKvHeads) *
                                            static_cast<std::size_t>(physical))))]);
                    }
                    const double gate = act_double(c.gate[static_cast<std::size_t>(d2i(d, h, t, b, kHeadDim, kQHeads, c.T, c.B))], c.activation);
                    ref[static_cast<std::size_t>(d2i(d, h, t, b, kHeadDim, kQHeads, c.T, c.B))] =
                        act_double(act_bits(attention * (1.0 / (1.0 + std::exp(-gate))),
                                            c.activation),
                                   c.activation);
                }
            }
        }
    }

    DeviceBuffer d_query = to_device(c.query);
    DeviceBuffer d_gate = to_device(c.gate);
    DeviceBuffer d_k = to_device(c.k_pages);
    DeviceBuffer d_v = to_device(c.v_pages);
    DeviceBuffer d_positions = to_device_i32(c.positions);
    DeviceBuffer d_valid = to_device_i32(c.valid);
    DeviceBuffer d_table_rows = to_device_i32(c.table_rows);
    DeviceBuffer d_page_table = to_device_i32(c.page_table);
    DeviceBuffer d_selection = to_device_i32(c.selection);
    DeviceBuffer d_counts = to_device_i32(c.counts);
    DeviceBuffer d_out(out_count * 2);
    d_out.fill(0);

    DeviceBuffer d_index = filled_device(static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize *
                                         c.index_pages * 2, 0);
    QsaCacheView cache{
        tensor(d_k.p, DType::BF16, {kHeadDim, kQsaTokenPageSize, kKvHeads, c.physical_pages}),
        tensor(d_v.p, DType::FP16, {kHeadDim, kQsaTokenPageSize, kKvHeads, c.physical_pages}),
        tensor(d_index.p, DType::BF16, {kIndexDim, kQsaIndexBlockPageSize, 1, c.index_pages}),
        tensor(d_page_table.p, DType::I32, {c.logical_pages, static_cast<std::int32_t>(c.table_rows.size())})};
    QsaBatchView batch{
        tensor(d_positions.p, DType::I32, {c.T, c.axes, c.B}),
        tensor(d_valid.p, DType::I32, {c.B}),
        tensor(d_table_rows.p, DType::I32, {c.B})};
    QsaBlockSelection selected{
        tensor(d_selection.p, DType::I32, {kCapacity, c.T, c.B}),
        tensor(d_counts.p, DType::I32, {c.T, c.B})};
    Tensor out = tensor(d_out.p, c.activation, {kHeadDim, kQHeads, c.T, c.B});

    ops::qsa_sparse_gqa(tensor(d_query.p, c.activation, {kHeadDim, kQHeads, c.T, c.B}),
                        tensor(d_gate.p, c.activation, {kHeadDim, kQHeads, c.T, c.B}), cache, batch,
                        selected, 4096, arena, out, stream);
    cuda_synchronize(stream);

    return verify_pointwise(c.label, read_activation(d_out.p, out_count, c.activation), ref,
                            kCriterion);
}

struct TailCase {
    std::string label;
    std::int32_t B;
    std::int32_t width;
    std::vector<std::int32_t> base;
    std::vector<std::int32_t> valid;
    std::vector<std::uint16_t> before_keys;
    std::vector<std::int32_t> before_rope;
    std::vector<std::uint16_t> record_keys;
    std::vector<std::int32_t> record_rope;
    std::vector<std::uint16_t> ref_keys;
    std::vector<std::int32_t> ref_rope;
};

int run_tail_extract(const TailCase& c, cudaStream_t stream) {
    DeviceBuffer d_before_keys = to_device(c.before_keys);
    DeviceBuffer d_before_rope = to_device_i32(c.before_rope);
    DeviceBuffer d_record_keys = to_device(c.record_keys);
    DeviceBuffer d_record_rope = to_device_i32(c.record_rope);
    DeviceBuffer d_base = to_device_i32(c.base);
    DeviceBuffer d_valid = to_device_i32(c.valid);
    DeviceBuffer d_after_keys(static_cast<std::size_t>(kIndexDim) * kCompress * c.B * 2);
    d_after_keys.fill(0);
    DeviceBuffer d_after_rope(static_cast<std::size_t>(3) * c.B * 4);
    d_after_rope.fill(0);

    ConstQsaTailStateView before{
        tensor(d_before_keys.p, DType::BF16, {kIndexDim, kCompress, c.B}),
        tensor(d_before_rope.p, DType::I32, {3, c.B})};
    QsaRawRecordReadView records{
        tensor(d_record_keys.p, DType::BF16, {kIndexDim, c.width, c.B}),
        tensor(d_record_rope.p, DType::I32, {3, c.width, c.B})};
    QsaTailStateView after{
        tensor(d_after_keys.p, DType::BF16, {kIndexDim, kCompress, c.B}),
        tensor(d_after_rope.p, DType::I32, {3, c.B})};

    ops::qsa_tail_extract(before, records, tensor(d_base.p, DType::I32, {c.B}),
                          tensor(d_valid.p, DType::I32, {c.B}), after, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact((c.label + " keys").c_str(),
                             from_device<std::uint16_t>(d_after_keys.p,
                                                        static_cast<std::size_t>(kIndexDim) * kCompress * c.B),
                             c.ref_keys);
    failures += verify_exact((c.label + " rope").c_str(),
                             from_device<std::int32_t>(d_after_rope.p, static_cast<std::size_t>(3) * c.B),
                             c.ref_rope);
    return failures;
}

PrepareCase fp16_prepare(PrepareCase c) {
    c.activation = DType::FP16;
    c.label += " (f16)";
    c.proj_query = to_activation(std::move(c.proj_query), DType::FP16);
    c.proj_key = to_activation(std::move(c.proj_key), DType::FP16);
    c.proj_value = to_activation(std::move(c.proj_value), DType::FP16);
    c.proj_index_query = to_activation(std::move(c.proj_index_query), DType::FP16);
    c.proj_index_key = to_activation(std::move(c.proj_index_key), DType::FP16);
    return c;
}

IndexerCase fp16_indexer(IndexerCase c) {
    c.activation = DType::FP16;
    c.label += " (f16)";
    c.index_query = to_activation(std::move(c.index_query), DType::FP16);
    return c;
}

GqaCase fp16_gqa(GqaCase c) {
    c.activation = DType::FP16;
    c.label += " (f16)";
    c.query = to_activation(std::move(c.query), DType::FP16);
    c.gate = to_activation(std::move(c.gate), DType::FP16);
    return c;
}

int run_prepare_cases(cudaStream_t stream, WorkspaceArena& arena) {
    int failures = 0;
    const std::int32_t logical_pages = 64;

    PrepareCase p1;
    p1.label = "qsa_prepare prefill";
    p1.T = 8;
    p1.B = 2;
    p1.axes = 3;
    p1.logical_pages = logical_pages;
    p1.record_width = 8;
    p1.physical_pages = 128;
    p1.index_pages = 128;
    p1.valid = {8, 6};
    p1.table_rows = {0, 1};
    p1.positions.assign(static_cast<std::size_t>(8) * 3 * 2, 0);
    for (std::int32_t t = 0; t < 8; ++t) {
        for (std::int32_t axis = 0; axis < 3; ++axis) {
            p1.positions[static_cast<std::size_t>((t * 3 + axis) * 2 + 0)] = t + axis;
            p1.positions[static_cast<std::size_t>((t * 3 + axis) * 2 + 1)] = 3 + t + axis * 2;
        }
    }
    p1.page_table.resize(static_cast<std::size_t>(logical_pages) * 2);
    for (std::int32_t page = 0; page < logical_pages; ++page) {
        for (std::int32_t row = 0; row < 2; ++row) {
            p1.page_table[static_cast<std::size_t>(page + logical_pages * row)] = page * 2 + row;
        }
    }
    p1.proj_query = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQHeads * 8 * 2, 11U, -2.0F, 2.0F);
    p1.proj_key = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kKvHeads * 8 * 2, 12U, -2.0F, 2.0F);
    p1.proj_value = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kKvHeads * 8 * 2, 13U, -2.0F, 2.0F);
    p1.proj_index_query = bf16_inputs(static_cast<std::size_t>(kIndexDim) * kIndexHeads * 8 * 2, 14U, -2.0F, 2.0F);
    p1.proj_index_key = bf16_inputs(static_cast<std::size_t>(kIndexDim) * 1 * 8 * 2, 15U, -2.0F, 2.0F);
    p1.norm_query = bf16_inputs(kHeadDim, 16U, 0.5F, 2.0F);
    p1.norm_key = bf16_inputs(kHeadDim, 17U, 0.5F, 2.0F);
    p1.norm_index_query = bf16_inputs(kIndexDim, 18U, 0.5F, 2.0F);
    p1.norm_index_key = bf16_inputs(kIndexDim, 19U, 0.5F, 2.0F);
    p1.tail_keys.assign(static_cast<std::size_t>(kIndexDim) * kCompress * 2, 0);
    for (std::int32_t d = 0; d < kIndexDim; ++d) {
        for (std::int32_t j = 0; j < 3; ++j) {
            p1.tail_keys[static_cast<std::size_t>(d + kIndexDim * (j + 4 * 1))] = double_bf16(1.0 + 0.5 * j);
        }
    }
    p1.tail_rope = {0, 0, 0, 1, 2, 3};
    failures += run_prepare(p1, stream, arena);
    failures += run_prepare(fp16_prepare(p1), stream, arena);

    PrepareCase p2;
    p2.label = "qsa_prepare block boundary";
    p2.T = 4;
    p2.B = 1;
    p2.axes = 1;
    p2.logical_pages = 64;
    p2.record_width = 4;
    p2.physical_pages = 64;
    p2.index_pages = 64;
    p2.valid = {2};
    p2.table_rows = {0};
    p2.positions = {2, 3, 4, 5};
    p2.page_table.resize(64);
    for (std::int32_t page = 0; page < 64; ++page) p2.page_table[static_cast<std::size_t>(page)] = page;
    p2.proj_query = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQHeads * 4, 21U, -2.0F, 2.0F);
    p2.proj_key = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kKvHeads * 4, 22U, -2.0F, 2.0F);
    p2.proj_value = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kKvHeads * 4, 23U, -2.0F, 2.0F);
    p2.proj_index_query = bf16_inputs(static_cast<std::size_t>(kIndexDim) * kIndexHeads * 4, 24U, -2.0F, 2.0F);
    p2.proj_index_key = bf16_inputs(static_cast<std::size_t>(kIndexDim) * 4, 25U, -2.0F, 2.0F);
    p2.norm_query = bf16_inputs(kHeadDim, 26U, 0.5F, 2.0F);
    p2.norm_key = bf16_inputs(kHeadDim, 27U, 0.5F, 2.0F);
    p2.norm_index_query = bf16_inputs(kIndexDim, 28U, 0.5F, 2.0F);
    p2.norm_index_key = bf16_inputs(kIndexDim, 29U, 0.5F, 2.0F);
    p2.tail_keys.assign(static_cast<std::size_t>(kIndexDim) * kCompress, 0);
    for (std::int32_t d = 0; d < kIndexDim; ++d) {
        for (std::int32_t j = 0; j < 2; ++j) {
            p2.tail_keys[static_cast<std::size_t>(d + kIndexDim * j)] = double_bf16(-1.0 - 0.5 * j);
        }
    }
    p2.tail_rope = {1, 2, 3};
    failures += run_prepare(p2, stream, arena);
    failures += run_prepare(fp16_prepare(p2), stream, arena);

    return failures;
}

int run_indexer_cases(cudaStream_t stream, WorkspaceArena& arena) {
    int failures = 0;

    IndexerCase i1;
    i1.label = "qsa_indexer all blocks";
    i1.T = 2;
    i1.B = 1;
    i1.axes = 3;
    i1.logical_pages = 64;
    i1.index_pages = 64;
    i1.valid = {2};
    i1.table_rows = {0};
    i1.positions = {100, 0, 0, 103, 0, 0};
    i1.page_table.resize(64);
    for (std::int32_t page = 0; page < 64; ++page) i1.page_table[static_cast<std::size_t>(page)] = page;
    i1.index_query = integer_inputs(static_cast<std::size_t>(kIndexDim) * kIndexHeads * 2, 31U);
    i1.index_keys = integer_inputs(static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * 64, 32U);
    failures += run_indexer(i1, stream, arena);
    failures += run_indexer(fp16_indexer(i1), stream, arena);

    IndexerCase i2;
    i2.label = "qsa_indexer top-512";
    i2.T = 1;
    i2.B = 1;
    i2.axes = 1;
    i2.logical_pages = 64;
    i2.index_pages = 64;
    i2.valid = {1};
    i2.table_rows = {0};
    i2.positions = {2051};
    i2.page_table.resize(64);
    for (std::int32_t page = 0; page < 64; ++page) i2.page_table[static_cast<std::size_t>(page)] = page;
    i2.index_query = integer_inputs(static_cast<std::size_t>(kIndexDim) * kIndexHeads, 41U);
    i2.index_keys = integer_inputs(static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * 64, 42U);
    failures += run_indexer(i2, stream, arena);
    failures += run_indexer(fp16_indexer(i2), stream, arena);

    IndexerCase i3;
    i3.label = "qsa_indexer tie rule";
    i3.T = 1;
    i3.B = 1;
    i3.axes = 1;
    i3.logical_pages = 64;
    i3.index_pages = 64;
    i3.valid = {1};
    i3.table_rows = {0};
    i3.positions = {2051};
    i3.page_table.resize(64);
    for (std::int32_t page = 0; page < 64; ++page) i3.page_table[static_cast<std::size_t>(page)] = page;
    i3.index_query.assign(static_cast<std::size_t>(kIndexDim) * kIndexHeads, double_bf16(1.0));
    i3.index_keys.assign(static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * 64, double_bf16(0.0));
    i3.index_keys[static_cast<std::size_t>(0 + kIndexDim * (4 + kQsaIndexBlockPageSize * 31))] = double_bf16(1.0);
    i3.index_keys[static_cast<std::size_t>(0 + kIndexDim * (0 + kQsaIndexBlockPageSize * 32))] = double_bf16(1.0);
    failures += run_indexer(i3, stream, arena);
    failures += run_indexer(fp16_indexer(i3), stream, arena);

    IndexerCase i5;
    i5.label = "qsa_indexer distinct scores";
    i5.T = 1;
    i5.B = 1;
    i5.axes = 1;
    i5.logical_pages = 64;
    i5.index_pages = 64;
    i5.valid = {1};
    i5.table_rows = {0};
    i5.positions = {2399};
    i5.page_table.resize(64);
    for (std::int32_t page = 0; page < 64; ++page) i5.page_table[static_cast<std::size_t>(page)] = page;
    i5.index_query.assign(static_cast<std::size_t>(kIndexDim) * kIndexHeads, double_bf16(0.0));
    i5.index_query[0] = double_bf16(1.0);
    i5.index_keys.assign(static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * 64,
                         double_bf16(0.0));
    for (std::int32_t g = 0; g < 600; ++g) {
        const float value = (1.0F + static_cast<float>(g % 256) / 256.0F) *
                            std::ldexp(1.0F, g / 256);
        i5.index_keys[static_cast<std::size_t>(g) * kIndexDim] = double_bf16(value);
    }
    failures += run_indexer(i5, stream, arena);
    failures += run_indexer(fp16_indexer(i5), stream, arena);

    IndexerCase i6;
    i6.label = "qsa_indexer scored rows";
    i6.T = 2;
    i6.B = 1;
    i6.axes = 1;
    i6.logical_pages = 64;
    i6.index_pages = 64;
    i6.valid = {2};
    i6.table_rows = {0};
    i6.positions = {2051, 2052};
    i6.page_table.resize(64);
    for (std::int32_t page = 0; page < 64; ++page) i6.page_table[static_cast<std::size_t>(page)] = page;
    i6.index_query = integer_inputs(static_cast<std::size_t>(kIndexDim) * kIndexHeads * 2, 61U);
    i6.index_keys = integer_inputs(static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * 64, 62U);
    failures += run_indexer(i6, stream, arena);
    failures += run_indexer(fp16_indexer(i6), stream, arena);

    IndexerCase i7;
    i7.label = "qsa_indexer uniform scores";
    i7.T = 1;
    i7.B = 1;
    i7.axes = 1;
    i7.logical_pages = 64;
    i7.index_pages = 64;
    i7.valid = {1};
    i7.table_rows = {0};
    i7.positions = {2051};
    i7.page_table.resize(64);
    for (std::int32_t page = 0; page < 64; ++page) i7.page_table[static_cast<std::size_t>(page)] = page;
    i7.index_query.assign(static_cast<std::size_t>(kIndexDim) * kIndexHeads, double_bf16(1.0));
    i7.index_keys.assign(static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * 64,
                         double_bf16(0.0));
    failures += run_indexer(i7, stream, arena);
    failures += run_indexer(fp16_indexer(i7), stream, arena);

    // A wave width smaller than the token grid exercises the multi-wave scratch offset: the score
    // and select kernels must index the wave-local scratch with (tb - token_base), not the global
    // token index. With the default 1 MiB arena the 4 MiB wave cap always covers a small grid, so
    // this case builds an arena sized for exactly two tokens.
    {
        IndexerCase i8;
        i8.label = "qsa_indexer multi-wave";
        i8.T = 64;
        i8.B = 1;
        i8.axes = 1;
        i8.logical_pages = 64;
        i8.index_pages = 64;
        i8.valid = {64};
        i8.table_rows = {0};
        i8.positions.resize(64);
        for (std::int32_t t = 0; t < 64; ++t) { i8.positions[static_cast<std::size_t>(t)] = 2051 + t; }
        i8.page_table.resize(64);
        for (std::int32_t page = 0; page < 64; ++page) i8.page_table[static_cast<std::size_t>(page)] = page;
        i8.index_query = integer_inputs(static_cast<std::size_t>(kIndexDim) * kIndexHeads * 64, 71U);
        i8.index_keys = integer_inputs(
            static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * 64, 72U);
        const std::size_t score_bytes =
            static_cast<std::size_t>(i8.index_pages * kQsaIndexBlockPageSize) * sizeof(float);
        // The wave scratch is exactly two tokens wide; the remaining address space is a guard so a
        // regression that indexes the scratch with the global token (rather than tb - token_base)
        // is caught even when the score and select kernels agree with each other.
        DeviceBuffer guard(score_bytes * 64);
        guard.fill(0x7f);
        DeviceArena wave_arena(DeviceSpan{guard.p, score_bytes * 2});
        failures += run_indexer(i8, stream, wave_arena);
        failures += run_indexer(fp16_indexer(i8), stream, wave_arena);
        const auto untouched = from_device<std::uint8_t>(
            static_cast<const std::uint8_t*>(guard.p) + score_bytes * 2,
            guard.bytes - score_bytes * 2);
        if (std::any_of(untouched.begin(), untouched.end(),
                        [](std::uint8_t value) { return value != 0x7f; })) {
            std::cerr << "qsa_indexer multi-wave: the score scratch escaped its wave window\n";
            ++failures;
        }
    }

    // More than 2^17 block groups: the selection key must keep the whole 20-bit block field, and
    // the score/select wave path must survive a grid larger than one wave.
    {
        IndexerCase i9;
        i9.label = "qsa_indexer 20-bit key";
        i9.T = 1;
        i9.B = 1;
        i9.axes = 1;
        i9.logical_pages = 8750;
        i9.index_pages = 8750;
        i9.valid = {1};
        i9.table_rows = {0};
        i9.positions = {559999};
        i9.page_table.resize(8750);
        for (std::int32_t page = 0; page < 8750; ++page) i9.page_table[static_cast<std::size_t>(page)] = page;
        i9.index_query = integer_inputs(static_cast<std::size_t>(kIndexDim) * kIndexHeads, 81U);
        i9.index_keys = integer_inputs(
            static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * 8750, 82U);
        failures += run_indexer(i9, stream, arena);
    }

    IndexerCase i4;
    i4.label = "qsa_indexer invalid row";    i4.T = 2;
    i4.B = 1;
    i4.axes = 1;
    i4.logical_pages = 64;
    i4.index_pages = 64;
    i4.valid = {1};
    i4.table_rows = {0};
    i4.positions = {100, 104};
    i4.page_table.resize(64);
    for (std::int32_t page = 0; page < 64; ++page) i4.page_table[static_cast<std::size_t>(page)] = page;
    i4.index_query = integer_inputs(static_cast<std::size_t>(kIndexDim) * kIndexHeads * 2, 51U);
    i4.index_keys = integer_inputs(static_cast<std::size_t>(kIndexDim) * kQsaIndexBlockPageSize * 64, 52U);
    failures += run_indexer(i4, stream, arena);
    failures += run_indexer(fp16_indexer(i4), stream, arena);

    return failures;
}

int run_gqa_cases(cudaStream_t stream, WorkspaceArena& arena) {
    int failures = 0;

    GqaCase g1;
    g1.label = "qsa_sparse_gqa tails";
    g1.T = 3;
    g1.B = 1;
    g1.axes = 1;
    g1.logical_pages = 2;
    g1.physical_pages = 2;
    g1.index_pages = 2;
    g1.valid = {3};
    g1.table_rows = {0};
    g1.positions = {7, 8, 10};
    g1.page_table = {0, 1};
    g1.query = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQHeads * 3, 61U, -2.0F, 2.0F);
    g1.gate = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQHeads * 3, 62U, -2.0F, 2.0F);
    g1.k_pages = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQsaTokenPageSize * kKvHeads * 2, 63U, -2.0F, 2.0F);
    g1.v_pages = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQsaTokenPageSize * kKvHeads * 2, 64U, -2.0F, 2.0F);
    g1.selection.assign(static_cast<std::size_t>(kCapacity) * 3, kSentinelI32);
    g1.counts.assign(static_cast<std::size_t>(3), 0);
    for (std::int32_t t = 0; t < 3; ++t) {
        const std::int32_t complete = (g1.positions[static_cast<std::size_t>(t)] + 1) / kCompress;
        g1.counts[static_cast<std::size_t>(t)] = complete;
        for (std::int32_t g = 0; g < complete; ++g) {
            g1.selection[static_cast<std::size_t>(kCapacity * t + g)] = g;
        }
    }
    failures += run_sparse_gqa(g1, stream, arena);
    failures += run_sparse_gqa(fp16_gqa(g1), stream, arena);

    GqaCase g2;
    g2.label = "qsa_sparse_gqa partial selection";
    g2.T = 1;
    g2.B = 1;
    g2.axes = 1;
    g2.logical_pages = 2;
    g2.physical_pages = 2;
    g2.index_pages = 2;
    g2.valid = {1};
    g2.table_rows = {0};
    g2.positions = {41};
    g2.page_table = {0, 1};
    g2.query = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQHeads, 71U, -2.0F, 2.0F);
    g2.gate = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQHeads, 72U, -2.0F, 2.0F);
    g2.k_pages = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQsaTokenPageSize * kKvHeads * 2, 73U, -2.0F, 2.0F);
    g2.v_pages = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQsaTokenPageSize * kKvHeads * 2, 74U, -2.0F, 2.0F);
    g2.selection.assign(kCapacity, kSentinelI32);
    g2.selection[0] = 1;
    g2.selection[1] = 4;
    g2.selection[2] = 7;
    g2.counts = {3};
    failures += run_sparse_gqa(g2, stream, arena);
    failures += run_sparse_gqa(fp16_gqa(g2), stream, arena);

    GqaCase g3;
    g3.label = "qsa_sparse_gqa empty set";
    g3.T = 2;
    g3.B = 1;
    g3.axes = 1;
    g3.logical_pages = 2;
    g3.physical_pages = 2;
    g3.index_pages = 2;
    g3.valid = {1};
    g3.table_rows = {0};
    g3.positions = {3, 4};
    g3.page_table = {0, 1};
    g3.query = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQHeads * 2, 81U, -2.0F, 2.0F);
    g3.gate = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQHeads * 2, 82U, -2.0F, 2.0F);
    g3.k_pages = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQsaTokenPageSize * kKvHeads * 2, 83U, -2.0F, 2.0F);
    g3.v_pages = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kQsaTokenPageSize * kKvHeads * 2, 84U, -2.0F, 2.0F);
    g3.selection.assign(static_cast<std::size_t>(kCapacity) * 2, kSentinelI32);
    g3.counts = {0, 0};
    failures += run_sparse_gqa(g3, stream, arena);
    failures += run_sparse_gqa(fp16_gqa(g3), stream, arena);

    return failures;
}

int run_tail_cases(cudaStream_t stream) {
    int failures = 0;

    TailCase e1;
    e1.label = "qsa_tail_extract cross-call";
    e1.B = 2;
    e1.width = 8;
    e1.base = {5, 4};
    e1.valid = {2, 5};
    e1.before_keys.assign(static_cast<std::size_t>(kIndexDim) * kCompress * 2, 0);
    e1.before_rope = {11, 12, 13, 21, 22, 23};
    for (std::int32_t d = 0; d < kIndexDim; ++d) {
        for (std::int32_t j = 0; j < 4; ++j) {
            e1.before_keys[static_cast<std::size_t>(d + kIndexDim * (j + 4 * 0))] = double_bf16(1.0 + j);
            e1.before_keys[static_cast<std::size_t>(d + kIndexDim * (j + 4 * 1))] = double_bf16(-1.0 - j);
        }
    }
    e1.record_keys.assign(static_cast<std::size_t>(kIndexDim) * e1.width * 2, 0);
    e1.record_rope.assign(static_cast<std::size_t>(3) * e1.width * 2, 0);
    for (std::int32_t b = 0; b < 2; ++b) {
        for (std::int32_t t = 0; t < e1.width; ++t) {
            for (std::int32_t d = 0; d < kIndexDim; ++d) {
                e1.record_keys[static_cast<std::size_t>(d + kIndexDim * (t + e1.width * b))] =
                    double_bf16(3.0 + t + b);
            }
            for (std::int32_t axis = 0; axis < 3; ++axis) {
                e1.record_rope[static_cast<std::size_t>(axis + 3 * (t + e1.width * b))] = 40 + axis + t + b;
            }
        }
    }
    e1.ref_keys.assign(static_cast<std::size_t>(kIndexDim) * kCompress * 2, 0);
    e1.ref_rope = {0, 0, 0, 0, 0, 0};
    for (std::int32_t b = 0; b < 2; ++b) {
        const std::int32_t F0 = e1.base[static_cast<std::size_t>(b)];
        const std::int32_t a = e1.valid[static_cast<std::size_t>(b)];
        const std::int32_t F1 = F0 + a;
        const std::int32_t r = F1 % kCompress;
        const std::int32_t s = F1 - r;
        for (std::int32_t j = 0; j < r; ++j) {
            const std::int32_t source = s + j;
            for (std::int32_t d = 0; d < kIndexDim; ++d) {
                if (source < F0) {
                    const std::int32_t column = source - (F0 - F0 % kCompress);
                    e1.ref_keys[static_cast<std::size_t>(d + kIndexDim * (j + 4 * b))] =
                        e1.before_keys[static_cast<std::size_t>(d + kIndexDim * (column + 4 * b))];
                } else {
                    e1.ref_keys[static_cast<std::size_t>(d + kIndexDim * (j + 4 * b))] =
                        e1.record_keys[static_cast<std::size_t>(d + kIndexDim * (source - F0 + e1.width * b))];
                }
            }
        }
        if (r != 0) {
            for (std::int32_t axis = 0; axis < 3; ++axis) {
                if (s < F0) {
                    e1.ref_rope[static_cast<std::size_t>(axis + 3 * b)] =
                        e1.before_rope[static_cast<std::size_t>(axis + 3 * b)];
                } else {
                    e1.ref_rope[static_cast<std::size_t>(axis + 3 * b)] =
                        e1.record_rope[static_cast<std::size_t>(axis + 3 * (s - F0 + e1.width * b))];
                }
            }
        }
    }
    failures += run_tail_extract(e1, stream);

    TailCase e2;
    e2.label = "qsa_tail_extract block boundary";
    e2.B = 1;
    e2.width = 4;
    e2.base = {2};
    e2.valid = {2};
    e2.before_keys.assign(static_cast<std::size_t>(kIndexDim) * kCompress, double_bf16(2.0));
    e2.before_rope = {5, 6, 7};
    e2.record_keys.assign(static_cast<std::size_t>(kIndexDim) * 4, double_bf16(3.0));
    e2.record_rope.assign(12, 9);
    e2.ref_keys.assign(static_cast<std::size_t>(kIndexDim) * kCompress, 0);
    e2.ref_rope = {0, 0, 0};
    failures += run_tail_extract(e2, stream);

    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    DeviceArena arena(1U << 20);

    int failures = 0;
    failures += run_prepare_cases(stream, arena);
    failures += run_indexer_cases(stream, arena);
    failures += run_gqa_cases(stream, arena);
    failures += run_tail_cases(stream);

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    if (failures == 0) std::cout << "qsa_test: PASS\n";
    return failures == 0 ? 0 : 1;
}
