#include "ninfer/ops/ple.h"

#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/ple/ple.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::ops {

PleRowAddress ple_row_address(const PleNf4Table& table, std::uint32_t global_row) {
    detail::require_ple_geometry(table.geometry);
    if (table.shard_count != kPleShards) {
        throw std::invalid_argument("ple: the table is not the registered 128-shard table");
    }
    if (global_row >= table.shard_count * table.geometry.rows) {
        throw std::invalid_argument("ple: the global row is outside the table");
    }
    return {static_cast<std::uint32_t>(global_row / table.geometry.rows),
            static_cast<std::uint32_t>(global_row % table.geometry.rows)};
}

namespace detail {
namespace {

constexpr int kLookupBlock = 80;

template <class A>
__global__ __launch_bounds__(kLookupBlock) void ple_lookup_kernel(
    std::uint32_t shard_rows, std::uint32_t table_rows, std::uint32_t row_bytes,
    std::uint32_t code_bytes_per_row, const float* divisors, const std::uint32_t* rows,
    const std::uint8_t* staged, A* embedding) {
    const std::size_t entry = blockIdx.x;
    const std::uint32_t global_row = rows[entry];
    const std::size_t column = entry / kPleHeads;
    const std::size_t head = entry % kPleHeads;
    A* out = embedding + column * (kPleFeatures * kPleHeads) + head * kPleFeatures;
    const std::uint8_t* source = staged + entry * row_bytes;

    if (global_row >= table_rows) {
        out[threadIdx.x * 2] = PleAct<A>::from_float(0.0F);
        out[threadIdx.x * 2 + 1] = PleAct<A>::from_float(0.0F);
        return;
    }

    const float divisor = divisors[global_row / shard_rows];
    const float scale = decode_nvfp4_e4m3(source[code_bytes_per_row + threadIdx.x / 8]);
    const float2 codes = decode_nvfp4_e2m1x2(source[threadIdx.x]);
    out[threadIdx.x * 2] = PleAct<A>::from_float(codes.x * scale / divisor);
    out[threadIdx.x * 2 + 1] = PleAct<A>::from_float(codes.y * scale / divisor);
}

} // namespace

void ple_lookup_launch(const ninfer::artifact::RowPagedNf4Geometry& geometry,
                       const float* divisors, const std::uint32_t* rows,
                       const std::uint8_t* staged_rows, const Tensor& embedding,
                       std::uint32_t shard_count, std::size_t entries, cudaStream_t stream) {
    if (embedding.dtype == DType::FP16) {
        ple_lookup_kernel<__half><<<static_cast<int>(entries), kLookupBlock, 0, stream>>>(
            static_cast<std::uint32_t>(geometry.rows),
            static_cast<std::uint32_t>(shard_count * geometry.rows),
            static_cast<std::uint32_t>(geometry.row_bytes),
            static_cast<std::uint32_t>(geometry.code_bytes_per_row), divisors, rows, staged_rows,
            static_cast<__half*>(embedding.data));
    } else {
        ple_lookup_kernel<__nv_bfloat16><<<static_cast<int>(entries), kLookupBlock, 0, stream>>>(
            static_cast<std::uint32_t>(geometry.rows),
            static_cast<std::uint32_t>(shard_count * geometry.rows),
            static_cast<std::uint32_t>(geometry.row_bytes),
            static_cast<std::uint32_t>(geometry.code_bytes_per_row), divisors, rows, staged_rows,
            static_cast<__nv_bfloat16*>(embedding.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace detail

void ple_lookup_rows(const PleNf4Table& table, const std::uint32_t* rows, std::size_t entries,
                     const std::uint8_t* staged_rows, Tensor& embedding, cudaStream_t stream) {
    detail::require_ple_geometry(table.geometry);
    if (table.shard_count != kPleShards) {
        throw std::invalid_argument("ple: the table is not the registered 128-shard table");
    }
    if (table.divisors == nullptr) {
        throw std::invalid_argument("ple: the table carries no divisor table");
    }
    if (entries == 0) {
        throw std::invalid_argument("ple: the lookup needs at least one row");
    }
    if (!detail::aligned_to(table.divisors, 16)) {
        throw std::invalid_argument("ple: divisors must be 16-byte aligned");
    }
    if (!detail::aligned_to(rows, 16)) {
        throw std::invalid_argument("ple: rows must be 16-byte aligned");
    }
    if (!detail::aligned_to(staged_rows, 16)) {
        throw std::invalid_argument("ple: staged_rows must be 16-byte aligned");
    }

    const std::int32_t tokens = embedding.ne[1];
    const std::int32_t batch = embedding.ne[2];
    detail::require_activation(embedding, "embedding");
    detail::require_shape(embedding, "embedding", embedding.dtype,
                          {static_cast<std::int32_t>(kPleFeatures * kPleHeads), tokens, batch});
    if (tokens <= 0 || batch <= 0) {
        throw std::invalid_argument("ple: the lookup needs a positive column extent");
    }
    if (entries != kPleHeads * static_cast<std::size_t>(tokens) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("ple: the entry count does not match the embedding extent");
    }

    detail::ple_lookup_launch(table.geometry, table.divisors, rows, staged_rows, embedding,
                              table.shard_count, entries, stream);
}

} // namespace ninfer::ops
