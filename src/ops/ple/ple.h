#pragma once

#include "ninfer/ops/ple.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

inline bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1U)) == 0U;
}

inline bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const std::uintptr_t left = reinterpret_cast<std::uintptr_t>(lhs.data);
    const std::uintptr_t right = reinterpret_cast<std::uintptr_t>(rhs.data);
    return left < right + rhs.bytes() && right < left + lhs.bytes();
}

inline void require_shape(const Tensor& tensor, const char* label, DType dtype,
                          std::initializer_list<std::int32_t> shape) {
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string("ple: ") + label + " has a wrong dtype");
    }
    std::size_t index = 0;
    for (const std::int32_t extent : shape) {
        if (tensor.ne[index] != extent) {
            throw std::invalid_argument(std::string("ple: ") + label + " has an unexpected shape");
        }
        ++index;
    }
    for (; index < 4; ++index) {
        if (tensor.ne[index] != 1) {
            throw std::invalid_argument(std::string("ple: ") + label + " has an unexpected rank");
        }
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("ple: ") + label + " must be contiguous");
    }
    if (!aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("ple: ") + label + " must be 16-byte aligned");
    }
}

inline void require_disjoint(const Tensor& output, const Tensor& input, const char* label) {
    if (overlaps(output, input)) {
        throw std::invalid_argument(std::string("ple: ") + label + " overlaps an input");
    }
}

inline void require_ple_geometry(const ninfer::artifact::RowPagedNf4Geometry& geometry) {
    if (geometry.rows != kPleShardRows || geometry.columns != kPleFeatures ||
        geometry.group_size != 16 || geometry.page_bytes != 4096 ||
        geometry.rows_per_page != 45 || geometry.code_bytes_per_row != 80 ||
        geometry.scale_bytes_per_row != 10 || geometry.matrix_scale_offset % 4 != 0) {
        throw std::invalid_argument("ple: the table is not the registered paged NF4 geometry");
    }
    if (geometry.encoded_bytes != geometry.matrix_scale_offset + 4) {
        throw std::invalid_argument("ple: the table geometry size does not match its layout");
    }
}

void ple_lookup_launch(const ninfer::artifact::RowPagedNf4Geometry& geometry,
                       const float* divisors, const std::uint32_t* rows,
                       const std::uint8_t* staged_rows, const Tensor& embedding,
                       std::uint32_t shard_count, std::size_t entries, cudaStream_t stream);

void ple_gate_launch(const Tensor& key, const Tensor& value, const Tensor& residual,
                     const Tensor& norm_key, const Tensor& norm_query, const Tensor& norm_conv,
                     float epsilon, const Tensor& gated_value, const Tensor& normalized_conv,
                     cudaStream_t stream);

void ple_conv_launch(const Tensor& history, const Tensor& values, const Tensor& residual,
                     const Tensor& gated_value, const Tensor& kernel, const Tensor& offsets,
                     const Tensor& counts, const void* next_norm, float epsilon,
                     const Tensor& updated_residual, void* normalized, bool normalize,
                     cudaStream_t stream);

void ple_state_extract_launch(const Tensor& before_history, const Tensor& records_values,
                              const Tensor& accepted_columns, const Tensor& after_history,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
