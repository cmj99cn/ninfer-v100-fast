#include "ninfer/ops/ple.h"

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/ple/ple.h"

#include "core/device.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

namespace detail {
namespace {

constexpr int kConvBlock = 256;
constexpr int kStateBlock = 256;

__global__ __launch_bounds__(kConvBlock) void ple_conv_kernel(
    const __nv_bfloat16* history, const __nv_bfloat16* values, const __nv_bfloat16* residual,
    const __nv_bfloat16* gated_value, const __nv_bfloat16* kernel, const std::int32_t* offsets,
    const std::int32_t* counts, const __nv_bfloat16* next_norm, float epsilon,
    __nv_bfloat16* updated_residual, __nv_bfloat16* normalized, std::int32_t tokens, bool normalize) {
    const std::size_t column = blockIdx.x;
    const std::int32_t b = static_cast<std::int32_t>(column / tokens);
    const std::int32_t t = static_cast<std::int32_t>(column % tokens);
    if (t >= counts[b]) return;

    __shared__ float inverse[kPleStreams];
    __shared__ float scratch[kConvBlock / kWarpSize];

    for (std::size_t s = 0; s < kPleStreams; ++s) {
        float sum_squares = 0.0F;
        for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < kPleHidden; d += kConvBlock) {
            const std::size_t c = s * kPleHidden + d;
            float conv = 0.0F;
            for (std::size_t tap = 0; tap < kPleConvTaps; ++tap) {
                const std::int32_t k = t - static_cast<std::int32_t>(kPleConvWindow) +
                                       static_cast<std::int32_t>(tap) *
                                           static_cast<std::int32_t>(kPleConvDilation);
                float source = 0.0F;
                if (k < 0) {
                    source = __bfloat162float(
                        history[c + kPleChannels * static_cast<std::size_t>(9 + k) +
                                kPleChannels * kPleConvWindow * b]);
                } else {
                    const std::size_t source_column =
                        static_cast<std::size_t>(offsets[b] + k) + static_cast<std::size_t>(tokens) * b;
                    source = __bfloat162float(values[c + kPleChannels * source_column]);
                }
                conv += __bfloat162float(kernel[tap * kPleChannels + c]) * source;
            }
            const std::size_t index = c + kPleChannels * column;
            const __nv_bfloat16 stored = __float2bfloat16_rn(
                __bfloat162float(residual[index]) +
                __bfloat162float(gated_value[c + kPleChannels * column]) + silu(conv));
            updated_residual[index] = stored;
            const float word = __bfloat162float(stored);
            sum_squares += word * word;
        }
        const float total = block_reduce_sum<kConvBlock>(sum_squares, scratch);
        if (threadIdx.x == 0) {
            inverse[s] = rsqrtf(total / static_cast<float>(kPleHidden) + epsilon);
        }
        __syncthreads();
    }

    if (!normalize) return;

    for (std::size_t s = 0; s < kPleStreams; ++s) {
        for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < kPleHidden; d += kConvBlock) {
            const std::size_t c = s * kPleHidden + d;
            const std::size_t index = c + kPleChannels * column;
            normalized[index] = __float2bfloat16_rn(
                __bfloat162float(updated_residual[index]) *
                (1.0F + __bfloat162float(next_norm[c])) * inverse[s]);
        }
    }
}

__global__ __launch_bounds__(kStateBlock) void ple_state_extract_kernel(
    const __nv_bfloat16* before_history, const __nv_bfloat16* records_values,
    const std::int32_t* accepted_columns, __nv_bfloat16* after_history, std::int32_t tokens,
    std::size_t rows) {
    const std::size_t flat = static_cast<std::size_t>(blockIdx.x) * kStateBlock + threadIdx.x;
    if (flat >= rows) return;

    const std::size_t b = flat / kPleChannels;
    const std::size_t c = flat % kPleChannels;
    const std::int32_t accepted = accepted_columns[b];

    for (std::size_t j = 0; j < kPleConvWindow; ++j) {
        const std::int32_t k = accepted - static_cast<std::int32_t>(kPleConvWindow) +
                               static_cast<std::int32_t>(j);
        const std::size_t destination = c + kPleChannels * j + kPleChannels * kPleConvWindow * b;
        if (k < 0) {
            after_history[destination] =
                before_history[c + kPleChannels * static_cast<std::size_t>(9 + k) +
                               kPleChannels * kPleConvWindow * b];
        } else {
            after_history[destination] =
                records_values[c + kPleChannels * static_cast<std::size_t>(k) +
                               kPleChannels * static_cast<std::size_t>(tokens) * b];
        }
    }
}

} // namespace

void ple_conv_launch(const Tensor& history, const Tensor& values, const Tensor& residual,
                     const Tensor& gated_value, const Tensor& kernel, const Tensor& offsets,
                     const Tensor& counts, const void* next_norm, float epsilon,
                     const Tensor& updated_residual, void* normalized, bool normalize,
                     cudaStream_t stream) {
    const std::size_t tokens = static_cast<std::size_t>(values.ne[1]);
    const std::size_t batch = static_cast<std::size_t>(values.ne[2]);
    ple_conv_kernel<<<static_cast<int>(tokens * batch), kConvBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(history.data), static_cast<const __nv_bfloat16*>(values.data),
        static_cast<const __nv_bfloat16*>(residual.data), static_cast<const __nv_bfloat16*>(gated_value.data),
        static_cast<const __nv_bfloat16*>(kernel.data),
        static_cast<const std::int32_t*>(offsets.data), static_cast<const std::int32_t*>(counts.data),
        static_cast<const __nv_bfloat16*>(next_norm), epsilon,
        static_cast<__nv_bfloat16*>(updated_residual.data), static_cast<__nv_bfloat16*>(normalized),
        static_cast<std::int32_t>(tokens), normalize);
    CUDA_CHECK(cudaGetLastError());
}

void ple_state_extract_launch(const Tensor& before_history, const Tensor& records_values,
                              const Tensor& accepted_columns, const Tensor& after_history,
                              cudaStream_t stream) {
    const std::size_t batch = static_cast<std::size_t>(accepted_columns.ne[0]);
    const std::size_t tokens = static_cast<std::size_t>(records_values.ne[1]);
    const std::size_t rows = kPleChannels * batch;
    const int grid = static_cast<int>((rows + kStateBlock - 1) / kStateBlock);
    ple_state_extract_kernel<<<grid, kStateBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(before_history.data),
        static_cast<const __nv_bfloat16*>(records_values.data),
        static_cast<const std::int32_t*>(accepted_columns.data),
        static_cast<__nv_bfloat16*>(after_history.data), static_cast<std::int32_t>(tokens), rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace detail

namespace {

void require_conv_shapes(const Tensor& history, const Tensor& normalized_conv, const Tensor& residual,
                         const Tensor& gated_value, const Tensor& kernel, const Tensor& offsets,
                         const Tensor& counts) {
    const std::int32_t tokens = normalized_conv.ne[1];
    const std::int32_t batch = normalized_conv.ne[2];

    detail::require_shape(history, "history", DType::BF16,
                          {static_cast<std::int32_t>(kPleChannels), static_cast<std::int32_t>(kPleConvWindow),
                           batch});
    detail::require_shape(normalized_conv, "normalized_conv", DType::BF16,
                          {static_cast<std::int32_t>(kPleChannels), tokens, batch});
    detail::require_shape(residual, "residual", DType::BF16,
                          {static_cast<std::int32_t>(kPleHidden), static_cast<std::int32_t>(kPleStreams),
                           tokens, batch});
    detail::require_shape(gated_value, "gated_value", DType::BF16,
                          {static_cast<std::int32_t>(kPleChannels), tokens, batch});
    detail::require_shape(kernel, "kernel", DType::BF16,
                          {static_cast<std::int32_t>(kPleConvTaps), static_cast<std::int32_t>(kPleChannels)});
    detail::require_shape(offsets, "offsets", DType::I32, {batch});
    detail::require_shape(counts, "counts", DType::I32, {batch});

    if (tokens <= 0 || batch <= 0) {
        throw std::invalid_argument("ple: the convolution needs a positive column extent");
    }
}

} // namespace

void ple_conv_add(const Tensor& history, const Tensor& normalized_conv, const Tensor& residual,
                  const Tensor& gated_value, const Tensor& kernel, const Tensor& offsets,
                  const Tensor& counts, Tensor& updated_residual, cudaStream_t stream) {
    require_conv_shapes(history, normalized_conv, residual, gated_value, kernel, offsets, counts);
    detail::require_shape(updated_residual, "updated_residual", DType::BF16,
                          {static_cast<std::int32_t>(kPleHidden), static_cast<std::int32_t>(kPleStreams),
                           normalized_conv.ne[1], normalized_conv.ne[2]});

    detail::require_disjoint(updated_residual, history, "updated_residual");
    detail::require_disjoint(updated_residual, normalized_conv, "updated_residual");
    detail::require_disjoint(updated_residual, residual, "updated_residual");
    detail::require_disjoint(updated_residual, gated_value, "updated_residual");
    detail::require_disjoint(updated_residual, kernel, "updated_residual");
    detail::require_disjoint(updated_residual, offsets, "updated_residual");
    detail::require_disjoint(updated_residual, counts, "updated_residual");

    detail::ple_conv_launch(history, normalized_conv, residual, gated_value, kernel, offsets, counts,
                            nullptr, 0.0F, updated_residual, nullptr, false, stream);
}

void ple_conv_add_norm(const Tensor& history, const Tensor& normalized_conv, const Tensor& residual,
                       const Tensor& gated_value, const Tensor& kernel, const Tensor& offsets,
                       const Tensor& counts, const Tensor& next_hc_norm_weight, float epsilon,
                       Tensor& updated_residual, Tensor& normalized, cudaStream_t stream) {
    require_conv_shapes(history, normalized_conv, residual, gated_value, kernel, offsets, counts);
    detail::require_shape(next_hc_norm_weight, "next_hc_norm_weight", DType::BF16,
                          {static_cast<std::int32_t>(kPleChannels)});
    detail::require_shape(updated_residual, "updated_residual", DType::BF16,
                          {static_cast<std::int32_t>(kPleHidden), static_cast<std::int32_t>(kPleStreams),
                           normalized_conv.ne[1], normalized_conv.ne[2]});
    detail::require_shape(normalized, "normalized", DType::BF16,
                          {static_cast<std::int32_t>(kPleHidden), static_cast<std::int32_t>(kPleStreams),
                           normalized_conv.ne[1], normalized_conv.ne[2]});
    if (!(epsilon > 0.0F) || !std::isfinite(epsilon)) {
        throw std::invalid_argument("ple: the convolution epsilon must be positive and finite");
    }

    detail::require_disjoint(updated_residual, history, "updated_residual");
    detail::require_disjoint(updated_residual, normalized_conv, "updated_residual");
    detail::require_disjoint(updated_residual, residual, "updated_residual");
    detail::require_disjoint(updated_residual, gated_value, "updated_residual");
    detail::require_disjoint(updated_residual, kernel, "updated_residual");
    detail::require_disjoint(updated_residual, offsets, "updated_residual");
    detail::require_disjoint(updated_residual, counts, "updated_residual");
    detail::require_disjoint(updated_residual, next_hc_norm_weight, "updated_residual");
    detail::require_disjoint(normalized, history, "normalized");
    detail::require_disjoint(normalized, normalized_conv, "normalized");
    detail::require_disjoint(normalized, residual, "normalized");
    detail::require_disjoint(normalized, gated_value, "normalized");
    detail::require_disjoint(normalized, kernel, "normalized");
    detail::require_disjoint(normalized, offsets, "normalized");
    detail::require_disjoint(normalized, counts, "normalized");
    detail::require_disjoint(normalized, next_hc_norm_weight, "normalized");
    detail::require_disjoint(normalized, updated_residual, "normalized");

    detail::ple_conv_launch(history, normalized_conv, residual, gated_value, kernel, offsets, counts,
                            next_hc_norm_weight.data, epsilon, updated_residual, normalized.data, true,
                            stream);
}

void ple_state_extract(const Tensor& before_history, const Tensor& records_values,
                       const Tensor& accepted_columns, Tensor& after_history, cudaStream_t stream) {
    const std::int32_t batch = accepted_columns.ne[0];
    const std::int32_t tokens = records_values.ne[1];

    detail::require_shape(before_history, "before_history", DType::BF16,
                          {static_cast<std::int32_t>(kPleChannels), static_cast<std::int32_t>(kPleConvWindow),
                           batch});
    detail::require_shape(records_values, "records_values", DType::BF16,
                          {static_cast<std::int32_t>(kPleChannels), tokens, batch});
    detail::require_shape(accepted_columns, "accepted_columns", DType::I32, {batch});
    detail::require_shape(after_history, "after_history", DType::BF16,
                          {static_cast<std::int32_t>(kPleChannels), static_cast<std::int32_t>(kPleConvWindow),
                           batch});

    if (batch <= 0 || tokens <= 0) {
        throw std::invalid_argument("ple: the state extraction needs a positive extent");
    }
    if (after_history.data != before_history.data) {
        detail::require_disjoint(after_history, before_history, "after_history");
    }
    detail::require_disjoint(after_history, records_values, "after_history");
    detail::require_disjoint(after_history, accepted_columns, "after_history");

    detail::ple_state_extract_launch(before_history, records_values, accepted_columns, after_history,
                                     stream);
}

} // namespace ninfer::ops
