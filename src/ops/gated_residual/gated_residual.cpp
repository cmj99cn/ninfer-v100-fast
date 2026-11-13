#include "ninfer/ops/gated_residual.h"

#include "ninfer/ops/linear.h"
#include "ops/gated_residual/gated_residual.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

bool same_storage(const Tensor& lhs, const Tensor& rhs) {
    return lhs.data == rhs.data && lhs.bytes() == rhs.bytes();
}

void require_shape(const Tensor& tensor, const char* label, DType dtype,
                   std::initializer_list<std::int32_t> shape) {
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string("gated_residual: ") + label + " has a wrong dtype");
    }
    int index = 0;
    for (const std::int32_t extent : shape) {
        if (tensor.ne[index] != extent) {
            throw std::invalid_argument(std::string("gated_residual: ") + label +
                                        " has an unexpected shape");
        }
        ++index;
    }
    for (; index < 4; ++index) {
        if (tensor.ne[index] != 1) {
            throw std::invalid_argument(std::string("gated_residual: ") + label +
                                        " has an unexpected rank");
        }
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("gated_residual: ") + label + " must be contiguous");
    }
    if (!aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gated_residual: ") + label +
                                    " must be 16-byte aligned");
    }
}

void require_disjoint(const Tensor& output, const Tensor& input, const char* label) {
    if (overlaps(output, input)) {
        throw std::invalid_argument(std::string("gated_residual: ") + label +
                                    " must not overlap its input");
    }
}

void require_epsilon(float epsilon) {
    if (!(epsilon > 0.0F) || !std::isfinite(epsilon)) {
        throw std::invalid_argument("gated_residual: epsilon must be positive and finite");
    }
}

void require_bf16_geometry(const Weight& weight, const char* label, std::int32_t rows,
                           std::int32_t columns) {
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument(std::string("gated_residual: ") + label +
                                    " must be a contiguous BF16 weight");
    }
    if (weight.n != rows || weight.k != columns || weight.ndim != 2) {
        throw std::invalid_argument(std::string("gated_residual: ") + label +
                                    " has an unexpected geometry");
    }
}

void require_bf16_weight(const Weight& weight, const char* label, std::int32_t rows,
                         std::int32_t columns) {
    require_bf16_geometry(weight, label, rows, columns);
    for (int index = 0; index < 4; ++index) {
        if (weight.shape[index] != weight.padded_shape[index]) {
            throw std::invalid_argument(std::string("gated_residual: ") + label +
                                        " must not be padded");
        }
    }
    if (weight.shape[0] != weight.n || weight.shape[1] != weight.k) {
        throw std::invalid_argument(std::string("gated_residual: ") + label +
                                    " has an inconsistent shape");
    }
    const auto required_bytes = static_cast<std::uint64_t>(weight.n) *
                                static_cast<std::uint64_t>(weight.k) * 2;
    if (weight.payload_bytes < required_bytes) {
        throw std::invalid_argument(std::string("gated_residual: ") + label +
                                    " has too small a payload");
    }
    if (weight.qdata == nullptr || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("gated_residual: ") + label +
                                    " must be 16-byte aligned");
    }
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

std::size_t workspace_bytes(std::int32_t rank, std::int32_t channels, std::int32_t columns) {
    const std::size_t low   = static_cast<std::size_t>(rank) * static_cast<std::size_t>(columns) * 2;
    const std::size_t gates = static_cast<std::size_t>(channels) * static_cast<std::size_t>(columns) * 2;
    return align_up(low, 256) + gates;
}

void require_interval(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gated_residual: invalid token interval");
    }
}

} // namespace

void hc_expand_norm(const Tensor& embedding, const Tensor& norm_weight, float epsilon,
                    Tensor& residual, Tensor& normalized, cudaStream_t stream) {
    require_epsilon(epsilon);
    const std::int32_t hidden  = embedding.ne[0];
    const std::int32_t columns = embedding.ne[1];
    const std::int32_t streams = norm_weight.ne[1];
    require_shape(embedding, "embedding", DType::BF16, {hidden, columns});
    require_shape(norm_weight, "norm_weight", DType::BF16, {hidden, streams});
    require_shape(residual, "residual", DType::BF16, {hidden, streams, columns});
    require_shape(normalized, "normalized", DType::BF16, {hidden, streams, columns});
    require_disjoint(residual, embedding, "residual");
    require_disjoint(residual, norm_weight, "residual");
    require_disjoint(normalized, embedding, "normalized");
    require_disjoint(normalized, norm_weight, "normalized");
    require_disjoint(normalized, residual, "normalized");
    detail::hc_expand_norm_launch(embedding, norm_weight, epsilon, residual, normalized, stream);
}

void hc_grouped_rmsnorm(const Tensor& residual, const Tensor& norm_weight, float epsilon,
                        Tensor& normalized, cudaStream_t stream) {
    require_epsilon(epsilon);
    const std::int32_t hidden  = residual.ne[0];
    const std::int32_t streams = residual.ne[1];
    const std::int32_t columns = residual.ne[2];
    require_shape(residual, "residual", DType::BF16, {hidden, streams, columns});
    require_shape(norm_weight, "norm_weight", DType::BF16, {hidden, streams});
    require_shape(normalized, "normalized", DType::BF16, {hidden, streams, columns});
    require_disjoint(normalized, residual, "normalized");
    require_disjoint(normalized, norm_weight, "normalized");
    detail::hc_grouped_rmsnorm_launch(residual, norm_weight, epsilon, normalized, stream);
}

void hc_combine(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                Tensor& updated_residual, cudaStream_t stream) {
    const std::int32_t hidden  = residual.ne[0];
    const std::int32_t streams = residual.ne[1];
    const std::int32_t columns = residual.ne[2];
    require_shape(residual, "residual", DType::BF16, {hidden, streams, columns});
    require_shape(block_output, "block_output", DType::BF16, {hidden, columns});
    require_shape(injection, "injection", DType::FP32, {streams, columns});
    require_shape(updated_residual, "updated_residual", DType::BF16, {hidden, streams, columns});
    if (!same_storage(updated_residual, residual) && overlaps(updated_residual, residual)) {
        throw std::invalid_argument(
            "gated_residual: updated_residual must alias residual exactly or be disjoint");
    }
    require_disjoint(updated_residual, block_output, "updated_residual");
    require_disjoint(updated_residual, injection, "updated_residual");
    detail::hc_combine_launch(residual, block_output, injection, updated_residual, stream);
}

void hc_combine_norm(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                     const Tensor& next_norm_weight, float epsilon, Tensor& updated_residual,
                     Tensor& normalized, cudaStream_t stream) {
    require_epsilon(epsilon);
    const std::int32_t hidden  = residual.ne[0];
    const std::int32_t streams = residual.ne[1];
    const std::int32_t columns = residual.ne[2];
    require_shape(residual, "residual", DType::BF16, {hidden, streams, columns});
    require_shape(block_output, "block_output", DType::BF16, {hidden, columns});
    require_shape(injection, "injection", DType::FP32, {streams, columns});
    require_shape(next_norm_weight, "next_norm_weight", DType::BF16, {hidden, streams});
    require_shape(updated_residual, "updated_residual", DType::BF16, {hidden, streams, columns});
    require_shape(normalized, "normalized", DType::BF16, {hidden, streams, columns});
    if (!same_storage(updated_residual, residual) && overlaps(updated_residual, residual)) {
        throw std::invalid_argument(
            "gated_residual: updated_residual must alias residual exactly or be disjoint");
    }
    require_disjoint(updated_residual, block_output, "updated_residual");
    require_disjoint(updated_residual, injection, "updated_residual");
    require_disjoint(updated_residual, next_norm_weight, "updated_residual");
    require_disjoint(normalized, residual, "normalized");
    require_disjoint(normalized, block_output, "normalized");
    require_disjoint(normalized, injection, "normalized");
    require_disjoint(normalized, next_norm_weight, "normalized");
    require_disjoint(normalized, updated_residual, "normalized");
    detail::hc_combine_launch(residual, block_output, injection, updated_residual, stream);
    detail::hc_grouped_rmsnorm_launch(updated_residual, next_norm_weight, epsilon, normalized,
                                      stream);
}

void hc_mix(const Tensor& normalized, const HcMixWeights& weights, WorkspaceArena& workspace,
            Tensor& block_input, Tensor& injection, cudaStream_t stream) {
    const std::int32_t hidden   = normalized.ne[0];
    const std::int32_t streams  = normalized.ne[1];
    const std::int32_t columns  = normalized.ne[2];
    const std::int32_t rank     = weights.up.k;
    const std::int32_t channels = hidden * streams;
    require_shape(normalized, "normalized", DType::BF16, {hidden, streams, columns});
    require_shape(block_input, "block_input", DType::BF16, {hidden, columns});
    require_shape(injection, "injection", DType::FP32, {streams, columns});
    require_bf16_weight(weights.down_injection, "down_injection", rank + streams, channels);
    require_bf16_weight(weights.up, "up", channels, rank);
    require_disjoint(block_input, normalized, "block_input");
    require_disjoint(block_input, injection, "block_input");
    require_disjoint(injection, normalized, "injection");

    const auto scope = workspace.scope();
    Tensor low        = workspace.alloc(DType::BF16, {rank, columns});
    Tensor gates      = workspace.alloc(DType::BF16, {channels, columns});
    const Tensor flattened = normalized.view({channels, columns});

    Weight down          = weights.down_injection;
    down.n               = rank;
    down.shape[0]        = rank;
    down.padded_shape[0] = rank;
    down.payload_bytes   = static_cast<std::uint64_t>(rank) * channels * 2;

    if (columns == 1) {
        detail::hc_project_gemv_launch(down.qdata, rank, channels, flattened.data, low, stream);
        detail::hc_mix_silu_launch(low, streams, stream);
        detail::hc_project_gemv_launch(weights.up.qdata, channels, rank, low.data, gates, stream);
    } else {
        linear(flattened, down, low, stream);
        detail::hc_mix_silu_launch(low, streams, stream);
        linear(low, weights.up, gates, stream);
    }
    detail::hc_mix_gate_launch(gates, normalized, block_input, hidden, streams, stream);

    const auto* injection_rows = static_cast<const std::uint8_t*>(weights.down_injection.qdata) +
                                  static_cast<std::size_t>(rank) * channels * 2;
    detail::hc_mix_injection_launch(injection_rows, flattened, channels, streams, injection, stream);
}

void hc_final_mix(const Tensor& normalized, const HcFinalMixWeights& weights,
                  WorkspaceArena& workspace, Tensor& sample_hidden, cudaStream_t stream) {
    const std::int32_t hidden   = normalized.ne[0];
    const std::int32_t streams  = normalized.ne[1];
    const std::int32_t columns  = normalized.ne[2];
    const std::int32_t rank     = weights.down.n;
    const std::int32_t channels = hidden * streams;
    require_shape(normalized, "normalized", DType::BF16, {hidden, streams, columns});
    require_shape(sample_hidden, "sample_hidden", DType::BF16, {hidden, columns});
    require_bf16_weight(weights.down, "down", rank, channels);
    require_bf16_weight(weights.up, "up", channels, rank);
    require_disjoint(sample_hidden, normalized, "sample_hidden");

    const auto scope = workspace.scope();
    Tensor low        = workspace.alloc(DType::BF16, {rank, columns});
    Tensor gates      = workspace.alloc(DType::BF16, {channels, columns});
    const Tensor flattened = normalized.view({channels, columns});

    if (columns == 1) {
        detail::hc_project_gemv_launch(weights.down.qdata, rank, channels, flattened.data, low,
                                        stream);
        detail::hc_mix_silu_launch(low, streams, stream);
        detail::hc_project_gemv_launch(weights.up.qdata, channels, rank, low.data, gates, stream);
    } else {
        linear(flattened, weights.down, low, stream);
        detail::hc_mix_silu_launch(low, streams, stream);
        linear(low, weights.up, gates, stream);
    }
    detail::hc_mix_gate_launch(gates, normalized, sample_hidden, hidden, streams, stream);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcMixWeights& weights, std::int32_t min_tokens,
                                            std::int32_t max_tokens) {
    require_interval(min_tokens, max_tokens);
    const std::int32_t rank     = weights.up.k;
    const std::int32_t channels = weights.down_injection.k;
    const std::int32_t streams  = weights.down_injection.n - rank;
    if (rank <= 0 || channels <= 0 || streams <= 0) {
        throw std::invalid_argument("gated_residual: invalid mixing weight geometry");
    }
    require_bf16_geometry(weights.down_injection, "down_injection", rank + streams, channels);
    require_bf16_geometry(weights.up, "up", channels, rank);
    return workspace_bytes(rank, channels, max_tokens);
}

std::size_t hc_final_mix_workspace_capacity_bytes(const HcFinalMixWeights& weights,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    require_interval(min_tokens, max_tokens);
    const std::int32_t rank     = weights.down.n;
    const std::int32_t channels = weights.down.k;
    if (rank <= 0 || channels <= 0) {
        throw std::invalid_argument("gated_residual: invalid mixing weight geometry");
    }
    require_bf16_geometry(weights.down, "down", rank, channels);
    require_bf16_geometry(weights.up, "up", channels, rank);
    return workspace_bytes(rank, channels, max_tokens);
}

} // namespace ninfer::ops
