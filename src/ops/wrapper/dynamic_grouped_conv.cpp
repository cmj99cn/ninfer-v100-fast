// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

#include "ninfer/ops/dynamic_grouped_conv.h"

#include "core/layout.h"
#include "ops/kernel/dflash2_gemm.cuh"
#include "ops/launcher/dynamic_grouped_conv.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kGroupWidth = 16;
constexpr std::int32_t kTaps       = 2;
constexpr std::int32_t kSides      = 2;
constexpr const char* kPrepareOp   = "rmsnorm_dynamic_grouped_conv_prepare";
constexpr const char* kAddOp       = "linear_dynamic_grouped_conv_add";
constexpr const char* kConvOp      = "dynamic_grouped_conv";

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t d0, std::int32_t d1,
                    std::int32_t d2, std::int32_t d3, const char* op, const char* label) {
    if (tensor.dtype != dtype || tensor.ne[0] != d0 || tensor.ne[1] != d1 || tensor.ne[2] != d2 ||
        tensor.ne[3] != d3 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

void require_bf16_projection(const Weight& weight, std::int32_t rows, std::int32_t columns,
                             const char* op, const char* label) {
    const std::uint64_t payload =
        static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) * 2U;
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != columns || weight.shape[0] != rows ||
        weight.shape[1] != columns || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != columns || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

void require_w8_projection(const Weight& weight, std::int32_t rows, std::int32_t columns,
                           const char* op, const char* label) {
    const std::uint64_t codes = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scales = static_cast<std::uint64_t>(rows) * (columns / 32) * 2U;
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != columns || weight.shape[0] != rows ||
        weight.shape[1] != columns || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != columns || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || weight.payload_bytes < codes + scales ||
        weight.qdata == nullptr || weight.scales == nullptr ||
        (columns % 32) != 0 || !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

std::size_t prepare_workspace_bytes(std::int32_t hidden, std::int32_t width, std::int32_t batch) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {hidden, width * batch}, 256);
    (void)layout.alloc(DType::FP32, {4 * (hidden / kGroupWidth), width * batch}, 256);
    return layout.peak_bytes();
}

std::size_t add_workspace_bytes(std::int32_t hidden, std::int32_t width, std::int32_t batch) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {hidden, width * batch}, 256);
    return layout.peak_bytes();
}

} // namespace

void dynamic_grouped_conv(const Tensor& x, const Tensor& coefficients, const Tensor& base_kernel,
                          std::int32_t side, Tensor& out, cudaStream_t stream) {
    const std::int32_t hidden = x.ne[0], width = x.ne[1], batch = x.ne[2];
    if (side != 0 && side != 1) {
        throw std::invalid_argument(std::string(kConvOp) + ": side must be 0 or 1");
    }
    if (hidden <= 0 || (hidden % kGroupWidth) != 0) {
        throw std::invalid_argument(std::string(kConvOp) + ": H must be a positive multiple of 16");
    }
    if (width < 1 || width > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument(std::string(kConvOp) + ": W must be in [1,16] and B in [1,8]");
    }
    if ((x.dtype != DType::BF16 && x.dtype != DType::FP32) || x.ne[3] != 1 ||
        !x.is_contiguous() || !aligned_to(x.data, 16)) {
        throw std::invalid_argument(std::string(kConvOp) + ": invalid x");
    }
    const std::int32_t rows = 4 * (hidden / kGroupWidth);
    if ((coefficients.dtype != DType::FP32 && coefficients.dtype != DType::BF16) ||
        coefficients.ne[0] != rows || coefficients.ne[1] != width ||
        coefficients.ne[2] != batch || coefficients.ne[3] != 1 ||
        !coefficients.is_contiguous() || !aligned_to(coefficients.data, 16)) {
        throw std::invalid_argument(std::string(kConvOp) + ": invalid coefficients");
    }
    if (base_kernel.dtype != DType::BF16 || base_kernel.ne[0] != hidden ||
        base_kernel.ne[1] != kTaps || base_kernel.ne[2] != kSides || base_kernel.ne[3] != 1 ||
        !base_kernel.is_contiguous() || !aligned_to(base_kernel.data, 16)) {
        throw std::invalid_argument(std::string(kConvOp) + ": invalid base_kernel");
    }
    require_tensor(out, DType::BF16, hidden, width, batch, 1, kConvOp, "out");
    const auto begin = [](const Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data); };
    const auto end   = [&](const Tensor& t) { return begin(t) + t.bytes(); };
    for (const Tensor* input : {&x, &coefficients, &base_kernel}) {
        if (!(end(*input) <= begin(out) || end(out) <= begin(*input))) {
            throw std::invalid_argument(std::string(kConvOp) + ": out must not overlap an input");
        }
    }
    detail::dynamic_grouped_conv_launch(x, coefficients, base_kernel, side, out, stream);
}

std::size_t rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t min_width, std::int32_t max_width, std::int32_t min_batch,
    std::int32_t max_batch) {
    if (hidden <= 0 || (hidden % kGroupWidth) != 0 || min_width < 2 || max_width > 16 ||
        min_width > max_width || min_batch < 1 || max_batch > 8 || min_batch > max_batch) {
        throw std::invalid_argument(std::string(kPrepareOp) + " workspace: invalid W/B interval");
    }
    return prepare_workspace_bytes(hidden, max_width, max_batch);
}

void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Weight& kernel_projection_weight, Tensor& prepared,
                                          Tensor& finish_delta, WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    const std::int32_t hidden = residual.ne[0];
    const std::int32_t width  = residual.ne[1];
    const std::int32_t batch  = residual.ne[2];
    if (hidden <= 0 || (hidden % kGroupWidth) != 0 || width < 2 || width > 16 || batch < 1 ||
        batch > 8) {
        throw std::invalid_argument(std::string(kPrepareOp) + ": invalid W/B or H");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument(std::string(kPrepareOp) + ": eps must be positive and finite");
    }
    require_tensor(residual, DType::BF16, hidden, width, batch, 1, kPrepareOp, "residual");
    require_tensor(norm_weight, DType::BF16, hidden, 1, 1, 1, kPrepareOp, "norm_weight");
    require_tensor(base_kernel, DType::BF16, hidden, kTaps, kSides, 1, kPrepareOp, "base_kernel");
    require_tensor(prepared, DType::BF16, hidden, width, batch, 1, kPrepareOp, "prepared");
    const std::int32_t groups = hidden / kGroupWidth;
    require_tensor(finish_delta, DType::BF16, groups, kTaps, width, batch, kPrepareOp,
                   "finish_delta");
    require_bf16_projection(kernel_projection_weight, 4 * groups, hidden, kPrepareOp,
                            "kernel_projection_weight");

    auto scope = workspace.scope();
    Tensor normalized = workspace.alloc(DType::BF16, {hidden, width * batch});
    Tensor coefficients = workspace.alloc(DType::FP32, {4 * groups, width * batch});
    detail::prepare_normalize_launch(residual, norm_weight, eps, normalized, hidden,
                                     width * batch, stream);
    detail::dflash2_bf16_f32_gemm(normalized.data, kernel_projection_weight.qdata,
                                  static_cast<float*>(coefficients.data), 4 * groups, hidden,
                                  width * batch, stream);
    detail::prepare_conv_launch(normalized, coefficients, base_kernel, prepared, finish_delta,
                                hidden, width, batch, stream);
}

std::size_t linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t input_rows, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch, std::int32_t max_batch) {
    if (hidden <= 0 || (hidden % kGroupWidth) != 0 || input_rows <= 0 || min_width < 2 ||
        max_width > 16 || min_width > max_width || min_batch < 1 || max_batch > 8 ||
        min_batch > max_batch) {
        throw std::invalid_argument(std::string(kAddOp) + " workspace: invalid W/B interval");
    }
    return add_workspace_bytes(hidden, max_width, max_batch);
}

void linear_dynamic_grouped_conv_add(const Tensor& x, const Weight& projection_weight,
                                     const Tensor& base_kernel, const Tensor& finish_delta,
                                     Tensor& residual, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    const std::int32_t input_rows = x.ne[0];
    const std::int32_t width      = x.ne[1];
    const std::int32_t batch      = x.ne[2];
    const std::int32_t hidden     = residual.ne[0];
    if (hidden <= 0 || (hidden % kGroupWidth) != 0 || input_rows <= 0 || width < 2 ||
        width > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument(std::string(kAddOp) + ": invalid W/B or H");
    }
    require_tensor(x, DType::BF16, input_rows, width, batch, 1, kAddOp, "x");
    require_tensor(base_kernel, DType::BF16, hidden, kTaps, kSides, 1, kAddOp, "base_kernel");
    const std::int32_t groups = hidden / kGroupWidth;
    require_tensor(finish_delta, DType::BF16, groups, kTaps, width, batch, kAddOp,
                   "finish_delta");
    require_tensor(residual, DType::BF16, hidden, width, batch, 1, kAddOp, "residual");
    require_w8_projection(projection_weight, hidden, input_rows, kAddOp, "projection_weight");

    auto scope = workspace.scope();
    Tensor projected = workspace.alloc(DType::BF16, {hidden, width * batch});
    detail::dflash2_w8_gemm(x.data, static_cast<const std::uint8_t*>(projection_weight.qdata),
                            static_cast<const std::uint8_t*>(projection_weight.scales),
                            projected.data, hidden, input_rows, width * batch, stream);
    detail::finish_add_launch(projected, base_kernel, finish_delta, residual, hidden, width, batch,
                              stream);
}

} // namespace ninfer::ops
