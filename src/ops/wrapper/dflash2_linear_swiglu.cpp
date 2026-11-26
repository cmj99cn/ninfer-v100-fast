// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

#include "ninfer/ops/dflash2_linear_swiglu.h"

#include "core/layout.h"
#include "ops/launcher/dflash2_linear_swiglu.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp = "dflash2_linear_swiglu";

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                    const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + label);
    }
}

void require_parent(const Weight& weight) {
    constexpr std::int32_t kRows = 15360;
    constexpr std::int32_t kColumns = 2560;
    const std::uint64_t codes  = static_cast<std::uint64_t>(kRows) * kColumns;
    const std::uint64_t scales = static_cast<std::uint64_t>(kRows) * (kColumns / 32) * 2U;
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != kRows || weight.k != kColumns ||
        weight.shape[0] != kRows || weight.shape[1] != kColumns ||
        weight.padded_shape[0] != kRows || weight.padded_shape[1] != kColumns ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < codes + scales || weight.qdata == nullptr ||
        weight.scales == nullptr || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid gate/up weight");
    }
}

} // namespace

std::size_t dflash2_linear_swiglu_workspace_capacity_bytes(std::int32_t min_tokens,
                                                            std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument(std::string(kOp) + " workspace: invalid token interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {7680, max_tokens}, 256);
    return layout.peak_bytes();
}

void dflash2_linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t columns = x.ne[0];
    const std::int32_t tokens  = x.ne[1];
    if (columns != 2560 || tokens <= 0) {
        throw std::invalid_argument(std::string(kOp) + ": x must be BF16 [2560,T], T>0");
    }
    require_matrix(x, 2560, tokens, "x");
    require_matrix(out, 7680, tokens, "out");
    require_parent(gate_up_weight);
    auto scope = workspace.scope();
    Tensor gate = workspace.alloc(DType::BF16, {7680, tokens});
    detail::dflash2_linear_swiglu_launch(x, gate_up_weight, out, gate, stream);
}

} // namespace ninfer::ops
