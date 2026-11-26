// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

#include "ninfer/ops/dflash2_attn_input.h"

#include "ops/launcher/dflash2_attn_input.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp = "dflash2_attn_input_proj";

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

void require_parent(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t codes  = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scales = static_cast<std::uint64_t>(rows) * (columns / 32) * 2U;
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != columns || weight.shape[0] != rows ||
        weight.shape[1] != columns || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != columns || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || weight.payload_bytes < codes + scales ||
        weight.qdata == nullptr || weight.scales == nullptr || (columns % 32) != 0 ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid query/key/value weight");
    }
}

} // namespace

void dflash2_attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q,
                             Tensor& k, Tensor& v, cudaStream_t stream) {
    const std::int32_t columns = x.ne[0];
    const std::int32_t tokens  = x.ne[1];
    if (columns != 2560 || tokens <= 0) {
        throw std::invalid_argument(std::string(kOp) + ": x must be BF16 [2560,T], T>0");
    }
    require_matrix(x, 2560, tokens, "x");
    require_matrix(q, 4096, tokens, "q");
    require_matrix(k, 1024, tokens, "k");
    require_matrix(v, 1024, tokens, "v");
    require_parent(query_key_value_weight, 6144, 2560);
    detail::dflash2_attn_input_proj_launch(x, query_key_value_weight, q, k, v, stream);
}

} // namespace ninfer::ops
