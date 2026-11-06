#include "ninfer/ops/cast.h"

#include "ops/launcher/cast.h"

#include <stdexcept>

namespace ninfer::ops {

void cast_fp32_to_bf16(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    if (source.dtype != DType::FP32 || destination.dtype != DType::BF16) {
        throw std::invalid_argument("cast_fp32_to_bf16: source must be FP32 and destination BF16");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (source.ne[dim] != destination.ne[dim]) {
            throw std::invalid_argument("cast_fp32_to_bf16: shapes must match");
        }
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr) {
        throw std::invalid_argument("cast_fp32_to_bf16: tensors must be contiguous and non-null");
    }
    if (source.data == destination.data) {
        throw std::invalid_argument("cast_fp32_to_bf16: source and destination must not alias");
    }
    detail::cast_fp32_to_bf16_launch(source, destination, stream);
}

void cast_bf16_to_fp16(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    if (source.dtype != DType::BF16 || destination.dtype != DType::FP16) {
        throw std::invalid_argument("cast_bf16_to_fp16: source must be BF16 and destination FP16");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (source.ne[dim] != destination.ne[dim]) {
            throw std::invalid_argument("cast_bf16_to_fp16: shapes must match");
        }
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr || source.data == destination.data) {
        throw std::invalid_argument("cast_bf16_to_fp16: tensors must be contiguous, non-null, disjoint");
    }
    detail::cast_bf16_to_fp16_launch(source, destination, stream);
}

void cast_fp16_to_bf16(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    if (source.dtype != DType::FP16 || destination.dtype != DType::BF16) {
        throw std::invalid_argument("cast_fp16_to_bf16: source must be FP16 and destination BF16");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (source.ne[dim] != destination.ne[dim]) {
            throw std::invalid_argument("cast_fp16_to_bf16: shapes must match");
        }
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr || source.data == destination.data) {
        throw std::invalid_argument("cast_fp16_to_bf16: tensors must be contiguous, non-null, disjoint");
    }
    detail::cast_fp16_to_bf16_launch(source, destination, stream);
}

} // namespace ninfer::ops
