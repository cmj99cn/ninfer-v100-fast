// ninfer::ops - hyper_connection_stream_mean wrapper: implements the public api, validates
// parameters, and dispatches to the launcher. Host-compiled; never includes the kernel header.
#include "ninfer/ops/hyper_connection_stream_mean.h"

#include "core/dtype.h"
#include "ops/launcher/hyper_connection_stream_mean.h" // detail::hyper_connection_stream_mean_launch

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require_stride(std::int64_t stride, std::size_t element, const char* name) {
    if (stride < 0 || static_cast<std::uint64_t>(stride) % element != 0) {
        throw std::invalid_argument(std::string("hyper_connection_stream_mean: ") + name +
                                    " is not a whole number of elements");
    }
}

} // namespace

void hyper_connection_stream_mean(const Tensor& input, Tensor& out, cudaStream_t stream) {
    if (input.dtype != DType::BF16 && input.dtype != DType::FP16) {
        throw std::invalid_argument("hyper_connection_stream_mean: input must be BF16 or FP16");
    }
    if (out.dtype != DType::BF16 && out.dtype != DType::FP16) {
        throw std::invalid_argument("hyper_connection_stream_mean: output must be BF16 or FP16");
    }
    if (input.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("hyper_connection_stream_mean: data must be non-null");
    }
    if (input.ne[0] <= 0 || input.ne[1] <= 0 || input.ne[2] <= 0 ||
        input.ne[3] <= 0) {
        throw std::invalid_argument("hyper_connection_stream_mean: input extents must be positive");
    }
    if (out.ne[0] != input.ne[0] || out.ne[1] != input.ne[2] || out.ne[2] != input.ne[3] ||
        out.ne[3] != 1) {
        throw std::invalid_argument(
            "hyper_connection_stream_mean: output shape must be [H,T,B] of the input [H,S,T,B]");
    }

    const std::size_t in_element  = dtype_size(input.dtype);
    const std::size_t out_element = dtype_size(out.dtype);
    if (static_cast<std::uint64_t>(input.nb[0]) != in_element ||
        static_cast<std::uint64_t>(out.nb[0]) != out_element) {
        throw std::invalid_argument(
            "hyper_connection_stream_mean: the contiguous channel axis must have unit stride");
    }
    require_stride(input.nb[1], in_element, "input row stride");
    require_stride(input.nb[2], in_element, "input token stride");
    require_stride(input.nb[3], in_element, "input batch stride");
    require_stride(out.nb[1], out_element, "output token stride");
    require_stride(out.nb[2], out_element, "output batch stride");

    detail::hyper_connection_stream_mean_launch(input, out, stream);
}

} // namespace ninfer::ops
