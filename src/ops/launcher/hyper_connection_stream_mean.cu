// Implements: include/ninfer/ops/hyper_connection_stream_mean.h
// The stream reduction is a pure sum over the second axis; the mean is formed once in FP32 and
// rounded to the output dtype. Strides are taken from the tensors, so a tapped feature slice can
// be written straight into its strided column block of the wide feature buffer.
#include "ops/launcher/hyper_connection_stream_mean.h"

#include "ops/common/math.h"
#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ float stream_mean_load(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
__device__ __forceinline__ float stream_mean_load(__half value) { return __half2float(value); }
__device__ __forceinline__ void stream_mean_store(__nv_bfloat16& destination, float value) {
    destination = __float2bfloat16_rn(value);
}
__device__ __forceinline__ void stream_mean_store(__half& destination, float value) {
    destination = __float2half_rn(value);
}

template <class TIn, class TOut>
__global__ void hyper_connection_stream_mean_kernel(
    const TIn* input, TOut* out, std::int64_t hidden, std::int64_t tokens, std::int64_t batch,
    std::int64_t streams, std::int64_t in_0, std::int64_t in_1, std::int64_t in_2,
    std::int64_t in_3, std::int64_t out_0, std::int64_t out_1, std::int64_t out_2,
    float inverse_streams) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = hidden * tokens * batch;
    if (index >= count) { return; }
    const std::int64_t h = index % hidden;
    const std::int64_t t = (index / hidden) % tokens;
    const std::int64_t b = index / (hidden * tokens);
    const std::int64_t in_base = h * in_0 + t * in_2 + b * in_3;
    float sum = 0.0F;
    for (std::int64_t s = 0; s < streams; ++s) {
        sum += stream_mean_load(input[in_base + s * in_1]);
    }
    stream_mean_store(out[h * out_0 + t * out_1 + b * out_2], sum * inverse_streams);
}

} // namespace

void hyper_connection_stream_mean_launch(const Tensor& input, Tensor& out, cudaStream_t stream) {
    const std::int64_t hidden  = input.ne[0];
    const std::int64_t streams = input.ne[1];
    const std::int64_t tokens  = input.ne[2];
    const std::int64_t batch   = input.ne[3];
    const std::int64_t count   = hidden * tokens * batch;
    const std::int64_t in_elem = static_cast<std::int64_t>(dtype_size(input.dtype));
    const std::int64_t out_elem = static_cast<std::int64_t>(dtype_size(out.dtype));
    const std::int64_t in_0    = input.nb[0] / in_elem;
    const std::int64_t in_1    = input.nb[1] / in_elem;
    const std::int64_t in_2    = input.nb[2] / in_elem;
    const std::int64_t in_3    = input.nb[3] / in_elem;
    const std::int64_t out_0   = out.nb[0] / out_elem;
    const std::int64_t out_1   = out.nb[1] / out_elem;
    const std::int64_t out_2   = out.nb[2] / out_elem;
    const float inverse        = 1.0F / static_cast<float>(streams);

    constexpr int kBlock   = 256;
    constexpr int kMaxGrid = 4096;
    const int grid = static_cast<int>(std::min<std::int64_t>(
        kMaxGrid, std::max<std::int64_t>(1, ninfer::ops::div_up(count, static_cast<std::int64_t>(kBlock)))));
    const auto launch = [&](auto* in_ptr, auto* out_ptr) {
        hyper_connection_stream_mean_kernel<<<grid, kBlock, 0, stream>>>(
            in_ptr, out_ptr, hidden, tokens, batch, streams, in_0, in_1, in_2, in_3, out_0, out_1,
            out_2, inverse);
    };
    if (input.dtype == DType::BF16 && out.dtype == DType::BF16) {
        launch(static_cast<const __nv_bfloat16*>(input.data), static_cast<__nv_bfloat16*>(out.data));
    } else if (input.dtype == DType::BF16) {
        launch(static_cast<const __nv_bfloat16*>(input.data), static_cast<__half*>(out.data));
    } else if (out.dtype == DType::BF16) {
        launch(static_cast<const __half*>(input.data), static_cast<__nv_bfloat16*>(out.data));
    } else {
        launch(static_cast<const __half*>(input.data), static_cast<__half*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
