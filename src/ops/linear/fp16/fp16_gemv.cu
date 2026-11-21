#include "ops/linear/fp16/fp16_launch.h"

#include "core/device.h"
#include "ops/linear/gemv/two_byte_gemv.cuh"

#include <cuda_fp16.h>

#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;

    const GemvContiguousOutput<__half> output{static_cast<__half*>(out.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    fp16_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __half*>(x.data), static_cast<const __half*>(weight.qdata),
        output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_fp16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == 124160 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<124160, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 10240 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<10240, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 2560 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<2560, 2560>>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("fp16 linear decode: unsupported exact problem");
}

} // namespace ninfer::ops::detail
