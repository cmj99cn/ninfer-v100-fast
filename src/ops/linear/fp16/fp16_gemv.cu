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

template <class Geometry, int Columns>
void launch_geometry_multi(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;

    const GemvColumnContiguousOutput<__half> output{static_cast<__half*>(out.data),
                                                    Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    fp16_gemv_multi_kernel<Geometry, Schedule, Columns><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __half*>(x.data), static_cast<const __half*>(weight.qdata), output);
}

template <class Geometry>
void launch_geometry_multi_columns(const Tensor& x, const Weight& weight, Tensor& out,
                                   std::int32_t columns, cudaStream_t stream) {
    switch (columns) {
    case 2: launch_geometry_multi<Geometry, 2>(x, weight, out, stream); break;
    case 3: launch_geometry_multi<Geometry, 3>(x, weight, out, stream); break;
    case 4: launch_geometry_multi<Geometry, 4>(x, weight, out, stream); break;
    case 5: launch_geometry_multi<Geometry, 5>(x, weight, out, stream); break;
    case 6: launch_geometry_multi<Geometry, 6>(x, weight, out, stream); break;
    case 7: launch_geometry_multi<Geometry, 7>(x, weight, out, stream); break;
    case 8: launch_geometry_multi<Geometry, 8>(x, weight, out, stream); break;
    default: throw std::invalid_argument("fp16 linear multi decode: unsupported column count");
    }
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

void launch_fp16_decode_multi(const Tensor& x, const Weight& weight, Tensor& out,
                              std::int32_t columns, cudaStream_t stream) {
    if (weight.n == 124160 && weight.k == 2560) {
        launch_geometry_multi_columns<Bf16GemvGeometry<124160, 2560>>(x, weight, out, columns, stream);
        return;
    }
    if (weight.n == 10240 && weight.k == 2560) {
        launch_geometry_multi_columns<Bf16GemvGeometry<10240, 2560>>(x, weight, out, columns, stream);
        return;
    }
    if (weight.n == 2560 && weight.k == 2560) {
        launch_geometry_multi_columns<Bf16GemvGeometry<2560, 2560>>(x, weight, out, columns, stream);
        return;
    }
    throw std::invalid_argument("fp16 linear multi decode: unsupported exact problem");
}

} // namespace ninfer::ops::detail
