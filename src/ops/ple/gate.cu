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

constexpr int kGateBlock = 256;

template <class A>
__global__ __launch_bounds__(kGateBlock) void ple_gate_kernel(
    const A* key, const A* value, const A* residual, const __nv_bfloat16* norm_key,
    const __nv_bfloat16* norm_query, const __nv_bfloat16* norm_conv, float epsilon,
    A* gated_value, A* normalized_conv) {
    const std::size_t column = blockIdx.x;
    const std::size_t channel_base = column * kPleChannels;
    const std::size_t hidden_base = column * kPleHidden;

    __shared__ float inverse_key[kPleStreams];
    __shared__ float inverse_query[kPleStreams];
    __shared__ float gate[kPleStreams];
    __shared__ float inverse_conv[kPleStreams];
    __shared__ float scratch[kGateBlock / kWarpSize];

    for (std::size_t s = 0; s < kPleStreams; ++s) {
        const std::size_t base = channel_base + s * kPleHidden;
        float sum_key = 0.0F;
        float sum_query = 0.0F;
        for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < kPleHidden; d += kGateBlock) {
            const float k = PleAct<A>::to_float(key[base + d]);
            const float q = PleAct<A>::to_float(residual[base + d]);
            sum_key += k * k;
            sum_query += q * q;
        }
        const float total_key = block_reduce_sum<kGateBlock>(sum_key, scratch);
        __syncthreads();
        const float total_query = block_reduce_sum<kGateBlock>(sum_query, scratch);
        if (threadIdx.x == 0) {
            inverse_key[s] = rsqrtf(total_key / static_cast<float>(kPleHidden) + epsilon);
            inverse_query[s] = rsqrtf(total_query / static_cast<float>(kPleHidden) + epsilon);
        }
        __syncthreads();
    }

    for (std::size_t s = 0; s < kPleStreams; ++s) {
        const std::size_t base = channel_base + s * kPleHidden;
        float dot = 0.0F;
        for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < kPleHidden; d += kGateBlock) {
            const std::size_t weight = s * kPleHidden + d;
            const float normalized_key =
                PleAct<A>::to_float(key[base + d]) * (1.0F + __bfloat162float(norm_key[weight]));
            const float normalized_query =
                PleAct<A>::to_float(residual[base + d]) *
                (1.0F + __bfloat162float(norm_query[weight]));
            dot += normalized_key * normalized_query;
        }
        const float total = block_reduce_sum<kGateBlock>(dot, scratch);
        if (threadIdx.x == 0) {
            const float a = total * inverse_key[s] * inverse_query[s] *
                            rsqrtf(static_cast<float>(kPleHidden));
            const float sign = a > 0.0F ? 1.0F : (a < 0.0F ? -1.0F : 0.0F);
            gate[s] = sigmoid(sign * sqrtf(fmaxf(fabsf(a), kPleGateFloor)));
        }
        __syncthreads();
    }

    for (std::size_t s = 0; s < kPleStreams; ++s) {
        const std::size_t base = channel_base + s * kPleHidden;
        float sum_squares = 0.0F;
        for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < kPleHidden; d += kGateBlock) {
            const A stored = PleAct<A>::from_float(gate[s] *
                                                   PleAct<A>::to_float(value[hidden_base + d]));
            gated_value[base + d] = stored;
            const float word = PleAct<A>::to_float(stored);
            sum_squares += word * word;
        }
        const float total = block_reduce_sum<kGateBlock>(sum_squares, scratch);
        if (threadIdx.x == 0) {
            inverse_conv[s] = rsqrtf(total / static_cast<float>(kPleHidden) + epsilon);
        }
        __syncthreads();
    }

    for (std::size_t s = 0; s < kPleStreams; ++s) {
        const std::size_t base = channel_base + s * kPleHidden;
        for (std::int32_t d = static_cast<std::int32_t>(threadIdx.x); d < kPleHidden; d += kGateBlock) {
            const float word = PleAct<A>::to_float(gated_value[base + d]);
            normalized_conv[base + d] = PleAct<A>::from_float(
                word * (1.0F + __bfloat162float(norm_conv[s * kPleHidden + d])) * inverse_conv[s]);
        }
    }
}

} // namespace

void ple_gate_launch(const Tensor& key, const Tensor& value, const Tensor& residual,
                     const Tensor& norm_key, const Tensor& norm_query, const Tensor& norm_conv,
                     float epsilon, const Tensor& gated_value, const Tensor& normalized_conv,
                     cudaStream_t stream) {
    const std::size_t columns = static_cast<std::size_t>(key.ne[1]) * static_cast<std::size_t>(key.ne[2]);
    if (key.dtype == DType::FP16) {
        ple_gate_kernel<__half><<<static_cast<int>(columns), kGateBlock, 0, stream>>>(
            static_cast<const __half*>(key.data), static_cast<const __half*>(value.data),
            static_cast<const __half*>(residual.data),
            static_cast<const __nv_bfloat16*>(norm_key.data),
            static_cast<const __nv_bfloat16*>(norm_query.data),
            static_cast<const __nv_bfloat16*>(norm_conv.data), epsilon,
            static_cast<__half*>(gated_value.data), static_cast<__half*>(normalized_conv.data));
    } else {
        ple_gate_kernel<__nv_bfloat16><<<static_cast<int>(columns), kGateBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(key.data),
            static_cast<const __nv_bfloat16*>(value.data),
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<const __nv_bfloat16*>(norm_key.data),
            static_cast<const __nv_bfloat16*>(norm_query.data),
            static_cast<const __nv_bfloat16*>(norm_conv.data), epsilon,
            static_cast<__nv_bfloat16*>(gated_value.data),
            static_cast<__nv_bfloat16*>(normalized_conv.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace detail

void ple_gate(const Tensor& projected_key, const Tensor& projected_value, const Tensor& residual,
              const Tensor& norm_key, const Tensor& norm_query, const Tensor& norm_conv,
              float epsilon, Tensor& gated_value, Tensor& normalized_conv, cudaStream_t stream) {
    const std::int32_t tokens = projected_key.ne[1];
    const std::int32_t batch = projected_key.ne[2];

    detail::require_activation(projected_key, "projected.key");
    detail::require_activation(projected_value, "projected.value");
    detail::require_activation(residual, "residual");
    const DType activation = projected_key.dtype;
    if (projected_value.dtype != activation || residual.dtype != activation) {
        throw std::invalid_argument("ple: the gate activation dtypes must agree");
    }
    detail::require_shape(projected_key, "projected.key", projected_key.dtype,
                          {static_cast<std::int32_t>(kPleChannels), tokens, batch});
    detail::require_shape(projected_value, "projected.value", projected_value.dtype,
                          {static_cast<std::int32_t>(kPleHidden), tokens, batch});
    detail::require_shape(residual, "residual", residual.dtype,
                          {static_cast<std::int32_t>(kPleHidden), static_cast<std::int32_t>(kPleStreams),
                           tokens, batch});
    detail::require_shape(norm_key, "norm_key", DType::BF16, {static_cast<std::int32_t>(kPleChannels)});
    detail::require_shape(norm_query, "norm_query", DType::BF16, {static_cast<std::int32_t>(kPleChannels)});
    detail::require_shape(norm_conv, "norm_conv", DType::BF16, {static_cast<std::int32_t>(kPleChannels)});
    detail::require_activation(gated_value, "gated_value");
    detail::require_activation(normalized_conv, "normalized_conv");
    if (gated_value.dtype != activation || normalized_conv.dtype != activation) {
        throw std::invalid_argument("ple: the gate activation dtypes must agree");
    }
    detail::require_shape(gated_value, "gated_value", gated_value.dtype,
                          {static_cast<std::int32_t>(kPleChannels), tokens, batch});
    detail::require_shape(normalized_conv, "normalized_conv", normalized_conv.dtype,
                          {static_cast<std::int32_t>(kPleChannels), tokens, batch});

    if (tokens <= 0 || batch <= 0) {
        throw std::invalid_argument("ple: the gate needs a positive column extent");
    }
    if (!(epsilon > 0.0F) || !std::isfinite(epsilon)) {
        throw std::invalid_argument("ple: the gate epsilon must be positive and finite");
    }
    detail::require_disjoint(gated_value, normalized_conv, "gated_value");
    detail::require_disjoint(gated_value, projected_key, "gated_value");
    detail::require_disjoint(gated_value, projected_value, "gated_value");
    detail::require_disjoint(gated_value, residual, "gated_value");
    detail::require_disjoint(gated_value, norm_key, "gated_value");
    detail::require_disjoint(gated_value, norm_query, "gated_value");
    detail::require_disjoint(gated_value, norm_conv, "gated_value");
    detail::require_disjoint(normalized_conv, projected_key, "normalized_conv");
    detail::require_disjoint(normalized_conv, projected_value, "normalized_conv");
    detail::require_disjoint(normalized_conv, residual, "normalized_conv");
    detail::require_disjoint(normalized_conv, norm_key, "normalized_conv");
    detail::require_disjoint(normalized_conv, norm_query, "normalized_conv");
    detail::require_disjoint(normalized_conv, norm_conv, "normalized_conv");

    detail::ple_gate_launch(projected_key, projected_value, residual, norm_key, norm_query, norm_conv,
                            epsilon, gated_value, normalized_conv, stream);
}

} // namespace ninfer::ops
