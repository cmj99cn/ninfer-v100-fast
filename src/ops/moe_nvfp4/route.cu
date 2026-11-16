#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <type_traits>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

struct RankedValue {
    float value;
    int id;
};

__device__ __forceinline__ bool ranked_better(const RankedValue& a, const RankedValue& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

inline constexpr int kRouteRows        = kMoeExperts + 1;
inline constexpr int kRouteGroups      = kMoeHidden / 16;
inline constexpr int kRoutePhases      = kRouteGroups / 32;
static_assert(kRouteGroups % 32 == 0);
inline constexpr int kRouteWarpsPerCta = 8;

template <class A>
__device__ __forceinline__ float route_row_dot(const std::uint32_t* weight_words,
                                               const A* activation, int lane) {
    const auto* activation_words = reinterpret_cast<const std::uint32_t*>(activation);
    float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (int phase = 0; phase < kRoutePhases; ++phase) {
        const int group = phase * 32 + lane;
        const uint4 weight_low  = load_vec<uint4>(weight_words + 8 * group);
        const uint4 weight_high = load_vec<uint4>(weight_words + 8 * group + 4);
        const uint4 value_low   = load_vec<uint4>(activation_words + 8 * group);
        const uint4 value_high  = load_vec<uint4>(activation_words + 8 * group + 4);
        const std::uint32_t weights[8] = {weight_low.x, weight_low.y, weight_low.z, weight_low.w,
                                          weight_high.x, weight_high.y, weight_high.z,
                                          weight_high.w};
        const std::uint32_t values[8]  = {value_low.x, value_low.y, value_low.z, value_low.w,
                                          value_high.x, value_high.y, value_high.z, value_high.w};
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const float2 weight = bf16x2_bits_to_float2(weights[pair]);
            const float2 value  = MoeAct<A>::pair_to_float2(values[pair]);
            accumulator[(2 * pair) & 3] =
                fmaf(weight.x, value.x, accumulator[(2 * pair) & 3]);
            accumulator[(2 * pair + 1) & 3] =
                fmaf(weight.y, value.y, accumulator[(2 * pair + 1) & 3]);
        }
    }
    float total = 0.0F;
#pragma unroll
    for (int chain = 0; chain < 4; ++chain) { total += accumulator[chain]; }
    return warp_reduce_sum(total);
}

template <class A>
__global__ __launch_bounds__(512) void moe_nvfp4_route_kernel(
    const A* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    std::int32_t* __restrict__ ids, float* __restrict__ coefficients,
    float* __restrict__ shared_gate) {
    const int token   = static_cast<int>(blockIdx.x);
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int warps   = static_cast<int>(blockDim.x) >> 5;
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int row     = static_cast<int>(threadIdx.x);

    __shared__ float logits[kRouteRows];
    __shared__ RankedValue warp_best[16];
    __shared__ float selected_value[kMoeTopK];

    const A* x_column = x + static_cast<std::int64_t>(token) * kMoeHidden;
    for (int index = warp; index < kRouteRows; index += warps) {
        const std::uint32_t* weight_words = reinterpret_cast<const std::uint32_t*>(
            control + static_cast<std::int64_t>(index) * kMoeHidden);
        logits[index] = route_row_dot<A>(weight_words, x_column, lane);
    }
    __syncthreads();

    for (int rank = 0; rank < kMoeTopK; ++rank) {
        RankedValue candidate{logits[row], row};
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            RankedValue other;
            other.value = __shfl_down_sync(kFullWarpMask, candidate.value, offset);
            other.id    = __shfl_down_sync(kFullWarpMask, candidate.id, offset);
            if (ranked_better(other, candidate)) candidate = other;
        }
        if (lane == 0) warp_best[warp] = candidate;
        __syncthreads();
        if (row == 0) {
            RankedValue overall = warp_best[0];
            for (int w = 1; w < warps; ++w) {
                if (ranked_better(warp_best[w], overall)) overall = warp_best[w];
            }
            ids[rank + kMoeTopK * token]   = overall.id;
            selected_value[rank]           = overall.value;
            logits[overall.id]             = -CUDART_INF_F;
        }
        __syncthreads();
    }

    if (warp == 0) {
        float exponential = lane < kMoeTopK ? expf(selected_value[lane] - selected_value[0]) : 0.0F;
        const float denominator = warp_sum(exponential);
        if (lane < kMoeTopK) coefficients[lane + kMoeTopK * token] = exponential / denominator;
    }
    if (row == 0) shared_gate[token] = sigmoid(logits[kMoeExperts]);
}

template <class A>
__global__ __launch_bounds__(kRouteWarpsPerCta * 32) void moe_nvfp4_route_logits_kernel(
    const A* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    float* __restrict__ logits) {
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * kRouteWarpsPerCta + warp;
    if (row >= kRouteRows) { return; }

    const std::uint32_t* weight_words =
        reinterpret_cast<const std::uint32_t*>(control + static_cast<std::int64_t>(row) * kMoeHidden);
    const A* x_column = x + static_cast<std::int64_t>(token) * kMoeHidden;
    logits[static_cast<std::int64_t>(token) * kRouteRows + row] =
        route_row_dot<A>(weight_words, x_column, lane);
}

__global__ __launch_bounds__(512) void moe_nvfp4_route_select_kernel(
    const float* __restrict__ logits_in, std::int32_t* __restrict__ ids,
    float* __restrict__ coefficients, float* __restrict__ shared_gate) {
    const int token = static_cast<int>(blockIdx.x);
    const int row   = static_cast<int>(threadIdx.x);
    const int lane  = row & 31;
    const int warp  = row >> 5;

    __shared__ float logits[kRouteRows];
    __shared__ RankedValue warp_best[16];
    __shared__ float selected_value[kMoeTopK];

    const float* source = logits_in + static_cast<std::int64_t>(token) * kRouteRows;
    logits[row] = source[row];
    if (row == 0) { logits[kMoeExperts] = source[kMoeExperts]; }
    __syncthreads();

    for (int rank = 0; rank < kMoeTopK; ++rank) {
        RankedValue candidate{logits[row], row};
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            RankedValue other;
            other.value = __shfl_down_sync(kFullWarpMask, candidate.value, offset);
            other.id    = __shfl_down_sync(kFullWarpMask, candidate.id, offset);
            if (ranked_better(other, candidate)) candidate = other;
        }
        if (lane == 0) warp_best[warp] = candidate;
        __syncthreads();
        if (row == 0) {
            RankedValue overall = warp_best[0];
            for (int w = 1; w < 16; ++w) {
                if (ranked_better(warp_best[w], overall)) overall = warp_best[w];
            }
            ids[rank + kMoeTopK * token]   = overall.id;
            selected_value[rank]           = overall.value;
            logits[overall.id]             = -CUDART_INF_F;
        }
        __syncthreads();
    }

    if (warp == 0) {
        float exponential = lane < kMoeTopK ? expf(selected_value[lane] - selected_value[0]) : 0.0F;
        const float denominator = warp_sum(exponential);
        if (lane < kMoeTopK) coefficients[lane + kMoeTopK * token] = exponential / denominator;
    }
    if (row == 0) shared_gate[token] = sigmoid(logits[kMoeExperts]);
}

} // namespace

void moe_nvfp4_route_launch(const Tensor& x, const void* control, MoeNvfp4RouteOutputs outputs,
                            float* logits, cudaStream_t stream) {
    const bool split = x.ne[1] == 1 && logits != nullptr;
    if (split) {
        const dim3 logits_grid((kRouteRows + kRouteWarpsPerCta - 1) / kRouteWarpsPerCta, x.ne[1]);
        if (x.dtype == DType::FP16) {
            moe_nvfp4_route_logits_kernel<__half>
                <<<logits_grid, kRouteWarpsPerCta * 32, 0, stream>>>(
                    static_cast<const __half*>(x.data),
                    static_cast<const __nv_bfloat16*>(control), logits);
        } else {
            moe_nvfp4_route_logits_kernel<__nv_bfloat16>
                <<<logits_grid, kRouteWarpsPerCta * 32, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const __nv_bfloat16*>(control), logits);
        }
        moe_nvfp4_route_select_kernel<<<x.ne[1], 512, 0, stream>>>(
            logits, outputs.ids, outputs.coefficients, outputs.shared_gate);
    } else if (x.dtype == DType::FP16) {
        moe_nvfp4_route_kernel<__half><<<x.ne[1], 512, 0, stream>>>(
            static_cast<const __half*>(x.data), static_cast<const __nv_bfloat16*>(control),
            outputs.ids, outputs.coefficients, outputs.shared_gate);
    } else {
        moe_nvfp4_route_kernel<__nv_bfloat16><<<x.ne[1], 512, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(control),
            outputs.ids, outputs.coefficients, outputs.shared_gate);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
