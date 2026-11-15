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

template <class A>
__device__ __forceinline__ float route_warp_dot(const std::uint32_t* weight_words,
                                                const A* activation, std::int32_t pairs) {
    float score = 0.0F;
#pragma unroll 64
    for (int pair = 0; pair < pairs; ++pair) {
        const float2 weight = bf16x2_bits_to_float2(weight_words[pair]);
        const float2 value =
            MoeAct<A>::pair_to_float2(*reinterpret_cast<const std::uint32_t*>(activation + 2 * pair));
        score = fmaf(weight.x, value.x, score);
        score = fmaf(weight.y, value.y, score);
    }
    return score;
}

template <class A>
__global__ __launch_bounds__(512) void moe_nvfp4_route_kernel(
    const A* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    std::int32_t* __restrict__ ids, float* __restrict__ coefficients,
    float* __restrict__ shared_gate) {
    const int token = static_cast<int>(blockIdx.x);
    const int row   = static_cast<int>(threadIdx.x);
    const int lane  = row & 31;
    const int warp  = row >> 5;

    __shared__ float logits[kMoeExperts + 1];
    __shared__ RankedValue warp_best[16];
    __shared__ float selected_value[kMoeTopK];

    const std::uint32_t* weight_words =
        reinterpret_cast<const std::uint32_t*>(control + static_cast<std::int64_t>(row) * kMoeHidden);
    const A* x_column = x + static_cast<std::int64_t>(token) * kMoeHidden;

    logits[row] = route_warp_dot<A>(weight_words, x_column, kMoeHidden / 2);
    if (row == 0) {
        const std::uint32_t* shared_score_row =
            reinterpret_cast<const std::uint32_t*>(control + kMoeExperts * kMoeHidden);
        logits[kMoeExperts] = route_warp_dot<A>(shared_score_row, x_column, kMoeHidden / 2);
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

constexpr int kRouteRowTile = 4;

template <class A>
__global__ __launch_bounds__(kRouteRowTile) void moe_nvfp4_route_logits_kernel(
    const A* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    float* __restrict__ logits) {
    const int token = static_cast<int>(blockIdx.y);
    const int row   = static_cast<int>(blockIdx.x) * kRouteRowTile + static_cast<int>(threadIdx.x);

    const std::uint32_t* weight_words =
        reinterpret_cast<const std::uint32_t*>(control + static_cast<std::int64_t>(row) * kMoeHidden);
    const A* x_column = x + static_cast<std::int64_t>(token) * kMoeHidden;

    logits[static_cast<std::int64_t>(token) * (kMoeExperts + 1) + row] =
        route_warp_dot<A>(weight_words, x_column, kMoeHidden / 2);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const std::uint32_t* shared_score_row =
            reinterpret_cast<const std::uint32_t*>(control + kMoeExperts * kMoeHidden);
        logits[static_cast<std::int64_t>(token) * (kMoeExperts + 1) + kMoeExperts] =
            route_warp_dot<A>(shared_score_row, x_column, kMoeHidden / 2);
    }
}

__global__ __launch_bounds__(512) void moe_nvfp4_route_select_kernel(
    const float* __restrict__ logits_in, std::int32_t* __restrict__ ids,
    float* __restrict__ coefficients, float* __restrict__ shared_gate) {
    const int token = static_cast<int>(blockIdx.x);
    const int row   = static_cast<int>(threadIdx.x);
    const int lane  = row & 31;
    const int warp  = row >> 5;

    __shared__ float logits[kMoeExperts + 1];
    __shared__ RankedValue warp_best[16];
    __shared__ float selected_value[kMoeTopK];

    const float* source = logits_in + static_cast<std::int64_t>(token) * (kMoeExperts + 1);
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
        const dim3 logits_grid(kMoeExperts / kRouteRowTile);
        if (x.dtype == DType::FP16) {
            moe_nvfp4_route_logits_kernel<__half><<<logits_grid, kRouteRowTile, 0, stream>>>(
                static_cast<const __half*>(x.data),
                static_cast<const __nv_bfloat16*>(control), logits);
        } else {
            moe_nvfp4_route_logits_kernel<__nv_bfloat16><<<logits_grid, kRouteRowTile, 0, stream>>>(
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
