#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>
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

__global__ __launch_bounds__(512) void moe_nvfp4_route_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    std::int32_t* __restrict__ ids, float* __restrict__ coefficients, float* __restrict__ shared_gate) {
    const int token = static_cast<int>(blockIdx.x);
    const int row   = static_cast<int>(threadIdx.x);
    const int lane  = row & 31;
    const int warp  = row >> 5;

    __shared__ float logits[kMoeExperts + 1];
    __shared__ RankedValue warp_best[16];
    __shared__ float selected_value[kMoeTopK];

    const std::uint32_t* weight_words =
        reinterpret_cast<const std::uint32_t*>(control + static_cast<std::int64_t>(row) * kMoeHidden);
    const std::uint32_t* activation_pairs =
        reinterpret_cast<const std::uint32_t*>(x + static_cast<std::int64_t>(token) * kMoeHidden);

    float score = 0.0F;
    for (int pair = 0; pair < kMoeHidden / 2; ++pair) {
        const float2 weight = bf16x2_bits_to_float2(weight_words[pair]);
        const float2 value  = bf16x2_bits_to_float2(activation_pairs[pair]);
        score = fmaf(weight.x, value.x, score);
        score = fmaf(weight.y, value.y, score);
    }
    logits[row] = score;
    if (row == 0) {
        const std::uint32_t* shared_score_row =
            reinterpret_cast<const std::uint32_t*>(control + kMoeExperts * kMoeHidden);
        float shared_score = 0.0F;
        for (int pair = 0; pair < kMoeHidden / 2; ++pair) {
            const float2 weight = bf16x2_bits_to_float2(shared_score_row[pair]);
            const float2 value  = bf16x2_bits_to_float2(activation_pairs[pair]);
            shared_score = fmaf(weight.x, value.x, shared_score);
            shared_score = fmaf(weight.y, value.y, shared_score);
        }
        logits[kMoeExperts] = shared_score;
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

} // namespace

void moe_nvfp4_route_launch(const Tensor& x, const void* control, MoeNvfp4RouteOutputs outputs,
                            cudaStream_t stream) {
    moe_nvfp4_route_kernel<<<x.ne[1], 512, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(control),
        outputs.ids, outputs.coefficients, outputs.shared_gate);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
