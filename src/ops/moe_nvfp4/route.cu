#include "ninfer/ops/moe_nvfp4.h"

#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <type_traits>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct RankedValue {
    float value;
    int id;
};

__device__ __forceinline__ bool ranked_better(const RankedValue& a, const RankedValue& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

__device__ __forceinline__ RankedValue rank_max(const RankedValue& a, const RankedValue& b) {
    const bool better = (b.value > a.value) | ((b.value == a.value) & (b.id < a.id));
    RankedValue result;
    result.value = better ? b.value : a.value;
    result.id    = better ? b.id : a.id;
    return result;
}

inline constexpr int kRouteRows        = kMoeExperts + 1;
inline constexpr int kRouteGroups      = kMoeHidden / 16;
inline constexpr int kRoutePhases      = kRouteGroups / 32;
static_assert(kRouteGroups % 32 == 0);
inline constexpr int kRouteWarpsPerCta = 8;
inline constexpr int kRouteCtas        = (kRouteRows + kRouteWarpsPerCta - 1) / kRouteWarpsPerCta;
inline constexpr int kRouteSlots       = (kMoeExperts + 31) / 32;

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

// Multi-column counterpart of route_row_dot. The router weight row is fetched once per phase and
// reused for every token column; each column keeps its own four-chain accumulator with the exact
// same pair order, chain sum and warp reduction as the T==1 kernel.
template <class A, int Columns>
__device__ __forceinline__ void route_row_dot_multi(const std::uint32_t* weight_words,
                                                    const A* activation_columns, int lane,
                                                    float (&totals)[Columns]) {
    const auto* activation_words = reinterpret_cast<const std::uint32_t*>(activation_columns);
#pragma unroll
    for (int column = 0; column < Columns; ++column) {
        totals[column] = 0.0F;
    }
    float accumulators[Columns][4] = {};
#pragma unroll
    for (int phase = 0; phase < kRoutePhases; ++phase) {
        const int group = phase * 32 + lane;
        const uint4 weight_low  = load_vec<uint4>(weight_words + 8 * group);
        const uint4 weight_high = load_vec<uint4>(weight_words + 8 * group + 4);
        const std::uint32_t weights[8] = {weight_low.x, weight_low.y, weight_low.z, weight_low.w,
                                          weight_high.x, weight_high.y, weight_high.z,
                                          weight_high.w};
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            const std::uint32_t* column_words =
                activation_words + static_cast<std::size_t>(column) * (kMoeHidden / 2);
            const uint4 value_low  = load_vec<uint4>(column_words + 8 * group);
            const uint4 value_high = load_vec<uint4>(column_words + 8 * group + 4);
            const std::uint32_t values[8] = {value_low.x, value_low.y, value_low.z, value_low.w,
                                             value_high.x, value_high.y, value_high.z,
                                             value_high.w};
#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                const float2 weight = bf16x2_bits_to_float2(weights[pair]);
                const float2 value  = MoeAct<A>::pair_to_float2(values[pair]);
                accumulators[column][(2 * pair) & 3] =
                    fmaf(weight.x, value.x, accumulators[column][(2 * pair) & 3]);
                accumulators[column][(2 * pair + 1) & 3] =
                    fmaf(weight.y, value.y, accumulators[column][(2 * pair + 1) & 3]);
            }
        }
    }
#pragma unroll
    for (int column = 0; column < Columns; ++column) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < 4; ++chain) { total += accumulators[column][chain]; }
        totals[column] = warp_reduce_sum(total);
    }
}

// Single-warp top-k tournament over kMoeExperts logits. Selection order and tie-breaking are
// (value desc, id asc); coefficients consume the same selected values in the same order, so the
// outputs are bitwise identical for identical logits.
__device__ __forceinline__ void route_select(const float* __restrict__ logits,
                                              std::int32_t* __restrict__ ids,
                                              float* __restrict__ coefficients,
                                              float* __restrict__ shared_gate) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    __shared__ float selected_value[kMoeTopK];

    float values[kRouteSlots];
#pragma unroll
    for (int slot = 0; slot < kRouteSlots; ++slot) {
        const int index = lane + 32 * slot;
        values[slot]    = index < kMoeExperts ? logits[index] : -CUDART_INF_F;
    }

    for (int rank = 0; rank < kMoeTopK; ++rank) {
        RankedValue candidate{values[0], lane};
#pragma unroll
        for (int slot = 1; slot < kRouteSlots; ++slot) {
            const int index = lane + 32 * slot;
            candidate       = rank_max(candidate, RankedValue{values[slot], index});
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            RankedValue other;
            other.value = __shfl_down_sync(kFullWarpMask, candidate.value, offset);
            other.id    = __shfl_down_sync(kFullWarpMask, candidate.id, offset);
            candidate   = rank_max(candidate, other);
        }
        const int chosen = __shfl_sync(kFullWarpMask, candidate.id, 0);
        if (lane == 0) {
            ids[rank]            = chosen;
            selected_value[rank] = candidate.value;
        }
#pragma unroll
        for (int slot = 0; slot < kRouteSlots; ++slot) {
            const int index = lane + 32 * slot;
            if (index == chosen) { values[slot] = -CUDART_INF_F; }
        }
    }

    const float exponential =
        lane < kMoeTopK ? expf(selected_value[lane] - selected_value[0]) : 0.0F;
    const float denominator = warp_sum(exponential);
    if (lane < kMoeTopK) { coefficients[lane] = exponential / denominator; }
    if (lane == 0) { shared_gate[0] = sigmoid(logits[kMoeExperts]); }
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

__global__ __launch_bounds__(32) void moe_nvfp4_route_select_kernel(
    const float* __restrict__ logits, std::int32_t* __restrict__ ids,
    float* __restrict__ coefficients, float* __restrict__ shared_gate) {
    route_select(logits, ids, coefficients, shared_gate);
}

template <class A>
__global__ __launch_bounds__(kRouteWarpsPerCta * 32) void moe_nvfp4_route_fused_kernel(
    const A* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    std::int32_t* __restrict__ ids, float* __restrict__ coefficients,
    float* __restrict__ shared_gate, float* __restrict__ logits,
    std::int32_t* __restrict__ counter) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * kRouteWarpsPerCta + warp;
    if (row < kRouteRows) {
        const std::uint32_t* weight_words = reinterpret_cast<const std::uint32_t*>(
            control + static_cast<std::int64_t>(row) * kMoeHidden);
        logits[row] = route_row_dot<A>(weight_words, x, lane);
    }
    __threadfence();

    __shared__ int is_last;
    if (threadIdx.x == 0) {
        is_last = atomicAdd(counter, 1) == kRouteCtas - 1 ? 1 : 0;
    }
    __syncthreads();
    if (is_last == 0) { return; }

    if (warp == 0) { route_select(logits, ids, coefficients, shared_gate); }
    if (threadIdx.x == 0) { *counter = 0; }
}

// True batched counterpart of moe_nvfp4_route_fused_kernel. One grid launch evaluates all token
// columns: each router weight row is loaded once and applied to every column with the T==1 dot
// order, and the last CTA runs the unchanged single-warp top-k tournament once per column.
template <class A, int Columns>
__global__ __launch_bounds__(kRouteWarpsPerCta * 32) void moe_nvfp4_route_fused_multi_kernel(
    const A* __restrict__ x, const __nv_bfloat16* __restrict__ control,
    std::int32_t* __restrict__ ids, float* __restrict__ coefficients,
    float* __restrict__ shared_gate, float* __restrict__ logits,
    std::int32_t* __restrict__ counter) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * kRouteWarpsPerCta + warp;
    if (row < kRouteRows) {
        const std::uint32_t* weight_words = reinterpret_cast<const std::uint32_t*>(
            control + static_cast<std::int64_t>(row) * kMoeHidden);
        float totals[Columns];
        route_row_dot_multi<A, Columns>(weight_words, x, lane, totals);
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            logits[static_cast<std::size_t>(column) * kRouteRows + row] = totals[column];
        }
    }
    __threadfence();

    __shared__ int is_last;
    if (threadIdx.x == 0) {
        is_last = atomicAdd(counter, 1) == kRouteCtas - 1 ? 1 : 0;
    }
    __syncthreads();
    if (is_last == 0) { return; }

    if (warp == 0) {
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            route_select(logits + static_cast<std::size_t>(column) * kRouteRows,
                         ids + column * kMoeTopK, coefficients + column * kMoeTopK,
                         shared_gate + column);
        }
    }
    if (threadIdx.x == 0) { *counter = 0; }
}

} // namespace

void moe_nvfp4_route_launch(const Tensor& x, const void* control, MoeNvfp4RouteOutputs outputs,
                            float* logits, std::int32_t* counter, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (logits != nullptr && counter != nullptr && tokens >= 1 &&
        tokens <= kMoeRouteFusedMaxTokens) {
        if (tokens == 1) {
            if (x.dtype == DType::FP16) {
                moe_nvfp4_route_fused_kernel<__half>
                    <<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(
                        static_cast<const __half*>(x.data),
                        static_cast<const __nv_bfloat16*>(control), outputs.ids,
                        outputs.coefficients, outputs.shared_gate, logits, counter);
            } else {
                moe_nvfp4_route_fused_kernel<__nv_bfloat16>
                    <<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(x.data),
                        static_cast<const __nv_bfloat16*>(control), outputs.ids,
                        outputs.coefficients, outputs.shared_gate, logits, counter);
            }
        } else {
            if (x.dtype == DType::FP16) {
                const auto* xp = static_cast<const __half*>(x.data);
                switch (tokens) {
                case 2: moe_nvfp4_route_fused_multi_kernel<__half, 2><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 3: moe_nvfp4_route_fused_multi_kernel<__half, 3><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 4: moe_nvfp4_route_fused_multi_kernel<__half, 4><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 5: moe_nvfp4_route_fused_multi_kernel<__half, 5><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 6: moe_nvfp4_route_fused_multi_kernel<__half, 6><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 7: moe_nvfp4_route_fused_multi_kernel<__half, 7><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 8: moe_nvfp4_route_fused_multi_kernel<__half, 8><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                default: throw std::invalid_argument("moe route multi: unsupported column count");
                }
            } else {
                const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
                switch (tokens) {
                case 2: moe_nvfp4_route_fused_multi_kernel<__nv_bfloat16, 2><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 3: moe_nvfp4_route_fused_multi_kernel<__nv_bfloat16, 3><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 4: moe_nvfp4_route_fused_multi_kernel<__nv_bfloat16, 4><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 5: moe_nvfp4_route_fused_multi_kernel<__nv_bfloat16, 5><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 6: moe_nvfp4_route_fused_multi_kernel<__nv_bfloat16, 6><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 7: moe_nvfp4_route_fused_multi_kernel<__nv_bfloat16, 7><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                case 8: moe_nvfp4_route_fused_multi_kernel<__nv_bfloat16, 8><<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(xp, static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients, outputs.shared_gate, logits, counter); break;
                default: throw std::invalid_argument("moe route multi: unsupported column count");
                }
            }
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool split = x.ne[1] == 1 && logits != nullptr;
    if (split && counter != nullptr) {
        if (x.dtype == DType::FP16) {
            moe_nvfp4_route_fused_kernel<__half>
                <<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(
                    static_cast<const __half*>(x.data),
                    static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients,
                    outputs.shared_gate, logits, counter);
        } else {
            moe_nvfp4_route_fused_kernel<__nv_bfloat16>
                <<<kRouteCtas, kRouteWarpsPerCta * 32, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const __nv_bfloat16*>(control), outputs.ids, outputs.coefficients,
                    outputs.shared_gate, logits, counter);
        }
    } else if (split) {
        const dim3 logits_grid(kRouteCtas, x.ne[1]);
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
        moe_nvfp4_route_select_kernel<<<x.ne[1], 32, 0, stream>>>(
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
