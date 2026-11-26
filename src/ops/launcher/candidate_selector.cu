// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime: ninfer::core Tensor/Weight/WorkspaceArena
// operands, sm_70 SIMT routes, and the real DFlash2 companion geometry (16 candidates over the
// target vocabulary, rank-256 selector). The candidate path/top-k mathematics is unchanged.
#include "ops/launcher/candidate_selector.h"
#include <type_traits>

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kCandidates = 16;
constexpr int kRank       = 256;
constexpr int kTopKBlock  = 128;
constexpr int kTopKSplit  = 32;

struct Candidate {
    float value;
    std::int32_t id;
};

__device__ __forceinline__ bool better(const Candidate& a, const Candidate& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

__device__ __forceinline__ void insert(Candidate (&top)[kCandidates], const Candidate& c) {
    if (!better(c, top[kCandidates - 1])) { return; }
    int p = kCandidates - 1;
    while (p > 0 && better(c, top[p - 1])) {
        top[p] = top[p - 1];
        --p;
    }
    top[p] = c;
}

__device__ void block_merge(Candidate* storage, const Candidate (&top)[kCandidates]) {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int k = 0; k < kCandidates; ++k) { storage[tid * kCandidates + k] = top[k]; }
    __syncthreads();
    for (int step = kTopKBlock / 2; step > 0; step >>= 1) {
        if (tid < step) {
            const Candidate* left  = storage + tid * kCandidates;
            const Candidate* right = storage + (tid + step) * kCandidates;
            Candidate out[kCandidates];
            int i = 0, j = 0;
#pragma unroll
            for (int k = 0; k < kCandidates; ++k) {
                const bool take_left = better(left[i], right[j]) || !better(right[j], left[i]);
                out[k]               = take_left ? left[i] : right[j];
                i += take_left ? 1 : 0;
                j += take_left ? 0 : 1;
            }
#pragma unroll
            for (int k = 0; k < kCandidates; ++k) { storage[tid * kCandidates + k] = out[k]; }
        }
        __syncthreads();
    }
}

template <typename T>
__device__ __forceinline__ float logit(const T* p, std::int64_t i) {
    if constexpr (std::is_same_v<T, float>) {
        return p[i];
    } else {
        return __bfloat162float(p[i]);
    }
}

template <typename T>
__global__ void __launch_bounds__(kTopKBlock)
    candidate_topk_partials(const T* __restrict__ logits, std::int64_t rows,
                            std::int32_t valid_rows, Candidate* __restrict__ partials) {
    __shared__ Candidate storage[kTopKBlock * kCandidates];
    const int column = static_cast<int>(blockIdx.y);
    const int chunk  = (valid_rows + kTopKSplit - 1) / kTopKSplit;
    const int begin  = static_cast<int>(blockIdx.x) * chunk;
    const int end    = min(begin + chunk, valid_rows);
    const T* source  = logits + std::int64_t(column) * rows;
    Candidate top[kCandidates];
#pragma unroll
    for (int k = 0; k < kCandidates; ++k) { top[k] = Candidate{-CUDART_INF_F, INT_MAX}; }
    for (int v = begin + static_cast<int>(threadIdx.x); v < end; v += kTopKBlock) {
        insert(top, Candidate{logit(source, v), v});
    }
    block_merge(storage, top);
    if (threadIdx.x < kCandidates) {
        partials[(std::int64_t(column) * kTopKSplit + blockIdx.x) * kCandidates + threadIdx.x] =
            storage[threadIdx.x];
    }
}

__global__ void __launch_bounds__(kTopKBlock)
    candidate_topk_combine(const Candidate* __restrict__ partials, std::int32_t* __restrict__ ids,
                           float* __restrict__ scores) {
    __shared__ Candidate storage[kTopKBlock * kCandidates];
    const int column = static_cast<int>(blockIdx.x);
    Candidate top[kCandidates];
#pragma unroll
    for (int k = 0; k < kCandidates; ++k) {
        top[k] = threadIdx.x < kTopKSplit
                     ? partials[(std::int64_t(column) * kTopKSplit + threadIdx.x) * kCandidates + k]
                     : Candidate{-CUDART_INF_F, INT_MAX};
    }
    block_merge(storage, top);
    if (threadIdx.x < kCandidates) {
        const Candidate c = storage[threadIdx.x];
        ids[std::int64_t(column) * kCandidates + threadIdx.x] =
            c.id == INT_MAX ? 0 : c.id;
        scores[std::int64_t(column) * kCandidates + threadIdx.x] = c.value;
    }
}

struct DeviceArgs {
    const std::int32_t* ids;
    const float* unary;
    const __nv_bfloat16* hidden;
    const std::int32_t* anchors;
    const __nv_bfloat16* predecessor;
    const __nv_bfloat16* successor;
    const std::int32_t* positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* q;
    float* confidence;
    int steps;
    float softcap;
};

struct alignas(16) SelectorShared {
    float product[kRank];
    float edge[kCandidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
};

__device__ int draw_from(float edge, float maximum, float temperature, unsigned long long seed,
                         int position, float* q) {
    const int lane = threadIdx.x & 31;
    if (temperature <= 0.0F) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kCandidates) q[lane] = lane == selected ? 1.0F : 0.0F;
        return selected;
    }
    const float weight      = lane < kCandidates ? __expf((edge - maximum) / temperature) : 0.0F;
    const float probability = weight / warp_sum(weight);
    if (lane < kCandidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, 0U) : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits = __ballot_sync(kFullWarpMask, lane < kCandidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kCandidates - 1;
}

__device__ int draw_rank(float edge, float temperature, unsigned long long seed, int position,
                         float* q, float* confidence) {
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    const int selected  = draw_from(edge, maximum, temperature, seed, position, q);
    if (confidence != nullptr) {
        const float weight = lane < kCandidates ? __expf(edge - maximum) : 0.0F;
        const float total  = warp_sum(weight);
        const float mine   = __shfl_sync(kFullWarpMask, weight, selected);
        if (lane == 0) *confidence = mine / total;
    }
    return selected;
}

__device__ void score_row(const DeviceArgs& a, int column, SelectorShared& shared) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int token = lane == 0 ? a.ids[column * kCandidates + warp] : 0;
    token     = __shfl_sync(kFullWarpMask, token, 0);
    __syncthreads();
    if (tid < kRank) {
        const std::int64_t predecessor = shared.predecessor;
        shared.product[tid] =
            __bfloat162float(a.predecessor[predecessor * kRank + tid]) *
            __bfloat162float(a.hidden[static_cast<std::int64_t>(column) * kRank + tid]);
    }
    __syncthreads();
    {
        const int c = warp;
        float sum   = 0;
#pragma unroll
        for (int r = lane; r < kRank; r += 32) {
            sum = fmaf(shared.product[r],
                       __bfloat162float(a.successor[static_cast<std::int64_t>(token) * kRank + r]),
                       sum);
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            const float edge = a.unary[column * kCandidates + c] + sum;
            shared.edge[c]   = a.softcap > 0.0F ? a.softcap * tanhf(edge / a.softcap) : edge;
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(512, 1) void selector_walk_kernel(DeviceArgs a) {
    __shared__ SelectorShared shared;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    if (tid == 0) {
        shared.predecessor   = a.anchors[batch];
        shared.base_position = a.positions[batch];
        shared.temperature   = a.configs[batch].temperature;
        shared.seed          = a.configs[batch].seed;
    }
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        score_row(a, column, shared);
        if (warp == 0) {
            const float edge   = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = draw_rank(edge, shared.temperature, shared.seed,
                                           shared.base_position + step,
                                           a.q + column * kCandidates,
                                           a.confidence ? a.confidence + column : nullptr);
            if (lane == 0) {
                shared.predecessor = a.ids[column * kCandidates + selected];
                a.drafts[column]   = shared.predecessor;
            }
        }
    }
}

__global__ __launch_bounds__(512, 2) void selector_lattice_kernel(DeviceArgs a, float* edges) {
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared shared;
    if (threadIdx.x == 0)
        shared.predecessor =
            step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kCandidates + p];
    score_row(a, column, shared);
    if (threadIdx.x < kCandidates)
        edges[(static_cast<std::int64_t>(column) * kCandidates + p) * kCandidates + threadIdx.x] =
            shared.edge[threadIdx.x];
}

__global__ __launch_bounds__(32) void selector_lattice_walk_kernel(DeviceArgs a,
                                                                   const float* edges) {
    const int lane = threadIdx.x, batch = blockIdx.x;
    const auto seed         = a.configs[batch].seed;
    const float temperature = a.configs[batch].temperature;
    const int position      = a.positions[batch];
    int predecessor_rank    = 0;
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const float edge =
            lane < kCandidates
                ? edges[(static_cast<std::int64_t>(column) * kCandidates + predecessor_rank) *
                            kCandidates +
                        lane]
                : -CUDART_INF_F;
        const int selected = draw_rank(edge, temperature, seed, position + step,
                                       a.q + column * kCandidates,
                                       a.confidence ? a.confidence + column : nullptr);
        predecessor_rank = selected;
        if (lane == 0) a.drafts[column] = a.ids[column * kCandidates + selected];
    }
}

} // namespace

void candidate_topk_launch(const Tensor& logits, std::int32_t valid_rows, Tensor& candidate_ids,
                           Tensor& candidate_scores, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t rows    = logits.ne[0];
    const std::int32_t columns = logits.ne[1];
    auto* partials = static_cast<Candidate*>(workspace.data);
    const dim3 grid(kTopKSplit, static_cast<unsigned>(columns));
    if (logits.dtype == DType::FP32) {
        candidate_topk_partials<float><<<grid, kTopKBlock, 0, stream>>>(
            static_cast<const float*>(logits.data), rows, valid_rows, partials);
    } else {
        candidate_topk_partials<__nv_bfloat16><<<grid, kTopKBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), rows, valid_rows, partials);
    }
    CUDA_CHECK(cudaGetLastError());
    candidate_topk_combine<<<static_cast<unsigned>(columns), kTopKBlock, 0, stream>>>(
        partials, static_cast<std::int32_t*>(candidate_ids.data),
        static_cast<float*>(candidate_scores.data));
    CUDA_CHECK(cudaGetLastError());
}

void candidate_selector_path_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                    const Tensor& projected_hidden, const Tensor& anchors,
                                    const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook,
                                    const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, float* confidence, float edge_softcap,
                                    DeviceSpan workspace, cudaStream_t stream) {
    DeviceArgs args{
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<const float*>(unary_scores.data),
        static_cast<const __nv_bfloat16*>(projected_hidden.data),
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
        static_cast<const __nv_bfloat16*>(successor_codebook.data),
        static_cast<const std::int32_t*>(base_positions.data),
        configs,
        static_cast<std::int32_t*>(drafts.data),
        static_cast<float*>(proposal_q.data),
        confidence,
        candidate_ids.ne[1],
        edge_softcap};
    if (candidate_ids.ne[1] <= 4) {
        selector_walk_kernel<<<static_cast<unsigned>(candidate_ids.ne[2]), 512, 0, stream>>>(args);
    } else {
        auto* edges = static_cast<float*>(workspace.data);
        selector_lattice_kernel<<<dim3(static_cast<unsigned>(args.steps * candidate_ids.ne[2]),
                                       kCandidates),
                                  512, 0, stream>>>(args, edges);
        CUDA_CHECK(cudaGetLastError());
        selector_lattice_walk_kernel<<<static_cast<unsigned>(candidate_ids.ne[2]), 32, 0, stream>>>(
            args, edges);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
