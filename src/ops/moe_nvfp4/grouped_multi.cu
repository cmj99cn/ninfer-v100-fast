#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <type_traits>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

inline constexpr int kMoeDecodeThreads    = 128;
inline constexpr int kMoeDecodeWarps      = kMoeDecodeThreads / 32;
inline constexpr int kMoeDecodeGateUpTile = kMoeDecodeWarps;
inline constexpr int kMoeDecodeDownTile   = 2 * kMoeDecodeWarps;
static_assert(kMoeIntermediate % kMoeDecodeGateUpTile == 0);
static_assert(kMoeHidden % kMoeDecodeDownTile == 0);

template <class A, int R>
__device__ __noinline__ void grouped_gate_up_rows(
    const std::uint8_t* __restrict__ gate_codes, const std::uint8_t* __restrict__ gate_scales,
    float inverse_gate, const std::uint8_t* __restrict__ up_codes,
    const std::uint8_t* __restrict__ up_scales, float inverse_up, const int (&rows)[R],
    const A* __restrict__ activation, const float2* __restrict__ decode_lut,
    const float* __restrict__ scale_lut, float (&gate)[R], float (&up)[R]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float gate_acc[R];
    float up_acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        gate_acc[r] = 0.0F;
        up_acc[r]   = 0.0F;
    }
    for (int group = lane; group < kMoeHidden / 16; group += 32) {
        const uint4 low  = load_vec<uint4>(activation + 16 * group);
        const uint4 high = load_vec<uint4>(activation + 16 * group + 8);
        const std::uint32_t words[8] = {low.x, low.y, low.z, low.w,
                                        high.x, high.y, high.z, high.w};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = rows[r];
            const float gate_scale =
                scale_lut[gate_scales[batched_nvfp4_scale_offset(row, group, kMoeHidden)]] *
                inverse_gate;
            const float up_scale =
                scale_lut[up_scales[batched_nvfp4_scale_offset(row, group, kMoeHidden)]] *
                inverse_up;
            const std::uint64_t gate_word = load_vec<std::uint64_t>(
                gate_codes + static_cast<std::size_t>(row) * (kMoeHidden / 2U) +
                static_cast<std::size_t>(group) * 8U);
            const std::uint64_t up_word = load_vec<std::uint64_t>(
                up_codes + static_cast<std::size_t>(row) * (kMoeHidden / 2U) +
                static_cast<std::size_t>(group) * 8U);
#pragma unroll
            for (int byte = 0; byte < 8; ++byte) {
                const float2 gate_code =
                    decode_lut[static_cast<std::uint8_t>(gate_word >> (8 * byte))];
                const float2 up_code =
                    decode_lut[static_cast<std::uint8_t>(up_word >> (8 * byte))];
                const float2 value = MoeAct<A>::pair_to_float2(words[byte]);
                gate_acc[r] = fmaf(gate_code.x * gate_scale, value.x, gate_acc[r]);
                gate_acc[r] = fmaf(gate_code.y * gate_scale, value.y, gate_acc[r]);
                up_acc[r] = fmaf(up_code.x * up_scale, value.x, up_acc[r]);
                up_acc[r] = fmaf(up_code.y * up_scale, value.y, up_acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        gate[r] = warp_reduce_sum(gate_acc[r]);
        up[r]   = warp_reduce_sum(up_acc[r]);
    }
}

template <int R>
__device__ __noinline__ void grouped_down_rows(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    float inverse, const int (&rows)[R], const float* __restrict__ activation,
    const float2* __restrict__ decode_lut, const float* __restrict__ scale_lut,
    float (&value)[R]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float accumulator[R];
#pragma unroll
    for (int r = 0; r < R; ++r) accumulator[r] = 0.0F;
    for (int group = lane; group < kMoeIntermediate / 16; group += 32) {
        const auto* values = reinterpret_cast<const float4*>(activation + 16 * group);
        const float4 a0 = load_vec<float4>(values);
        const float4 a1 = load_vec<float4>(values + 1);
        const float4 a2 = load_vec<float4>(values + 2);
        const float4 a3 = load_vec<float4>(values + 3);
        const float av[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                              a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = rows[r];
            const float scale =
                scale_lut[scales[batched_nvfp4_scale_offset(row, group, kMoeIntermediate)]] *
                inverse;
            const std::uint64_t codes_word = load_vec<std::uint64_t>(
                codes + static_cast<std::size_t>(row) * (kMoeIntermediate / 2U) +
                static_cast<std::size_t>(group) * 8U);
#pragma unroll
            for (int byte = 0; byte < 8; ++byte) {
                const float2 code = decode_lut[static_cast<std::uint8_t>(codes_word >> (8 * byte))];
                accumulator[r] = fmaf(code.x * scale, av[2 * byte], accumulator[r]);
                accumulator[r] = fmaf(code.y * scale, av[2 * byte + 1], accumulator[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) value[r] = warp_reduce_sum(accumulator[r]);
}

// One CTA builds the group list from the routed ids. A slot's bucket holds its member count in
// the first int followed by at most kMoeGroupMaxMembers route indices (route + kTopK * token).
// Top-k selection is without replacement within a token, so an expert can occur at most once per
// column; the member capacity is therefore the verify width, not kTopK * width.
__global__ __launch_bounds__(kMoeGroupPrepareThreads) void moe_nvfp4_grouped_prepare_kernel(
    const std::int32_t* __restrict__ ids, std::int32_t entries,
    const std::int32_t* __restrict__ lookup, std::int32_t slot_count,
    std::int32_t* __restrict__ buckets, std::int32_t* __restrict__ meta) {
    const int thread = static_cast<int>(threadIdx.x);
    for (int slot = thread; slot < slot_count; slot += kMoeGroupPrepareThreads) {
        buckets[static_cast<std::size_t>(slot) * kMoeGroupBucketStride] = 0;
    }
    __syncthreads();
    for (int entry = thread; entry < entries; entry += kMoeGroupPrepareThreads) {
        const std::int32_t global = ids[entry];
        const std::int32_t slot =
            global >= 0 && global < kMoeExperts ? lookup[global] : -1;
        if (slot < 0 || slot >= slot_count) { continue; }
        std::int32_t* bucket = buckets + static_cast<std::size_t>(slot) * kMoeGroupBucketStride;
        const int member = atomicAdd(bucket, 1);
        if (member < kMoeGroupMaxMembers) { bucket[1 + member] = entry; }
    }
    __syncthreads();
    if (thread == 0) { meta[0] = 0; }
    __syncthreads();
    for (int slot = thread; slot < slot_count; slot += kMoeGroupPrepareThreads) {
        const int count = buckets[static_cast<std::size_t>(slot) * kMoeGroupBucketStride];
        if (count <= 0) { continue; }
        const int group = atomicAdd(meta, 1);
        if (group < kMoeGroupMaxGroups) {
            meta[1 + group]                      = slot;
            meta[1 + kMoeGroupMaxGroups + group] = count < kMoeGroupMaxMembers
                                                       ? count
                                                       : kMoeGroupMaxMembers;
        }
    }
}

// Grouped gate/up: CTA loops over (expert group, row tile) work items with grid stride. For one
// group the resident expert's gate/up planes stay in L1 across the sequential column loop; each
// column uses one gate/up accumulator set and the exact cached-path reduction order, so the
// activation words are raw-bit identical to the ungrouped kernel.
template <class A>
__global__ __launch_bounds__(kMoeDecodeThreads, 10) void moe_nvfp4_grouped_gate_up_kernel(
    const A* __restrict__ x, const std::uint8_t* __restrict__ slots,
    std::uint64_t slot_bytes, const float* __restrict__ divisors,
    float* __restrict__ activation, const std::int32_t* __restrict__ buckets,
    const std::int32_t* __restrict__ meta) {
    constexpr int kRows = kMoeDecodeGateUpTile / kMoeDecodeWarps;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    __shared__ float2 decode_lut[256];
    __shared__ float scale_lut[256];
    for (int index = static_cast<int>(threadIdx.x); index < 256;
         index += static_cast<int>(blockDim.x)) {
        decode_lut[index] = decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(index));
        scale_lut[index]  = decode_nvfp4_e4m3(static_cast<std::uint8_t>(index));
    }
    __syncthreads();

    const int groups = meta[0];
    const int row_base = static_cast<int>(blockIdx.x) * kMoeDecodeGateUpTile;
    for (int group = static_cast<int>(blockIdx.y); group < groups;
         group += static_cast<int>(gridDim.y)) {
        const int slot     = meta[1 + group];
        const std::int32_t* bucket =
            buckets + static_cast<std::size_t>(slot) * kMoeGroupBucketStride;
        const int count = bucket[0] < kMoeGroupMaxMembers ? bucket[0] : kMoeGroupMaxMembers;

        const std::uint8_t* base   = slots + static_cast<std::size_t>(slot) * slot_bytes;
        const float* slot_divisors = divisors + static_cast<std::size_t>(slot) * 3;
        const float inverse_gate   = 1.0F / slot_divisors[0];
        const float inverse_up     = 1.0F / slot_divisors[1];

        int rows[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) rows[r] = row_base + warp + r * kMoeDecodeWarps;
        for (int member = 0; member < count; ++member) {
            const int route_idx = bucket[1 + member];
            float gate[kRows];
            float up[kRows];
            grouped_gate_up_rows<A, kRows>(
                base, base + kMoeGateScale, inverse_gate, base + kMoeUpCode,
                base + kMoeUpScale, inverse_up, rows,
                x + static_cast<std::size_t>(route_idx / kMoeTopK) * kMoeHidden, decode_lut,
                scale_lut, gate, up);
            if (lane == 0) {
                float* column = activation + static_cast<std::size_t>(route_idx) * kMoeIntermediate;
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    column[rows[r]] = silu(gate[r]) * up[r];
                }
            }
        }
    }
}

// Grouped down: same row helper and reduction as the cached kernel, but the resident expert's
// down plane is reused across the sequential column loop. Each (route, row) contribution is
// written to a partial buffer; the final reduction reproduces the cached path's route-order
// FP32 accumulation exactly.
template <class A>
__global__ __launch_bounds__(kMoeDecodeThreads, 9) void moe_nvfp4_grouped_down_kernel(
    const std::uint8_t* __restrict__ slots, std::uint64_t slot_bytes,
    const float* __restrict__ divisors, const float* __restrict__ coefficients,
    const float* __restrict__ activation, float* __restrict__ partial,
    const std::int32_t* __restrict__ buckets, const std::int32_t* __restrict__ meta) {
    constexpr int kRows = kMoeDecodeDownTile / kMoeDecodeWarps;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    __shared__ float2 decode_lut[256];
    __shared__ float scale_lut[256];
    for (int index = static_cast<int>(threadIdx.x); index < 256;
         index += static_cast<int>(blockDim.x)) {
        decode_lut[index] = decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(index));
        scale_lut[index]  = decode_nvfp4_e4m3(static_cast<std::uint8_t>(index));
    }
    __syncthreads();

    const int groups = meta[0];
    const int row_base = static_cast<int>(blockIdx.x) * kMoeDecodeDownTile;
    for (int group = static_cast<int>(blockIdx.y); group < groups;
         group += static_cast<int>(gridDim.y)) {
        const int slot     = meta[1 + group];
        const std::int32_t* bucket =
            buckets + static_cast<std::size_t>(slot) * kMoeGroupBucketStride;
        const int count = bucket[0] < kMoeGroupMaxMembers ? bucket[0] : kMoeGroupMaxMembers;

        const std::uint8_t* base   = slots + static_cast<std::size_t>(slot) * slot_bytes;
        const float* slot_divisors = divisors + static_cast<std::size_t>(slot) * 3;
        const float inverse_down   = 1.0F / slot_divisors[2];

        int rows[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) rows[r] = row_base + warp + r * kMoeDecodeWarps;
        for (int member = 0; member < count; ++member) {
            const int route_idx = bucket[1 + member];
            const float* source =
                activation + static_cast<std::size_t>(route_idx) * kMoeIntermediate;
            float value[kRows];
            grouped_down_rows<kRows>(base + kMoeDownCode, base + kMoeDownScale, inverse_down,
                                     rows, source, decode_lut, scale_lut, value);
            if (lane == 0) {
                float* column = partial + static_cast<std::size_t>(route_idx) * kMoeHidden;
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    column[rows[r]] = coefficients[route_idx] * value[r];
                }
            }
        }
    }
}

template <class A>
__global__ __launch_bounds__(256) void moe_nvfp4_grouped_reduce_kernel(
    const float* __restrict__ partial, A* __restrict__ destination) {
    const int row   = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (row >= kMoeHidden) { return; }
    float sum = 0.0F;
    const float* column =
        partial + static_cast<std::size_t>(token * kMoeTopK) * kMoeHidden + row;
#pragma unroll
    for (int route = 0; route < kMoeTopK; ++route) {
        sum += column[static_cast<std::size_t>(route) * kMoeHidden];
    }
    A* destination_column = destination + static_cast<std::size_t>(token) * kMoeHidden;
    destination_column[row] = MoeAct<A>::from_float(
        sum + MoeAct<A>::to_float(destination_column[row]));
}

} // namespace

void moe_nvfp4_grouped_prepare_launch(const std::int32_t* ids, std::int32_t entries,
                                       const std::int32_t* lookup, std::int32_t slot_count,
                                       MoeNvfp4GroupedScratch grouped, cudaStream_t stream) {
    if (entries < 1 || entries > kMoeGroupMaxGroups || lookup == nullptr ||
        grouped.buckets == nullptr || grouped.meta == nullptr) {
        throw std::invalid_argument("moe_nvfp4 grouped: invalid group prepare arguments");
    }
    moe_nvfp4_grouped_prepare_kernel<<<1, kMoeGroupPrepareThreads, 0, stream>>>(
        ids, entries, lookup, slot_count, grouped.buckets, grouped.meta);
    CUDA_CHECK(cudaGetLastError());
}

void moe_nvfp4_grouped_cached_launch(const Tensor& x, const std::uint8_t* slots,
                                     std::uint64_t slot_bytes, std::int32_t slot_count,
                                     const std::int32_t* lookup, const float* divisors,
                                     MoeNvfp4Scratch scratch, MoeNvfp4GroupedScratch grouped,
                                     const MoeNvfp4RouteOutputs& route, Tensor& destination,
                                     cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens < 2 || tokens > kMoeMaxSmallT || scratch.activation == nullptr ||
        grouped.buckets == nullptr || grouped.meta == nullptr || grouped.partial == nullptr ||
        route.ids == nullptr) {
        throw std::invalid_argument("moe_nvfp4 grouped: the grouped cached path needs 2-8 tokens");
    }
    const std::int32_t entries = tokens * kMoeTopK;
    moe_nvfp4_grouped_prepare_launch(route.ids, entries, lookup, slot_count, grouped, stream);
    CUDA_CHECK(cudaMemsetAsync(grouped.partial, 0,
                               static_cast<std::size_t>(entries) * kMoeHidden * sizeof(float),
                               stream));

    constexpr std::uint32_t kGroupGridY = 8;
    const dim3 gate_up_grid(kMoeIntermediate / kMoeDecodeGateUpTile, kGroupGridY);
    const dim3 down_grid(kMoeHidden / kMoeDecodeDownTile, kGroupGridY);
    if (x.dtype == DType::FP16) {
        moe_nvfp4_grouped_gate_up_kernel<__half><<<gate_up_grid, kMoeDecodeThreads, 0, stream>>>(
            static_cast<const __half*>(x.data), slots, slot_bytes, divisors, scratch.activation,
            grouped.buckets, grouped.meta);
        moe_nvfp4_grouped_down_kernel<__half><<<down_grid, kMoeDecodeThreads, 0, stream>>>(
            slots, slot_bytes, divisors, route.coefficients, scratch.activation, grouped.partial,
            grouped.buckets, grouped.meta);
        const dim3 reduce_grid(kMoeHidden / 256, static_cast<std::uint32_t>(tokens));
        moe_nvfp4_grouped_reduce_kernel<__half><<<reduce_grid, 256, 0, stream>>>(
            grouped.partial, static_cast<__half*>(destination.data));
    } else {
        moe_nvfp4_grouped_gate_up_kernel<__nv_bfloat16>
            <<<gate_up_grid, kMoeDecodeThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data), slots, slot_bytes, divisors,
                scratch.activation, grouped.buckets, grouped.meta);
        moe_nvfp4_grouped_down_kernel<__nv_bfloat16><<<down_grid, kMoeDecodeThreads, 0, stream>>>(
            slots, slot_bytes, divisors, route.coefficients, scratch.activation, grouped.partial,
            grouped.buckets, grouped.meta);
        const dim3 reduce_grid(kMoeHidden / 256, static_cast<std::uint32_t>(tokens));
        moe_nvfp4_grouped_reduce_kernel<__nv_bfloat16><<<reduce_grid, 256, 0, stream>>>(
            grouped.partial, static_cast<__nv_bfloat16*>(destination.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
