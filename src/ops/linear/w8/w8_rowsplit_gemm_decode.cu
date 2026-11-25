#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows          = 2048;
constexpr int kHidden        = 16384;

constexpr int kNarrowHidden        = 2560;
constexpr int kNarrowGroups        = kNarrowHidden / 32;
constexpr int kNarrowValuesPerLane = 8;
constexpr int kNarrowPhaseValues   = 32 * kNarrowValuesPerLane;
constexpr int kNarrowPhaseGroups   = kNarrowPhaseValues / 32;
constexpr int kNarrowPhases        = kNarrowHidden / kNarrowPhaseValues;

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 4) void w8_rowsplit_narrow_f16_decode_kernel(
    const __half* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __half* __restrict__ out, std::int32_t rows) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    if (row >= rows) { return; }
    const auto* code_row  = codes + static_cast<std::int64_t>(row) * kNarrowHidden;
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * kNarrowGroups * 2;

    float acc = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kNarrowPhases; ++phase) {
        unsigned scale_bits = 0;
        if (lane < kNarrowPhaseGroups) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + static_cast<std::int64_t>(phase * kNarrowPhaseGroups + lane) * 2);
        }
        scale_bits        = __shfl_sync(0xffffffffu, scale_bits, lane >> 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const int phase_k  = phase * kNarrowPhaseValues + lane * kNarrowValuesPerLane;
        const uint2 packed = load_vec<uint2>(code_row + phase_k);
        const uint4 values = load_vec<uint4>(x + phase_k);
        const float2 xv[4] = {
            __half22float2(*reinterpret_cast<const __half2*>(&values.x)),
            __half22float2(*reinterpret_cast<const __half2*>(&values.y)),
            __half22float2(*reinterpret_cast<const __half2*>(&values.z)),
            __half22float2(*reinterpret_cast<const __half2*>(&values.w)),
        };
#pragma unroll
        for (int byte = 0; byte < kNarrowValuesPerLane; ++byte) {
            const std::uint32_t word = (&packed.x)[byte >> 2];
            const float2 pair       = xv[byte >> 1];
            const float activation  = (byte & 1) == 0 ? pair.x : pair.y;
            const float weight = static_cast<float>(static_cast<std::int8_t>(word >> ((byte & 3) * 8))) *
                                 scale;
            acc = fmaf(weight, activation, acc);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2half_rn(acc); }
}

// True batched counterpart of w8_rowsplit_narrow_f16_decode_kernel. The W8 code/scale planes (the
// weight) are fetched once per row and phase, then applied to every token column. Each column keeps
// its own accumulator with the identical byte-to-weight decode, group scale, fmaf order, warp
// reduction and final rounding as the T==1 kernel.
template <int RowsPerCta, int Columns>
__global__ __launch_bounds__(RowsPerCta * 32, 4) void w8_rowsplit_narrow_f16_decode_multi_kernel(
    const __half* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __half* __restrict__ out, std::int32_t rows) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    if (row >= rows) { return; }
    const auto* code_row  = codes + static_cast<std::int64_t>(row) * kNarrowHidden;
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * kNarrowGroups * 2;

    float acc[Columns];
#pragma unroll
    for (int column = 0; column < Columns; ++column) { acc[column] = 0.0f; }
#pragma unroll
    for (int phase = 0; phase < kNarrowPhases; ++phase) {
        unsigned scale_bits = 0;
        if (lane < kNarrowPhaseGroups) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + static_cast<std::int64_t>(phase * kNarrowPhaseGroups + lane) * 2);
        }
        scale_bits        = __shfl_sync(0xffffffffu, scale_bits, lane >> 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const int phase_k  = phase * kNarrowPhaseValues + lane * kNarrowValuesPerLane;
        const uint2 packed = load_vec<uint2>(code_row + phase_k);
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            const uint4 values = load_vec<uint4>(
                x + static_cast<std::int64_t>(column) * kNarrowHidden + phase_k);
            const float2 xv[4] = {
                __half22float2(*reinterpret_cast<const __half2*>(&values.x)),
                __half22float2(*reinterpret_cast<const __half2*>(&values.y)),
                __half22float2(*reinterpret_cast<const __half2*>(&values.z)),
                __half22float2(*reinterpret_cast<const __half2*>(&values.w)),
            };
#pragma unroll
            for (int byte = 0; byte < kNarrowValuesPerLane; ++byte) {
                const std::uint32_t word = (&packed.x)[byte >> 2];
                const float2 pair       = xv[byte >> 1];
                const float activation  = (byte & 1) == 0 ? pair.x : pair.y;
                const float weight = static_cast<float>(
                                         static_cast<std::int8_t>(word >> ((byte & 3) * 8))) *
                                     scale;
                acc[column] = fmaf(weight, activation, acc[column]);
            }
        }
    }
#pragma unroll
    for (int column = 0; column < Columns; ++column) {
        acc[column] = warp_reduce_sum(acc[column]);
        if (lane == 0) {
            out[static_cast<std::int64_t>(column) * rows + row] = __float2half_rn(acc[column]);
        }
    }
}

constexpr int kWide3072Hidden        = 3072;
constexpr int kWide3072Groups        = kWide3072Hidden / 32;
constexpr int kWide3072ScaleBytesRow = kWide3072Groups * 2;
constexpr int kWide3072ValuesPerLane = 8;
constexpr int kWide3072PhaseValues   = 32 * kWide3072ValuesPerLane;
constexpr int kWide3072PhaseGroups   = kWide3072PhaseValues / 32;
constexpr int kWide3072Phases        = kWide3072Hidden / kWide3072PhaseValues;

template <class Act>
struct W8K3072Act;

template <>
struct W8K3072Act<__nv_bfloat16> {
    static __device__ __forceinline__ float2 to_float2(std::uint32_t bits) {
        return bf16x2_bits_to_float2(bits);
    }
    static __device__ __forceinline__ __nv_bfloat16 round(float value) {
        return __float2bfloat16_rn(value);
    }
};

template <>
struct W8K3072Act<__half> {
    static __device__ __forceinline__ float2 to_float2(std::uint32_t bits) {
        return __half22float2(*reinterpret_cast<const __half2*>(&bits));
    }
    static __device__ __forceinline__ __half round(float value) {
        return __float2half_rn(value);
    }
};

template <int RowsPerCta, class Act>
__global__ __launch_bounds__(RowsPerCta * 32, 4) void w8_rowsplit_k3072_decode_kernel(
    const Act* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Act* __restrict__ out, std::int32_t rows) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    if (row >= rows) { return; }
    const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * kWide3072Hidden;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kWide3072ScaleBytesRow;

    float accumulator = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kWide3072Phases; ++phase) {
        unsigned scale_bits = 0;
        if (lane < kWide3072PhaseGroups) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + static_cast<std::int64_t>(phase * kWide3072PhaseGroups + lane) * 2);
        }
        scale_bits        = __shfl_sync(0xffffffffu, scale_bits, lane >> 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const int phase_k  = phase * kWide3072PhaseValues + lane * kWide3072ValuesPerLane;
        const uint2 packed = load_vec<uint2>(code_row + phase_k);
        float weights[kWide3072ValuesPerLane];
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t word = (&packed.x)[word_index];
            weights[word_index * 4 + 0] =
                static_cast<float>(static_cast<std::int8_t>(word & 0xffu)) * scale;
            weights[word_index * 4 + 1] =
                static_cast<float>(static_cast<std::int8_t>((word >> 8) & 0xffu)) * scale;
            weights[word_index * 4 + 2] =
                static_cast<float>(static_cast<std::int8_t>((word >> 16) & 0xffu)) * scale;
            weights[word_index * 4 + 3] =
                static_cast<float>(static_cast<std::int8_t>((word >> 24) & 0xffu)) * scale;
        }

        const uint4 values = load_vec<uint4>(x + phase_k);
        const float2 x0    = W8K3072Act<Act>::to_float2(values.x);
        const float2 x1    = W8K3072Act<Act>::to_float2(values.y);
        const float2 x2    = W8K3072Act<Act>::to_float2(values.z);
        const float2 x3    = W8K3072Act<Act>::to_float2(values.w);
        accumulator        = fmaf(weights[0], x0.x, accumulator);
        accumulator        = fmaf(weights[1], x0.y, accumulator);
        accumulator        = fmaf(weights[2], x1.x, accumulator);
        accumulator        = fmaf(weights[3], x1.y, accumulator);
        accumulator        = fmaf(weights[4], x2.x, accumulator);
        accumulator        = fmaf(weights[5], x2.y, accumulator);
        accumulator        = fmaf(weights[6], x3.x, accumulator);
        accumulator        = fmaf(weights[7], x3.y, accumulator);
    }

    accumulator = warp_reduce_sum(accumulator);
    if (lane == 0) { out[row] = W8K3072Act<Act>::round(accumulator); }
}

constexpr int kGroups        = kHidden / 32;
constexpr int kValuesPerLane = 8;

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w8_rowsplit_k16384_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase = kValuesPerPhase / 32;
    constexpr int kPhases         = kHidden / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int row        = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const auto* code_row = codes + static_cast<std::int64_t>(row) * kHidden;
    const auto* scale_row =
        scales + static_cast<std::int64_t>(row) * kGroups * sizeof(std::uint16_t);

    float acc = 0.0f;
#pragma unroll 8
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned scale_bits = 0;
        if (lane < kGroupsPerPhase) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
        }
        scale_bits        = __shfl_sync(kMask, scale_bits, lane >> 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const int phase_k  = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 packed = load_vec<uint2>(code_row + phase_k);
        const uint4 values = load_vec<uint4>(x + phase_k);
        const float2 xv[4] = {
            bf16x2_bits_to_float2(values.x),
            bf16x2_bits_to_float2(values.y),
            bf16x2_bits_to_float2(values.z),
            bf16x2_bits_to_float2(values.w),
        };
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t word = (&packed.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const float2 activation_pair = xv[word_index * 2 + (byte >> 1)];
                const float activation = (byte & 1) == 0 ? activation_pair.x : activation_pair.y;
                const float weight_value =
                    static_cast<float>(static_cast<std::int8_t>(word >> (byte * 8))) * scale;
                acc = fmaf(weight_value, activation, acc);
            }
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16_rn(acc); }
}

template <int RowsPerCta>
void launch_decode(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.ne[0] != kHidden || x.ne[1] != 1 || out.ne[0] != kRows || out.ne[1] != 1 ||
        w.n != kRows || w.k != kHidden || w.padded_shape[1] != kHidden) {
        throw std::invalid_argument("W8 decode requires [2048,16384] and T=1");
    }
    static_assert((kRows % RowsPerCta) == 0);
    w8_rowsplit_k16384_decode_kernel<RowsPerCta>
        <<<kRows / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_w8_decode_r4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_decode<4>(x, w, out, stream);
}

void w8_rowsplit_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_decode<16>(x, w, out, stream);
}

void launch_w8_k3072_decode(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kRowsPerCta = 4;
    const std::int32_t group = w.group_size != 0 ? static_cast<std::int32_t>(w.group_size)
                                                 : w.group;
    if (x.ne[0] != kWide3072Hidden || x.ne[1] != 1 || x.ne[2] != 1 || x.ne[3] != 1 ||
        w.n != out.ne[0] || w.k != kWide3072Hidden || w.padded_shape[1] != kWide3072Hidden ||
        w.qtype != QType::W8G32_F16S || w.layout != QuantLayout::RowSplit || group != 32 ||
        w.qdata == nullptr || w.scales == nullptr || out.ne[1] != 1 || out.ne[2] != 1 ||
        out.ne[3] != 1 || !x.is_contiguous() || !out.is_contiguous() || x.dtype != out.dtype ||
        (x.dtype != DType::BF16 && x.dtype != DType::FP16)) {
        throw std::invalid_argument(
            "W8 k3072 decode requires contiguous [n,3072] W8G32_F16S RowSplit and T=1");
    }
    const unsigned grid = static_cast<unsigned>(div_up(w.n, kRowsPerCta));
    if (x.dtype == DType::BF16) {
        w8_rowsplit_k3072_decode_kernel<kRowsPerCta><<<grid, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            w.n);
    } else {
        w8_rowsplit_k3072_decode_kernel<kRowsPerCta><<<grid, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __half*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__half*>(out.data), w.n);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_w8_narrow_f16_decode(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    constexpr int kRowsPerCta = 4;
    if (x.ne[0] != kNarrowHidden || x.ne[1] != 1 || w.n != out.ne[0] ||
        w.padded_shape[1] != kNarrowHidden || x.dtype != DType::FP16 || out.dtype != DType::FP16) {
        throw std::invalid_argument("W8 narrow f16 decode requires [n,2560] and T=1");
    }
    w8_rowsplit_narrow_f16_decode_kernel<kRowsPerCta>
        <<<static_cast<unsigned>(div_up(w.n, kRowsPerCta)), kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __half*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__half*>(out.data), w.n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_w8_narrow_f16_decode_multi(const Tensor& x, const Weight& w, Tensor& out,
                                       std::int32_t columns, cudaStream_t stream) {
    constexpr int kRowsPerCta = 4;
    if (x.ne[0] != kNarrowHidden || w.n != out.ne[0] ||
        w.padded_shape[1] != kNarrowHidden || x.dtype != DType::FP16 || out.dtype != DType::FP16) {
        throw std::invalid_argument("W8 narrow f16 multi decode requires [n,2560]");
    }
    const unsigned grid = static_cast<unsigned>(div_up(w.n, kRowsPerCta));
    switch (columns) {
    case 2:
        w8_rowsplit_narrow_f16_decode_multi_kernel<kRowsPerCta, 2>
            <<<grid, kRowsPerCta * 32, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), static_cast<__half*>(out.data), w.n);
        break;
    case 3:
        w8_rowsplit_narrow_f16_decode_multi_kernel<kRowsPerCta, 3>
            <<<grid, kRowsPerCta * 32, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), static_cast<__half*>(out.data), w.n);
        break;
    case 4:
        w8_rowsplit_narrow_f16_decode_multi_kernel<kRowsPerCta, 4>
            <<<grid, kRowsPerCta * 32, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), static_cast<__half*>(out.data), w.n);
        break;
    case 5:
        w8_rowsplit_narrow_f16_decode_multi_kernel<kRowsPerCta, 5>
            <<<grid, kRowsPerCta * 32, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), static_cast<__half*>(out.data), w.n);
        break;
    case 6:
        w8_rowsplit_narrow_f16_decode_multi_kernel<kRowsPerCta, 6>
            <<<grid, kRowsPerCta * 32, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), static_cast<__half*>(out.data), w.n);
        break;
    case 7:
        w8_rowsplit_narrow_f16_decode_multi_kernel<kRowsPerCta, 7>
            <<<grid, kRowsPerCta * 32, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), static_cast<__half*>(out.data), w.n);
        break;
    case 8:
        w8_rowsplit_narrow_f16_decode_multi_kernel<kRowsPerCta, 8>
            <<<grid, kRowsPerCta * 32, 0, stream>>>(
                static_cast<const __half*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), static_cast<__half*>(out.data), w.n);
        break;
    default:
        throw std::invalid_argument("W8 narrow f16 multi decode: unsupported column count");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
