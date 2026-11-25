#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

// Flash-Next GDN qkvz W8G32_F16S multi-column GEMV, one launch for the whole per-rank parent.
//   x is [2560, T] with T in [1,4] (decode/verify, BF16 or FP16). The parent is
//   [8192, 2560] RowSplit W8 with FP16 scales, stored row-major per plane:
//   code row 2560 B, scale row 160 B. Row blocks map to the four projections
//   by output row counts: [0, q_rows) -> query, [q_rows, q_rows+k_rows) -> key,
//   then value, then z. Each output is an independent [rows, T] tensor of the x
//   dtype. One warp owns one row and one CTA owns the T token columns: the code,
//   scale and BF16 control rows are loaded once and reused across columns. Each
//   column keeps the same warp/row mapping and FP32 accumulation order as T=1.
//   K=2560, group=32. fp32 FMA into per-column per-lane accumulators, warp-shuffle
//   reduction, round-to-nearest store. No workspace: reductions live in registers.

constexpr int kK              = 2560;
constexpr int kGroup          = 32;
constexpr int kGroupsPerRow   = kK / kGroup;   // 80
constexpr int kScaleBytesRow  = kGroupsPerRow * 2;
constexpr int kValuesPerLane  = 8;
constexpr int kValuesPerPhase = 32 * kValuesPerLane;  // 256
constexpr int kGroupsPerPhase = kValuesPerPhase / kGroup;   // 8
constexpr int kPhases         = kK / kValuesPerPhase;       // 10
constexpr int kRowsPerCta     = 4;
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr int kGdnHeads       = 48;
constexpr int kControlRows    = 2 * kGdnHeads;   // A rows [0,48), B rows [48,96)

template <class Act>
struct DecodeAct;

template <>
struct DecodeAct<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    static __device__ __forceinline__ float2 to_float2(std::uint32_t bits) {
        return bf16x2_bits_to_float2(bits);
    }
    static __device__ __forceinline__ __nv_bfloat16 round(float value) {
        return __float2bfloat16_rn(value);
    }
};

template <>
struct DecodeAct<__half> {
    static __device__ __forceinline__ float to_float(__half value) { return __half2float(value); }
    static __device__ __forceinline__ float2 to_float2(std::uint32_t bits) {
        return __half22float2(*reinterpret_cast<const __half2*>(&bits));
    }
    static __device__ __forceinline__ __half round(float value) { return __float2half_rn(value); }
};

struct Segments {
    std::int32_t k_end;      // q_rows + k_rows
    std::int32_t v_end;      // q_rows + k_rows + v_rows
    std::int32_t q_rows;
    std::int32_t k_rows;
    std::int32_t v_rows;
    std::int32_t z_rows;
};

template <class Act>
struct GdnQkvzOutputs {
    Act* query;
    Act* key;
    Act* value;
    Act* z;
    Segments segments;
    const __nv_bfloat16* control;
    const float* a_log;
    const float* dt_bias;
    float* g;
    float* beta;
    std::int32_t qkvz_rows;
};

template <class Act, int Columns>
__device__ __forceinline__ void gdn_gating_rows(const Act* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ control,
                                                const float* __restrict__ a_log,
                                                const float* __restrict__ dt_bias,
                                                float* __restrict__ g, float* __restrict__ beta,
                                                int control_row, int lane) {
    const int head   = control_row < kGdnHeads ? control_row : control_row - kGdnHeads;
    const bool is_b  = control_row >= kGdnHeads;
    const auto* wrow = control + static_cast<std::int64_t>(control_row) * kK;

    // The BF16 control row is read once per CTA and reused across all token columns.
    // Each column keeps the same lane-strided fmaf order, shfl_down reduction and
    // epilogue as the single-token route, so every g/beta bit is unchanged.
    float accumulator[Columns];
#pragma unroll
    for (int column = 0; column < Columns; ++column) { accumulator[column] = 0.0f; }
    for (int k = lane; k < kK; k += 32) {
        const float weight = __bfloat162float(wrow[k]);
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            const float value = DecodeAct<Act>::to_float(x[static_cast<std::int64_t>(column) * kK + k]);
            accumulator[column] = fmaf(weight, value, accumulator[column]);
        }
    }
#pragma unroll
    for (int column = 0; column < Columns; ++column) {
        const float reduced = warp_reduce_sum(accumulator[column]);
        if (lane == 0) {
            const int slot = column * kGdnHeads + head;
            if (is_b) {
                beta[slot] = sigmoid(reduced);
            } else {
                g[slot] = -expf(a_log[head]) * softplus(reduced + dt_bias[head]);
            }
        }
    }
}

// True batched counterpart: the W8 code/scale planes and the BF16 control row are
// fetched once per output row, then reused for every token column. Each column has
// its own FP32 accumulator with the identical byte decode, group scale, fmaf order,
// warp reduction and rounding as the single-column path, so the results are bit-exact.
template <class Act, int Columns>
__global__ __launch_bounds__(kRowsPerCta * 32, 4) void w8_gdn_qkvz_decode_multi_kernel(
    const Act* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, GdnQkvzOutputs<Act> output) {
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int cta_row0 = static_cast<int>(blockIdx.x) * kRowsPerCta;
    const int row      = cta_row0 + warp;

    if (row >= output.qkvz_rows) {
        gdn_gating_rows<Act, Columns>(x, output.control, output.a_log, output.dt_bias, output.g,
                                      output.beta, row - output.qkvz_rows, lane);
        return;
    }

    const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * kK;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kScaleBytesRow;

    float accumulator[Columns];
#pragma unroll
    for (int column = 0; column < Columns; ++column) { accumulator[column] = 0.0f; }
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned scale_bits = 0;
        if (lane < kGroupsPerPhase) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
        }
        scale_bits        = __shfl_sync(kFullWarpMask, scale_bits, lane >> 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const int phase_k  = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 packed = load_vec<uint2>(code_row + phase_k);
        float weights[kValuesPerLane];
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

#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            const Act* token_x = x + static_cast<std::int64_t>(column) * kK;
            const uint4 values = load_vec<uint4>(token_x + phase_k);
            const float2 x0    = DecodeAct<Act>::to_float2(values.x);
            const float2 x1    = DecodeAct<Act>::to_float2(values.y);
            const float2 x2    = DecodeAct<Act>::to_float2(values.z);
            const float2 x3    = DecodeAct<Act>::to_float2(values.w);
            accumulator[column] = fmaf(weights[0], x0.x, accumulator[column]);
            accumulator[column] = fmaf(weights[1], x0.y, accumulator[column]);
            accumulator[column] = fmaf(weights[2], x1.x, accumulator[column]);
            accumulator[column] = fmaf(weights[3], x1.y, accumulator[column]);
            accumulator[column] = fmaf(weights[4], x2.x, accumulator[column]);
            accumulator[column] = fmaf(weights[5], x2.y, accumulator[column]);
            accumulator[column] = fmaf(weights[6], x3.x, accumulator[column]);
            accumulator[column] = fmaf(weights[7], x3.y, accumulator[column]);
        }
    }

    const Segments& s = output.segments;
#pragma unroll
    for (int column = 0; column < Columns; ++column) {
        const float reduced = warp_reduce_sum(accumulator[column]);
        if (lane != 0) { continue; }
        const Act projected = DecodeAct<Act>::round(reduced);
        if (row < s.q_rows) {
            output.query[static_cast<std::int64_t>(column) * s.q_rows + row] = projected;
        } else if (row < s.k_end) {
            output.key[static_cast<std::int64_t>(column) * s.k_rows + (row - s.q_rows)] = projected;
        } else if (row < s.v_end) {
            output.value[static_cast<std::int64_t>(column) * s.v_rows + (row - s.k_end)] = projected;
        } else {
            output.z[static_cast<std::int64_t>(column) * s.z_rows + (row - s.v_end)] = projected;
        }
    }
}

void validate_launch_inputs(const Tensor& x, const Weight& weight, const Tensor& query,
                             const Tensor& key, const Tensor& value, const Tensor& z,
                             const Weight& control, const Tensor& a_log, const Tensor& dt_bias,
                             const Tensor& g, const Tensor& beta) {
    const std::int32_t q_rows = query.ne[0];
    const std::int32_t k_rows = key.ne[0];
    const std::int32_t v_rows = value.ne[0];
    const std::int32_t z_rows = z.ne[0];
    const std::int64_t total  = static_cast<std::int64_t>(q_rows) + k_rows + v_rows + z_rows;
    const std::int32_t tokens = x.ne[1];
    const std::int32_t group = weight.group_size != 0 ? static_cast<std::int32_t>(weight.group_size)
                                                      : weight.group;
    if (x.dtype != DType::BF16 && x.dtype != DType::FP16) {
        throw std::invalid_argument("w8_gdn_qkvz_decode: x must be BF16 or FP16 [2560, T]");
    }
    if (!x.is_contiguous() || x.ne[0] != kK || tokens < 1 || tokens > kMaxQkvzDecodeTokens ||
        x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument(
            "w8_gdn_qkvz_decode: x must be contiguous [2560, T] with T in [1,4]");
    }
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        group != kGroup || weight.k != kK || weight.n != total || weight.qdata == nullptr ||
        weight.scales == nullptr) {
        throw std::invalid_argument(
            "w8_gdn_qkvz_decode: parent must be RowSplit W8G32_F16S [rows, 2560] with scales");
    }
    for (const Tensor* output : {&query, &key, &value, &z}) {
        if (output->dtype != x.dtype || !output->is_contiguous() || output->ne[1] != tokens ||
            output->ne[2] != 1 || output->ne[3] != 1 || output->data == nullptr) {
            throw std::invalid_argument(
                "w8_gdn_qkvz_decode: outputs must be contiguous x-dtype [rows, T]");
        }
    }
    if (control.qtype != QType::BF16_CTRL || control.layout != QuantLayout::Contiguous ||
        control.n != kControlRows || control.k != kK || control.shape[0] != kControlRows ||
        control.shape[1] != kK || control.qdata == nullptr) {
        throw std::invalid_argument(
            "w8_gdn_qkvz_decode: control must be contiguous BF16_CTRL [96, 2560]");
    }
    for (const Tensor* vector : {&a_log, &dt_bias}) {
        if (vector->dtype != DType::FP32 || vector->ne[0] != kGdnHeads || vector->ne[1] != 1 ||
            vector->ne[2] != 1 || vector->ne[3] != 1 || !vector->is_contiguous() ||
            vector->data == nullptr) {
            throw std::invalid_argument(
                "w8_gdn_qkvz_decode: a_log and dt_bias must be contiguous FP32 [48]");
        }
    }
    for (const Tensor* output : {&g, &beta}) {
        if (output->dtype != DType::FP32 || output->ne[0] != kGdnHeads ||
            output->ne[1] != tokens || output->ne[2] != 1 || output->ne[3] != 1 ||
            !output->is_contiguous() || output->data == nullptr) {
            throw std::invalid_argument(
                "w8_gdn_qkvz_decode: g and beta must be contiguous FP32 [48, T]");
        }
    }
}

template <class Act>
void launch_typed(const Tensor& x, const Weight& weight, Tensor& query, Tensor& key, Tensor& value,
                   Tensor& z, const Weight& control, const Tensor& a_log, const Tensor& dt_bias,
                   Tensor& g, Tensor& beta, cudaStream_t stream) {
    Segments segments{};
    segments.q_rows = query.ne[0];
    segments.k_rows = key.ne[0];
    segments.v_rows = value.ne[0];
    segments.z_rows = z.ne[0];
    segments.k_end  = segments.q_rows + segments.k_rows;
    segments.v_end  = segments.k_end + segments.v_rows;
    const GdnQkvzOutputs<Act> output{
        static_cast<Act*>(query.data), static_cast<Act*>(key.data), static_cast<Act*>(value.data),
        static_cast<Act*>(z.data), segments,
        static_cast<const __nv_bfloat16*>(control.qdata), static_cast<const float*>(a_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), weight.n};
    const int grid   = (weight.n + kControlRows) / kRowsPerCta;
    const int tokens = x.ne[1];
    const auto* activation = static_cast<const Act*>(x.data);
    const auto* codes      = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales     = static_cast<const std::uint8_t*>(weight.scales);
    switch (tokens) {
    case 1:
        w8_gdn_qkvz_decode_multi_kernel<Act, 1>
            <<<dim3(grid, 1), kRowsPerCta * 32, 0, stream>>>(activation, codes, scales, output);
        break;
    case 2:
        w8_gdn_qkvz_decode_multi_kernel<Act, 2>
            <<<dim3(grid, 1), kRowsPerCta * 32, 0, stream>>>(activation, codes, scales, output);
        break;
    case 3:
        w8_gdn_qkvz_decode_multi_kernel<Act, 3>
            <<<dim3(grid, 1), kRowsPerCta * 32, 0, stream>>>(activation, codes, scales, output);
        break;
    case 4:
        w8_gdn_qkvz_decode_multi_kernel<Act, 4>
            <<<dim3(grid, 1), kRowsPerCta * 32, 0, stream>>>(activation, codes, scales, output);
        break;
    default: throw std::invalid_argument("w8_gdn_qkvz_decode: unsupported token count");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_gdn_qkvz_decode_launch(const Tensor& x, const Weight& weight, Tensor& query, Tensor& key,
                                Tensor& value, Tensor& z, const Weight& control,
                                const Tensor& a_log, const Tensor& dt_bias, Tensor& g,
                                Tensor& beta, cudaStream_t stream) {
    validate_launch_inputs(x, weight, query, key, value, z, control, a_log, dt_bias, g, beta);
    if (weight.n % kRowsPerCta != 0) {
        throw std::invalid_argument("w8_gdn_qkvz_decode: parent rows must divide the CTA tile");
    }
    if (x.dtype == DType::BF16) {
        launch_typed<__nv_bfloat16>(x, weight, query, key, value, z, control, a_log, dt_bias, g,
                                     beta, stream);
        return;
    }
    launch_typed<__half>(x, weight, query, key, value, z, control, a_log, dt_bias, g, beta,
                          stream);
}

} // namespace ninfer::ops::detail
