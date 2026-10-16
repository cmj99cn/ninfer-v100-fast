// ninfer::ops::detail - Volta (sm_70) flash-attention prefill route.
//
// Drives the vendored llama.cpp MMA flash-attention kernel
// (third_party/llama_cpp_fattn, pinned at 62bf73d25) from a ninfer-native
// launcher. Rationale, phase gates and measurements: the V100 implementation and
// the V100 implementation.
//
// This is the only translation unit that sees the vendored headers, which is
// why the vendored include directory is scoped to this file in CMake: the
// vendored tree contains `common.cuh` and `mma.cuh`, names that also exist
// under src/ops.
//
// The vendored kernel reads K/V as plain strided FP16 and Q as strided FP32,
// while ninfer stores BF16 Q/out and a paged BF16 KV cache. Rather than inject
// page lookups into the kernel's hot loop, this file stages: append the whole
// width into the paged cache, gather the visible key range into a contiguous
// FP16 buffer once per layer, convert Q per Q-block, run the kernel, convert the
// FP32 result back to BF16.

#include "fattn-mma-f16.cuh" // vendored; must precede ninfer headers (defines WARP_SIZE)

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/gqa_attention_kv_iq4nl.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/launcher/gqa_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// Experimental sm_70 D256 Split-D route (gqa_attention_volta_splitd.cu). Selected with
// NINFER_VOLTA_SPLITD=0 to disable; the block call returns false when the kernel is
// unusable for this call, in which case the caller stays on the vendored llama.cpp kernel.
bool volta_splitd_requested();

bool splitd_positions_are_contiguous(const void* positions, std::int32_t base, std::int32_t width,
                                     cudaStream_t stream);

template <typename Geometry>
bool volta_splitd_block(const void* q_bf16, const void* k_f16, const void* v_f16, int rows,
                        int kv_len, int kv_offset, bool positions_contiguous, float scale,
                        void* out_bf16, cudaStream_t stream);

namespace {

// Spelled out rather than taken from gqa_attention_decode.cuh's kGqaHeadDim:
// including that header would pull in ops/common/mma.cuh and ops/common/warp.cuh,
// whose WARP_SIZE collides with the vendored common.cuh. Guarded by the
// static_asserts in VoltaFlashTiling below.
constexpr int kHeadDim = 256;
constexpr int kDKQ     = 256;
constexpr int kDV      = 256;

// The kernel picks ncols2 from the GQA ratio, and this table mirrors that choice
// per registered geometry:
//   24q/4kv -> ratio 6: 6 % 8 != 0, 6 % 4 != 0, then DKQ <= 256 and 6 % 2 == 0
//                       -> ncols2 = 2
//   16q/2kv -> ratio 8: 8 % 8 == 0                              -> ncols2 = 8
// ncols1 is then chosen so ncols1*ncols2 == 32, the smallest tile Volta compiles.
// A causal mask tensor is mandatory in both cases: mask == nullptr is only legal
// at ncols2 == 1.
template <typename Geometry>
struct VoltaFlashTiling;

template <>
struct VoltaFlashTiling<Gqa27Geometry> {
    static constexpr int ncols2 = 2;
    static constexpr int ncols1 = 16;
};

template <>
struct VoltaFlashTiling<Gqa27Tp2Geometry> : VoltaFlashTiling<Gqa27Geometry> {};

template <>
struct VoltaFlashTiling<Gqa27Tp4Geometry> : VoltaFlashTiling<Gqa27Geometry> {};

template <>
struct VoltaFlashTiling<Gqa35Geometry> {
    static constexpr int ncols2 = 8;
    static constexpr int ncols1 = 4;
};

template <typename Geometry>
struct VoltaFlashParams {
    static constexpr int kQHeads  = Geometry::QHeads;
    static constexpr int kKVHeads = Geometry::KVHeads;
    static constexpr int kGroup   = Geometry::GroupSize;
    static constexpr int kNcols2  = VoltaFlashTiling<Geometry>::ncols2;
    static constexpr int kNcols1  = VoltaFlashTiling<Geometry>::ncols1;
    static constexpr int kNcols   = kNcols1 * kNcols2;

    static_assert(kQHeads / kKVHeads == kGroup, "gqa ratio mismatch");
    static_assert(kGroup % kNcols2 == 0, "ncols2 must divide the GQA group");
    static_assert(kNcols >= 32, "Volta compiles no tile smaller than 32 columns");
};

// The key extent handed to the kernel is padded to FATTN_KQ_STRIDE. Upstream
// always satisfies this (llama.cpp pads its KV cache to 256 whenever FA is on), so
// the kernel's last key tile is free to read a whole tile past the true key count.
// Without the padding those reads would run off the end of a mask row and into the
// next one, which a multiple-of-32 key count happens to hide.
constexpr int kKeyPad = FATTN_KQ_STRIDE;

constexpr int round_up_keys(int n) { return ((n + kKeyPad - 1) / kKeyPad) * kKeyPad; }

// Report append/gather/attention CUDA-event deltas every 256 launches when
// NINFER_FLASH_TIMING=1. persist=1/0 tags which gather path is active.
inline void flash_timing_report(bool timing_on, cudaEvent_t t0, cudaEvent_t t1, cudaEvent_t t2,
                                cudaEvent_t t3, int* timing_n, int width, int n_kv,
                                bool persist_gather, cudaStream_t stream) {
    if (!timing_on) { return; }
    cudaEventRecord(t3, stream);
    const cudaError_t sync_err = cudaEventSynchronize(t3);
    // Timing is best-effort: never leave a sticky error for the next CUDA_CHECK.
    (void)cudaGetLastError();
    if (sync_err != cudaSuccess) { return; }
    float append_ms = 0, gather_ms = 0, attn_ms = 0;
    cudaEventElapsedTime(&append_ms, t0, t1);
    cudaEventElapsedTime(&gather_ms, t1, t2);
    cudaEventElapsedTime(&attn_ms, t2, t3);
    (void)cudaGetLastError();
    const int n = ++(*timing_n);
    if ((n & 0xF) == 1) {
        std::fprintf(stderr,
                     "[flash-timing] n=%d persist=%d width=%d n_kv=%d append=%.3fms "
                     "gather=%.3fms attn=%.3fms total=%.3fms\n",
                     n, persist_gather ? 1 : 0, width, n_kv, append_ms, gather_ms, attn_ms,
                     append_ms + gather_ms + attn_ms);
    }
    (void)n;
}

// ---------------------------------------------------------------------------
// Staging kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ const std::int32_t* select_block_table(
        const std::int32_t* block_tables, const std::int32_t* table_rows, int logical_pages) {
    return block_tables + static_cast<std::int64_t>(table_rows[0]) * logical_pages;
}

// Append the whole width's K/V into the paged cache in one launch. The chunked
// route does this per 5-token chunk; a flash tiling needs the full range resident
// before the first Q-block runs, and the causal mask is what keeps a query from
// seeing a key it must not.
template <int kKVHeads>
__global__ void volta_flash_append_kv_kernel(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ table_rows,
        __nv_bfloat16* __restrict__ k_pages, __nv_bfloat16* __restrict__ v_pages,
        const std::int32_t* __restrict__ block_tables, int logical_pages, int width) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (t >= width) { return; }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int32_t position     = positions[t];
    const std::int64_t dst =
        paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, position, d);
    const std::int64_t src =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * t);

    k_pages[dst] = k[src];
    v_pages[dst] = v[src];
}

__device__ __forceinline__ float volta_flash_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

// BF16 input -> paged INT8-G64 cache. Eight warps independently own one
// (token, kv-head, 64-d group), matching the cache's public quantization
// contract and the existing prefill append kernel.
template <int kKVHeads>
__launch_bounds__(256) __global__ void volta_flash_append_kv_i8_kernel(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ table_rows,
        std::int8_t* __restrict__ k_pages, std::int8_t* __restrict__ v_pages,
        half* __restrict__ k_scales, half* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, int logical_pages, int width) {
    constexpr int kWarps = 8;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int unit       = static_cast<int>(blockIdx.x) * kWarps + warp;
    const int units      = width * kKVHeads * kGqaKvQuantGroups;
    if (unit >= units) { return; }

    const int group   = unit % kGqaKvQuantGroups;
    const int tmp     = unit / kGqaKvQuantGroups;
    const int h       = tmp % kKVHeads;
    const int token   = tmp / kKVHeads;
    const int d0      = group * kGqaKvQuantGroup + lane;
    const std::int64_t src0 =
        static_cast<std::int64_t>(d0) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * token);
    const std::int64_t src1 = src0 + 32;

    const float k0 = __bfloat162float(k[src0]);
    const float k1 = __bfloat162float(k[src1]);
    const float v0 = __bfloat162float(v[src0]);
    const float v1 = __bfloat162float(v[src1]);
    const float ka = volta_flash_warp_max(fmaxf(fabsf(k0), fabsf(k1)));
    const float va = volta_flash_warp_max(fmaxf(fabsf(v0), fabsf(v1)));
    const half ks  = __float2half_rn(ka > 0.0f ? ka / 127.0f : 0.0f);
    const half vs  = __float2half_rn(va > 0.0f ? va / 127.0f : 0.0f);
    const float ksf = __half2float(ks);
    const float vsf = __half2float(vs);
    const float ki  = ksf > 0.0f ? 1.0f / ksf : 0.0f;
    const float vi  = vsf > 0.0f ? 1.0f / vsf : 0.0f;

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const int position               = positions[token];
    int physical_page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page     = __shfl_sync(0xffffffffu, physical_page, 0);
    const int page_offset = position & kPagedKVPageMask;
    const std::int64_t code_base = paged_kv_element_offset<kHeadDim, kKVHeads>(
        physical_page, h, page_offset, group * kGqaKvQuantGroup);

    k_pages[code_base + lane]      = gqa_kv_quant_code(k0, ki);
    k_pages[code_base + lane + 32] = gqa_kv_quant_code(k1, ki);
    v_pages[code_base + lane]      = gqa_kv_quant_code(v0, vi);
    v_pages[code_base + lane + 32] = gqa_kv_quant_code(v1, vi);
    if (lane == 0) {
        const std::int64_t scale_offset = paged_kv_element_offset<kGqaKvQuantGroups, kKVHeads>(
            physical_page, h, page_offset, group);
        k_scales[scale_offset] = ks;
        v_scales[scale_offset] = vs;
    }
}

// BF16 input -> paged IQ4_NL-G32 cache. One warp owns one (token, kv-head, 32-d group) with one
// value per lane; the encode is the shared gqa_i4nl_encode_warp pipeline, so the codes and scales
// are bit-identical to gqa_kv_append and the ChunkedSmallT fill. Eight warps per CTA.
template <int kKVHeads>
__launch_bounds__(256) __global__ void volta_flash_append_kv_i4nl_kernel(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ table_rows,
        std::int8_t* __restrict__ k_pages, std::int8_t* __restrict__ v_pages,
        half* __restrict__ k_scales, half* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, int logical_pages, int width) {
    constexpr int kWarps        = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int kGroups       = kGqaI4NLGroups;
    __shared__ std::int8_t lut_s[16];
    if (threadIdx.x < 16) { lut_s[threadIdx.x] = gqa_i4nl_value_at(static_cast<int>(threadIdx.x)); }
    __syncthreads();

    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int unit  = static_cast<int>(blockIdx.x) * kWarps + warp;
    const int units = width * kKVHeads * kGroups;
    if (unit >= units) { return; }

    const int group = unit % kGroups;
    const int tmp   = unit / kGroups;
    const int h     = tmp % kKVHeads;
    const int token = tmp / kKVHeads;
    const int d     = group * kGqaI4NLGroup + lane;

    const std::int64_t src =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * token);
    const float kx = __bfloat162float(k[src]);
    const float vx = __bfloat162float(v[src]);

    std::uint32_t k_byte, v_byte;
    half k_sh, v_sh;
    gqa_i4nl_encode_warp(kx, FullMask, lut_s, k_byte, k_sh);
    gqa_i4nl_encode_warp(vx, FullMask, lut_s, v_byte, v_sh);

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const int position              = positions[token];
    int physical_page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page     = __shfl_sync(FullMask, physical_page, 0);
    const int page_offset = position & kPagedKVPageMask;

    // The encode hands lanes [0,16) the packed code byte (lane b pairs 2b/2b+1), so each writer
    // stores byte (group, 2*lane) directly through the IQ4 code index.
    if (lane < 16) {
        const std::int64_t cb = gqa_i4nl_code_index<kKVHeads>(
            physical_page, h, group * kGqaI4NLGroup + 2 * lane, page_offset);
        k_pages[cb] = static_cast<std::int8_t>(k_byte & 0xffu);
        v_pages[cb] = static_cast<std::int8_t>(v_byte & 0xffu);
    }
    if (lane == 0) {
        const std::int64_t so =
            gqa_i4nl_scale_index<kKVHeads>(physical_page, h, group, page_offset);
        k_scales[so] = k_sh;
        v_scales[so] = v_sh;
    }
}

// Paged BF16 -> contiguous FP16, one launch per layer over the whole visible key
// range. ~50 MB of traffic per layer at 12K against the seconds the chunked route
// spends re-walking the key range; see the cost note in the plan.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_kernel(
        const __nv_bfloat16* __restrict__ k_pages, const __nv_bfloat16* __restrict__ v_pages,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded, int key_base = 0) {
    const int local = static_cast<int>(blockIdx.x);
    const int key   = local + key_base;
    const int h     = blockIdx.y;
    const int d     = threadIdx.x;
    if (local >= n_kv_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * local);

    // Padding rows are written as zero rather than left uninitialised: the mask
    // gives them weight zero, and 0 * NaN would still be NaN.
    if (key >= n_kv) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int64_t src = paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, key, d);

    k_out[dst] = __float2half(__bfloat162float(k_pages[src]));
    v_out[dst] = __float2half(__bfloat162float(v_pages[src]));
}

// Paged INT8-G64 -> contiguous FP16. Dequantizing once here preserves the
// flash kernel as the already-qualified arithmetic boundary and avoids the
// ChunkedSmallT path re-reading the full prefix for every five query tokens.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_i8_kernel(
        const std::int8_t* __restrict__ k_pages, const std::int8_t* __restrict__ v_pages,
        const half* __restrict__ k_scales, const half* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded, int key_base = 0) {
    const int local = static_cast<int>(blockIdx.x);
    const int key   = local + key_base;
    const int h     = blockIdx.y;
    const int d     = threadIdx.x;
    if (local >= n_kv_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * local);
    if (key >= n_kv) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int64_t code = paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, key, d);
    const int group         = d / kGqaKvQuantGroup;
    const std::int64_t scale =
        paged_kv_element_offset<kGqaKvQuantGroups, kKVHeads>(block_table, h, key, group);
    k_out[dst] = __float2half(static_cast<float>(k_pages[code]) * __half2float(k_scales[scale]));
    v_out[dst] = __float2half(static_cast<float>(v_pages[code]) * __half2float(v_scales[scale]));
}

// Paged IQ4_NL-G32 -> contiguous FP16, the page-aware strided counterpart of the INT8 gather.
// One thread dequantizes an 8-wide head-dim chunk (32 threads cover the 256 head dim); the four
// code bytes are one 32-bit load and the eight LUT entries expand through gqa_i4nl_staged_f16x8,
// the same single-RNE product the INT8 gather applies. Dequantizing once here keeps the vendored
// flash kernel as the already-qualified arithmetic boundary.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_i4nl_kernel(
        const std::int8_t* __restrict__ k_pages, const std::int8_t* __restrict__ v_pages,
        const half* __restrict__ k_scales, const half* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded, int key_base = 0) {
    __shared__ std::int8_t lut_s[16];
    if (threadIdx.x < 16) { lut_s[threadIdx.x] = gqa_i4nl_value_at(static_cast<int>(threadIdx.x)); }
    __syncthreads();

    const int local = static_cast<int>(blockIdx.x);
    const int key   = local + key_base;
    const int h     = blockIdx.y;
    const int chunk = threadIdx.x;
    if (local >= n_kv_padded) { return; }

    const int d0           = chunk * 8;
    const std::int64_t dst =
        static_cast<std::int64_t>(d0) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * local);
    if (key >= n_kv) {
        store_vec(k_out + dst, make_int4(0, 0, 0, 0));
        store_vec(v_out + dst, make_int4(0, 0, 0, 0));
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const int physical_page         = paged_kv_physical_page(block_table, key);
    const int page_offset           = key & kPagedKVPageMask;
    const int group                 = d0 >> 5;
    const std::int64_t code         = gqa_i4nl_code_index<kKVHeads>(physical_page, h, d0, page_offset);
    const std::int64_t ks           = gqa_i4nl_scale_index<kKVHeads>(physical_page, h, group, page_offset);
    const std::int64_t vs           = gqa_i4nl_scale_index<kKVHeads>(physical_page, h, group, page_offset);

    const std::uint32_t k_bytes = *reinterpret_cast<const std::uint32_t*>(k_pages + code);
    const std::uint32_t v_bytes = *reinterpret_cast<const std::uint32_t*>(v_pages + code);
    store_vec(k_out + dst, gqa_i4nl_staged_f16x8(k_bytes, __half2float(k_scales[ks]), lut_s));
    store_vec(v_out + dst, gqa_i4nl_staged_f16x8(v_bytes, __half2float(v_scales[vs]), lut_s));
}

// BF16 [256, QHeads, T] -> FP32, same element order (the kernel's Q and ninfer's q
// agree on layout; only the element type differs).
__global__ void volta_flash_convert_q_kernel(const __nv_bfloat16* __restrict__ q,
                                             float* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __bfloat162float(q[i]);
}

// FP32 -> BF16 on the way back out, same element order.
__global__ void volta_flash_convert_out_kernel(const float* __restrict__ in,
                                               __nv_bfloat16* __restrict__ out,
                                               std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __float2bfloat16(in[i]);
}

// Causal mask for one Q-block, built from the device-side positions rather than
// inferred from the envelope, so a non-sequential position layout cannot silently
// produce a wrong mask. Padding rows are fully masked; their outputs are discarded.
__global__ void volta_flash_build_mask_kernel(const std::int32_t* __restrict__ positions,
                                              int token_begin, int rows_valid,
                                              half* __restrict__ mask, int n_kv, int mask_stride) {
    const int row = blockIdx.x;
    const std::int32_t last_visible = row < rows_valid ? positions[token_begin + row] : -1;

    half* mask_row = mask + static_cast<std::int64_t>(row) * mask_stride;
    for (int key = threadIdx.x; key < n_kv; key += blockDim.x) {
        mask_row[key] = key <= last_visible ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

// ---------------------------------------------------------------------------
// Kernel launch: launch_fattn's stream-K decomposition, without the ggml types.
// ---------------------------------------------------------------------------

struct FlashLaunchConfig {
    int    nthreads       = 0;
    int    nwarps         = 0;
    int    nbatch_fa      = 0;
    size_t nbytes_shared  = 0;
    int    blocks_per_sm  = 0;
    int    nsm            = 0;
};

// One cached config per geometry: the shared-memory and occupancy figures depend
// only on the tiling, so they are resolved once per instantiation.
template <typename Geometry>
const FlashLaunchConfig& flash_launch_config() {
    using P = VoltaFlashParams<Geometry>;

    static std::mutex mutex;
    static std::map<int, FlashLaunchConfig> configs;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = configs.find(device); found != configs.end()) { return found->second; }
    const FlashLaunchConfig config = [device] {
        FlashLaunchConfig c;

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        c.nsm = prop.multiProcessorCount;

        const int cc = prop.major * 100 + prop.minor * 10;

        const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (kDKQ, kDV, P::kNcols, cc);
        const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (kDKQ, kDV, P::kNcols, cc);
        const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(kDKQ, kDV, P::kNcols, cc);
        const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (kDKQ, kDV, P::kNcols, cc);
        const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (kDKQ, kDV, P::kNcols1, P::kNcols2, cc);

        c.nthreads  = ggml_cuda_fattn_mma_get_nthreads (kDKQ, kDV, P::kNcols, cc);
        c.nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(kDKQ, kDV, P::kNcols, cc);
        c.nwarps    = c.nthreads / WARP_SIZE;

        const int cols_per_warp = std::min(P::kNcols, get_cols_per_warp(cc));

        const size_t shared_KV_1stage = size_t(c.nbatch_fa) * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_KV_2stage = size_t(c.nbatch_fa) *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_Q         = size_t(P::kNcols)      * (kDKQ/2 + 4)                            * sizeof(half2);
        const size_t shared_mask      = size_t(P::kNcols1)     * (c.nbatch_fa/2 + 4)                     * sizeof(half2);
        const size_t shared_combine   = size_t(c.nwarps)*cols_per_warp * (nbatch_combine + 4)         * sizeof(half2);
        const size_t shared_KV        = nstages <= 1 ? shared_KV_1stage : shared_KV_2stage;

        c.nbytes_shared = std::max(shared_combine, Q_in_reg
            ? std::max(shared_Q, shared_KV + shared_mask)
            :          shared_Q + shared_KV + shared_mask);

        auto kernel = flash_attn_ext_f16<kDKQ, kDV, P::kNcols1, P::kNcols2, false, false>;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, c.nbytes_shared));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &c.blocks_per_sm, reinterpret_cast<const void *>(kernel), c.nthreads, c.nbytes_shared));
        if (c.blocks_per_sm <= 0) { c.blocks_per_sm = 1; }
        return c;
    }();
    return configs.emplace(device, config).first->second;
}

// One Q-block through the vendored kernel plus its stream-K fixup.
template <typename Geometry>
void launch_flash_block(const float* q_f32, const half* k_f16, const half* v_f16, const half* mask,
                        float* dst, float2* dst_meta, int tokens, int n_kv, int mask_stride,
                        float scale, cudaStream_t stream) {
    using P                    = VoltaFlashParams<Geometry>;
    constexpr int kQHeads      = P::kQHeads;
    constexpr int kKVHeads     = P::kKVHeads;
    constexpr int kGroup       = P::kGroup;
    constexpr int kNcols1      = P::kNcols1;
    constexpr int kNcols2      = P::kNcols2;

    const FlashLaunchConfig& c = flash_launch_config<Geometry>();

    const int ntiles_x     = (tokens + kNcols1 - 1) / kNcols1;
    const int ntiles_z_gqa = (kGroup + kNcols2 - 1) / kNcols2;
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * kKVHeads;
    const int ntiles_KV    = (n_kv + c.nbatch_fa - 1) / c.nbatch_fa;

    const int max_blocks = c.blocks_per_sm * c.nsm;

    const int raw     = std::min(max_blocks, ntiles_KV * ntiles_dst);
    const int rounded = (raw / ntiles_dst) * ntiles_dst;
    const int loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    const int nblocks = loss <= 5 ? rounded : raw;

    const dim3 block_dim(WARP_SIZE, c.nwarps, 1);
    const dim3 blocks_num(nblocks, 1, 1);

    const std::int32_t nb01 = kDKQ * kQHeads  * sizeof(float);
    const std::int32_t nb02 = kDKQ            * sizeof(float);
    const std::int32_t nb11 = kDKQ * kKVHeads * sizeof(half);
    const std::int32_t nb12 = kDKQ            * sizeof(half);
    const std::int32_t nb31 = mask_stride     * sizeof(half);

    const uint3 ne01_fd = init_fastdiv_values(tokens);

    flash_attn_ext_f16<kDKQ, kDV, kNcols1, kNcols2, false, false>
        <<<blocks_num, block_dim, c.nbytes_shared, stream>>>(
            reinterpret_cast<const char *>(q_f32),
            reinterpret_cast<const char *>(k_f16),
            reinterpret_cast<const char *>(v_f16),
            reinterpret_cast<const char *>(mask),
            nullptr, nullptr, dst, dst_meta,
            scale, 0.0f, 1.0f, 1.0f, /*n_head_log2=*/16u, 0.0f,
            kDKQ, ne01_fd, kQHeads, 1, nb01, nb02, 0,
            kDKQ, n_kv, kKVHeads, 1, nb11, nb12, 0,
            nb11, nb12, 0,
            tokens, 1, 1,
            nb31, 0, 0);

    if (nblocks % ntiles_dst == 0 && nblocks > ntiles_dst) {
        const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * kKVHeads);
        const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
        const uint3 fd2 = init_fastdiv_values(ntiles_x);

        flash_attn_stream_k_fixup_uniform<kDV, kNcols1, kNcols2>
            <<<dim3((unsigned) ntiles_dst, kNcols1, kNcols2), dim3(kDV, 1, 1), 0, stream>>>(
                dst, dst_meta, tokens, kQHeads, kKVHeads, nblocks, kGroup,
                nblocks / ntiles_dst, fd0, fd1, fd2);
    } else if (ntiles_dst % nblocks != 0) {
        const int total_work = ntiles_KV * ntiles_dst;

        const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * kKVHeads);
        const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
        const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
        const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

        flash_attn_stream_k_fixup_general<kDV, kNcols1, kNcols2>
            <<<dim3((unsigned) nblocks, kNcols1, kNcols2), dim3(kDV, 1, 1), 0, stream>>>(
                dst, dst_meta, tokens, kQHeads, kGroup, total_work,
                fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
    }
}

// ---------------------------------------------------------------------------

template <typename Geometry>
std::size_t meta_elements_impl(std::int32_t tokens) {
    using P                    = VoltaFlashParams<Geometry>;
    const FlashLaunchConfig& c = flash_launch_config<Geometry>();
    const int ntiles_x         = (tokens + P::kNcols1 - 1) / P::kNcols1;
    const int ntiles_dst = ntiles_x * ((P::kGroup + P::kNcols2 - 1) / P::kNcols2) * P::kKVHeads;
    const int max_blocks = c.blocks_per_sm * c.nsm;
    // Upper bound: the launch never uses more blocks than this.
    const int nblocks = std::max(max_blocks, ntiles_dst);
    return static_cast<std::size_t>(nblocks) * P::kNcols * (2 + kDV / 2);
}

template <typename Geometry>
void volta_flash_launch_impl(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& table_rows, float scale,
                             PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                             std::int32_t q_block_tokens, Tensor& k_gathered, Tensor& v_gathered,
                             Tensor& mask, Tensor& q_f32, Tensor& out_f32, Tensor& dst_meta,
                             Tensor& out, cudaStream_t stream) {
    using P                          = VoltaFlashParams<Geometry>;
    constexpr int kQHeads            = P::kQHeads;
    constexpr int kKVHeads           = P::kKVHeads;
    const std::int32_t width         = q.ne[2];
    const std::int32_t n_kv_total    = static_cast<std::int32_t>(envelope.max_visible_keys);
    const std::int32_t logical_pages = cache.block_tables.ne[0];

    auto* block_tables = static_cast<const std::int32_t*>(cache.block_tables.data);
    auto* rows         = static_cast<const std::int32_t*>(table_rows.data);
    auto* position_ptr = static_cast<const std::int32_t*>(positions.data);

    // Phase timing (NINFER_FLASH_TIMING=1): CUDA-event deltas for append/gather/attention
    // on the current stream, reported every 128 launches. Cost when off: two getenv once.
    static const bool timing_on = std::getenv("NINFER_FLASH_TIMING") != nullptr;
    static cudaEvent_t t0, t1, t2, t3;
    static int timing_n = 0;
    if (timing_on && t0 == nullptr) {
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventCreate(&t2);
        cudaEventCreate(&t3);
        (void)cudaGetLastError();
    }

    // 1. Append this call's K/V for the whole width.
    if (timing_on) {
        cudaEventRecord(t0, stream);
        (void)cudaGetLastError(); // timing must not poison the append CUDA_CHECK below
    }
    if (cache.dtype == DType::I8) {
        constexpr int kWarpsPerBlock = 8;
        const int units              = width * kKVHeads * kGqaKvQuantGroups;
        const int blocks             = (units + kWarpsPerBlock - 1) / kWarpsPerBlock;
        volta_flash_append_kv_i8_kernel<kKVHeads><<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            position_ptr, rows, static_cast<std::int8_t*>(cache.k_pages.data),
            static_cast<std::int8_t*>(cache.v_pages.data),
            static_cast<half*>(cache.k_scale_pages.data),
            static_cast<half*>(cache.v_scale_pages.data), block_tables, logical_pages, width);
    } else if (cache.dtype == DType::I4NL) {
        constexpr int kWarpsPerBlock = 8;
        const int units              = width * kKVHeads * kGqaI4NLGroups;
        const int blocks             = (units + kWarpsPerBlock - 1) / kWarpsPerBlock;
        volta_flash_append_kv_i4nl_kernel<kKVHeads><<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            position_ptr, rows, static_cast<std::int8_t*>(cache.k_pages.data),
            static_cast<std::int8_t*>(cache.v_pages.data),
            static_cast<half*>(cache.k_scale_pages.data),
            static_cast<half*>(cache.v_scale_pages.data), block_tables, logical_pages, width);
    } else {
        volta_flash_append_kv_kernel<kKVHeads><<<dim3(width, kKVHeads), kHeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            position_ptr, rows, static_cast<__nv_bfloat16*>(cache.k_pages.data),
            static_cast<__nv_bfloat16*>(cache.v_pages.data), block_tables, logical_pages, width);
    }
    CUDA_CHECK(cudaGetLastError());
    if (timing_on) {
        cudaEventRecord(t1, stream);
        (void)cudaGetLastError();
    }

    // 2. Gather the visible key range paged -> contiguous FP16, once for every
    //    Q-block of this layer.
    //    NINFER_FLASH_PERSIST_GATHER=1 uses a persistent 512 MiB buffer (key_base
    //    incremental supported when base==flash_gathered); default uses per-call workspace.
    static const bool persist_gather =
        std::getenv("NINFER_FLASH_PERSIST_GATHER") != nullptr &&
        std::string(std::getenv("NINFER_FLASH_PERSIST_GATHER")) != "0";
    static half* flash_gather_k = nullptr;
    static half* flash_gather_v = nullptr;
    static int flash_gather_capacity = 0;
    static int flash_gathered = 0;
    static const void* flash_gather_layer = nullptr;

    const std::int32_t n_kv_alloc = round_up_keys(n_kv_total);
    half* gk = static_cast<half*>(k_gathered.data);
    half* gv = static_cast<half*>(v_gathered.data);
    int gather_begin = 0;
    if (persist_gather) {
        if (flash_gather_k == nullptr || flash_gather_capacity < n_kv_alloc) {
            if (flash_gather_k != nullptr) {
                cudaFree(flash_gather_k);
                cudaFree(flash_gather_v);
            }
            const int cap = std::max(n_kv_alloc, 262144 + 1024);
            const std::size_t elems =
                static_cast<std::size_t>(cap) * static_cast<std::size_t>(kKVHeads) * kHeadDim;
            CUDA_CHECK(cudaMalloc(&flash_gather_k, elems * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&flash_gather_v, elems * sizeof(half)));
            // Eager-map every page: an untouched cudaMalloc buffer can fault on first
            // touch inside the hot gather/attention path, which the phase timers then
            // mis-attribute to surrounding work.
            cudaMemset(flash_gather_k, 0, elems * sizeof(half));
            cudaMemset(flash_gather_v, 0, elems * sizeof(half));
            flash_gather_capacity = cap;
            flash_gathered        = 0;
            flash_gather_layer    = nullptr;
        }
        gk = flash_gather_k;
        gv = flash_gather_v;
        const void* layer_id = cache.k_pages.data;
        const std::int32_t call_base = n_kv_total - width;
        if (layer_id == flash_gather_layer && call_base == flash_gathered &&
            n_kv_total > flash_gathered && n_kv_total <= flash_gather_capacity) {
            gather_begin = flash_gathered;
        }
    }
    {
        const int gb = gather_begin;
        const int gp = round_up_keys(n_kv_total - gb);
        half* dk = gk + static_cast<std::int64_t>(gb) * kKVHeads * kHeadDim;
        half* dv = gv + static_cast<std::int64_t>(gb) * kKVHeads * kHeadDim;
        if (cache.dtype == DType::I8) {
            volta_flash_gather_kv_i8_kernel<kKVHeads>
                <<<dim3(gp, kKVHeads), kHeadDim, 0, stream>>>(
                    static_cast<const std::int8_t*>(cache.k_pages.data),
                    static_cast<const std::int8_t*>(cache.v_pages.data),
                    static_cast<const half*>(cache.k_scale_pages.data),
                    static_cast<const half*>(cache.v_scale_pages.data), block_tables, rows,
                    logical_pages, dk, dv, n_kv_total, gp, gb);
        } else if (cache.dtype == DType::I4NL) {
            volta_flash_gather_kv_i4nl_kernel<kKVHeads>
                <<<dim3(gp, kKVHeads), kHeadDim / 8, 0, stream>>>(
                    static_cast<const std::int8_t*>(cache.k_pages.data),
                    static_cast<const std::int8_t*>(cache.v_pages.data),
                    static_cast<const half*>(cache.k_scale_pages.data),
                    static_cast<const half*>(cache.v_scale_pages.data), block_tables, rows,
                    logical_pages, dk, dv, n_kv_total, gp, gb);
        } else {
            volta_flash_gather_kv_kernel<kKVHeads>
                <<<dim3(gp, kKVHeads), kHeadDim, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(cache.k_pages.data),
                    static_cast<const __nv_bfloat16*>(cache.v_pages.data), block_tables, rows,
                    logical_pages, dk, dv, n_kv_total, gp, gb);
        }
        CUDA_CHECK(cudaGetLastError());
        if (persist_gather && gb + (n_kv_total - gb) == n_kv_total) {
            flash_gather_layer = cache.k_pages.data;
            flash_gathered     = n_kv_total;
        }
    }
    if (timing_on) {
        cudaEventRecord(t2, stream);
        (void)cudaGetLastError();
    }
    half* k_contig = gk;
    half* v_contig = gv;

    // 3. Attention. Prefer one Split-D launch over the whole Q chunk (up to kMaxRows=8192)
    //    when the positions layout allows it — matches the upstream q8192 path. Otherwise
    //    fall through to 1024-row blocks on the vendored llama.cpp kernel.
    const std::int32_t base = n_kv_total - width;
    // Verify positions layout once for the whole launch (see splitd_positions_are_contiguous).
    // Per-Q-block verification would sync the stream ~width/1024 times per layer.
    const bool splitd_positions_ok =
        volta_splitd_requested() && splitd_positions_are_contiguous(position_ptr, base, width, stream);
    {
        static int calls = 0, ok = 0;
        ++calls;
        if (splitd_positions_ok) ++ok;
        if ((calls & 0xFF) == 1 && std::getenv("NINFER_SPLITD_DEBUG") != nullptr) {
            std::fprintf(stderr, "[splitd-flash] launches=%d pos_ok=%d width=%d n_kv=%d base=%d\n",
                         calls, ok, width, n_kv_total, base);
        }
    }
    if (splitd_positions_ok) {
        const auto* q_ptr = static_cast<const __nv_bfloat16*>(q.data);
        auto* out_ptr     = static_cast<__nv_bfloat16*>(out.data);
        bool all_done     = true;
        constexpr std::int32_t kMaxSplitdRows = 8192;
        for (std::int32_t begin = 0; begin < width; begin += kMaxSplitdRows) {
            const int rows = std::min<std::int32_t>(kMaxSplitdRows, width - begin);
            const auto* q_row = q_ptr + static_cast<std::int64_t>(begin) * kQHeads * kHeadDim;
            auto* out_row = out_ptr + static_cast<std::int64_t>(begin) * kQHeads * kHeadDim;
            if (!volta_splitd_block<Geometry>(q_row, k_contig, v_contig, rows,
                                              n_kv_total, base + begin,
                                              /*positions_contiguous=*/true, scale, out_row,
                                              stream)) {
                all_done = false;
                // Continue with the vendored path for the remaining rows via the loop below.
                // (In practice splitd either accepts or rejects uniformly.)
                break;
            }
        }
        if (all_done) {
            if (timing_on) flash_timing_report(timing_on, t0, t1, t2, t3, &timing_n, width,
                                               n_kv_total, persist_gather, stream);
            return;
        }
    }
    for (std::int32_t begin = 0; begin < width; begin += q_block_tokens) {
        const std::int32_t tokens = std::min(q_block_tokens, width - begin);
        const std::int32_t n_kv   = round_up_keys(base + begin + tokens);
        const std::int32_t rows_padded =
            ((tokens + kVoltaFlashMaskRowPad - 1) / kVoltaFlashMaskRowPad) * kVoltaFlashMaskRowPad;

        const std::int64_t q_count = static_cast<std::int64_t>(tokens) * kQHeads * kHeadDim;

        // Split-D route: the 1CatAI kernel masks causally from (kv_len, kv_offset) itself,
        // so it needs neither this route's mask nor its Q/out f32 staging conversions.
        if (splitd_positions_ok) {
            const auto* q_row = static_cast<const __nv_bfloat16*>(q.data) +
                                static_cast<std::int64_t>(begin) * kQHeads * kHeadDim;
            auto* out_row = static_cast<__nv_bfloat16*>(out.data) +
                            static_cast<std::int64_t>(begin) * kQHeads * kHeadDim;
            if (volta_splitd_block<Geometry>(
                    q_row, k_contig, v_contig, tokens, n_kv, base + begin,
                    /*positions_contiguous=*/true, scale, out_row, stream)) {
                continue;
            }
        }

        constexpr int kConvertThreads = 256;
        const int convert_blocks =
            static_cast<int>((q_count + kConvertThreads - 1) / kConvertThreads);

        volta_flash_convert_q_kernel<<<convert_blocks, kConvertThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data) +
                static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
            static_cast<float*>(q_f32.data), q_count);

        // The mask extent is the padded key count, so every key the kernel can touch
        // has a defined mask entry; keys past a row's causal limit -- padding included
        // -- get -inf from the same comparison.
        volta_flash_build_mask_kernel<<<rows_padded, 256, 0, stream>>>(
            position_ptr, begin, tokens, static_cast<half*>(mask.data), n_kv, n_kv);

        cudaMemsetAsync(out_f32.data, 0, static_cast<std::size_t>(q_count) * sizeof(float), stream);

        launch_flash_block<Geometry>(static_cast<const float*>(q_f32.data),
                           static_cast<const half*>(k_contig),
                           static_cast<const half*>(v_contig),
                           static_cast<const half*>(mask.data),
                           static_cast<float*>(out_f32.data),
                           static_cast<float2*>(dst_meta.data), tokens, n_kv, n_kv, scale, stream);

        volta_flash_convert_out_kernel<<<convert_blocks, kConvertThreads, 0, stream>>>(
            static_cast<const float*>(out_f32.data),
            static_cast<__nv_bfloat16*>(out.data) +
                static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
            q_count);
    }
    if (timing_on)
        flash_timing_report(timing_on, t0, t1, t2, t3, &timing_n, width, n_kv_total,
                            persist_gather, stream);
}

} // namespace

// The two registered geometries differ only in tiling; both are instantiated so
// the route can serve 27B (24q/4kv) and 35B-A3B (16q/2kv) from one launcher.
std::size_t gqa_attention_volta_flash_meta_elements(std::int32_t q_heads, std::int32_t tokens) {
    if (q_heads == Gqa27Geometry::QHeads) { return meta_elements_impl<Gqa27Geometry>(tokens); }
    if (q_heads == Gqa27Tp2Geometry::QHeads) { return meta_elements_impl<Gqa27Tp2Geometry>(tokens); }
    if (q_heads == Gqa27Tp4Geometry::QHeads) { return meta_elements_impl<Gqa27Tp4Geometry>(tokens); }
    if (q_heads == Gqa35Geometry::QHeads) { return meta_elements_impl<Gqa35Geometry>(tokens); }
    throw std::invalid_argument("gqa_attention volta flash: unsupported Q head geometry");
}

void gqa_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& positions, const Tensor& table_rows,
                                      float scale, PagedKVBatchLayerView cache,
                                      GqaExecutionEnvelope envelope, std::int32_t q_block_tokens,
                                      Tensor& k_gathered, Tensor& v_gathered, Tensor& mask,
                                      Tensor& q_f32, Tensor& out_f32, Tensor& dst_meta, Tensor& out,
                                      cudaStream_t stream) {
    if (q.ne[1] == Gqa27Tp2Geometry::QHeads) {
        volta_flash_launch_impl<Gqa27Tp2Geometry>(q, k, v, positions, table_rows, scale, cache,
                                                  envelope, q_block_tokens, k_gathered, v_gathered,
                                                  mask, q_f32, out_f32, dst_meta, out, stream);
        return;
    }
    if (q.ne[1] == Gqa27Tp4Geometry::QHeads) {
        volta_flash_launch_impl<Gqa27Tp4Geometry>(q, k, v, positions, table_rows, scale, cache,
                                                  envelope, q_block_tokens, k_gathered, v_gathered,
                                                  mask, q_f32, out_f32, dst_meta, out, stream);
        return;
    }
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        volta_flash_launch_impl<Gqa27Geometry>(q, k, v, positions, table_rows, scale, cache,
                                               envelope, q_block_tokens, k_gathered, v_gathered,
                                               mask, q_f32, out_f32, dst_meta, out, stream);
        return;
    }
    if (q.ne[1] == Gqa35Geometry::QHeads) {
        volta_flash_launch_impl<Gqa35Geometry>(q, k, v, positions, table_rows, scale, cache,
                                               envelope, q_block_tokens, k_gathered, v_gathered,
                                               mask, q_f32, out_f32, dst_meta, out, stream);
        return;
    }
    throw std::invalid_argument("gqa_attention volta flash: unsupported Q head geometry");
}

} // namespace ninfer::ops::detail
