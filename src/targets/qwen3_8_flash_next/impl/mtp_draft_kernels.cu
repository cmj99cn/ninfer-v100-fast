#include "targets/qwen3_8_flash_next/impl/mtp_draft_kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

__device__ __forceinline__ float decode_e4m3(std::uint8_t code) {
    const float sign            = (code & 0x80U) != 0 ? -1.0F : 1.0F;
    const std::uint8_t exponent = (code >> 3) & 0x0FU;
    const std::uint8_t fraction = code & 0x07U;
    if (exponent == 0) {
        if (fraction == 0) { return sign * 0.0F; }
        return sign * static_cast<float>(fraction) * (1.0F / 512.0F);
    }
    if (exponent == 0x0FU && fraction == 0x07U) {
        return __int_as_float(0x7FC00000);
    }
    return sign * (1.0F + static_cast<float>(fraction) / 8.0F) *
           ldexpf(1.0F, static_cast<int>(exponent) - 7);
}

__device__ __forceinline__ std::uint32_t pack_half2(float low, float high) {
    const __half2 pair = __floats2half2_rn(low, high);
    return *reinterpret_cast<const std::uint32_t*>(&pair);
}

__global__ void decode_fp8_block128_kernel(const std::uint8_t* __restrict__ codes,
                                           const std::uint16_t* __restrict__ scales,
                                           int rows, int columns,
                                           std::uint16_t* __restrict__ out) {
    const int groups_per_row = columns >> 7;
    const int words_per_row  = columns >> 2;
    const int total          = rows * words_per_row;
    const int index          = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                               static_cast<int>(threadIdx.x);
    if (index >= total) { return; }

    const int row  = index / words_per_row;
    const int word = index - row * words_per_row;
    const int scale_index = (row >> 7) * groups_per_row + (word >> 5);
    const float scale =
        __bfloat162float(__ushort_as_bfloat16(scales[scale_index]));
    const std::uint32_t packed =
        reinterpret_cast<const std::uint32_t*>(codes +
                                               static_cast<std::size_t>(row) * columns)[word];

    const float v0 = decode_e4m3(static_cast<std::uint8_t>(packed)) * scale;
    const float v1 = decode_e4m3(static_cast<std::uint8_t>(packed >> 8)) * scale;
    const float v2 = decode_e4m3(static_cast<std::uint8_t>(packed >> 16)) * scale;
    const float v3 = decode_e4m3(static_cast<std::uint8_t>(packed >> 24)) * scale;

    uint2 result;
    result.x = pack_half2(v0, v1);
    result.y = pack_half2(v2, v3);
    reinterpret_cast<uint2*>(out + static_cast<std::size_t>(row) * columns)[word] = result;
}

__global__ void silu_mul_f16_kernel(const std::uint16_t* __restrict__ gate_up,
                                    std::uint16_t* __restrict__ activation, int count) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
    if (index >= count) { return; }
    const float gate = __half2float(__ushort_as_half(gate_up[index]));
    const float up   = __half2float(__ushort_as_half(gate_up[count + index]));
    const float value = (gate / (1.0F + expf(-gate))) * up;
    activation[index] = __half_as_ushort(__float2half_rn(value));
}

__global__ void scale_add_f16_kernel(const std::uint16_t* __restrict__ source, float scale,
                                     float* __restrict__ total, int count) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
    if (index >= count) { return; }
    total[index] += scale * __half2float(__ushort_as_half(source[index]));
}

__global__ void round_f16_kernel(const float* __restrict__ total,
                                 std::uint16_t* __restrict__ destination, int count) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
    if (index >= count) { return; }
    destination[index] = __half_as_ushort(__float2half_rn(total[index]));
}

inline int block_count(int total, int threads) { return (total + threads - 1) / threads; }

} // namespace

void mtp_decode_fp8_block128(const std::uint8_t* codes, const std::uint16_t* scales,
                             std::int32_t rows, std::int32_t columns, void* out_fp16,
                             cudaStream_t stream) {
    if (rows <= 0 || columns <= 0 || (columns % 128) != 0) {
        throw std::invalid_argument("MTP draft: invalid block-FP8 decode geometry");
    }
    constexpr int kThreads = 256;
    const int words = rows * (columns / 4);
    decode_fp8_block128_kernel<<<block_count(words, kThreads), kThreads, 0, stream>>>(
        codes, scales, rows, columns, static_cast<std::uint16_t*>(out_fp16));
    CUDA_CHECK(cudaGetLastError());
}

void mtp_silu_mul_f16(const void* gate_up, void* activation, std::int32_t count,
                      cudaStream_t stream) {
    constexpr int kThreads = 256;
    silu_mul_f16_kernel<<<block_count(count, kThreads), kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(gate_up), static_cast<std::uint16_t*>(activation),
        count);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_scale_add_f16(const void* source, float scale, float* total, std::int32_t count,
                       cudaStream_t stream) {
    constexpr int kThreads = 256;
    scale_add_f16_kernel<<<block_count(count, kThreads), kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(source), scale, total, count);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_round_f16(const float* total, void* destination, std::int32_t count,
                   cudaStream_t stream) {
    constexpr int kThreads = 256;
    round_f16_kernel<<<block_count(count, kThreads), kThreads, 0, stream>>>(
        total, static_cast<std::uint16_t*>(destination), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
