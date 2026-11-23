#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

void mtp_decode_fp8_block128(const std::uint8_t* codes, const std::uint16_t* scales,
                             std::int32_t rows, std::int32_t columns, void* out_fp16,
                             cudaStream_t stream);

void mtp_silu_mul_f16(const void* gate_up, void* activation, std::int32_t count,
                      cudaStream_t stream);

void mtp_scale_add_f16(const void* source, float scale, float* total, std::int32_t count,
                       cudaStream_t stream);

void mtp_round_f16(const float* total, void* destination, std::int32_t count,
                   cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
