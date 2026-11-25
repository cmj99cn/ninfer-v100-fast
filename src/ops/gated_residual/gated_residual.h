#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void hc_expand_norm_launch(const Tensor& embedding, const Tensor& norm_weight, float epsilon,
                           Tensor& residual, Tensor& normalized, cudaStream_t stream);

void hc_grouped_rmsnorm_launch(const Tensor& residual, const Tensor& norm_weight, float epsilon,
                                Tensor& normalized, cudaStream_t stream);

void hc_combine_launch(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                       Tensor& updated_residual, cudaStream_t stream);

void hc_combine_norm_launch(const Tensor& residual, const Tensor& block_output,
                            const Tensor& injection, const Tensor& next_norm_weight, float epsilon,
                            Tensor& updated_residual, Tensor& normalized, cudaStream_t stream);

void hc_mix_silu_launch(Tensor& values, std::int32_t streams, cudaStream_t stream);

void hc_mix_up_gate_launch(const void* weight, std::int32_t hidden, std::int32_t rank,
                            std::int32_t streams, const void* low, const Tensor& normalized,
                            Tensor& block_input, DType dtype, cudaStream_t stream);

void hc_mix_up_gate_multi_launch(const void* weight, std::int32_t hidden, std::int32_t rank,
                                  std::int32_t streams, const void* low, const Tensor& normalized,
                                  Tensor& block_input, DType dtype, std::int32_t columns,
                                  cudaStream_t stream);

void hc_mix_down_silu_launch(const void* weight, std::int32_t rows, std::int32_t depth,
                              const void* activation, Tensor& output, const void* injection_rows,
                              DType weight_dtype, Tensor* injection, std::int32_t streams,
                              cudaStream_t stream);

void hc_mix_down_silu_multi_launch(const void* weight, std::int32_t rows, std::int32_t depth,
                                    const void* activation, Tensor& output,
                                    const void* injection_rows, DType weight_dtype,
                                    Tensor* injection, std::int32_t streams, std::int32_t columns,
                                    cudaStream_t stream);



void hc_project_gemv_launch(const void* weight, std::int32_t rows, std::int32_t depth,
                            const void* activation, Tensor& output, cudaStream_t stream);

void hc_mix_gate_launch(const Tensor& gates, const Tensor& normalized, Tensor& block_input,
                        std::int32_t hidden, std::int32_t streams, cudaStream_t stream);

void hc_mix_injection_launch(const void* injection_rows, DType weight_dtype,
                             const Tensor& flattened, std::int32_t channels, std::int32_t streams,
                             Tensor& injection, cudaStream_t stream);

} // namespace ninfer::ops::detail
