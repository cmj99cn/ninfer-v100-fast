#pragma once

#include <cuda_runtime.h>

namespace ninfer::third_party::onecat {

// Dense FP16 causal attention for head_dim=256, [B,H,M,D] x [B,H_KV,N,D].
void flash_attention_forward_f16_dense_launch(
    const void* q, const void* k, const void* v, void* out, float* softmax_lse,
    int B, int H, int H_KV, int M, int N, float softmax_scale, bool scalar_pv,
    int window_size_left, int window_size_right, cudaStream_t stream);

}  // namespace ninfer::third_party::onecat
