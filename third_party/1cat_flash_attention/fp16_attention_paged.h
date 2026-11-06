#pragma once

#include <cuda_runtime.h>

namespace ninfer::third_party::onecat {

void flash_attention_forward_f16_paged_launch(
    const void* q, const void* k_cache, const void* v_cache, void* out,
    float* softmax_lse, const int* block_table, const int* seq_lens, int B, int H,
    int M, int page_block_size, int num_kv_heads, int max_blocks,
    long long k_block_stride, long long k_token_stride, long long k_head_stride,
    long long v_block_stride, long long v_token_stride, long long v_head_stride,
    float softmax_scale, bool causal, float k_scale, float v_scale,
    int window_size_left, int window_size_right, cudaStream_t stream);

}  // namespace ninfer::third_party::onecat
