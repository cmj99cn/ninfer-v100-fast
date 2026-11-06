#pragma once

#include <cuda_runtime.h>

namespace ninfer::third_party::onecat {

void gdn_forward_f16_launch(const void* q, const void* k, const void* v,
                            const void* gate, const void* beta,
                            const void* initial_state, void* output,
                            float* final_state, int batch, int tokens, int q_heads,
                            int v_heads, float scale, bool gate_is_exp,
                            int column_groups_per_block, cudaStream_t stream);

}  // namespace ninfer::third_party::onecat
