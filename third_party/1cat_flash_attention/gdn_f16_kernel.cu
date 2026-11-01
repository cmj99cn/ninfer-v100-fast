// FP16 GDN recurrent kernel adapted from 1CatAI/1Cat-vLLM v1.5.1
// `flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu` (Apache-2.0).
// Only the raw-pointer device kernel and typed launcher are retained.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* ptr,
                                                         int64_t index) {
  return __half2float(reinterpret_cast<const __half*>(ptr)[index]);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(
    const __nv_bfloat16* ptr, int64_t index) {
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[index]);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* ptr,
                                                 int64_t index,
                                                 float value) {
  ptr[index] = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_from_float<__half>(__half* ptr,
                                                           int64_t index,
                                                           float value) {
  reinterpret_cast<__half*>(ptr)[index] = __float2half(value);
}

template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(
    __nv_bfloat16* ptr,
    int64_t index,
    float value) {
  reinterpret_cast<__nv_bfloat16*>(ptr)[index] = __float2bfloat16(value);
}

__device__ __forceinline__ float subgroup_sum(float value, int width) {
  constexpr unsigned mask = 0xffffffffU;
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, width);
  }
  return value;
}

__device__ __forceinline__ float subgroup_broadcast(float value, int width) {
  return __shfl_sync(0xffffffffU, value, 0, width);
}

template <typename scalar_t,
          typename gate_t,
          typename beta_t,
          typename state_t,
          int K,
          int V,
          int COLS,
          int WIDTH,
          bool GateIsExp>
__global__ void gdn_forward_kernel(const scalar_t* __restrict__ q,
                                   const scalar_t* __restrict__ k,
                                   const scalar_t* __restrict__ v,
                                   const gate_t* __restrict__ gate,
                                   const beta_t* __restrict__ beta,
                                   const state_t* __restrict__ initial_state,
                                   scalar_t* __restrict__ output,
                                   float* __restrict__ final_state,
                                   int batch,
                                   int tokens,
                                   int q_heads,
                                   int v_heads,
                                   float scale,
                                   bool gate_is_exp) {
  static_assert(K % WIDTH == 0);
  constexpr int subgroups_per_warp = 32 / WIDTH;
  constexpr int rows_per_lane = K / WIDTH;

  const int hv = blockIdx.x;
  const int b = blockIdx.y;
  const int subgroup = threadIdx.x / WIDTH;
  const int lane = threadIdx.x % WIDTH;
  const int group_base =
      (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
  const int col_base = group_base * COLS;
  const int hq = hv / (v_heads / q_heads);

  float state_shard[COLS][rows_per_lane];

#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float value = 0.0F;
      if (col < V) {
        const int64_t state_index =
            (((static_cast<int64_t>(b) * v_heads + hv) * K + row) * V) + col;
        value = initial_state == nullptr ? 0.0F
                                         : load_as_float(initial_state, state_index);
      }
      state_shard[c][r] = value;
    }
  }

  for (int t = 0; t < tokens; ++t) {
    const int64_t gate_index =
        ((static_cast<int64_t>(b) * tokens + t) * v_heads + hv);
    float gate_value = 0.0F;
    float beta_value = 0.0F;
    if (threadIdx.x == 0) {
      const float gate_raw = load_as_float(gate, gate_index);
      gate_value = GateIsExp ? gate_raw : __expf(gate_raw);
      beta_value = load_as_float(beta, gate_index);
    }
    gate_value = __shfl_sync(0xffffffffU, gate_value, 0);
    beta_value = __shfl_sync(0xffffffffU, beta_value, 0);

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float kv_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      kv_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      const int64_t qk_index =
          (((static_cast<int64_t>(b) * tokens + t) * q_heads + hq) * K) + row;
      const float q_value = load_as_float(q, qk_index);
      const float k_value = load_as_float(k, qk_index);
      q_reg[r] = q_value;
      k_reg[r] = k_value;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        kv_partial[c] += state_shard[c][r] * k_value;
      }
    }

    float delta[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
      const float kv_col = subgroup_sum(kv_partial[c], WIDTH);
      float delta_value = 0.0F;
      if (lane == 0 && col < V) {
        const int64_t v_index =
            (((static_cast<int64_t>(b) * tokens + t) * v_heads + hv) * V) + col;
        delta_value =
            (load_as_float(v, v_index) - gate_value * kv_col) * beta_value;
      }
      delta[c] = subgroup_broadcast(delta_value, WIDTH);
    }

    float attn_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const float new_state =
            fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
        state_shard[c][r] = new_state;
        attn_partial[c] += new_state * q_reg[r];
      }
    }

#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = subgroup_sum(attn_partial[c], WIDTH);
    }

    if (lane == 0) {
      const int64_t out_base =
          (((static_cast<int64_t>(b) * tokens + t) * v_heads + hv) * V);
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
        if (col < V) {
          store_from_float(output, out_base + col, attn_partial[c] * scale);
        }
      }
    }
  }

  if (final_state != nullptr) {
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
#pragma unroll
      for (int r = 0; r < rows_per_lane; ++r) {
        const int row = r * WIDTH + lane;
        if (col < V) {
          const int64_t state_index =
              (((static_cast<int64_t>(b) * v_heads + hv) * K + row) * V) + col;
          final_state[state_index] = state_shard[c][r];
        }
      }
    }
  }
}

template <typename scalar_t,
          typename gate_t,
          typename beta_t,
          typename state_t,
          int K,
          int V,
          int COLS,
          int WIDTH,
          bool GateIsExp>
__global__ void gdn_forward_vlk_varlen_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const gate_t* __restrict__ gate,
    const beta_t* __restrict__ beta,
    const state_t* __restrict__ initial_state,
    const int32_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ output,
    float* __restrict__ final_state,
    int q_heads,
    int v_heads,
    float scale,
    bool gate_is_exp) {
  static_assert(K % WIDTH == 0);
  constexpr int subgroups_per_warp = 32 / WIDTH;
  constexpr int rows_per_lane = K / WIDTH;

  const int hv = blockIdx.x;
  const int n = blockIdx.y;
  const int subgroup = threadIdx.x / WIDTH;
  const int lane = threadIdx.x % WIDTH;
  const int group_base =
      (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
  const int col_base = group_base * COLS;
  const int hq = hv / (v_heads / q_heads);
  const int seq_start = cu_seqlens[n];
  const int seq_end = cu_seqlens[n + 1];

  float state_shard[COLS][rows_per_lane];

#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float value = 0.0F;
      if (col < V) {
        const int64_t state_index =
            (((static_cast<int64_t>(n) * v_heads + hv) * V + col) * K) + row;
        value = initial_state == nullptr ? 0.0F
                                         : load_as_float(initial_state, state_index);
      }
      state_shard[c][r] = value;
    }
  }

  for (int t = seq_start; t < seq_end; ++t) {
    const int64_t gate_index = (static_cast<int64_t>(t) * v_heads + hv);
    float gate_value = 0.0F;
    float beta_value = 0.0F;
    if (threadIdx.x == 0) {
      const float gate_raw = load_as_float(gate, gate_index);
      gate_value = GateIsExp ? gate_raw : __expf(gate_raw);
      beta_value = load_as_float(beta, gate_index);
    }
    gate_value = __shfl_sync(0xffffffffU, gate_value, 0);
    beta_value = __shfl_sync(0xffffffffU, beta_value, 0);

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float kv_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      kv_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      const int64_t qk_index =
          ((static_cast<int64_t>(t) * q_heads + hq) * K) + row;
      const float q_value = load_as_float(q, qk_index);
      const float k_value = load_as_float(k, qk_index);
      q_reg[r] = q_value;
      k_reg[r] = k_value;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        kv_partial[c] += state_shard[c][r] * k_value;
      }
    }

    float delta[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
      const float kv_col = subgroup_sum(kv_partial[c], WIDTH);
      float delta_value = 0.0F;
      if (lane == 0 && col < V) {
        const int64_t v_index =
            ((static_cast<int64_t>(t) * v_heads + hv) * V) + col;
        delta_value =
            (load_as_float(v, v_index) - gate_value * kv_col) * beta_value;
      }
      delta[c] = subgroup_broadcast(delta_value, WIDTH);
    }

    float attn_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const float new_state =
            fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
        state_shard[c][r] = new_state;
        attn_partial[c] += new_state * q_reg[r];
      }
    }

#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = subgroup_sum(attn_partial[c], WIDTH);
    }

    if (lane == 0) {
      const int64_t out_base = (static_cast<int64_t>(t) * v_heads + hv) * V;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
        if (col < V) {
          store_from_float(output, out_base + col, attn_partial[c] * scale);
        }
      }
    }
  }

  if (final_state != nullptr) {
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
#pragma unroll
      for (int r = 0; r < rows_per_lane; ++r) {
        const int row = r * WIDTH + lane;
        if (col < V) {
          const int64_t state_index =
              (((static_cast<int64_t>(n) * v_heads + hv) * V + col) * K) + row;
          final_state[state_index] = state_shard[c][r];
        }
      }
    }
  }
}

template <typename scalar_t,
          typename bias_t,
          typename state_t,
          int K,
          int V,
          int COLS,
          int WIDTH>
__global__ void gdn_decode_mixed_qkv_global_state_kernel(
    const scalar_t* __restrict__ mixed_qkv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ A_log,
    const bias_t* __restrict__ dt_bias,
    state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    scalar_t* __restrict__ output,
    int tokens,
    int slots,
    int64_t state_slot_stride,
    int q_heads,
    int v_heads,
    int qkv_stride,
    float scale,
    bool use_qk_l2norm) {
  static_assert(K % WIDTH == 0);
  constexpr int subgroups_per_warp = 32 / WIDTH;
  constexpr int rows_per_lane = K / WIDTH;

  const int hv = blockIdx.x;
  const int t = blockIdx.y;
  const int subgroup = threadIdx.x / WIDTH;
  const int lane = threadIdx.x % WIDTH;
  const int group_base =
      (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
  const int col_base = group_base * COLS;
  const int hq = hv / (v_heads / q_heads);
  const int32_t slot = state_indices[t];

  if (slot < 0 || slot >= slots) {
    if (lane == 0) {
      const int64_t out_base = (static_cast<int64_t>(t) * v_heads + hv) * V;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
        if (col < V) {
          store_from_float(output, out_base + col, 0.0F);
        }
      }
    }
    return;
  }

  float state_shard[COLS][rows_per_lane];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float value = 0.0F;
      if (col < V) {
        const int64_t state_index =
            static_cast<int64_t>(slot) * state_slot_stride +
            ((static_cast<int64_t>(hv) * V + col) * K) + row;
        value = load_as_float(state, state_index);
      }
      state_shard[c][r] = value;
    }
  }

  const int64_t mixed_base = static_cast<int64_t>(t) * qkv_stride;
  float k_reg[rows_per_lane];
  float q_reg[rows_per_lane];
  float q_norm = 0.0F;
  float k_norm = 0.0F;
#pragma unroll
  for (int r = 0; r < rows_per_lane; ++r) {
    const int row = r * WIDTH + lane;
    const int64_t q_index = mixed_base + hq * K + row;
    const int64_t k_index = mixed_base + q_heads * K + hq * K + row;
    const float q_value = load_as_float(mixed_qkv, q_index);
    const float k_value = load_as_float(mixed_qkv, k_index);
    q_reg[r] = q_value;
    k_reg[r] = k_value;
    q_norm += q_value * q_value;
    k_norm += k_value * k_value;
  }

  if (use_qk_l2norm) {
    q_norm = subgroup_broadcast(subgroup_sum(q_norm, WIDTH), WIDTH);
    k_norm = subgroup_broadcast(subgroup_sum(k_norm, WIDTH), WIDTH);
    const float q_inv = rsqrtf(q_norm + 1.0e-6F);
    const float k_inv = rsqrtf(k_norm + 1.0e-6F);
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      q_reg[r] *= q_inv;
      k_reg[r] *= k_inv;
    }
  }

  const int64_t gate_index = static_cast<int64_t>(t) * v_heads + hv;
  float gate_value = 0.0F;
  float beta_value = 0.0F;
  if (threadIdx.x == 0) {
    const float x = load_as_float(a, gate_index) + load_as_float(dt_bias, hv);
    const float softplus_x =
        x <= 20.0F ? log1pf(__expf(x)) : x;
    const float g_value = -__expf(A_log[hv]) * softplus_x;
    gate_value = __expf(g_value);
    const float b_value = load_as_float(b, gate_index);
    beta_value = 1.0F / (1.0F + __expf(-b_value));
  }
  gate_value = __shfl_sync(0xffffffffU, gate_value, 0);
  beta_value = __shfl_sync(0xffffffffU, beta_value, 0);

  float kv_partial[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    kv_partial[c] = 0.0F;
  }
#pragma unroll
  for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      kv_partial[c] += state_shard[c][r] * k_reg[r];
    }
  }

  float delta[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
    const float kv_col = subgroup_sum(kv_partial[c], WIDTH);
    float delta_value = 0.0F;
    if (lane == 0 && col < V) {
      const int64_t v_index =
          mixed_base + 2 * q_heads * K + hv * V + col;
      delta_value =
          (load_as_float(mixed_qkv, v_index) - gate_value * kv_col) * beta_value;
    }
    delta[c] = subgroup_broadcast(delta_value, WIDTH);
  }

  float attn_partial[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    attn_partial[c] = 0.0F;
  }
#pragma unroll
  for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const float new_state =
          fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
      state_shard[c][r] = new_state;
      attn_partial[c] += new_state * q_reg[r];
    }
  }
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    attn_partial[c] = subgroup_sum(attn_partial[c], WIDTH);
  }

  if (lane == 0) {
    const int64_t out_base = (static_cast<int64_t>(t) * v_heads + hv) * V;
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
      if (col < V) {
        store_from_float(output, out_base + col, attn_partial[c] * scale);
      }
    }
  }

#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      if (col < V) {
        const int64_t state_index =
            static_cast<int64_t>(slot) * state_slot_stride +
            ((static_cast<int64_t>(hv) * V + col) * K) + row;
        store_from_float(state, state_index, state_shard[c][r]);
      }
    }
  }
}

template <typename scalar_t,
          typename bias_t,
          typename state_t,
          int K,
          int V,
          int COLS,
          int WIDTH>
__global__ void gdn_decode_mixed_qkv_ddtree_state_kernel(
    const scalar_t* __restrict__ mixed_qkv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ A_log,
    const bias_t* __restrict__ dt_bias,
    state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ parent_ids,
    const int32_t* __restrict__ num_accepted_tokens,
    const int32_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ output,
    int num_sequences,
    int max_state_tokens,
    int tokens,
    int slots,
    int64_t state_slot_stride,
    int q_heads,
    int v_heads,
    int qkv_stride,
    float scale,
    bool use_qk_l2norm) {
  static_assert(K % WIDTH == 0);
  constexpr int subgroups_per_warp = 32 / WIDTH;
  constexpr int rows_per_lane = K / WIDTH;

  const int hv = blockIdx.x;
  const int n = blockIdx.y;
  const int subgroup = threadIdx.x / WIDTH;
  const int lane = threadIdx.x % WIDTH;
  const int group_base =
      (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
  const int col_base = group_base * COLS;
  const int hq = hv / (v_heads / q_heads);
  if (n >= num_sequences) {
    return;
  }

  const int seq_start = cu_seqlens[n];
  const int seq_end = cu_seqlens[n + 1];
  const int seq_tokens = seq_end - seq_start;
  if (seq_tokens <= 0 || seq_start < 0 || seq_end > tokens) {
    return;
  }

  int selector = num_accepted_tokens[n] - 1;
  selector = selector < 0 ? 0 : selector;
  selector = selector >= max_state_tokens ? max_state_tokens - 1 : selector;
  int32_t state_idx = state_indices[n * max_state_tokens + selector];

  float state_shard[COLS][rows_per_lane];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float value = 0.0F;
      if (col < V && state_idx >= 0 && state_idx < slots) {
        const int64_t state_index =
            static_cast<int64_t>(state_idx) * state_slot_stride +
            ((static_cast<int64_t>(hv) * V + col) * K) + row;
        value = load_as_float(state, state_index);
      }
      state_shard[c][r] = value;
    }
  }

  for (int local_t = 0; local_t < seq_tokens && local_t < max_state_tokens;
       ++local_t) {
    if (local_t > 0) {
      int parent_t = parent_ids[n * max_state_tokens + local_t];
      parent_t = parent_t < 0 ? 0 : parent_t;
      parent_t =
          parent_t >= max_state_tokens ? max_state_tokens - 1 : parent_t;
      const int32_t parent_state_idx =
          state_indices[n * max_state_tokens + parent_t];
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
          const int row = r * WIDTH + lane;
          float value = 0.0F;
          if (col < V && parent_state_idx >= 0 && parent_state_idx < slots) {
            const int64_t state_index =
                static_cast<int64_t>(parent_state_idx) * state_slot_stride +
                ((static_cast<int64_t>(hv) * V + col) * K) + row;
            value = load_as_float(state, state_index);
          }
          state_shard[c][r] = value;
        }
      }
    }

    const int t = seq_start + local_t;
    const int64_t mixed_base = static_cast<int64_t>(t) * qkv_stride;
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float q_norm = 0.0F;
    float k_norm = 0.0F;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      const int64_t q_index = mixed_base + hq * K + row;
      const int64_t k_index = mixed_base + q_heads * K + hq * K + row;
      const float q_value = load_as_float(mixed_qkv, q_index);
      const float k_value = load_as_float(mixed_qkv, k_index);
      q_reg[r] = q_value;
      k_reg[r] = k_value;
      q_norm += q_value * q_value;
      k_norm += k_value * k_value;
    }

    if (use_qk_l2norm) {
      q_norm = subgroup_broadcast(subgroup_sum(q_norm, WIDTH), WIDTH);
      k_norm = subgroup_broadcast(subgroup_sum(k_norm, WIDTH), WIDTH);
      const float q_inv = rsqrtf(q_norm + 1.0e-6F);
      const float k_inv = rsqrtf(k_norm + 1.0e-6F);
#pragma unroll
      for (int r = 0; r < rows_per_lane; ++r) {
        q_reg[r] *= q_inv;
        k_reg[r] *= k_inv;
      }
    }

    const int64_t gate_index = static_cast<int64_t>(t) * v_heads + hv;
    float gate_value = 0.0F;
    float beta_value = 0.0F;
    if (threadIdx.x == 0) {
      const float x = load_as_float(a, gate_index) + load_as_float(dt_bias, hv);
      const float softplus_x = x <= 20.0F ? log1pf(__expf(x)) : x;
      const float g_value = -__expf(A_log[hv]) * softplus_x;
      gate_value = __expf(g_value);
      const float b_value = load_as_float(b, gate_index);
      beta_value = 1.0F / (1.0F + __expf(-b_value));
    }
    gate_value = __shfl_sync(0xffffffffU, gate_value, 0);
    beta_value = __shfl_sync(0xffffffffU, beta_value, 0);

    float kv_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      kv_partial[c] = 0.0F;
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        kv_partial[c] += state_shard[c][r] * k_reg[r];
      }
    }

    float delta[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
      const float kv_col = subgroup_sum(kv_partial[c], WIDTH);
      float delta_value = 0.0F;
      if (lane == 0 && col < V) {
        const int64_t v_index = mixed_base + 2 * q_heads * K + hv * V + col;
        delta_value =
            (load_as_float(mixed_qkv, v_index) - gate_value * kv_col) *
            beta_value;
      }
      delta[c] = subgroup_broadcast(delta_value, WIDTH);
    }

    float attn_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = 0.0F;
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const float new_state =
            fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
        state_shard[c][r] = new_state;
        attn_partial[c] += new_state * q_reg[r];
      }
    }
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = subgroup_sum(attn_partial[c], WIDTH);
    }

    if (lane == 0) {
      const int64_t out_base = (static_cast<int64_t>(t) * v_heads + hv) * V;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
        if (col < V) {
          store_from_float(output, out_base + col, attn_partial[c] * scale);
        }
      }
    }

    const int32_t dst_state_idx =
        state_indices[n * max_state_tokens + local_t];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
#pragma unroll
      for (int r = 0; r < rows_per_lane; ++r) {
        const int row = r * WIDTH + lane;
        if (col < V && dst_state_idx >= 0 && dst_state_idx < slots) {
          const int64_t state_index =
              static_cast<int64_t>(dst_state_idx) * state_slot_stride +
              ((static_cast<int64_t>(hv) * V + col) * K) + row;
          store_from_float(state, state_index, state_shard[c][r]);
        }
      }
    }
  }
}

template <typename scalar_t, typename gate_t, typename beta_t, typename state_t, int K, int V>
void launch_gdn_forward_typed(const scalar_t* q,
                              const scalar_t* k,
                              const scalar_t* v,
                              const gate_t* gate,
                              const beta_t* beta,
                              const state_t* initial_state,
                              scalar_t* output,
                              float* final_state,
                              int batch,
                              int tokens,
                              int q_heads,
                              int v_heads,
                              float scale,
                              bool gate_is_exp,
                              int column_groups_per_block,
                              cudaStream_t stream) {
  constexpr int cols = V == 128 ? 4 : 1;
  constexpr int width = K == 128 ? 16 : 32;
  constexpr int groups_per_warp = 32 / width;
  const dim3 block(32, column_groups_per_block);
  const int groups = (V + cols - 1) / cols;
  const int z = (groups + column_groups_per_block * groups_per_warp - 1) /
                (column_groups_per_block * groups_per_warp);
  const dim3 grid(v_heads, batch, z);
  if (gate_is_exp) {
    gdn_forward_kernel<scalar_t, gate_t, beta_t, state_t, K, V, cols, width, true>
        <<<grid, block, 0, stream>>>(q,
                                     k,
                                     v,
                                     gate,
                                     beta,
                                     initial_state,
                                     output,
                                     final_state,
                                     batch,
                                     tokens,
                                     q_heads,
                                     v_heads,
                                     scale,
                                     gate_is_exp);
  } else {
    gdn_forward_kernel<scalar_t, gate_t, beta_t, state_t, K, V, cols, width, false>
        <<<grid, block, 0, stream>>>(q,
                                     k,
                                     v,
                                     gate,
                                     beta,
                                     initial_state,
                                     output,
                                     final_state,
                                     batch,
                                     tokens,
                                     q_heads,
                                     v_heads,
                                     scale,
                                     gate_is_exp);
  }
}

}  // namespace

namespace ninfer::third_party::onecat {

void gdn_forward_f16_launch(const void* q, const void* k, const void* v,
                            const void* gate, const void* beta,
                            const void* initial_state, void* output,
                            float* final_state, int batch, int tokens, int q_heads,
                            int v_heads, float scale, bool gate_is_exp,
                            int column_groups_per_block, cudaStream_t stream) {
  launch_gdn_forward_typed<__half, float, float, float, 128, 128>(
      static_cast<const __half*>(q), static_cast<const __half*>(k),
      static_cast<const __half*>(v), static_cast<const float*>(gate),
      static_cast<const float*>(beta), static_cast<const float*>(initial_state),
      static_cast<__half*>(output), final_state, batch, tokens, q_heads, v_heads,
      scale, gate_is_exp, column_groups_per_block, stream);
}

}  // namespace ninfer::third_party::onecat
