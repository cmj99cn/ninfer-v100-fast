// FP16 V100 paged flash-attention kernel adapted from 1CatAI/1Cat-vLLM v1.5.1
// `flash-attention-v100/kernel/fused_mha_forward_paged.cu` (Apache-2.0).
// Only the raw-pointer device kernel is retained; the host launcher below is NInfer-specific.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>


#include <mma.h>
using namespace nvcuda::wmma;

#include "flash_v100_traits.cuh"
#include "fp8_kv_utils.cuh"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define MAX_THREADS_PER_WARP 32
#define MAX_THREADS_PER_SM 2048
#define MAX_THREAD_BLOCK_SIZE 1024
#define MAX_THREAD_BLOCK_PER_SM 32
#define MAX_WARPS_PER_SM 64
#define MAX_SM_PER_GPU 80
#define MAX_SMEM_PER_SM 98304

#define WARP_ALLOC_GROUP 4

namespace {

int kv_cache_dtype_code_from_string(const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto" || kv_cache_dtype == "bfloat16") {
    return flash_v100::KV_CACHE_DTYPE_FP16;
  }
  if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    return flash_v100::KV_CACHE_DTYPE_FP8_E4M3;
  }
  if (kv_cache_dtype == "fp8_e5m2") {
    return flash_v100::KV_CACHE_DTYPE_FP8_E5M2;
  }
  return -1;
}

}  // namespace

#define BLOCK_M_16 16
#define BLOCK_N_16 512
#define WARPS_16 16

#define BLOCK_M_32 32
#define BLOCK_N_32 256
#define WARPS_32 16

#define BLOCK_M_64 64
#define BLOCK_N_64 128
#define WARPS_64 16

#define BLOCK_M_128 32
#define BLOCK_N_128 176
#define WARPS_128 16

#define BLOCK_M_256 32
#define BLOCK_N_256 64
#define WARPS_256 16

#define BLOCK_M_256_LOW_SMEM 16
#define BLOCK_N_256_LOW_SMEM 128
#define KV_STAGE_N_256_LOW_SMEM 1
#define WARPS_256_LOW_SMEM 16
#define BLOCK_N_256_LOW_SMEM_SCALAR_QK 32
#define KV_STAGE_N_256_LOW_SMEM_SCALAR_QK BLOCK_N_256_LOW_SMEM_SCALAR_QK
#define LOW_SMEM_PAGE_SIZE 16

template <int D, bool LOW_SMEM = false, bool LOW_SMEM_SCALAR_QK = false,
          bool LOW_SMEM_BM32 = false, bool D256_OUTPUT_STRIDE_268 = false,
          bool D256_SW_PIPELINE_QK = false, bool D256_SW_PIPELINE_PV = false>
struct KernelConfig {
  static constexpr int BLOCK_M =
      (D == 16)   ? BLOCK_M_16
      : (D == 32) ? BLOCK_M_32
      : (D == 64) ? BLOCK_M_64
      : (D == 128)
          ? BLOCK_M_128
          : (LOW_SMEM ? (LOW_SMEM_BM32 ? BLOCK_M_256 : BLOCK_M_256_LOW_SMEM)
                      : BLOCK_M_256);
  static constexpr int BLOCK_N =
      (D == 16)   ? BLOCK_N_16
      : (D == 32) ? BLOCK_N_32
      : (D == 64) ? BLOCK_N_64
      : (D == 128)
          ? BLOCK_N_128
          : (LOW_SMEM ? (LOW_SMEM_SCALAR_QK ? BLOCK_N_256_LOW_SMEM_SCALAR_QK
                                            : BLOCK_N_256_LOW_SMEM)
                      : BLOCK_N_256);
  static constexpr int WARPS_PER_BLOCK =
      (D == 16)    ? WARPS_16
      : (D == 32)  ? WARPS_32
      : (D == 64)  ? WARPS_64
      : (D == 128) ? WARPS_128
                   : (LOW_SMEM ? WARPS_256_LOW_SMEM : WARPS_256);

  static constexpr int THREADS_PER_BLOCK =
      WARPS_PER_BLOCK * MAX_THREADS_PER_WARP;
  static constexpr int THREADS_PER_ROW = THREADS_PER_BLOCK / BLOCK_M;
  static constexpr int PAD = (8 - (D % 32) + 32) % 32;
  static constexpr int Q_STRIDE = D + PAD;
  static constexpr int KV_STRIDE = D + PAD;
  static constexpr int KV_STAGE_N =
      (D == 256 && LOW_SMEM)
          ? (LOW_SMEM_SCALAR_QK ? KV_STAGE_N_256_LOW_SMEM_SCALAR_QK
                                : KV_STAGE_N_256_LOW_SMEM)
          : BLOCK_N;
  static constexpr int S_STRIDE = BLOCK_N + PAD;
  static constexpr int P_SUB_TILE = 32;
  static constexpr int P_STRIDE = P_SUB_TILE + PAD;
  static constexpr int P_STRICT_ELEMENTS = (D == 256) ? BLOCK_M * P_STRIDE : 1;
  static constexpr int PIPELINE_S_ELEMENTS =
      (D == 256 && LOW_SMEM && D256_SW_PIPELINE_QK && D256_SW_PIPELINE_PV)
          ? BLOCK_M * S_STRIDE
          : 1;
  static constexpr int PIPELINE_PAGE_ELEMENTS =
      (D == 256 && LOW_SMEM && D256_SW_PIPELINE_QK && D256_SW_PIPELINE_PV)
          ? BLOCK_N / LOW_SMEM_PAGE_SIZE
          : 1;
  static constexpr bool PIPELINE_SWIZZLED_Q =
      D == 256 && LOW_SMEM && D256_SW_PIPELINE_QK && D256_SW_PIPELINE_PV;
  static constexpr int Q_ELEMENTS =
      PIPELINE_SWIZZLED_Q ? BLOCK_M * D : BLOCK_M * Q_STRIDE;
  // The D256 low-SMEM candidate changes only the output accumulator pitch
  // from 264 to 268 floats. This preserves all WMMA/softmax arithmetic and
  // targets the largest 8-way shared-memory load/store groups.
  static constexpr int OUTPUT_EXTRA_PAD =
      (D == 256 && LOW_SMEM && D256_OUTPUT_STRIDE_268) ? 4 : 0;
  static constexpr int O_STRIDE = D + PAD + OUTPUT_EXTRA_PAD;
  static constexpr int PER_UINT4 = 8;
  static constexpr int LOW_SMEM_PAGE_COUNT =
      (D == 256 && LOW_SMEM) ? (BLOCK_N / LOW_SMEM_PAGE_SIZE) : 1;
  struct alignas(128) SmemLayout {
    alignas(16) __half q[Q_ELEMENTS];
    union {
      alignas(16) __half k[KV_STAGE_N * KV_STRIDE];
      alignas(16) __half v[KV_STAGE_N * KV_STRIDE];
    } reuse_kv;
    union {
      alignas(16) float s[BLOCK_M * S_STRIDE];
      alignas(16) __half p[BLOCK_M * S_STRIDE];
    } reuse_sp;
    alignas(16) __half p_strict[P_STRICT_ELEMENTS];
    alignas(16) float pipeline_s[PIPELINE_S_ELEMENTS];
    alignas(16) int pipeline_page_idx[PIPELINE_PAGE_ELEMENTS];
    alignas(16) int pipeline_page_offset[PIPELINE_PAGE_ELEMENTS];
    // WMMA accumulator loads and stores require 256-bit alignment.
    alignas(32) float o[BLOCK_M * O_STRIDE];
    alignas(16) float row_max[BLOCK_M];
    alignas(16) float row_sum[BLOCK_M];
    alignas(16) int page_idx[LOW_SMEM_PAGE_COUNT];
    alignas(16) int page_offset[LOW_SMEM_PAGE_COUNT];
  };

  static constexpr size_t TOTAL_SMEM =
      ((sizeof(SmemLayout) + 127) & ~size_t(127));
  static_assert(offsetof(SmemLayout, o) % 32 == 0,
                "WMMA accumulator storage must be 256-bit aligned");
};

template <typename Config>
__device__ __forceinline__ void init_smem(char* smem_raw) {
  constexpr int N_U4 = Config::TOTAL_SMEM / 16;
  constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
  const int tid = threadIdx.x;

  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
#pragma unroll 1
  for (int i = tid; i < N_U4; i += THREADS_PER_BLOCK) {
    asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(addr + (i << 4)),
                 "r"(0)
                 : "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ int pipeline_swizzled_q_row_slot(int row) {
  return (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);
}

__device__ __forceinline__ int pipeline_swizzled_matrix_a_offset(int row,
                                                                 int col) {
  const int tile = col / WMMA_K;
  const int within_tile = col - tile * WMMA_K;
  const int plane = within_tile >> 3;
  const int inner = within_tile & 7;
  return tile * WMMA_M * WMMA_K + plane * WMMA_M * 8 +
         pipeline_swizzled_q_row_slot(row) * 8 + inner;
}

__device__ __forceinline__ void load_pipeline_swizzled_matrix_a_fragment(
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major>& frag,
    const __half* __restrict__ matrix, int k_offset) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 3) + ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 8;
  const int slot = pipeline_swizzled_q_row_slot(row);
  const int tile_offset = (k_offset / WMMA_K) * WMMA_M * WMMA_K;
  uint32_t address = static_cast<uint32_t>(
      __cvta_generic_to_shared(matrix + tile_offset + slot * 8));
  uint32_t* words = reinterpret_cast<uint32_t*>(&frag);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3])
               : "r"(address)
               : "memory");
  address += WMMA_M * 8 * sizeof(__half);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[4]), "=r"(words[5]), "=r"(words[6]), "=r"(words[7])
               : "r"(address)
               : "memory");
}

template <int Q_STRIDE, int S_STRIDE, int K_TILES, bool IS_CAUSAL,
          bool SWIZZLED_Q = false, bool ANCHORED_SWA = false>
__device__ __forceinline__ void pipeline_qk_slice(
    const __half* __restrict__ sQ, float* __restrict__ score,
    const __half* __restrict__ k_cache, const int* __restrict__ page_idx,
    const int* __restrict__ page_offset, int64_t k_block_stride,
    int64_t k_token_stride, int64_t k_head_stride, int kv_head_id,
    int start_col, int valid_k_rows, int start_row, int valid_q_rows,
    int causal_q_offset, int tile_n, int k_begin, bool load_partial,
    bool finalize, float softmax_scale, int window_size_left,
    int window_size_right, float neg_inf, int anchor_len = 0,
    int anchored_window = 0) {
  if (tile_n >= valid_k_rows) {
    return;
  }

  const int page_slot = tile_n >> 4;
  const int block_offset = page_offset[page_slot];
  const int physical_block_idx = page_idx[page_slot];
  const __half* k_tile_base = k_cache +
                              (int64_t)physical_block_idx * k_block_stride +
                              (int64_t)block_offset * k_token_stride +
                              (int64_t)kv_head_id * k_head_stride;

  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  if (load_partial) {
    load_matrix_sync(acc_frag, score + tile_n, S_STRIDE, mem_row_major);
  } else {
    fill_fragment(acc_frag, 0.0f);
  }

#pragma unroll
  for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
    const int k_offset = k_begin + k_tile * WMMA_K;
    if constexpr (SWIZZLED_Q) {
      load_pipeline_swizzled_matrix_a_fragment(a_frag, sQ, k_offset);
    } else {
      load_matrix_sync(a_frag, sQ + k_offset, Q_STRIDE);
    }
    load_matrix_sync(b_frag, k_tile_base + k_offset, k_token_stride);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  if (finalize) {
    const int lane_id = threadIdx.x & 31;
    const unsigned row_causal = (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8 +
                                ((lane_id >> 4) & 0b1) * 4;
    const unsigned col_causal =
        ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

#pragma unroll
    for (int i = 0; i < acc_frag.num_elements; ++i) {
      const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
      const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;
      const int global_m = start_row + row;
      const int global_n = start_col + tile_n + col;
      const int global_q_pos = global_m + causal_q_offset;

      const bool is_valid = global_m < start_row + valid_q_rows &&
                            global_n < start_col + valid_k_rows;
      bool is_causal_valid = true;
      if constexpr (IS_CAUSAL) {
        is_causal_valid = global_n <= global_q_pos;
      }
      bool is_window_valid = true;
      if (window_size_left >= 0) {
        is_window_valid = global_n >= global_q_pos - window_size_left;
      }
      if (window_size_right >= 0) {
        is_window_valid =
            is_window_valid && global_n <= global_q_pos + window_size_right;
      }
      bool is_anchor_valid = true;
      if constexpr (ANCHORED_SWA) {
        is_anchor_valid = (global_n < anchor_len) ||
                          (global_q_pos - global_n < anchored_window);
      }
      acc_frag.x[i] =
          (is_valid && is_causal_valid && is_window_valid && is_anchor_valid)
              ? acc_frag.x[i] * softmax_scale
              : neg_inf;
    }
  }

  store_matrix_sync(score + tile_n, acc_frag, S_STRIDE, mem_row_major);
}

template <int P_STRIDE, int O_STRIDE, bool SWIZZLED_P = false>
__device__ __forceinline__ void pipeline_pv_tile(
    const __half* __restrict__ sP, float* __restrict__ sO,
    const __half* __restrict__ v_cache, const int* __restrict__ page_idx,
    const int* __restrict__ page_offset, int64_t v_block_stride,
    int64_t v_token_stride, int64_t v_head_stride, int kv_head_id,
    int sub_start, int sub_valid_k_rows, int tile_d) {
  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  load_matrix_sync(acc_frag, sO + tile_d, O_STRIDE, mem_row_major);

#pragma unroll
  for (int tile_k = 0; tile_k < 2; ++tile_k) {
    const int k_offset = tile_k * WMMA_K;
    if (k_offset >= sub_valid_k_rows) {
      break;
    }
    const int token_offset = sub_start + k_offset;
    const int page_slot = token_offset >> 4;
    const int block_offset = page_offset[page_slot] + (token_offset & 15);
    const int physical_block_idx = page_idx[page_slot];
    const __half* v_tile_ptr = v_cache +
                               (int64_t)physical_block_idx * v_block_stride +
                               (int64_t)block_offset * v_token_stride +
                               (int64_t)kv_head_id * v_head_stride + tile_d;
    if constexpr (SWIZZLED_P) {
      load_pipeline_swizzled_matrix_a_fragment(a_frag, sP, k_offset);
    } else {
      load_matrix_sync(a_frag, sP + k_offset, P_STRIDE);
    }
    load_matrix_sync(b_frag, v_tile_ptr, v_token_stride);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }
  store_matrix_sync(sO + tile_d, acc_frag, O_STRIDE, mem_row_major);
}

template <int D, bool LOW_SMEM, bool LOW_SMEM_CONTIG_FAST,
          bool LOW_SMEM_SCALAR_QK, bool LOW_SMEM_BM32, bool SPLIT_KV,
          bool IS_CAUSAL, int KV_DTYPE, bool D256_OUTPUT_STRIDE_268 = false,
          bool D256_SW_PIPELINE_QK = false, bool D256_SW_PIPELINE_PV = false,
          bool ANCHORED_SWA = false>
__global__ void __launch_bounds__(
    KernelConfig<D, LOW_SMEM, LOW_SMEM_SCALAR_QK, LOW_SMEM_BM32,
                 D256_OUTPUT_STRIDE_268, D256_SW_PIPELINE_QK,
                 D256_SW_PIPELINE_PV>::THREADS_PER_BLOCK,
    2)
    flash_attention_forward_kernel_paged(
        const __half* __restrict__ Q, const void* __restrict__ K_cache,
        const void* __restrict__ V_cache, __half* __restrict__ Out,
        float* __restrict__ softmax_lse, const int* __restrict__ block_table,
        const int* __restrict__ seqused_k, const int B, const int H,
        const int M, const int N, const int* __restrict__ bfla_block_mask,
        const int bfla_mask_block_n, const int64_t bfla_mask_stride_b,
        const int64_t bfla_mask_stride_h, const int64_t bfla_mask_stride_q,
        const int64_t bfla_mask_stride_k, const int page_block_size,
        const int num_kv_heads, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const int64_t v_block_stride, const int64_t v_token_stride,
        const int64_t v_head_stride, const float softmax_scale,
        const float k_scale, const float v_scale, const int window_size_left,
        const int window_size_right, float* __restrict__ split_tmp_out,
        float* __restrict__ split_tmp_row_max,
        float* __restrict__ split_tmp_row_sum, const int split_kv_tiles,
        const int* __restrict__ anchor_lens, const int anchored_window) {
  static_assert(!ANCHORED_SWA ||
                    (IS_CAUSAL && KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16),
                "ANCHORED_SWA requires causal attention and an fp16 KV cache");
  using Config = KernelConfig<D, LOW_SMEM, LOW_SMEM_SCALAR_QK, LOW_SMEM_BM32,
                              D256_OUTPUT_STRIDE_268, D256_SW_PIPELINE_QK,
                              D256_SW_PIPELINE_PV>;
  using Traits = FlashV100Traits<D>;

  constexpr int BLOCK_M = Config::BLOCK_M;
  constexpr int BLOCK_N = Config::BLOCK_N;
  constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
  constexpr int THREADS_PER_ROW = Config::THREADS_PER_ROW;
  constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
  constexpr int Q_STRIDE = Config::Q_STRIDE;
  constexpr int KV_STRIDE = Config::KV_STRIDE;
  constexpr int S_STRIDE = Config::S_STRIDE;
  constexpr int P_SUB_TILE = Config::P_SUB_TILE;
  constexpr int P_STRIDE = Config::P_STRIDE;
  constexpr int O_STRIDE = Config::O_STRIDE;
  constexpr int PER_UINT4 = Config::PER_UINT4;
  constexpr bool USE_SW_PIPELINE = D256_SW_PIPELINE_QK || D256_SW_PIPELINE_PV;
  if constexpr (USE_SW_PIPELINE) {
    static_assert(D == 256 && LOW_SMEM && !LOW_SMEM_CONTIG_FAST &&
                      !LOW_SMEM_SCALAR_QK && !LOW_SMEM_BM32 && !SPLIT_KV &&
                      BLOCK_M == 16 && BLOCK_N == 128 && WARPS_PER_BLOCK == 16,
                  "software pipeline is specialized for D256 BM16 BN128");
  }
  const float NEG_INF = -1e30f;

  const int batch_head_id = blockIdx.z;
  if (batch_head_id >= B * H) return;

  const int batch_id = batch_head_id / H;
  const int q_head_id = batch_head_id % H;
  const int kv_group_size = H / num_kv_heads;
  const int kv_head_id = q_head_id / kv_group_size;

  const int block_m = blockIdx.x;
  const int start_row = block_m * BLOCK_M;
  if (start_row >= M) return;

  const int actual_N = seqused_k[batch_id];
  int num_n_tiles = (actual_N + BLOCK_N - 1) / BLOCK_N;
  const int valid_q_rows = min(BLOCK_M, M - start_row);
  const int causal_q_offset = max(actual_N - M, 0);

  int anchor_len = 0;
  if constexpr (ANCHORED_SWA) {
    anchor_len = anchor_lens[batch_id];
  }

  int max_key_pos = actual_N - 1;
  if constexpr (IS_CAUSAL) {
    max_key_pos =
        min(max_key_pos, start_row + valid_q_rows - 1 + causal_q_offset);
  }
  if (window_size_right >= 0) {
    max_key_pos = min(max_key_pos, start_row + valid_q_rows - 1 +
                                       causal_q_offset + window_size_right);
  }
  if (max_key_pos < 0) {
    num_n_tiles = 0;
  } else {
    num_n_tiles = min(num_n_tiles, (max_key_pos + BLOCK_N) / BLOCK_N);
  }
  const int min_key_pos =
      window_size_left >= 0
          ? max(0, start_row + causal_q_offset - window_size_left)
          : 0;

  const int tid = threadIdx.x;
  const int warp_id = tid / MAX_THREADS_PER_WARP;
  const int lane_id = tid % MAX_THREADS_PER_WARP;

  const size_t q_head_linear = (size_t)batch_id * H + q_head_id;
  const __half* q_ptr = Q + q_head_linear * M * D + start_row * D;
  __half* out_ptr = Out + q_head_linear * M * D + start_row * D;
  float* softmax_lse_ptr = softmax_lse + q_head_linear * M + start_row;

  const int max_num_blocks_per_seq =
      (N + page_block_size - 1) / page_block_size;
  const int* block_table_seq = block_table + batch_id * max_num_blocks_per_seq;

  const int64_t k_row_stride = k_token_stride;
  const int64_t v_row_stride = v_token_stride;

  extern __shared__ __align__(128) char smem_raw[];
  init_smem<Config>(smem_raw);
  auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

  __half* sQ = smem.q;
  __half* sK = smem.reuse_kv.k;
  __half* sV = smem.reuse_kv.v;
  float* sS = smem.reuse_sp.s;
  __half* sP = (D == 256) ? smem.p_strict : smem.reuse_sp.p;
  const int p_stride = (D == 256) ? P_STRIDE : S_STRIDE;
  const int p_tile_capacity = (D == 256) ? P_SUB_TILE : BLOCK_N;
  float* sO = smem.o;
  float* sRowMax = smem.row_max;
  float* sRowSum = smem.row_sum;
  int* sPageIdx = smem.page_idx;
  int* sPageOffset = smem.page_offset;

  const int d_stride_uint4 = (D + PER_UINT4 - 1) / PER_UINT4;
  const int q_stride_uint4 = (Q_STRIDE + PER_UINT4 - 1) / PER_UINT4;
  const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

  if (tid < BLOCK_M) {
    sRowMax[tid] = NEG_INF;
    sRowSum[tid] = 0.0f;
  }
  constexpr int O_UINT4_PER_ROW = D / 4;
  constexpr int O_UINT4_STRIDE = O_STRIDE / 4;
  for (int idx = tid; idx < BLOCK_M * O_UINT4_PER_ROW;
       idx += THREADS_PER_BLOCK) {
    const int row = idx / O_UINT4_PER_ROW;
    const int col = idx % O_UINT4_PER_ROW;
    reinterpret_cast<uint4*>(sO)[row * O_UINT4_STRIDE + col] =
        make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);

#pragma unroll 4
  for (int idx = tid; idx < (valid_q_rows * d_stride_uint4);
       idx += THREADS_PER_BLOCK) {
    const int row = idx / d_stride_uint4;
    const int vec_col = idx % d_stride_uint4;
    uint4 q_val = make_uint4(0, 0, 0, 0);
    if (row < valid_q_rows && vec_col < d_stride_uint4) {
      q_val = __ldg(&q_vec[row * d_stride_uint4 + vec_col]);
    }
    if constexpr (Config::PIPELINE_SWIZZLED_Q) {
      const int k_tile = vec_col >> 1;
      const int plane = vec_col & 1;
      const int slot = pipeline_swizzled_q_row_slot(row);
      sQ_vec[k_tile * (2 * BLOCK_M) + plane * BLOCK_M + slot] = q_val;
    } else {
      sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
    }
  }
  __syncthreads();

  int cross_block_first_n_tile = 0;
  if constexpr (D256_SW_PIPELINE_QK && D256_SW_PIPELINE_PV) {
    const bool use_cross_block_pipeline =
        valid_q_rows == BLOCK_M && actual_N > 0 && page_block_size == 784 &&
        bfla_block_mask == nullptr && window_size_left < 0 &&
        window_size_right < 0;
    if (use_cross_block_pipeline) {
      const __half* k_cache_h = reinterpret_cast<const __half*>(K_cache);
      const __half* v_cache_h = reinterpret_cast<const __half*>(V_cache);

      if (tid < Config::LOW_SMEM_PAGE_COUNT) {
        const int global_token_idx = tid * LOW_SMEM_PAGE_SIZE;
        const int virtual_block_idx = global_token_idx / page_block_size;
        sPageIdx[tid] = __ldg(&block_table_seq[virtual_block_idx]);
        sPageOffset[tid] =
            global_token_idx - virtual_block_idx * page_block_size;
      }
      __syncthreads();

      if (warp_id < BLOCK_N / WMMA_N) {
        const int valid_k_rows = min(BLOCK_N, actual_N);
        pipeline_qk_slice<Q_STRIDE, S_STRIDE, D / WMMA_K, IS_CAUSAL, true,
                          ANCHORED_SWA>(
            sQ, sS, k_cache_h, sPageIdx, sPageOffset, k_block_stride,
            k_token_stride, k_head_stride, kv_head_id, 0, valid_k_rows,
            start_row, valid_q_rows, causal_q_offset, warp_id * WMMA_N, 0,
            false, true, softmax_scale, window_size_left, window_size_right,
            NEG_INF, anchor_len, anchored_window);
      }
      __syncthreads();

      // Keep the hot loop on a compile-time steady-state path. The
      // existing exact path below drains the final block, avoiding a
      // loop-carried has_next predicate in every softmax/PV panel.
      const int steady_n_tiles = num_n_tiles - 1;
      for (int block_n = 0; block_n < steady_n_tiles; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        const int valid_k_rows = BLOCK_N;
        const int next_start_col = start_col + BLOCK_N;
        const int next_valid_k_rows = min(BLOCK_N, actual_N - next_start_col);
        const bool odd_block = (block_n & 1) != 0;
        float* const score_current = odd_block ? smem.pipeline_s : sS;
        float* const score_next = odd_block ? sS : smem.pipeline_s;
        int* const page_idx_current =
            odd_block ? smem.pipeline_page_idx : sPageIdx;
        int* const page_offset_current =
            odd_block ? smem.pipeline_page_offset : sPageOffset;
        int* const page_idx_next =
            odd_block ? sPageIdx : smem.pipeline_page_idx;
        int* const page_offset_next =
            odd_block ? sPageOffset : smem.pipeline_page_offset;

        if (tid < Config::LOW_SMEM_PAGE_COUNT) {
          const int page_token_offset = tid * LOW_SMEM_PAGE_SIZE;
          const int global_token_idx = next_start_col + page_token_offset;
          const int virtual_block_idx = global_token_idx / page_block_size;
          page_idx_next[tid] = __ldg(&block_table_seq[virtual_block_idx]);
          page_offset_next[tid] =
              global_token_idx - virtual_block_idx * page_block_size;
        }
        __syncthreads();

        for (int sub_start = 0; sub_start < valid_k_rows;
             sub_start += P_SUB_TILE) {
          const int sub_valid_k_rows =
              min(P_SUB_TILE, valid_k_rows - sub_start);
          const int row = warp_id;
          const int thread_in_row = lane_id;
          const unsigned mask = 0xFFFFFFFFU;
          float* sS_row_f = score_current + row * S_STRIDE + sub_start;

          const int vec_cols = sub_valid_k_rows >> 2;
          const int vecs_per_thread =
              (vec_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
          const int tail_start = vec_cols << 2;
          float thread_max = NEG_INF;
          float4* sS_vec4 = reinterpret_cast<float4*>(sS_row_f);

#pragma unroll 4
          for (int j = 0; j < vecs_per_thread; ++j) {
            const int vc = thread_in_row + j * THREADS_PER_ROW;
            if (vc < vec_cols) {
              const float4 v4 = sS_vec4[vc];
              thread_max = fmaxf(thread_max,
                                 fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
            }
          }
#pragma unroll
          for (int c = tail_start + thread_in_row; c < sub_valid_k_rows;
               c += THREADS_PER_ROW) {
            thread_max = fmaxf(thread_max, sS_row_f[c]);
          }
#pragma unroll
          for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
            thread_max =
                fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o));
          }

          const float row_max = __shfl_sync(mask, thread_max, 0);
          const float old_max = sRowMax[row];
          const float new_max = fmaxf(old_max, row_max);
          const float exp_diff = __expf(old_max - new_max);
          float thread_sum = 0.0f;
          int vc_base = thread_in_row;

#pragma unroll 4
          for (int j = 0; j < vecs_per_thread;
               ++j, vc_base += THREADS_PER_ROW) {
            if (vc_base < vec_cols) {
              const float4 v4 = sS_vec4[vc_base];
              const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
              const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
              const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
              const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
              thread_sum += (e0 + e1) + (e2 + e3);
              const int p_col = vc_base * 4;
              __half2* p_half2 = reinterpret_cast<__half2*>(
                  sP + pipeline_swizzled_matrix_a_offset(row, p_col));
              p_half2[0] = __float22half2_rn(make_float2(e0, e1));
              p_half2[1] = __float22half2_rn(make_float2(e2, e3));
            }
          }
#pragma unroll 4
          for (int c = tail_start + thread_in_row; c < sub_valid_k_rows;
               c += THREADS_PER_ROW) {
            const float e = __expf(fmaxf(sS_row_f[c] - new_max, -80.0f));
            thread_sum += e;
            sP[pipeline_swizzled_matrix_a_offset(row, c)] = __float2half_rn(e);
          }
#pragma unroll
          for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
            thread_sum += __shfl_down_sync(mask, thread_sum, o);
          }
          const float row_sum = __shfl_sync(mask, thread_sum, 0);
          if (thread_in_row == 0) {
            sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
            sRowMax[row] = new_max;
          }

#pragma unroll 4
          for (int c = tail_start + thread_in_row; c < P_SUB_TILE;
               c += THREADS_PER_ROW) {
            if (c >= sub_valid_k_rows) {
              sP[pipeline_swizzled_matrix_a_offset(row, c)] =
                  __float2half(0.0f);
            }
          }

          if (block_n > 0 || sub_start > 0) {
            float4* sO_vec = reinterpret_cast<float4*>(sO + row * O_STRIDE);
#pragma unroll 4
            for (int ov = thread_in_row; ov < D / 4; ov += THREADS_PER_ROW) {
              float4 v = sO_vec[ov];
              v.x *= exp_diff;
              v.y *= exp_diff;
              v.z *= exp_diff;
              v.w *= exp_diff;
              sO_vec[ov] = v;
            }
          }
          __syncthreads();

          if (sub_start == 0) {
            if (warp_id < BLOCK_N / WMMA_N) {
              pipeline_qk_slice<Q_STRIDE, S_STRIDE, D / WMMA_K, IS_CAUSAL, true,
                                ANCHORED_SWA>(
                  sQ, score_next, k_cache_h, page_idx_next, page_offset_next,
                  k_block_stride, k_token_stride, k_head_stride, kv_head_id,
                  next_start_col, next_valid_k_rows, start_row, valid_q_rows,
                  causal_q_offset, warp_id * WMMA_N, 0, false, true,
                  softmax_scale, window_size_left, window_size_right, NEG_INF,
                  anchor_len, anchored_window);
            } else {
              const int pv_warp = warp_id - BLOCK_N / WMMA_N;
              pipeline_pv_tile<P_STRIDE, O_STRIDE, true>(
                  sP, sO, v_cache_h, page_idx_current, page_offset_current,
                  v_block_stride, v_token_stride, v_head_stride, kv_head_id,
                  sub_start, sub_valid_k_rows, pv_warp * WMMA_N);
              pipeline_pv_tile<P_STRIDE, O_STRIDE, true>(
                  sP, sO, v_cache_h, page_idx_current, page_offset_current,
                  v_block_stride, v_token_stride, v_head_stride, kv_head_id,
                  sub_start, sub_valid_k_rows,
                  (pv_warp + BLOCK_N / WMMA_N) * WMMA_N);
            }
          } else {
            pipeline_pv_tile<P_STRIDE, O_STRIDE, true>(
                sP, sO, v_cache_h, page_idx_current, page_offset_current,
                v_block_stride, v_token_stride, v_head_stride, kv_head_id,
                sub_start, sub_valid_k_rows, warp_id * WMMA_N);
          }
          __syncthreads();
        }
      }
      cross_block_first_n_tile = steady_n_tiles;
    }
  }

  int first_n_tile = cross_block_first_n_tile;
  int last_n_tile = num_n_tiles;
  if constexpr (SPLIT_KV) {
    const int partition_id = blockIdx.y;
    const int tiles_per_partition = split_kv_tiles > 1 ? split_kv_tiles : 1;
    first_n_tile = partition_id * tiles_per_partition;
    last_n_tile = min(num_n_tiles, first_n_tile + tiles_per_partition);
  }
  if constexpr (!ANCHORED_SWA) {
    // DFlash2 draft attention is non-causal with a short sliding window. Jump
    // directly to the first tile that can intersect the window instead of
    // visiting every historical tile only to reject it inside the loop.
    first_n_tile = max(first_n_tile, min_key_pos / BLOCK_N);
  }

  for (int block_n = first_n_tile; block_n < last_n_tile; ++block_n) {
    const int start_col = block_n * BLOCK_N;
    if (start_col >= actual_N) break;
    const int valid_k_rows = min(BLOCK_N, actual_N - start_col);

    if (start_col + valid_k_rows <= min_key_pos) {
      continue;
    }

    if (bfla_block_mask != nullptr && bfla_mask_block_n > 0) {
      const int mask_q_idx = start_row / bfla_mask_block_n;
      const int mask_k_idx = start_col / bfla_mask_block_n;
      const int keep_tile =
          __ldg(bfla_block_mask + (int64_t)batch_id * bfla_mask_stride_b +
                (int64_t)kv_head_id * bfla_mask_stride_h +
                (int64_t)mask_q_idx * bfla_mask_stride_q +
                (int64_t)mask_k_idx * bfla_mask_stride_k);
      if (keep_tile == 0) {
        continue;
      }
    }

    const int partial_block_size =
        (block_n == num_n_tiles - 1 && actual_N % BLOCK_N != 0)
            ? (actual_N % BLOCK_N)
            : -1;

    uint4* sK_vec = reinterpret_cast<uint4*>(sK);
    const int64_t row_stride_uint4 = k_row_stride / PER_UINT4;
    const int start_page = start_col / page_block_size;
    const int page_offset = start_col % page_block_size;
    const bool single_page_tile =
        (page_offset + valid_k_rows) <= page_block_size;
    const bool two_page_tile =
        !single_page_tile &&
        (page_offset + valid_k_rows) <= (page_block_size * 2);
    const int first_page_rows =
        single_page_tile ? valid_k_rows
                         : min(valid_k_rows, page_block_size - page_offset);
    const int second_page_rows = valid_k_rows - first_page_rows;
    const int physical_block_idx0 = __ldg(&block_table_seq[start_page]);
    const int physical_block_idx1 =
        second_page_rows > 0 ? __ldg(&block_table_seq[start_page + 1]) : -1;
    const bool four_page_aligned_tile =
        D == 256 && page_block_size == 16 && page_offset == 0 &&
        BLOCK_N == page_block_size * 4 && valid_k_rows == BLOCK_N &&
        k_block_stride == (int64_t)page_block_size * k_token_stride &&
        v_block_stride == (int64_t)page_block_size * v_token_stride;
    const int physical_block_idx2 =
        four_page_aligned_tile ? __ldg(&block_table_seq[start_page + 2]) : -1;
    const int physical_block_idx3 =
        four_page_aligned_tile ? __ldg(&block_table_seq[start_page + 3]) : -1;
    const bool four_page_contiguous_tile =
        four_page_aligned_tile &&
        physical_block_idx1 == physical_block_idx0 + 1 &&
        physical_block_idx2 == physical_block_idx0 + 2 &&
        physical_block_idx3 == physical_block_idx0 + 3;
    bool low_smem_contiguous_page_tile = false;
    if constexpr (LOW_SMEM) {
      static_assert(D == 256, "low-smem paged prefill is D=256 only");
      static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16,
                    "low-smem paged prefill is fp16-KV only");
      static_assert(BLOCK_N % LOW_SMEM_PAGE_SIZE == 0,
                    "low-smem BLOCK_N must be page aligned");

      constexpr int LOW_SMEM_PAGE_COUNT = Config::LOW_SMEM_PAGE_COUNT;
      if (tid < LOW_SMEM_PAGE_COUNT) {
        const int page_token_offset = tid * LOW_SMEM_PAGE_SIZE;
        if (page_token_offset < valid_k_rows) {
          const int global_token_idx = start_col + page_token_offset;
          const int virtual_block_idx = global_token_idx / page_block_size;
          sPageIdx[tid] = __ldg(&block_table_seq[virtual_block_idx]);
          sPageOffset[tid] =
              global_token_idx - virtual_block_idx * page_block_size;
        } else {
          sPageIdx[tid] = -1;
          sPageOffset[tid] = 0;
        }
      }
      __syncthreads();

      if constexpr (LOW_SMEM_CONTIG_FAST) {
        low_smem_contiguous_page_tile =
            valid_k_rows == BLOCK_N &&
            k_block_stride == (int64_t)page_block_size * k_token_stride &&
            v_block_stride == (int64_t)page_block_size * v_token_stride &&
            sPageIdx[0] >= 0;
#pragma unroll
        for (int i = 1; i < LOW_SMEM_PAGE_COUNT; ++i) {
          const int linear_offset = sPageOffset[0] + i * LOW_SMEM_PAGE_SIZE;
          const int expected_page_delta = linear_offset / page_block_size;
          const int expected_page_offset =
              linear_offset - expected_page_delta * page_block_size;
          low_smem_contiguous_page_tile =
              low_smem_contiguous_page_tile &&
              sPageIdx[i] == sPageIdx[0] + expected_page_delta &&
              sPageOffset[i] == expected_page_offset;
        }
      } else {
        low_smem_contiguous_page_tile = false;
      }

      const __half* K_cache_h = reinterpret_cast<const __half*>(K_cache);

      if constexpr (LOW_SMEM_SCALAR_QK) {
        uint4* sK_vec = reinterpret_cast<uint4*>(sK);
        for (int idx = tid; idx < valid_k_rows * d_stride_uint4;
             idx += THREADS_PER_BLOCK) {
          const int row = idx / d_stride_uint4;
          const int vec_col = idx % d_stride_uint4;
          uint4 k_val = make_uint4(0, 0, 0, 0);
          if (row < valid_k_rows && vec_col < d_stride_uint4) {
            const int page_slot = row >> 4;
            const int block_offset =
                sPageOffset[page_slot] + (row - page_slot * LOW_SMEM_PAGE_SIZE);
            const int physical_block_idx_direct = sPageIdx[page_slot];
            const __half* k_row_ptr;
            if constexpr (LOW_SMEM_CONTIG_FAST) {
              k_row_ptr =
                  low_smem_contiguous_page_tile
                      ? K_cache_h + (int64_t)sPageIdx[0] * k_block_stride +
                            (int64_t)(sPageOffset[0] + row) * k_token_stride +
                            (int64_t)kv_head_id * k_head_stride
                      : K_cache_h +
                            (int64_t)physical_block_idx_direct *
                                k_block_stride +
                            (int64_t)block_offset * k_token_stride +
                            (int64_t)kv_head_id * k_head_stride;
            } else {
              k_row_ptr = K_cache_h +
                          (int64_t)physical_block_idx_direct * k_block_stride +
                          (int64_t)block_offset * k_token_stride +
                          (int64_t)kv_head_id * k_head_stride;
            }
            const uint4* k_row_vec = reinterpret_cast<const uint4*>(k_row_ptr);
            k_val = __ldg(&k_row_vec[vec_col]);
          }
          sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
        }
        __syncthreads();

        const int total_scores = valid_q_rows * valid_k_rows;
        for (int idx = tid; idx < total_scores; idx += THREADS_PER_BLOCK) {
          const int row = idx / valid_k_rows;
          const int col = idx - row * valid_k_rows;
          const int global_m = start_row + row;
          const int global_n = start_col + col;
          const int global_q_pos = global_m + causal_q_offset;

          bool is_valid = true;
          if constexpr (IS_CAUSAL) {
            is_valid = global_n <= global_q_pos;
          }
          if (window_size_left >= 0) {
            is_valid = is_valid && global_n >= global_q_pos - window_size_left;
          }
          if (window_size_right >= 0) {
            is_valid = is_valid && global_n <= global_q_pos + window_size_right;
          }
          if constexpr (ANCHORED_SWA) {
            is_valid =
                is_valid && ((global_n < anchor_len) ||
                             (global_q_pos - global_n < anchored_window));
          }

          float acc = 0.0f;
          if (is_valid) {
#pragma unroll 8
            for (int d = 0; d < D; d += 2) {
              const __half2 q_h2 =
                  *reinterpret_cast<const __half2*>(sQ + row * Q_STRIDE + d);
              const __half2 k_h2 =
                  *reinterpret_cast<const __half2*>(sK + col * KV_STRIDE + d);
              const float2 q_f2 = __half22float2(q_h2);
              const float2 k_f2 = __half22float2(k_h2);
              acc = fmaf(q_f2.x, k_f2.x, acc);
              acc = fmaf(q_f2.y, k_f2.y, acc);
            }
          }
          sS[row * S_STRIDE + col] = is_valid ? acc * softmax_scale : NEG_INF;
        }
      } else if constexpr (D256_SW_PIPELINE_QK) {
        constexpr int QK_TILES = BLOCK_N / WMMA_N;
        if (warp_id < QK_TILES) {
          pipeline_qk_slice<Q_STRIDE, S_STRIDE, D / WMMA_K, IS_CAUSAL,
                            D256_SW_PIPELINE_PV, ANCHORED_SWA>(
              sQ, sS, K_cache_h, sPageIdx, sPageOffset, k_block_stride,
              k_token_stride, k_head_stride, kv_head_id, start_col,
              valid_k_rows, start_row, valid_q_rows, causal_q_offset,
              warp_id * WMMA_N, 0, false, true, softmax_scale, window_size_left,
              window_size_right, NEG_INF, anchor_len, anchored_window);
        }
      } else {
        const int num_tiles_m_qk = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_qk = (BLOCK_N + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_qk = (D + WMMA_K - 1) / WMMA_K;
        const int total_tiles_qk = num_tiles_m_qk * num_tiles_n_qk;
        const int tiles_per_warp_qk =
            (total_tiles_qk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned row_causal = (lane_id & 0b1) +
                                    ((lane_id >> 2) & 0b1) * 8 +
                                    ((lane_id >> 4) & 0b1) * 4;
        const unsigned col_causal =
            ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

        for (int tile_idx = 0; tile_idx < tiles_per_warp_qk; ++tile_idx) {
          const int global_tile_idx = warp_id * tiles_per_warp_qk + tile_idx;
          if (global_tile_idx >= total_tiles_qk) break;

          const int tile_m_idx = global_tile_idx / num_tiles_n_qk;
          const int tile_n_idx = global_tile_idx % num_tiles_n_qk;

          const int tile_m = tile_m_idx * WMMA_M;
          const int tile_n = tile_n_idx * WMMA_N;

          if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) {
            continue;
          }

          const int page_slot = tile_n >> 4;
          const int block_offset = sPageOffset[page_slot];
          const int physical_block_idx_direct = sPageIdx[page_slot];

          fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
          fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
          fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
          fill_fragment(acc_frag, 0.0f);

#pragma unroll
          for (int k_tile = 0; k_tile < num_tiles_k_qk; ++k_tile) {
            const int k_offset = k_tile * WMMA_K;
            if (k_offset >= D) break;

            const __half* k_tile_ptr;
            if constexpr (LOW_SMEM_CONTIG_FAST) {
              k_tile_ptr =
                  low_smem_contiguous_page_tile
                      ? K_cache_h + (int64_t)sPageIdx[0] * k_block_stride +
                            (int64_t)(sPageOffset[0] + tile_n) *
                                k_token_stride +
                            (int64_t)kv_head_id * k_head_stride + k_offset
                      : K_cache_h +
                            (int64_t)physical_block_idx_direct *
                                k_block_stride +
                            (int64_t)block_offset * k_token_stride +
                            (int64_t)kv_head_id * k_head_stride + k_offset;
            } else {
              k_tile_ptr = K_cache_h +
                           (int64_t)physical_block_idx_direct * k_block_stride +
                           (int64_t)block_offset * k_token_stride +
                           (int64_t)kv_head_id * k_head_stride + k_offset;
            }

            load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_offset,
                             Q_STRIDE);
            load_matrix_sync(b_frag, k_tile_ptr, k_token_stride);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
          }

#pragma unroll
          for (int i = 0; i < acc_frag.num_elements; ++i) {
            const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
            const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;

            const int global_m = start_row + tile_m + row;
            const int global_n = start_col + tile_n + col;
            const int global_q_pos = global_m + causal_q_offset;

            const bool is_valid = (global_m < start_row + valid_q_rows) &&
                                  (global_n < start_col + valid_k_rows);
            bool is_causal_valid = true;
            if constexpr (IS_CAUSAL) {
              is_causal_valid = global_n <= global_q_pos;
            }
            bool is_window_valid = true;
            if (window_size_left >= 0) {
              is_window_valid = is_window_valid &&
                                global_n >= global_q_pos - window_size_left;
            }
            if (window_size_right >= 0) {
              is_window_valid = is_window_valid &&
                                global_n <= global_q_pos + window_size_right;
            }

            bool is_anchor_valid = true;
            if constexpr (ANCHORED_SWA) {
              is_anchor_valid = (global_n < anchor_len) ||
                                (global_q_pos - global_n < anchored_window);
            }

            acc_frag.x[i] = (is_valid && is_causal_valid && is_window_valid &&
                             is_anchor_valid)
                                ? acc_frag.x[i] * softmax_scale
                                : NEG_INF;
          }

          store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE,
                            mem_row_major);
        }
      }
      __syncthreads();
    } else {
      if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
        const __half* K_cache_h = reinterpret_cast<const __half*>(K_cache);
        const uint4* k_page0_vec = reinterpret_cast<const uint4*>(
            K_cache_h + (int64_t)physical_block_idx0 * k_block_stride +
            (int64_t)page_offset * k_token_stride +
            (int64_t)kv_head_id * k_head_stride);
        const uint4* k_page1_vec =
            two_page_tile && second_page_rows > 0
                ? reinterpret_cast<const uint4*>(
                      K_cache_h +
                      (int64_t)physical_block_idx1 * k_block_stride +
                      (int64_t)kv_head_id * k_head_stride)
                : nullptr;

#pragma unroll 2
        for (int idx = tid; idx < (valid_k_rows * d_stride_uint4);
             idx += THREADS_PER_BLOCK) {
          const int row = idx / d_stride_uint4;
          const int vec_col = idx % d_stride_uint4;

          uint4 k_val = make_uint4(0, 0, 0, 0);
          if (row < valid_k_rows && vec_col < d_stride_uint4) {
            if (four_page_contiguous_tile) {
              k_val = __ldg(&k_page0_vec[row * row_stride_uint4 + vec_col]);
            } else if (single_page_tile) {
              k_val = __ldg(&k_page0_vec[row * row_stride_uint4 + vec_col]);
            } else if (two_page_tile) {
              if (row < first_page_rows) {
                k_val = __ldg(&k_page0_vec[row * row_stride_uint4 + vec_col]);
              } else {
                const int row_page1 = row - first_page_rows;
                k_val =
                    __ldg(&k_page1_vec[row_page1 * row_stride_uint4 + vec_col]);
              }
            } else {
              const uint4* k_vec = reinterpret_cast<const uint4*>(K_cache_h);
              const int global_token_idx = start_col + row;
              const int virtual_block_idx = global_token_idx / page_block_size;
              const int block_offset = global_token_idx % page_block_size;
              const int physical_block_idx_slow =
                  __ldg(&block_table_seq[virtual_block_idx]);
              const int64_t physical_offset_halves =
                  (int64_t)physical_block_idx_slow * k_block_stride +
                  (int64_t)block_offset * k_token_stride +
                  (int64_t)kv_head_id * k_head_stride;
              const int64_t physical_offset_uint4 =
                  (physical_offset_halves / PER_UINT4) + vec_col;
              k_val = __ldg(&k_vec[physical_offset_uint4]);
            }
          }
          sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
        }
      } else {
        for (int idx = tid; idx < valid_k_rows * D; idx += THREADS_PER_BLOCK) {
          const int row = idx / D;
          const int col = idx % D;
          const int global_token_idx = start_col + row;
          const int virtual_block_idx = global_token_idx / page_block_size;
          const int block_offset = global_token_idx % page_block_size;
          const int physical_block_idx_slow =
              __ldg(&block_table_seq[virtual_block_idx]);
          const int64_t physical_offset =
              (int64_t)physical_block_idx_slow * k_block_stride +
              (int64_t)block_offset * k_token_stride +
              (int64_t)kv_head_id * k_head_stride + col;
          sK[row * KV_STRIDE + col] = flash_v100::load_kv_cache_half<KV_DTYPE>(
              K_cache, physical_offset, k_scale);
        }
      }
      __syncthreads();

      const int num_tiles_m_qk = (BLOCK_M + WMMA_M - 1) / WMMA_M;
      const int num_tiles_n_qk = (BLOCK_N + WMMA_N - 1) / WMMA_N;
      const int num_tiles_k_qk = (D + WMMA_K - 1) / WMMA_K;
      const int total_tiles_qk = num_tiles_m_qk * num_tiles_n_qk;
      const int tiles_per_warp_qk =
          (total_tiles_qk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
      const unsigned row_causal = (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8 +
                                  ((lane_id >> 4) & 0b1) * 4;
      const unsigned col_causal =
          ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

      for (int tile_idx = 0; tile_idx < tiles_per_warp_qk; ++tile_idx) {
        const int global_tile_idx = warp_id * tiles_per_warp_qk + tile_idx;
        if (global_tile_idx >= total_tiles_qk) break;

        const int tile_m_idx = global_tile_idx / num_tiles_n_qk;
        const int tile_n_idx = global_tile_idx % num_tiles_n_qk;

        const int tile_m = tile_m_idx * WMMA_M;
        const int tile_n = tile_n_idx * WMMA_N;

        if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) continue;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        fill_fragment(acc_frag, 0.0f);

#pragma unroll
        for (int k_tile = 0; k_tile < num_tiles_k_qk; ++k_tile) {
          const int k_offset = k_tile * WMMA_K;
          if (k_offset >= D) break;

          load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
          load_matrix_sync(b_frag, sK + tile_n * KV_STRIDE + k_offset,
                           KV_STRIDE);
          mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

#pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
          const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
          const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;

          const int global_m = start_row + tile_m + row;
          const int global_n = start_col + tile_n + col;
          const int global_q_pos = global_m + causal_q_offset;

          const bool is_valid = (global_m < start_row + valid_q_rows) &&
                                (global_n < start_col + valid_k_rows);
          bool is_causal_valid = true;
          if constexpr (IS_CAUSAL) {
            is_causal_valid = global_n <= global_q_pos;
          }
          bool is_window_valid = true;
          if (window_size_left >= 0) {
            is_window_valid =
                is_window_valid && global_n >= global_q_pos - window_size_left;
          }
          if (window_size_right >= 0) {
            is_window_valid =
                is_window_valid && global_n <= global_q_pos + window_size_right;
          }

          bool is_anchor_valid = true;
          if constexpr (ANCHORED_SWA) {
            is_anchor_valid = (global_n < anchor_len) ||
                              (global_q_pos - global_n < anchored_window);
          }

          acc_frag.x[i] = (is_valid && is_causal_valid && is_window_valid &&
                           is_anchor_valid)
                              ? acc_frag.x[i] * softmax_scale
                              : NEG_INF;
        }

        store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE,
                          mem_row_major);
      }
      __syncthreads();
    }

    uint4* sV_vec = reinterpret_cast<uint4*>(sV);
    const int64_t v_row_stride_uint4 = v_row_stride / PER_UINT4;
    if constexpr (!LOW_SMEM) {
      if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
        const __half* V_cache_h = reinterpret_cast<const __half*>(V_cache);
        const uint4* v_page0_vec = reinterpret_cast<const uint4*>(
            V_cache_h + (int64_t)physical_block_idx0 * v_block_stride +
            (int64_t)page_offset * v_token_stride +
            (int64_t)kv_head_id * v_head_stride);
        const uint4* v_page1_vec =
            two_page_tile && second_page_rows > 0
                ? reinterpret_cast<const uint4*>(
                      V_cache_h +
                      (int64_t)physical_block_idx1 * v_block_stride +
                      (int64_t)kv_head_id * v_head_stride)
                : nullptr;

#pragma unroll 2
        for (int idx = tid; idx < (valid_k_rows * d_stride_uint4);
             idx += THREADS_PER_BLOCK) {
          const int row = idx / d_stride_uint4;
          const int vec_col = idx % d_stride_uint4;

          uint4 v_val = make_uint4(0, 0, 0, 0);
          if (row < valid_k_rows && vec_col < d_stride_uint4) {
            if (four_page_contiguous_tile) {
              v_val = __ldg(&v_page0_vec[row * v_row_stride_uint4 + vec_col]);
            } else if (single_page_tile) {
              v_val = __ldg(&v_page0_vec[row * v_row_stride_uint4 + vec_col]);
            } else if (two_page_tile) {
              if (row < first_page_rows) {
                v_val = __ldg(&v_page0_vec[row * v_row_stride_uint4 + vec_col]);
              } else {
                const int row_page1 = row - first_page_rows;
                v_val = __ldg(
                    &v_page1_vec[row_page1 * v_row_stride_uint4 + vec_col]);
              }
            } else {
              const uint4* v_vec = reinterpret_cast<const uint4*>(V_cache_h);
              const int global_token_idx = start_col + row;
              const int virtual_block_idx = global_token_idx / page_block_size;
              const int block_offset = global_token_idx % page_block_size;
              const int physical_block_idx_slow =
                  __ldg(&block_table_seq[virtual_block_idx]);
              const int64_t physical_offset_halves =
                  (int64_t)physical_block_idx_slow * v_block_stride +
                  (int64_t)block_offset * v_token_stride +
                  (int64_t)kv_head_id * v_head_stride;
              const int64_t physical_offset_uint4 =
                  (physical_offset_halves / PER_UINT4) + vec_col;
              v_val = __ldg(&v_vec[physical_offset_uint4]);
            }
          }
          sV_vec[row * kv_stride_uint4 + vec_col] = v_val;
        }
      } else {
        for (int idx = tid; idx < valid_k_rows * D; idx += THREADS_PER_BLOCK) {
          const int row = idx / D;
          const int col = idx % D;
          const int global_token_idx = start_col + row;
          const int virtual_block_idx = global_token_idx / page_block_size;
          const int block_offset = global_token_idx % page_block_size;
          const int physical_block_idx_slow =
              __ldg(&block_table_seq[virtual_block_idx]);
          const int64_t physical_offset =
              (int64_t)physical_block_idx_slow * v_block_stride +
              (int64_t)block_offset * v_token_stride +
              (int64_t)kv_head_id * v_head_stride + col;
          sV[row * KV_STRIDE + col] = flash_v100::load_kv_cache_half<KV_DTYPE>(
              V_cache, physical_offset, v_scale);
        }
      }
      __syncthreads();
    }

    const int num_tiles_m_pv = (BLOCK_M + WMMA_M - 1) / WMMA_M;
    const int num_tiles_n_pv = (D + WMMA_N - 1) / WMMA_N;
    const int total_tiles_pv = num_tiles_m_pv * num_tiles_n_pv;
    const int tiles_per_warp_pv =
        (total_tiles_pv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int softmax_sub_tile = (D == 256) ? P_SUB_TILE : p_tile_capacity;

    for (int sub_start = 0; sub_start < valid_k_rows;
         sub_start += softmax_sub_tile) {
      const int sub_valid_k_rows =
          min(softmax_sub_tile, valid_k_rows - sub_start);

      // ================================================================
      // Patch 0108 (surgical) — sS/sP union race fix, D-conditional.
      // ISSUE-0007 / issues/0090. The race (W(__half2) P-commit racing
      // R(float4) sS reads with no cross-warp barrier) exists ONLY for
      // D != 256, where sP = reuse_sp.p aliases sS byte-for-byte.
      //
      // D == 256 is race-IMMUNE: sP = p_strict, a SEPARATE buffer
      // (SmemLayout::p_strict, line ~134; selected at line ~491). So the
      // D==256 arm below is the ORIGINAL monolithic block VERBATIM — no
      // barrier, no variable hoisting. This is the ONLY difference from
      // archived/patches/0011: 0011 split the guard and hoisted vars for
      // EVERY D, which perturbed the D=256 register/spill allocation and
      // cost the immune path +0.47..+2.79% for zero benefit. Guarding the
      // whole transform under `if constexpr (D != 256)` makes the D=256
      // codegen byte-identical to baseline (proved via SASS diff) => the
      // immune path pays EXACTLY zero. The D!=256 arm is 0011's
      // two-phase-commit fix, verbatim (barrier ALONE, no riders).
      // ================================================================
      if constexpr (D == 256) {
        if (tid < valid_q_rows * THREADS_PER_ROW) {
          const int row = tid / THREADS_PER_ROW;
          const int thread_in_row = tid % THREADS_PER_ROW;
          const unsigned mask =
              (valid_q_rows == BLOCK_M) ? 0xFFFFFFFFU : __activemask();
          const int row_leader = __ffs(mask) - 1;

          float* sS_row_f = sS + row * S_STRIDE + sub_start;
          __half* sP_row_h = sP + row * p_stride;

          const int vec_cols = sub_valid_k_rows >> 2;
          const int vecs_per_thread =
              (vec_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
          const int tail_start = vec_cols << 2;

          float thread_max = NEG_INF;
          float4* sS_vec4 = reinterpret_cast<float4*>(sS_row_f);

#pragma unroll 4
          for (int j = 0; j < vecs_per_thread; ++j) {
            int vc = thread_in_row + j * THREADS_PER_ROW;
            if (vc < vec_cols) {
              float4 v4 = sS_vec4[vc];
              thread_max = fmaxf(thread_max,
                                 fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
            }
          }

#pragma unroll
          for (int c = tail_start + thread_in_row; c < sub_valid_k_rows;
               c += THREADS_PER_ROW) {
            thread_max = fmaxf(thread_max, sS_row_f[c]);
          }

#pragma unroll
          for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o,
                                                            THREADS_PER_ROW));
          }

          const float row_max =
              __shfl_sync(mask, thread_max, row_leader, THREADS_PER_ROW);
          const float old_max = sRowMax[row];
          const float new_max = fmaxf(old_max, row_max);
          const float exp_diff = __expf(old_max - new_max);

          float thread_sum = 0.0f;
          __half2 half_buffer[20];
          int vc_base = thread_in_row;
          int h2_idx = 0;
          int tail_col = -1;
          __half tail_value = __float2half(0.f);
          __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);

#pragma unroll 4
          for (int j = 0; j < vecs_per_thread;
               ++j, vc_base += THREADS_PER_ROW) {
            if (vc_base < vec_cols) {
              float4 v4 = sS_vec4[vc_base];

              float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
              float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
              float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
              float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));

              thread_sum += (e0 + e1) + (e2 + e3);

              const __half2 p01 = __float22half2_rn(make_float2(e0, e1));
              const __half2 p23 = __float22half2_rn(make_float2(e2, e3));
              if constexpr (D256_SW_PIPELINE_PV) {
                // D256 owns a separate probability buffer, so the
                // values can leave registers before row reduction.
                const int base_offset = vc_base * 2;
                sP_half2[base_offset] = p01;
                sP_half2[base_offset + 1] = p23;
              } else {
                half_buffer[h2_idx++] = p01;
                half_buffer[h2_idx++] = p23;
              }
            }
          }

#pragma unroll 4
          for (int c = tail_start + thread_in_row; c < sub_valid_k_rows;
               c += THREADS_PER_ROW) {
            float v = sS_row_f[c];
            float e = __expf(fmaxf(v - new_max, -80.0f));
            thread_sum += e;
            if constexpr (D256_SW_PIPELINE_PV) {
              sP_row_h[c] = __float2half_rn(e);
            } else {
              tail_col = c;
              tail_value = __float2half_rn(e);
            }
          }

#pragma unroll
          for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
            thread_sum +=
                __shfl_down_sync(mask, thread_sum, o, THREADS_PER_ROW);
          }

          float row_sum =
              __shfl_sync(mask, thread_sum, row_leader, THREADS_PER_ROW);

          if (thread_in_row == 0) {
            sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
            sRowMax[row] = new_max;
          }

          if constexpr (!D256_SW_PIPELINE_PV) {
            h2_idx = 0;
            vc_base = thread_in_row;
#pragma unroll 4
            for (int j = 0; j < vecs_per_thread;
                 ++j, vc_base += THREADS_PER_ROW) {
              if (vc_base < vec_cols) {
                int base_offset = vc_base * 2;
                sP_half2[base_offset] = half_buffer[h2_idx++];
                sP_half2[base_offset + 1] = half_buffer[h2_idx++];
              }
            }

            if (tail_col >= 0) {
              sP_row_h[tail_col] = tail_value;
            }
          }

#pragma unroll 4
          for (int c = tail_start + thread_in_row; c < p_tile_capacity;
               c += THREADS_PER_ROW) {
            if (c >= sub_valid_k_rows) {
              sP_row_h[c] = __float2half(0.f);
            }
          }

          if (block_n > 0 || sub_start > 0) {
            float* sO_row = sO + row * O_STRIDE;
            float4* sO_vec = reinterpret_cast<float4*>(sO_row);
            const int o_vec_count = D / 4;
            float scale = exp_diff;

#pragma unroll 4
            for (int ov = thread_in_row; ov < o_vec_count;
                 ov += THREADS_PER_ROW) {
              float4 v = sO_vec[ov];
              v.x *= scale;
              v.y *= scale;
              v.z *= scale;
              v.w *= scale;
              sO_vec[ov] = v;
            }
          }
        }
      } else {
        // ----------------------------------------------------------------
        // ISSUE-0007 race-class fix (patch 0011,
        // patches/0011-paged-race-fix-issue0007/ — PoC: poc/race_poc.cu).
        //
        // For D != 256, sP aliases sS byte-for-byte (union reuse_sp;
        // sP = reuse_sp.p selected above with p_stride == S_STRIDE). The
        // softmax phase reads sS as float4 (max pass, exp pass, scalar
        // tail) and then commits P into the SAME union bytes as
        // __half2/__half. Row r's half P row occupies union bytes
        // [2*S_STRIDE*r, +2*S_STRIDE) — inside the FLOAT score row of row
        // r/2 — so the P commit of warp w lands exactly on the score row
        // still being read as float4 by warp w/2. The register staging
        // (half_buffer) orders read-then-write only WITHIN one thread;
        // cross-warp there was NO ordering until the __syncthreads() at
        // the end of this phase, i.e. AFTER both: a W(__half2) x R(float4)
        // data race. Evidence: compute-sanitizer racecheck 3 errors /
        // 87,712 hazard instances on the PoC extraction of this exact
        // access pattern, and the REAL kernel 10/10-runs bitwise
        // non-deterministic at D=128 (q=512/kv=4096 and q=1024/kv=8192).
        //
        // Fix, proved racecheck-clean AND bit-equal in the PoC: two-phase
        // P commit. Phase A = all sS reads + reductions with P staged in
        // registers exactly as before; ONE uniform block-scope
        // __syncthreads(); phase B = all sP stores. The former single
        // guarded region is split in two so the barrier sits in UNIFORM
        // control flow, and the per-thread state below is hoisted to
        // survive across the barrier. No arithmetic is changed or
        // reordered. D == 256 keeps its original barrier-free structure:
        // sP = p_strict there, a separate buffer that never aliases sS
        // (constexpr guard on the barrier below).
        // ----------------------------------------------------------------
        const int row = tid / THREADS_PER_ROW;
        const int thread_in_row = tid % THREADS_PER_ROW;
        __half* sP_row_h = sP + row * p_stride;
        __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
        const int vec_cols = sub_valid_k_rows >> 2;
        const int vecs_per_thread =
            (vec_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
        const int tail_start = vec_cols << 2;
        float exp_diff = 1.0f;
        __half2 half_buffer[20];
        int tail_col = -1;
        __half tail_value = __float2half(0.f);

        if (tid < valid_q_rows * THREADS_PER_ROW) {
          const unsigned mask =
              (valid_q_rows == BLOCK_M) ? 0xFFFFFFFFU : __activemask();
          const int row_leader = __ffs(mask) - 1;

          float* sS_row_f = sS + row * S_STRIDE + sub_start;

          float thread_max = NEG_INF;
          float4* sS_vec4 = reinterpret_cast<float4*>(sS_row_f);

#pragma unroll 4
          for (int j = 0; j < vecs_per_thread; ++j) {
            int vc = thread_in_row + j * THREADS_PER_ROW;
            if (vc < vec_cols) {
              float4 v4 = sS_vec4[vc];
              thread_max = fmaxf(thread_max,
                                 fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
            }
          }

#pragma unroll
          for (int c = tail_start + thread_in_row; c < sub_valid_k_rows;
               c += THREADS_PER_ROW) {
            thread_max = fmaxf(thread_max, sS_row_f[c]);
          }

#pragma unroll
          for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o,
                                                            THREADS_PER_ROW));
          }

          const float row_max =
              __shfl_sync(mask, thread_max, row_leader, THREADS_PER_ROW);
          const float old_max = sRowMax[row];
          const float new_max = fmaxf(old_max, row_max);
          // 0011: exp_diff hoisted above the phase split (used by the
          // sO rescale in phase B). Same expression, same position in
          // the arithmetic order.
          exp_diff = __expf(old_max - new_max);

          float thread_sum = 0.0f;
          int vc_base = thread_in_row;
          int h2_idx = 0;

#pragma unroll 4
          for (int j = 0; j < vecs_per_thread;
               ++j, vc_base += THREADS_PER_ROW) {
            if (vc_base < vec_cols) {
              float4 v4 = sS_vec4[vc_base];

              float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
              float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
              float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
              float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));

              thread_sum += (e0 + e1) + (e2 + e3);

              const __half2 p01 = __float22half2_rn(make_float2(e0, e1));
              const __half2 p23 = __float22half2_rn(make_float2(e2, e3));
              if constexpr (D256_SW_PIPELINE_PV) {
                // D256 owns a separate probability buffer, so the
                // values can leave registers before row reduction.
                const int base_offset = vc_base * 2;
                sP_half2[base_offset] = p01;
                sP_half2[base_offset + 1] = p23;
              } else {
                half_buffer[h2_idx++] = p01;
                half_buffer[h2_idx++] = p23;
              }
            }
          }

#pragma unroll 4
          for (int c = tail_start + thread_in_row; c < sub_valid_k_rows;
               c += THREADS_PER_ROW) {
            float v = sS_row_f[c];
            float e = __expf(fmaxf(v - new_max, -80.0f));
            thread_sum += e;
            if constexpr (D256_SW_PIPELINE_PV) {
              sP_row_h[c] = __float2half_rn(e);
            } else {
              tail_col = c;
              tail_value = __float2half_rn(e);
            }
          }

#pragma unroll
          for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
            thread_sum +=
                __shfl_down_sync(mask, thread_sum, o, THREADS_PER_ROW);
          }

          float row_sum =
              __shfl_sync(mask, thread_sum, row_leader, THREADS_PER_ROW);

          if (thread_in_row == 0) {
            sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
            sRowMax[row] = new_max;
          }
        }  // end phase A: every sS read of this sub-tile is done; P is
           // staged in registers (half_buffer / tail_col / tail_value).

        // ISSUE-0007 fix (patch 0011): THE one uniform __syncthreads()
        // between the sS read phase and the sP commit phase — the only
        // behavioral change of this patch. UNIFORMITY: this point is at
        // the body scope of the sub_start loop; its trip count
        // (valid_k_rows, softmax_sub_tile) and every enclosing control
        // transfer (kernel early returns on blockIdx.z/M; block_n loop
        // break/continue on start_col/min_key_pos/bfla block mask) are
        // block-uniform — the same uniformity the pre-existing
        // __syncthreads() at the end of this loop body already requires.
        // All threads of the block reach this barrier exactly once per
        // sub_start iteration. D == 256 is immune (sP = p_strict,
        // separate buffer) and keeps its barrier-free codegen.
        if constexpr (D != 256) {
          __syncthreads();
        }

        if (tid < valid_q_rows * THREADS_PER_ROW) {
          // phase B: commit P to the union bytes. Bit-identical values
          // (straight from the phase-A registers), same store order,
          // same addresses as the pre-fix code.
          if constexpr (!D256_SW_PIPELINE_PV) {
            int h2_idx = 0;
            int vc_base = thread_in_row;
#pragma unroll 4
            for (int j = 0; j < vecs_per_thread;
                 ++j, vc_base += THREADS_PER_ROW) {
              if (vc_base < vec_cols) {
                int base_offset = vc_base * 2;
                sP_half2[base_offset] = half_buffer[h2_idx++];
                sP_half2[base_offset + 1] = half_buffer[h2_idx++];
              }
            }

            if (tail_col >= 0) {
              sP_row_h[tail_col] = tail_value;
            }
          }

#pragma unroll 4
          for (int c = tail_start + thread_in_row; c < p_tile_capacity;
               c += THREADS_PER_ROW) {
            if (c >= sub_valid_k_rows) {
              sP_row_h[c] = __float2half(0.f);
            }
          }

          if (block_n > 0 || sub_start > 0) {
            float* sO_row = sO + row * O_STRIDE;
            float4* sO_vec = reinterpret_cast<float4*>(sO_row);
            const int o_vec_count = D / 4;
            float scale = exp_diff;

#pragma unroll 4
            for (int ov = thread_in_row; ov < o_vec_count;
                 ov += THREADS_PER_ROW) {
              float4 v = sO_vec[ov];
              v.x *= scale;
              v.y *= scale;
              v.z *= scale;
              v.w *= scale;
              sO_vec[ov] = v;
            }
          }
        }
      }
      __syncthreads();

      const int num_tiles_k_pv = (p_tile_capacity + WMMA_K - 1) / WMMA_K;

      for (int tile_idx = 0; tile_idx < tiles_per_warp_pv; ++tile_idx) {
        const int global_tile_idx = warp_id * tiles_per_warp_pv + tile_idx;
        if (global_tile_idx >= total_tiles_pv) break;

        const int tile_m_idx = global_tile_idx / num_tiles_n_pv;
        const int tile_d_idx = global_tile_idx % num_tiles_n_pv;

        const int tile_m = tile_m_idx * WMMA_M;
        const int tile_d = tile_d_idx * WMMA_N;

        if (tile_m >= valid_q_rows) continue;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        load_matrix_sync(acc_frag, sO + tile_m * O_STRIDE + tile_d, O_STRIDE,
                         mem_row_major);

#pragma unroll
        for (int tile_k = 0; tile_k < num_tiles_k_pv; ++tile_k) {
          const int k_offset = tile_k * WMMA_K;
          if (k_offset >= sub_valid_k_rows) break;

          load_matrix_sync(a_frag, sP + tile_m * p_stride + k_offset, p_stride);
          if constexpr (LOW_SMEM) {
            const __half* V_cache_h = reinterpret_cast<const __half*>(V_cache);
            const int token_offset = sub_start + k_offset;
            const int page_slot = token_offset >> 4;
            const int block_offset = sPageOffset[page_slot];
            const int physical_block_idx_direct = sPageIdx[page_slot];
            const __half* v_tile_ptr;
            if constexpr (LOW_SMEM_CONTIG_FAST) {
              v_tile_ptr =
                  low_smem_contiguous_page_tile
                      ? V_cache_h + (int64_t)sPageIdx[0] * v_block_stride +
                            (int64_t)(sPageOffset[0] + token_offset) *
                                v_token_stride +
                            (int64_t)kv_head_id * v_head_stride + tile_d
                      : V_cache_h +
                            (int64_t)physical_block_idx_direct *
                                v_block_stride +
                            (int64_t)block_offset * v_token_stride +
                            (int64_t)kv_head_id * v_head_stride + tile_d;
            } else {
              v_tile_ptr = V_cache_h +
                           (int64_t)physical_block_idx_direct * v_block_stride +
                           (int64_t)block_offset * v_token_stride +
                           (int64_t)kv_head_id * v_head_stride + tile_d;
            }
            load_matrix_sync(b_frag, v_tile_ptr, v_token_stride);
          } else {
            load_matrix_sync(b_frag,
                             sV + (sub_start + k_offset) * KV_STRIDE + tile_d,
                             KV_STRIDE);
          }
          mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        store_matrix_sync(sO + tile_m * O_STRIDE + tile_d, acc_frag, O_STRIDE,
                          mem_row_major);
      }
      __syncthreads();
    }
  }

  if constexpr (SPLIT_KV) {
    const int partition_id = blockIdx.y;
    const int num_partitions = gridDim.y;
    const int64_t row_base =
        ((int64_t)batch_head_id * num_partitions + partition_id) * M +
        start_row;

    for (int i = tid; i < valid_q_rows * D; i += THREADS_PER_BLOCK) {
      const int row = i / D;
      const int col = i - row * D;
      split_tmp_out[(row_base + row) * D + col] = sO[row * O_STRIDE + col];
    }

    if (tid < valid_q_rows) {
      split_tmp_row_max[row_base + tid] = sRowMax[tid];
      split_tmp_row_sum[row_base + tid] = sRowSum[tid];
    }
    return;
  }

  const int total_fp16_x4 = (valid_q_rows * D) / 4;

  for (int i = tid; i < total_fp16_x4; i += THREADS_PER_BLOCK) {
    const int row = i / (D / 4);
    const int col = (i % (D / 4)) * 4;

    const float sum_clamped = fmaxf(sRowSum[row], 1e-24f);
    const float inv_sum = 1.0f / sum_clamped;
    const float* sO_row = sO + row * O_STRIDE;

    const __half h0 = __float2half_rn(sO_row[col + 0] * inv_sum);
    const __half h1 = __float2half_rn(sO_row[col + 1] * inv_sum);
    const __half h2 = __float2half_rn(sO_row[col + 2] * inv_sum);
    const __half h3 = __float2half_rn(sO_row[col + 3] * inv_sum);

    asm volatile("st.global.v4.u16 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(out_ptr + row * D + col), "h"(__half_as_ushort(h0)),
                   "h"(__half_as_ushort(h1)), "h"(__half_as_ushort(h2)),
                   "h"(__half_as_ushort(h3))
                 : "memory");
  }

  if (tid < valid_q_rows) {
    const float sum = fmaxf(sRowSum[tid], 1e-24f);
    softmax_lse_ptr[tid] = sRowMax[tid] + logf(sum);
  }
}

inline bool env_flag_enabled(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && std::strcmp(raw, "0") != 0;
}

inline bool env_flag_default_enabled(const char* name) {
  const char* raw = std::getenv(name);
  return raw == nullptr || std::strcmp(raw, "0") != 0;
}


namespace ninfer::third_party::onecat {

void flash_attention_forward_f16_paged_launch(
    const void* q, const void* k_cache, const void* v_cache, void* out,
    float* softmax_lse, const int* block_table, const int* seq_lens, int B, int H,
    int M, int page_block_size, int num_kv_heads, int max_blocks,
    long long k_block_stride, long long k_token_stride, long long k_head_stride,
    long long v_block_stride, long long v_token_stride, long long v_head_stride,
    float softmax_scale, bool causal, float k_scale, float v_scale,
    int window_size_left, int window_size_right, cudaStream_t stream) {
  constexpr int D = 256;
  constexpr bool LOW_SMEM = false;
  constexpr bool LOW_SMEM_CONTIG_FAST = false;
  constexpr bool LOW_SMEM_SCALAR_QK = false;
  constexpr bool LOW_SMEM_BM32 = false;
  constexpr bool IS_CAUSAL = true;
  constexpr int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16;
  using Config = KernelConfig<D, LOW_SMEM, LOW_SMEM_SCALAR_QK, LOW_SMEM_BM32>;
  const int N = max_blocks * page_block_size;
  const dim3 grid((M + Config::BLOCK_M - 1) / Config::BLOCK_M, 1, B * H);
  const dim3 block(Config::THREADS_PER_BLOCK);
  const size_t smem = Config::TOTAL_SMEM;
  auto kernel = flash_attention_forward_kernel_paged<
      D, LOW_SMEM, LOW_SMEM_CONTIG_FAST, LOW_SMEM_SCALAR_QK, LOW_SMEM_BM32,
      false, IS_CAUSAL, KV_DTYPE>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));
  kernel<<<grid, block, smem, stream>>>(
      static_cast<const __half*>(q), k_cache, v_cache, static_cast<__half*>(out),
      softmax_lse, block_table, seq_lens, B, H, M, N, nullptr, 0, 0, 0, 0, 0,
      page_block_size, num_kv_heads, k_block_stride, k_token_stride,
      k_head_stride, v_block_stride, v_token_stride, v_head_stride,
      softmax_scale, k_scale, v_scale, window_size_left, window_size_right,
      nullptr, nullptr, nullptr, 0, nullptr, 0);
}

}  // namespace ninfer::third_party::onecat
