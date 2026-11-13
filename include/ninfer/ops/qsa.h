#pragma once

// ninfer::ops - QSA (indexed sparse attention) preparation, selection, attention and tail state.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Main K/V pages hold 64 token rows; an index page holds the 16 four-token blocks of the same
// 64-token logical span. One page-table row maps both the token page and its block rows.
inline constexpr std::int32_t kQsaTokenPageSize       = 64;
inline constexpr std::int32_t kQsaIndexBlockPageSize  = 16;
inline constexpr std::int32_t kQsaIndexCompress       = 4;
inline constexpr std::int32_t kQsaSelectionCapacity   = 512;

/**
 * The five projected QSA tensors. All are contiguous BF16, dimension 0 fastest.
 *
 *   query       [head_dim, q_heads, T, B]
 *   key         [head_dim, kv_heads, T, B]
 *   value       [head_dim, kv_heads, T, B]
 *   index_query [index_head_dim, index_heads, T, B]
 *   index_key   [index_head_dim, index_kv_heads, T, B]
 *
 * q_heads is divisible by kv_heads and index_kv_heads must be one. The registered Flash-Next
 * geometry is head_dim=256, q_heads=24, kv_heads=2, index_head_dim=128, index_heads=4.
 */
struct QsaProjectionView {
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor index_query;
    Tensor index_key;
};

/**
 * Norm weights and rotation constants consumed by qsa_prepare.
 *
 * All four weights are contiguous BF16 gains of length head_dim, head_dim, index_head_dim and
 * index_head_dim. rms_epsilon, rotary_dim and theta are the text Qwen MRoPE constants; rotary_dim
 * is exactly 64 and theta is positive and finite.
 */
struct QsaPrepareParameters {
    Tensor query_norm;
    Tensor key_norm;
    Tensor index_query_norm;
    Tensor index_key_norm;
    float rms_epsilon = 0.0F;
    std::int32_t rotary_dim = 0;
    float theta = 0.0F;
};

/**
 * Per-call batch coordinates.
 *
 *   positions     I32 [T, A, B] with A in {1, 3}; contiguous. Axis 0 is the token ordinal used by
 *                 every block/tail computation; axis 1 carries the A rotary coordinates. A=1 is
 *                 broadcast to all three rotary axes.
 *   valid_columns I32 [B] or an empty Tensor meaning every row has T valid columns; each value is
 *                 in [1, T].
 *   table_rows    I32 [B], selecting a row of the page tables.
 */
struct QsaBatchView {
    Tensor positions;
    Tensor valid_columns;
    Tensor table_rows;
};

/**
 * The canonical four-token QSA tail state at the first token of the next call.
 *
 *   raw_keys   BF16 [index_head_dim, 4, B]; the first F%4 columns hold the unfinished block in
 *              chronological order and the rest are BF16 +0.
 *   first_rope I32 [3, B]; the first token coordinates of that unfinished block, or three I32
 *              zeros when F%4 == 0.
 */
struct QsaTailStateView {
    Tensor raw_keys;
    Tensor first_rope;
};

/**
 * Persistent QSA cache planes and page table. All tensors are contiguous and dimension 0 fastest.
 *
 *   k_pages       BF16 [head_dim, 64, kv_heads, physical_pages]
 *   v_pages       FP16 [head_dim, 64, kv_heads, physical_pages]
 *   index_k_pages BF16 [index_head_dim, 16, index_kv_heads, physical_pages]
 *   page_table    I32 [logical_pages, table_rows]
 *
 * Main token p uses logical page p/64, row p%64; index block g uses logical page g/16, row g%16.
 * For an index block g, the first token of that block is token 4g.
 */
struct QsaCacheView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor index_k_pages;
    Tensor page_table;
};

/** Read-only projection of the index plane and page table used by qsa_indexer. */
struct ConstQsaIndexCacheView {
    Tensor index_k_pages;
    Tensor page_table;
};

/** Raw index-key words and rotary coordinates recorded by qsa_prepare for one call. */
struct QsaRawRecordWriteView {
    Tensor raw_keys;       // BF16 [index_head_dim, W, B]
    Tensor rope_positions; // I32 [3, W, B]
    Tensor base_positions; // I32 [B]; F = positions[0,0,b]
};

/** The read-only record form consumed by qsa_tail_extract. */
struct QsaRawRecordReadView {
    Tensor raw_keys;       // BF16 [index_head_dim, W, B]
    Tensor rope_positions; // I32 [3, W, B]
};

/**
 * Shared block selection. selection is I32 [512, T, B]; counts is I32 [T, B]. Only the first
 * counts[t,b] entries of selection[:,t,b] are meaningful and they are written in ascending block
 * order. Entries beyond counts are unspecified.
 */
struct QsaBlockSelection {
    Tensor selection;
    Tensor counts;
};

/**
 * Op: qsa_prepare
 *
 * Math / indexing:
 *   N_D(x,w) = x * (1 + FP32(w)) / sqrt( sum(x*x)/D + rms_epsilon )        (per head)
 *
 *   R(z,rho) rotates pairs (i,i+32), i in [0,32), by rho[i%3] * theta^(-2i/64):
 *     R[i]    = z[i]    * cos - z[i+32] * sin
 *     R[i+32] = z[i+32] * cos + z[i]    * sin
 *   Dimensions [64,D) retain z.
 *
 *   query[:,h,t,b]       = BF16(R(N_256(query_projection[:,h,t,b], query_norm), rho[t,:,b]))
 *   index_query[:,h,t,b] = BF16(R(N_128(index_query_projection[:,h,t,b], index_query_norm),
 *                                 rho[t,:,b]))
 *   k = R(N_256(key_projection[:,h,t,b], key_norm), rho[t,:,b]); cache K = BF16(k)
 *   v = value_projection[:,h,t,b];                              cache V = FP16_RNE(v)
 *
 *   Index blocks, with F = positions[0,0,b], n = valid_columns[b] and
 *   g in [floor(F/4), floor((F+n)/4)):
 *     u = BF16( mean(raw_index_key[4g:4g+4]) )
 *     index.K[g] = BF16( R( N_128(u, index_key_norm), rho_block ) )
 *   Raw keys preceding F come from before.raw_keys; the others from index_key_projection. A block
 *   starting before F rotates with before.first_rope, otherwise with the coordinate of its first
 *   token. Raw index-key words and actual coordinates are copied to records for every valid token.
 *
 * Logical shapes:
 *   T is the call width, B the batch size; W is records.raw_keys.ne[1].
 *   The main cache planes and page table follow QsaCacheView. positions/batch and the output
 *   tensors follow their view structs.
 *
 * Supported domain:
 *   BF16 inputs and outputs, FP16 value cache, BF16 key/index-key cache. q_heads%kv_heads == 0,
 *   index_kv_heads == 1, rotary_dim == 64 and rotary_dim <= head_dim and <= index_head_dim,
 *   rms_epsilon positive and finite, A in {1,3}. Every written token/block/page must be in range.
 *
 * Numeric:
 *   Normalization statistics accumulate in FP32; the gain 1 + FP32(w) is not rounded to BF16.
 *   The mean that forms u is rounded to BF16 before normalization, which is an observable
 *   boundary. The oracle evaluates the complete formula in FP64 from the represented inputs and
 *   reproduces that BF16 u rounding; the BF16 outputs and BF16 key cache are promoted and compared
 *   against it. The FP16 value cache stores the FP16_RNE representation of v, and its comparison
 *   decodes that stored word.
 *
 * Effects:
 *   Writes query, index_query, the addressed K/V/index-K cache rows, and every valid record
 *   column. before and the projection tensors are read-only and preserved.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory, keeps no persistent state and
 *   supports CUDA Graph capture and replay at bound addresses.
 */
void qsa_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const QsaTailStateView& before, QsaCacheView cache,
                 WorkspaceArena& workspace, QsaRawRecordWriteView& records, Tensor& query,
                 Tensor& index_query, cudaStream_t stream);

/**
 * Op: qsa_indexer
 *
 * Math / indexing:
 *   For every live (t,b), p = positions[t,0,b], C = floor((p+1)/4), and g in [0,C):
 *     score[g] = sum_{h=0..3} max( 0, sum_{d=0..127} index_query[d,h,t,b] * index.K[d,g] )
 *   The selection keeps min(512,C) highest scores. Exact ties prefer the smaller g. Block g reads
 *   index_k_pages[d, g%16, 0, page_table[g/16, table_rows[b]]]. The four index heads share one
 *   selection. When C <= 512 every block is selected without reading the query, keys or page
 *   table.
 *
 * Logical shapes:
 *   index_query BF16 [index_head_dim, index_heads, T, B]; selection I32 [512,T,B]; counts I32 [T,B].
 *   The cache follows ConstQsaIndexCacheView. index_heads is four and index_head_dim 128.
 *
 * Supported domain:
 *   BF16 index_query and index keys, I32 page table, index_kv_heads == 1, index_head_dim == 128.
 *   max_visible_tokens is a host launch-resource promise and does not alter any row's C.
 *
 * Numeric:
 *   Scores accumulate in FP32. Selection and counts are integer outputs and are compared exactly.
 *   The oracle evaluates the scores naively in FP64 and applies the same tie rule.
 *
 * Effects:
 *   Writes counts and the first counts[t,b] selection entries of every live row. Invalid rows have
 *   counts zero. The cache and index_query are read-only.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory, keeps no persistent state and
 *   supports CUDA Graph capture and replay at bound addresses.
 */
void qsa_indexer(const Tensor& index_query, const ConstQsaIndexCacheView& cache,
                 const QsaBatchView& batch, std::uint32_t max_visible_tokens,
                 WorkspaceArena& workspace, QsaBlockSelection& selected, cudaStream_t stream);

/**
 * Op: qsa_sparse_gqa
 *
 * Math / indexing:
 *   For every live (t,b), p = positions[t,0,b], C = floor((p+1)/4), c = counts[t,b]:
 *     J = { 4g .. 4g+3 : g in the first c selection ids } union [4C, p+1)
 *   For query head h, kvh = floor(h / (q_heads/kv_heads)):
 *     score[j]       = dot(query[:,h,t,b], K[:,j,kvh]) / sqrt(head_dim)
 *     attention[d,h] = sum_{j in J} softmax(score)[j] * V[d,j,kvh]
 *     out[d,h,t,b]   = BF16( attention[d,h] * sigmoid(FP32(output_gate[d,h,t,b])) )
 *   The indexer scores never enter the logits. Token j reads k_pages/v_pages at logical page j/64,
 *   row j%64. An empty J writes exact BF16 +0.
 *
 * Logical shapes:
 *   query/output_gate/out BF16 [head_dim, q_heads, T, B]; selection I32 [512,T,B]; counts I32 [T,B].
 *   The cache follows QsaCacheView.
 *
 * Supported domain:
 *   BF16 query, gate and output, BF16 K cache, FP16 V cache, head_dim == 256, q_heads%kv_heads == 0.
 *   Invalid token columns write exact BF16 zero.
 *
 * Numeric:
 *   Scores accumulate in FP32 and softmax is evaluated in FP32 with a max subtraction. The value
 *   reduction accumulates in FP32. The BF16 output is promoted and compared with an FP64 oracle
 *   that consumes the represented BF16 query/gate and the stored BF16/FP16 cache words.
 *
 * Effects:
 *   Writes all of out. Cache, selection, counts, query and output_gate are read-only.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory, keeps no persistent state and
 *   supports CUDA Graph capture and replay at bound addresses.
 */
void qsa_sparse_gqa(const Tensor& query, const Tensor& output_gate, const QsaCacheView& cache,
                    const QsaBatchView& batch, const QsaBlockSelection& selected,
                    std::uint32_t max_visible_tokens, WorkspaceArena& workspace, Tensor& out,
                    cudaStream_t stream);

/**
 * Op: qsa_tail_extract
 *
 * Math / indexing:
 *   For row b: F0 = base_positions[b], a = valid_columns[b], F1 = F0 + a, r = F1%4, s = F1-r.
 *
 *   after.raw_keys[c,j,b] = BF16 +0                              if j >= r
 *                           before.raw_keys[c, s+j-(F0-F0%4), b] if s+j < F0
 *                           records.raw_keys[c, s+j-F0, b]       otherwise
 *
 *   after.first_rope[:,b] = [0,0,0]                              if r == 0
 *                           before.first_rope[:,b]               if s < F0
 *                           records.rope_positions[:, s-F0, b]    otherwise
 *
 * Logical shapes:
 *   before/after raw_keys BF16 [index_head_dim, 4, B], first_rope I32 [3, B];
 *   records raw_keys BF16 [index_head_dim, W, B], rope_positions I32 [3, W, B];
 *   base_positions and valid_columns I32 [B].
 *
 * Supported domain:
 *   index_head_dim 128, B positive. base_positions nonnegative, valid_columns in [1,W], and every
 *   referenced record column in range.
 *
 * Numeric:
 *   The extraction copies BF16 words and I32 coordinates exactly; no floating-point evaluation is
 *   involved. It is compared with an exact oracle.
 *
 * Effects:
 *   Writes all four after.raw_keys columns and after.first_rope. before and records are read-only.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues on the given stream, allocates no device memory, keeps no persistent state and
 *   supports CUDA Graph capture and replay at bound addresses.
 */
void qsa_tail_extract(const QsaTailStateView& before, const QsaRawRecordReadView& records,
                      const Tensor& base_positions, const Tensor& valid_columns,
                      QsaTailStateView& after, cudaStream_t stream);

} // namespace ninfer::ops
