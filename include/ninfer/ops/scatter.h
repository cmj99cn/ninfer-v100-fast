#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: scatter
 *
 * Math / indexing:
 *   destination[:, indices[i]] = source[:, i] for 0 <= i < V.
 *
 * Logical shapes:
 *   BF16 source [D,V], I32 indices [V], BF16 destination [D,T], all contiguous.
 *
 * Numeric:
 *   Exact BF16 element copies. Callers provide indices in [0,T) with no duplicates when
 *   deterministic overwrite order is required.
 *
 * Effects:
 *   Overwrites only the selected destination columns; all other columns retain their old values.
 *   Inputs and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect beyond the selected destination writes.
 *
 * Supported routes:
 *   D divisible by eight with 16-byte-aligned source/destination uses BF16x8 copies; other aligned
 *   even D uses BF16x2; remaining contiguous dimensions use scalar copies.
 */
void scatter(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream);

/**
 * Scatter compact request-major BF16 blocks into lane-owned fixed storage.
 *
 * source is contiguous [D,W,B], lanes and valid_columns are contiguous I32 [B], and destination
 * is a possibly row-sliced [D,W,C] view. For each b and j<valid_columns[b]:
 *
 *   destination[:,j,lanes[b]] = source[:,j,b]
 *
 * Columns j>=valid_columns[b] and every unselected lane remain unchanged. The caller guarantees
 * 0<=valid_columns[b]<=W and 0<=lanes[b]<C. D is divisible by eight and all column starts are
 * 16-byte aligned. Inputs and destination do not alias.
 */
void scatter_bf16_batch(const Tensor& source, const Tensor& lanes, const Tensor& valid_columns,
                        Tensor& destination, cudaStream_t stream);

/**
 * Op: extract_bf16_columns
 *
 * Math / indexing:
 *   For source [D,T] and destination [D',T]:
 *   destination[d,t] = source[source_column + d,t], 0<=d<D', 0<=t<T.
 *
 * Logical shapes:
 *   Contiguous BF16 or FP16 source [D,T] and destination [D',T], sharing one dtype. Despite the
 *   historical parameter name, source_column is an offset in the fastest (ne[0]) dimension;
 *   0<=source_column and source_column+D'<=D.
 *
 * Numeric:
 *   Exact activation-word copies.
 *
 * Effects:
 *   Writes the full destination; source and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void extract_bf16_columns(const Tensor& source, std::int32_t source_column, Tensor& destination,
                          cudaStream_t stream);

/**
 * Fuse the decode GDN staging copies into one launch:
 *   conv_in[0:key_width]            = q
 *   conv_in[key_width:2*key_width]  = k
 *   conv_in[2*key_width:channels]   = v
 *   g_rank                          = g[rank*value_heads:(rank+1)*value_heads]
 *   beta_rank                       = beta[rank*value_heads:(rank+1)*value_heads]
 * All tensors are contiguous with T=1. Copies are exact activation/FP32 word moves.
 */
void stage_gdn_decode(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, std::int32_t rank, Tensor& conv_in, Tensor& g_rank,
                      Tensor& beta_rank, cudaStream_t stream);

/**
 * Op: stage_gdn_span
 *
 * Fuse the multi-token GDN staging copies into one launch. For every t in [0,T):
 *   conv_in[0:key_width,t]            = q[:,t]
 *   conv_in[key_width:2*key_width,t]  = k[:,t]
 *   conv_in[2*key_width:channels,t]   = v[:,t]
 *   g_rank[:,t]                       = g[rank*value_heads:(rank+1)*value_heads,t]
 *   beta_rank[:,t]                    = beta[rank*value_heads:(rank+1)*value_heads,t]
 *
 * Logical shapes: contiguous FP16 q/k [key_width,T], v [channels-2*key_width,T],
 * conv_in [channels,T]; contiguous FP32 g/beta [g_heads,T] and g_rank/beta_rank
 * [value_heads,T]. T>=2; the T=1 route is stage_gdn_decode.
 *
 * Numeric: exact activation/FP32 word moves, bit-identical to the five replaced
 * cudaMemcpy2DAsync calls. No persistent state; inputs are not modified.
 */
void stage_gdn_span(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, std::int32_t rank, Tensor& conv_in, Tensor& g_rank,
                    Tensor& beta_rank, cudaStream_t stream);

/**
 * Op: extract_gdn_columns
 *
 * Fuse the GDN post-convolution extracts into one launch. For every t in [0,T):
 *   q[:,t] = conv_out[0:key_width,t]
 *   k[:,t] = conv_out[key_width:2*key_width,t]
 *   v[:,t] = conv_out[2*key_width:channels,t]
 *
 * Logical shapes: contiguous FP16 conv_out [channels,T], q/k [key_width,T], and
 * v [channels-2*key_width,T]. T>=1; layouts and strides are unchanged from three
 * separate extract_bf16_columns calls.
 *
 * Numeric: exact activation word moves. The source and destinations must not alias.
 */
void extract_gdn_columns(const Tensor& source, Tensor& q, Tensor& k, Tensor& v,
                         cudaStream_t stream);

} // namespace ninfer::ops
