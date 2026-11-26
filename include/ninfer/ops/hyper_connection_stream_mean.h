#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Op: hyper_connection_stream_mean
 *
 * Reduces the hyper-connection streams of one tapped residual block to their mean:
 *
 *   ideal[h,t,b] = (1/S) * sum_s input[h,s,t,b].
 *
 * `input` is a BF16 or FP16 tensor `[H,S,T,B]`; `out` has the same element dtype and shape
 * `[H,T,B]` (its fourth extent is 1). Every extent is taken from `ne`; `nb` is honoured, so
 * `out` may be a strided column block of a wider feature buffer. Only the first `ne[0]`
 * elements of each row are written; trailing padding is untouched. The oracle evaluates the
 * mean in FP64 from the represented input words. The updated output is promoted and compared
 * with that result; output storage rounding belongs to the Op's numerical criterion.
 *
 * The Op uses no workspace or persistent state.
 */
void hyper_connection_stream_mean(const Tensor& input, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
