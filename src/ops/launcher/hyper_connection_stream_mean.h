#pragma once

// ninfer::ops::detail - private launch prototype for hyper_connection_stream_mean. Included by
// the wrapper (host) and defined by the launcher (.cu). Not part of the public api.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void hyper_connection_stream_mean_launch(const Tensor& input, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
