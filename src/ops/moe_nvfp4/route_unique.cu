#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kMaxExperts = 512;
static_assert(kMaxExperts == kMoeExperts);

__global__ void moe_nvfp4_route_probe_kernel(
    const std::int32_t* __restrict__ ids, std::int32_t entries,
    const std::int32_t* __restrict__ lookup, std::int32_t expert_begin,
    std::int32_t expert_end, std::int32_t* __restrict__ active_ids,
    std::int32_t* __restrict__ miss_ids, std::int32_t* __restrict__ active_count,
    std::int32_t* __restrict__ miss_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
    moe_nvfp4_probe_routes(ids, entries, lookup, expert_begin, expert_end, active_ids, miss_ids,
                           active_count, miss_count);
}

} // namespace

void moe_nvfp4_route_probe_launch(const std::int32_t* ids, std::int32_t entries,
                                  const std::int32_t* lookup, std::int32_t expert_begin,
                                  std::int32_t expert_end, std::int32_t* active_ids,
                                  std::int32_t* miss_ids, std::int32_t* active_count,
                                  std::int32_t* miss_count, cudaStream_t stream) {
    if (entries < 0 || entries > kMaxExperts) {
        throw std::invalid_argument("moe_nvfp4 probe: entries exceeds the expert count");
    }
    moe_nvfp4_route_probe_kernel<<<1, 1, 0, stream>>>(ids, entries, lookup, expert_begin,
                                                      expert_end, active_ids, miss_ids,
                                                      active_count, miss_count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
