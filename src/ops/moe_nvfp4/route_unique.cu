#include "ops/moe_nvfp4/moe_nvfp4.cuh"
#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cstdint>

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
    std::int32_t active = 0;
    std::int32_t miss   = 0;
    for (std::int32_t entry = 0; entry < entries; ++entry) {
        const std::int32_t id = ids[entry];
        if (id < expert_begin || id >= expert_end) { continue; }
        bool duplicate = false;
        for (std::int32_t other = 0; other < active; ++other) {
            if (active_ids[other] == id) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) { continue; }
        active_ids[active++] = id;
        if (lookup[id] < 0) { miss_ids[miss++] = id; }
    }
    __threadfence_system();
    *active_count = active;
    *miss_count   = miss;
}

} // namespace

void moe_nvfp4_route_probe_launch(const std::int32_t* ids, std::int32_t entries,
                                  const std::int32_t* lookup, std::int32_t expert_begin,
                                  std::int32_t expert_end, std::int32_t* active_ids,
                                  std::int32_t* miss_ids, std::int32_t* active_count,
                                  std::int32_t* miss_count, cudaStream_t stream) {
    moe_nvfp4_route_probe_kernel<<<1, 1, 0, stream>>>(ids, entries, lookup, expert_begin,
                                                      expert_end, active_ids, miss_ids,
                                                      active_count, miss_count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
