#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kMaxExperts = 512;

__global__ void moe_nvfp4_route_unique_kernel(
    const std::int32_t* __restrict__ ids, std::int32_t entries,
    std::int32_t* __restrict__ unique_ids, std::int32_t* __restrict__ unique_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
    std::int32_t seen[kMaxExperts];
    std::int32_t count = 0;
    for (std::int32_t entry = 0; entry < entries; ++entry) {
        const std::int32_t id = ids[entry];
        bool duplicate = false;
        for (std::int32_t other = 0; other < count; ++other) {
            if (seen[other] == id) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) { continue; }
        seen[count++] = id;
    }
    for (std::int32_t index = 0; index < count; ++index) { unique_ids[index] = seen[index]; }
    __threadfence_system();
    *unique_count = count;
}

} // namespace

void moe_nvfp4_route_unique_launch(const std::int32_t* ids, std::int32_t entries,
                                   std::int32_t* unique_ids, std::int32_t* unique_count,
                                   cudaStream_t stream) {
    moe_nvfp4_route_unique_kernel<<<1, 1, 0, stream>>>(ids, entries, unique_ids, unique_count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
