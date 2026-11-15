#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/op_tester.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

using namespace ninfer::test;

namespace {

__device__ float2 reference_e2m1(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ float reference_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

__global__ void compare_codes(std::uint32_t* e2m1_failures, std::uint32_t* e4m3_failures) {
    const int code = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (code >= 256) { return; }
    const auto storage = static_cast<std::uint8_t>(code);
    const float2 expected = reference_e2m1(storage);
    const float2 actual   = ninfer::ops::detail::decode_nvfp4_e2m1x2(storage);
    if (__float_as_uint(expected.x) != __float_as_uint(actual.x) ||
        __float_as_uint(expected.y) != __float_as_uint(actual.y)) {
        atomicAdd(e2m1_failures, 1U);
    }
    const float expected_scale = reference_e4m3(storage);
    const float actual_scale   = ninfer::ops::detail::decode_nvfp4_e4m3(storage);
    if (__float_as_uint(expected_scale) != __float_as_uint(actual_scale)) {
        atomicAdd(e4m3_failures, 1U);
    }
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    std::uint32_t* device_e2m1 = nullptr;
    std::uint32_t* device_e4m3 = nullptr;
    cuda_check(cudaMalloc(&device_e2m1, sizeof(std::uint32_t)), "cudaMalloc");
    cuda_check(cudaMalloc(&device_e4m3, sizeof(std::uint32_t)), "cudaMalloc");
    cuda_check(cudaMemset(device_e2m1, 0, sizeof(std::uint32_t)), "cudaMemset");
    cuda_check(cudaMemset(device_e4m3, 0, sizeof(std::uint32_t)), "cudaMemset");
    compare_codes<<<1, 256>>>(device_e2m1, device_e4m3);
    cuda_check(cudaGetLastError(), "compare_codes launch");
    cuda_synchronize();
    std::uint32_t host_e2m1 = 0;
    std::uint32_t host_e4m3 = 0;
    cuda_check(cudaMemcpy(&host_e2m1, device_e2m1, sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    cuda_check(cudaMemcpy(&host_e4m3, device_e4m3, sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    cuda_check(cudaFree(device_e2m1), "cudaFree");
    cuda_check(cudaFree(device_e4m3), "cudaFree");
    if (host_e2m1 != 0 || host_e4m3 != 0) {
        std::cerr << "nvfp4 codec: " << host_e2m1 << " E2M1 bytes and " << host_e4m3
                  << " E4M3 codes differ from the reference conversion\n";
        return 1;
    }
    std::cout << "OK nvfp4 codec\n";
    return 0;
}
