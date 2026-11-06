#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <class E>
struct ElemDtype;
template <>
struct ElemDtype<__nv_bfloat16> {
    static constexpr DType value = DType::BF16;
};
template <>
struct ElemDtype<half> {
    static constexpr DType value = DType::FP16;
};

template <class Geometry>
struct CutlassSections;

template <>
struct CutlassSections<Nvfp4GdnInputGeometry> {
    static constexpr std::int32_t kQkvRows = Nvfp4GdnInputOutput::kQkvRows;
    static constexpr std::int32_t kZRows   = Nvfp4GdnInputOutput::kZRows;
};

template <>
struct CutlassSections<Nvfp4GdnInputTp2ColumnGeometry> {
    static constexpr std::int32_t kQkvRows = 5120;
    static constexpr std::int32_t kZRows   = 3072;
};

template <class Geometry, class E, class Allocator>
Tensor allocate_projected(Allocator& allocator, int tokens) {
    return allocator.alloc(ElemDtype<E>::value, {Geometry::kOutputRows, tokens});
}

template <class Geometry, class E>
__global__ void split_gdn_output(const E* __restrict__ projected, E* __restrict__ qkv,
                                 E* __restrict__ z, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    using Sections = CutlassSections<Geometry>;
    const int row   = static_cast<int>(i % Geometry::kOutputRows);
    const int token = static_cast<int>(i / Geometry::kOutputRows);
    if (row < Sections::kQkvRows) {
        qkv[static_cast<std::int64_t>(token) * Sections::kQkvRows + row] = projected[i];
    } else {
        z[static_cast<std::int64_t>(token) * Sections::kZRows + row - Sections::kQkvRows] =
            projected[i];
    }
}

template <class Geometry, class E>
std::size_t workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected<Geometry, E>(layout, tokens);
    if constexpr (std::is_same_v<E, half>) {
        return layout.peak_bytes(1) +
               nvfp4_cutlass_sm70_f16_workspace_bytes(Geometry::kOutputRows, Geometry::kInputRows,
                                                      tokens);
    } else {
        return layout.peak_bytes(1) +
               nvfp4_cutlass_sm70_workspace_bytes(Geometry::kOutputRows, Geometry::kInputRows,
                                                  tokens);
    }
}

template <class Geometry, class E>
void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
            WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected<Geometry, E>(workspace, x.ne[1]);
    if constexpr (std::is_same_v<E, half>) {
        nvfp4_cutlass_sm70_f16_launch(x, weight, projected, workspace, stream);
    } else {
        nvfp4_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    }
    const std::int64_t count = static_cast<std::int64_t>(x.ne[1]) * Geometry::kOutputRows;
    split_gdn_output<Geometry, E><<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const E*>(projected.data), static_cast<E*>(qkv.data),
        static_cast<E*>(z.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t nvfp4_gdn_input_cutlass_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4GdnInputGeometry, __nv_bfloat16>(tokens);
}

std::size_t nvfp4_gdn_input_cutlass_shard_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4GdnInputTp2ColumnGeometry, __nv_bfloat16>(tokens);
}

std::size_t nvfp4_gdn_input_cutlass_f16_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4GdnInputGeometry, half>(tokens);
}

std::size_t nvfp4_gdn_input_cutlass_f16_shard_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4GdnInputTp2ColumnGeometry, half>(tokens);
}

void nvfp4_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    launch<Nvfp4GdnInputGeometry, __nv_bfloat16>(x, weight, qkv, z, workspace, stream);
}

void nvfp4_gdn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                               Tensor& z, WorkspaceArena& workspace,
                                               cudaStream_t stream) {
    launch<Nvfp4GdnInputTp2ColumnGeometry, __nv_bfloat16>(x, weight, qkv, z, workspace, stream);
}

void nvfp4_gdn_input_cutlass_f16_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                        Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    launch<Nvfp4GdnInputGeometry, half>(x, weight, qkv, z, workspace, stream);
}

void nvfp4_gdn_input_cutlass_f16_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                              Tensor& z, WorkspaceArena& workspace,
                                              cudaStream_t stream) {
    launch<Nvfp4GdnInputTp2ColumnGeometry, half>(x, weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail
