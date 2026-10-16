#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_cutlass_sm70.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4GdnInputRoute : std::uint8_t {
    A16,
    W4A4,
};

#ifdef NINFER_VOLTA_BUILD
// Same threshold the fp8 route measured on V100: at T>=33 the split-output cutlass projection
// (one dequant + one GEMM over the whole call) beats repeated <=32-token QPN/small_t launches.
constexpr std::int32_t kVoltaCutlassMinT = 33;
#endif

Nvfp4GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Nvfp4GdnInputRoute::A16; }
    if (policy == LinearPolicy::AllowA4) { return Nvfp4GdnInputRoute::W4A4; }
    throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported policy");
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                WorkspaceArena* workspace, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    if (x.ne[1] >= kVoltaCutlassMinT && workspace != nullptr) {
        nvfp4_gdn_input_cutlass_sm70_launch(x, weight, qkv, z, *workspace, stream);
        return;
    }
#endif
    constexpr std::int32_t kChunk   = kNvfp4LastSmallT;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* qkv_output =
            static_cast<std::uint8_t*>(qkv.data) +
            static_cast<std::int64_t>(token_begin) * kQkvRows * sizeof(std::uint16_t);
        auto* z_output = static_cast<std::uint8_t*>(z.data) +
                         static_cast<std::int64_t>(token_begin) * kZRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor qkv_chunk(qkv_output, DType::BF16, {kQkvRows, active});
        Tensor z_chunk(z_output, DType::BF16, {kZRows, active});
#ifdef NINFER_VOLTA_BUILD
        // Prefill chunks up to kNvfp4LastSmallT=32 fit the QPN tensor-core schedule; the
        // Nvfp4GdnInputOutput policy is token-indexed so active>1 chunks are valid. The SIMT
        // small_t kernel measures ~3.6 TFLOP/s at T=32 on V100 -- keep it only as fallback.
        if (nvfp4_volta_qpn_supported(weight.n, weight.k, active)) {
            nvfp4_gdn_input_decode_launch(input_chunk, weight, qkv_chunk, z_chunk, stream);
            continue;
        }
#endif
        if (active == 1) {
            nvfp4_gdn_input_decode_launch(input_chunk, weight, qkv_chunk, z_chunk, stream);
        } else {
            nvfp4_gdn_input_small_t_launch(input_chunk, weight, qkv_chunk, z_chunk, stream);
        }
    }
}

// The tp2 column shard -- same chunking discipline, halved row counts
// (Nvfp4GdnInputTp2ColumnGeometry's own qkv=5120=1024+1024+3072, z=3072).
void launch_a16_shard(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                      WorkspaceArena* workspace, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    if (x.ne[1] >= kVoltaCutlassMinT && workspace != nullptr) {
        nvfp4_gdn_input_cutlass_sm70_launch_shard(x, weight, qkv, z, *workspace, stream);
        return;
    }
#endif
    constexpr std::int32_t kChunk   = kNvfp4LastSmallT;
    constexpr std::int32_t kQkvRows = 5120;
    constexpr std::int32_t kZRows   = 3072;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* qkv_output =
            static_cast<std::uint8_t*>(qkv.data) +
            static_cast<std::int64_t>(token_begin) * kQkvRows * sizeof(std::uint16_t);
        auto* z_output = static_cast<std::uint8_t*>(z.data) +
                         static_cast<std::int64_t>(token_begin) * kZRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor qkv_chunk(qkv_output, DType::BF16, {kQkvRows, active});
        Tensor z_chunk(z_output, DType::BF16, {kZRows, active});
#ifdef NINFER_VOLTA_BUILD
        if (nvfp4_volta_qpn_supported(weight.n, weight.k, active)) {
            nvfp4_gdn_input_decode_launch_shard(input_chunk, weight, qkv_chunk, z_chunk, stream);
            continue;
        }
#endif
        if (active == 1) {
            nvfp4_gdn_input_decode_launch_shard(input_chunk, weight, qkv_chunk, z_chunk, stream);
        } else {
            nvfp4_gdn_input_small_t_launch_shard(input_chunk, weight, qkv_chunk, z_chunk, stream);
        }
    }
}

} // namespace

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    if (resolve_route(policy, max_tokens) == Nvfp4GdnInputRoute::W4A4) {
        return nvfp4_w4a4_workspace_capacity_bytes(max_tokens, Nvfp4GdnInputGeometry::kInputRows);
    }
#ifdef NINFER_VOLTA_BUILD
    if (max_tokens >= kVoltaCutlassMinT) {
        return nvfp4_gdn_input_cutlass_workspace_bytes(max_tokens);
    }
#endif
    return 0;
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4GdnInputRoute::A16) {
        launch_a16(x, weight, qkv, z, workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 gdn_input_proj requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_w4a4_launch(x, weight, qkv, z, scratch, stream);
}

// The tp2 column shard. The W4A4 activation-quantize workspace is a pure function of
// (tokens, K), and K=5120 is unchanged by the shard (only the output row count N halves) -- so
// nvfp4_gdn_input_workspace_capacity_bytes (tp1) is reused unchanged, the same rule
// attn_input_proj's own shard follows; no `_shard` capacity function exists or is needed.
void nvfp4_gdn_input_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    LinearPolicy policy, WorkspaceArena* workspace,
                                    cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4GdnInputRoute::A16) {
        launch_a16_shard(x, weight, qkv, z, workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 gdn_input_proj column-parallel requires caller "
                                    "workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_w4a4_launch_shard(x, weight, qkv, z, scratch, stream);
}

} // namespace ninfer::ops::detail
