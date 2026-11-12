#include "targets/qwen3_8_flash_next/impl/leaves.h"

#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

[[noreturn]] void not_implemented(const char* leaf) {
    throw std::logic_error(std::string("qwen3.8-flash-next: the ") + leaf +
                           " leaf has no implementation yet; it belongs to the M2 text leaf "
                           "milestone, the tasks that carry the Flash-Next text trunk");
}

} // namespace

void AttentionProjectionLeaf::project(const Tensor& hidden,
                                      const AttentionProjectionWeights& weights,
                                      const Tensor& positions, Tensor& query, Tensor& output_gate,
                                      Tensor& key, Tensor& value, Tensor& index_query,
                                      Tensor& index_key, WorkspaceArena& workspace,
                                      cudaStream_t stream) {
    (void)hidden;
    (void)weights;
    (void)positions;
    (void)query;
    (void)output_gate;
    (void)key;
    (void)value;
    (void)index_query;
    (void)index_key;
    (void)workspace;
    (void)stream;
    not_implemented("attention projection");
}

std::size_t AttentionProjectionLeaf::workspace_capacity_bytes(std::int32_t tokens) {
    (void)tokens;
    not_implemented("attention projection");
}

void GdnProjectionLeaf::project(const Tensor& hidden, const GdnProjectionWeights& weights,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z, Tensor& a,
                                Tensor& b, WorkspaceArena& workspace, cudaStream_t stream) {
    (void)hidden;
    (void)weights;
    (void)query;
    (void)key;
    (void)value;
    (void)z;
    (void)a;
    (void)b;
    (void)workspace;
    (void)stream;
    not_implemented("GDN projection and control");
}

std::size_t GdnProjectionLeaf::workspace_capacity_bytes(std::int32_t tokens) {
    (void)tokens;
    not_implemented("GDN projection and control");
}

void PostMixerLeaf::mix(const Tensor& hidden, const MoeWeights& weights, Tensor& output,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    (void)hidden;
    (void)weights;
    (void)output;
    (void)workspace;
    (void)stream;
    not_implemented("post-mixer");
}

std::size_t PostMixerLeaf::workspace_capacity_bytes(std::int32_t tokens) {
    (void)tokens;
    not_implemented("post-mixer");
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
