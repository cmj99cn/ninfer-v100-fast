// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

#include "ninfer/ops/candidate_selector.h"
#include <cmath>

#include "ops/launcher/candidate_selector.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp = "candidate_selector";

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t d0, std::int32_t d1,
                    std::int32_t d2, std::int32_t d3, const char* label) {
    if (tensor.dtype != dtype || tensor.ne[0] != d0 || tensor.ne[1] != d1 || tensor.ne[2] != d2 ||
        tensor.ne[3] != d3 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + label);
    }
}

std::size_t topk_workspace_bytes(std::int32_t columns) {
    return static_cast<std::size_t>(columns) * 32 * kCandidateCount *
           (sizeof(float) + sizeof(std::int32_t));
}

std::size_t selector_workspace_bytes(std::int32_t steps, std::int32_t batch) {
    if (steps <= 4) { return 0; }
    return static_cast<std::size_t>(steps) * batch * kCandidateCount * kCandidateCount *
           sizeof(float);
}

void require_positions(const Tensor& tensor, std::int32_t d0, std::int32_t d1, const char* label) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != d0 || tensor.ne[1] != d1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, alignof(std::int32_t))) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + label);
    }
}

} // namespace

std::size_t candidate_topk_workspace_capacity_bytes(std::int32_t columns) {
    if (columns <= 0) {
        throw std::invalid_argument("candidate_topk: columns must be positive");
    }
    return topk_workspace_bytes(columns) + 256;
}

void candidate_topk(const Tensor& logits, std::int32_t valid_rows, Tensor& candidate_ids,
                    Tensor& candidate_scores, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t rows    = logits.ne[0];
    const std::int32_t columns = logits.ne[1];
    if (logits.data == nullptr || !logits.is_contiguous() || logits.ne[2] != 1 ||
        logits.ne[3] != 1 || (logits.dtype != DType::BF16 && logits.dtype != DType::FP32)) {
        throw std::invalid_argument("candidate_topk: logits must be contiguous BF16 or FP32 [rows,U]");
    }
    if (columns <= 0 || valid_rows < kCandidateCount || valid_rows > rows) {
        throw std::invalid_argument("candidate_topk: U must be positive and valid_rows in [16,rows]");
    }
    require_tensor(candidate_ids, DType::I32, kCandidateCount, columns, 1, 1, "candidate_ids");
    require_tensor(candidate_scores, DType::FP32, kCandidateCount, columns, 1, 1,
                   "candidate_scores");
    auto scope = workspace.scope();
    const DeviceSpan scratch = workspace.alloc_bytes(topk_workspace_bytes(columns));
    detail::candidate_topk_launch(logits, valid_rows, candidate_ids, candidate_scores, scratch,
                                  stream);
}

std::size_t candidate_selector_path_workspace_capacity_bytes(int min_steps, int max_steps,
                                                             int min_batch, int max_batch) {
    if (min_steps < 1 || max_steps > 15 || max_steps < min_steps || min_batch < 1 ||
        max_batch > 8 || max_batch < min_batch) {
        throw std::invalid_argument("candidate_selector_path workspace: invalid K/B interval");
    }
    std::size_t capacity = 0;
    for (int k = min_steps; k <= max_steps; ++k) {
        for (int b = min_batch; b <= max_batch; ++b) {
            capacity = std::max(capacity, selector_workspace_bytes(k, b));
        }
    }
    return capacity;
}

void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                             cudaStream_t stream, Tensor* confidence, float edge_softcap) {
    if (!(edge_softcap >= 0.0f) || !std::isfinite(edge_softcap)) {
        throw std::invalid_argument(
            "candidate_selector_path: edge_softcap must be finite and nonnegative");
    }
    const std::int32_t steps = candidate_ids.ne[1];
    const std::int32_t batch = candidate_ids.ne[2];
    if (steps < 1 || steps > 15) {
        throw std::invalid_argument("candidate_selector_path: K must be in [1,15]");
    }
    if (batch < 1 || batch > 8) {
        throw std::invalid_argument("candidate_selector_path: B must be in [1,8]");
    }
    require_tensor(candidate_ids, DType::I32, kCandidateCount, steps, batch, 1, "candidate_ids");
    require_tensor(unary_scores, DType::FP32, kCandidateCount, steps, batch, 1, "unary_scores");
    require_tensor(projected_hidden, DType::BF16, kSelectorRank, steps, batch, 1,
                   "projected_hidden");
    require_positions(anchors, batch, 1, "anchors");
    require_positions(base_positions, batch, 1, "base_positions");
    require_tensor(drafts, DType::I32, steps, batch, 1, 1, "drafts");
    require_tensor(proposal_q, DType::FP32, kCandidateCount, steps, batch, 1, "proposal_q");
    if (confidence != nullptr) {
        require_tensor(*confidence, DType::FP32, steps, batch, 1, 1, "confidence");
    }
    if (configs == nullptr || !aligned_to(configs, alignof(SamplingConfig))) {
        throw std::invalid_argument("candidate_selector_path: invalid configs");
    }
    const std::int32_t codebook_rows = predecessor_codebook.ne[0];
    if (codebook_rows < kCandidateCount || predecessor_codebook.ne[1] != kSelectorRank ||
        successor_codebook.ne[0] != codebook_rows ||
        successor_codebook.ne[1] != kSelectorRank) {
        throw std::invalid_argument(
            "candidate_selector_path: codebooks must be BF16 [rows,256] with equal rows");
    }
    if (predecessor_codebook.dtype != DType::BF16 || successor_codebook.dtype != DType::BF16 ||
        predecessor_codebook.ne[2] != 1 || predecessor_codebook.ne[3] != 1 ||
        successor_codebook.ne[2] != 1 || successor_codebook.ne[3] != 1) {
        throw std::invalid_argument("candidate_selector_path: invalid codebook dtype/shape");
    }
    if (!predecessor_codebook.is_contiguous() || !successor_codebook.is_contiguous() ||
        !aligned_to(predecessor_codebook.data, 16) || !aligned_to(successor_codebook.data, 16)) {
        throw std::invalid_argument("candidate_selector_path: codebooks must be contiguous");
    }

    auto scope = workspace.scope();
    DeviceSpan scratch{};
    const std::size_t bytes = selector_workspace_bytes(steps, batch);
    if (bytes != 0) { scratch = workspace.alloc_bytes(bytes); }
    detail::candidate_selector_path_launch(
        candidate_ids, unary_scores, projected_hidden, anchors, predecessor_codebook,
        successor_codebook, base_positions, configs, drafts, proposal_q,
        confidence != nullptr ? static_cast<float*>(confidence->data) : nullptr, edge_softcap,
        scratch, stream);
}

} // namespace ninfer::ops
