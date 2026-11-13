#include "ninfer/ops/linear_split.h"

#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct RowSplitPlanes {
    std::uint64_t code_row  = 0;
    std::uint64_t high_row  = 0;
    std::uint64_t scale_row = 0;
};

int row_split_bits(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return 4;
    case QType::Q5G64_F16S:
        return 5;
    case QType::Q6G64_F16S:
        return 6;
    case QType::W8G32_F16S:
        return 8;
    default:
        throw std::invalid_argument("linear_split: unsupported row-split weight format");
    }
}

RowSplitPlanes row_split_planes(const Weight& w) {
    const std::int32_t group = w.group_size != 0 ? static_cast<std::int32_t>(w.group_size)
                                                 : w.group;
    if (group <= 0 || w.padded_shape[1] <= 0 || (w.padded_shape[1] % group) != 0) {
        throw std::invalid_argument("linear_split: inconsistent row-split group geometry");
    }
    const std::int32_t groups_per_row = w.padded_shape[1] / group;
    const int bits                    = row_split_bits(w.qtype);
    RowSplitPlanes planes;
    planes.code_row  = static_cast<std::uint64_t>(groups_per_row) * (bits == 8 ? group : group / 2);
    planes.high_row  = bits <= 4 ? 0
                                 : static_cast<std::uint64_t>(groups_per_row) *
                                       static_cast<std::uint64_t>(group * (bits - 4) / 8);
    planes.scale_row = static_cast<std::uint64_t>(groups_per_row) * sizeof(std::uint16_t);
    return planes;
}

void validate_parent(const Tensor& x, const Weight& w, std::span<const Tensor> outputs,
                     std::span<const LinearSplitSegment> segments) {
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != w.k || x.ne[1] <= 0 ||
        x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear_split: x must be contiguous BF16 [K,T]");
    }
    if (outputs.size() != segments.size() || segments.empty()) {
        throw std::invalid_argument("linear_split: outputs and segments must align and be non-empty");
    }
    if (w.n <= 0 || w.k <= 0 || w.qdata == nullptr) {
        throw std::invalid_argument("linear_split: parent weight is incomplete");
    }

    const bool contiguous_bf16 =
        w.qtype == QType::BF16_CTRL && w.layout == QuantLayout::Contiguous;
    const bool row_split =
        w.layout == QuantLayout::RowSplit &&
        (w.qtype == QType::Q4G64_F16S || w.qtype == QType::Q5G64_F16S ||
         w.qtype == QType::Q6G64_F16S || w.qtype == QType::W8G32_F16S);
    if (!contiguous_bf16 && !row_split) {
        throw std::invalid_argument(
            "linear_split: supports only BF16 contiguous or Q4/Q5/Q6/W8 row-split parents");
    }
    if (row_split) { (void)row_split_planes(w); }

    for (std::size_t index = 0; index < segments.size(); ++index) {
        const LinearSplitSegment& segment = segments[index];
        const Tensor& out                 = outputs[index];
        if (segment.row_begin < 0 || segment.row_count <= 0 ||
            segment.row_begin > w.n - segment.row_count) {
            throw std::invalid_argument("linear_split: segment rows out of range");
        }
        if (out.dtype != DType::BF16 || !out.is_contiguous() ||
            out.ne[0] != segment.row_count || out.ne[1] != x.ne[1] || out.ne[2] != 1 ||
            out.ne[3] != 1 || out.data == nullptr) {
            throw std::invalid_argument(
                "linear_split: every output must be contiguous BF16 [row_count,T]");
        }
    }
}

Weight segment_weight(const Weight& parent, const LinearSplitSegment& segment) {
    Weight sub = parent;
    sub.n      = segment.row_count;
    sub.shape[0]        = segment.row_count;
    sub.padded_shape[0] = segment.row_count;
    if (parent.layout == QuantLayout::Contiguous) {
        sub.qdata = static_cast<const std::uint8_t*>(parent.qdata) +
                    static_cast<std::size_t>(segment.row_begin) * parent.k * sizeof(std::uint16_t);
        sub.payload = sub.qdata;
        return sub;
    }

    const RowSplitPlanes planes = row_split_planes(parent);
    sub.qdata = static_cast<const std::uint8_t*>(parent.qdata) +
                static_cast<std::size_t>(segment.row_begin) * planes.code_row;
    sub.qhigh = planes.high_row == 0
                    ? nullptr
                    : static_cast<const std::uint8_t*>(parent.qhigh) +
                          static_cast<std::size_t>(segment.row_begin) * planes.high_row;
    sub.scales = static_cast<const std::uint8_t*>(parent.scales) +
                 static_cast<std::size_t>(segment.row_begin) * planes.scale_row;
    return sub;
}

} // namespace

std::size_t linear_split_workspace_capacity_bytes(
    QType qtype, std::int32_t parent_rows, std::int32_t input_rows,
    std::span<const LinearSplitSegment> segments, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens) {
    if (parent_rows <= 0 || input_rows <= 0 || segments.empty()) {
        throw std::invalid_argument("linear_split workspace: invalid parent geometry");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear_split workspace: invalid token interval");
    }
    if (qtype != QType::Q4G64_F16S && qtype != QType::Q5G64_F16S &&
        qtype != QType::Q6G64_F16S && qtype != QType::W8G32_F16S &&
        qtype != QType::BF16_CTRL) {
        throw std::invalid_argument("linear_split workspace: unsupported weight format");
    }
    std::size_t capacity = 0;
    for (const LinearSplitSegment& segment : segments) {
        if (segment.row_begin < 0 || segment.row_count <= 0 ||
            segment.row_begin > parent_rows - segment.row_count) {
            throw std::invalid_argument("linear_split workspace: segment rows out of range");
        }
        capacity = std::max(
            capacity, linear_workspace_capacity_bytes(qtype, segment.row_count, input_rows, policy,
                                                      min_tokens, max_tokens));
    }
    return capacity;
}

void linear_split(const Tensor& x, const Weight& w, std::span<const LinearSplitSegment> segments,
                  std::span<Tensor> outputs, LinearPolicy policy, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    validate_parent(x, w, outputs, segments);
    for (std::size_t index = 0; index < segments.size(); ++index) {
        const Weight sub = segment_weight(w, segments[index]);
        const auto scope = workspace.scope();
        linear(x, sub, outputs[index], policy, workspace, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
