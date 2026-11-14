#include "ninfer/ops/moe_nvfp4.h"

#include "ops/moe_nvfp4/moe_nvfp4_launch.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kExperts      = 512;
constexpr std::int32_t kTopK         = 10;
constexpr std::int32_t kHidden       = 2560;
constexpr std::int32_t kIntermediate = 640;
constexpr std::int32_t kParentRows   = 1793;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end   = 0;
    std::string name;
};

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

std::size_t align_up_256(std::size_t value) { return ((value + 255U) / 256U) * 256U; }

AddressRange address_range(const void* pointer, std::size_t bytes, std::string name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument("moe_nvfp4: " + name + " storage must be non-empty");
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("moe_nvfp4: " + name + " address range overflows");
    }
    return {begin, begin + bytes, std::move(name)};
}

void require_disjoint(const std::vector<AddressRange>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (ranges[i].begin < ranges[j].end && ranges[j].begin < ranges[i].end) {
                throw std::invalid_argument("moe_nvfp4: " + ranges[i].name + " overlaps " +
                                            ranges[j].name);
            }
        }
    }
}

std::int32_t require_tokens(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kHidden || tensor.ne[1] < 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("moe_nvfp4: invalid ") + name);
    }
    return tensor.ne[1];
}

void require_output(const Tensor& tensor, DType dtype, std::int32_t n, std::int32_t tokens,
                    const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n || tensor.ne[1] != tokens ||
        !tensor.is_contiguous() || tensor.data == nullptr || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("moe_nvfp4: invalid ") + name);
    }
}

void require_vector(const Tensor& tensor, DType dtype, std::int32_t tokens, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != tokens || tensor.ne[1] != 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("moe_nvfp4: invalid ") + name);
    }
}

void require_parent(const Weight& router_shared, std::vector<AddressRange>& ranges) {
    if (router_shared.qtype != QType::BF16_CTRL || router_shared.layout != QuantLayout::Contiguous ||
        router_shared.ndim != 2 || router_shared.n != kParentRows ||
        router_shared.k != kHidden || router_shared.shape[0] != kParentRows ||
        router_shared.shape[1] != kHidden || router_shared.qdata == nullptr ||
        router_shared.qhigh != nullptr || router_shared.scales != nullptr ||
        router_shared.payload_bytes < static_cast<std::size_t>(kParentRows) * kHidden * 2U ||
        !aligned_to(router_shared.qdata, 16)) {
        throw std::invalid_argument("moe_nvfp4: router_shared must be aligned contiguous BF16 "
                                    "[1793, 2560]");
    }
    ranges.push_back(address_range(router_shared.qdata, static_cast<std::size_t>(kParentRows) * kHidden * 2U,
                                   "router_shared"));
}

void require_shared_down(const Weight& shared_down, std::vector<AddressRange>& ranges) {
    if (shared_down.qtype != QType::BF16_CTRL || shared_down.layout != QuantLayout::Contiguous ||
        shared_down.ndim != 2 || shared_down.n != kHidden || shared_down.k != kIntermediate ||
        shared_down.shape[0] != kHidden || shared_down.shape[1] != kIntermediate ||
        shared_down.qdata == nullptr || shared_down.qhigh != nullptr ||
        shared_down.scales != nullptr ||
        shared_down.payload_bytes < static_cast<std::size_t>(kHidden) * kIntermediate * 2U ||
        !aligned_to(shared_down.qdata, 16)) {
        throw std::invalid_argument("moe_nvfp4: shared_down must be aligned contiguous BF16 "
                                    "[2560, 640]");
    }
    ranges.push_back(address_range(shared_down.qdata, static_cast<std::size_t>(kHidden) * kIntermediate * 2U,
                                   "shared_down"));
}

void require_batched(const BatchedNvfp4Weight& weight, std::int32_t rows, std::int32_t columns,
                     const char* name, std::vector<AddressRange>& ranges) {
    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BatchedBlockScaleK16M128x4 ||
        weight.group_size != 16U || weight.matrices < 1 || weight.matrices > kExperts ||
        weight.expert_begin < 0 || weight.expert_begin > kExperts - weight.matrices ||
        weight.rows != rows || weight.columns != columns || (rows % 128) != 0 ||
        (columns % 64) != 0 || weight.payload == nullptr || weight.codes == nullptr ||
        weight.scales == nullptr || weight.divisors == nullptr) {
        throw std::invalid_argument(std::string("moe_nvfp4: invalid batched NVFP4 ") + name);
    }
    const std::size_t matrices = static_cast<std::size_t>(weight.matrices);
    const std::size_t code_bytes  = matrices * rows * columns / 2U;
    const std::size_t scale_bytes = matrices * rows * columns / 16U;
    const std::size_t scale_offset = align_up_256(code_bytes);
    if (weight.payload_bytes < scale_offset + scale_bytes + 4U * matrices) {
        throw std::invalid_argument(std::string("moe_nvfp4: invalid payload size for ") + name);
    }
    const auto* payload = static_cast<const std::uint8_t*>(weight.payload);
    if (static_cast<const void*>(weight.codes) != weight.payload ||
        static_cast<const void*>(weight.scales) != payload + scale_offset ||
        reinterpret_cast<const std::uint8_t*>(weight.divisors) != payload + scale_offset + scale_bytes) {
        throw std::invalid_argument(std::string("moe_nvfp4: plane bases of ") + name +
                                    " must follow the batched layout");
    }
    if (!aligned_to(weight.codes, 16) || !aligned_to(weight.scales, 16) ||
        !aligned_to(weight.divisors, 16)) {
        throw std::invalid_argument(std::string("moe_nvfp4: batched planes of ") + name +
                                    " must be 16-byte aligned");
    }
    ranges.push_back(address_range(weight.codes, code_bytes, std::string(name) + " code"));
    ranges.push_back(address_range(weight.scales, scale_bytes, std::string(name) + " scales"));
    ranges.push_back(address_range(weight.divisors, 4U * matrices, std::string(name) + " divisors"));
}

const BatchedNvfp4Weight& require_expert_bank(const MoeNvfp4Weights& weights,
                                              std::vector<AddressRange>& ranges) {
    require_parent(weights.router_shared, ranges);
    require_shared_down(weights.shared_down, ranges);
    require_batched(weights.routed_gate_up, 2 * kIntermediate, kHidden, "routed_gate_up", ranges);
    require_batched(weights.routed_down, kHidden, kIntermediate, "routed_down", ranges);
    if (weights.routed_gate_up.expert_begin != weights.routed_down.expert_begin ||
        weights.routed_gate_up.matrices != weights.routed_down.matrices) {
        throw std::invalid_argument("moe_nvfp4: the routed expert banks disagree on their range");
    }
    return weights.routed_gate_up;
}

void validate_weights(const MoeNvfp4Weights& weights, std::vector<AddressRange>& ranges) {
    (void)require_expert_bank(weights, ranges);
}

void require_route(const MoeNvfp4RouteView& route, std::int32_t tokens,
                   std::vector<AddressRange>& ranges) {
    require_output(route.expert_ids, DType::I32, kTopK, tokens, "expert_ids");
    require_output(route.coefficients, DType::FP32, kTopK, tokens, "coefficients");
    require_vector(route.shared_gate, DType::FP32, tokens, "shared_gate");
    ranges.push_back(address_range(route.expert_ids.data, route.expert_ids.bytes(), "expert_ids"));
    ranges.push_back(address_range(route.coefficients.data, route.coefficients.bytes(),
                                   "coefficients"));
    ranges.push_back(address_range(route.shared_gate.data, route.shared_gate.bytes(),
                                   "shared_gate"));
}

detail::MoeNvfp4RouteOutputs route_outputs(const MoeNvfp4RouteView& route) {
    return {static_cast<std::int32_t*>(route.expert_ids.data),
            static_cast<float*>(route.coefficients.data),
            static_cast<float*>(route.shared_gate.data)};
}

detail::MoeNvfp4BatchedPlanes planes_of(const BatchedNvfp4Weight& weight) {
    return {static_cast<const std::uint8_t*>(weight.codes),
            static_cast<const std::uint8_t*>(weight.scales), weight.divisors};
}

std::size_t route_workspace_bytes(std::int32_t tokens) {
    return align_up_256(static_cast<std::size_t>(tokens) * (4U * kTopK + 4U * kTopK + 4U));
}

detail::MoeNvfp4RouteOutputs route_outputs(std::uint8_t* base, std::int32_t tokens) {
    detail::MoeNvfp4RouteOutputs outputs;
    outputs.ids          = reinterpret_cast<std::int32_t*>(base);
    outputs.coefficients = reinterpret_cast<float*>(base + 4U * kTopK * tokens);
    outputs.shared_gate  = reinterpret_cast<float*>(base + 8U * kTopK * tokens);
    return outputs;
}

} // namespace

std::size_t moe_nvfp4_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("moe_nvfp4 workspace: invalid token interval");
    }
    return route_workspace_bytes(max_tokens);
}

void moe_nvfp4_route(const Tensor& x, const MoeNvfp4Weights& weights, Tensor& expert_ids,
                     Tensor& coefficients, Tensor& shared_gate, cudaStream_t stream) {
    const std::int32_t tokens = require_tokens(x, "x");
    require_output(expert_ids, DType::I32, kTopK, tokens, "expert_ids");
    require_output(coefficients, DType::FP32, kTopK, tokens, "coefficients");
    require_vector(shared_gate, DType::FP32, tokens, "shared_gate");

    std::vector<AddressRange> ranges;
    ranges.reserve(12);
    ranges.push_back(address_range(x.data, x.bytes(), "x"));
    ranges.push_back(address_range(expert_ids.data, expert_ids.bytes(), "expert_ids"));
    ranges.push_back(address_range(coefficients.data, coefficients.bytes(), "coefficients"));
    ranges.push_back(address_range(shared_gate.data, shared_gate.bytes(), "shared_gate"));
    validate_weights(weights, ranges);
    require_disjoint(ranges);

    detail::MoeNvfp4RouteOutputs outputs{
        static_cast<std::int32_t*>(expert_ids.data),
        static_cast<float*>(coefficients.data),
        static_cast<float*>(shared_gate.data)};
    detail::moe_nvfp4_route_launch(x, weights.router_shared.qdata, outputs, stream);
}

void moe_routed_nvfp4(const Tensor& x, const MoeNvfp4Weights& weights, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = require_tokens(x, "x");
    if (require_tokens(destination, "destination") != tokens) {
        throw std::invalid_argument("moe_nvfp4: x and destination token counts must match");
    }

    const std::size_t required = route_workspace_bytes(tokens);
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("moe_nvfp4: insufficient workspace capacity");
    }

    std::vector<AddressRange> ranges;
    ranges.reserve(12);
    ranges.push_back(address_range(x.data, x.bytes(), "x"));
    ranges.push_back(address_range(destination.data, destination.bytes(), "destination"));
    validate_weights(weights, ranges);
    ranges.push_back(address_range(workspace.base(), workspace.capacity(), "workspace"));
    require_disjoint(ranges);

    auto scope = workspace.scope();
    const DeviceSpan span = workspace.alloc_bytes(required, 256);
    const detail::MoeNvfp4RouteOutputs outputs = route_outputs(static_cast<std::uint8_t*>(span.data), tokens);

    detail::moe_nvfp4_route_launch(x, weights.router_shared.qdata, outputs, stream);
    detail::moe_nvfp4_grouped_launch(x, planes_of(weights.routed_gate_up),
                                     planes_of(weights.routed_down), outputs,
                                     weights.routed_gate_up.expert_begin,
                                     weights.routed_gate_up.matrices, destination, stream);
    detail::moe_nvfp4_shared_launch(x, weights.router_shared.qdata, weights.shared_down.qdata,
                                    outputs, destination, stream);
}

void moe_routed_nvfp4_partial(const Tensor& x, const MoeNvfp4Weights& weights,
                              const MoeNvfp4RouteView& route, Tensor& destination,
                              cudaStream_t stream) {
    const std::int32_t tokens = require_tokens(x, "x");
    if (require_tokens(destination, "destination") != tokens) {
        throw std::invalid_argument("moe_nvfp4: x and destination token counts must match");
    }

    std::vector<AddressRange> ranges;
    ranges.reserve(12);
    ranges.push_back(address_range(x.data, x.bytes(), "x"));
    ranges.push_back(address_range(destination.data, destination.bytes(), "destination"));
    require_route(route, tokens, ranges);
    const BatchedNvfp4Weight& bank = require_expert_bank(weights, ranges);
    require_disjoint(ranges);

    const detail::MoeNvfp4RouteOutputs outputs = route_outputs(route);
    detail::moe_nvfp4_grouped_launch(x, planes_of(weights.routed_gate_up),
                                     planes_of(weights.routed_down), outputs, bank.expert_begin,
                                     bank.matrices, destination, stream);
}

void moe_shared_nvfp4(const Tensor& x, const MoeNvfp4Weights& weights,
                      const MoeNvfp4RouteView& route, Tensor& destination, cudaStream_t stream) {
    const std::int32_t tokens = require_tokens(x, "x");
    if (require_tokens(destination, "destination") != tokens) {
        throw std::invalid_argument("moe_nvfp4: x and destination token counts must match");
    }

    std::vector<AddressRange> ranges;
    ranges.reserve(12);
    ranges.push_back(address_range(x.data, x.bytes(), "x"));
    ranges.push_back(address_range(destination.data, destination.bytes(), "destination"));
    require_route(route, tokens, ranges);
    (void)require_expert_bank(weights, ranges);
    require_disjoint(ranges);

    const detail::MoeNvfp4RouteOutputs outputs = route_outputs(route);
    detail::moe_nvfp4_shared_launch(x, weights.router_shared.qdata, weights.shared_down.qdata,
                                    outputs, destination, stream);
}

} // namespace ninfer::ops
