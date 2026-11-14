// The per-rank shard byte plan of the text roles at tp 2, and the materialized views of the
// loaded prefix of those layers.
//
// Every expected byte mapping is computed here from the storage-layout formulas of
// docs/maintainer/storage-layouts.md (sections 3.7, 4 and 11) over the artifact's own stored
// bytes, read through the reader. The package's own slice arithmetic is only the thing under
// test, so a geometry mistake shows up as a byte mismatch instead of cancelling out.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen4_exp/frontend.h>
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/geometry.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"
#include "targets/qwen3_8_flash_next/impl/load/sharding.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace artifact = ninfer::artifact;

namespace {

using ninfer::artifact::Binder;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::PlaneCopy;
using ninfer::artifact::Reader;
using ninfer::artifact::SliceRange;
using ninfer::artifact::StorageLayout;

namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;
namespace shard = ninfer::targets::qwen3_8_flash_next::detail::sharding;

using ninfer::Weight;
using ninfer::targets::qwen3_8_flash_next::detail::RoleGeometry;
using ninfer::targets::qwen3_8_flash_next::detail::role_geometry;

int g_failures = 0;

void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
}

void require(bool condition, const std::string& message) {
    if (!condition) { fail(message); }
}

constexpr std::uint64_t kAlign      = 256;
constexpr std::uint64_t kRowTile    = 128;
constexpr std::uint64_t kTileColumn = 64;
constexpr std::uint64_t kTileBytes  = 512;
constexpr std::uint64_t kGroup      = 32;   // W8G32 group width
constexpr std::uint64_t kKAlign     = 128;  // row-split K alignment
constexpr std::size_t kLoadedLayers = 4;

std::uint64_t align_up(std::uint64_t value, std::uint64_t unit) {
    return (value + unit - 1) / unit * unit;
}

// storage-layouts.md section 4: one scale word's position inside one matrix's swizzled plane.
std::uint64_t swizzle_word(std::uint64_t row, std::uint64_t group, std::uint64_t columns) {
    const std::uint64_t tiles = columns / kTileColumn;
    return (row / kRowTile * tiles + group / 4) * kTileBytes + (row % 32) * 16 +
           ((row % kRowTile) / 32) * 4 + group % 4;
}

using Copies = std::vector<PlaneCopy>;

Copies ordered(Copies copies) {
    std::sort(copies.begin(), copies.end(), [](const PlaneCopy& a, const PlaneCopy& b) {
        return std::tie(a.source_offset, a.dest_offset, a.bytes) <
               std::tie(b.source_offset, b.dest_offset, b.bytes);
    });
    return copies;
}

bool same(const Copies& left, const Copies& right) {
    if (left.size() != right.size()) { return false; }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].source_offset != right[index].source_offset ||
            left[index].dest_offset != right[index].dest_offset ||
            left[index].bytes != right[index].bytes) {
            return false;
        }
    }
    return true;
}

Copies coalesced(Copies copies) {
    copies = ordered(copies);
    Copies out;
    for (const auto& copy : copies) {
        if (!out.empty()) {
            const auto& last = out.back();
            if (last.source_offset + last.bytes == copy.source_offset &&
                last.dest_offset + last.bytes == copy.dest_offset) {
                out.back().bytes += copy.bytes;
                continue;
            }
        }
        out.push_back(copy);
    }
    return out;
}

std::uint64_t total(const std::vector<SliceRange>& ranges) {
    return std::accumulate(ranges.begin(), ranges.end(), std::uint64_t{0},
                           [](std::uint64_t value, const SliceRange& range) {
                               return value + range.count;
                           });
}

std::uint64_t word_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return 2;
    case NumericFormat::FP32:
        return 4;
    default:
        throw std::runtime_error("the contiguous oracle covers BF16 and FP32 only");
    }
}

std::uint64_t oracle_bytes(StorageLayout layout, NumericFormat format,
                           const std::vector<std::uint64_t>& shape) {
    if (shape.size() == 1) { return shape[0] * word_bytes(format); }
    if (layout == StorageLayout::ContiguousLeV1) { return shape[0] * shape[1] * word_bytes(format); }
    if (layout == StorageLayout::RowSplitK128V1) {
        const auto groups = align_up(shape[1], kKAlign) / kGroup;
        return align_up(shape[0] * groups * kGroup, kAlign) + shape[0] * groups * 2;
    }
    if (layout == StorageLayout::BatchedBlockScaleK16M128x4V1) {
        const auto code   = shape[1] * shape[2] / 2;
        const auto scale  = shape[1] * shape[2] / 16;
        return align_up(shape[0] * code, kAlign) + shape[0] * scale + shape[0] * 4;
    }
    if (layout == StorageLayout::RowPagedG16x4KV1) {
        const auto rows_per_page = 4096 / (shape[1] / 2 + shape[1] / 16);
        const auto pages = (shape[0] + rows_per_page - 1) / rows_per_page;
        return pages * 4096 + 4;
    }
    throw std::runtime_error("the oracle does not cover this layout");
}

std::vector<std::uint64_t> narrowed_shape(const RoleGeometry& geometry, shard::SplitAxis split,
                                          std::uint64_t selected) {
    std::vector<std::uint64_t> shape(geometry.object_shape().begin(), geometry.object_shape().end());
    if (shape.size() == 1) { shape[0] = selected; return shape; }
    if (split == shard::SplitAxis::Input) { shape[1] = selected; } else { shape[0] = selected; }
    return shape;
}

// The byte mapping the layout formulas give for one object's shard on one rank.
Copies expected_copies(const RoleGeometry& geometry, shard::SplitAxis split,
                       const std::vector<SliceRange>& ranges) {
    const auto shape = std::vector<std::uint64_t>(geometry.object_shape().begin(),
                                                  geometry.object_shape().end());
    Copies out;
    if (ranges.empty()) {
        out.push_back({0, 0, oracle_bytes(geometry.layout, geometry.format, shape)});
        return out;
    }
    const std::uint64_t selected = total(ranges);

    // A shard that takes the whole axis is the whole object: a full copy has no internal structure
    // to re-derive, whatever the layout.
    const auto extent = split == shard::SplitAxis::Input ? shape[1] : shape[0];
    if (ranges.size() == 1 && ranges.front().begin == 0 && ranges.front().count == extent) {
        out.push_back({0, 0, oracle_bytes(geometry.layout, geometry.format, shape)});
        return out;
    }

    if (geometry.layout == StorageLayout::ContiguousLeV1) {
        const std::uint64_t word = word_bytes(geometry.format);
        if (split == shard::SplitAxis::Input) {
            for (std::uint64_t row = 0; row < shape[0]; ++row) {
                std::uint64_t column = 0;
                for (const auto& range : ranges) {
                    out.push_back({(row * shape[1] + range.begin) * word,
                                   (row * selected + column) * word, range.count * word});
                    column += range.count;
                }
            }
            return out;
        }
        std::uint64_t dest_row = 0;
        for (const auto& range : ranges) {
            out.push_back({range.begin * shape[1] * word, dest_row * shape[1] * word,
                           range.count * shape[1] * word});
            dest_row += range.count;
        }
        return out;
    }

    if (geometry.layout == StorageLayout::RowSplitK128V1) {
        const auto groups_per_row = align_up(shape[1], kKAlign) / kGroup;
        const auto parent_scale   = align_up(shape[0] * groups_per_row * kGroup, kAlign);
        if (split == shard::SplitAxis::Input) {
            // storage-layouts.md 3.8: a K-divided shard is a standalone tensor whose own K_pad and
            // groups_per_row are recomputed from the selected columns.
            const auto shard_groups = align_up(selected, kKAlign) / kGroup;
            const auto shard_scale  = align_up(shape[0] * shard_groups * kGroup, kAlign);
            const auto groups       = ranges.front().count / kGroup;
            for (std::uint64_t row = 0; row < shape[0]; ++row) {
                out.push_back({row * groups_per_row * kGroup + ranges.front().begin,
                               row * shard_groups * kGroup, ranges.front().count});
                out.push_back({parent_scale + row * groups_per_row * 2 + ranges.front().begin / kGroup * 2,
                               shard_scale + row * shard_groups * 2, groups * 2});
            }
            return out;
        }
        const auto shard_scale = align_up(selected * groups_per_row * kGroup, kAlign);
        std::uint64_t dest_row = 0;
        for (const auto& range : ranges) {
            for (std::uint64_t index = 0; index < range.count; ++index) {
                out.push_back({(range.begin + index) * groups_per_row * kGroup,
                               (dest_row + index) * groups_per_row * kGroup, groups_per_row * kGroup});
                out.push_back({parent_scale + (range.begin + index) * groups_per_row * 2,
                               shard_scale + (dest_row + index) * groups_per_row * 2, groups_per_row * 2});
            }
            dest_row += range.count;
        }
        return out;
    }

    if (geometry.layout == StorageLayout::BatchedBlockScaleK16M128x4V1) {
        const auto matrix_code  = shape[1] * shape[2] / 2;
        const auto matrix_scale = shape[1] * shape[2] / 16;
        const auto parent_scale   = align_up(shape[0] * matrix_code, kAlign);
        const auto parent_divisor = parent_scale + shape[0] * matrix_scale;
        const auto shard_scale    = align_up(selected * matrix_code, kAlign);
        const auto shard_divisor  = shard_scale + selected * matrix_scale;
        std::uint64_t index = 0;
        for (const auto& range : ranges) {
            for (std::uint64_t matrix = range.begin; matrix < range.begin + range.count; ++matrix) {
                out.push_back({matrix * matrix_code, index * matrix_code, matrix_code});
                out.push_back({parent_scale + matrix * matrix_scale, shard_scale + index * matrix_scale,
                               matrix_scale});
                out.push_back({parent_divisor + matrix * 4, shard_divisor + index * 4, 4});
                ++index;
            }
        }
        return out;
    }

    throw std::runtime_error("the oracle does not cover this layout");
}

std::vector<std::byte> read_object(const Reader& reader, const std::string& name) {
    const auto* descriptor = reader.find(name);
    if (descriptor == nullptr) { throw std::runtime_error("the artifact has no object " + name); }
    if (std::holds_alternative<artifact::ResourceDescriptor>(*descriptor)) {
        throw std::runtime_error("the object " + name + " is a resource, not a tensor");
    }
    std::vector<std::byte> out(artifact::object_bytes(*descriptor));
    const auto offset = reader.payload_offset() + artifact::object_offset(*descriptor);
    if (reader.read_direct(offset, out) != out.size()) {
        throw std::runtime_error("short artifact read for " + name);
    }
    return out;
}

std::vector<std::byte> assemble(const std::vector<std::byte>& parent, const Copies& copies) {
    std::uint64_t size = 0;
    for (const auto& copy : copies) {
        size = std::max(size, copy.dest_offset + copy.bytes);
    }
    std::vector<std::byte> out(size, std::byte{0});
    for (const auto& copy : copies) {
        if (copy.source_offset + copy.bytes > parent.size()) {
            throw std::runtime_error("a copy reads past the parent object: " + std::to_string(copy.source_offset) +
                                     "+" + std::to_string(copy.bytes) + " > " + std::to_string(parent.size()));
        }
        for (std::uint64_t index = 0; index < copy.bytes; ++index) {
            out[static_cast<std::size_t>(copy.dest_offset + index)] =
                parent[static_cast<std::size_t>(copy.source_offset + index)];
        }
    }
    return out;
}

std::vector<std::byte> device_bytes(const void* base, std::size_t size) {
    std::vector<std::byte> out(size);
    if (size == 0) { return out; }
    if (cudaMemcpy(out.data(), base, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("the shard view could not be read back to the host");
    }
    return out;
}

const shard::ShardMapping& mapping_of(const flash::ShardPlanByName& shards, std::string_view role) {
    const auto found = shards.find(role);
    if (found == shards.end()) { throw std::runtime_error("the shard map has no role " + std::string(role)); }
    return found->second;
}

struct BoundObject {
    std::string name;
    RoleGeometry geometry;
    std::vector<roles::Role> owners;
    std::vector<shard::SplitAxis> axes;
};


struct ExpectedRange {
    std::uint64_t begin;
    std::uint64_t count;
};

struct ExpectedRoleRanges {
    roles::Role role;
    std::array<ExpectedRange, 2> ranks;
};

// What each rank owns of each role, read off the role geometry table in `test_load_plan.cpp` and the
// shard spans in `test_sharding.cpp`. This is the independent anchor for the ranges; the byte plan
// must agree with it, not with a mirror of the implementation.
constexpr ExpectedRoleRanges kExpectedRanges[] = {
    {roles::Role::TokenEmbedding, {{{0, 248320}, {0, 248320}}}},
    {roles::Role::OutputHead, {{{0, 124160}, {124160, 124160}}}},
    {roles::Role::FinalHcNorm, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::FinalHcDown, {{{0, 320}, {0, 320}}}},
    {roles::Role::FinalHcUp, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::AttentionHcNorm, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::AttentionHcDown, {{{0, 320}, {0, 320}}}},
    {roles::Role::AttentionHcInject, {{{320, 4}, {320, 4}}}},
    {roles::Role::AttentionHcUp, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::MoeHcNorm, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::MoeHcDown, {{{0, 320}, {0, 320}}}},
    {roles::Role::MoeHcInject, {{{320, 4}, {320, 4}}}},
    {roles::Role::MoeHcUp, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::GdnQuery, {{{0, 1024}, {1024, 1024}}}},
    {roles::Role::GdnKey, {{{2048, 1024}, {3072, 1024}}}},
    {roles::Role::GdnValue, {{{4096, 3072}, {7168, 3072}}}},
    {roles::Role::GdnZ, {{{10240, 3072}, {13312, 3072}}}},
    {roles::Role::GdnAProjection, {{{0, 48}, {0, 48}}}},
    {roles::Role::GdnBProjection, {{{48, 48}, {48, 48}}}},
    {roles::Role::GdnALog, {{{0, 48}, {0, 48}}}},
    {roles::Role::GdnDtBias, {{{0, 48}, {0, 48}}}},
    {roles::Role::GdnConvolution, {{{0, 4}, {0, 4}}}},
    {roles::Role::GdnNorm, {{{0, 128}, {0, 128}}}},
    {roles::Role::GdnOutput, {{{0, 3072}, {3072, 3072}}}},
    {roles::Role::QsaQuery, {{{0, 3072}, {3072, 3072}}}},
    {roles::Role::QsaKey, {{{0, 256}, {256, 256}}}},
    {roles::Role::QsaValue, {{{512, 256}, {768, 256}}}},
    {roles::Role::QsaIndexQuery, {{{1024, 512}, {1024, 512}}}},
    {roles::Role::QsaIndexKey, {{{1536, 128}, {1536, 128}}}},
    {roles::Role::QsaOutput, {{{0, 3072}, {3072, 3072}}}},
    {roles::Role::QsaOutputGate, {{{6144, 3072}, {9216, 3072}}}},
    {roles::Role::QsaQueryNorm, {{{0, 256}, {0, 256}}}},
    {roles::Role::QsaIndexQueryNorm, {{{0, 128}, {0, 128}}}},
    {roles::Role::QsaKeyNorm, {{{0, 256}, {0, 256}}}},
    {roles::Role::QsaIndexKeyNorm, {{{0, 128}, {0, 128}}}},
    {roles::Role::MoeRouter, {{{0, 512}, {0, 512}}}},
    {roles::Role::MoeSharedScore, {{{512, 1}, {512, 1}}}},
    {roles::Role::MoeSharedGate, {{{513, 640}, {513, 640}}}},
    {roles::Role::MoeSharedUp, {{{1153, 640}, {1153, 640}}}},
    {roles::Role::MoeSharedDown, {{{0, 640}, {0, 640}}}},
    {roles::Role::MoeRoutedGate, {{{0, 256}, {256, 256}}}},
    {roles::Role::MoeRoutedUp, {{{0, 256}, {256, 256}}}},
    {roles::Role::MoeRoutedDown, {{{0, 256}, {256, 256}}}},
    {roles::Role::PleTable, {{{0, 2500012}, {0, 2500012}}}},
    {roles::Role::PleKey, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::PleValue, {{{10240, 2560}, {10240, 2560}}}},
    {roles::Role::PleNormKey, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::PleNormQuery, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::PleNormConv, {{{0, 10240}, {0, 10240}}}},
    {roles::Role::PleConvolution, {{{0, 4}, {0, 4}}}},
};

const ExpectedRoleRanges& expected_ranges_of(roles::Role role) {
    for (const auto& entry : kExpectedRanges) {
        if (entry.role == role) { return entry; }
    }
    throw std::runtime_error("the expected range table has no role");
}

std::map<std::string, BoundObject> bound_objects(const flash::BindingPlan& plan,
                                                  const flash::ShardPlanByName& shards) {
    std::map<std::string, BoundObject> by_object;
    for (const auto* parameter : plan.parameters()) {
        for (const auto& part : parameter->parts) {
            const auto& object = plan.objects.at(part.object);
            auto& entry = by_object[object.name];
            if (entry.owners.empty()) { entry.geometry = role_geometry(parameter->role); }
            if (std::find(entry.owners.begin(), entry.owners.end(), parameter->role) == entry.owners.end()) {
                entry.owners.push_back(parameter->role);
                entry.axes.push_back(mapping_of(shards, roles::role_name(parameter->role)).split);
            }
            entry.name = object.name;
        }
    }
    return by_object;
}

std::vector<SliceRange> rank_ranges(const BoundObject& object, int rank) {
    std::vector<SliceRange> ranges;
    for (const auto role : object.owners) {
        const auto& own = expected_ranges_of(role).ranks[static_cast<std::size_t>(rank)];
        ranges.push_back({own.begin, own.count});
    }
    std::sort(ranges.begin(), ranges.end(), [](const SliceRange& a, const SliceRange& b) {
        return a.begin < b.begin;
    });
    ranges.erase(std::unique(ranges.begin(), ranges.end(), [](const SliceRange& a, const SliceRange& b) {
        return a.begin == b.begin && a.count == b.count;
    }), ranges.end());
    return ranges;
}

void test_partition(const flash::ShardPlanByName& shards) {
    std::size_t checked = 0;
    for (const auto role : roles::all_roles()) {
        const auto name = roles::role_name(role);
        const auto mapping = mapping_of(shards, name);
        if (mapping.axis == shard::ShardAxis::Replicated) { continue; }
        ++checked;
        const auto& expected = expected_ranges_of(role);
        require(mapping.split == shard::SplitAxis::Batch || mapping.split == shard::SplitAxis::Input ||
                    mapping.split == shard::SplitAxis::Output,
                std::string(name) + " has no split axis");
        for (int rank = 0; rank < 2; ++rank) {
            const auto& part = mapping.ranks[static_cast<std::size_t>(rank)];
            const auto& own = expected.ranks[static_cast<std::size_t>(rank)];
            require(part.begin == static_cast<std::int64_t>(own.begin) &&
                        part.end - part.begin == static_cast<std::int64_t>(own.count),
                    std::string(name) + " rank " + std::to_string(rank) + " names " +
                        std::to_string(part.begin) + "+" + std::to_string(part.end - part.begin) +
                        "; the expected table gives " + std::to_string(own.begin) + "+" +
                        std::to_string(own.count));
            require(part.begin % 128 == 0, std::string(name) + " rank starts inside a 128-row band");
            require((part.end - part.begin) % 128 == 0,
                    std::string(name) + " rank covers part of a 128-row band");
        }
    }
    require(checked == 14, "the shard map names " + std::to_string(checked) +
                               " split roles; the expected table gives 14");
}

void test_expert_subsets(const flash::ShardPlanByName& shards) {
    std::size_t seen = 0;
    for (const auto role : roles::all_roles()) {
        const auto name = roles::role_name(role);
        const auto mapping = mapping_of(shards, name);
        if (mapping.axis != shard::ShardAxis::Expert) { continue; }
        ++seen;
        require(mapping.split == shard::SplitAxis::Batch, std::string(name) + " is not split on the batch axis");
        require(mapping.ranks.size() == 2, std::string(name) + " does not name two ranks");
        for (std::size_t rank = 0; rank < mapping.ranks.size(); ++rank) {
            const auto& part = mapping.ranks[rank];
            require(part.end - part.begin == 256, std::string(name) + " rank " + std::to_string(rank) +
                                                      " owns " + std::to_string(part.end - part.begin) +
                                                      " matrices; the expert set gives 256");
            require(part.begin == static_cast<std::int64_t>(rank) * 256,
                   std::string(name) + " rank " + std::to_string(rank) + " owns matrices starting at " +
                       std::to_string(part.begin) + "; the expert set gives " + std::to_string(rank * 256));
        }
    }
    require(seen == 3, "the shard map names " + std::to_string(seen) + " expert roles; the MoE has three");
}

void test_replicated(const flash::ShardPlanByName& shards,
                     const std::map<std::string, BoundObject>& by_object,
                     const std::vector<flash::ObjectShardPlan>& byte_plan) {
    std::size_t checked = 0;
    for (const auto& entry : byte_plan) {
        const auto found = by_object.find(entry.object);
        if (found == by_object.end()) { continue; }
        const auto& object = found->second;
        for (const auto role : object.owners) {
            const auto name = roles::role_name(role);
            const auto mapping = mapping_of(shards, name);
            if (mapping.axis != shard::ShardAxis::Replicated) { continue; }
            ++checked;
            const auto geometry = role_geometry(role);
            const auto begin = mapping.split == shard::SplitAxis::Input ? geometry.column_begin
                                                                       : geometry.row_begin;
            const auto count = mapping.split == shard::SplitAxis::Input ? geometry.column_count
                                                                        : geometry.row_count;
            for (int rank = 0; rank < 2; ++rank) {
                const auto& ranges = entry.ranges[static_cast<std::size_t>(rank)];
                const auto owns = std::find_if(ranges.begin(), ranges.end(),
                                               [&](const SliceRange& range) {
                                                   return range.begin == begin && range.count == count;
                                               });
                require(owns != ranges.end(), std::string(name) + " rank " + std::to_string(rank) +
                                                 " does not own its whole role rectangle in " +
                                                 entry.object);
                if (object.axes.size() == 1) {
                    require(same(entry.copies[static_cast<std::size_t>(rank)], entry.copies[0]),
                            std::string(name) + " rank " + std::to_string(rank) +
                                " maps to a different object offset");
                }
            }
        }
    }
    require(checked > 0, "no replicated role was checked");
}

void test_byte_plan(const flash::BindingPlan& plan, const flash::ShardPlanByName& shards,
                    const std::map<std::string, BoundObject>& by_object,
                    const std::vector<flash::ObjectShardPlan>& byte_plan) {
    require(byte_plan.size() == plan.objects.size(),
            "the byte plan covers " + std::to_string(byte_plan.size()) + " objects; the plan binds " +
                std::to_string(plan.objects.size()));

    std::size_t matched = 0;
    for (const auto& entry : byte_plan) {
        const auto found = by_object.find(entry.object);
        if (found == by_object.end()) { continue; }
        const auto& object = found->second;
        if (object.owners.empty()) {
            require(entry.copies[0].size() == 1 && entry.copies[0].front().source_offset == 0 &&
                    entry.copies[0].front().dest_offset == 0,
                    "the unbound object " + entry.object + " is not a whole-object copy");
            continue;
        }
        const auto split = object.axes.front();
        for (int rank = 0; rank < 2; ++rank) {
            const auto ranges   = rank_ranges(object, rank);
            const auto expected = coalesced(expected_copies(object.geometry, split, ranges));
            const Copies actual = coalesced(entry.copies[static_cast<std::size_t>(rank)]);
            require(same(expected, actual),
                    "object " + entry.object + " rank " + std::to_string(rank) +
                        " does not match the layout formulas for split axis " +
                        std::to_string(static_cast<int>(split)));
            require(same(expected, actual), "the byte plan for " + entry.object + " rank " +
                                            std::to_string(rank) + " holds " +
                                            std::to_string(actual.size()) + " ranges; the layout formulas give " +
                                            std::to_string(expected.size()));
            const auto shape = narrowed_shape(object.geometry, split, total(ranges));
            require(entry.encoded_bytes[static_cast<std::size_t>(rank)] ==
                    oracle_bytes(object.geometry.layout, object.geometry.format, shape),
                    "the shard of " + entry.object + " rank " + std::to_string(rank) + " is " +
                        std::to_string(entry.encoded_bytes[static_cast<std::size_t>(rank)]) +
                        " bytes; the oracle gives " +
                        std::to_string(oracle_bytes(object.geometry.layout, object.geometry.format, shape)));
            ++matched;
        }
    }
    std::cout << "flash-next shard byte plan: " << matched << " object ranks match the layout formulas\n";
}


std::vector<SliceRange> ordered_ranges(std::vector<SliceRange> ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const SliceRange& a, const SliceRange& b) {
        return a.begin < b.begin;
    });
    return ranges;
}

void test_role_blocks(const flash::BindingPlan& plan, const flash::ShardPlanByName& shards,
                      const std::map<std::string, BoundObject>& by_object,
                      const std::vector<flash::ObjectShardPlan>& byte_plan) {
    const auto placements = flash::shard_placements(plan, shards, 2);
    std::size_t checked = 0;
    std::size_t shared = 0;
    std::size_t offset_blocks = 0;
    for (const auto& entry : byte_plan) {
        const auto found = by_object.find(entry.object);
        if (found == by_object.end()) { continue; }
        const auto& object = found->second;
        if (object.owners.empty()) { continue; }
        const auto placement = placements.find(entry.object);
        require(placement != placements.end(), "the object " + entry.object + " has no placement");
        bool split_role = false;
        for (const auto role : object.owners) {
            if (mapping_of(shards, roles::role_name(role)).axis != shard::ShardAxis::Replicated) { split_role = true; }
        }
        if (!split_role) {
            require(placement->second.axis == artifact::ShardAxis::Replicated,
                    "the object " + entry.object + " is replicated but its placement is split");
            continue;
        }
        const auto expected_axis = object.axes.front() == shard::SplitAxis::Input
            ? artifact::ShardAxis::Columns
            : artifact::ShardAxis::Rows;
        require(placement->second.axis == expected_axis,
                "the object " + entry.object + " splits axis " +
                    std::to_string(static_cast<int>(object.axes.front())) +
                    " but its placement names " + std::to_string(static_cast<int>(placement->second.axis)));
        for (int rank = 0; rank < 2; ++rank) {
            const auto expected_ranges = ordered_ranges(rank_ranges(object, rank));
            const auto& actual_ranges = placement->second.device_ranges[static_cast<std::size_t>(rank)];
            require(expected_ranges.size() == actual_ranges.size(),
                    "the placement for " + entry.object + " rank " + std::to_string(rank) + " names " +
                        std::to_string(actual_ranges.size()) + " ranges; the byte plan copies " +
                        std::to_string(expected_ranges.size()));
            for (std::size_t index = 0; index < expected_ranges.size(); ++index) {
                require(expected_ranges[index].begin == actual_ranges[index].begin &&
                            expected_ranges[index].count == actual_ranges[index].count,
                        "the placement for " + entry.object + " rank " + std::to_string(rank) +
                            " range " + std::to_string(index) + " is " +
                            std::to_string(actual_ranges[index].begin) + "+" +
                            std::to_string(actual_ranges[index].count) + "; the byte plan copies " +
                            std::to_string(expected_ranges[index].begin) + "+" +
                            std::to_string(expected_ranges[index].count));
            }
        }
        const auto split = object.axes.front();
        if (object.owners.size() > 1) { ++shared; }
        for (int rank = 0; rank < 2; ++rank) {
            const auto object_copies = coalesced(expected_copies(object.geometry, split,
                                                                 rank_ranges(object, rank)));
            require(entry.blocks[static_cast<std::size_t>(rank)].size() == object.owners.size(),
                    "the object " + entry.object + " rank " + std::to_string(rank) + " names " +
                        std::to_string(entry.blocks[static_cast<std::size_t>(rank)].size()) +
                        " role blocks; the plan binds " + std::to_string(object.owners.size()));
            for (const auto& block : entry.blocks[static_cast<std::size_t>(rank)]) {
                const auto& own = expected_ranges_of(block.role).ranks[static_cast<std::size_t>(rank)];
                const auto geometry = role_geometry(block.role);
                require(block.bytes == oracle_bytes(geometry.layout, geometry.format,
                                                    narrowed_shape(geometry, split, own.count)),
                        "the role block " + std::string(roles::role_name(block.role)) + " in " +
                            entry.object + " rank " + std::to_string(rank) + " is " +
                            std::to_string(block.bytes) + " bytes; the oracle gives " +
                            std::to_string(oracle_bytes(geometry.layout, geometry.format,
                                                        narrowed_shape(geometry, split, own.count))));
                const auto start = expected_copies(geometry, split, {{own.begin, own.count}}).front().source_offset;
                const auto carrier = std::find_if(object_copies.begin(), object_copies.end(),
                                                  [&](const PlaneCopy& copy) {
                                                      return copy.source_offset <= start &&
                                                             start < copy.source_offset + copy.bytes;
                                                  });
                require(carrier != object_copies.end(),
                        "no copy carries the role block " + std::string(roles::role_name(block.role)));
                require(carrier->dest_offset + (start - carrier->source_offset) == block.offset,
                        "the role block " + std::string(roles::role_name(block.role)) + " in " +
                            entry.object + " rank " + std::to_string(rank) + " lands at " +
                            std::to_string(block.offset) + "; the oracle gives " +
                            std::to_string(carrier->dest_offset + (start - carrier->source_offset)));
                if (block.offset != 0) { ++offset_blocks; }
                ++checked;
            }
        }
    }
    require(shared >= 1, "no object with more than one role was checked");
    require(offset_blocks >= 1, "no role block sits at a non-zero offset");
    std::cout << "flash-next role blocks: " << checked << " role blocks over " << shared <<
               " multi-role objects, " << offset_blocks << " at a non-zero offset\n";
}

const std::vector<std::pair<roles::Role, std::uint32_t>> kSampled = {
    {roles::Role::MoeRoutedGate, 0},
    {roles::Role::MoeRoutedUp, 0},
    {roles::Role::MoeRoutedDown, 0},
    {roles::Role::GdnQuery, 0},
    {roles::Role::QsaKey, 3},
    {roles::Role::TokenEmbedding, 0},
};

void test_real_bytes(const flash::BindingPlan& plan, const flash::ShardPlanByName& shards,
                     const Reader& reader,
                     const std::vector<flash::ObjectShardPlan>& byte_plan) {
    const auto by_object = bound_objects(plan, shards);
    const auto parameters = plan.parameters();

    for (const auto [role, layer] : kSampled) {
        const auto name = std::string(roles::parameter_name(role, layer));
        const auto found = std::find_if(parameters.begin(), parameters.end(),
                                        [&](const flash::BoundParameter* candidate) {
                                            return candidate->parameter == name;
                                        });
        if (found == parameters.end()) { throw std::runtime_error("no parameter " + name); }
        const auto& parameter = *found;
        const auto& object = plan.objects.at(parameter->parts.front().object);
        const auto entry_found = std::find_if(byte_plan.begin(), byte_plan.end(),
                                              [&](const flash::ObjectShardPlan& candidate) {
                                                  return candidate.object == object.name;
                                              });
        if (entry_found == byte_plan.end()) { throw std::runtime_error("no byte plan for " + object.name); }
        const auto& entry = *entry_found;
        const auto info_found = by_object.find(object.name);
        if (info_found == by_object.end()) { throw std::runtime_error("no bound entry for object " + object.name); }
        const auto& info   = info_found->second;
        const auto split   = mapping_of(shards, roles::role_name(parameter->role)).split;
        const auto parent  = read_object(reader, object.name);
        for (int rank = 0; rank < 2; ++rank) {
            const auto ranges = rank_ranges(info, rank);
            const auto expected = ordered(expected_copies(info.geometry, split, ranges));
            const Copies actual = ordered(entry.copies[static_cast<std::size_t>(rank)]);
            require(assemble(parent, actual) == assemble(parent, expected),
                    "the shard of " + object.name + " rank " + std::to_string(rank) +
                        " does not hold the bytes the layout formulas give");
        }
    }
}

void test_scale_plane(const flash::ShardPlanByName& shards, const Reader& reader,
                      const std::vector<flash::ObjectShardPlan>& byte_plan) {
    const auto found = std::find_if(byte_plan.begin(), byte_plan.end(),
                                    [&](const flash::ObjectShardPlan& candidate) {
                                        return candidate.object == "weight/000000";
                                    });
    if (found == byte_plan.end()) { throw std::runtime_error("no byte plan for weight/000000"); }
    const auto& entry = *found;

    const std::vector<std::uint64_t> shape{512, 1280, 2560};
    const auto matrix_code  = shape[1] * shape[2] / 2;
    const auto matrix_scale = shape[1] * shape[2] / 16;
    const auto parent_scale = align_up(shape[0] * matrix_code, kAlign);
    const auto parent       = read_object(reader, "weight/000000");
    const auto& mapping     = mapping_of(shards, roles::role_name(roles::Role::MoeRoutedGate));

    for (int rank = 0; rank < 2; ++rank) {
        const auto shard_scale = align_up(256 * matrix_code, kAlign);
        require(entry.scale_plane_offset[static_cast<std::size_t>(rank)] == shard_scale,
                "the scale plane of weight/000000 rank " + std::to_string(rank) + " starts at " +
                    std::to_string(entry.scale_plane_offset[static_cast<std::size_t>(rank)]) +
                    "; the batched formula gives " + std::to_string(shard_scale));

        for (std::uint64_t index = 0; index < 256; index += 31) {
            const std::uint64_t matrix = static_cast<std::uint64_t>(mapping.ranks[rank].begin) + index;
            for (const std::uint64_t row : {0ULL, 128ULL, 639ULL}) {
                for (const std::uint64_t group : {0ULL, 3ULL, 159ULL}) {
                    const auto word = swizzle_word(row, group, shape[2]);
                    const auto source = parent_scale + matrix * matrix_scale + word;
                    const auto dest   = shard_scale + index * matrix_scale + word;
                    const auto carrier = std::find_if(entry.copies[static_cast<std::size_t>(rank)].begin(),
                                                      entry.copies[static_cast<std::size_t>(rank)].end(),
                                                      [&](const PlaneCopy& copy) {
                                                          return copy.source_offset <= source &&
                                                                 source < copy.source_offset + copy.bytes;
                                                      });
                    require(carrier != entry.copies[static_cast<std::size_t>(rank)].end(),
                            "the shard of weight/000000 rank " + std::to_string(rank) +
                                " has no copy carrying the parent scale word for matrix " +
                                std::to_string(matrix));
                    require(carrier->dest_offset + (source - carrier->source_offset) == dest,
                            "the shard scale word for rank " + std::to_string(rank) + " index " +
                                std::to_string(index) + " row " + std::to_string(row) + " group " +
                                std::to_string(group) + " lands at " +
                                std::to_string(carrier->dest_offset + (source - carrier->source_offset)) +
                                ", not at the shard position " + std::to_string(dest) +
                                " for parent matrix " + std::to_string(matrix));
                }
            }
        }
    }
}

const void* entry_payload(const flash::Variant::ModelView& view, roles::Role role, std::size_t layer) {
    switch (role) {
    case roles::Role::MoeRoutedGate:
    case roles::Role::MoeRoutedUp:
        return view.layers[layer].post_mixer.routed_gate_up.payload;
    case roles::Role::MoeRoutedDown:
        return view.layers[layer].post_mixer.routed_down.payload;
    case roles::Role::GdnQuery:
        return view.layers[layer].gdn_projection.qkvz.payload;
    case roles::Role::QsaKey:
        return view.layers[layer].attention_projection.key_value.payload;
    case roles::Role::TokenEmbedding:
        return view.embedding.payload;
    default:
        return nullptr;
    }
}

Weight role_weight(const flash::Variant::ModelView& view, roles::Role role, std::size_t layer) {
    switch (role) {
    case roles::Role::GdnQuery:
        return view.layers[layer].gdn_projection.qkvz;
    case roles::Role::QsaKey:
        return view.layers[layer].attention_projection.key_value;
    case roles::Role::TokenEmbedding:
        return view.embedding;
    default:
        throw std::runtime_error("the test has no view for this role");
    }
}

// The materialized per-rank views. Needs two devices; skipped when they cannot be taken.
int test_materialized_views(const char* path) {
    ninfer::ExecutionContext execution({0, 1});
    if (execution.tp != 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }

    Reader reader(path);
    Binder binder(reader, 2);
    const auto plan       = flash::bind_artifact(binder, flash::WeightsProfile::Nvfp4, 2);
    const auto shards     = flash::shard_plan_by_name(2);
    const auto byte_plan  = flash::shard_byte_plan(plan, shards, 2);
    (void)ninfer::targets::qwen4_exp::bind_frontend_resources(binder);
    const auto by_object  = bound_objects(plan, shards);
    const auto placements = flash::shard_placements(plan, shards, 2);
    const auto parameters = plan.parameters();

    const auto needed = [&](const std::string& name) {
        for (const auto* parameter : parameters) {
            for (const auto& part : parameter->parts) {
                if (plan.objects.at(part.object).name != name) { continue; }
                if (parameter->role == roles::Role::PleTable) { continue; }
                if (!roles::is_layer_scoped(parameter->role) || parameter->layer < kLoadedLayers) {
                    return true;
                }
            }
        }
        return false;
    };
    for (const auto& object : plan.objects) {
        require(placements.count(object.name) != 0,
                "the tensor object " + object.name + " has no shard placement");
        if (needed(object.name)) {
            binder.materialize_on_device(object.handle);
        } else {
            binder.validate_only(object.handle);
        }
    }
    binder.validate_unconsumed_matching("text/");
    binder.validate_unconsumed_matching("resource/text/");
    for (const auto& object : plan.unclaimed.object_names) {
        binder.validate_only(object);
    }

    artifact::MaterializedArtifact materialized;
    try {
        materialized = artifact::materialize(reader, binder.finish(), execution);
    } catch (const std::exception& error) {
        std::cout << "skip: the two devices cannot be taken: " << error.what() << "\n";
        return 77;
    }
    const auto views = flash::shard_views(plan, materialized, shards, 2, kLoadedLayers);

    for (const auto [role, layer] : kSampled) {
        const auto name = std::string(roles::parameter_name(role, layer));
        const auto found = std::find_if(parameters.begin(), parameters.end(),
                                        [&](const flash::BoundParameter* candidate) {
                                            return candidate->parameter == name;
                                        });
        if (found == parameters.end()) { throw std::runtime_error("no parameter " + name); }
        const auto& parameter = *found;
        const auto& object = plan.objects.at(parameter->parts.front().object);
        const auto entry_found = std::find_if(byte_plan.begin(), byte_plan.end(),
                                              [&](const flash::ObjectShardPlan& candidate) {
                                                  return candidate.object == object.name;
                                              });
        if (entry_found == byte_plan.end()) { throw std::runtime_error("no byte plan for " + object.name); }
        const auto& entry = *entry_found;
        const auto info_found = by_object.find(object.name);
        if (info_found == by_object.end()) { throw std::runtime_error("no bound entry for object " + object.name); }
        const auto& info   = info_found->second;
        const auto split   = mapping_of(shards, roles::role_name(parameter->role)).split;
        const auto parent  = read_object(reader, object.name);
        for (int rank = 0; rank < 2; ++rank) {
            const auto& view = views.views[static_cast<std::size_t>(rank)];
            const auto ranges = rank_ranges(info, rank);
            const auto expected = assemble(parent, ordered(expected_copies(info.geometry, split, ranges)));
            const Copies actual = ordered(entry.copies[static_cast<std::size_t>(rank)]);
            const auto base = entry_payload(view, parameter->role, parameter->layer);
            require(base != nullptr, "rank " + std::to_string(rank) + " has no view for " + name);
            if (parameter->library.has_value()) {
                const auto& bank = parameter->role == roles::Role::MoeRoutedDown
                                       ? view.layers[parameter->layer].post_mixer.routed_down
                                       : view.layers[parameter->layer].post_mixer.routed_gate_up;
                require(bank.matrices == 256, name + " rank " + std::to_string(rank) + " view holds " +
                                                  std::to_string(bank.matrices) + " matrices");
                require(bank.rows == static_cast<std::int32_t>(info.geometry.object_shape()[1]),
                        name + " rank " + std::to_string(rank) + " view spans " +
                            std::to_string(bank.rows) + " rows; the fused role rectangles together span " +
                            std::to_string(info.geometry.object_shape()[1]));
                require(bank.columns == static_cast<std::int32_t>(info.geometry.object_shape()[2]),
                        name + " rank " + std::to_string(rank) + " view spans " +
                            std::to_string(bank.columns) + " columns; the role rectangle spans " +
                            std::to_string(info.geometry.object_shape()[2]));
                require(bank.codes == base, name + " rank " + std::to_string(rank) +
                                                " code plane is not the shard base");
                require(bank.scales == static_cast<const std::uint8_t*>(base) +
                        entry.scale_plane_offset[static_cast<std::size_t>(rank)],
                        name + " rank " + std::to_string(rank) + " scale plane is not the shard's plane");
                require(reinterpret_cast<const std::uint8_t*>(bank.divisors) ==
                        static_cast<const std::uint8_t*>(base) +
                        entry.divisor_plane_offset[static_cast<std::size_t>(rank)],
                        name + " rank " + std::to_string(rank) + " divisor plane is not the shard's plane");
            } else {
                const auto rows = static_cast<std::int32_t>(total(ranges));
                const auto weight = role_weight(view, parameter->role, parameter->layer);
                require(weight.n == rows, name + " rank " + std::to_string(rank) + " view spans " +
                                              std::to_string(weight.n) + " rows; the shard gives " +
                                              std::to_string(rows));
                require(weight.k == static_cast<std::int32_t>(info.geometry.column_count),
                        name + " rank " + std::to_string(rank) + " view spans " +
                            std::to_string(weight.k) + " columns");
            }
            const auto actual_bytes = device_bytes(base, expected.size());
            require(actual_bytes == expected,
                    "the shard view of " + object.name + " rank " + std::to_string(rank) +
                        " is not the byte sequence the shard map places there");
        }
    }

    for (int rank = 0; rank < 2; ++rank) {
        const auto& view = views.views[static_cast<std::size_t>(rank)];
        require(view.tp == 2, "rank " + std::to_string(rank) + " view does not carry the shard width");
        require(view.layers[kLoadedLayers].post_mixer.routed_gate_up.matrices == 0,
                "rank " + std::to_string(rank) + " fills a layer past the loaded prefix");
    }
    return g_failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    const bool artifact_mode = argc > 1 && std::string_view(argv[1]) == "artifact";
    const bool views_mode    = argc > 1 && std::string_view(argv[1]) == "views";

    try {
        const auto shards = flash::shard_plan_by_name(2);
        test_partition(shards);
        test_expert_subsets(shards);
        if (!artifact_mode && !views_mode) { return g_failures == 0 ? 0 : 1; }

        const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        if (artifact == nullptr || *artifact == '\0') {
            std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
            return 77;
        }

        Reader reader(artifact);
        Binder binder(reader, 2);
        const auto plan = flash::bind_artifact(binder, flash::WeightsProfile::Nvfp4, 1);
        const auto byte_plan = flash::shard_byte_plan(plan, shards, 2);
        const auto by_object = bound_objects(plan, shards);
        test_replicated(shards, by_object, byte_plan);
        test_byte_plan(plan, shards, by_object, byte_plan);
        test_role_blocks(plan, shards, by_object, byte_plan);
        test_real_bytes(plan, shards, reader, byte_plan);
        test_scale_plane(shards, reader, byte_plan);

        if (views_mode) { return test_materialized_views(artifact); }
        if (g_failures != 0) { return 1; }
        std::cout << "flash-next shard byte plan: " << plan.objects.size() << " objects x 2 ranks, "
                  << shards.size() << " roles\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
