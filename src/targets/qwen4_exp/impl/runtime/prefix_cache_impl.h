#pragma once

// Ported and adapted from Neroued/ninfer, commit 40dec5ff
// (https://github.com/Neroued/ninfer), Apache License, Version 2.0.
// The upstream qwen4_exp Session checkpoints/state-image/reclaim algorithms were re-expressed
// against this fork's PrefixIndex/SharedHostTier host cache: KV pages are published as 64-token
// blocks, the continuation state as one page-aligned endpoint snapshot, both into pinned host
// slabs. The consumer side restores those slabs into device KV/continuation state and prefills
// only the suffix tokens.

#include <ninfer/types.h>
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/shared_host_tier.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail {

namespace pc = ninfer::prefix_cache;

// Device KV pages in qwen4_exp are fixed slices of TextState's cache planes, not pool-allocated
// pages. Dropping the index's reference therefore needs no release: the next request's reset
// reuses those slices after the pinned host copy has been made.
class PrefixCacheSink final : public pc::PrefixPageSink {
public:
    void release_block_pages(const pc::PoolPages&) noexcept override {}
    void release_tail_pages(const pc::PoolPages&) noexcept override {}
    void release_snapshot_slot(pc::SlotId) noexcept override {}
};

struct PrefixPublishResult {
    std::vector<pc::NodeId> block_nodes;
    pc::NodeId anchor          = pc::kInvalidNode;
    pc::SnapshotId snapshot    = pc::kInvalidSnapshot;
    std::uint32_t frontier     = 0;
};

// Commits every full 64-token block of `tokens` named by `hashes` and publishes one page-aligned
// endpoint snapshot on the deepest block. `reserved_slot` (when valid) is settled here: the new
// snapshot consumes it, and every other outcome returns it to the index's free list.
// Block page ids are the logical QSA page indices, which this runtime maps one-to-one onto physical
// pages through the identity page table until the next state reset.
inline PrefixPublishResult commit_prefix_blocks(pc::PrefixIndex& index,
                                                std::span<const std::uint64_t> hashes,
                                                std::span<const TokenId> tokens,
                                                pc::SlotId reserved_slot) {
    PrefixPublishResult result;
    const auto release_reserved = [&]() noexcept {
        if (reserved_slot != pc::kInvalidSlot) { index.release_reserved_slot(reserved_slot); }
    };
    const std::size_t blocks = std::min<std::size_t>(hashes.size(), tokens.size() / pc::kBlockTokens);
    pc::NodeId parent = index.root();
    for (std::size_t block = 0; block < blocks; ++block) {
        pc::BlockCommit commit;
        commit.parent     = parent;
        commit.lookup_hash = hashes[block];
        commit.tokens     = tokens.subspan(block * pc::kBlockTokens, pc::kBlockTokens);
        commit.pages.page[0] = static_cast<std::int32_t>(block);
        bool adopted = false;
        parent = index.commit_block(commit, adopted);
        if (parent == pc::kInvalidNode) {
            release_reserved();
            return {};
        }
        result.block_nodes.push_back(parent);
    }
    result.anchor   = parent;
    result.frontier = static_cast<std::uint32_t>(blocks * pc::kBlockTokens);
    if (result.anchor == pc::kInvalidNode || result.frontier == 0) {
        release_reserved();
        return result;
    }

    pc::SnapshotRequest request;
    request.anchor        = result.anchor;
    request.frontier      = result.frontier;
    request.kind          = pc::SnapshotKind::Endpoint;
    request.reserved_slot = reserved_slot;
    result.snapshot = index.add_snapshot(request);
    if (result.snapshot == pc::kInvalidSnapshot) {
        release_reserved();
    } else if (reserved_slot != pc::kInvalidSlot) {
        const pc::Snapshot* published = index.snapshot(result.snapshot);
        if (published == nullptr || published->device_slot != reserved_slot) { release_reserved(); }
    }
    return result;
}

// Same contract as the released Engine prefix probe: a 64-token frontier from local residency and
// a 64-token frontier plus manifest id from the process-level pinned host tier.
inline PrefixProbe probe_prefix(const pc::PrefixIndex& index, SharedHostTier* tier,
                                std::uint32_t tp, std::span<const std::uint64_t> hashes) {
    PrefixProbe probe;
    probe.block_tokens    = static_cast<std::uint32_t>(pc::kBlockTokens);
    probe.device_frontier = index.probe_device(hashes);
    if (tier != nullptr) {
        const auto manifest = tier->match(
            tp, hashes, static_cast<std::uint32_t>(hashes.size()) * probe.block_tokens);
        if (manifest.has_value()) {
            probe.manifest_frontier = manifest->frontier;
            probe.manifest_id       = manifest->id;
        }
    }
    return probe;
}

struct RetainedPrefix {
    bool valid     = false;
    std::uint32_t frontier = 0;
    pc::SnapshotId snapshot = pc::kInvalidSnapshot;
    std::uint32_t generation = 0;
    std::uint64_t manifest_id = 0;
    std::vector<std::uint64_t> hashes;
};

// Holds a shared-tier manifest alive for the duration of a host-to-device prefix restore. The
// publisher's own reference is independent; releasing this lease does not disturb publication.
class ManifestLease {
public:
    ManifestLease(SharedHostTier& tier, std::uint64_t id) noexcept
        : tier_(tier), id_(id), held_(tier.retain_manifest(id)) {}
    ~ManifestLease() noexcept {
        if (held_) { (void)tier_.release_manifest(id_); }
    }
    ManifestLease(const ManifestLease&) = delete;
    ManifestLease& operator=(const ManifestLease&) = delete;
    [[nodiscard]] bool held() const noexcept { return held_; }

private:
    SharedHostTier& tier_;
    std::uint64_t id_;
    bool held_;
};

} // namespace ninfer::targets::qwen4_exp::detail
