#pragma once

// Pinned host memory slab pool for the hybrid prefix cache host tier (M2).
//
// Holds KV blocks and snapshot images in pinned host RAM. Allocations are fixed-size slabs:
//   - Block slabs: one per KV page group (64 tokens); sized from the real pool layout per rank.
//   - Snapshot slabs: one per state image (GDN recurrent + conv history); sized from the real
//     linear-attention layout per rank.
//
// The pool is storage only: it carves a pinned allocation into slabs and tracks occupancy plus
// GDSF eviction metadata. All policy decisions (which node/snapshot owns a slab, when a slab is
// dead or superseded) live in PrefixIndex, which calls mark_dead/mark_alive/supersede/hit/pin.
// Eviction order inside the pool: slabs with a non-zero pin count are never touched; among the
// rest, dead slabs first (oldest), then superseded slabs (oldest), then by (frequency, recency) —
// GreedyDual-Size-Frequency.

#include "runtime/prefix_cache/host_page_allocator.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::prefix_cache {

struct HostSlabPoolOptions {
    // Total pinned memory in bytes. 0 = disabled (device-only mode).
    std::size_t total_bytes = 0;
    // Size of one KV block slab in bytes (per rank).
    std::size_t block_slab_bytes = 0;
    // Size of one snapshot image slab in bytes (per rank).
    std::size_t snapshot_slab_bytes = 0;
    // Number of block slabs to carve.
    std::size_t block_slab_count = 0;
    // Number of snapshot slabs to carve.
    std::size_t snapshot_slab_count = 0;
    // When true, allocation evicts an unpinned slab if the pool is full.
    // When false, allocation returns kInvalidHostPage on exhaustion instead of evicting.
    // Shared tiers use false so that cross-owner local pages are never evicted.
    bool evict_on_exhaustion = true;
    // When false, failed allocations are not counted. A shared tier that retries after evicting a
    // manifest counts only its own final failure, so its pool must not count the first attempt.
    bool count_allocation_failures = true;
};

class HostSlabPool : public HostPageAllocator {
public:
    explicit HostSlabPool(HostSlabPoolOptions options);
    ~HostSlabPool() override;

    HostSlabPool(const HostSlabPool&) = delete;
    HostSlabPool& operator=(const HostSlabPool&) = delete;

    [[nodiscard]] bool enabled() const noexcept override { return host_memory_ != nullptr; }
    // Requested budget from the caller; may exceed the bytes actually pinned.
    [[nodiscard]] std::size_t total_bytes() const noexcept { return options_.total_bytes; }
    // Bytes actually pinned for slab data (excludes the heap-side HostSlab metadata).
    [[nodiscard]] std::size_t pinned_bytes() const noexcept override { return pinned_bytes_; }
    [[nodiscard]] std::size_t used_bytes() const noexcept override;
    [[nodiscard]] std::size_t free_bytes() const noexcept;
    [[nodiscard]] std::size_t block_slab_count() const noexcept override {
        return block_slabs_.size();
    }
    [[nodiscard]] std::size_t snapshot_slab_count() const noexcept override {
        return snapshot_slabs_.size();
    }
    [[nodiscard]] std::uint64_t block_alloc_failures() const noexcept override {
        return block_alloc_failures_;
    }
    [[nodiscard]] std::uint64_t snapshot_alloc_failures() const noexcept override {
        return snapshot_alloc_failures_;
    }

    // Allocate a block slab, evicting another BLOCK slab if the pool is full. Returns
    // kInvalidHostPage when no block slab can be obtained (all pinned).
    [[nodiscard]] SlabAllocation allocate_block() override;
    // Allocate a snapshot slab, evicting another SNAPSHOT slab if the pool is full.
    [[nodiscard]] SlabAllocation allocate_snapshot() override;

    // Release a slab back to the free list. Resets all eviction metadata. When the slab is
    // still pinned, the release is deferred until the last unpin.
    void release(HostPageId page) noexcept override;

    // Raw pointer and size of a slab's data (for D2H/H2D transfers).
    [[nodiscard]] void* data(HostPageId page) const noexcept override;
    [[nodiscard]] std::size_t bytes(HostPageId page) const noexcept override;

    // --- eviction metadata (driven by PrefixIndex policy) ---

    // Protect a slab from eviction while a transfer is in flight or a shared manifest owns it.
    // Pins nest: each unpin releases one reference, and a deferred release completes on the last.
    void pin(HostPageId page) noexcept override;
    void unpin(HostPageId page) noexcept override;

    // Mark a slab dead (no live snapshot below its node) or alive again.
    void mark_dead(HostPageId page) noexcept override;
    void mark_alive(HostPageId page) noexcept override;

    // Mark a slab superseded by a newer version (sticky until re-allocated).
    void supersede(HostPageId page) noexcept override;

    // Clear the superseded mark after the owner is hit again.
    void clear_superseded(HostPageId page) noexcept override;

    // Record a hit: bumps frequency and recency. Does not clear dead/superseded.
    void hit(HostPageId page, std::uint64_t tick) noexcept override;

    // Slab metadata access.
    [[nodiscard]] const HostSlab* slab(HostPageId page) const noexcept override;
    [[nodiscard]] HostSlab* slab_mut(HostPageId page) noexcept;

private:
    HostPageId find_free(std::size_t offset, std::size_t count) noexcept;
    // Pick a victim among slabs [offset, offset+count): dead first (oldest), then superseded
    // (oldest), then (min freq, oldest tick). Pinned and unoccupied slabs are skipped.
    HostPageId pick_victim(std::size_t offset, std::size_t count) noexcept;
    void occupy(HostPageId page) noexcept;
    void reset_slab(HostSlab& slab) noexcept;

    HostSlabPoolOptions options_;
    // Slab metadata is heap-side; only slab DATA lives in the pinned allocation.
    std::vector<HostSlab> block_slabs_;
    std::vector<HostSlab> snapshot_slabs_;
    void* host_memory_ = nullptr;
    std::size_t pinned_bytes_ = 0;
    std::uint64_t block_alloc_failures_ = 0;
    std::uint64_t snapshot_alloc_failures_ = 0;
    std::uint64_t tick_ = 0;
};

} // namespace ninfer::prefix_cache
