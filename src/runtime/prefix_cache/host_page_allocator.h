#pragma once

// Host-tier page allocator interface for the hybrid prefix cache.
//
// HostPageAllocator owns fixed-size pinned host slabs and their eviction metadata. Policy
// decisions (which node/snapshot owns a slab, when it is dead or superseded) live in
// PrefixIndex, which calls mark_dead/mark_alive/supersede/hit/pin.
//
// HostSlabPool is the private per-Engine implementation; a shared host tier can provide another
// implementation of the same contract.

#include <cstddef>
#include <cstdint>

namespace ninfer::prefix_cache {

using HostPageId = std::int32_t;

inline constexpr HostPageId kInvalidHostPage = -1;

// One slab in the host tier. Each slab is a fixed-size pinned buffer.
struct HostSlab {
    void* data = nullptr;
    std::size_t bytes = 0;
    bool occupied = false;
    // In-flight transfer destination/source or shared-cache manifest ownership. The slab is never
    // evicted while any pin is held; `release_pending` defers a release requested while pinned
    // until the last unpin.
    std::uint32_t pin_count = 0;
    bool release_pending = false;
    // Dead-KV: unreachable from any live snapshot; evicted before live slabs.
    bool dead = false;
    std::uint64_t dead_at = 0;
    // Superseded by a newer version of the same state; evicted before live slabs. Sticky until
    // the slab is freed and re-allocated.
    bool superseded = false;
    std::uint64_t superseded_at = 0;
    // GDSF recency/frequency metadata.
    std::uint64_t freq = 0;
    std::uint64_t last_hit_tick = 0;

    [[nodiscard]] bool pinned() const noexcept { return pin_count > 0; }
};

// Result of a slab allocation: the page obtained, and the page that was evicted to make room
// (kInvalidHostPage when none). The caller must reconcile the evicted page's owner before the
// new slab's data is written.
struct SlabAllocation {
    HostPageId page = kInvalidHostPage;
    HostPageId evicted = kInvalidHostPage;
};

class HostPageAllocator {
public:
    virtual ~HostPageAllocator() = default;
    virtual SlabAllocation allocate_block() = 0;
    virtual SlabAllocation allocate_snapshot() = 0;
    virtual void release(HostPageId page) noexcept = 0;
    virtual void pin(HostPageId page) noexcept = 0;
    virtual void unpin(HostPageId page) noexcept = 0;
    virtual void mark_dead(HostPageId page) noexcept = 0;
    virtual void mark_alive(HostPageId page) noexcept = 0;
    virtual void supersede(HostPageId page) noexcept = 0;
    virtual void clear_superseded(HostPageId page) noexcept = 0;
    virtual void hit(HostPageId page, std::uint64_t tick) noexcept = 0;
    virtual const HostSlab* slab(HostPageId page) const noexcept = 0;
    virtual void* data(HostPageId page) const noexcept = 0;
    virtual std::size_t bytes(HostPageId page) const noexcept = 0;
    virtual std::size_t used_bytes() const noexcept = 0;
    virtual bool enabled() const noexcept = 0;
    virtual std::size_t pinned_bytes() const noexcept = 0;
    virtual std::size_t block_slab_count() const noexcept = 0;
    virtual std::size_t snapshot_slab_count() const noexcept = 0;
};

} // namespace ninfer::prefix_cache
