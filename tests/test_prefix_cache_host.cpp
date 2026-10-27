// Unit tests for the M2 host tier: HostSlabPool allocation/eviction and PrefixIndex
// host-residency state transitions. No GPU required. Checks are evaluated unconditionally
// (no assert), so they hold in NDEBUG builds as well.

#include "core/paged_kv_cache.h"
#include "runtime/prefix_cache/host_slab_pool.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/shared_host_tier.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace pc = ninfer::prefix_cache;

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

struct TestSink : pc::PrefixPageSink {
    int released_blocks = 0;
    int released_tails  = 0;
    int released_slots  = 0;
    void release_block_pages(const pc::PoolPages&) noexcept override { ++released_blocks; }
    void release_tail_pages(const pc::PoolPages&) noexcept override { ++released_tails; }
    void release_snapshot_slot(pc::SlotId) noexcept override { ++released_slots; }
};

pc::HostSlabPool make_test_pool() {
    // 4 MiB budget; the slab data (4 x 512 KiB blocks + 2 x 128 KiB snapshots = 2.25 MiB) fits.
    return pc::HostSlabPool({.total_bytes = 4 * 1024 * 1024,
                             .block_slab_bytes = 512 * 1024,
                             .snapshot_slab_bytes = 128 * 1024,
                             .block_slab_count = 4,
                             .snapshot_slab_count = 2});
}

// PagedKVPool only performs pointer arithmetic in the paths under test, so a host buffer can back
// it: the accounting and eviction tests never touch CUDA.
struct TestPagedPool {
    std::vector<unsigned char> backing;
    ninfer::PagedKVPoolLayout layout;
    std::unique_ptr<ninfer::PagedKVPool> pool;
};

TestPagedPool make_test_paged_pool(std::uint32_t pages) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(
        builder, {.page_group_count      = pages,
                  .logical_page_capacity = pages,
                  .table_rows            = 1,
                  .plane_order           = ninfer::PagedKVPlaneOrder::PageMajor,
                  .planes                = {{ninfer::DType::BF16, 16, 1}}});
    const std::size_t bytes = builder.finish(256);

    TestPagedPool fixture;
    fixture.backing.assign(bytes + 256U, 0);
    const auto address  = reinterpret_cast<std::uintptr_t>(fixture.backing.data());
    const auto aligned  = (address + 255U) & ~static_cast<std::uintptr_t>(255U);
    fixture.layout      = std::move(layout);
    fixture.pool        = std::make_unique<ninfer::PagedKVPool>(
        ninfer::DeviceSpan{reinterpret_cast<void*>(aligned), bytes}, fixture.layout);
    return fixture;
}

template <typename Fn>
int run_with_watchdog(Fn&& fn, const char* label) {
    auto future = std::async(std::launch::async, std::forward<Fn>(fn));
    if (future.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
        std::cerr << label << " timed out (possible eviction spin)\n";
        std::_Exit(1);
    }
    future.get();
    return 0;
}

std::vector<std::int32_t> block_tokens(std::int32_t seed) {
    std::vector<std::int32_t> tokens(pc::kBlockTokens);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = seed + static_cast<std::int32_t>(i);
    }
    return tokens;
}

pc::NodeId commit_block(pc::PrefixIndex& index, pc::NodeId parent, std::int32_t seed,
                        std::int32_t text_page) {
    auto tokens = block_tokens(seed);
    const pc::BlockNode* parent_node = index.node(parent);
    const std::uint64_t parent_hash =
        parent_node != nullptr ? parent_node->lookup_hash : pc::kBlockHashSeed;
    bool adopted = false;
    const pc::NodeId node = index.commit_block(
        pc::BlockCommit{.parent = parent,
                        .lookup_hash = pc::block_hash(parent_hash, tokens, 0),
                        .extra = 0,
                        .tokens = tokens,
                        .pages = {std::array<std::int32_t, pc::kMaxPools>{text_page, -1}}},
        adopted);
    return node;
}

pc::SnapshotId add_tap(pc::PrefixIndex& index, pc::NodeId anchor, std::uint32_t frontier,
                       std::uint64_t priority = 5) {
    return index.add_snapshot(
        pc::SnapshotRequest{.anchor = anchor,
                            .frontier = frontier,
                            .tail_tokens = {},
                            .tail_pages = {},
                            .kind = pc::SnapshotKind::Tap,
                            .priority = priority});
}

void write_block_to_host(pc::PrefixIndex& index, pc::NodeId node) {
    const auto target = index.begin_host_write(node);
    (void)target;
    index.finish_host_write(node, true);
}

void write_snapshot_to_host(pc::PrefixIndex& index, pc::SnapshotId id) {
    const auto target = index.begin_snapshot_write(id);
    (void)target;
    index.finish_snapshot_write(id, true);
}

// --- HostSlabPool tests ---

int test_host_allocator_contract() {
    int failures = 0;
    auto pool = make_test_pool();
    pc::HostPageAllocator& alloc = pool;
    const auto before = alloc.used_bytes();
    auto a = alloc.allocate_block();
    failures += check(a.page >= 0, "block allocation");
    failures += check(alloc.used_bytes() > before, "allocation must occupy a slab");
    alloc.release(a.page);
    failures += check(alloc.used_bytes() == before, "release must return the slab");
    return failures;
}

int test_slab_pool_basic() {
    int failures = 0;
    pc::HostSlabPool pool = make_test_pool();
    failures += check(pool.enabled(), "pool should be enabled after construction");

    pc::SlabAllocation blk = pool.allocate_block();
    failures += check(blk.page != pc::kInvalidHostPage, "block slab allocation failed");
    void* data = pool.data(blk.page);
    failures += check(data != nullptr, "allocated slab has no data pointer");
    if (data != nullptr) { std::memset(data, 0xAB, pool.bytes(blk.page)); }

    pc::SlabAllocation snap = pool.allocate_snapshot();
    failures += check(snap.page != pc::kInvalidHostPage, "snapshot slab allocation failed");
    failures += check(pool.data(snap.page) != nullptr, "snapshot slab has no data pointer");

    pool.release(blk.page);
    pool.release(snap.page);
    return failures;
}

int test_slab_pool_eviction() {
    int failures = 0;
    pc::HostSlabPool pool({.total_bytes = 2 * 1024 * 1024,
                           .block_slab_bytes = 512 * 1024,
                           .snapshot_slab_bytes = 64 * 1024,
                           .block_slab_count = 2,
                           .snapshot_slab_count = 1});
    failures += check(pool.enabled(), "pool should be enabled after construction");

    pc::SlabAllocation a = pool.allocate_block();
    pc::SlabAllocation b = pool.allocate_block();
    failures += check(a.page != pc::kInvalidHostPage && b.page != pc::kInvalidHostPage,
                      "initial block slab allocations failed");

    // Pool is full; allocating a third should evict the oldest (a).
    pc::SlabAllocation c = pool.allocate_block();
    failures += check(c.page != pc::kInvalidHostPage, "third block slab allocation failed");
    failures += check(c.evicted == a.page, "LRU slab was not evicted");
    return failures;
}

int test_slab_pool_gdsf() {
    int failures = 0;
    pc::HostSlabPool pool({.total_bytes = 2 * 1024 * 1024,
                           .block_slab_bytes = 512 * 1024,
                           .snapshot_slab_bytes = 64 * 1024,
                           .block_slab_count = 3,
                           .snapshot_slab_count = 1});
    failures += check(pool.enabled(), "pool should be enabled after construction");

    pc::SlabAllocation a = pool.allocate_block();
    pc::SlabAllocation b = pool.allocate_block();
    pc::SlabAllocation c = pool.allocate_block();
    failures += check(a.page != pc::kInvalidHostPage && b.page != pc::kInvalidHostPage &&
                          c.page != pc::kInvalidHostPage,
                      "initial block slab allocations failed");

    pool.mark_dead(a.page);

    // Allocating a fourth block should evict the dead slab first.
    pc::SlabAllocation d = pool.allocate_block();
    failures += check(d.page != pc::kInvalidHostPage, "fourth block slab allocation failed");
    failures += check(d.evicted == a.page, "dead slab was not evicted first");
    return failures;
}

// --- PrefixIndex host-residency tests ---

int test_prefix_index_host_residency() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_allocator(&pool);

    const pc::NodeId node = commit_block(index, index.root(), 42, 100);
    failures += check(node != pc::kInvalidNode, "block commit failed");
    const pc::BlockNode* record = index.node(node);
    failures += check(record != nullptr, "committed node is missing");
    if (record != nullptr) {
        failures += check(record->device_state == pc::Residency::Resident,
                          "committed node is not device-resident");
        failures += check(record->host_state == pc::Residency::Absent,
                          "fresh node unexpectedly has a host copy");
    }

    pc::HostWriteTarget wt = index.begin_host_write(node);
    failures += check(wt.node == node && wt.page != pc::kInvalidHostPage,
                      "begin_host_write did not reserve a slab");
    record = index.node(node);
    failures += check(record != nullptr && record->host_state == pc::Residency::Filling,
                      "node is not Filling during write-through");

    index.finish_host_write(node, true);
    record = index.node(node);
    failures += check(record != nullptr && record->host_state == pc::Residency::Host,
                      "write-through did not land the host copy");
    return failures;
}

int test_device_fill_roundtrip() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_allocator(&pool);

    const pc::NodeId node = commit_block(index, index.root(), 7, 100);
    failures += check(node != pc::kInvalidNode, "block commit failed");
    write_block_to_host(index, node);

    // Demote the device copy: the node is backed, so eviction keeps it host-only.
    const pc::NodeId evicted = index.evict_device_lru();
    failures += check(evicted == node, "backed node was not demoted");
    const pc::BlockNode* record = index.node(node);
    failures += check(record != nullptr && record->device_state == pc::Residency::Absent &&
                          record->host_state == pc::Residency::Host,
                      "demoted node should be host-only");

    // Restore: fill the device copy from the host slab.
    index.begin_device_fill(node, {std::array<std::int32_t, pc::kMaxPools>{200, -1}});
    record = index.node(node);
    failures += check(record != nullptr && record->device_state == pc::Residency::Filling,
                      "node is not Filling during restore");
    index.finish_device_fill(node, true);
    record = index.node(node);
    failures += check(record != nullptr && record->device_state == pc::Residency::Resident &&
                          record->device_pages.page[0] == 200,
                      "restore did not make the node device-resident");

    // Rollback path: demote again, start a fill, abort it.
    (void)index.evict_device_lru();
    const int releases_before = sink.released_blocks;
    index.begin_device_fill(node, {std::array<std::int32_t, pc::kMaxPools>{201, -1}});
    index.finish_device_fill(node, false);
    record = index.node(node);
    failures += check(record != nullptr && record->device_state == pc::Residency::Absent,
                      "aborted restore did not roll the node back");
    failures += check(sink.released_blocks == releases_before + 1,
                      "aborted restore did not return its pages");
    return failures;
}

int test_evict_demote_vs_erase() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_allocator(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId b = commit_block(index, a, 2, 101);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode, "block commits failed");
    const pc::SnapshotId snap = add_tap(index, b, 128);
    failures += check(snap != pc::kInvalidSnapshot, "tap snapshot was rejected");

    // Unbacked subtree: eviction erases the oldest node and everything below it.
    const int releases_before = sink.released_blocks;
    const int slots_before    = sink.released_slots;
    const pc::NodeId evicted  = index.evict_device_lru();
    failures += check(evicted == a, "LRU did not evict the oldest node");
    failures += check(index.node(a) == nullptr && index.node(b) == nullptr,
                      "unbacked subtree was not erased");
    failures += check(index.snapshot(snap) == nullptr,
                      "snapshot anchored in the erased subtree survived");
    failures += check(sink.released_blocks == releases_before + 2,
                      "erased nodes did not return their pages");
    failures += check(sink.released_slots == slots_before + 1,
                      "erased snapshot did not return its slot");
    return failures;
}

int test_cache_page_accounting() {
    int failures = 0;
    auto fixture = make_test_paged_pool(6);
    ninfer::PagedKVPool& pool = *fixture.pool;

    const std::uint32_t before = pool.mapped_pages();
    const auto pages           = pool.acquire_cache_pages(2);
    failures += check(pool.mapped_pages() == before,
                      "cache pages must not count as allocation-mapped");
    failures += check(pool.cached_pages() == 2, "acquired pages were not counted as cache-owned");

    pool.release_cache_pages(pages);
    failures += check(pool.mapped_pages() == before,
                      "releasing cache pages must not change mapped count");
    failures += check(pool.cached_pages() == 0, "released pages are still counted as cache-owned");
    return failures;
}

int test_evict_skips_pinned_subtree() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_allocator(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId b = commit_block(index, a, 2, 101);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode, "block commits failed");

    // b is deeper in the subtree than the LRU head a; its in-flight transfer pins the subtree.
    index.pin_restore(b);

    pc::NodeId first = pc::kInvalidNode;
    failures += run_with_watchdog([&] { first = index.evict_device_lru(); },
                                  "evict_device_lru with a pinned descendant");
    failures += check(first == pc::kInvalidNode,
                      "pinned subtree must yield no eviction, not a phantom victim");
    failures += check(index.node(a) != nullptr && index.node(b) != nullptr,
                      "partially erased a subtree containing a pinned descendant");
    failures += check(sink.released_blocks == 0, "phantom eviction released block pages");

    // Once the pin is gone the same node is a real victim and the whole subtree goes.
    index.unpin_restore(b);
    pc::NodeId second = pc::kInvalidNode;
    failures += run_with_watchdog([&] { second = index.evict_device_lru(); },
                                  "evict_device_lru after the pin was released");
    failures += check(second == a, "unpinned LRU head was not evicted");
    failures += check(index.node(a) == nullptr && index.node(b) == nullptr,
                      "unpinned subtree was not erased");
    failures += check(sink.released_blocks == 2,
                      "erased subtree did not release every block page");
    return failures;
}

int test_slab_eviction_reconciliation() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    // Exactly two block slabs so the third write-through must evict one.
    pc::HostSlabPool pool({.total_bytes = 2 * 1024 * 1024,
                           .block_slab_bytes = 512 * 1024,
                           .snapshot_slab_bytes = 64 * 1024,
                           .block_slab_count = 2,
                           .snapshot_slab_count = 1});
    index.set_host_allocator(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId b = commit_block(index, index.root(), 2, 101);
    const pc::NodeId c = commit_block(index, index.root(), 3, 102);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode &&
                          c != pc::kInvalidNode,
                      "block commits failed");

    write_block_to_host(index, a);
    write_block_to_host(index, b);
    const pc::HostPageId slab_a = index.node(a)->host_page;
    failures += check(slab_a != pc::kInvalidHostPage, "first write-through did not land");

    // The pool is full; writing c through evicts the oldest slab (a's). Node a is still
    // device-resident, so it survives as device-only.
    write_block_to_host(index, c);
    const pc::BlockNode* na = index.node(a);
    const pc::BlockNode* nc = index.node(c);
    failures += check(na != nullptr && na->host_state == pc::Residency::Absent &&
                          na->host_page == pc::kInvalidHostPage,
                      "evicted slab was not reconciled off the owner");
    failures += check(nc != nullptr && nc->host_state == pc::Residency::Host,
                      "new write-through did not land");

    // Now demote a (device-only) and force its... a has no host copy; instead demote c's
    // sibling b: b still owns a slab? No - b's slab was never evicted. Demote b to host-only,
    // then evict b's slab by writing through a fourth node.
    (void)index.evict_device_lru(); // demotes the oldest backed node (b or a; a has no host copy)
    const pc::BlockNode* nb = index.node(b);
    failures += check(nb != nullptr && nb->host_state == pc::Residency::Host,
                      "b should be host-backed before slab eviction");
    if (nb != nullptr && nb->device_state == pc::Residency::Resident) {
        (void)index.evict_device_lru();
    }
    const pc::HostPageId slab_b = index.node(b) != nullptr ? index.node(b)->host_page
                                                           : pc::kInvalidHostPage;
    const pc::NodeId d = commit_block(index, index.root(), 4, 103);
    failures += check(d != pc::kInvalidNode, "fourth block commit failed");
    write_block_to_host(index, d);
    // b was host-only; losing its slab leaves it with no data anywhere, so it is deleted.
    failures += check(index.node(b) == nullptr,
                      "host-only node was not deleted when its slab was evicted");
    (void)slab_b;
    return failures;
}

int test_slot_demote_host_backed() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 1}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_allocator(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId b = commit_block(index, a, 2, 101);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode, "block commits failed");

    const pc::SnapshotId snap_a = add_tap(index, a, 64, 10);
    failures += check(snap_a != pc::kInvalidSnapshot, "first snapshot was rejected");
    write_snapshot_to_host(index, snap_a);

    // Fill the only remaining slot with a second snapshot.
    const pc::SnapshotId snap_b = add_tap(index, b, 128, 10);
    failures += check(snap_b != pc::kInvalidSnapshot, "second snapshot was rejected");

    // No free slot: reserving one must evict a victim. snap_a has a host copy, so it is
    // demoted (kept host-resident) rather than deleted.
    const pc::SlotId slot = index.reserve_snapshot_slot(10);
    failures += check(slot != pc::kInvalidSlot, "slot reservation found no victim");
    const pc::Snapshot* ra = index.snapshot(snap_a);
    failures += check(ra != nullptr, "host-backed snapshot was deleted instead of demoted");
    if (ra != nullptr) {
        failures += check(ra->device_slot == pc::kInvalidSlot &&
                              ra->device_state == pc::Residency::Absent &&
                              ra->host_state == pc::Residency::Host,
                          "demoted snapshot should be host-only");
    }
    index.release_reserved_slot(slot);
    return failures;
}

int test_snapshot_restore_roundtrip() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_allocator(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId b = commit_block(index, a, 2, 101);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode, "block commits failed");

    const pc::SnapshotId snap_a = add_tap(index, a, 64, 10);
    const pc::SnapshotId snap_b = add_tap(index, b, 128, 10);
    failures += check(snap_a != pc::kInvalidSnapshot && snap_b != pc::kInvalidSnapshot,
                      "snapshot admission failed");
    write_snapshot_to_host(index, snap_a);

    // Exhaust the slots so snap_a (host-backed) is demoted.
    const pc::SlotId slot = index.reserve_snapshot_slot(10);
    failures += check(slot != pc::kInvalidSlot, "slot reservation found no victim");
    index.release_reserved_slot(slot);
    const pc::Snapshot* ra = index.snapshot(snap_a);
    failures += check(ra != nullptr && ra->device_state == pc::Residency::Absent,
                      "snap_a should be demoted to host-only");

    // Restore snap_a's image into a device slot.
    const auto target = index.begin_snapshot_restore(snap_a);
    failures += check(target.slot != pc::kInvalidSlot &&
                          target.page != pc::kInvalidHostPage,
                      "snapshot restore did not reserve a slot");
    index.finish_snapshot_restore(snap_a, true);
    ra = index.snapshot(snap_a);
    failures += check(ra != nullptr && ra->device_resident() &&
                          ra->device_slot != pc::kInvalidSlot,
                      "restored snapshot is not device-resident");

    // Rollback path: demote again, start a restore, abort it.
    const pc::SlotId slot2 = index.reserve_snapshot_slot(10);
    failures += check(slot2 != pc::kInvalidSlot, "second slot reservation found no victim");
    index.release_reserved_slot(slot2);
    const auto aborted = index.begin_snapshot_restore(snap_a);
    failures += check(aborted.slot != pc::kInvalidSlot, "second restore did not reserve a slot");
    index.finish_snapshot_restore(snap_a, false);
    ra = index.snapshot(snap_a);
    failures += check(ra != nullptr && ra->device_state == pc::Residency::Absent &&
                          ra->host_state == pc::Residency::Host,
                      "aborted restore did not roll the snapshot back");
    return failures;
}

int test_best_host_admission() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_allocator(&pool);

    const auto block0 = block_tokens(1);
    const auto block1 = block_tokens(1000);
    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    failures += check(a != pc::kInvalidNode, "block commit failed");
    const pc::SnapshotId snap = add_tap(index, a, 64, 5);
    failures += check(snap != pc::kInvalidSnapshot, "tap snapshot was rejected");
    write_block_to_host(index, a);
    write_snapshot_to_host(index, snap);

    // Demote the block: the path is now host-only, so device admission must fail while host
    // admission still finds the snapshot.
    (void)index.evict_device_lru();
    const pc::BlockNode* record = index.node(a);
    failures += check(record != nullptr && record->host_state == pc::Residency::Host &&
                          record->device_state == pc::Residency::Absent,
                      "block should be host-only after demotion");

    std::vector<std::int32_t> prompt(block0.begin(), block0.end());
    prompt.insert(prompt.end(), block1.begin(), block1.end());
    const auto hashes = pc::chain_block_hashes(prompt);
    const pc::PrefixMatch match = index.match(prompt, hashes, {});
    failures += check(match.matched_blocks == 1, "match did not walk the committed block");
    failures += check(match.best_device == pc::kInvalidSnapshot,
                      "device-resident admission should fail on a demoted path");
    failures += check(match.best_host == snap && match.best_host_frontier == 64,
                      "host-resident admission did not select the snapshot");
    return failures;
}

int test_dead_kv_and_supersede_marking() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 1}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_allocator(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    failures += check(a != pc::kInvalidNode, "block commit failed");
    write_block_to_host(index, a);
    const pc::HostPageId slab_a = index.node(a)->host_page;
    failures += check(slab_a != pc::kInvalidHostPage, "write-through did not land");

    // While a snapshot reaches the node, its slab stays alive.
    const pc::SnapshotId snap = add_tap(index, a, 64, 10);
    failures += check(snap != pc::kInvalidSnapshot, "tap snapshot was rejected");
    failures += check(pool.slab(slab_a) != nullptr && !pool.slab(slab_a)->dead,
                      "slab marked dead while a snapshot still reaches it");

    // Removing the snapshot leaves the block unreachable: the slab becomes dead-KV.
    index.remove_snapshot(snap);
    failures += check(pool.slab(slab_a) != nullptr && pool.slab(slab_a)->dead,
                      "slab not marked dead after its last snapshot left");

    // Supersede: with one slot, adding a second snapshot that names the first as superseded
    // marks the victim's slab superseded and demotes it (it has a host copy).
    const pc::NodeId b = commit_block(index, index.root(), 2, 101);
    failures += check(b != pc::kInvalidNode, "second block commit failed");
    const pc::SnapshotId s1 = add_tap(index, b, 64, 10);
    failures += check(s1 != pc::kInvalidSnapshot, "supersede victim snapshot was rejected");
    write_snapshot_to_host(index, s1);
    const pc::HostPageId slab_s1 = index.snapshot(s1) != nullptr
                                       ? index.snapshot(s1)->host_page
                                       : pc::kInvalidHostPage;
    failures += check(slab_s1 != pc::kInvalidHostPage &&
                          !pool.slab(slab_s1)->superseded,
                      "victim slab marked superseded too early");

    const pc::NodeId c = commit_block(index, b, 3, 102);
    failures += check(c != pc::kInvalidNode, "third block commit failed");
    const pc::SnapshotId s2 = index.add_snapshot(
        pc::SnapshotRequest{.anchor = c,
                            .frontier = 192,
                            .tail_tokens = {},
                            .tail_pages = {},
                            .kind = pc::SnapshotKind::Tap,
                            .priority = 5,
                            .supersede = s1});
    failures += check(s2 != pc::kInvalidSnapshot, "superseding snapshot was rejected");
    failures += check(pool.slab(slab_s1) != nullptr && pool.slab(slab_s1)->superseded,
                      "victim slab not marked superseded");
    const pc::Snapshot* victim = index.snapshot(s1);
    failures += check(victim != nullptr && victim->device_state == pc::Residency::Absent &&
                          victim->host_state == pc::Residency::Host,
                      "superseded snapshot with a host copy should be demoted, not deleted");

    // A hit clears the superseded mark on both the snapshot and its slab.
    index.hit(s1);
    failures += check(pool.slab(slab_s1) != nullptr && !pool.slab(slab_s1)->superseded,
                      "hit did not clear the superseded mark");
    return failures;
}

} // namespace

int test_overlapping_manifests_share_block_page() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);

    ninfer::SharedHostTier::Config cfg;
    cfg.host_bytes          = 1 << 20;
    cfg.block_slab_bytes    = 64 * 1024;
    cfg.snapshot_slab_bytes = 16 * 1024;
    cfg.block_slabs         = 4;
    cfg.snapshot_slabs      = 4;
    cfg.tp                  = 1;
    cfg.manifest_ttl        = std::chrono::minutes(30);
    auto tier = ninfer::SharedHostTier::create(cfg);
    index.set_host_allocator(tier.get());
    index.set_shared_host_tier(tier.get());

    const pc::NodeId node_a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId node_b = commit_block(index, node_a, 2, 101);
    failures += check(node_a != pc::kInvalidNode && node_b != pc::kInvalidNode,
                      "block commit failed");

    write_block_to_host(index, node_a);
    write_block_to_host(index, node_b);

    const pc::SnapshotId tap1 = add_tap(index, node_a, 64);
    failures += check(tap1 != pc::kInvalidSnapshot, "tap1 failed");
    write_snapshot_to_host(index, tap1);

    const pc::SnapshotId tap2 = add_tap(index, node_b, 128);
    failures += check(tap2 != pc::kInvalidSnapshot, "tap2 failed");
    write_snapshot_to_host(index, tap2);

    const std::uint64_t m1 = index.publish_snapshot_manifest(tap1, 1, 1, 1);
    const std::uint64_t m2 = index.publish_snapshot_manifest(tap2, 1, 1, 1);
    failures += check(m1 != 0 && m2 != 0, "manifest publish failed");

    const pc::BlockNode* record_a = index.node(node_a);
    failures += check(record_a != nullptr && record_a->host_shared &&
                      record_a->shared_manifests.size() == 2,
                      "node_a must hold both manifest ids");

    index.remove_snapshot(tap1);
    record_a = index.node(node_a);
    failures += check(record_a != nullptr && record_a->host_shared &&
                      record_a->shared_manifests.size() == 1 &&
                      record_a->shared_manifests[0] == m2,
                      "node_a must still be protected by manifest2 after manifest1 release");

    index.remove_snapshot(tap2);
    record_a = index.node(node_a);
    failures += check(record_a != nullptr && !record_a->host_shared &&
                      record_a->shared_manifests.empty(),
                      "node_a must be cleared after both manifests released");
    return failures;
}

int test_import_foreign() {
    int failures = 0;
    ninfer::SharedHostTier::Config tier_config;
    tier_config.host_bytes          = 1 << 20;
    tier_config.block_slab_bytes    = 64 * 1024;
    tier_config.snapshot_slab_bytes = 16 * 1024;
    tier_config.block_slabs         = 8;
    tier_config.snapshot_slabs      = 8;
    tier_config.tp                  = 1;
    auto tier = ninfer::SharedHostTier::create(tier_config);
    TestSink sink;
    pc::PrefixIndex index({.device_snapshot_slots = 2}, &sink);
    index.set_shared_host_tier(tier.get());
    index.set_host_allocator(tier.get());

    // The hashes must be the real chain hashes, otherwise `find_child` cannot locate the committed
    // nodes. Block 0 has a local node (rule 6); block 1 only exists on the foreign chain.
    const auto tokens0 = block_tokens(10);
    const auto tokens1 = block_tokens(20);
    std::vector<std::uint64_t> hashes(2);
    hashes[0] = pc::block_hash(pc::kBlockHashSeed, tokens0, 0);
    hashes[1] = pc::block_hash(hashes[0], tokens1, 0);
    const pc::HostPageId page0 = tier->allocate_block().page;
    const pc::HostPageId page1 = tier->allocate_block().page;
    const pc::HostPageId state = tier->allocate_snapshot().page;

    ninfer::SharedHostTier::PublishData data;
    data.tp          = 1;
    data.frontier    = 128;
    data.hashes      = hashes;
    data.block_pages = {page0, page1};
    data.state_page  = state;
    const std::uint64_t manifest = tier->publish(1, std::move(data));
    failures += check(manifest != 0, "foreign test publish failed");

    // A hint naming a manifest the tier does not hold cannot be imported.
    pc::PrefixIndex::ForeignView bad;
    bad.manifest_id  = manifest + 1000U;
    bad.frontier     = 128;
    bad.hashes       = hashes;
    bad.block_pages  = {page0, page1};
    bad.state_page   = state;
    failures += check(!index.import_foreign(bad), "import must fail when retain fails");

    pc::PrefixIndex::ForeignView view;
    view.manifest_id  = manifest;
    view.frontier     = 128;
    view.hashes       = hashes;
    view.block_pages  = {page0, page1};
    view.state_page   = state;
    failures += check(index.import_foreign(view), "import_foreign failed");
    failures += check(index.live_snapshot_count() == 1,
                      "import must count the foreign snapshot as live");

    const pc::NodeId foreign0 = index.foreign_node(manifest, 0);
    const pc::NodeId foreign1 = index.foreign_node(manifest, 1);
    const pc::BlockNode* node0 = index.node(foreign0);
    const pc::BlockNode* node1 = index.node(foreign1);
    failures += check(node0 != nullptr && node1 != nullptr, "foreign nodes missing");
    failures += check(node0 != nullptr && node0->foreign && node0->host_shared &&
                          node0->host_state == pc::Residency::Host &&
                          node0->device_state == pc::Residency::Absent &&
                          node0->host_page == page0 && node0->depth == 0 &&
                          node0->shared_manifests.size() == 1 &&
                          node0->shared_manifests[0] == manifest,
                      "foreign node 0 state is wrong");
    failures += check(node1 != nullptr && node1->foreign && node1->depth == 1 &&
                          node1->parent == foreign0,
                      "foreign chain parent/depth is wrong");

    // Foreign nodes are not in the local tree: the local match sees nothing, and the device LRU
    // has nothing to evict.
    const pc::PrefixMatch local = index.match({}, hashes, {});
    failures += check(local.matched_blocks == 0, "foreign chain must not be locally matchable");
    failures += check(index.evict_device_lru() == pc::kInvalidNode,
                      "foreign node must not be in the local LRU");

    // The manifest ref import_foreign took is still held: pages stay pinned and are not
    // release-pending through the allocator.
    failures += check(tier->slab(page1) != nullptr && tier->slab(page1)->pinned() &&
                          !tier->slab(page1)->release_pending,
                      "import must not release a manifest-owned page through the allocator");

    // A device fill on a foreign node keeps the manifest reference.
    index.begin_device_fill(foreign1, pc::PoolPages{{std::array<std::int32_t, pc::kMaxPools>{7, -1}}});
    index.finish_device_fill(foreign1, true);
    node1 = index.node(foreign1);
    failures += check(node1 != nullptr && node1->foreign && node1->host_shared &&
                          node1->host_page == page1 &&
                          node1->device_state == pc::Residency::Resident,
                      "foreign node must stay manifest-owned after a device fill");

    // Rule 6: block 0 already has a local node, so the local node keeps its own storage; block 1
    // has no local match, so the filled foreign node becomes the local node.
    const pc::NodeId local0 = commit_block(index, index.root(), 10, 3);
    failures += check(local0 != pc::kInvalidNode, "local block 0 commit failed");

    std::vector<std::int32_t> all_tokens;
    all_tokens.insert(all_tokens.end(), tokens0.begin(), tokens0.end());
    all_tokens.insert(all_tokens.end(), tokens1.begin(), tokens1.end());
    failures += check(index.commit_foreign(manifest, std::span<const std::int32_t>(all_tokens)),
                      "commit_foreign failed");

    const pc::ChildMatch replay0 = index.find_child(index.root(), hashes[0], tokens0, 0);
    const pc::ChildMatch replay1 = index.find_child(local0, hashes[1], tokens1, 0);
    const pc::NodeId local1 = replay1.node;
    failures += check(replay0.node == local0 && local0 != foreign0,
                      "rule 6 must keep the existing local node for block 0");
    failures += check(replay1.node == local1 &&
                          index.node(local1)->device_pages.page[0] == 7,
                      "committed foreign block 1 must carry its device pages into the local tree");
    const pc::BlockNode* committed = index.node(local1);
    failures += check(committed != nullptr && !committed->foreign && !committed->host_shared &&
                          committed->host_page == pc::kInvalidHostPage &&
                          committed->host_state == pc::Residency::Absent &&
                          committed->device_state == pc::Residency::Resident &&
                          committed->live,
                      "committed node must drop the manifest host reference");

    // The foreign snapshot is now a local snapshot anchored at the deepest committed node, and the
    // manifest ref import took is still held until release_foreign.
    const pc::SnapshotId snap = index.foreign_snapshot(manifest);
    const pc::Snapshot* record = index.snapshot(snap);
    failures += check(record != nullptr && !record->foreign && !record->host_shared &&
                          record->shared_manifest_id == 0 &&
                          record->host_page == pc::kInvalidHostPage && record->live &&
                          record->anchor == local1,
                      "committed foreign snapshot must stop being manifest-owned");

    // The retained ref is still held: pages stay pinned, the manifest is still recorded.
    failures += check(tier->manifest_count() == 1, "manifest must survive until release_foreign");
    failures += check(tier->slab(page1) != nullptr && tier->slab(page1)->pinned(),
                      "manifest pages must stay pinned until release_foreign");

    // `release_foreign` performs the single release for the ref `import_foreign` took, so the
    // publisher ref is the last one and the manifest becomes evictable.
    index.release_foreign(manifest);
    failures += check(tier->manifest_count() == 1, "release_foreign must not erase the manifest");
    failures += check(!tier->release_manifest(manifest), "the last ref must reach zero");
    failures += check(tier->evict_one() == 1, "the released manifest must be evictable");
    failures += check(tier->slab(page1) != nullptr && !tier->slab(page1)->occupied,
                      "evicted manifest page must return to the pool");

    // Bookkeeping: the committed snapshot is still live in the local table, so the count stays 1;
    // detaching it returns to zero without wrapping.
    failures += check(index.live_snapshot_count() == 1,
                      "committed foreign snapshot must still be counted live");
    index.remove_snapshot(snap);
    failures += check(index.live_snapshot_count() == 0,
                      "detaching the committed snapshot must return the count to zero");

    // Cancel path: import then release without commit deletes the snapshot and the count returns
    // to zero.
    std::vector<std::uint64_t> cancel_hashes = {hashes[0]};
    std::vector<pc::HostPageId> cancel_pages = {tier->allocate_block().page};
    const pc::HostPageId cancel_state = tier->allocate_snapshot().page;
    ninfer::SharedHostTier::PublishData cancel_data;
    cancel_data.tp          = 1;
    cancel_data.frontier    = 64;
    cancel_data.hashes      = cancel_hashes;
    cancel_data.block_pages = cancel_pages;
    cancel_data.state_page  = cancel_state;
    const std::uint64_t cancel_manifest = tier->publish(1, std::move(cancel_data));
    pc::PrefixIndex::ForeignView cancel_view;
    cancel_view.manifest_id  = cancel_manifest;
    cancel_view.frontier     = 64;
    cancel_view.hashes       = cancel_hashes;
    cancel_view.block_pages  = cancel_pages;
    cancel_view.state_page   = cancel_state;
    failures += check(index.import_foreign(cancel_view), "cancel-path import failed");
    failures += check(index.live_snapshot_count() == 1, "cancel-path import must count the snapshot");
    index.release_foreign(cancel_manifest);
    failures += check(index.live_snapshot_count() == 0,
                      "cancel-path release must return the count to zero");
    return failures;
}

// probe_device: longest local device-resident prefix along a hash chain, 64-aligned; stops at the
// first missing or non-resident block. No GPU: the committed nodes carry raw page ids and the
// residency transitions are pure bookkeeping.
int test_probe_device() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index({.device_snapshot_slots = 1}, &sink);

    std::vector<std::int32_t> all;
    for (std::int32_t seed = 10; seed <= 30; seed += 10) {
        const auto tokens = block_tokens(seed);
        all.insert(all.end(), tokens.begin(), tokens.end());
    }
    const std::vector<std::uint64_t> hashes = pc::chain_block_hashes(std::span<const std::int32_t>(all));
    failures += check(hashes.size() == 3, "three full blocks must chain three hashes");
    failures += check(index.probe_device(hashes) == 0, "cold index must probe an empty frontier");

    auto fixture = make_test_paged_pool(4);
    const std::vector<std::int32_t> pages = fixture.pool->acquire_cache_pages(3);
    pc::NodeId parent = index.root();
    for (std::size_t block = 0; block < hashes.size(); ++block) {
        const std::span<const std::int32_t> tokens =
            std::span<const std::int32_t>(all).subspan(block * pc::kBlockTokens, pc::kBlockTokens);
        bool adopted = false;
        const pc::NodeId node = index.commit_block(
            pc::BlockCommit{.parent = parent,
                            .lookup_hash = hashes[block],
                            .extra = 0,
                            .tokens = tokens,
                            .pages = {std::array<std::int32_t, pc::kMaxPools>{pages[block], -1}}},
            adopted);
        failures += check(node != pc::kInvalidNode && adopted, "probe test commit failed");
        parent = node;
    }
    failures += check(index.probe_device(hashes) == 3U * static_cast<std::uint32_t>(pc::kBlockTokens),
                      "three committed resident blocks must probe the full frontier");
    failures += check(index.probe_device(hashes) % static_cast<std::uint32_t>(pc::kBlockTokens) == 0,
                      "probe frontier must be block aligned");

    // Evicting the LRU head (block 0, unbacked, no host allocator) erases its subtree; the probe
    // stops at the root even though blocks 1 and 2 were resident.
    failures += check(index.evict_device_lru() != pc::kInvalidNode,
                      "unbacked LRU head must be evictable");
    failures += check(index.probe_device(hashes) == 0,
                      "probe must stop at the first missing block after eviction");

    // Re-committing only the head advances the frontier to exactly one block.
    const std::span<const std::int32_t> head_tokens =
        std::span<const std::int32_t>(all).subspan(0, pc::kBlockTokens);
    bool adopted = false;
    const pc::NodeId head = index.commit_block(
        pc::BlockCommit{.parent = index.root(),
                        .lookup_hash = hashes[0],
                        .extra = 0,
                        .tokens = head_tokens,
                        .pages = {std::array<std::int32_t, pc::kMaxPools>{pages[0], -1}}},
        adopted);
    failures += check(head != pc::kInvalidNode && adopted, "head re-commit failed");
    failures += check(index.probe_device(hashes) == static_cast<std::uint32_t>(pc::kBlockTokens),
                      "probe must stop at the first missing block, not skip to a deeper one");

    // A chain shorter than the committed path reports only its own blocks.
    failures += check(index.probe_device(std::span<const std::uint64_t>(hashes).subspan(0, 1)) ==
                          static_cast<std::uint32_t>(pc::kBlockTokens),
                      "probe over a shorter chain must report only its own blocks");
    return failures;
}

int main() {
    int failures = 0;
    failures += test_probe_device();
    failures += test_host_allocator_contract();
    failures += test_slab_pool_basic();
    failures += test_slab_pool_eviction();
    failures += test_slab_pool_gdsf();
    failures += test_prefix_index_host_residency();
    failures += test_device_fill_roundtrip();
    failures += test_evict_demote_vs_erase();
    failures += test_cache_page_accounting();
    failures += test_evict_skips_pinned_subtree();
    failures += test_slab_eviction_reconciliation();
    failures += test_slot_demote_host_backed();
    failures += test_snapshot_restore_roundtrip();
    failures += test_best_host_admission();
    failures += test_overlapping_manifests_share_block_page();
    failures += test_import_foreign();
    if (failures == 0) {
        std::printf("All host-tier unit tests passed.\n");
        return 0;
    }
    std::printf("%d host-tier check(s) failed.\n", failures);
    return 1;
}
