// Host-only unit tests for the shared host tier: slab pin refcounting, manifest publish /
// retain / release, prefix matching, and eviction. No GPU and no artifact are required.

#include "runtime/prefix_cache/host_slab_pool.h"
#include "runtime/prefix_cache/shared_host_tier.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace pc = ninfer::prefix_cache;
using ninfer::SharedHostTier;

namespace ninfer {
struct SharedHostTierTestAccess {
    static prefix_cache::HostSlabPool& pool(SharedHostTier& tier) noexcept { return tier.pool(); }
};
} // namespace ninfer

namespace {

pc::HostSlabPool& pool(SharedHostTier& tier) noexcept {
    return ninfer::SharedHostTierTestAccess::pool(tier);
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

SharedHostTier::Config config(std::size_t bytes) {
    SharedHostTier::Config value;
    value.host_bytes           = bytes;
    value.block_slab_bytes     = 64 * 1024;
    value.snapshot_slab_bytes  = 16 * 1024;
    value.block_slabs          = 8;
    value.snapshot_slabs       = 8;
    value.block_tokens         = 64;
    value.tp                   = 1;
    value.manifest_ttl         = std::chrono::minutes(30);
    return value;
}

std::uint64_t hash_for(std::size_t index) {
    return 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(index + 1);
}

std::vector<std::uint64_t> hashes_for(std::size_t blocks) {
    std::vector<std::uint64_t> hashes(blocks);
    for (std::size_t i = 0; i < blocks; ++i) { hashes[i] = hash_for(i); }
    return hashes;
}

// Hashes are indexed, so a manifest whose block list is a prefix of another manifest's block
// list automatically shares that prefix's hashes.
SharedHostTier::PublishData manifest(std::uint32_t frontier,
                                     std::vector<pc::HostPageId> block_pages,
                                     pc::HostPageId state_page, std::uint32_t tp = 1) {
    SharedHostTier::PublishData data;
    data.tp          = tp;
    data.frontier    = frontier;
    data.hashes      = hashes_for(block_pages.size());
    data.block_pages = std::move(block_pages);
    data.state_page  = state_page;
    return data;
}

pc::HostPageId allocate_block(SharedHostTier& tier) {
    return pool(tier).allocate_block().page;
}

pc::HostPageId allocate_snapshot(SharedHostTier& tier) {
    return pool(tier).allocate_snapshot().page;
}

int test_publish_retain_release() {
    int failures = 0;
    auto tier    = SharedHostTier::create(config(1 << 20));

    const pc::HostPageId page  = allocate_block(*tier);
    const pc::HostPageId state = allocate_snapshot(*tier);
    failures += check(page != pc::kInvalidHostPage && state != pc::kInvalidHostPage,
                      "publish test allocations failed");

    const std::uint64_t id = tier->publish(1, manifest(2, {page}, state));
    failures += check(id != 0, "publish returned no manifest id");
    failures += check(tier->manifest_count() == 1, "publish did not record one manifest");

    failures += check(tier->retain_manifest(id), "retain of a live manifest failed");
    failures += check(tier->release_manifest(id), "release must report the remaining source ref");
    failures += check(tier->manifest_count() == 1, "manifest vanished before its last ref");
    failures += check(!tier->release_manifest(id),
                      "release of the last ref must report no remaining source ref");
    failures += check(tier->manifest_count() == 1,
                      "manifest became unevictable state too early (must stay until evicted)");

    failures += check(tier->evict_one() == 1, "evict_one did not evict the released manifest");
    failures += check(tier->manifest_count() == 0, "evicted manifest is still recorded");
    failures += check(pool(*tier).slab(page) != nullptr && !pool(*tier).slab(page)->occupied,
                      "evicted block slab was not returned to the pool");
    failures += check(pool(*tier).slab(state) != nullptr && !pool(*tier).slab(state)->occupied,
                      "evicted snapshot slab was not returned to the pool");
    return failures;
}

int test_eviction_skips_referenced() {
    int failures = 0;
    auto tier    = SharedHostTier::create(config(1 << 20));

    const pc::HostPageId pa = allocate_block(*tier);
    const pc::HostPageId sa = allocate_snapshot(*tier);
    const pc::HostPageId pb = allocate_block(*tier);
    const pc::HostPageId sb = allocate_snapshot(*tier);
    failures += check(pa != pc::kInvalidHostPage && sa != pc::kInvalidHostPage &&
                          pb != pc::kInvalidHostPage && sb != pc::kInvalidHostPage,
                      "eviction test allocations failed");

    const std::uint64_t a = tier->publish(1, manifest(256, {pa}, sa));
    const std::uint64_t b = tier->publish(1, manifest(512, {pb}, sb));
    failures += check(a != 0 && b != 0, "eviction test publishes failed");
    failures += check(tier->retain_manifest(a), "retain of the surviving manifest failed");
    failures += check(!tier->release_manifest(b), "second manifest should have no source refs");
    failures += check(tier->manifest_count() == 2, "both manifests should still be recorded");

    failures += check(tier->evict_one() == 1, "evict_one found no victim");
    failures += check(tier->manifest_count() == 1, "eviction removed the wrong number of manifests");
    failures += check(pool(*tier).slab(pb) != nullptr && !pool(*tier).slab(pb)->occupied,
                      "released manifest's block slab was not freed");
    failures += check(pool(*tier).slab(sb) != nullptr && !pool(*tier).slab(sb)->occupied,
                      "released manifest's snapshot slab was not freed");
    failures += check(pool(*tier).slab(pa) != nullptr && pool(*tier).slab(pa)->occupied &&
                          pool(*tier).slab(pa)->pinned(),
                      "retained manifest's block slab was released");
    failures += check(pool(*tier).slab(sa) != nullptr && pool(*tier).slab(sa)->occupied &&
                          pool(*tier).slab(sa)->pinned(),
                      "retained manifest's snapshot slab was released");

    const auto view = tier->match(1, hashes_for(1), 512);
    failures += check(view.has_value() && view->id == a, "retained manifest is no longer matchable");
    failures += check(tier->evict_one() == 0, "evict_one evicted a referenced manifest");

    failures += check(tier->release_manifest(a), "first release of the retained manifest failed");
    failures += check(!tier->release_manifest(a), "second release should drop the last ref");
    failures += check(tier->evict_one() == 1, "evict_one did not evict the now-released manifest");
    failures += check(tier->manifest_count() == 0, "released manifest is still recorded");
    failures += check(pool(*tier).slab(pa) != nullptr && !pool(*tier).slab(pa)->occupied,
                      "released retained slab was not returned to the pool");
    return failures;
}

int test_match_longest_prefix() {
    int failures = 0;
    auto cfg     = config(1 << 20);
    cfg.block_slabs    = 8;
    cfg.snapshot_slabs = 4;
    auto tier = SharedHostTier::create(cfg);

    std::vector<pc::HostPageId> long_pages;
    for (int i = 0; i < 8; ++i) { long_pages.push_back(allocate_block(*tier)); }
    const std::vector<pc::HostPageId> short_pages(long_pages.begin(), long_pages.begin() + 4);
    const pc::HostPageId short_state = allocate_snapshot(*tier);
    const pc::HostPageId long_state  = allocate_snapshot(*tier);

    const std::uint64_t short_id = tier->publish(1, manifest(256, short_pages, short_state));
    const std::uint64_t long_id  = tier->publish(1, manifest(512, long_pages, long_state));
    failures += check(short_id != 0 && long_id != 0, "match test publishes failed");

    auto eight = tier->match(1, hashes_for(8), 512);
    failures += check(eight.has_value() && eight->id == long_id && eight->frontier == 512,
                      "match did not select the longest prefix");
    failures += check(eight.has_value() && eight->hashes.size() == 8 &&
                          eight->block_pages.size() == 8,
                      "match returned an incomplete manifest view");

    auto two = tier->match(1, hashes_for(2), 512);
    failures += check(!two.has_value(), "match returned a manifest longer than the query prefix");

    auto capped = tier->match(1, hashes_for(8), 256);
    failures += check(capped.has_value() && capped->id == short_id && capped->frontier == 256,
                      "match ignored max_frontier");

    auto wrong_tp = tier->match(2, hashes_for(8), 512);
    failures += check(!wrong_tp.has_value(), "match ignored the tensor-parallel width");

    // The view owns its vectors: mutating it must not corrupt the stored manifest.
    if (eight.has_value()) {
        eight->hashes[0] = 0;
        auto again       = tier->match(1, hashes_for(8), 512);
        failures += check(again.has_value() && again->hashes[0] == hash_for(0),
                          "match exposed alias to the stored manifest vectors");
    }
    return failures;
}

int test_pin_refcount() {
    int failures = 0;
    pc::HostSlabPool pool({.total_bytes           = 1 << 20,
                           .block_slab_bytes     = 64 * 1024,
                           .snapshot_slab_bytes  = 16 * 1024,
                           .block_slab_count     = 2,
                           .snapshot_slab_count  = 1});
    failures += check(pool.enabled(), "pin refcount pool is disabled");

    const pc::SlabAllocation a = pool.allocate_block();
    const pc::SlabAllocation b = pool.allocate_block();
    failures += check(a.page != pc::kInvalidHostPage && b.page != pc::kInvalidHostPage,
                      "pin refcount allocations failed");

    pool.pin(a.page);
    pool.pin(a.page);
    pool.unpin(a.page);
    failures += check(pool.slab(a.page) != nullptr && pool.slab(a.page)->pin_count == 1 &&
                          pool.slab(a.page)->pinned(),
                      "one unpin must leave a doubly-pinned slab pinned");

    // The pool is full; the pinned slab must not be picked as the victim.
    const pc::SlabAllocation c = pool.allocate_block();
    failures += check(c.page != pc::kInvalidHostPage && c.evicted == b.page,
                      "eviction did not skip the pinned slab");

    pool.unpin(a.page);
    failures += check(pool.slab(a.page) != nullptr && pool.slab(a.page)->pin_count == 0 &&
                          !pool.slab(a.page)->pinned(),
                      "last unpin did not clear the pin");

    const pc::SlabAllocation d = pool.allocate_block();
    failures += check(d.page != pc::kInvalidHostPage && d.evicted == a.page,
                      "unpinned slab was not evictable");

    // release() while pinned defers the actual free until the final unpin.
    pc::HostSlabPool deferred({.total_bytes          = 1 << 20,
                               .block_slab_bytes     = 64 * 1024,
                               .snapshot_slab_bytes  = 16 * 1024,
                               .block_slab_count     = 1,
                               .snapshot_slab_count  = 1});
    const pc::SlabAllocation e = deferred.allocate_block();
    failures += check(e.page != pc::kInvalidHostPage, "deferred release allocation failed");
    deferred.pin(e.page);
    deferred.release(e.page);
    failures += check(deferred.slab(e.page) != nullptr && deferred.slab(e.page)->occupied &&
                          deferred.slab(e.page)->release_pending,
                      "release of a pinned slab did not defer the free");
    deferred.unpin(e.page);
    failures += check(deferred.slab(e.page) != nullptr && !deferred.slab(e.page)->occupied,
                      "final unpin did not complete the deferred release");
    return failures;
}

int test_erase_requires_zero_refs() {
    int failures = 0;
    auto tier    = SharedHostTier::create(config(1 << 20));
    const pc::HostPageId page  = allocate_block(*tier);
    const pc::HostPageId state = allocate_snapshot(*tier);
    const std::uint64_t id     = tier->publish(1, manifest(64, {page}, state));
    failures += check(id != 0, "erase test publish failed");

    failures += check(!tier->erase_manifest(id), "erase accepted a manifest with a live ref");
    failures += check(tier->manifest_count() == 1, "failed erase dropped the manifest");
    failures += check(pool(*tier).slab(page) != nullptr && pool(*tier).slab(page)->occupied &&
                          pool(*tier).slab(page)->pinned(),
                      "failed erase partially released the block slab");
    failures += check(pool(*tier).slab(state) != nullptr && pool(*tier).slab(state)->occupied &&
                          pool(*tier).slab(state)->pinned(),
                      "failed erase partially released the snapshot slab");

    failures += check(!tier->release_manifest(id), "release of the only ref must report none left");
    failures += check(tier->erase_manifest(id), "erase rejected a manifest with no refs");
    failures += check(tier->manifest_count() == 0, "successful erase left the manifest recorded");
    failures += check(pool(*tier).slab(page) != nullptr && !pool(*tier).slab(page)->occupied,
                      "successful erase did not return the block slab");
    failures += check(pool(*tier).slab(state) != nullptr && !pool(*tier).slab(state)->occupied,
                      "successful erase did not return the snapshot slab");
    failures += check(!tier->erase_manifest(id), "erase of a missing manifest reported success");
    return failures;
}

int test_release_retain_cycle() {
    int failures = 0;
    auto tier    = SharedHostTier::create(config(1 << 20));
    const pc::HostPageId page  = allocate_block(*tier);
    const pc::HostPageId state = allocate_snapshot(*tier);
    const std::uint64_t id     = tier->publish(1, manifest(64, {page}, state));
    failures += check(id != 0, "release/retain cycle publish failed");

    failures += check(!tier->release_manifest(id), "first release should drop to zero refs");
    failures += check(tier->manifest_count() == 1,
                      "zero-ref manifest must remain until evicted");
    failures += check(pool(*tier).slab(page) != nullptr && pool(*tier).slab(page)->occupied,
                      "zero-ref manifest page was freed before eviction");
    failures += check(pool(*tier).slab(page) != nullptr && pool(*tier).slab(page)->release_pending,
                      "zero-ref manifest page did not enter deferred release");

    const auto matched = tier->match(1, hashes_for(1), 64);
    failures += check(matched.has_value() && matched->id == id,
                      "zero-ref manifest is no longer matchable");
    failures += check(tier->retain_manifest(id), "retain could not revive a zero-ref manifest");
    failures += check(tier->match(1, hashes_for(1), 64).has_value(),
                      "revived manifest is not matchable");
    failures += check(tier->evict_one() == 0, "evict_one evicted a revived manifest");
    failures += check(pool(*tier).slab(page) != nullptr && pool(*tier).slab(page)->occupied,
                      "revived manifest page was freed while referenced");

    failures += check(!tier->release_manifest(id), "second release should drop to zero refs");
    failures += check(tier->manifest_count() == 1,
                      "manifest disappeared before the eviction that frees its pages");
    failures += check(pool(*tier).slab(page) != nullptr && pool(*tier).slab(page)->occupied,
                      "released manifest page was freed before eviction");
    failures += check(tier->evict_one() == 1, "evict_one did not evict the released manifest");
    failures += check(tier->manifest_count() == 0, "evicted manifest is still recorded");
    failures += check(pool(*tier).slab(page) != nullptr && !pool(*tier).slab(page)->occupied,
                      "evicted manifest page was not returned to the pool");
    failures += check(pool(*tier).slab(state) != nullptr && !pool(*tier).slab(state)->occupied,
                      "evicted manifest state page was not returned to the pool");
    return failures;
}

int test_retire_owner() {
    int failures = 0;
    auto tier    = SharedHostTier::create(config(1 << 20));
    constexpr std::uint64_t owner_id = 7;

    const pc::HostPageId pa = allocate_block(*tier);
    const pc::HostPageId sa = allocate_snapshot(*tier);
    const pc::HostPageId pb = allocate_block(*tier);
    const pc::HostPageId sb = allocate_snapshot(*tier);

    const std::uint64_t a = tier->publish(owner_id, manifest(64, {pa}, sa));
    const std::uint64_t b = tier->publish(owner_id, manifest(128, {pb}, sb));
    failures += check(a != 0 && b != 0, "retire_owner publishes failed");
    failures += check(!tier->release_manifest(b), "released manifest should have zero refs");
    failures += check(tier->manifest_count() == 2, "both owner manifests should be recorded");

    tier->retire_owner(owner_id);
    failures += check(tier->manifest_count() == 1,
                      "retire_owner did not delete exactly the zero-ref manifest");
    failures += check(!tier->retain_manifest(b), "retired manifest is still present");
    failures += check(tier->retain_manifest(a), "referenced manifest was retired");
    failures += check(pool(*tier).slab(pb) != nullptr && !pool(*tier).slab(pb)->occupied,
                      "retired manifest's block slab was not freed");
    failures += check(pool(*tier).slab(sb) != nullptr && !pool(*tier).slab(sb)->occupied,
                      "retired manifest's state slab was not freed");
    failures += check(pool(*tier).slab(pa) != nullptr && pool(*tier).slab(pa)->occupied,
                      "referenced manifest's block slab was freed");

    failures += check(tier->release_manifest(a), "release should leave one source ref");
    failures += check(!tier->release_manifest(a), "final release should drop to zero refs");
    failures += check(tier->evict_one() == 1, "evict_one did not evict the released manifest");
    failures += check(tier->manifest_count() == 0, "released manifest is still recorded");
    failures += check(pool(*tier).slab(pa) != nullptr && !pool(*tier).slab(pa)->occupied,
                      "released manifest's block slab was not freed");
    failures += check(pool(*tier).slab(sa) != nullptr && !pool(*tier).slab(sa)->occupied,
                      "released manifest's state slab was not freed");
    return failures;
}

int test_no_cross_owner_eviction() {
    int failures = 0;
    auto cfg     = config(1 << 20);
    cfg.block_slabs    = 2;
    cfg.snapshot_slabs = 1;
    auto tier = SharedHostTier::create(cfg);

    const pc::HostPageId pa = tier->allocate_block().page;
    const pc::HostPageId pb = tier->allocate_block().page;
    failures += check(pa != pc::kInvalidHostPage && pb != pc::kInvalidHostPage,
                      "cross-owner eviction test allocations failed");

    std::memset(pool(*tier).data(pa), 0xAA, pool(*tier).bytes(pa));
    std::memset(pool(*tier).data(pb), 0xBB, pool(*tier).bytes(pb));

    const pc::HostPageId pc_page = tier->allocate_block().page;
    failures += check(pc_page == pc::kInvalidHostPage,
                      "shared tier must not evict another owner's local page");

    failures += check(pool(*tier).slab(pa) != nullptr && pool(*tier).slab(pa)->occupied,
                      "first page must remain occupied after failed allocation");
    failures += check(pool(*tier).slab(pb) != nullptr && pool(*tier).slab(pb)->occupied,
                      "second page must remain occupied after failed allocation");
    failures += check(pool(*tier).data(pa) != nullptr &&
                      static_cast<unsigned char*>(pool(*tier).data(pa))[0] == 0xAA,
                      "first page data must not be overwritten");
    failures += check(pool(*tier).data(pb) != nullptr &&
                      static_cast<unsigned char*>(pool(*tier).data(pb))[0] == 0xBB,
                      "second page data must not be overwritten");
    return failures;
}

int test_pressure_evicts_zero_ref_manifest() {
    int failures = 0;
    auto cfg     = config(1 << 20);
    cfg.block_slabs    = 1;
    cfg.snapshot_slabs = 1;
    auto tier = SharedHostTier::create(cfg);

    const pc::HostPageId pa = tier->allocate_block().page;
    const pc::HostPageId sa = tier->allocate_snapshot().page;
    failures += check(pa != pc::kInvalidHostPage && sa != pc::kInvalidHostPage,
                      "pressure eviction test allocations failed");

    const std::uint64_t id = tier->publish(1, manifest(64, {pa}, sa));
    failures += check(id != 0, "pressure eviction test publish failed");

    // Release the only ref: manifest becomes evictable but pages stay pinned until evicted.
    failures += check(!tier->release_manifest(id), "first release should drop to zero refs");

    // Pool is full; allocation must evict the zero-ref manifest and succeed.
    const pc::HostPageId pb = tier->allocate_block().page;
    failures += check(pb != pc::kInvalidHostPage,
                      "allocation under pressure must evict the zero-ref manifest");

    // A retained manifest must not be evicted.
    const pc::HostPageId sb = tier->allocate_snapshot().page;
    const std::uint64_t id2 = tier->publish(1, manifest(64, {pb}, sb));
    failures += check(id2 != 0, "second publish failed");
    failures += check(tier->retain_manifest(id2), "retain failed");
    failures += check(tier->release_manifest(id2), "first release should leave one ref");
    // Pool still full with a pinned retained manifest; no evictable victim.
    const pc::HostPageId pc_page = tier->allocate_block().page;
    failures += check(pc_page == pc::kInvalidHostPage,
                      "allocation must fail when all manifests are pinned");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_publish_retain_release();
    failures += test_eviction_skips_referenced();
    failures += test_match_longest_prefix();
    failures += test_pin_refcount();
    failures += test_erase_requires_zero_refs();
    failures += test_release_retain_cycle();
    failures += test_retire_owner();
    failures += test_no_cross_owner_eviction();
    failures += test_pressure_evicts_zero_ref_manifest();
    if (failures == 0) {
        std::printf("All shared-host-tier unit tests passed.\n");
        return 0;
    }
    std::printf("%d shared-host-tier check(s) failed.\n", failures);
    return 1;
}
