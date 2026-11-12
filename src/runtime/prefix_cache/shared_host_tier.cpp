#include "runtime/prefix_cache/shared_host_tier.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer {

namespace {

bool valid_page(const prefix_cache::HostSlabPool& pool,
                prefix_cache::HostPageId page) noexcept {
    const prefix_cache::HostSlab* slab = pool.slab(page);
    return slab != nullptr && slab->occupied;
}

} // namespace

SharedHostTier::SharedHostTier(Config config)
    : config_(std::move(config)),
      pool_({.total_bytes          = config_.host_bytes,
             .block_slab_bytes     = config_.block_slab_bytes,
             .snapshot_slab_bytes  = config_.snapshot_slab_bytes,
             .block_slab_count     = config_.block_slabs,
             .snapshot_slab_count  = config_.snapshot_slabs,
             .evict_on_exhaustion  = false,
             .count_allocation_failures = false}) {}

std::shared_ptr<SharedHostTier> SharedHostTier::create(Config config) {
    if (config.host_bytes == 0) {
        throw std::invalid_argument("SharedHostTier requires host_bytes > 0");
    }
    return std::shared_ptr<SharedHostTier>(new SharedHostTier(std::move(config)));
}

std::uint64_t SharedHostTier::allocate_owner() {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_owner_++;
}

void SharedHostTier::release_pages_locked(const Entry& entry) noexcept {
    for (const prefix_cache::HostPageId page : entry.block_pages) {
        pool_.release(page);
    }
    pool_.release(entry.state_page);
}

SharedHostTier::ManifestMap::iterator SharedHostTier::erase_locked(
    ManifestMap::iterator it) noexcept {
    const Entry& entry = it->second;
    for (const prefix_cache::HostPageId page : entry.block_pages) {
        pool_.unpin(page);
    }
    pool_.unpin(entry.state_page);
    return manifests_.erase(it);
}

std::uint64_t SharedHostTier::publish(std::uint64_t owner, PublishData data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (data.tp != config_.tp) {
        return 0;
    }
    if (data.block_pages.size() != data.hashes.size()) {
        return 0;
    }
    if (!valid_page(pool_, data.state_page)) {
        return 0;
    }
    for (const prefix_cache::HostPageId page : data.block_pages) {
        if (!valid_page(pool_, page)) {
            return 0;
        }
    }

    Entry entry;
    entry.id           = next_manifest_id_++;
    entry.owner        = owner;
    entry.tp           = data.tp;
    entry.block_tokens = config_.block_tokens;
    entry.frontier     = data.frontier;
    entry.hashes       = std::move(data.hashes);
    entry.block_pages  = std::move(data.block_pages);
    entry.state_page   = data.state_page;
    entry.refs         = 1;
    entry.expires_at   = std::chrono::steady_clock::now() + config_.manifest_ttl;
    entry.freq         = 1;
    entry.last_hit     = ++access_tick_;

    // Register first: emplace may allocate and throw. Pinning after a successful insertion keeps
    // the failure path free of leaked pins because pin() itself is noexcept and every page was
    // validated above.
    const std::uint64_t id = entry.id;
    const auto inserted = manifests_.emplace(id, std::move(entry));
    if (!inserted.second) {
        return 0;
    }
    Entry& stored = inserted.first->second;
    for (const prefix_cache::HostPageId page : stored.block_pages) {
        pool_.pin(page);
    }
    pool_.pin(stored.state_page);
    return id;
}

bool SharedHostTier::retain_manifest(std::uint64_t manifest_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = manifests_.find(manifest_id);
    if (it == manifests_.end()) {
        return false;
    }
    ++it->second.refs;
    return true;
}

bool SharedHostTier::release_manifest(std::uint64_t manifest_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = manifests_.find(manifest_id);
    if (it == manifests_.end() || it->second.refs == 0) {
        return false;
    }
    Entry& entry = it->second;
    --entry.refs;
    if (entry.refs == 0) {
        // The manifest is now evictable, but its pages stay pinned until it is actually erased.
        release_pages_locked(entry);
        return false;
    }
    return true;
}

bool SharedHostTier::erase_manifest(std::uint64_t manifest_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = manifests_.find(manifest_id);
    if (it == manifests_.end() || it->second.refs != 0) {
        return false;
    }
    erase_locked(it);
    return true;
}

void SharedHostTier::retire_owner(std::uint64_t owner) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = manifests_.begin(); it != manifests_.end();) {
        if (it->second.owner == owner && it->second.refs == 0) {
            it = erase_locked(it);
        } else {
            ++it;
        }
    }
}

std::optional<SharedHostTier::ManifestView> SharedHostTier::match(
    std::uint32_t tp, std::span<const std::uint64_t> hashes, std::uint32_t max_frontier) {
    std::lock_guard<std::mutex> lock(mutex_);
    Entry* best = nullptr;
    for (auto& [id, entry] : manifests_) {
        (void)id;
        if (entry.tp != tp || entry.frontier > max_frontier) {
            continue;
        }
        if (entry.hashes.size() > hashes.size()) {
            continue;
        }
        if (!std::equal(entry.hashes.begin(), entry.hashes.end(), hashes.begin())) {
            continue;
        }
        if (best == nullptr || entry.frontier > best->frontier ||
            (entry.frontier == best->frontier &&
             (entry.freq > best->freq ||
              (entry.freq == best->freq && entry.last_hit > best->last_hit)))) {
            best = &entry;
        }
    }
    if (best == nullptr) {
        return std::nullopt;
    }

    ++best->freq;
    best->last_hit = ++access_tick_;

    ManifestView view;
    view.id           = best->id;
    view.tp           = best->tp;
    view.block_tokens = best->block_tokens;
    view.frontier     = best->frontier;
    view.hashes       = best->hashes;
    view.block_pages  = best->block_pages;
    view.state_page   = best->state_page;
    return view;
}

std::size_t SharedHostTier::manifest_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return manifests_.size();
}

std::size_t SharedHostTier::evict_one_locked() noexcept {
    const auto now = std::chrono::steady_clock::now();

    auto expired = manifests_.end();
    auto best    = manifests_.end();
    for (auto it = manifests_.begin(); it != manifests_.end(); ++it) {
        Entry& entry = it->second;
        if (entry.refs != 0) {
            continue;
        }
        if (entry.expires_at <= now) {
            if (expired == manifests_.end() ||
                entry.expires_at < expired->second.expires_at) {
                expired = it;
            }
            continue;
        }
        if (best == manifests_.end() || entry.freq < best->second.freq ||
            (entry.freq == best->second.freq && entry.last_hit < best->second.last_hit)) {
            best = it;
        }
    }

    const auto victim = expired != manifests_.end() ? expired : best;
    if (victim == manifests_.end()) {
        return 0;
    }
    erase_locked(victim);
    return 1;
}

std::size_t SharedHostTier::evict_one() {
    std::lock_guard<std::mutex> lock(mutex_);
    return evict_one_locked();
}

bool SharedHostTier::enabled() const noexcept { return pool_.enabled(); }

std::uint32_t SharedHostTier::tp() const noexcept { return config_.tp; }

std::size_t SharedHostTier::block_slab_bytes() const noexcept {
    return config_.block_slab_bytes;
}

std::size_t SharedHostTier::snapshot_slab_bytes() const noexcept {
    return config_.snapshot_slab_bytes;
}

prefix_cache::SlabAllocation SharedHostTier::allocate_block() {
    std::lock_guard<std::mutex> lock(mutex_);
    prefix_cache::SlabAllocation alloc = pool_.allocate_block();
    if (alloc.page == prefix_cache::kInvalidHostPage) {
        if (evict_one_locked() != 0) {
            alloc = pool_.allocate_block();
        }
        if (alloc.page == prefix_cache::kInvalidHostPage) { ++block_alloc_failures_; }
    }
    return alloc;
}

prefix_cache::SlabAllocation SharedHostTier::allocate_snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    prefix_cache::SlabAllocation alloc = pool_.allocate_snapshot();
    if (alloc.page == prefix_cache::kInvalidHostPage) {
        if (evict_one_locked() != 0) {
            alloc = pool_.allocate_snapshot();
        }
        if (alloc.page == prefix_cache::kInvalidHostPage) { ++snapshot_alloc_failures_; }
    }
    return alloc;
}

void SharedHostTier::release(prefix_cache::HostPageId page) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.release(page);
}

void SharedHostTier::pin(prefix_cache::HostPageId page) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.pin(page);
}

void SharedHostTier::unpin(prefix_cache::HostPageId page) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.unpin(page);
}

void SharedHostTier::mark_dead(prefix_cache::HostPageId page) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.mark_dead(page);
}

void SharedHostTier::mark_alive(prefix_cache::HostPageId page) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.mark_alive(page);
}

void SharedHostTier::supersede(prefix_cache::HostPageId page) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.supersede(page);
}

void SharedHostTier::clear_superseded(prefix_cache::HostPageId page) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.clear_superseded(page);
}

void SharedHostTier::hit(prefix_cache::HostPageId page, std::uint64_t tick) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.hit(page, tick);
}

const prefix_cache::HostSlab* SharedHostTier::slab(
    prefix_cache::HostPageId page) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.slab(page);
}

void* SharedHostTier::data(prefix_cache::HostPageId page) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.data(page);
}

std::size_t SharedHostTier::bytes(prefix_cache::HostPageId page) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.bytes(page);
}

std::size_t SharedHostTier::used_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.used_bytes();
}

std::size_t SharedHostTier::pinned_bytes() const noexcept { return pool_.pinned_bytes(); }

std::size_t SharedHostTier::block_slab_count() const noexcept {
    return pool_.block_slab_count();
}

std::size_t SharedHostTier::snapshot_slab_count() const noexcept {
    return pool_.snapshot_slab_count();
}

std::uint64_t SharedHostTier::block_alloc_failures() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return block_alloc_failures_;
}

std::uint64_t SharedHostTier::snapshot_alloc_failures() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_alloc_failures_;
}

} // namespace ninfer
