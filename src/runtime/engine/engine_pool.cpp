#include "runtime/engine/engine_pool.h"

#include "runtime/engine/shared_model.h"
#include "runtime/prefix_cache/shared_host_tier.h"

#include <algorithm>
#include <cassert>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {

using Clock = std::chrono::steady_clock;

namespace {

// Production instance view: the pool reads one Engine through its published probe and capacity
// boundary and never mutates it.
class EngineInstanceView final : public PoolInstanceView {
public:
    explicit EngineInstanceView(Engine& engine) : engine_(engine) {}

    PrefixProbe probe(std::span<const std::uint64_t> hashes) const override {
        return engine_.prefix_probe(hashes);
    }

    CapacitySnapshot capacity() const override { return engine_.capacity_snapshot(); }

private:
    Engine& engine_;
};

} // namespace

void Permit::release() noexcept {
    if (released_) { return; }
    released_ = true;
    if (pool_ == nullptr) { return; }
    {
        std::lock_guard lock(pool_->mutex_);
        auto& held = pool_->permits_by_instance_[instance_];
        const auto it = std::find(held.begin(), held.end(), this);
        if (it != held.end()) { held.erase(it); }
    }
    pool_->cv_.notify_all();
}

void Permit::record_completion(std::uint64_t prefix_hit_tokens,
                              std::uint64_t manifest_restore_tokens) noexcept {
    if (pool_ == nullptr) { return; }
    pool_->record_completion(instance_, prefix_hit_tokens, manifest_restore_tokens);
}

PoolHandle::PoolHandle(PoolHandle&&) noexcept            = default;
PoolHandle& PoolHandle::operator=(PoolHandle&&) noexcept = default;

void PoolHandle::attach(std::shared_ptr<Permit> permit, std::uint32_t instance,
                        PrefixReuseHint hint, std::uint64_t pending_id) noexcept {
    permit_     = std::move(permit);
    instance_   = instance;
    hint_       = hint;
    pending_id_ = pending_id;
}

const ResolvedSamplingParameters& PoolHandle::resolved_sampling() const {
    static const ResolvedSamplingParameters kEmpty{};
    return handle_ != nullptr ? handle_->resolved_sampling() : kEmpty;
}

void PoolHandle::attach(GenerationHandle&& handle) noexcept {
    handle_ = std::make_unique<GenerationHandle>(std::move(handle));
}

// The Engine handle goes first, so the Engine drops that request's lane and pages before the pool
// gives them back.
void PoolHandle::reset() noexcept {
    handle_.reset();
    if (permit_ != nullptr) { permit_->release(); }
}

GenerationResult PoolHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (handle_ == nullptr) {
        if (permit_ != nullptr) { permit_->release(); }
        return {};
    }
    std::shared_ptr<Permit> permit = std::move(permit_);
    GenerationResult result      = handle_->wait(sink, cancellation);
    if (permit != nullptr) {
        const std::uint64_t manifest_restore =
            hint_.manifest_id != 0 &&
                    result.prefix_reuse_reason == PrefixReuseReason::ReusedBlockSnapshot
                ? static_cast<std::uint64_t>(hint_.frontier)
                : 0;
        permit->record_completion(result.reused_prompt_tokens, manifest_restore);
        permit->release();
    }
    return result;
}

EnginePool::EnginePool(Options options) : options_(std::move(options)) {
    initialize();

    shared_model_ = std::make_shared<SharedModel>(options_.groups[0].engine.artifact_path);
    for (std::size_t index = 0; index < instance_options_.size(); ++index) {
        EngineOptions instance = instance_options_[index];
        instance.shared_model = shared_model_;
        if (index == 0) {
            instance.create_shared_host_tier = instance.prefix_cache.host_cache_mib > 0;
        } else {
            instance.shared_host_tier      = shared_host_tier_;
            instance.create_shared_host_tier = false;
        }
        // The shared reader is not safe for concurrent materialization, so the Engines are built in
        // instance order here and released in reverse order by the destructor.
        engines_.push_back(std::make_unique<Engine>(std::move(instance)));
        if (index == 0) { shared_host_tier_ = engines_[0]->shared_host_tier(); }
        views_.push_back(std::make_unique<EngineInstanceView>(*engines_[index]));
    }
}

EnginePool::EnginePool(Options options, std::vector<std::unique_ptr<PoolInstanceView>> views)
    : options_(std::move(options)) {
    initialize();
    if (views.size() != instance_ids_.size()) {
        throw std::invalid_argument("engine pool test views must match its groups");
    }
    views_ = std::move(views);
}

EnginePool::~EnginePool() {
    {
        std::lock_guard lock(mutex_);
        // Admission callers must all have returned before the pool dies: an entry left in the queue
        // would mean a waiter is still blocked on this mutex.
        assert(pending_.empty());
        stopping_ = true;
    }
    cv_.notify_all();
    // Views reference Engines; drop the views first, then release the Engines in reverse
    // construction order.
    views_.clear();
    for (auto it = engines_.rbegin(); it != engines_.rend(); ++it) { it->reset(); }
}

void EnginePool::initialize() {
    validate_groups();
    normalize_pending();

    tp_ = options_.groups[0].tp;
    config_.affinity_slack_tokens = options_.affinity_slack_tokens;
    config_.fairness_aging        = options_.fairness_aging;
    router_ = runtime::EngineRouter(config_);

    for (std::size_t index = 0; index < options_.groups.size(); ++index) {
        instance_ids_.push_back(static_cast<std::uint32_t>(index));
    }
    permits_by_instance_.assign(instance_ids_.size(), {});
    completion_by_instance_.assign(instance_ids_.size(), {});
    provider_ = [this](const runtime::PendingRequest& pending, std::uint32_t instance) {
        return candidate_locked(pending, instance);
    };
}

void EnginePool::validate_groups() const {
    if (options_.groups.empty()) {
        throw std::invalid_argument("engine pool needs at least one group");
    }
    const std::int32_t tp = options_.groups[0].tp;
    if (tp != 1 && tp != 2 && tp != 4) {
        throw std::invalid_argument("engine pool group tp must be 1, 2, or 4");
    }

    std::set<int> devices;
    for (const auto& group : options_.groups) {
        if (group.engine.artifact_path != options_.groups[0].engine.artifact_path) {
            throw std::invalid_argument("every engine pool group must use the same artifact");
        }
        if (group.tp != tp) {
            throw std::invalid_argument("every engine pool group must use the same tensor-parallel width");
        }
        const std::vector<int> group_devices =
            group.devices.empty() ? std::vector<int>{group.engine.device} : group.devices;
        if (group_devices.size() != static_cast<std::size_t>(tp)) {
            throw std::invalid_argument("engine pool group device count must equal tp");
        }
        for (const int device : group_devices) {
            if (!devices.insert(device).second) {
                throw std::invalid_argument("engine pool groups must not share a device");
            }
        }
    }
    if (devices.size() != options_.groups.size() * static_cast<std::size_t>(tp)) {
        throw std::invalid_argument("engine pool device count does not match groups * tp");
    }
}

void EnginePool::normalize_pending() {
    for (auto& group : options_.groups) {
        group.engine.devices = group.devices.empty() ? std::vector<int>{group.engine.device} : group.devices;
        group.engine.tp      = group.tp;
        // Multi-group: the pool queue is the only admission queue, so each Engine keeps exactly the
        // one request already routed to it. A single group keeps its own pending behavior.
        if (options_.groups.size() > 1) { group.engine.max_pending_requests = 1; }
        instance_options_.push_back(group.engine);
    }
}

// The pool's candidate view for one (pending, instance) pair. Lane occupancy is the pool's own
// authority: one held permit is one lane claim. Pages are the instance's reported free pages minus
// the pages the pool reserved for permits that instance has not yet mapped, so a permit is counted
// once and the view never over-admits.
runtime::RouteCandidate EnginePool::candidate_locked(const runtime::PendingRequest& pending,
                                                     std::uint32_t instance) const {
    CapacitySnapshot capacity = views_[instance]->capacity();
    const auto& held = permits_by_instance_[instance];
    const std::size_t mapped =
        std::min(held.size(), static_cast<std::size_t>(capacity.active_requests));
    std::uint32_t reserved = 0;
    for (std::size_t index = mapped; index < held.size(); ++index) {
        reserved += held[index]->required_pages();
    }
    capacity.free_pages = capacity.free_pages > reserved ? capacity.free_pages - reserved : 0;
    capacity.pending_requests = 0;
    capacity.active_requests  = static_cast<std::uint32_t>(held.size());
    return runtime::RouteCandidate{instance, views_[instance]->probe(pending.block_hashes), capacity};
}

std::optional<runtime::RouteDecision> EnginePool::pick_locked() const {
    std::vector<runtime::PendingRequest> view;
    view.reserve(pending_.size());
    for (const auto& entry : pending_) { view.push_back(entry->route); }
    return router_.pick(view, provider_, instance_ids_, Clock::now());
}

void EnginePool::apply_locked(const std::shared_ptr<Entry>& entry,
                              const runtime::RouteDecision& decision) {
    if (entry->routed) { return; }
    auto permit = std::make_shared<Permit>(*this, decision.instance, decision.reserved_pages);
    entry->permit   = permit;
    entry->instance = decision.instance;
    if (decision.use_hint && decision.manifest_frontier > 0) {
        entry->hint = PrefixReuseHint{
            .manifest_id  = decision.manifest_id,
            .frontier     = decision.manifest_frontier,
            .tp           = tp_,
            .block_tokens = config_.block_tokens,
        };
    }
    entry->routed = true;
    erase_locked(entry);
    permits_by_instance_[decision.instance].push_back(permit.get());
}

void EnginePool::erase_locked(const std::shared_ptr<Entry>& entry) {
    const auto it = std::find(pending_.begin(), pending_.end(), entry);
    if (it != pending_.end()) { pending_.erase(it); }
}

std::uint32_t EnginePool::home_locked(std::span<const std::uint64_t> hashes) const {
    std::uint32_t best_instance = 0;
    std::uint32_t best_frontier = 0;
    for (std::uint32_t instance = 0; instance < views_.size(); ++instance) {
        const std::uint32_t frontier = views_[instance]->probe(hashes).device_frontier;
        if (frontier > best_frontier) {
            best_frontier   = frontier;
            best_instance   = instance;
        }
    }
    return best_frontier == 0 ? 0 : best_instance;
}

std::uint32_t EnginePool::required_pages_for(std::uint32_t prompt_tokens,
                                                std::uint32_t output_tokens) const {
    const std::uint64_t tokens =
        static_cast<std::uint64_t>(prompt_tokens) + output_tokens + config_.safety_margin_tokens;
    return static_cast<std::uint32_t>((tokens + config_.block_tokens - 1) / config_.block_tokens);
}

PoolHandle EnginePool::admit(runtime::PendingRequest pending,
                              std::chrono::steady_clock::time_point pending_deadline) {
    std::unique_lock lock(mutex_);
    if (stopping_) {
        throw RequestError(RequestErrorKind::Unavailable, "engine pool is unavailable");
    }
    if (options_.global_pending_requests == 0 ||
        pending_.size() >= options_.global_pending_requests) {
        throw RequestError(RequestErrorKind::Overloaded, "engine pool pending queue is full");
    }

    const auto submitted = Clock::now();
    if (pending_deadline == Clock::time_point{}) {
        pending_deadline = submitted + std::chrono::milliseconds(instance_options_[0].pending_timeout_ms);
    }
    if (submitted >= pending_deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "inference request expired before admission");
    }

    auto entry = std::make_shared<Entry>();
    entry->hashes.assign(pending.block_hashes.begin(), pending.block_hashes.end());
    entry->route           = pending;
    entry->route.block_hashes = entry->hashes;
    entry->deadline        = pending_deadline;
    if (!entry->route.has_home) {
        const std::uint32_t home = home_locked(entry->hashes);
        if (home != 0) {
            entry->route.has_home     = true;
            entry->route.home_instance = home;
        }
    }
    entry->route.id = next_pending_id_++;
    pending_.push_back(entry);

    for (;;) {
        const auto decision = pick_locked();
        if (decision.has_value()) {
            const auto target = pending_[decision->pending_index];
            apply_locked(target, *decision);
            if (target == entry) { break; }
            // The decision belongs to another waiter; its own call finishes that permit.
            cv_.notify_all();
            continue;
        }
        const auto now = Clock::now();
        if (now >= entry->deadline) {
            erase_locked(entry);
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired while waiting for a pool permit");
        }
        const auto remaining = std::min<std::chrono::milliseconds>(
            std::chrono::duration_cast<std::chrono::milliseconds>(entry->deadline - now),
            std::chrono::milliseconds(25));
        cv_.wait_for(lock, remaining, [&] { return stopping_ || entry->routed; });
        if (entry->routed) { break; }
    }

    PoolHandle handle;
    handle.attach(entry->permit, entry->instance, entry->hint, entry->route.id);
    lock.unlock();
    return handle;
}

PoolHandle EnginePool::submit(PreparedPrompt prompt, RequestOptions options,
                              std::chrono::steady_clock::time_point pending_deadline) {
    if (engines_.empty()) {
        throw std::logic_error("engine pool has no engine to execute a prepared prompt");
    }

    const auto submitted = Clock::now();
    const auto deadline = pending_deadline == Clock::time_point{}
                              ? submitted + std::chrono::milliseconds(instance_options_[0].pending_timeout_ms)
                              : pending_deadline;

    runtime::PendingRequest pending;
    pending.prompt_tokens  = prompt.token_count();
    pending.max_tokens     = options.execution.requested_output_tokens;
    pending.required_pages = required_pages_for(pending.prompt_tokens, pending.max_tokens);
    pending.enqueued       = submitted;
    pending.block_hashes   = prompt.block_hashes();

    PoolHandle handle = admit(std::move(pending), deadline);
    Engine& engine = *engines_[handle.instance()];

    RequestOptions routed = options;
    routed.execution.prefix_reuse_hint = handle.hint();
    try {
        GenerationHandle generation = engine.submit(std::move(prompt), std::move(routed), deadline);
        handle.attach(std::move(generation));
    } catch (...) {
        handle.reset();
        throw;
    }
    return handle;
}

PreparedPrompt EnginePool::prepare(PromptInput input, const PreparationControl& control) const {
    if (engines_.empty()) {
        throw std::logic_error("engine pool has no engine to prepare a prompt");
    }
    return engines_[0]->prepare(std::move(input), control);
}

PreparedPrompt EnginePool::prepare_tokens(std::vector<TokenId> token_ids,
                                          bool allow_prefix_identity) const {
    if (engines_.empty()) {
        throw std::logic_error("engine pool has no engine to prepare a prompt");
    }
    return engines_[0]->prepare_tokens(std::move(token_ids), allow_prefix_identity);
}

MetricsSnapshot aggregate_instance_metrics(std::span<const PoolInstanceMetrics> instances) {
    MetricsSnapshot out;
    for (const PoolInstanceMetrics& row : instances) {
        const MetricsSnapshot& instance = row.metrics;
        out.instances.push_back(InstanceMetrics{
            .free_kv_pages           = instance.main_kv.free_pages,
            .active_requests         = instance.scheduler.running_requests,
            .pending_requests        = instance.scheduler.waiting_requests,
            .prefix_hit_tokens       = row.prefix_hit_tokens,
            .manifest_restore_tokens = row.manifest_restore_tokens,
            .committed_tokens        = instance.scheduler.committed_decode_tokens,
        });
        out.devices.insert(out.devices.end(), instance.devices.begin(), instance.devices.end());

        out.scheduler.computed_prefill_tokens += instance.scheduler.computed_prefill_tokens;
        out.scheduler.committed_decode_tokens += instance.scheduler.committed_decode_tokens;
        out.scheduler.decode_rounds           += instance.scheduler.decode_rounds;
        out.scheduler.decode_row_rounds       += instance.scheduler.decode_row_rounds;
        out.scheduler.running_requests        += instance.scheduler.running_requests;
        out.scheduler.prefilling_requests     += instance.scheduler.prefilling_requests;
        out.scheduler.decode_ready_requests   += instance.scheduler.decode_ready_requests;
        out.scheduler.waiting_requests        += instance.scheduler.waiting_requests;

        out.main_kv.capacity_pages += instance.main_kv.capacity_pages;
        out.main_kv.mapped_pages   += instance.main_kv.mapped_pages;
        out.main_kv.cached_pages   += instance.main_kv.cached_pages;
        out.main_kv.free_pages     += instance.main_kv.free_pages;
        out.main_kv.page_bytes     = std::max(out.main_kv.page_bytes, instance.main_kv.page_bytes);

        out.backend_kv.capacity_pages += instance.backend_kv.capacity_pages;
        out.backend_kv.mapped_pages   += instance.backend_kv.mapped_pages;
        out.backend_kv.cached_pages   += instance.backend_kv.cached_pages;
        out.backend_kv.free_pages     += instance.backend_kv.free_pages;
        out.backend_kv.page_bytes     = std::max(out.backend_kv.page_bytes, instance.backend_kv.page_bytes);

        out.prefix_cache.enabled = out.prefix_cache.enabled || instance.prefix_cache.enabled;
        out.prefix_cache.device_snapshot_slots      += instance.prefix_cache.device_snapshot_slots;
        out.prefix_cache.device_snapshot_slots_used += instance.prefix_cache.device_snapshot_slots_used;
        // The pool shares one host tier across its instances, so every Engine reports the same
        // process-wide host figures. Take the maximum instead of summing them.
        out.prefix_cache.host.enabled = out.prefix_cache.host.enabled || instance.prefix_cache.host.enabled;
        out.prefix_cache.host.pinned_bytes =
            std::max(out.prefix_cache.host.pinned_bytes, instance.prefix_cache.host.pinned_bytes);
        out.prefix_cache.host.used_bytes =
            std::max(out.prefix_cache.host.used_bytes, instance.prefix_cache.host.used_bytes);
        out.prefix_cache.host.block_slab_count =
            std::max(out.prefix_cache.host.block_slab_count, instance.prefix_cache.host.block_slab_count);
        out.prefix_cache.host.snapshot_slab_count = std::max(
            out.prefix_cache.host.snapshot_slab_count, instance.prefix_cache.host.snapshot_slab_count);
    }
    return out;
}

void EnginePool::record_completion(std::uint32_t instance, std::uint64_t prefix_hit_tokens,
                                   std::uint64_t manifest_restore_tokens) {
    std::lock_guard lock(mutex_);
    if (instance >= completion_by_instance_.size()) { return; }
    completion_by_instance_[instance].prefix_hit_tokens += prefix_hit_tokens;
    completion_by_instance_[instance].manifest_restore_tokens += manifest_restore_tokens;
}

MetricsSnapshot EnginePool::metrics() const {
    std::vector<PoolInstanceMetrics> rows;
    rows.reserve(engines_.size());
    for (const auto& engine : engines_) { rows.push_back(PoolInstanceMetrics{.metrics = engine->metrics()}); }
    {
        std::lock_guard lock(mutex_);
        for (std::size_t index = 0; index < rows.size() && index < completion_by_instance_.size();
             ++index) {
            rows[index].prefix_hit_tokens =
                completion_by_instance_[index].prefix_hit_tokens;
            rows[index].manifest_restore_tokens =
                completion_by_instance_[index].manifest_restore_tokens;
        }
    }
    return aggregate_instance_metrics(rows);
}

std::size_t EnginePool::pending_count() const {
    std::lock_guard lock(mutex_);
    return pending_.size();
}

std::uint64_t EnginePool::pending_front_id() const {
    std::lock_guard lock(mutex_);
    return pending_.empty() ? 0 : pending_.front()->route.id;
}

std::size_t EnginePool::permits_held() const {
    std::lock_guard lock(mutex_);
    std::size_t held = 0;
    for (const auto& instance : permits_by_instance_) { held += instance.size(); }
    return held;
}

} // namespace ninfer
