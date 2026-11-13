#pragma once

// Global admission queue for a multi-instance Engine pool.
//
// One EnginePool owns N Engines that execute the same artifact at the same tensor-parallel width,
// share one `SharedModel` reader and one `SharedHostTier`, and each keep a single-slot pending queue
// (`max_pending_requests == 1`) that only carries the request already routed to it. The pool is the
// only admission queue: `submit` turns a prepared prompt into a pending entry, waits for
// `EngineRouter::pick` to return a feasible candidate, and then reserves a lane and the decision's
// net `reserved_pages` KV pages on that instance under one pool lock. A `Permit` is that reservation
// and `PoolHandle` owns it for the request's lifetime.
//
// The pool's capacity view for one instance is its own authority for lane occupancy -- one permit is
// one lane claim -- and for page availability: the instance's reported free pages minus the pages
// reserved for permits that instance has not yet mapped. A permit holder is therefore never blocked
// by pending capacity inside its Engine.

#include "ninfer/engine.h"
#include "ninfer/types.h"
#include "runtime/engine/engine_router.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

class EnginePool;

// One instance's contribution to the pool snapshot. `metrics` is that Engine's own boundary
// snapshot; the two counters are pool-owned completions for the instance.
struct PoolInstanceMetrics {
    MetricsSnapshot metrics;
    std::uint64_t prefix_hit_tokens = 0;
    std::uint64_t manifest_restore_tokens = 0;
};

// Aggregate per-instance snapshots into the pool-wide MetricsSnapshot. Counters, KV pages and
// scheduler totals are summed; `instances` carries the derived per-instance rows. Device memory is
// concatenated in instance order. A single Engine uses the same function with one row.
[[nodiscard]] MetricsSnapshot aggregate_instance_metrics(
    std::span<const PoolInstanceMetrics> instances);

// Per-instance view the pool's admission core reads. Production binds it to a real Engine; host tests
// inject a scripted view so the permit logic runs with no artifact, no device and no Engine
// construction.
// Probe and capacity captured together under one non-blocking execution-lock attempt.
struct InstanceProbe {
    PrefixProbe probe;
    CapacitySnapshot capacity;
};

class PoolInstanceView {
public:
    virtual ~PoolInstanceView() = default;
    // `nullopt` when the instance is mid-unit: the router skips it that dispatch round rather
    // than wait behind its execution lock.
    [[nodiscard]] virtual std::optional<InstanceProbe>
    try_probe(std::span<const std::uint64_t> hashes) const = 0;
};

// One applied admission decision: a lane plus the decision's net `reserved_pages` on one instance,
// held until the request completes or its handle is destroyed.
class Permit {
public:
    Permit(EnginePool& pool, std::uint32_t instance, std::uint32_t reserved_pages) noexcept
        : pool_(&pool), instance_(instance), reserved_pages_(reserved_pages) {}

    ~Permit() { release(); }

    Permit(const Permit&)            = delete;
    Permit& operator=(const Permit&) = delete;

    [[nodiscard]] std::uint32_t instance() const noexcept { return instance_; }
    [[nodiscard]] std::uint32_t reserved_pages() const noexcept { return reserved_pages_; }

    // Idempotent; called by `PoolHandle::wait()` and by the handle destructor.
    void release() noexcept;

    // Accumulate this instance's completion counters before the permit is released.
    void record_completion(std::uint64_t prefix_hit_tokens,
                           std::uint64_t manifest_restore_tokens) noexcept;

private:
    friend class EnginePool;

    EnginePool* pool_;
    std::uint32_t instance_;
    std::uint32_t reserved_pages_;
    bool released_ = false;
};

// The handle for one pool-routed request: the permit plus, on the product path, the Engine handle for
// the instance the request was routed to. It does not extend the EnginePool's lifetime, so the pool
// must outlive every handle it produced.
class PoolHandle {
public:
    PoolHandle() noexcept = default;
    ~PoolHandle() { reset(); }

    PoolHandle(PoolHandle&&) noexcept;
    PoolHandle& operator=(PoolHandle&&) noexcept;

    PoolHandle(const PoolHandle&)            = delete;
    PoolHandle& operator=(const PoolHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept { return permit_ != nullptr; }
    [[nodiscard]] std::uint32_t instance() const noexcept { return instance_; }
    // Admission id assigned by the pool when the request entered the global queue.
    [[nodiscard]] std::uint64_t pending_id() const noexcept { return pending_id_; }
    [[nodiscard]] const PrefixReuseHint& hint() const noexcept { return hint_; }
    // The routed Engine's resolved sampling for this request, available once `submit` attached the
    // GenerationHandle. The serve request log records the same fields as the single-Engine path.
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    friend class EnginePool;

    void attach(std::shared_ptr<Permit> permit, std::uint32_t instance, PrefixReuseHint hint,
                std::uint64_t pending_id) noexcept;
    void attach(GenerationHandle&& handle) noexcept;
    void reset() noexcept;

    std::shared_ptr<Permit> permit_;
    std::unique_ptr<GenerationHandle> handle_;
    PrefixReuseHint hint_;
    std::uint32_t instance_ = 0;
    std::uint64_t pending_id_ = 0;
};

class EnginePool {
public:
    struct GroupOptions {
        std::vector<int> devices;
        int tp = 1;
        ninfer::EngineOptions engine;
    };

    struct Options {
        std::vector<GroupOptions> groups;
        std::uint32_t global_pending_requests = 16;
        std::uint32_t affinity_slack_tokens   = 2048;
        std::chrono::milliseconds fairness_aging{1000};
    };

    explicit EnginePool(Options options);

    // Host-test seam: scripted instance views in place of Engines. No artifact, no device and no
    // Engine construction; `prepare` and `metrics` have no source on this path.
    EnginePool(Options options, std::vector<std::unique_ptr<PoolInstanceView>> views);

    ~EnginePool();

    EnginePool(const EnginePool&)            = delete;
    EnginePool& operator=(const EnginePool&) = delete;

    // Preparation runs on instance 0, which owns the tokenizer for the shared artifact. A prepared
    // prompt carries no instance identity and any Engine in the pool can consume it.
    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // Product path: the prepared prompt becomes a pending entry, the pool admits it, and the chosen
    // Engine executes it. If that Engine rejects the request, the permit is returned.
    [[nodiscard]] PoolHandle submit(PreparedPrompt prompt, RequestOptions options,
                                    std::chrono::steady_clock::time_point pending_deadline = {});

    // Admission core: the global queue plus the permit. `required_pages` on the pending is the
    // reuse-blind gross upper bound the caller computed; the router derives each candidate's net
    // bill from it and the permit holds that net bill. Host tests drive this directly.
    [[nodiscard]] PoolHandle admit(runtime::PendingRequest pending,
                                   std::chrono::steady_clock::time_point pending_deadline = {});

    [[nodiscard]] MetricsSnapshot metrics() const;

    [[nodiscard]] const std::vector<std::unique_ptr<Engine>>& engines() const noexcept {
        return engines_;
    }
    // The per-instance EngineOptions the pool actually used, after the single-slot pending
    // normalization.
    [[nodiscard]] const std::vector<EngineOptions>& instance_options() const noexcept {
        return instance_options_;
    }
    [[nodiscard]] std::size_t pending_count() const;
    // Admission id at the front of the global queue (0 when empty). The queue is in enqueue order.
    [[nodiscard]] std::uint64_t pending_front_id() const;
    [[nodiscard]] std::size_t permits_held() const;

    [[nodiscard]] std::shared_ptr<SharedHostTier> shared_host_tier() const noexcept {
        return shared_host_tier_;
    }
    [[nodiscard]] std::shared_ptr<SharedModel> shared_model() const noexcept {
        return shared_model_;
    }

private:
    friend class Permit;

    struct Entry {
        runtime::PendingRequest route;
        std::vector<std::uint64_t> hashes;
        std::chrono::steady_clock::time_point deadline;
        std::shared_ptr<Permit> permit;
        PrefixReuseHint hint;
        std::uint32_t instance = 0;
        bool routed = false;
    };

    void initialize();
    void validate_groups() const;
    void normalize_pending();
    std::optional<runtime::RouteCandidate> candidate_locked(
        const runtime::PendingRequest& pending, std::uint32_t instance) const;
    std::optional<runtime::RouteDecision> pick_locked() const;
    void apply_locked(const std::shared_ptr<Entry>& entry, const runtime::RouteDecision& decision);
    void erase_locked(const std::shared_ptr<Entry>& entry);
    std::uint32_t home_locked(std::span<const std::uint64_t> hashes) const;
    std::uint32_t required_pages_for(std::uint32_t prompt_tokens, std::uint32_t output_tokens) const;
    void record_completion(std::uint32_t instance, std::uint64_t prefix_hit_tokens,
                           std::uint64_t manifest_restore_tokens);

    Options options_;
    std::vector<EngineOptions> instance_options_;
    std::vector<std::unique_ptr<Engine>> engines_;
    std::vector<std::unique_ptr<PoolInstanceView>> views_;
    std::shared_ptr<SharedModel> shared_model_;
    std::shared_ptr<SharedHostTier> shared_host_tier_;

    runtime::EngineRouter router_;
    runtime::EngineRouter::Config config_;
    runtime::CandidateProvider provider_;
    std::vector<std::uint32_t> instance_ids_;
    std::uint32_t tp_ = 1;
    std::uint64_t next_pending_id_ = 1;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::shared_ptr<Entry>> pending_;
    std::vector<std::vector<Permit*>> permits_by_instance_;
    std::vector<PoolInstanceMetrics> completion_by_instance_;
    bool stopping_ = false;
};

} // namespace ninfer
