// Multi-instance shared-cache gate on a real Qwen3.8 27B artifact.
//
// Two tp1 Engines own separate Programs on separate devices but share one artifact reader and one
// pinned-host cache tier. The first Engine publishes a full host tap for an 8192-token prompt; the
// second Engine proves the process-level handles are actually shared. A second request on the
// first Engine proves local prefix reuse still works while manifest ownership is active.
//
// `test_capacity_and_probe` runs first, while Engine B is still cold: `prefix_probe()` and
// `capacity_snapshot()` must report an empty local frontier, then the committed 8192-token prefix
// after B's first request, with pool and scheduler counters matching the Engine facts.
//
// The cross-instance tests then drive the `PrefixReuseHint` path: Engine B restores Engine A's
// manifest through `import_foreign`, commits the restored device pages back to its own index, and
// a stale hint falls back to the cold path.
//
// `test_pool_two_tp1` runs the same two devices through `EnginePool`: two requests fill the two
// lanes, 16 short requests queue behind them in the single global admission queue, a routed handle
// that has not completed is destroyed to release its permit, the queue head is not starved, and the
// second round reuses the 8192-token prefix through the shared tier.
//
// NINFER_V100X2_ARTIFACT=/path/to/model.ninfer ctest -R multi_instance_real

#include "ninfer/engine.h"

#include "runtime/engine/engine_pool.h"
#include "runtime/prefix_cache/shared_host_tier.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kCapacity = 262144;
constexpr std::uint32_t kPromptTokens = 8192;
constexpr std::size_t kTokenDomain = 248077;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<ninfer::TokenId> long_prompt(std::size_t count) {
    std::vector<ninfer::TokenId> tokens(count);
    std::uint64_t state = 0x9E37'79B9'7F4A'7C15ULL;
    for (ninfer::TokenId& token : tokens) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        token = static_cast<ninfer::TokenId>((state >> 33U) % kTokenDomain);
    }
    return tokens;
}

// A prompt that is not a prefix of `long_prompt`, so a second Engine that never ran it cold has
// no local entry for it; any reuse on its second request can only come from the foreign commit.
std::vector<ninfer::TokenId> distinct_prompt(std::size_t count) {
    std::vector<ninfer::TokenId> tokens(count);
    std::uint64_t state = 0xC2B2'AE3D'27D4'EB4FULL;
    for (ninfer::TokenId& token : tokens) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        token = static_cast<ninfer::TokenId>((state >> 33U) % kTokenDomain);
    }
    return tokens;
}

ninfer::EngineOptions engine_options(const char* artifact, int device) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.device        = device;
    options.tp            = 1;
    options.max_context   = kCapacity;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    options.max_concurrency = 1;
    options.kv_cache        = ninfer::KvCacheStorage::Int8Group64;
    options.prefix_cache.enabled        = true;
    options.prefix_cache.host_cache_mib = 4096;
    // One tail tap per prompt and four snapshot slabs let the tier hold the original and the
    // foreign prompt while B pins the original manifest.
    options.prefix_cache.max_new_taps   = 1;
    return options;
}

ninfer::RequestOptions request_options(std::uint32_t output_tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = output_tokens;
    options.execution.allow_prefix_reuse      = true;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.presence_penalty  = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults          = false;
    return options;
}

// Greedy completion token ids, the level the cross-instance equality is asserted at.
std::vector<ninfer::TokenId> greedy_tokens(const ninfer::GenerationResult& result) {
    return result.generated_token_ids;
}

std::uint32_t block_frontier(const std::vector<std::uint64_t>& hashes) {
    return static_cast<std::uint32_t>(hashes.size()) * 64U;
}

// Probe and capacity on a cold Engine B: the device frontier is empty before any request and the
// tier reports no manifest; after one request on B, its committed blocks show up as the longest
// device-resident prefix. The capacity fields must match the Engine's startup-fixed facts.
void test_capacity_and_probe(ninfer::Engine& b) {
    auto prompt = b.prepare_tokens(long_prompt(kPromptTokens));
    const auto probe = b.prefix_probe(prompt);
    require(probe.block_tokens == 64, "probe must report 64-token blocks");
    require(probe.device_frontier == 0, "cold engine must report an empty device frontier");
    require(probe.device_frontier % 64 == 0, "device frontier must be block aligned");
    require(probe.manifest_frontier == 0, "cold engine must report no shared manifest prefix");
    require(probe.manifest_id == 0, "cold probe must not name a manifest");

    const auto cap = b.capacity_snapshot();
    require(cap.total_pages >= kCapacity / 64ULL, "capacity snapshot lost the pool total");
    require(cap.free_pages <= cap.total_pages, "free pages cannot exceed total pages");
    require(cap.max_concurrency == 1, "capacity snapshot must report the Engine concurrency");
    require(cap.max_context == kCapacity, "capacity snapshot lost the exact max context");
    require(cap.prefill_chunk == 1024, "capacity snapshot lost the prefill chunk");
    require(cap.recent_prefill_tps == 0.0, "no prefill sample yet must report zero tps");
    require(cap.active_requests == 0 && cap.pending_requests == 0,
            "cold scheduler must report no active or pending requests");

    auto first = b.generate(std::move(prompt), request_options(1));
    require(first.generated_token_ids.size() == 1, "first request must emit one token");

    const auto warm = b.prefix_probe(b.prepare_tokens(long_prompt(kPromptTokens)));
    require(warm.device_frontier >= kPromptTokens,
            "committed blocks must show up as the longest device-resident prefix");
    require(warm.device_frontier % 64 == 0, "warm device frontier must be block aligned");

    const auto after = b.capacity_snapshot();
    require(after.recent_prefill_tps > 0.0, "a completed prefill must leave a tps sample");
    require(after.active_requests == 0 && after.pending_requests == 0,
            "scheduler must be idle after the request completed");
    std::cout << "capacity/probe: total=" << cap.total_pages << " free=" << cap.free_pages
              << " tps=" << after.recent_prefill_tps
              << " warm_frontier=" << warm.device_frontier << std::endl;
}

void test_shared_state_and_publish(ninfer::Engine& a, ninfer::Engine& b) {
    require(a.shared_host_tier() != nullptr, "engine A must create its tier");
    require(a.shared_model() != nullptr, "engine A must create its reader");
    require(a.shared_model() == b.shared_model(), "one process-level artifact reader");
    require(a.shared_host_tier() == b.shared_host_tier(), "one process-level host tier");

    const std::vector<ninfer::TokenId> prompt_tokens = long_prompt(kPromptTokens);
    auto prompt = a.prepare_tokens(prompt_tokens);
    require(prompt.debug_token_ids().size() == kPromptTokens,
            "long prompt did not retain its exact token count");
    auto first = a.generate(std::move(prompt), request_options(1));
    require(first.generated_token_ids.size() == 1, "first request must emit one token");
    require(a.shared_host_tier()->manifest_count() > 0,
            "a full tap at 8192 tokens must publish a manifest");

    auto second = a.generate(a.prepare_tokens(prompt_tokens), request_options(1));
    require(second.generated_token_ids.size() == 1, "second request must emit one token");
    require(second.prefix_reuse_path != ninfer::PrefixReusePath::FullReset,
            "local prefix reuse regressed under shared manifest ownership");
    require(second.reused_prompt_tokens > 0,
            "local prefix reuse reported no reused prompt tokens");
    std::cout << "manifest_count=" << a.shared_host_tier()->manifest_count()
              << " reused=" << second.reused_prompt_tokens
              << " path=" << static_cast<int>(second.prefix_reuse_path) << std::endl;
}

void test_cross_instance_restore(ninfer::Engine& a, ninfer::Engine& b) {
    const std::vector<ninfer::TokenId> prompt_tokens = long_prompt(kPromptTokens);

    // B's cold run first: it is B's own oracle and proves the two Engines agree on greedy
    // continuation for the same prompt before any hint is involved.
    auto cold_b = b.generate(b.prepare_tokens(prompt_tokens), request_options(8));
    require(cold_b.generated_token_ids.size() == 8, "B cold run must emit eight tokens");

    // A publishes the full host tap for the same prompt.
    auto prompt_a = a.prepare_tokens(prompt_tokens);
    require(prompt_a.token_count() == kPromptTokens, "A prompt lost its token count");
    const std::vector<std::uint64_t> hashes(prompt_a.block_hashes().begin(),
                                           prompt_a.block_hashes().end());
    require(hashes.size() == kPromptTokens / 64ULL, "A prompt did not chain 128 block hashes");
    auto cold_a = a.generate(std::move(prompt_a), request_options(8));
    require(cold_a.generated_token_ids.size() == 8, "A cold run must emit eight tokens");

    auto m = a.shared_host_tier()->match(1, hashes, kPromptTokens);
    require(m.has_value(), "the shared tier must still hold a manifest for A's prompt");
    require(m->frontier > 0 && m->frontier % 64 == 0, "manifest frontier must be block-aligned");

    ninfer::RequestOptions warm = request_options(8);
    warm.execution.prefix_reuse_hint = {m->id, m->frontier, m->tp, m->block_tokens};

    auto heated = b.generate(b.prepare_tokens(prompt_tokens), warm);
    require(heated.reused_prompt_tokens == m->frontier,
            "target must restore the full manifest frontier before decoding");
    require(greedy_tokens(heated) == greedy_tokens(cold_a),
            "cross-instance greedy continuation must match A's cold baseline");
    require(greedy_tokens(heated) == greedy_tokens(cold_b),
            "cross-instance greedy continuation must match B's own cold baseline");
    std::cout << "cross-instance: manifest=" << m->id << " frontier=" << m->frontier
              << " reused=" << heated.reused_prompt_tokens
              << " path=" << static_cast<int>(heated.prefix_reuse_path) << std::endl;

    // Commit-back proof on a prompt B has never run cold. B's local index had no entry for this
    // prompt before the warm restore, so any reuse on B's second request can only come from the
    // foreign commit; the cold_b snapshot path above is a different prompt and cannot satisfy it.
    // The commit runs through the saved foreign prompt tokens, independent of the tap plan.
    const std::vector<ninfer::TokenId> foreign_prompt = distinct_prompt(kPromptTokens);
    auto oracle_a2 = a.generate(a.prepare_tokens(foreign_prompt), request_options(8));
    require(oracle_a2.generated_token_ids.size() == 8, "A cold run on the foreign prompt failed");

    auto warm_prompt_b = b.prepare_tokens(foreign_prompt);
    const std::vector<std::uint64_t> foreign_hashes(warm_prompt_b.block_hashes().begin(),
                                                    warm_prompt_b.block_hashes().end());
    auto m2 = a.shared_host_tier()->match(1, foreign_hashes, kPromptTokens);
    require(m2.has_value(), "the shared tier must hold a manifest for the foreign prompt");

    ninfer::RequestOptions warm2 = request_options(8);
    warm2.execution.prefix_reuse_hint = {m2->id, m2->frontier, m2->tp, m2->block_tokens};
    auto heated2 = b.generate(std::move(warm_prompt_b), warm2);
    require(heated2.reused_prompt_tokens == m2->frontier,
            "warm restore on a prompt B never ran cold must reuse the full manifest frontier");
    require(greedy_tokens(heated2) == greedy_tokens(oracle_a2),
            "cross-instance greedy continuation on the foreign prompt must match A");

    auto committed = b.generate(b.prepare_tokens(foreign_prompt), request_options(1));
    require(committed.reused_prompt_tokens >= m2->frontier,
            "restored foreign pages were not committed back to B's local index");
    require(greedy_tokens(committed)[0] == greedy_tokens(oracle_a2)[0],
            "the committed snapshot must reproduce A's greedy continuation on the foreign prompt");
    std::cout << "cross-instance commit: reused=" << committed.reused_prompt_tokens
              << " path=" << static_cast<int>(committed.prefix_reuse_path) << std::endl;
}

void test_stale_hint_falls_back(ninfer::Engine& a, ninfer::Engine& b) {
    // A prompt B has never run, so B's local index has no snapshot for it and the only possible
    // hit is the stale manifest the hint names.
    const std::vector<ninfer::TokenId> prompt_tokens = long_prompt(4096);

    auto oracle = a.generate(a.prepare_tokens(prompt_tokens), request_options(8));
    require(oracle.generated_token_ids.size() == 8, "A cold run must emit eight tokens");

    auto tier = a.shared_host_tier();
    auto stale_prompt = b.prepare_tokens(prompt_tokens);
    const std::vector<std::uint64_t> hashes(stale_prompt.block_hashes().begin(),
                                           stale_prompt.block_hashes().end());

    ninfer::SharedHostTier::PublishData data;
    data.tp        = 1;
    data.frontier  = static_cast<std::uint32_t>(hashes.size()) * 64U;
    data.hashes    = hashes;
    data.block_pages.reserve(hashes.size());
    for (std::size_t block = 0; block < hashes.size(); ++block) {
        data.block_pages.push_back(tier->allocate_block().page);
    }
    data.state_page = tier->allocate_snapshot().page;
    const std::uint64_t stale_id = tier->publish(tier->allocate_owner(), std::move(data));
    require(stale_id != 0, "stale manifest publish failed");

    // The last ref goes, then the manifest is evicted; its pages return to the pool.
    require(!tier->release_manifest(stale_id), "stale manifest must reach zero refs");
    require(tier->erase_manifest(stale_id), "stale manifest must be erasable at zero refs");

    ninfer::RequestOptions stale;
    stale.execution.requested_output_tokens = 8;
    stale.execution.allow_prefix_reuse      = true;
    stale.execution.sampling.temperature    = 0.0F;
    stale.execution.sampling.presence_penalty  = 0.0F;
    stale.execution.sampling.frequency_penalty = 0.0F;
    stale.stop.include_model_defaults          = false;
    stale.execution.prefix_reuse_hint = {stale_id, block_frontier(hashes), 1, 64};

    auto fell_back = b.generate(std::move(stale_prompt), stale);
    require(fell_back.reused_prompt_tokens == 0,
            "a stale hint must fall back to the cold path with zero reused tokens");
    require(greedy_tokens(fell_back) == greedy_tokens(oracle),
            "stale-hint cold output must equal the cold baseline on the same prompt");
    std::cout << "stale hint: reused=" << fell_back.reused_prompt_tokens
              << " path=" << static_cast<int>(fell_back.prefix_reuse_path) << std::endl;
}

// The pool route on the same two devices: one global admission queue in front of two single-lane
// Engines. Two in-flight requests fill both lanes, 16 short requests queue behind them, a routed
// handle that has not completed releases its permit when destroyed, the queue head is not starved,
// and the repeated 8192-token prefix is reused.
void test_pool_two_tp1(const char* artifact) {
    using Clock = std::chrono::steady_clock;

    constexpr std::size_t kQueuedRequests = 16;
    // Two TP1 weight sets plus a 262144-token KV pool do not fit on this 16 GiB target, so the test
    // profile uses 131072 KV tokens. The product default is unchanged.
    constexpr std::uint32_t kPoolKvCapacity = 131072;

    ninfer::EnginePool::Options options;
    options.global_pending_requests = 16;
    for (int device = 0; device < 2; ++device) {
        ninfer::EnginePool::GroupOptions group;
        group.devices = {device};
        group.tp      = 1;
        group.engine  = engine_options(artifact, device);
        group.engine.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kPoolKvCapacity);
        group.engine.max_context = kPoolKvCapacity;
        // The host tier has two snapshot slabs, one per warm-up engine; capture only the tail tap
        // so both 8128 manifests survive for the second round.
        group.engine.prefix_cache.host_cache_mib = 2048;
        group.engine.prefix_cache.max_new_taps   = 1;
        // The two 8192 warm-up requests hold the lanes while the sixteen waiters queue; the default
        // 30 s admission timeout is shorter than their prefill.
        group.engine.pending_timeout_ms = 300000;
        options.groups.push_back(group);
    }

    ninfer::EnginePool pool(options);
    std::cout << "pool: groups=" << options.groups.size() << " kv_capacity=" << kPoolKvCapacity
              << " global_pending=" << options.global_pending_requests << std::endl;

    require(pool.engines().size() == 2, "the pool must own two engines");
    for (const auto& instance_options : pool.instance_options()) {
        require(instance_options.max_pending_requests == 1,
                "a pool group must keep the single-slot pending queue");
    }
    require(pool.shared_model() != nullptr, "the pool must own the shared artifact reader");
    require(pool.engines()[0]->shared_model() == pool.engines()[1]->shared_model(),
            "both engines must share one artifact reader");
    require(pool.shared_host_tier() != nullptr, "the first group must publish the shared host tier");
    require(pool.shared_host_tier() == pool.engines()[0]->shared_host_tier(),
            "the pool tier must be the tier instance 0 created");
    require(pool.shared_host_tier() == pool.engines()[1]->shared_host_tier(),
            "the second group must consume the same tier");
    require(pool.metrics().instances.size() == 2, "metrics must report one row per instance");

    const auto started  = Clock::now();
    const auto deadline = started + std::chrono::seconds(600);

    const std::vector<ninfer::TokenId> long_tokens  = long_prompt(kPromptTokens);
    const std::vector<ninfer::TokenId> short_tokens = long_prompt(1024);

    // Two requests take the only permit on each instance. Placement is asserted while both permits
    // are still held, so each instance was really routed one request.
    ninfer::PoolHandle first  = pool.submit(pool.prepare_tokens(long_tokens), request_options(8));
    ninfer::PoolHandle second = pool.submit(pool.prepare_tokens(long_tokens), request_options(8));
    require(first.instance() == 0 && second.instance() == 1,
            "the first two requests did not fill the two lanes");

    // 16 short requests queue behind the two held permits. Each submit blocks until the pool admits
    // it, so no queued thread holds a handle until one of the initial permits is released.
    std::vector<ninfer::PreparedPrompt> prompts;
    // The queue probes must not publish taps: the two snapshot slabs are reserved for the 8192
    // warm-up manifests that the second round restores through the shared tier.
    for (std::size_t i = 0; i < kQueuedRequests; ++i) {
        prompts.push_back(pool.prepare_tokens(short_tokens, false));
    }

    struct Slot {
        std::optional<ninfer::PoolHandle> handle;
        ninfer::GenerationResult result;
        std::uint64_t pending_id = 0;
        std::atomic<bool> ready{false};
        std::atomic<bool> cancelled{false};
        std::atomic<bool> done{false};
    };
    std::vector<std::unique_ptr<Slot>> slots(kQueuedRequests);
    for (auto& slot : slots) { slot = std::make_unique<Slot>(); }

    std::vector<int> completion_order(kQueuedRequests, -1);
    std::mutex order_mutex;
    int completed_rank = 0;
    std::atomic<bool> gate{false};

    std::vector<std::thread> threads;
    threads.reserve(kQueuedRequests);
    for (std::size_t i = 0; i < kQueuedRequests; ++i) {
        threads.emplace_back([&, i] {
            slots[i]->handle = pool.submit(std::move(prompts[i]), request_options(8));
            slots[i]->pending_id = slots[i]->handle->pending_id();
            slots[i]->ready.store(true, std::memory_order_release);
            while (!gate.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (slots[i]->cancelled.load()) { return; }
            slots[i]->result = slots[i]->handle->wait();
            {
                std::lock_guard lock(order_mutex);
                completion_order[i] = completed_rank++;
            }
            slots[i]->done.store(true);
        });
    }

    // The pool's lanes are permits, not Engine completion: while the two initial permits are held,
    // every one of the sixteen requests must stay in the global queue and no thread may have a
    // handle yet.
    while (Clock::now() < deadline) {
        if (pool.pending_count() == kQueuedRequests && pool.permits_held() == 2) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    require(pool.pending_count() == kQueuedRequests && pool.permits_held() == 2,
            "the sixteen requests did not all queue behind the two initial permits");
    for (const auto& slot : slots) {
        require(!slot->ready.load(std::memory_order_acquire),
                "a queued request was admitted while both initial permits were held");
    }
    const std::uint64_t head_id = pool.pending_front_id();
    require(head_id != 0, "the global queue must report a head id");

    // Releasing the two initial permits lets the queue take the lanes. Only then can any queued
    // submit return.
    first.wait();
    second.wait();

    // Two of the sixteen threads win the two lanes (which two is not specified) and the rest stay
    // in the global queue.
    std::size_t ready_count = 0;
    std::size_t cancel_slot = slots.size();
    std::size_t head_slot   = slots.size();
    while (Clock::now() < deadline) {
        ready_count = 0;
        cancel_slot = slots.size();
        head_slot   = slots.size();
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (!slots[i]->ready.load(std::memory_order_acquire)) { continue; }
            ++ready_count;
            if (slots[i]->pending_id == head_id) { head_slot = i; }
            if (cancel_slot == slots.size() && slots[i]->pending_id != head_id) {
                cancel_slot = i;
            }
        }
        if (ready_count >= 2 && head_slot != slots.size() && pool.pending_count() >= 1) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    require(ready_count >= 2 && head_slot != slots.size() && pool.pending_count() >= 1,
            "the pool did not route the queue head while keeping queued requests behind it");
    require(cancel_slot != slots.size(), "no non-head routed handle was available to cancel");

    // Destroying a routed handle that has not completed releases its permit.
    slots[cancel_slot]->cancelled.store(true);
    slots[cancel_slot]->handle.reset();
    gate.store(true, std::memory_order_release);

    for (auto& thread : threads) { thread.join(); }
    std::size_t completed = 0;
    for (const auto& slot : slots) {
        if (!slot->done.load()) { continue; }
        ++completed;
        require(slot->result.generated_token_ids.size() == 8,
                "a queued request did not emit its eight tokens");
    }
    require(completed == kQueuedRequests - 1,
            "the queued requests did not all complete after the cancelled handle");
    const int last_rank = *std::max_element(completion_order.begin(), completion_order.end());
    require(completion_order[head_slot] >= 0 && completion_order[head_slot] < last_rank,
            "the queue head was skipped until the end of the queue");
    require(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - started).count() < 600,
            "the pool queue run exceeded its 600 second bound");
    // Second round on the same 8192-token prefix: at least one instance reuses the whole prefix.
    auto round2_first  = pool.submit(pool.prepare_tokens(long_tokens), request_options(8)).wait();
    auto round2_second = pool.submit(pool.prepare_tokens(long_tokens), request_options(8)).wait();
    // A full tap leaves the final block for the decode opener, so the largest reusable frontier is
    // kPromptTokens - kBlockTokens, matching the cross-instance restore assertion above.
    std::cout << "pool round2 reused=" << round2_first.reused_prompt_tokens << "/"
              << round2_second.reused_prompt_tokens << std::endl;
    require(round2_first.reused_prompt_tokens >= kPromptTokens - 64 ||
            round2_second.reused_prompt_tokens >= kPromptTokens - 64,
            "the repeated 8192 prefix was not reused by any instance");
    require(pool.metrics().instances.size() == 2,
            "metrics must still report two instances after the queue run");
    std::cout << "pool queue: head_rank=" << completion_order[head_slot] << " reused="
              << round2_first.reused_prompt_tokens << "/" << round2_second.reused_prompt_tokens
              << std::endl;
}

void test_cross_instance_mtp(const char* artifact) {
    ninfer::EngineOptions options_a = engine_options(artifact, 0);
    options_a.create_shared_host_tier = true;
    options_a.speculative             = ninfer::SpeculativeOptions{ninfer::SpeculativeBackend::Mtp,
                                                                   2, ninfer::ProposalHead::Full};
    options_a.kv_capacity             = ninfer::KvCapacityPolicy::explicit_capacity(131072);
    options_a.max_context             = 131072;

    std::shared_ptr<ninfer::Engine> a;
    try {
        a = std::make_shared<ninfer::Engine>(options_a);
    } catch (const std::exception& error) {
        std::cout << "skip cross-instance MTP: engine A did not fit (" << error.what()
                  << "); recorded as unverified" << std::endl;
        return;
    }

    ninfer::EngineOptions options_b = engine_options(artifact, 1);
    options_b.shared_model     = a->shared_model();
    options_b.shared_host_tier = a->shared_host_tier();
    options_b.speculative      = options_a.speculative;
    options_b.kv_capacity      = options_a.kv_capacity;
    options_b.max_context      = options_a.max_context;

    std::shared_ptr<ninfer::Engine> b;
    try {
        b = std::make_shared<ninfer::Engine>(options_b);
    } catch (const std::exception& error) {
        std::cout << "skip cross-instance MTP: engine B did not fit (" << error.what()
                  << "); recorded as unverified" << std::endl;
        return;
    }

    const std::vector<ninfer::TokenId> prompt_tokens = long_prompt(kPromptTokens);
    auto cold_a = a->generate(a->prepare_tokens(prompt_tokens), request_options(8));
    require(cold_a.generated_token_ids.size() == 8, "A cold MTP run must emit eight tokens");

    const auto prepared = a->prepare_tokens(prompt_tokens);
    const std::vector<std::uint64_t> hashes(prepared.block_hashes().begin(),
                                           prepared.block_hashes().end());
    auto m = a->shared_host_tier()->match(1, hashes, kPromptTokens);
    require(m.has_value(), "the shared tier must hold an MTP manifest for the prompt");

    ninfer::RequestOptions warm = request_options(8);
    warm.execution.prefix_reuse_hint = {m->id, m->frontier, m->tp, m->block_tokens};
    auto heated = b->generate(b->prepare_tokens(prompt_tokens), warm);
    require(heated.reused_prompt_tokens == m->frontier,
            "MTP target must restore the full manifest frontier");
    require(greedy_tokens(heated) == greedy_tokens(cold_a),
            "cross-instance MTP committed tokens must match A's cold MTP output");
    std::cout << "cross-instance MTP: reused=" << heated.reused_prompt_tokens
              << " path=" << static_cast<int>(heated.prefix_reuse_path) << std::endl;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_V100X2_ARTIFACT");
    if (artifact == nullptr || artifact[0] == '\0') {
        std::cout << "skip: set NINFER_V100X2_ARTIFACT to the product .ninfer artifact"
                  << std::endl;
        return 77;
    }
    try {
        test_pool_two_tp1(artifact);

        ninfer::EngineOptions options_a = engine_options(artifact, 0);
        options_a.create_shared_host_tier = true;
        ninfer::Engine a(options_a);

        ninfer::EngineOptions options_b = engine_options(artifact, 1);
        options_b.shared_model     = a.shared_model();
        options_b.shared_host_tier = a.shared_host_tier();
        ninfer::Engine b(options_b);

        test_capacity_and_probe(b);
        test_shared_state_and_publish(a, b);
        test_cross_instance_restore(a, b);
        test_stale_hint_falls_back(a, b);
        test_cross_instance_mtp(artifact);
    } catch (const std::exception& error) {
        std::cerr << "multi-instance shared-cache test failed: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
