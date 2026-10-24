// Host-only EnginePool admission gate: no artifact, no device, no Engine construction.
//
// The pool's admission core runs against scripted `PoolInstanceView` objects, so the global pending
// queue, the lane+KV permit, the queue-full path and the permit timeout are exercised directly.
// The router is the real one and the real per-(pending, candidate) provider path; only the instance
// view is scripted. `EnginePool::admit` is the seam the product `submit` also runs through.

#include "runtime/engine/engine_pool.h"
#include "runtime/engine/engine_router.h"

#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using ninfer::EnginePool;
using ninfer::PoolHandle;
using ninfer::PoolInstanceView;
using ninfer::RequestError;
using ninfer::RequestErrorKind;
using ninfer::runtime::PendingRequest;

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    std::cerr << message << '\n';
    ++failures;
}

// One scripted instance view: a fixed capacity reading, no probe chain, no device.
class FakeView : public PoolInstanceView {
public:
    FakeView(std::uint32_t max_concurrency, std::uint32_t free_pages)
        : max_concurrency_(max_concurrency), free_pages_(free_pages) {}

    ninfer::PrefixProbe probe(std::span<const std::uint64_t> hashes) const override {
        return ninfer::PrefixProbe{.block_tokens = 64};
    }

    ninfer::CapacitySnapshot capacity() const override {
        return ninfer::CapacitySnapshot{
            .free_pages = free_pages_,
            .total_pages = 4096,
            .max_concurrency = max_concurrency_,
            .max_context = 262144,
            .prefill_chunk = 1024,
        };
    }

    void set_free_pages(std::uint32_t free_pages) noexcept { free_pages_ = free_pages; }

private:
    std::uint32_t max_concurrency_;
    std::uint32_t free_pages_;
};

EnginePool::GroupOptions group(int device) {
    EnginePool::GroupOptions options;
    options.devices = {device};
    options.tp      = 1;
    options.engine.device              = device;
    options.engine.artifact_path       = "host-test-artifact";
    options.engine.max_concurrency     = 1;
    // Pre-normalization value: the pool must rewrite it to one routed request per instance.
    options.engine.max_pending_requests = 16;
    return options;
}

// `required_pages` is the authoritative router input the caller supplies; prompt_tokens and
// max_tokens stay consistent with the pool formula so the pending is self-consistent. The id is
// assigned by `admit`, so the caller leaves it 0.
PendingRequest pending(std::uint32_t required_pages, std::uint32_t home_instance, bool has_home) {
    return PendingRequest{
        .id             = 0,
        .prompt_tokens  = required_pages * 64U - 64U,
        .max_tokens     = 0,
        .required_pages = required_pages,
        .enqueued       = Clock::now(),
        .home_instance  = home_instance,
        .has_home       = has_home,
        .block_hashes   = {},
    };
}

// Two instances, one lane and two pages each. The first two requests take the only two permits;
// the third waits in the global queue until one permit is released.
void test_permit_reserves_lane_and_kv() {
    EnginePool::Options options;
    options.groups                 = {group(0), group(1)};
    options.global_pending_requests = 16;

    std::vector<std::unique_ptr<PoolInstanceView>> views;
    views.push_back(std::make_unique<FakeView>(1, 2));
    views.push_back(std::make_unique<FakeView>(1, 2));
    EnginePool pool(options, std::move(views));

    const auto deadline = Clock::now() + std::chrono::seconds(10);
    PoolHandle first  = pool.admit(pending(2, 0, false), deadline);
    PoolHandle second = pool.admit(pending(2, 0, false), deadline);
    check(first.instance() == 0, "the first request did not reserve a permit on instance 0");
    check(second.instance() == 1, "the second request did not reserve a permit on instance 1");
    check(pool.permits_held() == 2, "both permits were not held at the same time");

    std::atomic<bool> third_routed{false};
    std::uint32_t third_instance = 0;
    std::optional<PoolHandle> third;
    std::thread waiter([&] {
        third = pool.admit(pending(2, 0, false), deadline);
        third_instance = third->instance();
        third_routed = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    check(!third_routed.load(), "the third request took a permit while both lanes were reserved");
    check(pool.pending_count() == 1, "the third request was not waiting in the global queue");

    first = PoolHandle{};
    waiter.join();
    check(third_routed.load(), "the third request was not routed after a permit was released");
    check(third_instance == 0, "the third request did not route to the instance that released its permit");
    check(pool.permits_held() == 2, "the released lane and pages were not re-reserved by the third request");
}

// The pool admits as soon as a lane is free, so a permit holder is never blocked by pending
// capacity: every group's own pending queue is normalized to the one already-routed request.
void test_no_double_queue() {
    EnginePool::Options options;
    options.groups                  = {group(0), group(1)};
    options.global_pending_requests = 16;

    std::vector<std::unique_ptr<PoolInstanceView>> views;
    views.push_back(std::make_unique<FakeView>(2, 4));  // two lanes, room for two permits
    views.push_back(std::make_unique<FakeView>(1, 0));  // never routable
    EnginePool pool(options, std::move(views));

    for (const auto& instance_options : pool.instance_options()) {
        check(instance_options.max_pending_requests == 1,
              "a pool group kept a pending queue larger than one routed request");
    }

    const auto deadline = Clock::now() + std::chrono::seconds(10);
    const auto started  = Clock::now();
    PoolHandle first  = pool.admit(pending(2, 0, false), deadline);
    PoolHandle second = pool.admit(pending(2, 0, false), deadline);
    check(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started) <
              std::chrono::milliseconds(100),
          "a permit holder was blocked instead of being admitted on a free lane");
    check(first.instance() == 0 && second.instance() == 0,
          "two permits were not both held on the instance with two lanes");

    std::atomic<bool> third_routed{false};
    std::optional<PoolHandle> third;
    std::thread waiter([&] {
        third = pool.admit(pending(2, 0, false), deadline);
        third_routed = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    check(!third_routed.load(), "a request was admitted past the instance's lane bound");
    check(pool.pending_count() == 1, "the over-bound request was not waiting in the global queue");

    first = PoolHandle{};
    waiter.join();
    check(third_routed.load(), "the over-bound request was not admitted after a lane was released");
}

// Destroying a handle that never completed releases its permit, so a later request can route.
void test_cancel_waiter_releases_permit() {
    EnginePool::Options options;
    options.groups                  = {group(0), group(1)};
    options.global_pending_requests = 16;

    std::vector<std::unique_ptr<PoolInstanceView>> views;
    views.push_back(std::make_unique<FakeView>(1, 2));
    views.push_back(std::make_unique<FakeView>(1, 0));
    EnginePool pool(options, std::move(views));

    const auto deadline = Clock::now() + std::chrono::seconds(10);
    PoolHandle held = pool.admit(pending(2, 0, false), deadline);
    check(held.instance() == 0, "the first request did not hold the only lane");

    std::atomic<bool> routed{false};
    std::uint32_t routed_instance = 0;
    std::optional<PoolHandle> waiting;
    std::thread waiter([&] {
        waiting = pool.admit(pending(2, 0, false), deadline);
        routed_instance = waiting->instance();
        routed = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    check(!routed.load(), "a second request took a permit while the first handle was alive");

    held = PoolHandle{};
    waiter.join();
    check(routed.load(), "the waiter was not admitted after the cancelled handle released its permit");
    check(routed_instance == 0, "the waiter did not route to the instance that released its permit");
}

// The global queue is the only admission queue: when it is full, submit fails immediately.
void test_global_queue_full_is_overloaded() {
    EnginePool::Options options;
    options.groups                  = {group(0), group(1)};
    options.global_pending_requests = 2;

    auto first_view  = std::make_unique<FakeView>(1, 0);
    auto second_view = std::make_unique<FakeView>(1, 0);
    FakeView* first  = first_view.get();
    FakeView* second = second_view.get();
    std::vector<std::unique_ptr<PoolInstanceView>> views;
    views.push_back(std::move(first_view));
    views.push_back(std::move(second_view));
    EnginePool pool(options, std::move(views));

    const auto deadline = Clock::now() + std::chrono::seconds(10);
    std::atomic<bool> done{false};
    std::vector<std::thread> waiters;
    for (int index = 0; index < 2; ++index) {
        waiters.emplace_back([&] {
            (void)pool.admit(pending(2, 0, false), deadline);
            done = true;
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    check(!done.load(), "a request was admitted even though no instance had room");
    check(pool.pending_count() == 2, "the two waiting requests were not in the global queue");

    bool overloaded = false;
    try {
        (void)pool.admit(pending(2, 0, false), deadline);
    } catch (const RequestError& error) {
        overloaded = error.kind() == RequestErrorKind::Overloaded;
    }
    check(overloaded, "a submit on a full global queue did not report Overloaded");

    first->set_free_pages(2);
    second->set_free_pages(2);
    for (auto& waiter : waiters) { waiter.join(); }
    check(done.load(), "the waiting requests did not complete after the queue drained");
}

// A waiter whose deadline has already passed reports QueueTimeout, not Overloaded.
void test_permit_timeout_is_queue_timeout() {
    EnginePool::Options options;
    options.groups                  = {group(0), group(1)};
    options.global_pending_requests = 16;

    std::vector<std::unique_ptr<PoolInstanceView>> views;
    views.push_back(std::make_unique<FakeView>(1, 2));
    views.push_back(std::make_unique<FakeView>(1, 2));
    EnginePool pool(options, std::move(views));

    const auto deadline = Clock::now() + std::chrono::seconds(10);
    PoolHandle first  = pool.admit(pending(2, 0, false), deadline);
    PoolHandle second = pool.admit(pending(2, 0, false), deadline);
    check(pool.permits_held() == 2, "both permits were not held before the timeout check");

    bool timed_out = false;
    try {
        (void)pool.admit(pending(2, 0, false), Clock::now() - std::chrono::milliseconds(1));
    } catch (const RequestError& error) {
        timed_out = error.kind() == RequestErrorKind::QueueTimeout;
    }
    check(timed_out, "an expired pending deadline did not report QueueTimeout");
    check(pool.pending_count() == 0, "the expired request stayed in the global queue");
    check(pool.permits_held() == 2, "the held permits were disturbed by the expired request");
    (void)first.wait();
    (void)second.wait();
    check(pool.permits_held() == 0, "wait() did not release the permits");
}

} // namespace

int main() {
    test_permit_reserves_lane_and_kv();
    test_no_double_queue();
    test_cancel_waiter_releases_permit();
    test_global_queue_full_is_overloaded();
    test_permit_timeout_is_queue_timeout();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
