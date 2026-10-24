// Host-tier metrics formatting: the /metrics wire contract. Builds a known snapshot and checks
// both the Prometheus text exposition and the JSON object carry the device, KV, host, and
// scheduler figures. No GPU or artifact required.

#include "runtime/engine/engine_pool.h"
#include "serve/metrics_format.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::MetricsSnapshot sample() {
    ninfer::MetricsSnapshot m;
    m.scheduler.running_requests      = 2;
    m.scheduler.prefilling_requests   = 1;
    m.scheduler.decode_ready_requests = 2;
    m.scheduler.waiting_requests      = 3;
    m.scheduler.computed_prefill_tokens = 1000;
    m.scheduler.committed_decode_tokens = 2000;
    m.scheduler.decode_rounds           = 50;
    m.scheduler.decode_row_rounds       = 75;
    m.devices.push_back(ninfer::DeviceMemoryMetrics{
        .device = 4, .total_bytes = 16ULL << 30, .free_bytes = 6ULL << 30, .used_bytes = 10ULL << 30});
    m.main_kv = ninfer::KvPoolMetrics{.capacity_pages = 16384,
                                      .mapped_pages = 100,
                                      .cached_pages = 20,
                                      .free_pages = 16264,
                                      .page_bytes = 2ULL << 20};
    m.prefix_cache.enabled                             = true;
    m.prefix_cache.device_snapshot_slots               = 5;
    m.prefix_cache.device_snapshot_slots_used          = 2;
    m.prefix_cache.host.enabled                        = true;
    m.prefix_cache.host.pinned_bytes                   = 28130ULL << 20;
    m.prefix_cache.host.used_bytes                     = 7ULL << 30;
    m.prefix_cache.host.block_slab_count               = 10291;
    m.prefix_cache.host.snapshot_slab_count            = 38;
    m.instances.push_back(ninfer::InstanceMetrics{.free_kv_pages = 4096,
                                                  .active_requests = 2,
                                                  .pending_requests = 1,
                                                  .prefix_hit_tokens = 8192,
                                                  .manifest_restore_tokens = 4096,
                                                  .committed_tokens = 200});
    m.instances.push_back(ninfer::InstanceMetrics{.free_kv_pages = 2048,
                                                  .active_requests = 1,
                                                  .pending_requests = 0,
                                                  .prefix_hit_tokens = 1024,
                                                  .manifest_restore_tokens = 0,
                                                  .committed_tokens = 150});
    return m;
}

ninfer::MetricsSnapshot instance_snapshot(std::uint64_t committed, std::uint32_t running,
                                          std::uint32_t waiting, std::uint32_t free_pages) {
    ninfer::MetricsSnapshot snapshot;
    snapshot.scheduler.running_requests        = running;
    snapshot.scheduler.waiting_requests        = waiting;
    snapshot.scheduler.computed_prefill_tokens = committed * 2;
    snapshot.scheduler.committed_decode_tokens = committed;
    snapshot.main_kv.free_pages                = free_pages;
    // Both instances share one host tier, so each reports the same process-wide figures.
    snapshot.prefix_cache.host.enabled           = true;
    snapshot.prefix_cache.host.pinned_bytes      = 8ULL << 30;
    snapshot.prefix_cache.host.used_bytes        = 3ULL << 30;
    snapshot.prefix_cache.host.block_slab_count  = 10291;
    snapshot.prefix_cache.host.snapshot_slab_count = 38;
    return snapshot;
}

int test_prometheus() {
    const std::string text = ninfer::serve::metrics_to_prometheus(sample());
    int failures           = 0;
    const auto has         = [&](const std::string& needle) {
        return text.find(needle) != std::string::npos;
    };
    failures += check(has("# TYPE ninfer_device_memory_bytes gauge"),
                      "device metric type missing");
    failures += check(has("ninfer_device_memory_bytes{device=\"4\",kind=\"used\"} " +
                          std::to_string(10ULL << 30)),
                      "device used bytes sample missing");
    failures += check(has("ninfer_kv_pages{pool=\"main\",kind=\"cached\"} 20"),
                      "main KV cached pages missing");
    failures += check(has("ninfer_kv_page_bytes{pool=\"main\"} 2097152"),
                      "KV page bytes missing");
    failures += check(has("ninfer_prefix_cache_device_snapshot_slots{kind=\"used\"} 2"),
                      "device snapshot used missing");
    failures += check(has("ninfer_prefix_cache_host_bytes{kind=\"pinned\"} " +
                          std::to_string(28130ULL << 20)),
                      "host pinned bytes missing");
    failures += check(has("ninfer_prefix_cache_host_slabs{kind=\"block\"} 10291"),
                      "host block slab count missing");
    failures += check(has("ninfer_requests{state=\"waiting\"} 3"), "waiting requests missing");
    failures += check(has("# TYPE ninfer_decode_rounds_total counter"),
                      "decode rounds counter missing");
    failures += check(has("ninfer_instance_free_kv_pages{instance=\"0\"} 4096"),
                      "instance 0 free KV pages missing");
    failures += check(has("ninfer_instance_active_requests{instance=\"0\"} 2"),
                      "instance 0 active requests missing");
    failures += check(has("ninfer_instance_pending_requests{instance=\"1\"} 0"),
                      "instance 1 pending requests missing");
    failures += check(has("ninfer_instance_prefix_hit_tokens_total{instance=\"0\"} 8192"),
                      "instance 0 prefix hit tokens missing");
    failures += check(has("ninfer_instance_manifest_restore_tokens_total{instance=\"0\"} 4096"),
                      "instance 0 manifest restore tokens missing");
    failures += check(has("ninfer_instance_committed_tokens_total{instance=\"1\"} 150"),
                      "instance 1 committed tokens missing");
    return failures;
}

int test_json() {
    int failures = 0;
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(ninfer::serve::metrics_to_json(sample()).dump());
    } catch (const std::exception& e) {
        std::cerr << "json parse failed: " << e.what() << '\n';
        return 1;
    }
    failures += check(parsed["devices"][0]["device"] == 4, "json device ordinal wrong");
    failures += check(parsed["devices"][0]["used_bytes"] == (10ULL << 30),
                      "json device used wrong");
    failures += check(parsed["main_kv"]["capacity_pages"] == 16384,
                      "json kv capacity wrong");
    failures += check(parsed["prefix_cache"]["host"]["pinned_bytes"] == (28130ULL << 20),
                      "json host pinned wrong");
    failures += check(parsed["prefix_cache"]["host"]["block_slab_count"] == 10291,
                      "json host slab count wrong");
    failures += check(parsed["scheduler"]["waiting_requests"] == 3,
                      "json scheduler waiting wrong");
    failures += check(parsed["instances"].size() == 2, "json instance array size wrong");
    failures += check(parsed["instances"][0]["active_requests"] == 2,
                      "json instance active requests wrong");
    failures += check(parsed["instances"][0]["pending_requests"] == 1,
                      "json instance pending requests wrong");
    failures += check(parsed["instances"][1]["prefix_hit_tokens"] == 1024,
                      "json instance prefix hit tokens wrong");
    failures += check(parsed["instances"][1]["manifest_restore_tokens"] == 0,
                      "json instance manifest restore tokens wrong");
    failures += check(parsed["instances"][0]["committed_tokens"] == 200,
                      "json instance committed tokens wrong");
    return failures;
}

// Pool aggregation is exercised with two fake per-instance snapshots: the aggregate committed
// counter must equal the sum of the instance rows, and every row must carry the fields the
// pool derives. This is aggregation, not rendering.
int test_pool_metrics_aggregate() {
    int failures = 0;
    const std::array<ninfer::PoolInstanceMetrics, 2> rows{{
        ninfer::PoolInstanceMetrics{
            .metrics = instance_snapshot(100, 2, 1, 4096),
            .prefix_hit_tokens = 7000,
            .manifest_restore_tokens = 3000,
        },
        ninfer::PoolInstanceMetrics{
            .metrics = instance_snapshot(250, 1, 3, 2048),
            .prefix_hit_tokens = 1000,
            .manifest_restore_tokens = 0,
        },
    }};

    const ninfer::MetricsSnapshot pool = ninfer::aggregate_instance_metrics(rows);

    failures += check(pool.instances.size() == 2,
                      "the aggregate did not carry one row per instance");
    failures += check(pool.scheduler.committed_decode_tokens == 350,
                      "the aggregate committed decode counter is not the sum");
    failures += check(pool.scheduler.computed_prefill_tokens == 700,
                      "the aggregate prefill counter is not the sum");
    if (pool.instances.size() == 2) {
        failures += check(pool.instances[0].committed_tokens == 100 &&
                              pool.instances[1].committed_tokens == 250,
                          "an instance row carries the wrong committed token count");
        failures += check(pool.instances[0].committed_tokens +
                              pool.instances[1].committed_tokens ==
                          pool.scheduler.committed_decode_tokens,
                          "the instance committed rows do not sum to the aggregate");
        failures += check(pool.instances[0].active_requests == 2 &&
                              pool.instances[1].active_requests == 1,
                          "an instance row carries the wrong active request count");
        failures += check(pool.instances[0].pending_requests == 1 &&
                              pool.instances[1].pending_requests == 3,
                          "an instance row carries the wrong pending request count");
        failures += check(pool.instances[0].free_kv_pages == 4096 &&
                              pool.instances[1].free_kv_pages == 2048,
                          "an instance row carries the wrong free KV page count");
        failures += check(pool.instances[0].prefix_hit_tokens == 7000 &&
                              pool.instances[1].prefix_hit_tokens == 1000,
                          "an instance row carries the wrong prefix hit token count");
        failures += check(pool.instances[0].manifest_restore_tokens == 3000 &&
                              pool.instances[1].manifest_restore_tokens == 0,
                          "an instance row carries the wrong manifest restore token count");
    }
    failures += check(pool.main_kv.free_pages == 6144,
                      "the aggregate free KV pages are not the sum");
    failures += check(pool.prefix_cache.host.pinned_bytes == (8ULL << 30) &&
                          pool.prefix_cache.host.used_bytes == (3ULL << 30),
                      "the aggregate multiplied the shared host tier");
    failures += check(pool.prefix_cache.host.block_slab_count == 10291 &&
                          pool.prefix_cache.host.snapshot_slab_count == 38,
                      "the aggregate summed shared host slab counts");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_prometheus();
    failures += test_json();
    failures += test_pool_metrics_aggregate();
    if (failures == 0) {
        std::printf("metrics format tests passed.\n");
        return 0;
    }
    std::printf("%d metrics format check(s) failed.\n", failures);
    return 1;
}
