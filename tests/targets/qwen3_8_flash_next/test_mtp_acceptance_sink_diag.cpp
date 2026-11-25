#include "flash_model_harness.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DeviceBuffer;
using ninfer::Tensor;
using ninfer::TokenId;
using flash_harness::FlashHarness;
using flash_harness::require;
namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
namespace rt = flash_harness::rt;

constexpr std::size_t kConditioningWords = 10240;  // MtpDraft::hidden_elements = 2560*4
constexpr std::size_t kSinkBytes = kConditioningWords * 2;
constexpr std::size_t kVerify = 2;                  // draft=1 -> [next_input, proposal]
constexpr std::size_t kOutputTokens = 6;

const std::vector<std::vector<TokenId>> kPrompts{
    {1024, 7, 98140, 33, 4097, 512, 100, 2048},
    {100, 200, 300, 400, 500, 600, 700, 800},
    {1, 2, 3, 4, 5, 6, 7, 8},
};

struct RouteResult {
    std::vector<std::uint16_t> sink;
    std::vector<std::uint16_t> sink_after_forward;
    std::vector<std::uint16_t> verification_record;
    TokenId next_input = 0;
    std::uint32_t next_position = 0;
    std::vector<TokenId> predictions;
    std::uint32_t accepted = 0;
    std::uint32_t committed_count = 0;
    std::array<std::uint64_t, 2> kv_digest{0, 0};
    std::vector<std::uint16_t> next_logits;
    std::vector<std::uint16_t> unsynced_sink;
};

std::vector<std::uint16_t> read_sink_unsynced(void* device) {
    std::vector<std::uint16_t> words(kConditioningWords);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(words.data(), device, kSinkBytes, cudaMemcpyDeviceToHost));
    return words;
}

std::vector<std::uint16_t> read_sink(FlashHarness& harness, void* device) {
    rt::TextState& state = *harness.states[0];
    CUDA_CHECK(cudaSetDevice(state.device_id()));
    CUDA_CHECK(cudaStreamSynchronize(state.stream()));
    std::vector<std::uint16_t> words(kConditioningWords);
    CUDA_CHECK(cudaMemcpy(words.data(), device, kSinkBytes, cudaMemcpyDeviceToHost));
    return words;
}

std::vector<std::uint16_t> capture_next_logits(FlashHarness& harness, TokenId token) {
    flash_harness::capture_buffer().clear();
    flash_harness::g_logits_sink = &flash_harness::capture_buffer();
    (void)harness.decode(token);
    flash_harness::g_logits_sink = nullptr;
    return flash_harness::capture_buffer().at(0);
}

void fill_kv_digest(FlashHarness& harness, RouteResult& out, const std::vector<TokenId>& prompt) {
    const auto position = static_cast<std::uint32_t>(prompt.size());
    for (int rank = 0; rank < 2; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        out.kv_digest[static_cast<std::size_t>(rank)] =
            flash_harness::kv_region_digest(*harness.states[slot], position, out.committed_count);
    }
}

std::vector<TokenId> reference_sequence(FlashHarness& harness, const std::vector<TokenId>& prompt) {
    harness.reset();
    TokenId next = harness.prefill(prompt);
    std::vector<TokenId> sequence;
    for (std::size_t step = 0; step < kOutputTokens + 2; ++step) {
        sequence.push_back(next);
        next = harness.decode(next);
    }
    return sequence;
}

// Route A: the superseded sequential T=1 verify plus the old per-decode sink write. The old
// StateTransaction restored the full pre-verify state before replaying the committed prefix, so
// the committed live state (and the sink written by its last T=1 decode) is exactly what a fresh
// T=1 pass over verify_tokens[0..committed_count-1] produces from the pre-verify state.
RouteResult route_sequential(FlashHarness& harness, const std::vector<TokenId>& prompt,
                             const std::vector<TokenId>& verify_tokens, DeviceBuffer& sink,
                             const RouteResult& batched) {
    const std::vector<TokenId> proposals(verify_tokens.begin() + 1, verify_tokens.end());

    // First pass: derive the sequential predictions / acceptance for reporting only.
    harness.reset();
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::UpdateInPlace);
    (void)harness.prefill(prompt);
    RouteResult probe;
    for (const TokenId token : verify_tokens) { probe.predictions.push_back(harness.decode(token)); }
    probe.accepted = flash::accepted_prefix(proposals, probe.predictions);
    probe.committed_count = probe.accepted + 1;
    probe.next_input = flash::commit_tokens(proposals, probe.predictions).back();

    // Second pass: emulate the old commit's resulting live state from the pre-verify state, with
    // the old per-decode sink write active.
    harness.reset();
    CUDA_CHECK(cudaMemset(sink.p, 0, sink.bytes));
    harness.trunk->set_residual_sink(0, sink.p, sink.bytes);
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::UpdateInPlace);
    (void)harness.prefill(prompt);
    for (std::uint32_t index = 0; index < batched.committed_count; ++index) {
        (void)harness.decode(verify_tokens[index]);
    }
    probe.next_position = harness.states[0]->next_position();
    probe.unsynced_sink = read_sink_unsynced(sink.p);
    probe.sink_after_forward = read_sink(harness, sink.p);
    probe.sink = probe.sink_after_forward;
    require(probe.accepted == batched.accepted && probe.committed_count == batched.committed_count,
            "sequential and batched routes disagree on acceptance");
    fill_kv_digest(harness, probe, prompt);
    probe.next_logits = capture_next_logits(harness, probe.next_input);
    return probe;
}

// Route B: production batched T=2 verify (RecordForReplay) plus commit_verified_span, whose tail
// relocates the residual sink to verification_residual(committed_count-1).
RouteResult route_batched(FlashHarness& harness, const std::vector<TokenId>& prompt,
                          const std::vector<TokenId>& verify_tokens, DeviceBuffer& sink) {
    harness.reset();
    CUDA_CHECK(cudaMemset(sink.p, 0, sink.bytes));
    harness.trunk->set_residual_sink(0, sink.p, sink.bytes);
    (void)harness.prefill(prompt);

    const std::vector<TokenId> proposals(verify_tokens.begin() + 1, verify_tokens.end());
    const auto baseline = harness.trunk->capture_verification_baseline();
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::RecordForReplay);
    RouteResult out;
    out.predictions = harness.trunk->decode_span(verify_tokens);
    harness.trunk->set_gdn_state_action(rt::GdnStateAction::UpdateInPlace);
    out.accepted = flash::accepted_prefix(proposals, out.predictions);
    out.committed_count = out.accepted + 1;
    const std::vector<TokenId> committed = flash::commit_tokens(proposals, out.predictions);
    out.next_input = committed.back();

    out.sink_after_forward = read_sink(harness, sink.p);
    harness.trunk->commit_verified_span(verify_tokens, out.committed_count, baseline);
    out.next_position = harness.states[0]->next_position();
    out.unsynced_sink = read_sink_unsynced(sink.p);
    out.sink = read_sink(harness, sink.p);

    rt::TextState& state = *harness.states[0];
    CUDA_CHECK(cudaSetDevice(state.device_id()));
    CUDA_CHECK(cudaStreamSynchronize(state.stream()));
    const Tensor record = state.verification_residual(out.committed_count - 1);
    out.verification_record.resize(record.bytes() / 2);
    CUDA_CHECK(cudaMemcpy(out.verification_record.data(), record.data, record.bytes(),
                          cudaMemcpyDeviceToHost));
    fill_kv_digest(harness, out, prompt);
    out.next_logits = capture_next_logits(harness, out.next_input);
    return out;
}

struct SinkDiff {
    std::size_t bit_mismatch = 0;
    std::size_t first_mismatch = 0;
    float max_abs = 0.0F;
    double sum_abs = 0.0;
    bool equal = false;
};

float fp16_word(std::uint16_t word) {
    __half value;
    std::memcpy(&value, &word, sizeof(value));
    return __half2float(value);
}

SinkDiff compare_words(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
    require(a.size() == b.size(), "sink slice extents differ");
    SinkDiff diff;
    diff.first_mismatch = a.size();
    for (std::size_t index = 0; index < a.size(); ++index) {
        if (a[index] != b[index]) {
            ++diff.bit_mismatch;
            diff.first_mismatch = std::min(diff.first_mismatch, index);
        }
        const float delta =
            std::abs(fp16_word(a[index]) - fp16_word(b[index]));
        diff.max_abs = std::max(diff.max_abs, delta);
        diff.sum_abs += delta;
    }
    diff.equal = diff.bit_mismatch == 0;
    return diff;
}

void report(const char* scenario, std::size_t prompt_id, const RouteResult& a,
            const RouteResult& b) {
    const SinkDiff final_diff = compare_words(a.sink, b.sink);
    const SinkDiff pre_commit_diff = compare_words(a.sink, b.sink_after_forward);
    const SinkDiff record_diff = compare_words(b.verification_record, b.sink);
    const SinkDiff next_logits_diff = compare_words(a.next_logits, b.next_logits);
    const SinkDiff seq_race = compare_words(a.unsynced_sink, a.sink);
    const SinkDiff batch_race = compare_words(b.unsynced_sink, b.sink);
    const SinkDiff batch_stale_source = compare_words(b.unsynced_sink, b.sink_after_forward);
    std::cout << "SINK_DIAG scenario=" << scenario << " prompt=" << prompt_id
              << " accepted=" << a.accepted << "/" << b.accepted
              << " committed=" << a.committed_count << "/" << b.committed_count
              << " sink_words=" << a.sink.size()
              << " final_bit_mismatch=" << final_diff.bit_mismatch
              << " final_first_mismatch=" << final_diff.first_mismatch
              << " final_max_abs=" << final_diff.max_abs
              << " final_mean_abs=" << (a.sink.empty() ? 0.0 : final_diff.sum_abs / a.sink.size())
              << " final_equal=" << final_diff.equal
              << " precommit_bit_mismatch=" << pre_commit_diff.bit_mismatch
              << " record_vs_sink_bit_mismatch=" << record_diff.bit_mismatch
              << " next_input=" << a.next_input << "/" << b.next_input
              << " next_input_equal=" << (a.next_input == b.next_input)
              << " next_position=" << a.next_position << "/" << b.next_position
              << " next_position_equal=" << (a.next_position == b.next_position)
              << " predictions_equal=" << (a.predictions == b.predictions)
              << " kv_equal=" << (a.kv_digest == b.kv_digest)
              << " kv_seq=" << a.kv_digest[0] << "," << a.kv_digest[1]
              << " kv_batched=" << b.kv_digest[0] << "," << b.kv_digest[1]
              << " next_logits_bit_mismatch=" << next_logits_diff.bit_mismatch
              << " next_logits_max_abs=" << next_logits_diff.max_abs
              << " next_logits_equal=" << next_logits_diff.equal
              << " unsynced_vs_synced_seq=" << seq_race.bit_mismatch
              << " unsynced_vs_synced_batched=" << batch_race.bit_mismatch
              << " unsynced_vs_precommit_batched=" << batch_stale_source.bit_mismatch << '\n';
    if (!final_diff.equal) {
        const std::size_t at = final_diff.first_mismatch;
        std::cout << "SINK_SAMPLE scenario=" << scenario << " prompt=" << prompt_id
                  << " index=" << at << " seq_word=" << a.sink[at]
                  << " batched_word=" << b.sink[at]
                  << " seq_float=" << fp16_word(a.sink[at])
                  << " batched_float=" << fp16_word(b.sink[at]) << '\n';
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (artifact == nullptr || artifact[0] == '\0') {
        std::cout << "skip: residual sink diagnosis needs the artifact\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }
    try {
        FlashHarness harness(artifact, static_cast<std::uint32_t>(kVerify));
        DeviceBuffer sink_a(kSinkBytes);
        DeviceBuffer sink_b(kSinkBytes);
        for (std::size_t prompt_id = 0; prompt_id < kPrompts.size(); ++prompt_id) {
            const auto& prompt = kPrompts[prompt_id];
            const auto reference = reference_sequence(harness, prompt);
            std::vector<TokenId> verify_tokens{reference[0], reference[1]};
            const RouteResult batched_full = route_batched(harness, prompt, verify_tokens, sink_b);
            report("full", prompt_id,
                   route_sequential(harness, prompt, verify_tokens, sink_a, batched_full),
                   batched_full);

            std::vector<TokenId> rejected = verify_tokens;
            rejected[1] = static_cast<TokenId>(reference[1] + 1000);
            const RouteResult batched_reject = route_batched(harness, prompt, rejected, sink_b);
            report("reject", prompt_id,
                   route_sequential(harness, prompt, rejected, sink_a, batched_reject),
                   batched_reject);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK residual sink diagnosis completed (report only)\n";
    return 0;
}
