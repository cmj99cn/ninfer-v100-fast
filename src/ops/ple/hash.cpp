#include "ninfer/ops/ple.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::ops {

PleHashControls parse_ple_hash_controls(std::span<const std::byte> payload) {
    if (payload.size() != kPleHashControlsBytes) {
        throw std::invalid_argument("ple: the hash controls payload is not 280 bytes");
    }

    PleHashControls controls{};
    for (std::size_t word = 0; word < 35; ++word) {
        std::uint64_t value = 0;
        for (std::size_t byte = 0; byte < 8; ++byte) {
            value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(payload[word * 8 + byte]))
                   << (8U * byte);
        }
        if (word < 3) controls.multipliers[word] = value;
        else if (word < 19) controls.vocab_sizes[word - 3] = value;
        else controls.offsets[word - 19] = value;
    }
    return controls;
}

void ple_hash_tokens(std::span<const std::uint64_t> ids, std::array<std::uint64_t, 2> tail,
                     std::uint64_t eos, const PleHashControls& controls,
                     std::span<std::uint32_t> rows) {
    if (ids.empty()) {
        throw std::invalid_argument("ple: the hash needs at least one token");
    }
    if (rows.size() != ids.size() * kPleHeads) {
        throw std::invalid_argument("ple: the row output does not hold 16 rows per token");
    }

    std::array<std::uint64_t, 2> previous = tail;
    for (std::size_t token = 0; token < ids.size(); ++token) {
        const std::uint64_t id = ids[token];
        const std::uint64_t older = previous[1] == eos ? eos : previous[0];
        const std::uint64_t pair = id * controls.multipliers[0] ^ previous[1] * controls.multipliers[1];
        const std::uint64_t triple = pair ^ older * controls.multipliers[2];

        for (std::size_t head = 0; head < kPleHeads; ++head) {
            const std::int64_t word = static_cast<std::int64_t>(head < 8 ? pair : triple);
            const std::int64_t vocabulary = static_cast<std::int64_t>(controls.vocab_sizes[head]);
            std::int64_t index = word % vocabulary;
            if (index < 0) index += vocabulary;
            rows[head + kPleHeads * token] =
                static_cast<std::uint32_t>(index + static_cast<std::int64_t>(controls.offsets[head]));
        }
        previous = {previous[1], id};
    }
}

} // namespace ninfer::ops
