#include "product/media_acquire/acquire.h"

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    throw std::runtime_error(
        "media acquire unavailable: built without NINFER_BUILD_MEDIA_ACQUIRE");
}

} // namespace ninfer::product::media_acquire
