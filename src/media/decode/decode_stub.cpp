#include "media/decode/decode.h"

namespace ninfer::media::decode {

Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    throw std::runtime_error(
        "media decode unavailable: built with NINFER_SKIP_FFMPEG=ON");
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw std::runtime_error(
        "media decode unavailable: built with NINFER_SKIP_FFMPEG=ON");
}

} // namespace ninfer::media::decode
