#pragma once

// Process-level artifact reader shared by Engines that execute the same `.ninfer` file.
//
// The reader owns the mapped file and descriptor table; it is not safe for concurrent
// materialization. Engines that share one must be constructed and destroyed sequentially, which is
// exactly the ownership contract of the multi-instance Engine pool.

#include "artifact/reader.h"

#include <filesystem>

namespace ninfer {

class SharedModel {
public:
    explicit SharedModel(std::filesystem::path artifact_path);

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] artifact::Reader& reader() noexcept { return reader_; }

private:
    std::filesystem::path path_;
    artifact::Reader reader_;
};

} // namespace ninfer
