#include "runtime/engine/shared_model.h"

#include <utility>

namespace ninfer {

SharedModel::SharedModel(std::filesystem::path artifact_path)
    : path_(std::move(artifact_path)), reader_(path_) {}

} // namespace ninfer
