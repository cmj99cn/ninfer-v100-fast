add_library(ninfer_flash_next_package STATIC
  impl/package.cpp
  impl/leaves.cpp
  impl/load/bindings.cpp
  impl/load/geometry.cpp
  impl/load/roles.cpp)

ninfer_internal_includes(ninfer_flash_next_package)
target_include_directories(ninfer_flash_next_package PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/export)
target_link_libraries(ninfer_flash_next_package PUBLIC ninfer_artifact)
