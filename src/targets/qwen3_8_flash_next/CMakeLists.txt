add_library(ninfer_flash_next_package STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/leaves.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/mtp_draft.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/speculative.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state_transaction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/expert_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/expert_runtime.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/geometry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/mtp_bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/roles.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/sharding.cpp)

ninfer_internal_includes(ninfer_flash_next_package)
target_include_directories(ninfer_flash_next_package
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/export)
target_link_libraries(ninfer_flash_next_package
  PUBLIC
    ninfer_qwen4_exp_family
    ninfer_artifact
    ninfer_core
    ninfer_ops)
