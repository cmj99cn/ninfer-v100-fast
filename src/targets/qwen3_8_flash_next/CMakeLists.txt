add_library(ninfer_flash_next_package INTERFACE)
target_include_directories(ninfer_flash_next_package INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/export)
target_link_libraries(ninfer_flash_next_package INTERFACE ninfer_qwen4_exp_runtime)

target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/leaves.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/geometry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/roles.cpp)

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
