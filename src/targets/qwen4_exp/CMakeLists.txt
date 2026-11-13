add_library(ninfer_qwen4_exp_runtime INTERFACE)
target_include_directories(ninfer_qwen4_exp_runtime INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/export)

target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/frontend.cpp)

target_include_directories(ninfer_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/impl)
