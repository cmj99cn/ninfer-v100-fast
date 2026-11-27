add_library(ninfer_qwen4_exp_family STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/frontend.cpp)

ninfer_internal_includes(ninfer_qwen4_exp_family)
target_include_directories(ninfer_qwen4_exp_family PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/export)
target_link_libraries(ninfer_qwen4_exp_family
  PUBLIC
    ninfer_artifact
    ninfer_core
    ninfer_qwen_frontend)
