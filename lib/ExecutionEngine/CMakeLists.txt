add_llvm_library(comet_runner_utils
  SHARED
  blis_interface.cpp
  StatUtils.cpp
  SparseUtils.cpp
  TransposeUtils.cpp
)

target_compile_definitions(comet_runner_utils PRIVATE comet_runner_utils_EXPORTS comet_blis_interface_EXPORTS)
target_link_libraries(comet_runner_utils  ${COMET_BLIS_LIB})
