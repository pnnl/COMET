# 
add_llvm_library(COMET_BLIS
  bli_gemm_haswell_asm_d6x8.c
  bli_cpuid.c
  bli_arch.c
)

target_include_directories (COMET_BLIS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

