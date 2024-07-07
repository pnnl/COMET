set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(comet-opt
  comet.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
)

include_directories(include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

llvm_update_compile_flags(comet-opt)
option(ENABLE_GPU_TARGET OFF) 

if(${ENABLE_GPU_TARGET})
  set(TRITON_BUILD_PATH "" CACHE PATH "Path to Triton Installation")
  if(NOT EXISTS "${TRITON_BUILD_PATH}/bin/triton-opt")
  message(FATAL_ERROR "triton-opt  not found: ${TRITON_BUILD_PATH}/bin/triton-opt")
  endif()
  configure_file(TritonConfig.h.in ${CMAKE_BINARY_DIR}/include/comet/TritonConfig.h)
  add_definitions(-DENABLE_GPU_TARGET)
endif()

set(LIBS
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  COMETUtils
  COMETTensorAlgebraDialect
  COMETIndexTreeDialect
  COMETIndexTreeToSCF
)

target_link_libraries(comet-opt 
    PRIVATE MLIRIR 
    ${LIBS} 
    ${dialect_libs} 
    ${conversion_libs}
    )