set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(comet-opt
  comet.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
)

include_directories(include/)
if(ENABLE_GPU_TARGET)
  include_directories("${TRITON_PATH}")
  include_directories("${TRITON_PATH}/include")
  include_directories("${TRITON_BUILD_PATH}/include")
  get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

llvm_update_compile_flags(comet-opt)

set(LIBS
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRFuncAllExtensions
  MLIRTransforms
  COMETUtils
  COMETTensorAlgebraDialect
  COMETIndexTreeDialect
  # COMETIndexTreeToSCF
)

if(ENABLE_FPGA_TARGET)
  set(LIBS
    ${LIBS}
    COMETParallelLoopsToGpuFPGA
  )
endif()

if(ENABLE_GPU_TARGET OR ENABLE_FPGA_TARGET)
  set(LIBS
    ${LIBS}
    MLIRGPUToLLVMIRTranslation
  )
endif()


if(ENABLE_GPU_TARGET)
  set(LIBS
    ${LIBS}
    COMETGPUUtils
    COMETParallelLoopsToGpu
    COMETGpuToBlockedGpu
    COMETBlockedGpuToTriton
    #COMETGpuToTriton
    COMETPrepareGpuHost
    ${triton_libs}
  )
endif()


if(ENABLE_NVIDIA_GPU_BACKEND)
  set(LIBS
    ${LIBS}
    COMETTritonToCuda
  )
endif()

if(ENABLE_AMD_GPU_BACKEND)
  set(LIBS
    ${LIBS}
    COMETTritonToHIP
  )
endif()

if(ENABLE_FPGA_TARGET)
  set(LIBS
    ${LIBS}
    COMETGpuToOCLSPIRV
    COMETGpuHostToMCLRT
  )
endif()

target_link_libraries(comet-opt 
    PRIVATE MLIRIR 
    ${LIBS} 
    ${dialect_libs} 
    ${conversion_libs}
    )