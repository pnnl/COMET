add_mlir_conversion_library(COMETSCFToGPU
  SCFToGPU.cpp
  GPUVectorization.cpp
  GPUReduceBankConflicts.cpp
  GPUOptimizeVectorTransfer.cpp
  GPUPipelining.cpp
  GPUAffineVectorize.cpp
  GPUHostRegister.cpp
  GPUMemrefCopy.cpp


  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/TensorAlgebraToIndexTree
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/TensorAlgebraToSCF
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/IndexTreeToSCF

  DEPENDS
  COMETConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRGPUOps
  MLIRGPUTransformOps
  MLIRPass
  MLIRTransforms
  )