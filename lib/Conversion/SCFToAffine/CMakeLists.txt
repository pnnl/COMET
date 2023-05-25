add_mlir_conversion_library(COMETSCFToAffine
  SCFToAffine.cpp


  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/TensorAlgebraToIndexTree
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/TensorAlgebraToSCF
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/IndexTreeToSCF
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/TensorAlgebraToGPU
  
  DEPENDS
  COMETConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  )