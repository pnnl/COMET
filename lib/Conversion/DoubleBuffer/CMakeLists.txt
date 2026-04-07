add_mlir_conversion_library(COMETDoubleBuffer
  DoubleBuffer.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/DoubleBuffer

  DEPENDS
  COMETConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRPass
  MLIRTransforms
  COMETIndexTreeDialect
  COMETTensorAlgebraDialect
)
