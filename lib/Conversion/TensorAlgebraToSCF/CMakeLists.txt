add_mlir_conversion_library(COMETTensorAlgebraToSCF
  EarlyLowering.cpp
  LateLowering.cpp
  LowerFunc.cpp
  LowerPCToLoops.cpp 
  TensorAlgebraToSCF.cpp


  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/TensorAlgebraToIndexTree

  DEPENDS
  COMETConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  )
