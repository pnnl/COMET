add_mlir_conversion_library(COMETTensorAlgebraToIndexTree
  TensorAlgebraToIndexTree.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/TensorAlgebraToIndexTree

  DEPENDS
  COMETConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  COMETIndexTreeDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  )
