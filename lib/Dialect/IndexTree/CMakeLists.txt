add_llvm_library(COMETIndexTree
  IR/ITDialect.cpp
  IR/IndexTree.cpp

  Conversion/LowerIndexTreeIRToSCF.cpp
  Conversion/WorkspaceTransforms.cpp 

  Transforms/IndexTreePass.cpp
  Transforms/IterationDomain.cpp
  Transforms/Tensor.cpp
  Transforms/UnitExpression.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Dialect/IndexTree
  )


add_dependencies(
  COMETIndexTree
  COMETITOpsIncGen
  )

target_link_libraries(COMETIndexTree MLIRIR)

