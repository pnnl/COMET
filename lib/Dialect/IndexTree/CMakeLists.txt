add_llvm_library(COMETIndexTreeDialect
  IR/IndexTreeDialect.cpp
  IR/IndexTree.cpp

  Transforms/IterationDomain.cpp
  Transforms/Tensor.cpp
  Transforms/UnitExpression.cpp
  Transforms/WorkspaceTransforms.cpp 
  Transforms/Fusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Dialect/IndexTree
  )


add_dependencies(
  COMETIndexTreeDialect
  COMETIndexTreeOpsIncGen
  COMETIndexTreeTypesIncGen
  COMETIndexTreePassIncGen
  MLIRSupport
  )

target_link_libraries(COMETIndexTreeDialect MLIRIR)

