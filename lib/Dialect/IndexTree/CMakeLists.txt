add_mlir_dialect_library(COMETIndexTreeDialect
  IR/IndexTreeDialect.cpp
  IR/IndexTree.cpp

  # Transforms/IterationDomain.cpp
  # Transforms/Tensor.cpp
  # Transforms/UnitExpression.cpp
  # Transforms/WorkspaceTransforms.cpp 
  # Transforms/Fusion.cpp
  Transforms/IterationDomainInference.cpp
  Transforms/DomainConcretization.cpp
  Transforms/SymbolicCompute.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Dialect/IndexTree

  DEPENDS
  COMETIndexTreeOpsIncGen
  COMETIndexTreeTypesIncGen
  COMETIndexTreePassIncGen
  MLIRSupport

  LINK_LIBS PUBLIC
  MLIRIR
)
