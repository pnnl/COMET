# add_mlir_dialect_library(COMETTensorAlgebraDialect
#   IR/TADialect.cpp

#   Transforms/Transforms.cpp
#   Transforms/LinalgTransforms.cpp
#   Transforms/TCtoTTGT.cpp
#   Transforms/Passes.cpp

#   Transforms/CheckImplicitTensorDecls.cpp
#   Transforms/TensorDeclLowering.cpp

#   ADDITIONAL_HEADER_DIRS
#   ${COMET_MAIN_INCLUDE_DIR}/comet/Dialect/TensorAlgebra

#   DEPENDS
#   COMETTensorAlgebraOpsIncGen
#   COMETTensorAlgebraPassIncGen

#   LINK_LIBS PUBLIC
#   MLIRLinalgTransformOpsIncGen
#   # # MLIR
#   # ${BLAS_LIBRARIES} 
# )

# target_link_libraries(COMETTensorAlgebraDialect MLIRIR ${BLAS_LIBRARIES})

add_llvm_library(COMETTensorAlgebraDialect
  IR/TADialect.cpp

  Transforms/Transforms.cpp
  Transforms/LinalgTransforms.cpp
  Transforms/TCtoTTGT.cpp
  Transforms/Passes.cpp

  Transforms/CheckImplicitTensorDecls.cpp
  Transforms/TensorDeclLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Dialect/TensorAlgebra
  )

add_dependencies(
  COMETTensorAlgebraDialect
  COMETTensorAlgebraOpsIncGen
  COMETTensorAlgebraPassIncGen
  )

target_link_libraries(COMETTensorAlgebraDialect MLIRIR ${BLAS_LIBRARIES})

