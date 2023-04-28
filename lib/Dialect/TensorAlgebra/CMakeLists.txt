add_llvm_library(COMETTensorAlgebraDialect
  IR/TADialect.cpp

  # Conversion/EarlyLowering.cpp
  # Conversion/PreLowering.cpp
  # Conversion/LowerPCToLoops.cpp
  # Conversion/TensorOpsLowering.cpp
  # Conversion/LateLowering.cpp 
  # Conversion/LowerFunc.cpp 

  Transforms/Transforms.cpp
  #Transforms/LinalgTransforms.cpp
  #Transforms/TCtoTTGT.cpp
  Transforms/Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Dialect/TensorAlgebra
  )


add_dependencies(
  COMETTensorAlgebraDialect
  COMETTensorAlgebraOpsIncGen
  )

target_link_libraries(COMETTensorAlgebraDialect MLIRIR)

