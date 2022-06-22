add_llvm_library(COMETTensorAlgebra
  IR/TADialect.cpp

  Conversion/EarlyLowering.cpp
  Conversion/PreLowering.cpp
  Conversion/SUMLowerToSCF.cpp
  Conversion/TransposeLowering.cpp
  Conversion/LateLowering.cpp 

  Transforms/Transforms.cpp
  Transforms/LinalgTransforms.cpp
  Transforms/TCtoTTGT.cpp
  Transforms/Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Dialect/TensorAlgebra
  )


add_dependencies(
  COMETTensorAlgebra
  COMETTAOpsIncGen
  )

target_link_libraries(COMETTensorAlgebra MLIRIR)

