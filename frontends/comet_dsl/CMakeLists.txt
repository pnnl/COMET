set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(comet-opt
  comet.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
)

include_directories(include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

llvm_update_compile_flags(comet-opt)

set(LIBS
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  COMETUtils
  COMETTensorAlgebraDialect
  COMETIndexTreeDialect
  COMETIndexTreeToSCF
)

target_link_libraries(comet-opt 
    PRIVATE MLIRIR 
    ${LIBS} 
    ${dialect_libs} 
    ${conversion_libs}
    )