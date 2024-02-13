set(LLVM_TARGET_DEFINITIONS SymbolicDomainPatterns.td)
mlir_tablegen(SymbolicDomainPatterns.inc -gen-rewriters)
add_public_tablegen_target(COMETSymbolicDomainPatternsIncGen)

add_mlir_conversion_library(COMETIndexTreeToSCF
  IndexTreeToSCF.cpp
  ConvertDomains.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/IndexTreeToSCF

  DEPENDS
  COMETConversionPassIncGen
  COMETSymbolicDomainPatternsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIndexDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  )
