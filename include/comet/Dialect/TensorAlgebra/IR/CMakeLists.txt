set(LLVM_TARGET_DEFINITIONS TAOps.td)
mlir_tablegen(TAOps.h.inc -gen-op-decls)
mlir_tablegen(TAOps.cpp.inc -gen-op-defs)
mlir_tablegen(TADialect.h.inc -gen-dialect-decls)
mlir_tablegen(TADialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(COMETTensorAlgebraOpsIncGen)