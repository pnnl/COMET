set(LLVM_TARGET_DEFINITIONS ITOps.td)
mlir_tablegen(ITOps.h.inc -gen-op-decls)
mlir_tablegen(ITOps.cpp.inc -gen-op-defs)
mlir_tablegen(ITDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ITDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(COMETITOpsIncGen)


