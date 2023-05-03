set(LLVM_TARGET_DEFINITIONS IndexTreeOps.td)
mlir_tablegen(IndexTreeOps.h.inc -gen-op-decls)
mlir_tablegen(IndexTreeOps.cpp.inc -gen-op-defs)
mlir_tablegen(IndexTreeDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IndexTreeDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(COMETIndexTreeOpsIncGen)


