set(LLVM_TARGET_DEFINITIONS TAOps.td)
mlir_tablegen(TAOps.h.inc -gen-op-decls)
mlir_tablegen(TAOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(COMETTAOpsIncGen)
