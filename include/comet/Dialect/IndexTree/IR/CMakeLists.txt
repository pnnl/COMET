set(LLVM_TARGET_DEFINITIONS ITOps.td)
mlir_tablegen(ITOps.h.inc -gen-op-decls)
mlir_tablegen(ITOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(COMETITOpsIncGen)

