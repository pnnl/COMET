set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(CometTransforms.h.inc -gen-pass-decls -name Comet)
add_public_tablegen_target(CometTransformsIncGen)