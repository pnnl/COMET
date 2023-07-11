#ifndef COMET_TRANSFORMS_PASSES
#define COMET_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def InsertMCLCalls : Pass<"insert-mcl-calls", "ModuleOp"> {
    let summary = "Insert calls for the MCL runtime";
    let constructor = "mlir::createInsertMCLCallsPass()";
    let dependentDialects = ["gpu::GPUDialect", "LLVM::LLVMDialect", "memref::MemRefDialect", "scf::SCFDialect"];
}

def SCFWhileToFor : Pass<"raise-scf-while-to-for"> {
    let summary = "Attempt to find while ops that can be raised to for ops";
    let constructor = "mlir::createRaiseSCFWhileToForPass()";
    let dependentDialects = ["scf::SCFDialect"];
}

def GPUToOCLSPIRV : Pass<"gpu-to-ocl-spirv", "ModuleOp"> {
    let summary = "Set spirv targetEnv and entryPointABI attributes on gpu functions before lowering";
    let constructor = "mlir::createGPUToOCLSPIRVPass()";
    let dependentDialects = ["gpu::GPUDialect", "spirv::SPIRVDialect"];
}

// from https://github.com/llvm/Polygeist/blob/main/include/polygeist/Passes/Passes.td
def AffineCFG : Pass<"affine-cfg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::replaceAffineCFGPass()";
}

def SCFRaiseToAffine : Pass<"raise-scf-to-affine"> {
  let summary = "Raise SCF to affine";
  let constructor = "mlir::createRaiseSCFToAffinePass()";
  let dependentDialects = ["AffineDialect"];
}

#endif // COMET_TRANSFORMS_PASSES