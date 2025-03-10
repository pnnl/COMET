#ifndef PREPARE_GPU_HOST_PASSES
#define PREPARE_GPU_HOST_PASSES

include "mlir/Pass/PassBase.td"

def PrepareGpuHost: Pass<"prepare-gpu-host", "mlir::ModuleOp"> {
       let summary = "Prepare GPU host";
       let description = [{}];
       let constructor = "mlir::comet::createPrepareGpuHostPass()";

       let dependentDialects = ["mlir::gpu::GPUDialect", "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}
#endif
