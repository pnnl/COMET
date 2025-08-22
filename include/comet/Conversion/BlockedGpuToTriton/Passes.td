#ifndef COMET_BLOCKED_GPU_TO_TRITON_CONVERSION_PASSES
#define COMET_BLOCKED_GPU_TO_TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def CometBlockedGpuToTriton : Pass<"comet-blocked-gpu-to-triton", "mlir::ModuleOp"> {
  let summary = "Map blocked gpu to trtion";

  let constructor = "mlir::comet::createConvertBlockedGpuToTriton()";

  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::gpu::GPUDialect", "mlir::tensor::TensorDialect"];
}

#endif
