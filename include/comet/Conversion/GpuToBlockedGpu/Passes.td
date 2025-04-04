#ifndef COMET_GPU_TO_BLOCKED_GPU_CONVERSION_PASSES
#define COMET_GPU_TO_BLOCKED_GPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def CometGpuToBlockedGpu : Pass<"comet-gpu-to-blocked-gpu", "mlir::gpu::GPUFuncOp"> {
  let summary = "Map loops to gpu";

  let constructor = "mlir::comet::createConvertGpuToBlockedGpuPass()";

  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::affine::AffineDialect", "mlir::gpu::GPUDialect", "mlir::vector::VectorDialect", "mlir::tensor::TensorDialect"];
}

#endif
