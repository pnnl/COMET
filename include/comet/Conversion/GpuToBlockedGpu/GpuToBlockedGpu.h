#ifndef COMET_CONVERSION_GPUTOBLOCKEDGPU_H
#define COMET_CONVERSION_GPUTOBLOCKEDGPU_H

#include <memory>
namespace mlir {
class ModuleOp;
namespace gpu{
class GPUFuncOp;
}
template <typename T> class OperationPass;
namespace comet {
std::unique_ptr<OperationPass<mlir::gpu::GPUFuncOp> > createConvertGpuToBlockedGpuPass();
}
}

#endif