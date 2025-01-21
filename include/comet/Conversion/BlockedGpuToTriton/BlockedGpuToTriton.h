#ifndef COMET_CONVERSION_BLOCKEDGPUTOTRITON_H
#define COMET_CONVERSION_BLOCKEDGPUTOTRITON_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include <memory>
namespace mlir {
class ModuleOp;
namespace gpu{
class GPUFuncOp;
}
template <typename T> class OperationPass;
namespace comet {
std::unique_ptr<OperationPass<mlir::ModuleOp> > createConvertBlockedGpuToTritonPass();
}
}

#endif