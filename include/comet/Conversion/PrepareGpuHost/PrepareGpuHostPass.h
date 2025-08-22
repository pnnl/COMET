#ifndef COMET_PREPARE_GPU_HOST_PASS
#define COMET_PREPARE_GPU_HOST_PASS

#include <memory>
#include "comet/Dialect/Utils/Utils.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace comet {


std::unique_ptr<OperationPass<mlir::ModuleOp>> createPrepareGpuHostPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createPrepareGpuHostPass(bool generateAllocsAndTransfers);

} // namespace triton
} // namespace mlir

#endif