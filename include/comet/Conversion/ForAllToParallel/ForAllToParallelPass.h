
#ifndef COMET_CONVERSION_FORALLTOPARALLEL_H
#define COMET_CONVERSION_FORALLTOPARALLEL_H

#include <memory>
namespace mlir {
class ModuleOp;
namespace func{
class FuncOp;
}

template <typename T> class OperationPass;
namespace comet {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertForAllToParallelPass();
}
}

#endif