#ifndef TRITON_CONVERSION_TRITONTOCUDA_H
#define TRITON_CONVERSION_TRITONTOCUDA_H

#include <memory>
#include "comet/Dialect/Utils/Utils.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace comet {


std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerGpuHostToHIPPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerTritonDeviceToHIPPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerTritonDeviceToHIPPass(int numWarps,
                                                int threadsPerWarp,
                                                int numCTAs,
                                                int numStages,
                                                int computeCapability,
                                                mlir::tensorAlgebra::GPUCompilationFormat format);

} // namespace triton
} // namespace mlir

#endif