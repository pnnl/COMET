//===- GPUSharedMemPlacement.cpp------===//
//
// Copyright 2022 Battelle Memorial Institute
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// create shared memory copies for gpu kernel operands
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Conversion/TensorAlgebraToSCF/TensorAlgebraToSCF.h"
#include "comet/Conversion/SCFToGPU/SCFToGPU.h"
#include "comet/Conversion/Utils/MarkerUtils.h"
#include "comet/Conversion/TensorAlgebraToIndexTree/TensorAlgebraToIndexTree.h"
#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"


#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::gpu;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_GPUSharedMemPlacement
// #define DEBUG_MODE_GPUSharedMemPlacement
// #endif

#ifdef DEBUG_MODE_GPUSharedMemPlacement
#define comet_debug() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n)                                \
  llvm::errs() << __FILE__ << " " << __LINE__ << " "; \
  n->dump()
#define comet_vdump(n)                                \
  llvm::errs() << __FILE__ << " " << __LINE__ << " "; \
  n.dump()
#else
#define comet_debug() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

namespace {
struct GPUSharedMemPlacementPass
    : public PassWrapper<GPUSharedMemPlacementPass, OperationPass<func::FuncOp>> {


  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUSharedMemPlacementPass)

private:
  
 public:
  GPUSharedMemPlacementPass() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
  }

  // Creates fast buffers (shared memory)
  static void createAndPlaceFastBuffers(AffineForOp rootForOp,
                                        OpBuilder opBuilder) {
    
    SmallVector<AffineForOp, 6> loopNest;
    getPerfectlyNestedLoops(loopNest, rootForOp);

    // Checks if the loop nest is perfectly nested or not. The pass doesn't work
    // in case of imperfect loop nest.
    assert(loopNest.size() > 2 && "Expected perfectly nested loop nest.");

    SmallVector<Value, 4> inputMemrefs;
    Value outputMemRef;
    // Identify the input and output memrefs.
    rootForOp.walk(
      [&](AffineStoreOp storeOp) { outputMemRef = storeOp.getMemRef(); });
    
    rootForOp.walk([&](AffineLoadOp loadOp) {
        // Checks if the loadOp's memref is equal to output memref, if yes then
        // it's the output matrix's memref and skip it.
        if (outputMemRef == loadOp.getMemRef())
            return;
        inputMemrefs.push_back(loadOp.getMemRef());
    });
    comet_debug() << "[DEBUG][GPU] done collecting input and output memrefs\n";

    // Intialize the copy options for placing matrices into fast buffers.
    AffineCopyOptions copyOptions = {/*generateDma=*/false,
                                   /*slowMemorySpace=*/0,
                                   /*fastMemorySpace=*/3,  // GPU shared memory
                                   /*tagMemorySpace=*/0,
                                   /*fastMemCapacityBytes=*/32 * 1024 * 1024UL};
    
    comet_debug() << "[DEBUG][GPU] making a call to affineDataCopyGenerate()\n";
    DenseSet<Operation *> copyNests;
    // TODO: make this more general for a variety of kernels.
    for (int i = 0; i < inputMemrefs.size(); i++) {
      if (failed(affineDataCopyGenerate(loopNest[2], copyOptions, inputMemrefs[i], copyNests)))
        return;
    }

    for (Operation *copyNest : copyNests)
      copyNest->setAttr("isCopyLoopNest", opBuilder.getBoolAttr(true));
    
    // Checks if the loop nest to be marked is present or not.
    if (loopNest[3])
       loopNest[3]->setAttr("isComputeLoopNest", opBuilder.getBoolAttr(true));
    
    comet_debug() << "[DEBUG][GPU] DONE call to affineDataCopyGenerate()\n";

  }

  void runOnOperation() override {
    comet_debug() << "[DEBUG][GPU][Start] GPUSharedMemPlacement\n";

    MLIRContext *context = &getContext();

    auto funcOp = getOperation();
    for (Block &block : funcOp) {
        for (Operation &op : block) {
            // the outer for loop.
            if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
                if (!forOp->getParentOfType<AffineForOp>()) {
                    OpBuilder opBuilder(forOp);
                    createAndPlaceFastBuffers(forOp, opBuilder);
                }         
            }
        }
    }
    comet_debug() << "[DEBUG][GPU][End] GPUSharedMemPlacement\n";
  }
};

}  // namespace


/// Create a pass to place shared memory buffers using affineDataCopyGenerate().
std::unique_ptr<Pass> mlir::comet::createGPUSharedMemPlacementPass() {
  return std::make_unique<GPUSharedMemPlacementPass>();
}