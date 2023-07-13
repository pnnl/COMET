//===- GPUHostRegister.cpp------===//
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
// This file adds gpu host_register op for mapping host memory to device space.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Conversion/TensorAlgebraToSCF/TensorAlgebraToSCF.h"
#include "comet/Conversion/SCFToGPU/SCFToGPU.h"
#include "comet/Conversion/Utils/GPUUtils.h"
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
// #ifndef DEBUG_MODE_GPUHostRegister
// #define DEBUG_MODE_GPUHostRegister
// #endif

#ifdef DEBUG_MODE_GPUHostRegister
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
struct GPUHostRegisterPass
    : public PassWrapper<GPUHostRegisterPass, OperationPass<func::FuncOp>> {


  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUHostRegisterPass)

private:
  
 public:
  GPUHostRegisterPass() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    comet_debug() << "[DEBUG][GPU][Start] GPUHostRegister\n";

    MLIRContext *context = &getContext();

    auto funcOp = getOperation();
    DenseSet<Operation *> allocOps;
    funcOp.walk([&allocOps](memref::AllocOp allocOp) { allocOps.insert(allocOp); });

    IRRewriter rewriter(context);
    FloatType f64Type = FloatType::getF64(context);
    Type unrankedMemrefType_f64 = UnrankedMemRefType::get(f64Type, 0);

    // look for memref::allocs use in gpu.launch_func.
    // if found being used in the launch_func, then a host_register op needs to be created.
    bool found_use = false;
    bool found_insert = false;
    for (auto alloc : allocOps) {
      for (auto n : alloc->getUsers()) {
        if (isa<gpu::LaunchFuncOp>(n)) {
          //comet_debug() << "Found use of alloc in launch func\n";
          found_use = true;
        }

        if (isa<linalg::FillOp>(n)) {
          // insert the new ops after linalg.fill 
          found_insert = true;
          rewriter.setInsertionPointAfter(n);
        }
      }

      if (found_use) {

        Location loc = alloc->getLoc();
        if (!found_insert) 
          rewriter.setInsertionPointAfter(alloc);

        auto alloc_op = cast<memref::AllocOp>(alloc);
        auto newCastOp = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_f64, alloc_op);
        comet_vdump(newCastOp);

        auto hostRegOp = rewriter.create<gpu::HostRegisterOp>(loc, newCastOp);
        comet_vdump(hostRegOp);
      }

      found_use = false; // reset for next alloc
      found_insert = false; 
    }

    comet_debug() << "[DEBUG][GPU][End] GPUHostRegister\n";
  }
};

}  // namespace


/// Create a pass to add host register op to map host memory to device address space.
std::unique_ptr<Pass> mlir::comet::createGPUHostRegisterOpPass() {
  return std::make_unique<GPUHostRegisterPass>();
}