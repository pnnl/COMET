//===- SCFToGPU.cpp------===//
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
// This file SCF loops to GPU Ops.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Conversion/TensorAlgebraToSCF/TensorAlgebraToSCF.h"
#include "comet/Conversion/SCFToGPU/SCFToGPU.h"
#include "comet/Conversion/TensorAlgebraToIndexTree/TensorAlgebraToIndexTree.h"
#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_SCFToGPUPass
// #define DEBUG_MODE_SCFToGPUPass
// #endif

#ifdef DEBUG_MODE_SCFToGPUPass
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

/// This is a lowering of SCF ops to GPU ops

namespace
{
  struct LowerSCFToGPUPass
      : public PassWrapper<LowerSCFToGPUPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSCFToGPUPass)

    void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, AffineDialect, func::FuncDialect, memref::MemRefDialect>();
  }

    void runOnOperation() override;
  };
} // end anonymous namespace.

void LowerSCFToGPUPass::runOnOperation()
{
  func::FuncOp function = getOperation();

  // debug
  //auto module = function.getOperation()->getParentOfType<ModuleOp>();
  //module->dump();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<gpu::GPUDialect,
                         LinalgDialect,
                         AffineDialect,
                         scf::SCFDialect,
                         ArithDialect,
                         memref::MemRefDialect,
                         bufferization::BufferizationDialect>();

  RewritePatternSet patternsAlloc(&getContext());
  //patternsAlloc.insert<>(&getContext());
 
  RewritePatternSet patterns(&getContext());
  populateParallelLoopToGPUPatterns(patterns);  // gpu.launch_func created here
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  configureParallelLoopToGPULegality(target);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
  {
    signalPassFailure();
  }

  finalizeParallelLoopToGPUConversion(getOperation());
}

/// Create a pass for lowering SCF operations in the SCF dialect to GPU ops in the GPU dialect
std::unique_ptr<Pass> mlir::comet::createLowerSCFToGPUPass()
{
  return std::make_unique<LowerSCFToGPUPass>();
}