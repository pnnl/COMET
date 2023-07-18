//===- GPUVectorization.cpp------===//
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTICE: The source code has been modified for integration with COMET.
//===----------------------------------------------------------------------===//
//
// This file performs various vectorizations.
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
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
// #ifndef DEBUG_MODE_GPUVectorization
// #define DEBUG_MODE_GPUVectorization
// #endif

#ifdef DEBUG_MODE_GPUVectorization
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

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(RewritePatternSet &patterns,
                                          int64_t maxVectorSize) {
  MLIRContext *ctx = patterns.getContext();

  // vectorization of memref.copyOp
  patterns.add<mlir::linalg::CopyVectorizationPattern>(ctx);
  mlir::linalg::populatePadOpVectorizationPatterns(patterns);
}

namespace {
struct GPUVectorizationPass
    : public PassWrapper<GPUVectorizationPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUVectorizationPass)

  GPUVectorizationPass(int64_t maxVectorSize) {
    this->maxVectorSize = maxVectorSize;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    comet_debug() << "[DEBUG][GPU][Start] GPUVectorization\n";
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    
    RewritePatternSet vectorizationPatterns(context);
    populateVectorizationPatterns(vectorizationPatterns, maxVectorSize);
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(
          vectorizationPatterns);
    mlir::vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
                          
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorizationPatterns)))) {
      return signalPassFailure();
    }

    mlir::linalg::hoistRedundantVectorTransfersOnTensor(funcOp);

    comet_debug() << "[DEBUG][GPU][End] GPUVectorization\n";
  }

  private:
    int64_t maxVectorSize;
  
};

}  // namespace


/// Create a pass to do vectorization
std::unique_ptr<Pass> mlir::comet::createGPUVectorizationPass(int64_t maxVectorSize) {
  return std::make_unique<GPUVectorizationPass>(maxVectorSize);
}