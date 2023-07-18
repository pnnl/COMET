//===- GPUOptimizeVectorTransfer.cpp------===//
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTICE: The source code has been modified for integration with COMET.
//===----------------------------------------------------------------------===//
//
// This file optimizes vector transfers.
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
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_GPUOptimizeVectorTransfer
// #define DEBUG_MODE_GPUOptimizeVectorTransfer
// #endif

#ifdef DEBUG_MODE_GPUOptimizeVectorTransfer
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

static void loopInvariantCodeMotion(func::FuncOp funcOp) {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  funcOp.walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}

// Return true if all the uses of op are either Store/transfer_write.
// There can be SubviewOp users as long as all its users are also
// StoreOp/transfer_write. If return true it also fills out the uses, if it
// returns false uses is unchanged.
static bool allUsesAreStores(Operation* op, std::vector<Operation*>& uses) {
  std::vector<Operation*> opUses;
  for (OpOperand& use : op->getUses()) {
    Operation* useOp = use.getOwner();
    if (isa<vector::TransferWriteOp, memref::StoreOp>(useOp) ||
        (isa<memref::SubViewOp>(useOp) && allUsesAreStores(useOp, opUses))) {
      opUses.push_back(useOp);
      continue;
    }
    return false;
  }
  uses.insert(uses.end(), opUses.begin(), opUses.end());
  return true;
}

// Track temporary allocations that are never read from. If this is the case
// it means both the allocations and associated stores can be removed.
static void eraseDeadAllocAndStores(func::FuncOp funcOp) {
  std::vector<Operation*> opToErase;
  funcOp.walk([&](memref::AllocOp op) {
    if (allUsesAreStores(op, opToErase)) {
      opToErase.push_back(op.getOperation());
    }
  });
  for (Operation* op : opToErase) {
    op->erase();
  }
}

namespace {
struct GPUVectorTransferPass
    : public PassWrapper<GPUVectorTransferPass, OperationPass<func::FuncOp>> {


  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUVectorTransferPass)

private:
  bool flatten;
  bool optionFlatten = false;

 public:
  GPUVectorTransferPass(bool flatten)
      : flatten(flatten) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    comet_debug() << "[DEBUG][GPU][Begin] GPUOptimizeVectorTransfer\n";
    func::FuncOp funcOp = getOperation();
    // Generate vector.shape_cast for dropping leading one dimensions in vector
    // ops. This increases the chance that we can forward more transfer writes
    // to transfer reads.
    {
      RewritePatternSet patterns(&getContext());
      mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, &getContext());
      mlir::vector::
          populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
              patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Workaround, run loop invariant code motion before hoist redudant vector
    // transfer to workaround a bug upstream.
    // TODO(thomasraoux): Remove it once the fix is merged.
    loopInvariantCodeMotion(funcOp);
    linalg::hoistRedundantVectorTransfers(funcOp);
    IRRewriter rewriter(funcOp->getContext());
    vector::transferOpflowOpt(funcOp);

    // Move bitcast inwards from loop region boundaries to increase chances to
    // cancel them.
    {
      RewritePatternSet patterns(&getContext());
      vector::populateBubbleVectorBitCastOpPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Second stage of patterns to flatten transfer ops.
    if (flatten) {
      RewritePatternSet patterns(&getContext());
      mlir::vector::populateVectorTransferDropUnitDimsPatterns(patterns);
      mlir::vector::populateFlattenVectorTransferPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    // Delete potential dead alloc and associated ops after store to load
    // forwarding.
    eraseDeadAllocAndStores(funcOp);

    comet_debug() << "[DEBUG][GPU][End] GPUOptimizeVectorTransfer\n";
  }

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    // `flatten` may have been set to `true` in the constructor already.
    // The |= is so we preserve that rather than overwrite it with the default
    // value `false` of `optionFlatten`.
    flatten |= optionFlatten;
    return success();
  }
};

}  // namespace



/// Create a pass to do vectorization and padding
std::unique_ptr<Pass> mlir::comet::createGPUOptimizeVectorTransferPass(bool flatten) {
  return std::make_unique<GPUVectorTransferPass>(flatten);
}