// Back ported from LLVM v21
//===- ForallToParallel.cpp - scf.forall to scf.parallel loop conversion --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms SCF.ForallOp's into SCF.ParallelOps's.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "comet/Conversion/ForAllToParallel/ForAllToParallelPass.h"

#define GEN_PASS_CLASSES
#include "comet/Conversion/ForAllToParallel/Passes.h.inc"


using namespace mlir;

LogicalResult forallToParallelLoop(RewriterBase &rewriter,
                                              scf::ForallOp forallOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);

  Location loc = forallOp.getLoc();
  if (!forallOp.getOutputs().empty())
    return rewriter.notifyMatchFailure(
        forallOp,
        "only fully bufferized scf.forall ops can be lowered to scf.parallel");

  // Convert mixed bounds and steps to SSA values.
  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);

  // Create empty scf.parallel op.
  auto parallelOp = rewriter.create<scf::ParallelOp>(loc, lbs, ubs, steps);
  rewriter.eraseBlock(&parallelOp.getRegion().front());
  rewriter.inlineRegionBefore(forallOp.getRegion(), parallelOp.getRegion(),
                              parallelOp.getRegion().begin());
  // Replace the terminator.
  rewriter.setInsertionPointToEnd(&parallelOp.getRegion().front());
  rewriter.replaceOpWithNewOp<scf::ReduceOp>(
      parallelOp.getRegion().front().getTerminator());

  // If the mapping attribute is present, propagate to the new parallelOp.
  if (forallOp.getMapping())
    parallelOp->setAttr("mapping", *forallOp.getMapping());

  // Erase the scf.forall op.
  rewriter.replaceOp(forallOp, parallelOp);

  // if (result)
  //   *result = parallelOp;

  return success();
}

namespace {
struct ForallToParallelLoop final
    : public ForAllToParallelBase<ForallToParallelLoop> {
  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    parentOp->walk([&](scf::ForallOp forallOp) {
      if (failed(forallToParallelLoop(rewriter, forallOp))) {
        return signalPassFailure();
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::comet::createConvertForAllToParallelPass() {
  return std::make_unique<ForallToParallelLoop>();
}
