//===- SCFToAffine.cpp------===//
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
// This file implements the pass to raise scf.for to affine.for ops.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Conversion/TensorAlgebraToSCF/TensorAlgebraToSCF.h"
#include "comet/Conversion/SCFToAffine/SCFToAffine.h"
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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AffineExpr.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::tensorAlgebra;
using namespace mlir::scf;
using namespace mlir::memref;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_RaiseSCFToAffinePass
// #define DEBUG_MODE_RaiseSCFToAffinePass
// #endif

#ifdef DEBUG_MODE_RaiseSCFToAffinePass
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

namespace
{
// Raise scf.for to affine.for & replace scf.yield with affine.yield
struct SCFForRaising : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  SCFForRaising(MLIRContext *context)
      : OpRewritePattern(context){};

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override;
};

// Raise memref.load to affine.load
struct LoadRaising : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LoadRaising(MLIRContext *context)
      : OpRewritePattern(context){};

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override;
};

// Raise memref.store to affine.store
struct StoreRaising : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  StoreRaising(MLIRContext *context)
      : OpRewritePattern(context){};

  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override;
};


// Rewriting Pass
  struct SCFForRaisingPass
      : public PassWrapper<SCFForRaisingPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFForRaisingPass)
    void runOnOperation() override;
  };

  struct LoadStoreRaisingPass
      : public PassWrapper<LoadStoreRaisingPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoadStoreRaisingPass)
    void runOnOperation() override;
  };

} // end anonymous namespace

// Raise scf.for to affine.for & replace scf.yield with affine.yield
// Conversion:
// Raise scf.for to affine.for:
// scf.for step -> affine.for step
// scf.for lb -> affine.for lb
// scf.for ub -> affine.for ub
// scf.for iter_ags -> affine.for iter_args
// scf.yield -> affine.yield
// scf.for body -> affine.for body
// create a map for both lb, ub: #map1 = affine_map<()[s0] -> (s0)>
LogicalResult SCFForRaising::matchAndRewrite(ForOp forOp,
                                             PatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();
  // map scf.for operands to affine.for
  comet_debug() << "mapping scf.for to affine.for\n";
  auto step = forOp.getStep();
  auto lb = forOp.getLowerBound();
  auto ub = forOp.getUpperBound();
  // affine.for only accepts a step as a literal
  int stepNum = dyn_cast<ConstantOp>(step.getDefiningOp())
                    .getValue()
                    .cast<IntegerAttr>()
                    .getInt();
  // the loop bounds are both valid symbols - direct map #map <()[s0] -> (s0)>
  AffineMap directSymbolMap =
      AffineMap::get(0, 1, getAffineSymbolExpr(0, rewriter.getContext()));
  auto f = rewriter.create<AffineForOp>(loc, lb, directSymbolMap, ub,
                                              directSymbolMap, stepNum);
    
  rewriter.eraseBlock(f.getBody());
  Operation *loopTerminator = forOp.getBody()->getTerminator();
  ValueRange terminatorOperands = loopTerminator->getOperands();
  rewriter.setInsertionPointToEnd(forOp.getBody());
  rewriter.create<AffineYieldOp>(loc, terminatorOperands);
  rewriter.inlineRegionBefore(forOp.getRegion(), f.getRegion(), f.getRegion().end());
  rewriter.eraseOp(loopTerminator);
  rewriter.eraseOp(forOp);
 
  return success();
}

// Extract the affine expression from a number of std operations
AffineExpr getAffineExpr(Value value, PatternRewriter &rewriter,
                         std::vector<Value> *dims,
                         std::vector<Value> *symbols) {
  auto op = value.getDefiningOp();
  assert(op != NULL || isValidDim(value));
  if (isValidSymbol(value)) {
    int symbolIdx;
    auto symbolIter = std::find(symbols->begin(), symbols->end(), value);
    if (symbolIter == symbols->end()) {
      symbolIdx = symbols->size();
      symbols->push_back(value);
    } else
      symbolIdx = std::distance(symbols->begin(), symbolIter);
    return getAffineSymbolExpr(symbolIdx, rewriter.getContext());
  } else if (isValidDim(value)) {
    int dimIdx;
    auto dimIter = std::find(dims->begin(), dims->end(), value);
    if (dimIter == dims->end()) {
      dimIdx = dims->size();
      dims->push_back(value);
    } else
      dimIdx = std::distance(dims->begin(), dimIter);
    return getAffineDimExpr(dimIdx, rewriter.getContext());
  } else if (isa<AddIOp>(op))
    return getAffineExpr(op->getOperand(0), rewriter, dims, symbols) +
           getAffineExpr(op->getOperand(1), rewriter, dims, symbols);
  else if (isa<MulIOp>(op))
    return getAffineExpr(op->getOperand(0), rewriter, dims, symbols) *
           getAffineExpr(op->getOperand(1), rewriter, dims, symbols);
  else
    return NULL;
}

// Raise std.load to affine.load
// for each index of the loadOp, extract its affine expression
// construct a map with the dim and symbol count for the affine.load
// replace the loadOp with the newly constructed affine.load
LogicalResult LoadRaising::matchAndRewrite(LoadOp loadOp,
                                           PatternRewriter &rewriter) const {
  std::vector<Value> indices, dims, symbols;
  std::vector<AffineExpr> exprs;

  for (auto index : loadOp.getIndices()) {
    indices.push_back(index);
    exprs.push_back(getAffineExpr(index, rewriter, &dims, &symbols));
  }
  ArrayRef<AffineExpr> results(exprs);
  AffineMap affineMap = AffineMap::get(dims.size(), symbols.size(), results,
                                       rewriter.getContext());
  dims.insert(dims.end(), symbols.begin(), symbols.end());
  rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, loadOp.getMemRef(),
                                            affineMap, dims);
  return success();
}

// Raise std.store to affine.store
LogicalResult StoreRaising::matchAndRewrite(StoreOp storeOp,
                                            PatternRewriter &rewriter) const {

  std::vector<Value> indices, dims, symbols;
  std::vector<AffineExpr> exprs;

  for (auto index : storeOp.getIndices()) {
    indices.push_back(index);
    exprs.push_back(getAffineExpr(index, rewriter, &dims, &symbols));
  }
  ArrayRef<AffineExpr> results(exprs);
  AffineMap affineMap = AffineMap::get(dims.size(), symbols.size(), results,
                                       rewriter.getContext());
  dims.insert(dims.end(), symbols.begin(), symbols.end());
  rewriter.replaceOpWithNewOp<AffineStoreOp>(
      storeOp, storeOp.getValueToStore(), storeOp.getMemRef(), affineMap, dims);
  return success();
}


void mlir::comet::SCFForRaisingPatterns(RewritePatternSet &patterns,
                                 MLIRContext *ctx) {
  patterns.insert<SCFForRaising>(ctx);
}

void mlir::comet::LoadStoreRaisingPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.insert<LoadRaising, StoreRaising>(ctx);
}

void SCFForRaisingPass::runOnOperation() {
  
  RewritePatternSet patterns(&getContext());
  mlir::comet::SCFForRaisingPatterns(patterns, &getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<SCFDialect, AffineDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

void LoadStoreRaisingPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  mlir::comet::LoadStoreRaisingPatterns(patterns, &getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<SCFDialect, AffineDialect>();
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::comet::createRaiseSCFForPass() {
  return std::make_unique<SCFForRaisingPass>();
}

std::unique_ptr<Pass> mlir::comet::createRaiseLoadStorePass() {
  return std::make_unique<LoadStoreRaisingPass>();
}