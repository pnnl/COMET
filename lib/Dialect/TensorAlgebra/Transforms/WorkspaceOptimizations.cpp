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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/IndexedMap.h"

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Patterns.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"

using namespace mlir;
using namespace mlir::tensorAlgebra;

namespace mlir {
  namespace comet{
    #define GEN_PASS_DEF_TENSORALGEBRAWORKSPACEOPTIMIZATIONS
    #include "comet/Dialect/TensorAlgebra/Passes.h.inc"
  }
}

struct WorkspaceMergeExtract : public OpRewritePattern<TensorInsertOp> {
  WorkspaceMergeExtract(MLIRContext *context)
      : OpRewritePattern<TensorInsertOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(TensorInsertOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Value workspace = op.getTensor();
    WorkspaceType workspace_type = llvm::dyn_cast<WorkspaceType>(workspace.getType());
    if(!workspace_type) {
      return failure();
    }

    Value to_insert = op.getValue();

    //TODO: Right now we only support merging add operations
    auto accumulate_op = to_insert.getDefiningOp<arith::AddFOp>();
    if(!accumulate_op || !accumulate_op->hasOneUse()) {
      return failure();
    }

    auto extract_op = accumulate_op.getOperand(0).getDefiningOp<TensorExtractOp>();
    if(!extract_op || !extract_op->hasOneUse()) {
      return failure();
    }

    if(extract_op.getTensor() != op.getTensor()){
      return failure();
    }

    rewriter.replaceOpWithNewOp<WorkspaceAccumulateOp>(op, workspace_type, workspace, extract_op.getPos(), extract_op.getCrds(), accumulate_op.getOperand(1));
    rewriter.eraseOp(accumulate_op);
    rewriter.eraseOp(extract_op);
    return success();
  }
};

struct WorkspaceExtractNoCheck : public OpRewritePattern<TensorExtractOp> {
  WorkspaceExtractNoCheck(MLIRContext *context)
      : OpRewritePattern<TensorExtractOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(TensorExtractOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Value workspace = op.getTensor();
    WorkspaceType workspace_type = llvm::dyn_cast<WorkspaceType>(workspace.getType());
    if(!workspace_type) {
      return failure();
    }

    Value crd = op.getCrds()[0]; // TODO: Deal with greater than 1 dimensional workspace

    Operation* src = crd.getDefiningOp();
    // Deal with casting
    if(llvm::isa<arith::IndexCastOp>(src)){
      src = src->getOperand(0).getDefiningOp();
    }

    SpTensorGetCrd getCrdOp = llvm::dyn_cast<SpTensorGetCrd>(src);
    if(!getCrdOp) {
      return failure();
    }

    if(getCrdOp.getTensor() != workspace){
      return failure();
    }

    rewriter.replaceOpWithNewOp<WorkspaceReadOp>(op, workspace_type.getElementType(), workspace, getCrdOp.getIdx(), crd);
    return success();
  }
};


void tensorAlgebra::populateWorkspaceOptimizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<WorkspaceMergeExtract, WorkspaceExtractNoCheck>(context);
}

struct TensorAlgebraWorkspaceOptimizations : comet::impl::TensorAlgebraWorkspaceOptimizationsBase<TensorAlgebraWorkspaceOptimizations> {
  using TensorAlgebraWorkspaceOptimizationsBase::TensorAlgebraWorkspaceOptimizationsBase;

  void runOnOperation() override {
    mlir::RewritePatternSet workspace_optimization_patterns(&getContext());
    tensorAlgebra::populateWorkspaceOptimizationPatterns(&getContext(), workspace_optimization_patterns);

    if(failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(workspace_optimization_patterns)))) {
      return signalPassFailure();
    }
  }
};

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createWorkspaceOptimizationsPass()
{
  return std::make_unique<TensorAlgebraWorkspaceOptimizations>();
}