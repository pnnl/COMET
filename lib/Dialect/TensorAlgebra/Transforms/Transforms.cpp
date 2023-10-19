//===- Transforms.cpp - Tensor Algebra High Level Optimizer --------------------------===//
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
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the TA dialect.
//
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/Sequence.h"

#include <limits>
#include <map>
#include <stack>
#include <set>
#include <unordered_map>
#include <numeric>

#define DEBUG_TYPE "comet-transforms"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::bufferization;

using namespace mlir::tensorAlgebra;
using namespace mlir::indexTree;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

std::vector<Value> dim_format;

namespace
{
  /// Chain of multiplication operations produces ChainSetOp
  /// and it needs to be lowered
  struct ChainSetOpLowering : public ConversionPattern
  {
    ChainSetOpLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::ChainSetOp::getOperationName(), 1,
                            ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::ChainSetOp>(op));

      comet_debug() << "ChainSetOpLowering begin\n";
      comet_pdump(op);
      auto ctx = rewriter.getContext();
      auto loc = op->getLoc();

      auto lhs = operands[0].getDefiningOp();
      auto rhs = operands[1].getDefiningOp();

      assert(isa<tensorAlgebra::LabeledTensorOp>(lhs));
      auto lhsLT = cast<tensorAlgebra::LabeledTensorOp>(lhs);

      Operation *rhsOp;
      if (isa<tensorAlgebra::ChainMulOp>(rhs))
      {
        comet_debug() << "\n";
        rhsOp = cast<tensorAlgebra::ChainMulOp>(rhs);
      }
      else if (isa<tensorAlgebra::TensorAddOp>(rhs))
      {
        comet_debug() << "\n";
        rhsOp = cast<tensorAlgebra::TensorAddOp>(rhs);
      }
      else if (isa<tensorAlgebra::LabeledTensorOp>(rhs))
      {
        comet_debug() << "\n";
        auto rhsLT = cast<tensorAlgebra::LabeledTensorOp>(rhs);

        auto lhsLabels = lhsLT.getLabels();
        auto rhsLabels = rhsLT.getLabels();
        std::vector<Operation *> lhsLabelOps, rhsLabelOps;
        for (const auto lbl : lhsLabels)
        {
          lhsLabelOps.push_back(lbl.getDefiningOp());
        }
        for (const auto lbl : rhsLabels)
        {
          rhsLabelOps.push_back(lbl.getDefiningOp());
        }
        auto outPerm = constructPermutationMapAttr(rhsLabelOps, lhsLabelOps);
        auto inPerm = constructPermutationMapAttr(rhsLabelOps, rhsLabelOps);

        auto inPermAttr = AffineMapAttr::get(AffineMap::getPermutationMap(inPerm, ctx));
        auto outPermAttr = AffineMapAttr::get(AffineMap::getPermutationMap(outPerm, ctx));

        auto new_op = rewriter.create<tensorAlgebra::TensorCopyOp>(loc, lhsLT.getTensor(), rhsLT.getTensor(), inPermAttr, outPermAttr);
        comet_debug() << "\n";
        comet_vdump(new_op);

        double alpha = 1.0;
        auto betaAttr = op->getAttr("__beta__");
        double beta = betaAttr.cast<FloatAttr>().getValueAsDouble();

        new_op.getOperation()->setAttr("__alpha__", rewriter.getF64FloatAttr(alpha));
        new_op.getOperation()->setAttr("__beta__", rewriter.getF64FloatAttr(beta));

        rewriter.eraseOp(op);
        return success();
      }
      else
      {
        comet_debug() << "Neither MulOp, AddOp, nor LabeledTensorOp, it is: ";
        comet_pdump(rhs);
        comet_debug() << "ChainSetOpLowering end\n";
        return success();
      }

      comet_debug() << "\n";
      auto lhsTensor = lhsLT.getTensor();
      comet_debug() << "\n";
      auto labels = lhsLT.getLabels();
      comet_debug() << "\n";
      std::vector<Value> lhsLabels(labels.begin(), labels.end());

      comet_pdump(rhsOp);
      comet_debug() << "\n";
      replaceSetOp(rhsOp, lhsTensor, lhsLabels, loc, rewriter);
      rewriter.eraseOp(op);
      comet_debug() << "ChainSetOpLowering end\n";
      return success();
    }
  };

  /// TODO(gkestor): test TensorCopyLowering
  struct TensorCopyLowering : public ConversionPattern
  {
    TensorCopyLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::TensorCopyOp::getOperationName(), 1,
                            ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::TensorCopyOp>(op));

      auto loc = op->getLoc();
      auto tensorCopyOp = cast<tensorAlgebra::TensorCopyOp>(op);

      auto lhsTensorOperand = operands[0];
      auto lhsTensorLoadOp = cast<ToTensorOp>(lhsTensorOperand.getDefiningOp());
      auto lhsMemref = lhsTensorLoadOp.getMemref();

      auto rhsTensorOperand = operands[1];
      auto rhsTensorLoadOp = cast<ToTensorOp>(rhsTensorOperand.getDefiningOp());
      ;
      auto rhsMemref = rhsTensorLoadOp.getMemref();

      /// TODO(gkestor): better way to cast AffineMap to ArrayRef<int64_t>
      auto outPermMap = tensorCopyOp.getOutputPermAttr();
      std::vector<std::vector<int64_t>> allPerms = getAllPerms(dyn_cast<ArrayAttr>(outPermMap));
      auto copyOp = rewriter.create<linalg::TransposeOp>(loc, rhsMemref, lhsMemref, llvm::ArrayRef<int64_t>(allPerms[0]));

      auto alphaAttr = tensorCopyOp.getOperation()->getAttr("__alpha__");
      auto betaAttr = tensorCopyOp.getOperation()->getAttr("__beta__");

      copyOp.getOperation()->setAttr("__alpha__", alphaAttr);
      copyOp.getOperation()->setAttr("__beta__", betaAttr);

      rewriter.eraseOp(op);

      return success();
    }
  };

  //===----------------------------------------------------------------------===//
  /// STCRemoveDeadOps RewritePatterns: SparseTensor Constant operations
  //===----------------------------------------------------------------------===//

  struct RemoveDeadOpLowering : public OpRewritePattern<tensorAlgebra::TensorMultOp>
  {
    using OpRewritePattern<tensorAlgebra::TensorMultOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tensorAlgebra::TensorMultOp op,
                                  PatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::TensorMultOp>(op));
      comet_debug() << " erase TensorMultOp \n";
      comet_debug() << "--------------TensorContractionLowering in format\n";
      /// Here, should check the operands, at least one operand should be sparse;
      /// Otherwise, if all dense operands, just return.
      return success();
    }
  }; /// TensorContractionLowering

  template <typename TAOp>
  struct RemoveDeadTAOpLowering : public ConversionPattern
  {
    RemoveDeadTAOpLowering(MLIRContext *ctx)
        : ConversionPattern(TAOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      comet_debug() << " erase op \n";
      rewriter.eraseOp(op);

      return success();
    }
  };

}

/// =============================================================================
//
/// These patterns lowers tensor multiplication chains into s series of ta.tc operations.
//
//===----------------------------------------------------------------------===//
void mlir::tensorAlgebra::populateLowerTAMulChainPatterns(
    RewritePatternSet &patterns, MLIRContext *context)
{
  patterns.insert<ChainSetOpLowering>(context);
}

void mlir::tensorAlgebra::populateSTCRemoveDeadOpsPatterns(
    RewritePatternSet &patterns, MLIRContext *context)
{
  patterns.insert<RemoveDeadTAOpLowering<tensorAlgebra::ChainMulOp>>(context);
  patterns.insert<RemoveDeadTAOpLowering<tensorAlgebra::IndexLabelDynamicOp>>(context);
  patterns.insert<RemoveDeadTAOpLowering<tensorAlgebra::IndexLabelStaticOp>>(context);
}