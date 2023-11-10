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

#include <cassert>

#include "comet/Conversion/TensorAlgebraToIndexTree/TensorAlgebraToIndexTree.h"

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/Transforms/UnitExpression.h"
#include "comet/Dialect/IndexTree/IR/IndexTree.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

using namespace mlir;

namespace
{
  struct LowerTensorAlgebraToIndexTreePass
      : public PassWrapper<LowerTensorAlgebraToIndexTreePass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorAlgebraToIndexTreePass)
    void runOnOperation() override;
  };
} /// namespace

Value getRealLhs(Operation *op)
{
  assert(isa<TensorMultOp>(op) || isa<TensorElewsMultOp>(op) || isa<TensorAddOp>(op) || isa<TensorSubtractOp>(op));
  Operation *firstUser;
  for (auto user : op->getResult(0).getUsers())
  {
    firstUser = user;
    break;
  }

  comet_pdump(firstUser);
  assert(isa<TensorSetOp>(firstUser));
  TensorSetOp setOp = cast<TensorSetOp>(firstUser);
  return setOp.getOperand(1);
}

Value getRealRhs(Operation *op)
{
  /// this will return set_op for transpose, but messes up getUsers() or subsequent calls to it.
  Operation *firstUser = op->getNextNode();
  comet_pdump(firstUser);
  /// TODO(gkestor): need to find out why user set_op is not showing up in users of TransposeOp
  ///       from the for loop below. once resolved, remove getNextNode().
  /// Operation *firstUser;
  /// for (auto user : op->getResult(0).getUsers())
  //{
  ///  firstUser = user;
  ///  break;
  //}

  if (isa<tensorAlgebra::TransposeOp>(op))
  {
    if (isa<TensorSetOp>(firstUser))
    {
      TensorSetOp setOp = cast<TensorSetOp>(firstUser);
      return setOp.getOperand(1);
    }
    else
    {
      llvm::errs() << "ERROR: Transpose has no set_op after it!\n";
    }
  }
  else
  {
    /// do nothing
    return op->getResult(0);
  }
  return op->getResult(0);
}

struct TensorMultOpLowering : public mlir::ConversionPattern {
  TensorMultOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TensorMultOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto context = rewriter.getContext();
    TensorMultOp mult_op = llvm::dyn_cast<TensorMultOp>(op);

    Value rhs1_tensor = getRealRhs(mult_op.getRhs1().getDefiningOp());
    Value rhs2_tensor = getRealRhs(mult_op.getRhs2().getDefiningOp());
    Value lhs_tensor = getRealLhs(op);
    Value mask_tensor = mult_op.getMask();

    auto indexing_maps = mult_op.getIndexingMaps();
    // auto allFormats = getAllFormats(op.getFormatsAttr(), allPerms);
    auto semiring = mult_op.getSemiringAttr().cast<mlir::StringAttr>().getValue();
    auto MaskingTypeAttr = mult_op.getMaskTypeAttr();

    auto tensor_type = op->getResultTypes()[0];
    auto itree_op = rewriter.create<IndexTreeOp>(loc, tensor_type, SmallVector<Value, 3>{lhs_tensor, rhs1_tensor, rhs2_tensor});
    Region* body = &itree_op.getRegion();
    loc = body->getLoc();
    Block* block = rewriter.createBlock(body, {}, {tensor_type, tensor_type, tensor_type},  SmallVector<Location, 3>(3, loc));

    indexTree::IndexTreeType tree_type = indexTree::IndexTreeType::get(context);
    Value parent = rewriter.create<indexTree::IndexTreeRootOp>(loc, tree_type);

    //Construct each index variable
    auto lhsMap = indexing_maps[2].cast<AffineMapAttr>().getValue();
    indexTree::IndexNodeType index_node_type = indexTree::IndexNodeType::get(context); 
    std::vector<Value> index_nodes;
    for (unsigned i = 0; i < lhsMap.getNumDims(); i++)
    {
      parent = rewriter.create<indexTree::IndexTreeIndicesOp>(loc, index_node_type, parent);
      index_nodes.push_back(parent);
    }

    //Construct LHS Operand
    indexTree::OperandType operand_type = indexTree::OperandType::get(context);
    std::vector<Value> indices;
    for (size_t i = 0; i < lhsMap.getNumResults(); i++)
    {
      auto expr = lhsMap.getResult(i);
      indices.push_back(index_nodes[expr.cast<AffineDimExpr>().getPosition()]);
    }

    Value lhs_val = block->getArgument(0);
    if(mask_tensor != nullptr)
    {
      lhs_val = rewriter.create<indexTree::IndexTreeMaskOp>(loc, tensor_type, block->getArgument(0), mask_tensor, MaskingTypeAttr);
    }
    Value lhs_operand;
    lhs_operand = rewriter.create<indexTree::IndexTreeOperandOp>(loc, operand_type, lhs_val, indices);

    //Construct RHS operands
    std::vector<Value> rhs_operands;
    indices.clear();
    auto affineMap = indexing_maps[0].cast<AffineMapAttr>().getValue();
    for (size_t i = 0; i < affineMap.getNumResults(); i++)
    {
      auto expr = affineMap.getResult(i);
      indices.push_back(index_nodes[expr.cast<AffineDimExpr>().getPosition()]);
    }
    rhs_operands.push_back(rewriter.create<indexTree::IndexTreeOperandOp>(
                           loc, operand_type, block->getArgument(1), indices));
    
    indices.clear();
    affineMap = indexing_maps[1].cast<AffineMapAttr>().getValue();
    for (size_t i = 0; i < affineMap.getNumResults(); i++)
    {
      auto expr = affineMap.getResult(i);
      indices.push_back(index_nodes[expr.cast<AffineDimExpr>().getPosition()]);
    }
    rhs_operands.push_back(rewriter.create<indexTree::IndexTreeOperandOp>(
                           loc, operand_type, block->getArgument(2), indices));

    Value compute_op = rewriter.create<indexTree::IndexTreeComputeOp>(
        loc,
        tensor_type,
        parent,
        lhs_operand,
        rhs_operands,
        rewriter.getBoolAttr(false),
        rewriter.getStringAttr(semiring)
    );

    rewriter.create<indexTree::YieldOp>(loc, TypeRange(), compute_op);
    rewriter.replaceOp(op, itree_op.getResult(0));
    return success();
  }
};

void LowerTensorAlgebraToIndexTreePass::runOnOperation()
{
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<indexTree::IndexTreeDialect>();
  target.addIllegalOp<tensorAlgebra::TensorMultOp, tensorAlgebra::TensorElewsMultOp,
                      tensorAlgebra::TensorAddOp, tensorAlgebra::TensorSubtractOp>();
  
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<TensorMultOpLowering>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// create all the passes.
std::unique_ptr<Pass> mlir::comet::createLowerTensorAlgebraToIndexTreePass()
{
  comet_debug() << " Calling createLowerTensorAlgebraToIndexTreePass\n";
  return std::make_unique<LowerTensorAlgebraToIndexTreePass>();
}
