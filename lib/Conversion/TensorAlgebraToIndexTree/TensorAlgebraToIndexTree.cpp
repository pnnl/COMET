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

#include <algorithm>
#include <cassert>
#include <unordered_set>

#include "comet/Conversion/TensorAlgebraToIndexTree/TensorAlgebraToIndexTree.h"

#include "comet/Dialect/IndexTree/Transforms/Tensor.h"
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
// #define COMET_DEBUG_MODE
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
    LowerTensorAlgebraToIndexTreePass(TargetDevice device) : device(device){};
    void runOnOperation() override;
  
    TargetDevice device;
  };

} /// namespace

/**
 * @brief Check if the given allPerms is from one of the chosen operations.
 * Current algorithm is to check the pattern of the allPerms to see if it is tensor contraction.
 * Current chosen operations includes:
 * dense matrix-matrix multiplication (MM),
 * dense matrix-vector multiplication (MV),
 * sparse matrix-dense matrix multiplication (SpMM),
 * sparse matrix-dense vector multiplication (SpMV)
 *
 * @param allPerms allPerms from the operation. For example, [[d0, d1], [d1, d2], [d0, d2]]
 * @return true : it is one of the chosen operations.
 * @return false : it is not.
 */
bool check_chosen_operations(const std::vector<std::vector<int64_t>> &allPerms,
                             const std::vector<std::vector<std::string>> &allFormats)
{
  if (allPerms.size() != 3)
  {
    return false;
  }

  /// The output tensor should be dense.
  auto lhs_formats = allFormats[2];
  for (auto &f : lhs_formats)
  {
    if (f != "D")
    {
      return false;
    }
  }

  /// do lhs = op(rhs1, rhs2)
  const std::vector<int64_t> &rhs1_perms = allPerms[0];
  const std::vector<int64_t> &rhs2_perms = allPerms[1];
  const std::vector<int64_t> &lhs_perms = allPerms[2];

  if (rhs1_perms.size() == 2)
  {
    if (rhs2_perms.size() == 2 && lhs_perms.size() == 2)
    {
      /// If it is op(matrix, matrix)
      if (rhs1_perms[0] == lhs_perms[0] &&
          rhs1_perms[1] == rhs2_perms[0] &&
          rhs2_perms[1] == lhs_perms[1])
      {
        /// Then op is MM or SpMM
        return true;
      }
    }
    else if (rhs2_perms.size() == 1 && lhs_perms.size() == 1)
    {
      /// If it is op(matrix, vector)
      if (rhs1_perms[0] == lhs_perms[0] &&
          rhs1_perms[1] == rhs2_perms[0])
      {
        /// Then op is MV or SpMV
        return true;
      }
    }
  }

  return false;
}

Value getRealLhs(Operation *op)
{
  assert(isa<TensorMultOp>(op) || isa<TensorElewsMultOp>(op) || isa<TensorAddOp>(op) || isa<TensorSubtractOp>(op));
  Operation *firstUser = nullptr;
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

void buildDefUseInfo(UnitExpression *e)
{
  auto lhs = e->getLHS();
  lhs->setDefiningExpr(e);
  for (auto operand : e->getOperands())
  {
    if (auto def = operand->getDefiningExpr())
    {
      def->addUser(e);
    }
  }
}

IndicesType getUnion(IndicesType indices1, IndicesType indices2)
{
  sort(indices1.begin(), indices1.end());
  sort(indices2.begin(), indices2.end());

  IndicesType allIndices(indices1.size() * 4);

  IndicesType::iterator it = set_union(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(), allIndices.begin());
  allIndices.resize(it - allIndices.begin());
  return allIndices;
}

IndicesType gpuIndices(IndicesType indices1, IndicesType indices2)
{
  sort(indices1.begin(), indices1.end());
  sort(indices2.begin(), indices2.end());

  IndicesType interIndices;
  IndicesType unIndices;
  IndicesType difIndices;
  IndicesType allIndices;

  std::set_intersection(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(),  std::back_inserter(interIndices));
  set_union(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(), std::back_inserter(unIndices));
  std::set_difference(unIndices.begin(), unIndices.end(), interIndices.begin(), interIndices.end(), std::back_inserter(difIndices));
  allIndices = difIndices;
  allIndices.insert(allIndices.end(), interIndices.begin(), interIndices.end());

  // allIndices.resize(it - allIndices.begin());
  return allIndices;  
}

void doTensorMultOp(TensorMultOp op, unique_ptr<Index_Tree> &tree, TargetDevice device = CPU)
{
  Value rhs1_tensor = getRealRhs(op.getRhs1().getDefiningOp());
  Value rhs2_tensor = getRealRhs(op.getRhs2().getDefiningOp());
  Value lhs_tensor = getRealLhs(op);
  Value mask_tensor = op.getMask();
  // rhs1_tensor.getDefiningOp
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
    // auto MaskingTypeAttr = mult_op.getMaskTypeAttr();

    auto tensor_type = op->getResultTypes()[0];
    auto itree_op = rewriter.create<IndexTreeOp>(loc, tensor_type);
    rewriter.createBlock(&itree_op.getRegion());

    indexTree::IndexTreeType tree_type = indexTree::IndexTreeType::get(context);
    Value parent = rewriter.create<indexTree::IndexTreeRootOp>(loc, tree_type);

    //Construct each index variable
    auto lhsMap = indexing_maps[0].cast<AffineMapAttr>().getValue();
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
    Value lhs_operand;
    if(mask_tensor == nullptr)
    {
      lhs_operand = rewriter.create<indexTree::IndexTreeOperandOp>(loc, operand_type, lhs_tensor, indices);
    }
    else
    {
      lhs_operand = rewriter.create<indexTree::IndexTreeMaskedOperandOp>(loc, operand_type, lhs_tensor, mask_tensor, indices);
    }

    //Construct RHS operands
    std::vector<Value> rhs_operands;
    indices.clear();
    auto affineMap = indexing_maps[1].cast<AffineMapAttr>().getValue();
    for (size_t i = 0; i < affineMap.getNumResults(); i++)
    {
      auto expr = affineMap.getResult(i);
      indices.push_back(index_nodes[expr.cast<AffineDimExpr>().getPosition()]);
    }
    rhs_operands.push_back(rewriter.create<indexTree::IndexTreeOperandOp>(loc, operand_type, rhs1_tensor, indices));
    
    indices.clear();
    affineMap = indexing_maps[2].cast<AffineMapAttr>().getValue();
    for (size_t i = 0; i < affineMap.getNumResults(); i++)
    {
      auto expr = affineMap.getResult(i);
      indices.push_back(index_nodes[expr.cast<AffineDimExpr>().getPosition()]);
    }
    rhs_operands.push_back(rewriter.create<indexTree::IndexTreeOperandOp>(loc, operand_type, rhs2_tensor, indices));

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
std::unique_ptr<Pass> mlir::comet::createLowerTensorAlgebraToIndexTreePass(TargetDevice device)
{
  comet_debug() << " Calling createLowerTensorAlgebraToIndexTreePass\n";
  return std::make_unique<LowerTensorAlgebraToIndexTreePass>(device);
}
