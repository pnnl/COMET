//===- WorkspaceTransforms.cpp  ------===//
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
// This pass performs workspace transformations on index tree dialect for sparse-sparse computation
//===----------------------------------------------------------------------===//

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/Support/Debug.h"
#include <iostream>
#include <algorithm>
#include <vector>

#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <string>
#include <utility>
#include <queue>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::arith;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

using llvm::SmallVector;
using llvm::StringRef;

#define DEBUG_TYPE "loop-invariant-transformations"

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//



///  Apply workspace transformation on the lhs
///  Consider CSR first
///  ikj Cij += Aik * Bkj ===> i (j Wj = 0; kj Wj += Aik * Bkj; kj Cij=Wj)
///  ij Cij = Aij * Bij =====> i (j Wj = 0; j Wj = Aij * Bij; j Cij=Wj)

///  Apply workspace transformation on the rhs
///  Consider CSR first
///  j Wj = Aij * Bij ===> j Vj = 0; j Vj = Bij; j Wj = Aij * Vj;

//===----------------------------------------------------------------------===//
/// WorkspaceTransforms Pass
//===----------------------------------------------------------------------===//

///  Apply workspace transformations on the ta.tc and tc.elews_mul
namespace
{
  struct IndexTreeLoopInvariantCodeMotionPass
      : public PassWrapper<IndexTreeLoopInvariantCodeMotionPass, OperationPass<mlir::func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexTreeLoopInvariantCodeMotionPass)
    void runOnOperation() override;
  };
} /// end anonymous namespace.

struct MoveInvariantComputeOp : public OpRewritePattern<IndexTreeComputeOp> {
  MoveInvariantComputeOp (MLIRContext *context)
    : OpRewritePattern<IndexTreeLHSOperandOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(IndexTreeComputeOp compute_op, mlir::PatternRewriter &rewriter) const override {
    // Collect all indices used in this compute expression
    llvm::SmallDenseVector<Value> used_indices;
    IndexTreeLHSOperandOp lhs_op = op.getLhs().getDefiningOp<IndexTreeLHSOperandOp>();
    for(auto pos : lhs_op.getPos()) {
      auto index_to_tensor = pos.getDefiningOp<IndexTreeIndexToTensorOp>();
      if(!index_to_tensor)
        return failure();
      used_indices.insert(index_to_tensor.getIndex());
    }
    
    auto rhs_operands = op.getRhs();
    for(Value rhs : rhs_operands) {
      IndexTreeOperandOp operand_op = rhs.getDefiningOp<IndexTreeOperandOp>();
      for(auto pos : operand_op.getPos()) {
        auto index_to_tensor = pos.getDefiningOp<IndexTreeIndexToTensorOp>();
        if(!index_to_tensor)
          return failure();
        used_indices.insert(index_to_tensor.getIndex());
      }
    }

    // We want to find all of the indices that this compute op is nested under
    // and check if they are used in this compute expression. Every time we come
    // across an unused index, the index nodes that we have seen so far need to be copied
    // to form a new branch of the tree. We also keep track of the parent at the fork
    llvm::SmallVector<IndexTreeIndicesOp> seen_indices;
    llvm::SmallVector<IndexTreeIndicesOp> indices_to_copy;
    Value parent = compute_op.getParent();
    Value fork = parent;
    IndexTreeIndicesOp node = parent.getDefiningOp<IndexTreeIndicesOp>();
    while(node) {
      if(used_indices.find(parent) != used_indices.end()) {
        // Used index variable
        seen_indices.push_back(parent);
      } else if {
        // Unused index variable
        fork = node.getParent();
        indices_to_copy.insert(indices_to_copy.begin(). seen_indices.rbegin(), seen_indices.rend());
        seen_indices.clear();
      }

      parent = node.getParent();
      IndexTreeIndicesOp node = parent.getDefiningOp<IndexTreeIndicesOp>();
    }

    if(fork == compute_op.getParent()) {
      return failure(); // Match failed, no indces to move.
    }

    // Success!
    IRMapping map;
    auto context = rewriter.getContext();
    auto loc = op.getLoc();
    IndexNodeType index_node_type = IndexNodeType::get(context); 
    for(auto index : indices_to_copy)
    {
      Value new_index = rewriter.create<IndexTreeIndicesOp>(loc, index_node_type, parent);
      map.map(index, new_index);
    }

    for(auto pos : lhs_op.getPos()) {
      auto index_to_tensor = pos.getDefiningOp<IndexTreeIndexToTensorOp>();
      rewriter.clone(index_to_tensor, map);
    }
    rewriter.clone(lhs_op, map);
    
    auto rhs_operands = op.getRhs();
    for(Value rhs : rhs_operands) {
      IndexTreeOperandOp operand_op = rhs.getDefiningOp<IndexTreeOperandOp>();
      for(auto pos : operand_op.getPos()) {
        auto index_to_tensor = pos.getDefiningOp<IndexTreeIndexToTensorOp>();
        rewriter.clone(index_to_tensor, map);
      }
      rewriter.clone(operand_op, map);
    }
    Operation* new_compute_op = rewriter.clone(compute_op, map);
    rewriter.replaceOp(compute_op, new_compute_op);
    return success();    
  }
};

void comet::indexTree::populateLoopInvariantCodeMotionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<MoveInvariantComputeOp>(context);
}


void IndexTreeWorkspaceTransformationsPass::runOnOperation()
{
  comet_debug() << __FILE__ << " " << __LINE__ << " starting CompressedWorkspaceTransforms pass \n";
  func::FuncOp function = getOperation();
  mlir::RewritePatternSet code_motion_patterns(&getContext());

  // Add patterns to move invariant compute ops out of a loop
  indexTree::populateLoopInvariantCodeMotionPatterns(&getContext(), code_motion_patterns);

  // Add patterns to infer the domain of any new index variables
  indexTree::populateDomainInferencePatterns(&getContext(), code_motion_patterns);
  indexTree::populateDomainConcretizationPatterns(&getContext(), code_motion_patterns);

  // Apply the collected patterns
  mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(code_motion_patterns));
  comet_debug() << __FILE__ << " " << __LINE__ << " ending CompressedWorkspaceTransforms pass \n";
}

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeWorkspaceTransformationsPass()
{
  return std::make_unique<IndexTreeWorkspaceTransformationsPass>();
}