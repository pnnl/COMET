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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ScopedPrinter.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/IndexTree/Patterns.h"

#define DEBUG_TYPE "mask-domain"

using namespace mlir;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

struct MoveInvariantMaskOp : public mlir::OpRewritePattern<IndexTreeFillMaskOp> {
  MoveInvariantMaskOp(mlir::MLIRContext *context)
      : OpRewritePattern<IndexTreeFillMaskOp>(context, /*benefit=*/1) {}

    mlir::LogicalResult 
    liftAccessOp(mlir::Operation *dependent_op, 
                 IndexTreeIndexToTensorOp access_op,
                 mlir::PatternRewriter &rewriter ) const {
      if(access_op->isBeforeInBlock(dependent_op))
        return success();
  
      Value prev_access_value;
      if((prev_access_value = access_op.getPrevDim()))
      {
        if(mlir::failed(
              liftAccessOp(
                dependent_op,
                llvm::cast<IndexTreeIndexToTensorOp>(prev_access_value.getDefiningOp()),
                rewriter
              )
            ))
          return failure();
      }
      rewriter.modifyOpInPlace(access_op, [&]() {access_op->moveBefore(dependent_op);});
      return success();
    }

  mlir::LogicalResult
  matchAndRewrite(IndexTreeFillMaskOp op,
                  mlir::PatternRewriter &rewriter) const override 
  {
    if(!llvm::isa<ConcreteDomain>(op.getDomain().getDefiningOp())) {
      return failure();
    }

    llvm::ScopedPrinter logger{llvm::dbgs()};
    LLVM_DEBUG({
      Operation* domain_op = op.getDomain().getDefiningOp();
      logger.getOStream() << "\n";
      logger.startLine() << "Looking at : '" << domain_op->getName() << "'("
                         << op << ") {\n";
      logger.indent();

      // If the operation has no regions, just print it here.
      logger.startLine() << "Concrete domain : '" << domain_op->getName() << "'("<< domain_op << ")\n";

    });

    // We want to find all the indices necessary to compute the mask
    llvm::SmallDenseSet<Value> used_indices;
    llvm::SmallVector<Value> work_list;
    work_list.push_back(op.getDomain());
    while(!work_list.empty()){
      Value domain = work_list.back();
      work_list.pop_back();

      llvm::TypeSwitch<Operation*, void>(domain.getDefiningOp())
      .Case<IndexTreeDenseDomainOp>([&](IndexTreeDenseDomainOp op) {
        return;
      })
      .Case<IndexTreeSparseDomainOp>([&](IndexTreeSparseDomainOp op) {
        auto access = op.getParent().getDefiningOp<IndexTreeIndexToTensorOp>();
        used_indices.insert(access.getIndex());
        return;
      })
      .Case<IndexTreeWorkspaceDomainOp>([&](IndexTreeWorkspaceDomainOp op) {
        return;
      })
      .Case<IndexTreeDomainIntersectionOp>([&](IndexTreeDomainIntersectionOp op) {
        work_list.append(op.getDomains().begin(), op.getDomains().end());
        return;
      })
      .Case<IndexTreeDomainUnionOp>([&](IndexTreeDomainUnionOp op) {
        work_list.append(op.getDomains().begin(), op.getDomains().end());
        return;
      })
      .Default([](Operation *op) {
        assert(false && "IndexNode not given a valid domain");
        return nullptr;
      });
    }

    LLVM_DEBUG({
      if(used_indices.size() == 0) {
        op->getParentOp()->print(logger.startLine());
      }

    });

    // We then move the fill mask op into the highest loop possible
    Value parent = op.getParent();
    while(parent != nullptr) {
      if(used_indices.contains(parent)){
        break;
      }
      parent = parent.getDefiningOp<IndexTreeIndicesOp>().getParent();
    }
    assert(parent != nullptr && "Could not find proper nesting for creating mask.");
    if(parent == op.getParent()) {
      return failure();
    }

    // Match success!
    // We then find the zero op and move it to the same nesting so that the logic remains correct.
    Value previous_parent = op.getParent();
    LLVM_DEBUG({
        op->getParentOp()->print(logger.startLine());
        logger.startLine() << "\n";
    });
    
    rewriter.modifyOpInPlace(op, [&](){
      Value cur_parent = op.getParent();
      while(cur_parent != parent) {
        op->moveBefore(cur_parent.getDefiningOp());
        cur_parent = llvm::cast<IndexTreeIndicesOp>(cur_parent.getDefiningOp()).getParent();
      }
      op.getParentMutable().assign(parent);
    });
    if(!op.getDomain().getDefiningOp()->isBeforeInBlock(op)){
      Operation* domain = op.getDomain().getDefiningOp();
      rewriter.modifyOpInPlace(domain, [&](){domain->moveBefore(op);});
      IndexTreeSparseDomainOp sparse_domain;
      if((sparse_domain = llvm::dyn_cast<IndexTreeSparseDomainOp>(domain))){
        liftAccessOp(sparse_domain, llvm::cast<IndexTreeIndexToTensorOp>(sparse_domain.getParent().getDefiningOp()), rewriter);
      }
    }

    IndexTreeZeroMaskOp zero_op = nullptr;
    for(auto user : previous_parent.getUsers()) {
      if((zero_op = llvm::dyn_cast<IndexTreeZeroMaskOp>(user)) && zero_op.getInit() == op.getResult()) {
        rewriter.modifyOpInPlace(zero_op, [&](){zero_op.getParentMutable().assign(parent);});
      }
    }
    return success();
  }
};

struct CreateFillMaskOp : public mlir::OpRewritePattern<IndexTreeMaskedDomainOp> {
  CreateFillMaskOp(mlir::MLIRContext *context)
      : OpRewritePattern<IndexTreeMaskedDomainOp>(context, /*benefit=*/0) {}

  mlir::LogicalResult
  matchAndRewrite(IndexTreeMaskedDomainOp op,
                  mlir::PatternRewriter &rewriter) const override 
  {
    if(!llvm::isa<DomainType>(op.getMask().getType())) {
      return failure();
    }

    if(!llvm::isa<ConcreteDomain>(op.getBase().getDefiningOp())) {
      return failure();
    }

    IndexTreeOp tree_op = op->getParentOfType<IndexTreeOp>();
    if(!tree_op) {
      return failure();
    }

    auto mask_domain = op.getMask().getDefiningOp<ConcreteDomain>();
    if(!mask_domain) {
      return failure();
    }

    auto loc = op.getLoc();
    
    // Create bit tensor outside of index tree
    auto cur = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(tree_op);
    auto bit_tensor_type = RankedTensorType::get({ShapedType::kDynamic}, rewriter.getI1Type());
    Value f = rewriter.create<arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
    Value init_bit_tensor = rewriter.create<tensor::SplatOp>(loc, bit_tensor_type, f, mask_domain.getDimensionSize());
    SmallVector<Value> tree_temps(tree_op.getIntermediates());
    SmallVector<Type> tree_types(tree_op->getResultTypes());
    tree_temps.push_back(init_bit_tensor);
    tree_types.push_back(bit_tensor_type);
    auto new_op = rewriter.create<IndexTreeOp>(loc, tree_types, tree_op.getInputs(), tree_temps);
    rewriter.inlineRegionBefore(tree_op.getRegion(), new_op.getRegion(), new_op.getRegion().end());

    rewriter.restoreInsertionPoint(cur);
    rewriter.modifyOpInPlace(new_op, [&](){new_op.getBody()->addArgument(bit_tensor_type, loc);});
    init_bit_tensor = new_op.getBody()->getArgument(new_op.getBody()->getNumArguments() - 1);

    Value domain = op.getMask();
    Value mask_tensor = rewriter.create<indexTree::IndexTreeFillMaskOp>(loc, bit_tensor_type, op.getParentNode(), domain, init_bit_tensor);
    rewriter.modifyOpInPlace(op, [&](){op.getMaskMutable().assign(mask_tensor);});
    indexTree::YieldOp yield = llvm::cast<indexTree::YieldOp>(new_op.getBody()->getTerminator());
    rewriter.setInsertionPoint(yield);
    mask_tensor = rewriter.create<indexTree::IndexTreeZeroMaskOp>(loc, bit_tensor_type, op.getParentNode(), domain, mask_tensor);
    rewriter.modifyOpInPlace(new_op, [&](){yield.getResultsMutable().append(ValueRange(mask_tensor));});

    for(unsigned i = 0; i < tree_op.getNumResults(); i++){
      rewriter.replaceAllUsesWith(tree_op.getResult(i), new_op.getResult(i));
    }
    rewriter.eraseOp(tree_op);

    return success();
  }
};

void indexTree::populateMaskDomainTransformationPatterns(MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<CreateFillMaskOp, MoveInvariantMaskOp>(context);
}