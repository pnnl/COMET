#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringSet.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/Passes.h"

using namespace mlir;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

namespace mlir {
    namespace comet{
    #define GEN_PASS_DEF_INDEXTREEDOMAINCONCRETIZATION
    #include "comet/Dialect/IndexTree/Passes.h.inc"
    }
}

struct ConcretizeTensorDomain :  public OpRewritePattern<IndexTreeTensorDomainOp> {
  ConcretizeTensorDomain(MLIRContext *context)
      : OpRewritePattern<IndexTreeTensorDomainOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult 
  liftAccessOp(mlir::Operation *dependent_op, 
               IndexTreeIndexToTensorOp access_op) const {
    if(access_op->isBeforeInBlock(dependent_op))
      return success();

    Value prev_access_value;
    if((prev_access_value = access_op.getPrevDim()))
    {
      if(mlir::failed(
            liftAccessOp(
              dependent_op,
              llvm::cast<IndexTreeIndexToTensorOp>(prev_access_value.getDefiningOp())
            )
          ))
        return failure();
    }
    access_op->moveBefore(dependent_op);
    return success();
  }

  mlir::LogicalResult
  matchAndRewrite(IndexTreeTensorDomainOp domain_op, 
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = domain_op->getLoc();
    auto context = rewriter.getContext();
    indexTree::DomainType domain_type = indexTree::DomainType::get(context);
    uint32_t dim = domain_op.getDim();
    Value new_domain;

    Value tensor = domain_op.getTensor();
    SparseTensorConstructOp construct_op = tensor.getDefiningOp<SparseTensorConstructOp>();
    if(construct_op)
    {
      //Domain comes from a sparse tensor (may still be dense)
      int32_t rank = construct_op.getTensorRank();
      TensorFormatEnum format = construct_op.getDimensionFormats()[2 * dim].cast<TensorFormatEnumAttr>().getValue();

      if(format == TensorFormatEnum::D)
      {
        Value max = construct_op.getOperand((8*rank) + 2 + dim); //TODO: Fix magic numbers
        new_domain = rewriter.create<IndexTreeDenseDomainOp>(loc, domain_type, max);
      } else
      {
        Value pos = construct_op.getOperand(4 * dim);
        Value crd = construct_op.getOperand((4 * dim) + 1);
        Value pos_size = construct_op.getOperand((4 * rank) + (4 * dim) + 1);
        Value crd_size = construct_op.getOperand((4 * rank) + (4 * dim) + 2);
        Value parent = domain_op.getParent();
        if(!parent)
        {
          // Get associated index
          IndexTreeIndicesOp index_op; 
          Operation* use = *(domain_op->user_begin());
          // TODO: Fix danger of infinite loop!!!
          while(!(index_op = llvm::dyn_cast<indexTree::IndexTreeIndicesOp>(use)))
          {
            use = *(use->user_begin());
          }
          assert(index_op);

          if(dim == 0)
          {
            parent = nullptr;
          } else
          {
            // Infer parent index variable
            for(Operation* use : index_op->getUsers())
            {
              IndexTreeIndexToTensorOp access_op = llvm::dyn_cast<indexTree::IndexTreeIndexToTensorOp>(use);
              if(!access_op || access_op.getTensor() != tensor || access_op.getDim() != dim)
                continue;

              parent = access_op.getPrevDim();
              IndexTreeIndexToTensorOp prev_access_op = 
                llvm::cast<indexTree::IndexTreeIndexToTensorOp>(parent.getDefiningOp());
              if(mlir::failed(this->liftAccessOp(domain_op, prev_access_op)))
                return failure();

              break;
            }
          }
        }
        new_domain = rewriter.create<IndexTreeSparseDomainOp>(
          loc, domain_type, tensor, domain_op.getDimAttr(), 
          TensorFormatEnumAttr::get(context, format), 
          pos, crd, pos_size, crd_size, parent);
      }
    }
    else
    {
      //Domain is dense
      //TODO (alokvk2): Figure out if we need to take the root index variable or the allocation
      //Right now I don't know how to get back to the root index variable.
      Operation* toTensor = tensor.getDefiningOp();
      memref::AllocOp alloc = toTensor->getOperand(0).getDefiningOp<memref::AllocOp>();
      Value max = alloc.getOperand(0);
      new_domain = rewriter.create<IndexTreeDenseDomainOp>(loc, domain_type, max);
    }
    rewriter.replaceOp(domain_op, new_domain);
    return success();
  }
};


struct SimplifyIntersectionOp : public OpRewritePattern<IndexTreeDomainIntersectionOp> {
  SimplifyIntersectionOp(MLIRContext *context)
      : OpRewritePattern<IndexTreeDomainIntersectionOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(IndexTreeDomainIntersectionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    Value first_domain = op->getOperand(0);
    SmallVector<Value, 2> domains;
    SmallVector<Operation*> to_remove;
    Operation* operand_op;
    for(auto operand : op->getOperands())
    {
      if((operand_op = operand.getDefiningOp<IndexTreeDenseDomainOp>()))
        to_remove.push_back(operand_op);
      else
        domains.push_back(operand);
    }

    if(domains.size() == 0) // All domains are dense
      domains.push_back(first_domain); // Preserve first one

    if(domains.size() == 1)
    {
      // Remove intersection op completely
      rewriter.replaceOp(op, {domains[0]});
    } else if (domains.size() < op->getNumOperands())
    {
      // Keep only non-dense operands
      auto loc = op->getLoc();
      auto context = rewriter.getContext();
      indexTree::DomainType domain_type = indexTree::DomainType::get(context);
      Value new_domain = rewriter.create<IndexTreeDomainIntersectionOp>(loc, domain_type, domains);
      rewriter.replaceOp(op, {new_domain});
    } else
    {
      // The intersection can't be simplified
      return failure();
    }

    // Delete newly unused values
    for(Operation* unused_op : to_remove)
    {
      if(unused_op->use_empty())
        rewriter.eraseOp(unused_op);
    }

    return success();
  }
};

struct SimplifyUnionOp : public mlir::OpRewritePattern<IndexTreeDomainUnionOp> {
  SimplifyUnionOp(mlir::MLIRContext *context)
      : OpRewritePattern<IndexTreeDomainUnionOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(IndexTreeDomainUnionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool success = false;
    Operation* operand_op;
    for(auto operand : op->getOperands())
    {
      if((operand_op = operand.getDefiningOp<IndexTreeDenseDomainOp>())){
        rewriter.replaceOp(op, {operand});
        success = true;
        break;
      }
    }

    if(!success)
      return failure();

    
    for(auto operand : op->getOperands())
    {
      operand_op = operand.getDefiningOp();
      if(operand_op->use_empty())
        rewriter.eraseOp(operand_op);
    }
    return failure();
  }
};

struct IndexTreeDomainConcretization : comet::impl::IndexTreeDomainConcretizationBase<IndexTreeDomainConcretization> {
  using IndexTreeDomainConcretizationBase::IndexTreeDomainConcretizationBase;

  void runOnOperation() {
    mlir::RewritePatternSet tensor_domain_patterns(&getContext());
    tensor_domain_patterns.add<ConcretizeTensorDomain>(&getContext());
    mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(tensor_domain_patterns));

    mlir::RewritePatternSet simplify_domain_patterns(&getContext());
    simplify_domain_patterns.add<SimplifyIntersectionOp>(&getContext());
    mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(simplify_domain_patterns));
  }
};

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeDomainConcretizationPass()
{
  return std::make_unique<IndexTreeDomainConcretization>();
}