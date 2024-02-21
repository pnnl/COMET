#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/IndexedMap.h"

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
        new_domain = rewriter.create<IndexTreeDenseDomainOp>(loc, domain_type, max, tensor, rewriter.getI32ArrayAttr({dim}));
      } else
      {
        Value pos = construct_op.getOperand(4 * dim);
        Value crd = construct_op.getOperand((4 * dim) + 1);
        Value pos_size = construct_op.getOperand((4 * rank) + (4 * dim) + 1);
        Value crd_size = construct_op.getOperand((4 * rank) + (4 * dim) + 2);
        Value dim_size = construct_op.getOperand((8*rank) + 2 + dim);
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
          pos, crd, pos_size, crd_size, dim_size, parent);
      }
    } else if(llvm::isa<tensorAlgerba::WorkspaceTensor>(tensor.getType())) {
      auto index_type = rewriter.getIndexType()
      Value dim_size = rewriter.create<tensorAlgebra::WorkspaceGetDimSize>(loc, index_type, tensorm rewriter.getI32Attr(dim));
      
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

      new_domain = rewriter.create<indexTree::IndexTreeWorkspaceDomainOp>(
        loc,
        domain_type,
        tensor,
        max,
        rewriter.getI32Attr(dim),
        parent,
      );
    } else {
      //Domain is dense
      //TODO (alokvk2): Figure out if we need to take the root index variable or the allocation
      //Right now I don't know how to get back to the root index variable.
      auto tensor_type = llvm::cast<TensorType>(tensor.getType());
      auto max = tensor_type.getShape()[dim];
      Value max_val;
      if(max < 0) {
        Operation* toTensor = tensor.getDefiningOp();
        memref::AllocOp alloc = toTensor->getOperand(0).getDefiningOp<memref::AllocOp>();
        max_val = alloc.getOperand(0);
      } else {
        max_val = rewriter.create<index::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(max));
      }
      new_domain = rewriter.create<IndexTreeDenseDomainOp>(loc, domain_type, max_val, tensor, rewriter.getI32ArrayAttr({dim}));
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
    SmallVector<Value> tensors;
    SmallVector<Attribute> dims;
    SmallVector<Value> maximums;
    IndexTreeDenseDomainOp operand_op;
    for(auto operand : op->getOperands())
    {
      if((operand_op = operand.getDefiningOp<IndexTreeDenseDomainOp>()))
      {
        to_remove.push_back(operand_op.getOperation());
        auto operand_tensors = operand_op.getTensors();
        tensors.insert(tensors.end(), operand_tensors.begin(), operand_tensors.end());
        auto tensor_dims = operand_op.getDimsAttr();
        dims.insert(dims.end(), tensor_dims.begin(), tensor_dims.end());
        maximums.push_back(operand_op.getDimSize());
      }
      else
      {
        domains.push_back(operand);
      }
    }

    if(domains.size() == 0) // All domains are dense
    {
      if(to_remove.size() > 1){
        auto loc = op->getLoc();
        auto context = rewriter.getContext();
        auto index_type = rewriter.getIndexType();
        Value max = maximums[0];
        // TODO: Do we need this to check if the domains are compatible?
        // for(Value new_max : maximums){
        //   max = rewriter.create<index::MaxUOp>(loc, index_type, max, new_max);
        // }
        indexTree::DomainType domain_type = indexTree::DomainType::get(context);
        Value new_domain = rewriter.create<IndexTreeDenseDomainOp>(loc, domain_type, max, tensors, rewriter.getArrayAttr(dims));
        rewriter.replaceOp(op, {new_domain});
      } else {
        rewriter.replaceOp(op, {first_domain});
      }
    } else if(domains.size() == 1)
    {
      // Remove intersection op completely
      rewriter.replaceOp(op, {domains[0]});
    } else
    {
      // Keep only non-dense operands
      auto loc = op->getLoc();
      auto context = rewriter.getContext();
      indexTree::DomainType domain_type = indexTree::DomainType::get(context);
      Value dim_size = llvm::dyn_cast<indexTree::ConcreteDomain>(domains[0].getDefiningOp()).getDimensionSize();
      Value new_domain = rewriter.create<IndexTreeDomainIntersectionOp>(loc, domain_type, domains, dim_size);
      rewriter.replaceOp(op, {new_domain});
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
    bool can_replace = false;
    Operation* operand_op;
    for(auto operand : op.getDomains())
    {
      if((operand_op = operand.getDefiningOp<IndexTreeDenseDomainOp>())){
        rewriter.replaceOp(op, {operand});
        can_replace = true;
        break;
      }
    }

    if(can_replace)
    {
      for(auto operand : op.getDomains())
      {
        operand_op = operand.getDefiningOp();
        if(operand_op->use_empty())
          rewriter.eraseOp(operand_op);
      }
    } else {
      Value dim_size = llvm::dyn_cast<ConcreteDomain>(op.getDomains()[0].getDefiningOp()).getDimensionSize();
      auto loc = op->getLoc();
      auto context = rewriter.getContext();
      indexTree::DomainType domain_type = indexTree::DomainType::get(context);
      Value new_op = rewriter.create<IndexTreeDomainUnionOp>(loc, domain_type, op.getDomains(), dim_size);
      rewriter.replaceOp(op, {new_op});
    }
    return success();
  }
};

struct InferOutputDomains : public OpRewritePattern<IndexTreeSparseTensorOp> {
  InferOutputDomains(MLIRContext *context)
      : OpRewritePattern<IndexTreeSparseTensorOp>(context, /*benefit=*/1) {}

  Value copyDomain(Value domain,
                   mlir::PatternRewriter &rewriter,
                   IRMapping& map,
                   Location loc,
                   llvm::SmallDenseMap<Value, Value>& index_vars) const
  {
    Value new_domain;
    Operation* domain_op = domain.getDefiningOp();
    if(llvm::isa<IndexTreeDomainIntersectionOp>(domain_op) || llvm::isa<IndexTreeDomainUnionOp>(domain_op))
    {
      for(Value subdomain : domain_op->getOperands()){
        copyDomain(subdomain, rewriter, map, loc, index_vars);
      }
    }
    
    if(llvm::isa<IndexTreeSparseDomainOp>(domain_op))
    {
      auto sparse_domain_op = llvm::cast<IndexTreeSparseDomainOp>(domain_op);

      // Ensure parent domain will also be copied. Otherwise create it
      Value new_parent_domain = nullptr;
      if(sparse_domain_op.getParent())
      {
        auto index_to_tensor_op = sparse_domain_op.getParent().getDefiningOp<IndexTreeIndexToTensorOp>();
        auto index_var = index_to_tensor_op.getIndex().getDefiningOp<IndexTreeIndicesOp>();
        if(index_vars.find(index_var) == index_vars.end()){
          new_parent_domain = copyDomain(index_var.getDomain(), rewriter, map, loc, index_vars);
          index_vars.insert(std::make_pair(index_var.getResult(), index_var.getDomain()));
        }
      }

      // Clone without parent
      new_domain = rewriter.create<IndexTreeSparseDomainOp>(loc, 
                                                            domain_op->getResultTypes(),
                                                            sparse_domain_op.getTensor(),
                                                            sparse_domain_op.getDimAttr(),
                                                            sparse_domain_op.getFormatAttr(),
                                                            sparse_domain_op.getPos(),
                                                            sparse_domain_op.getCrd(),
                                                            sparse_domain_op.getPosSize(),
                                                            sparse_domain_op.getCrdSize(),
                                                            sparse_domain_op.getDimSize(),
                                                            nullptr);
      map.map(sparse_domain_op, new_domain);

      if(new_parent_domain)
      {
        // Create or fold so multiple levels of nested domains are foleded into one
        new_domain = rewriter.createOrFold<IndexTreeNestedDomainOp>(loc, 
                                                                    domain_op->getResultTypes(),
                                                                    new_parent_domain, 
                                                                    new_domain);
      }
    } else {
      // Clone
      new_domain = rewriter.clone(*domain_op, map)->getResult(0);
    }
    return new_domain;
  }

  mlir::LogicalResult
  matchAndRewrite(IndexTreeSparseTensorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    for(auto domain : op.getDomains())
    {
      if(!llvm::isa<IndexTreeEmptyDomainOp>(domain.getDefiningOp()))
      {
        return failure();
      }
    }

    // Get the LHSOperandOp which creates thise tensor
    Value tensor = op->getResult(0);
    IndexTreeLHSOperandOp lhs_op = nullptr;
    for(Operation* op : tensor.getUsers())
    {
      if(llvm::isa<IndexTreeLHSOperandOp>(op))
      {
        lhs_op = llvm::cast<IndexTreeLHSOperandOp>(op);
      }
    }

    if(lhs_op == nullptr)
      return failure();

    
    auto crds = lhs_op.getCrds();
    unsigned dims = (lhs_op.getNumOperands() - 1) / 2;
    Value empty_domain = lhs_op.getOperand(0);
    llvm::IndexedMap<Value> domains(empty_domain);
    llvm::SmallDenseMap<Value, Value> index_vars;
    domains.resize(dims);
    for(Value crd : crds){
      auto access_op = llvm::dyn_cast<IndexTreeIndexToTensorOp>(crd.getDefiningOp());
      if(access_op == nullptr){
        return failure();
      }
      auto index_op = llvm::dyn_cast<IndexTreeIndicesOp>(access_op.getIndex().getDefiningOp());
      if(index_op == nullptr){
        return failure();
      }

      Value domain = index_op.getDomain();
      index_vars.insert(std::make_pair(index_op.getResult(), domain));
      domains[access_op.getDim()] = domain;
    }

    // Successfully matched! Cannot fail after this point.
    auto loc = op->getLoc();
    auto context = rewriter.getContext();
    SmallVector<Value> new_args;
    IRMapping map;
    for(unsigned dim = 0; dim < dims; dim++){
      Value domain_copy = copyDomain(domains[dim], rewriter, map, loc, index_vars);
      new_args.push_back(domain_copy);
    }
    auto new_tensor = rewriter.create<IndexTreeSparseTensorOp>(loc, op->getResult(0).getType(), new_args);
    rewriter.replaceOp(op, new_tensor->getResults());
    return success();
  }
};

void comet::indexTree::populateDomainConcretizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<ConcretizeTensorDomain, SimplifyIntersectionOp, InferOutputDomains>(context);
}

struct IndexTreeDomainConcretization : comet::impl::IndexTreeDomainConcretizationBase<IndexTreeDomainConcretization> {
  using IndexTreeDomainConcretizationBase::IndexTreeDomainConcretizationBase;

  void runOnOperation() {
    mlir::RewritePatternSet domain_concretization_patterns(&getContext());
    indexTree::populateDomainConcretizationPatterns(&getContext(), domain_concretization_patterns);
    tensor_domain_patterns.add<domain_concretization_patterns>(&getContext());
    mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(domain_concretization_patterns));
  }
};

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeDomainConcretizationPass()
{
  return std::make_unique<IndexTreeDomainConcretization>();
}