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
    #define GEN_PASS_DEF_INDEXTREESYMBOLICCOMPUTEPASS
    #include "comet/Dialect/IndexTree/Passes.h.inc"
  }
}

struct CreateSymbolicTree :  public OpRewritePattern<IndexTreeSparseTensorOp> {
  CreateSymbolicTree(MLIRContext *context)
      : OpRewritePattern<IndexTreeSparseTensorOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(IndexTreeSparseTensorOp it_tensor_decl_op, 
                  mlir::PatternRewriter &rewriter) const override {
    for(auto domain : it_tensor_decl_op.getDomains())
    {
      if(llvm::isa<IndexTreeOp>(domain.getDefiningOp()))
      {
        return failure();
      }
    }

    auto loc = it_tensor_decl_op->getLoc();
    auto context = rewriter.getContext();

    auto domain_type = DomainType::get(context);
    auto itree_op = rewriter.create<IndexTreeOp>(loc, it_tensor_decl_op->getOperandTypes());
    Region* body = &itree_op.getRegion();
    loc = body->getLoc();
    Block* block = rewriter.createBlock(body);
    rewriter.setInsertionPointToStart(block);

    indexTree::IndexTreeType tree_type = indexTree::IndexTreeType::get(context);
    Value parent = rewriter.create<indexTree::IndexTreeRootOp>(loc, tree_type);
    indexTree::IndexNodeType index_node_type = indexTree::IndexNodeType::get(context); 
    std::vector<Value> index_nodes;
    for (Value domain : it_tensor_decl_op.getDomains())
    {
      parent = rewriter.create<indexTree::IndexTreeIndicesOp>(loc, index_node_type, parent, domain);
      index_nodes.push_back(parent);
    }
    auto symbolic_domain = rewriter.create<indexTree::SymbolicComputeOp>(loc, it_tensor_decl_op->getOperandTypes(), parent);
    rewriter.create<indexTree::YieldOp>(loc, TypeRange(), symbolic_domain->getResults());

    rewriter.setInsertionPointAfter(itree_op);
    auto new_tensor = rewriter.create<indexTree::IndexTreeSparseTensorOp>(loc, it_tensor_decl_op->getResultTypes(), itree_op->getResults());
    rewriter.replaceOp(it_tensor_decl_op, new_tensor->getResults());

    return success();
  }
};

struct IndexTreeSymbolicComputePass : comet::impl::IndexTreeSymbolicComputePassBase<IndexTreeSymbolicComputePass> {
  using IndexTreeSymbolicComputePassBase::IndexTreeSymbolicComputePassBase;

  void runOnOperation() {
    mlir::RewritePatternSet sp_output_patterns(&getContext());
    sp_output_patterns.add<CreateSymbolicTree>(&getContext());
    mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(sp_output_patterns));
  }
};

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeSymbolicComputePass()
{
  return std::make_unique<IndexTreeSymbolicComputePass>();
}