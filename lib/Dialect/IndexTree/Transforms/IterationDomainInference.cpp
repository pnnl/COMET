#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringSet.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/IndexTree/Patterns.h"

using namespace mlir;
using namespace mlir::indexTree;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
// *********** For debug purpose *********//

namespace mlir {
    namespace comet{
    #define GEN_PASS_DEF_INDEXTREEDOMAININFERENCE
    #include "comet/Dialect/IndexTree/Passes.h.inc"
    }
}

struct IndexTreeDomainInference : comet::impl::IndexTreeDomainInferenceBase<IndexTreeDomainInference> {
  using IndexTreeDomainInferenceBase::IndexTreeDomainInferenceBase;
  void runOnOperation() override;
};

struct InferIndexDomain : public OpRewritePattern<IndexTreeIndicesOp> {
  InferIndexDomain(MLIRContext *context)
    : OpRewritePattern<IndexTreeIndicesOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(IndexTreeIndicesOp op, mlir::PatternRewriter &builder) const override {
    if(op.getDomain())
      return failure();

    comet_vdump(op);

    Location loc = op.getLoc();
    auto context = builder.getContext();
    indexTree::DomainType domain_type = indexTree::DomainType::get(context);

    // Map operands to domains
    llvm::SmallDenseMap<Operation*, Value, 8> operands_to_domains;
    // Set of all compute operands
    llvm::SmallPtrSet<Operation*, 4> compute_ops;
    for(Operation* tensor_access_op : op->getUsers())
    {
      if(!llvm::isa<indexTree::IndexTreeIndexToTensorOp>(tensor_access_op))
        continue;

      for(Operation* operand_op : tensor_access_op->getUsers())
      {
        if(!llvm::isa<indexTree::IndexTreeOperandOp>(operand_op))
          continue;

        auto tensor_val = llvm::cast<indexTree::IndexTreeIndexToTensorOp>(tensor_access_op).getTensor();
        unsigned dim = llvm::cast<indexTree::IndexTreeIndexToTensorOp>(tensor_access_op).getDim();;
        Value domain = builder.create<indexTree::IndexTreeTensorDomainOp>(loc,
            domain_type,
            tensor_val,
            builder.getUI32IntegerAttr(dim),
            tensorAlgebra::TensorFormatEnumAttr::get(context, tensorAlgebra::TensorFormatEnum::UNK),
            nullptr);
        comet_vdump(domain);

        operands_to_domains.insert(std::pair<Operation*, Value>(operand_op, domain));
        compute_ops.insert(operand_op->user_begin(), operand_op->user_end());
        break;
      }
    }

    llvm::SmallVector<Value, 8> domains;
    Value zero = builder.create<index::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0));
    for(auto compute_op : compute_ops)
    {
      // Check if compute op needs intersection
      auto itComputeOp = cast<indexTree::IndexTreeComputeOp>(compute_op);
      auto semiringParts = itComputeOp.getSemiring().split('_');

      if(itComputeOp.getComputeMissing()){
        for(auto operand_op_val : itComputeOp.getRhs())
        {
          auto operand_op = operand_op_val.getDefiningOp();
          if(operands_to_domains.find(operand_op) != operands_to_domains.end())
            domains.push_back(operands_to_domains[operand_op]);
        }
      } else {
        SmallVector<Value, 4> intersection_domains;
        for(auto operand_op_val : itComputeOp.getRhs())
        {
          auto operand_op = operand_op_val.getDefiningOp();
          if(operands_to_domains.find(operand_op) != operands_to_domains.end())
            intersection_domains.push_back(operands_to_domains[operand_op]);
        }
        if(intersection_domains.size() > 1)
          domains.push_back(builder.create<indexTree::IndexTreeDomainIntersectionOp>(
                          loc, domain_type, intersection_domains, nullptr));
        else
          domains.push_back(intersection_domains[0]);
      }
    }

    Value final_domain;
    if(domains.size() > 1) {
      final_domain = builder.create<indexTree::IndexTreeDomainUnionOp>(loc,
            domain_type, domains, nullptr);
    } else {
      final_domain = domains[0];
    }
    comet_vdump(final_domain);

    indexTree::IndexNodeType index_node_type = indexTree::IndexNodeType::get(context);
    builder.replaceOpWithNewOp<indexTree::IndexTreeIndicesOp>(
                          op, index_node_type, op.getParent(), final_domain);
    return success();
  }
};

void mlir::indexTree::populateDomainInferencePatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<InferIndexDomain>(context);
}

void IndexTreeDomainInference::runOnOperation(){
  mlir::RewritePatternSet domain_inference_patterns(&getContext());
  populateDomainInferencePatterns(&getContext(), domain_inference_patterns);
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(domain_inference_patterns))))
    signalPassFailure();
}

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeDomainInferencePass()
{
  return std::make_unique<IndexTreeDomainInference>();
}