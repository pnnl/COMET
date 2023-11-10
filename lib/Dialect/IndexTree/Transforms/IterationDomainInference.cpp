#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/StringSet.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"

using namespace mlir;
using namespace mlir::indexTree;

namespace mlir {
    namespace comet{
    #define GEN_PASS_DEF_INDEXTREEDOMAININFERENCE
    #include "comet/Dialect/IndexTree/Passes.h.inc"
    }
}

static const llvm::StringSet<> Semiring_intersectOps{"land", "times"};

struct IndexTreeDomainInference : comet::impl::IndexTreeDomainInferenceBase<IndexTreeDomainInference> {
  using IndexTreeDomainInferenceBase::IndexTreeDomainInferenceBase;

  void runOnOperation() {
    // Get the current operation being operated on.
    indexTree::IndexTreeOp treeOp = getOperation();
    treeOp.walk([](indexTree::IndexTreeIndicesOp op){
        OpBuilder builder(op);
        Location loc = op.getLoc();
        auto context = builder.getContext();
        indexTree::DomainType domain_type = indexTree::DomainType::get(context);
        
        // Map operands to domains
        llvm::SmallDenseMap<Operation*, Value, 8> operands_to_domains;
        // Set of all compute operands
        llvm::SmallPtrSet<Operation*, 4> compute_ops;
        for(OpOperand& use : op->getUses())
        {
            Operation* operand_op = use.getOwner();
            if(!llvm::isa<indexTree::IndexTreeOperandOp>(operand_op))
              continue;

            auto tensor_val = llvm::cast<indexTree::IndexTreeOperandOp>(operand_op).getTensor();
            unsigned dim = use.getOperandNumber() - 1;
            Value domain = builder.create<indexTree::IndexTreeTensorDomainOp>(loc, 
                domain_type, tensor_val, builder.getUI32IntegerAttr(dim), nullptr);
            operands_to_domains.insert(std::pair<Operation*, Value>(operand_op, domain));
            compute_ops.insert(operand_op->user_begin(), operand_op->user_end());
        }

        llvm::SmallVector<Value, 8> domains;
        for(auto compute_op : compute_ops)
        {
          // Check if compute op needs intersection
          auto itComputeOp = cast<indexTree::IndexTreeComputeOp>(compute_op);
          auto semiringParts = itComputeOp.getSemiring().split('_');

          //TODO (alokvk2): Figure out what semantics we want for the Left hand side?
          if(operands_to_domains.find(itComputeOp.getLhs().getDefiningOp()) != operands_to_domains.end())
            domains.push_back(operands_to_domains[itComputeOp.getLhs().getDefiningOp()]);

          if(!Semiring_intersectOps.contains(semiringParts.second)){
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
                              loc, domain_type, intersection_domains));
            else
              domains.push_back(intersection_domains[0]);
          }
        }

       Value final_domain;
        if(domains.size() > 1) {
          final_domain = builder.create<indexTree::IndexTreeDomainUnionOp>(loc, 
                domain_type, domains);
        } else {
          final_domain = domains[0];
        }

        indexTree::IndexNodeType index_node_type = indexTree::IndexNodeType::get(context); 
        Value index_node = builder.create<indexTree::IndexTreeIndicesOp>(
                              loc, index_node_type, op.getParent(), final_domain);
        op.replaceAllUsesWith(index_node);
        op.erase();

    });
  }
};

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeDomainInferencePass()
{
  return std::make_unique<IndexTreeDomainInference>();
}