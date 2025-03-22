#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseMap.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Analysis/CopiedDomainAnalysis.h"

using namespace mlir;
using namespace mlir::indexTree;

CopiedDomainAnalysis::CopiedDomainAnalysis(Operation* op)
{
    op->walk([&](IndexTreeOp tree_op) {
        tree_op.getBody()->walk([&](IndexTreeComputeOp compute_op) {
            analyzeDomains(compute_op);
        });
    });
}

void CopiedDomainAnalysis::analyzeDomains(IndexTreeComputeOp compute_op)
{
llvm::SmallDenseSet<Value> output_vars;
llvm::SmallDenseMap<Value, uint32_t> candidates;
IndexTreeLHSOperandOp lhs = compute_op.getLhs().getDefiningOp<IndexTreeLHSOperandOp>();
Value tensor = compute_op.getResult();
for(Value pos : lhs.getPos()){
    auto index_to_tensor = pos.getDefiningOp<IndexTreeIndexToTensorOp>();
    output_vars.insert(index_to_tensor.getIndex());
    candidates.insert(std::make_pair(index_to_tensor.getIndex(), index_to_tensor.getDim()));
}

llvm::SmallDenseSet<Value> appeared_once;
for(Value rhs : compute_op.getRhs()){
    auto positions = rhs.getDefiningOp<IndexTreeOperandOp>().getPos();
    for(auto pos : positions) {
        auto index_to_tensor = pos.getDefiningOp<IndexTreeIndexToTensorOp>();
        Value index = index_to_tensor.getIndex();

        // Reduction variable appears twice in the same expression, and not in the output
        if(!output_vars.contains(index) && appeared_once.contains(index)) { 
            reductionVars.insert(std::make_pair(compute_op, index));
        }
        appeared_once.insert(index);
    }
}

for(Value rhs : compute_op.getRhs()){
    auto positions = rhs.getDefiningOp<IndexTreeOperandOp>().getPos();
    uint32_t dim = 0;
    for(; dim < positions.size(); dim += 1) {
        auto index_to_tensor = positions[dim].getDefiningOp<IndexTreeIndexToTensorOp>();
        Value index = index_to_tensor.getIndex();
        if(reductionVars.contains(std::make_pair(compute_op, index))) { 
            candidates.erase(index);
            break;
        }
    }

    // Erase everything that comes after a reduction variable
    for(dim += 1; dim < positions.size(); dim += 1) {
        auto index_to_tensor = positions[dim].getDefiningOp<IndexTreeIndexToTensorOp>();
        Value index = index_to_tensor.getIndex();
        candidates.erase(index);
    }
}

for(auto it = candidates.begin(); it != candidates.end(); it++) {
    uint32_t dim = it->getSecond();
    copiedDomains.insert(std::make_pair(tensor, dim));
}
}

bool CopiedDomainAnalysis::isCopiedDomain(Value tensor, unsigned dim)
{
    return copiedDomains.contains(std::make_pair(tensor, dim));
}

bool CopiedDomainAnalysis::isReductionVar(IndexTreeComputeOp op, Value index_var)
{
   return reductionVars.contains(std::make_pair(op, index_var));
}