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
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseMap.h"

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
  CopiedDomainAnalysis& copiedDomains;
  InferIndexDomain(MLIRContext *context, CopiedDomainAnalysis& copiedDomains)
    : OpRewritePattern<IndexTreeIndicesOp>(context, /*benefit=*/1), copiedDomains(copiedDomains) {}

  mlir::LogicalResult
  matchAndRewrite(IndexTreeIndicesOp op, mlir::PatternRewriter &builder) const override {
    if(op.getDomain())
      return failure();

    comet_vdump(op);

    Location loc = op.getLoc();
    auto context = builder.getContext();
    indexTree::DomainType domain_type = indexTree::DomainType::get(context);

    // Map operands to domains
    // Will break if operand appears multiple times for the same index variable
    llvm::SmallDenseMap<Operation*, Value, 8> operands_to_domains;

    // Set of tensors that we need to infer domain
    llvm::SmallDenseSet<Value> intermediate_tensors;
    
    // Set of all compute operands
    llvm::SmallPtrSet<Operation*, 4> compute_ops;
    for(Operation* tensor_access_op : op->getUsers())
    {
      if(!llvm::isa<indexTree::IndexTreeIndexToTensorOp>(tensor_access_op))
        continue;

      comet_pdump(tensor_access_op);
      for(Operation* operand_op : tensor_access_op->getUsers())
      {
        if(!llvm::isa<indexTree::IndexTreeOperandOp>(operand_op))
          continue;

        comet_pdump(operand_op);
        auto tensor_val = llvm::cast<indexTree::IndexTreeIndexToTensorOp>(tensor_access_op).getTensor();
        unsigned dim = llvm::cast<indexTree::IndexTreeIndexToTensorOp>(tensor_access_op).getDim();
        comet_vdump(tensor_val);
        comet_debug() << "dim: " << dim << "\n";
        Value domain;
        if(llvm::isa<IndexTreeComputeOp>(tensor_val.getDefiningOp()) && op->isBeforeInBlock(tensor_val.getDefiningOp()))
        {
          if(copiedDomains.isCopiedDomain(tensor_val, dim)){
            // The domain is the same on the LHS as it is on the RHS
            // We need to find the domain of this index variable on the RHS
            intermediate_tensors.insert(tensor_val);
          } else {
            // We promote this domain to dense because we cannot narrow the domain further
            // Use negative one to indicate to use the maximums of the other tensors
            Value neg_one = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(-1));
            domain = builder.create<indexTree::IndexTreeDenseDomainOp>(
              loc,
              domain_type,
              neg_one,
              ValueRange(),
              builder.getI32ArrayAttr(llvm::ArrayRef<int32_t>())
            );
            operands_to_domains.insert(std::pair<Operation*, Value>(operand_op, domain));
          }
        } else {
          domain = builder.create<indexTree::IndexTreeTensorDomainOp>(loc,
            domain_type,
            tensor_val,
            builder.getUI32IntegerAttr(dim),
            tensorAlgebra::TensorFormatEnumAttr::get(context, tensorAlgebra::TensorFormatEnum::UNK),
            nullptr
          );
          operands_to_domains.insert(std::pair<Operation*, Value>(operand_op, domain));
        }
        compute_ops.insert(operand_op->user_begin(), operand_op->user_end());
        break;
      }
    }

    llvm::SmallVector<Value, 4> domains;
    Value zero = builder.create<index::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0));
    auto tree_op = op->getParentOfType<IndexTreeOp>();
    tree_op.getBody()->walk([&](IndexTreeComputeOp compute_op) {
      if(!compute_ops.contains(compute_op.getOperation())){
        return WalkResult::advance();
      }
      // Check if compute op needs intersection
      auto itComputeOp = cast<indexTree::IndexTreeComputeOp>(compute_op);
      auto semiringParts = itComputeOp.getSemiring().split('_');
      comet_vdump(itComputeOp);

      if(itComputeOp.getComputeMissing()){
        SmallVector<Value, 4> union_domains;
        for(auto operand_op_val : itComputeOp.getRhs())
        {
          auto operand_op = operand_op_val.getDefiningOp();
          if(operands_to_domains.find(operand_op) != operands_to_domains.end()) {
            domains.push_back(operands_to_domains[operand_op]);
            union_domains.push_back(operands_to_domains[operand_op]);
          }

          if(intermediate_tensors.contains(itComputeOp.getResult())){
            Value new_domain = builder.create<IndexTreeDomainUnionOp>(
              loc,
              domain_type,
              union_domains
            );
            for(auto user : itComputeOp.getResult().getUsers()) {
              if(llvm::isa<IndexTreeOperandOp>(user)){
                operands_to_domains.insert(std::make_pair(user, new_domain));
              }
            }
          }
        }
      } else {
        Value temp_domain;
        SmallVector<Value, 4> intersection_domains;
        for(auto operand_op_val : itComputeOp.getRhs())
        {
          auto operand_op = operand_op_val.getDefiningOp();
          comet_pdump(operand_op);
          if(operands_to_domains.find(operand_op) != operands_to_domains.end())
            intersection_domains.push_back(operands_to_domains[operand_op]);
        }
        if(intersection_domains.size() > 1)
          temp_domain = builder.create<indexTree::IndexTreeDomainIntersectionOp>(loc, domain_type, intersection_domains, nullptr);
        else
          temp_domain = intersection_domains[0];

        if(itComputeOp.getMask()) {
          auto operand_op = itComputeOp.getMask().getDefiningOp();
          if(operands_to_domains.find(operand_op) != operands_to_domains.end())
          {
            temp_domain = builder.create<indexTree::IndexTreeDomainIntersectionOp>(
              loc, 
              domain_type, 
              ValueRange{temp_domain, operands_to_domains[operand_op]}, 
              nullptr
            );
          }
        }

        if(intermediate_tensors.contains(itComputeOp.getResult())){
          for(auto user : itComputeOp.getResult().getUsers()) {
            if(llvm::isa<IndexTreeOperandOp>(user)){
              operands_to_domains.insert(std::make_pair(user, temp_domain));
            }
          }
        }
        domains.push_back(temp_domain);
      }
      return WalkResult::advance();
    });

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
                          op, index_node_type, op.getParent(), final_domain, op.getIsParallelAttr());
    return success();
  }
};

void mlir::indexTree::populateDomainInferencePatterns(
    MLIRContext *context, RewritePatternSet &patterns, CopiedDomainAnalysis &copiedDomains) {
  patterns.add<InferIndexDomain>(context, copiedDomains);
}

void IndexTreeDomainInference::runOnOperation() {
  mlir::RewritePatternSet domain_inference_patterns(&getContext());
  CopiedDomainAnalysis& copiedDomains = getAnalysis<CopiedDomainAnalysis>();
  populateDomainInferencePatterns(&getContext(), domain_inference_patterns, copiedDomains);
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(domain_inference_patterns))))
    signalPassFailure();
}

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeDomainInferencePass()
{
  return std::make_unique<IndexTreeDomainInference>();
}