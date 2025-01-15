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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringSet.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/IndexTree/Patterns.h"

#include <set>
#include <unordered_map>

using namespace mlir;
using namespace mlir::indexTree;

// *********** For debug purpose *********//
#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
// *********** For debug purpose *********//

namespace mlir {
  namespace comet{
#define GEN_PASS_DEF_INDEXTREEKERNELFUSION
#include "comet/Dialect/IndexTree/Passes.h.inc"
  }
}

/// DimIndex = <tensor, dim>
namespace {

using DimCompound = std::pair<uint32_t, Value>;  /// <i, Tensor>, i.e., Tensor's i-th dimension.

template<class T>
uint32_t getIdxInVector(const llvm::SmallVector<T> &array, T value)
{
//  for (uint32_t i = 0; i < array.size(); ++i) {
//    if (array[i] == value) {
//      return i;
//    }
//  }
//  return (uint32_t) -1;

  auto it = std::find(array.begin(), array.end(), value);
  if (it != array.end()) {
    return it - array.begin();
  } else {
    return (uint32_t) -1;
  }
}


Value getRealLhsTensor(IndexTreeOp itree,
                       Value lhs_tensor)
{
  /// get the block arguments and itree arguments
  llvm::SmallVector<Value> block_args;
  for (Value arg : itree.getRegion().getBlocks().front().getArguments()) {
    block_args.push_back(arg);
  }
  llvm::SmallVector<Value> itree_args;
  for (Value arg : itree.getInputs()) {
    itree_args.push_back(arg);
  }

  /// Find the true operand if it is one of block arguments
  uint32_t idx = getIdxInVector(block_args, lhs_tensor);
  if (idx != (uint32_t) -1) {
    return itree_args[idx];
  } else {
    return lhs_tensor;
  }
}


void collectDimCompound(IndexTreeOp itree_op,
                        uint32_t &num_indexOps /*out*/,
                        llvm::SmallVector<llvm::DenseSet<DimCompound>> &indexOp_to_DimCompound /*out*/)
{
//  /// get the block arguments and itree arguments
//  llvm::SmallVector<Value> block_args;
//  for (Value arg : itree_op.getRegion().getBlocks().front().getArguments()) {
//    block_args.push_back(arg);
//  }
//  llvm::SmallVector<Value> itree_args;
//  for (Value arg : itree_op.getInputs()) {
//    itree_args.push_back(arg);
//  }

  /// Go through each IndexOp's `it.IndexToTensorDim`
  itree_op.getOperation()->walk([&](IndexTreeIndicesOp index_op) {
    ++num_indexOps;
    comet_vdump(index_op);
    indexOp_to_DimCompound.push_back(llvm::DenseSet<DimCompound>());
    auto &dims_set = indexOp_to_DimCompound.back();
    for (auto user : index_op->getUsers()) {
      if (auto indexToTensorDimOp = llvm::dyn_cast<IndexTreeIndexToTensorOp>(user)) {
        comet_vdump(indexToTensorDimOp);
        uint32_t dim = indexToTensorDimOp.getDim();
//        Value operand = indexToTensorDimOp.getTensor();
        Value operand = getRealLhsTensor(itree_op, indexToTensorDimOp.getTensor());
//        /// Find the true operand if it is one of block arguments
//        uint32_t idx = getIdxInVector(block_args, operand);
//        if (idx != (uint32_t) -1) {
//          operand = itree_args[idx];
//        }
        dims_set.insert(std::make_pair(dim, operand));
      }
    }
  });

}


bool isSameDimension(const llvm::DenseSet<DimCompound> &dim1, const llvm::DenseSet<DimCompound> &dim2)
{
  {
    comet_debug() << "dim1:\n";
    for (DimCompound dim : dim1) {
      comet_debug() << dim.first << " : " << dim.second << "\n";
    }
    comet_debug() << "dim2:\n";
    for (DimCompound dim : dim2) {
      comet_debug() << dim.first << " : " << dim.second << "\n";
    }
  }
  /// Find the intersection
  if (std::any_of(dim1.begin(), dim1.end(), [&](const DimCompound &entry) { return dim2.contains(entry); })) {
    return true;
  } else {
    return false;
  }
}


llvm::SmallVector<uint32_t> findCommonIndex(const llvm::SmallVector<llvm::DenseSet<DimCompound>> &indices_list1,
                                      const llvm::SmallVector<llvm::DenseSet<DimCompound>> &indices_list2)
{
  llvm::SmallVector<uint32_t> common_indices;

  uint32_t bound = std::min(indices_list1.size(), indices_list2.size());
  for (uint32_t idx = 0; idx < bound; ++idx) {
    if (isSameDimension(indices_list1[idx], indices_list2[idx])) {
      common_indices.push_back(idx);
    } else {
      break;
    }
  }

  comet_debug() << "common_indices.size(): " << common_indices.size() << "\n";
  return common_indices;
}


void collectCommonIndices(const llvm::SmallVector<IndexTreeOp> &itree_list,
                          llvm::SmallVector<uint32_t> &itree_to_num_indexOps /*out*/,
                          std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices /*out*/)
{
  llvm::SmallVector<llvm::SmallVector<llvm::DenseSet<DimCompound>>> itree_to_DimCompound(itree_list.size());
  for (uint32_t tree_i = 0; tree_i < itree_list.size(); ++tree_i) {
    collectDimCompound(itree_list[tree_i],
                       itree_to_num_indexOps[tree_i] /*out*/,
                       itree_to_DimCompound[tree_i] /*out*/);
  }

  for (uint32_t tree_i = 1; tree_i < itree_list.size(); ++tree_i) {
    itree_to_common_indices[tree_i] = findCommonIndex(itree_to_DimCompound[0], itree_to_DimCompound[tree_i]);
    {/// test
      for (uint32_t idx : itree_to_common_indices[tree_i]) {
        comet_debug() << idx << "\n";
      }
    }
  }

  {/// test
    comet_debug() << "itree_to_common_indices.size(): " << itree_to_common_indices.size() << "\n";
    for (auto v : itree_to_num_indexOps) {
      comet_debug() << v << "\n";
    }
  }

}


void collectOperandsDims(IndexTreeOp itree,
                         Value &lhs_tensor /*out*/,
                         llvm::SmallVector<DimCompound> &lhs_dims /*out*/,
                         llvm::SmallVector<uint32_t> &lhs_index_idx /*out*/,
                         llvm::SmallVector<Value> &rhs_tensors /*out*/,
                         llvm::SmallVector<llvm::SmallVector<DimCompound>> &rhs_dims /*out*/,
                         llvm::SmallVector<llvm::SmallVector<uint32_t>> &rhs_index_idx /*out*/)
{
  /// Collect all IndexOp
  llvm::SmallVector<Value> indexOp_list;
  itree.walk([&](IndexTreeIndicesOp op) {
    indexOp_list.push_back(op);
  });

//  /// get the block arguments and itree arguments
//  llvm::SmallVector<Value> block_args;
//  for (Value arg : itree.getRegion().getBlocks().front().getArguments()) {
//    block_args.push_back(arg);
//  }
//  llvm::SmallVector<Value> itree_args;
//  for (Value arg : itree.getInputs()) {
//    itree_args.push_back(arg);
//  }

  itree.walk([&](IndexTreeComputeOp computeOp) {
    /// LHS
    Value lhs_OperandOp = computeOp.getLhs();
    {
      auto lhs_op = llvm::cast<IndexTreeLHSOperandOp>(lhs_OperandOp.getDefiningOp());
//      lhs_tensor = lhs_op.getTensor();
      lhs_tensor = getRealLhsTensor(itree, lhs_op.getTensor());
//      /// Find the true operand if it is one of block arguments
//      uint32_t idx = getIdxInVector(block_args, lhs_tensor);
//      if (idx != (uint32_t) -1) {
//        lhs_tensor = itree_args[idx];
//      }
      auto positions = lhs_op.getPos();
      for (Value pos: positions)
      {
        auto indexToTensorDimOp = llvm::cast<IndexTreeIndexToTensorOp>(pos.getDefiningOp());
        uint32_t dim = indexToTensorDimOp.getDim();
        lhs_dims.push_back(std::make_pair(dim, lhs_tensor));

        /// Find out which IndexOp this IndexToTensorDim links to
        Value indexOp = indexToTensorDimOp.getIndex();
        uint32_t idx = getIdxInVector(indexOp_list, indexOp);
        assert(idx != (uint32_t) -1 && "Error: index_op did not exist.");
        lhs_index_idx.push_back(idx);

        {
          comet_vdump(lhs_op);
          comet_vdump(pos);
          comet_vdump(lhs_tensor);
          comet_debug() << "dim: " << dim << "\n";
          comet_debug() << "idx: " << idx << "\n";
          comet_debug() << "\n";
        }
      }
    }

    /// RHS
    llvm::SmallVector<Value> rhs_OperandOp;
    for (Value rhs : computeOp.getRhs()) {
      rhs_OperandOp.push_back(rhs);
    }

    /// The mask, if exists, is another OperandOp in Index Tree dialect. It will be the first generated `it.OperandOp`,
    /// but will be the last argument of generated `it.ComputeOp`.
    Value mask_tensor = computeOp.getMask();
    if (mask_tensor) {
      rhs_OperandOp.push_back(mask_tensor);
    }

    for (Value rhs_value : rhs_OperandOp) {
      auto rhs_op = llvm::cast<IndexTreeOperandOp>(rhs_value.getDefiningOp());
      rhs_tensors.push_back(rhs_op.getTensor());
      rhs_dims.push_back(llvm::SmallVector<DimCompound>());
      rhs_index_idx.push_back(llvm::SmallVector<uint32_t>());

      auto positions = rhs_op.getPos();
      for (Value pos : positions) {
        auto indexToTensorDimOp = llvm::cast<IndexTreeIndexToTensorOp>(pos.getDefiningOp());
        uint32_t dim = indexToTensorDimOp.getDim();
        rhs_dims.back().push_back(std::make_pair(dim, rhs_tensors.back()));

        /// Find out which IndexOp this IndexToTensorDim links to
        Value indexOp = indexToTensorDimOp.getIndex();
        uint32_t idx = getIdxInVector(indexOp_list, indexOp);
        assert(idx != (uint32_t) -1 && "Error: index_op did not exist.");
        rhs_index_idx.back().push_back(idx);

        {
          comet_vdump(rhs_op);
          comet_vdump(pos);
          comet_vdump(rhs_tensors.back());
          comet_debug() << "dim: " << dim << "\n";
          comet_debug() << "idx: " << idx << "\n";
          comet_debug() << "\n";
        }
      }
    }
  });
}


void collectComputeOpInfo(IndexTreeOp itree,
                          mlir::StringRef &semiring /*out*/,
                          bool &compute_missing /*out*/)
{
  itree.walk([&](IndexTreeComputeOp op) {
    semiring = op.getSemiring();
    compute_missing = op.getComputeMissing();
    comet_debug() << semiring << "\n";
    comet_debug() << compute_missing << "\n";
  });
}


IndexTreeOp createNewITree(IndexTreeOp host_itree,
                           const llvm::SmallVector<Type> &tree_types,
                           const llvm::SmallVector<Value> &itree_arguments,
                           mlir::IRRewriter &rewriter,
                           mlir::Location &loc)
{
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(host_itree);
  /// Create the new itree
  llvm::SmallVector<Location> locs;
  for (Value arg : itree_arguments) {
    locs.push_back(arg.getLoc());
  }
  IndexTreeOp new_itree = rewriter.create<indexTree::IndexTreeOp>(loc, tree_types, itree_arguments);
  Region *body = &new_itree.getRegion();
  Block *block = rewriter.createBlock(body, {}, TypeRange(tree_types), locs);
  YieldOp dummy_yield = rewriter.create<indexTree::YieldOp>(loc, TypeRange(), itree_arguments);  /// create a dummy YieldOp to make the following inlineBlockBefore() work.
  comet_vdump(new_itree);
  /// Move the host itree's block to the new itree. The host itree's argument happens to be the first argument of new_itree.
  rewriter.inlineBlockBefore(&host_itree.getRegion().front(),
                             block->getTerminator(),
                             {block->getArgument(0)} /*argValues to replace host itree's block arguments*/);
  rewriter.eraseOp(dummy_yield);  /// delete the dummy YieldOp, so the block will contain only one YieldOp.
  comet_vdump(new_itree);

  return new_itree;
}


  llvm::SmallVector<Value> createIndexOps(
      uint32_t tree_i,
      llvm::SmallVector<IndexTreeIndicesOp> &host_index_ops,
      const llvm::SmallVector<uint32_t> &itree_to_num_indexOps,
      std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
      MLIRContext *context,
      mlir::IRRewriter &rewriter,
      mlir::Location &loc)
{
  /// 1) Record the common
  llvm::SmallVector<Value> index_ops;
  for (uint32_t common_i = 0; common_i < itree_to_common_indices[tree_i].size(); ++common_i) {
    index_ops.push_back(host_index_ops[common_i]);
  }
  /// 2) Add the new index ops
  assert(!index_ops.empty() && "Common indices should not be empty.");
  Value parent = index_ops.back();
  indexTree::IndexNodeType index_node_type = indexTree::IndexNodeType::get(context);
  uint32_t num_new_index_op = itree_to_num_indexOps[tree_i] - itree_to_common_indices[tree_i].size();
  for (uint32_t op_i = 0; op_i < num_new_index_op; ++op_i) {
    parent = rewriter.create<indexTree::IndexTreeIndicesOp>(loc, index_node_type, parent);
    index_ops.push_back(parent);
  }

  return index_ops;
}


Value createLHSOperand(
    uint32_t tree_i,
    llvm::SmallVector<Value> &index_ops,
//    const llvm::SmallVector<Value> &itree_to_lhs_tensors,
    Value lhs_tensor,
    llvm::SmallVector<llvm::SmallVector<DimCompound>> &itree_to_lhs_dims,
    llvm::SmallVector<llvm::SmallVector<uint32_t>> &itree_to_lhs_index_idx,
    MLIRContext *context,
    mlir::IRRewriter &rewriter,
    mlir::Location &loc)
{
  llvm::SmallVector<Value> pos;
  llvm::SmallVector<Value> crds;
  Value prev_dim = nullptr;
  auto access_type = rewriter.getIndexType();
//  Value lhs_tensor = itree_to_lhs_tensors[tree_i];
  llvm::SmallVector<DimCompound> &lhs_dims = itree_to_lhs_dims[tree_i];
  llvm::SmallVector<uint32_t> &lhs_index_idx = itree_to_lhs_index_idx[tree_i];
  for (uint32_t d_i = 0; d_i < lhs_dims.size(); ++d_i)
  {
    Value index_node = index_ops[lhs_index_idx[d_i]];
    uint32_t dim = lhs_dims[d_i].first;
    auto access_op = rewriter.create<indexTree::IndexTreeIndexToTensorOp>(
        loc,
        TypeRange({access_type, access_type}),
        lhs_tensor,
        index_node,
        rewriter.getUI32IntegerAttr(dim),
        prev_dim);
    pos.push_back(access_op.getPos());
    crds.push_back(access_op.getCrd());
    prev_dim = pos.back();
  }

  indexTree::OperandType operand_type = indexTree::OperandType::get(context);
  Value lhs_operand = rewriter.create<indexTree::IndexTreeLHSOperandOp>(loc,
                                                                  operand_type,
                                                                  lhs_tensor,
                                                                  pos,
                                                                  crds);

  return lhs_operand;
}


llvm::SmallVector<Value> createRHSOperands(
    uint32_t tree_i,
    uint32_t intermediate_idx,
    Value prev_computeOp,
    llvm::SmallVector<Value> &index_ops,
    llvm::SmallVector<llvm::SmallVector<Value>> &itree_to_rhs_tensors,
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<DimCompound>>> &itree_to_rhs_dims,
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<uint32_t>>> &itree_to_rhs_index_idx,
    MLIRContext *context,
    mlir::IRRewriter &rewriter,
    mlir::Location &loc)
{
  llvm::SmallVector<Value> rhs_operands;
  llvm::SmallVector<Value> &rhs_tensors = itree_to_rhs_tensors[tree_i];

  indexTree::OperandType operand_type = indexTree::OperandType::get(context);
  for (uint32_t rhs_i = 0; rhs_i < rhs_tensors.size(); ++rhs_i) {
    llvm::SmallVector<Value> pos;
    llvm::SmallVector<Value> crds;
    Value prev_dim = nullptr;
    auto access_type = rewriter.getIndexType();
    llvm::SmallVector<DimCompound> &rhs_dims = itree_to_rhs_dims[tree_i][rhs_i];
    llvm::SmallVector<uint32_t> &rhs_index_idx = itree_to_rhs_index_idx[tree_i][rhs_i];
    Value rhs_tensor = rhs_tensors[rhs_i];
    if (rhs_i == intermediate_idx) {
      rhs_tensor = prev_computeOp;
    }
    for (uint32_t d_i = 0; d_i < rhs_dims.size(); ++d_i) {
      Value index_node = index_ops[rhs_index_idx[d_i]];
      uint32_t dim = rhs_dims[d_i].first;
      auto access_op = rewriter.create<indexTree::IndexTreeIndexToTensorOp>(
          loc,
          TypeRange({access_type, access_type}),
          rhs_tensor,
          index_node,
          rewriter.getUI32IntegerAttr(dim),
          prev_dim);
      pos.push_back(access_op.getPos());
      crds.push_back(access_op.getCrd());
      prev_dim = pos.back();
    }

    rhs_operands.push_back(
        rewriter.create<indexTree::IndexTreeOperandOp>(loc,
                                                       operand_type,
                                                       rhs_tensor,
                                                       pos,
                                                       crds));
  }
  return rhs_operands;
}


void createComputeOp(
    uint32_t tree_i,
    mlir::Type &tensor_type,
    llvm::SmallVector<Value> &index_ops,
    Value lhs_operand,
    llvm::SmallVector<Value> &rhs_operands,
    const llvm::SmallVector<mlir::StringRef> &itree_to_semiring,
    const llvm::SmallVector<bool> &itree_to_compute_missing,
    mlir::IRRewriter &rewriter,
    mlir::Location &loc,
    llvm::SmallVector<Value> &compute_ops /*out*/)
{
//  auto tensor_type = tree_types[tree_i];
  Value parent = index_ops.back();
  Value mask_operand = nullptr;
  if (rhs_operands.size() == 3) {
    mask_operand = rhs_operands.back();
    rhs_operands.pop_back();
  }
  mlir::StringRef semiring = itree_to_semiring[tree_i];
  bool compute_missing = itree_to_compute_missing[tree_i];
  Value compute_op = rewriter.create<indexTree::IndexTreeComputeOp>(
      loc,
      tensor_type,
      parent,
      lhs_operand,
      rhs_operands,
      mask_operand,
      rewriter.getStringAttr(semiring),
      rewriter.getBoolAttr(compute_missing));
  compute_ops.push_back(compute_op);
}


void fuseITrees(IndexTreeOp new_itree,
                uint32_t num_itrees,
                const llvm::SmallVector<Type> &tree_types,
                const llvm::SmallVector<uint32_t> &itree_to_num_indexOps,
                std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
                const llvm::SmallVector<Value> &itree_to_lhs_tensors,
                llvm::SmallVector<llvm::SmallVector<DimCompound>> &itree_to_lhs_dims,
                llvm::SmallVector<llvm::SmallVector<uint32_t>> &itree_to_lhs_index_idx,
                llvm::SmallVector<llvm::SmallVector<Value>> &itree_to_rhs_tensors,
                llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<DimCompound>>> &itree_to_rhs_dims,
                llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<uint32_t>>> &itree_to_rhs_index_idx,
                const llvm::SmallVector<mlir::StringRef> &itree_to_semiring,
                const llvm::SmallVector<bool> &itree_to_compute_missing,
                MLIRContext *context,
                mlir::IRRewriter &rewriter,
                mlir::Location &loc)
{
  /// Get the current ComputeOp
  llvm::SmallVector<Value> compute_ops;
  new_itree.walk([&](IndexTreeComputeOp op) {
    compute_ops.push_back(op);
  });
  /// Get all IndexOp of the new itree
  llvm::SmallVector<IndexTreeIndicesOp> host_index_ops;
  new_itree.walk([&](IndexTreeIndicesOp op) {
    host_index_ops.push_back(op);
  });

  indexTree::YieldOp yield_op = llvm::cast<indexTree::YieldOp>(new_itree.getRegion().getBlocks().front().getTerminator());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(yield_op);

  /// Fuse each other itree to the new itree.
  for (uint32_t tree_i = 1; tree_i < num_itrees; ++tree_i) {
    /// Create Index Ops: 1) Record the common index ops, then 2) add the new index ops.
    llvm::SmallVector<Value> index_ops = createIndexOps(tree_i,
                                                        host_index_ops,
                                                        itree_to_num_indexOps,
                                                        itree_to_common_indices,
                                                        context,
                                                        rewriter,
                                                        loc);
    comet_vdump(new_itree);

    /// Create LHS
    Value lhs_tensor = new_itree.getRegion().getBlocks().front().getArgument(tree_i);
    Value lhs_operand = createLHSOperand(tree_i,
                                         index_ops,
                                         lhs_tensor,
                                         itree_to_lhs_dims,
                                         itree_to_lhs_index_idx,
                                         context,
                                         rewriter,
                                         loc);
    comet_vdump(new_itree);

    /// Create RHS
    /// After fusion, one rhs operand should come from the intermediate variable.
    Value prev_computeOp = compute_ops.back();  /// the previous ComputeOp
    uint32_t intermediate_idx = getIdxInVector(itree_to_rhs_tensors[tree_i], itree_to_lhs_tensors[tree_i - 1]);  /// Which rhs operand is from the previous lhs operand
    llvm::SmallVector<Value> rhs_operands = createRHSOperands(tree_i,
                                                              intermediate_idx,
                                                              prev_computeOp,
                                                              index_ops,
                                                              itree_to_rhs_tensors,
                                                              itree_to_rhs_dims,
                                                              itree_to_rhs_index_idx,
                                                              context,
                                                              rewriter,
                                                              loc);
    comet_vdump(new_itree);

    /// Create Compute Ops
    auto tensor_type = tree_types[tree_i];
    createComputeOp(tree_i,
                    tensor_type,
                    index_ops,
                    lhs_operand,
                    rhs_operands,
                    itree_to_semiring,
                    itree_to_compute_missing,
                    rewriter,
                    loc,
                    compute_ops /*out*/);
    comet_vdump(new_itree);
    comet_debug() << "\n";
  }

  /// Create yield op
  rewriter.create<indexTree::YieldOp>(loc, TypeRange(), compute_ops);
  rewriter.eraseOp(yield_op);
  comet_vdump(new_itree);
}


void createITree(
    uint32_t num_itrees,
    const llvm::SmallVector<Value> &itree_arguments,
    llvm::SmallVector<IndexTreeOp> &itree_list,
    const llvm::SmallVector<uint32_t> &itree_to_num_indexOps,
    std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
    const llvm::SmallVector<Value> &itree_to_lhs_tensors,
    llvm::SmallVector<llvm::SmallVector<DimCompound>> &itree_to_lhs_dims,
    llvm::SmallVector<llvm::SmallVector<uint32_t>> &itree_to_lhs_index_idx,
    llvm::SmallVector<llvm::SmallVector<Value>> &itree_to_rhs_tensors,
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<DimCompound>>> &itree_to_rhs_dims,
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<uint32_t>>> &itree_to_rhs_index_idx,
    const llvm::SmallVector<mlir::StringRef> &itree_to_semiring,
    const llvm::SmallVector<bool> &itree_to_compute_missing)
{
  /// The 1st itree is the host.
  /// 1) Its region will be moved to the new itree.
  /// 2) Each other itree will be fused to the new itree.
  IndexTreeOp host_itree = itree_list.front();
  mlir::OpBuilder builder(host_itree);
  mlir::IRRewriter rewriter(builder);
  MLIRContext *context = rewriter.getContext();
  llvm::SmallVector<Type> tree_types;
  for (Value arg : itree_arguments) {
    tree_types.push_back(arg.getType());
  }
  auto loc = host_itree->getLoc();

  /// Create the new itree.
  IndexTreeOp new_itree = createNewITree(host_itree,
                                         tree_types,
                                         itree_arguments,
                                         rewriter,
                                         loc);

  /// Fuse other itrees to the new itree
  fuseITrees(new_itree,
             num_itrees,
             tree_types,
             itree_to_num_indexOps,
             itree_to_common_indices,
             itree_to_lhs_tensors,
             itree_to_lhs_dims,
             itree_to_lhs_index_idx,
             itree_to_rhs_tensors,
             itree_to_rhs_dims,
             itree_to_rhs_index_idx,
             itree_to_semiring,
             itree_to_compute_missing,
             context,
             rewriter,
             loc);

  /// Update the usage of results of itree
  uint32_t new_r_i = 0;
  for (uint32_t tree_i = 0; tree_i < num_itrees; ++tree_i) {
    for (uint32_t r_i = 0; r_i < itree_list[tree_i]->getNumResults(); ++r_i) {
      rewriter.replaceAllUsesWith(itree_list[tree_i].getResult(r_i), new_itree.getResult(new_r_i++));
    }
  }

  /// Delete the old itrees
  for (auto itree : itree_list) {
    rewriter.eraseOp(itree);
  }
}

}  /// anonymous namespace




struct IndexTreeKernelFusion : comet::impl::IndexTreeKernelFusionBase<IndexTreeKernelFusion>
{
  using IndexTreeKernelFusionBase::IndexTreeKernelFusionBase;
  void runOnOperation() override;
};

void IndexTreeKernelFusion::runOnOperation()
{
  comet_debug() << "IndexTreeKernelFusion::runOnOperation()\n";
  func::FuncOp funcOp = getOperation();
  comet_vdump(funcOp->getParentOfType<ModuleOp>());

  /// Collect all itrees.
  llvm::SmallVector<IndexTreeOp> itree_list;
  funcOp.walk([&](IndexTreeOp itree){
    comet_vdump(itree);
    itree_list.push_back(itree);
  });

  uint32_t num_itrees = itree_list.size();
  if (num_itrees < 2) {
    /// Need at least two itrees to fuse.
    return;
  }

  /// Find the overlap of IndicesOp;
  /// Each IndicesOp corresponds to a set of Dimension Compound <Tensor, Dim>.
  /// For example,
  ///   %14 = "it.IndexOp"(%13) : (!it.index_tree) -> !it.index
  ///   %crd_0, %pos_1 = "it.IndexToTensorDim"(%10, %14, %pos) <{dim = 1 : ui32}> : (tensor<?x4xf64>, !it.index, index) -> (index, index)
  ///   %crd_8, %pos_9 = "it.IndexToTensorDim"(%7, %14, %pos_7) <{dim = 1 : ui32}> : (tensor<?x4xf64>, !it.index, index) -> (index, index)
  /// Here %14 corresponds to <%10, 1> and <%7, 1>, meaning that %14 is the tensor %10's dim 1 and tensor %7's dim 1.
  /// The set of Dimension Compound will be used to tell if two IndexOps are the common index in two itrees.
  llvm::SmallVector<uint32_t> itree_to_num_indexOps(num_itrees);
  std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> itree_to_common_indices;
  collectCommonIndices(itree_list,
                       itree_to_num_indexOps /*out*/,
                       itree_to_common_indices /*out*/);
  bool has_common_indices = false;
  for (uint32_t tree_i = 1; tree_i < num_itrees; ++tree_i) {
    if (!itree_to_common_indices[tree_i].empty()) {
      has_common_indices = true;
      break;
    }
  }
  if (!has_common_indices) {
    /// No need to fuse if itrees don't have any common indices.
    return;
  }

  /// Collect all itrees' arguments
  llvm::SmallVector<Value> itree_arguments;
  for (IndexTreeOp &itree : itree_list) {
    for (Value arg : itree.getInputs()) {
      comet_vdump(arg);
      itree_arguments.push_back(arg);
    }
  }

  /// Collect all LHS and RHS operands

  /// itree_to_lhs_index_idx is used to tell which IndexOp are used by a LHSOperandOp's dimensions.
  /// Similarly, itree_to_rhs_index_idx is to tell which IndexOp are used by a OperandOp's dimensions.
  llvm::SmallVector<Value> itree_to_lhs_tensors(num_itrees);
  llvm::SmallVector<llvm::SmallVector<DimCompound>> itree_to_lhs_dims(num_itrees);
  llvm::SmallVector<llvm::SmallVector<uint32_t>> itree_to_lhs_index_idx(num_itrees);
  llvm::SmallVector<llvm::SmallVector<Value>> itree_to_rhs_tensors(num_itrees);
  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<DimCompound>>> itree_to_rhs_dims(num_itrees);
  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<uint32_t>>> itree_to_rhs_index_idx(num_itrees);
  for (uint32_t tree_i = 0; tree_i < num_itrees; ++tree_i) {
    collectOperandsDims(itree_list[tree_i],
                        itree_to_lhs_tensors[tree_i] /*out*/,
                        itree_to_lhs_dims[tree_i] /*out*/,
                        itree_to_lhs_index_idx[tree_i] /*out*/,
                        itree_to_rhs_tensors[tree_i] /*out*/,
                        itree_to_rhs_dims[tree_i] /*out*/,
                        itree_to_rhs_index_idx[tree_i] /*out*/);
  }

  /// Collect ComputeOp's information: semiring, and compute_missing.
  llvm::SmallVector<mlir::StringRef> itree_to_semiring(num_itrees);
  llvm::SmallVector<bool> itree_to_compute_missing(num_itrees, false);
  for (uint32_t tree_i = 0; tree_i < num_itrees; ++tree_i) {
    collectComputeOpInfo(itree_list[tree_i],
                         itree_to_semiring[tree_i] /*out*/,
                         itree_to_compute_missing[tree_i] /*out*/);
  }

  /// TODO: reduce tensor dimension before here

  /// Build the new itree
  createITree(num_itrees,
              itree_arguments,
              itree_list,
              itree_to_num_indexOps,
              itree_to_common_indices,
              itree_to_lhs_tensors,
              itree_to_lhs_dims,
              itree_to_lhs_index_idx,
              itree_to_rhs_tensors,
              itree_to_rhs_dims,
              itree_to_rhs_index_idx,
              itree_to_semiring,
              itree_to_compute_missing);

  comet_vdump(funcOp->getParentOfType<ModuleOp>());
}


/// Apply the redundancy-aware kernel fusion on index tree dialect for some compound expressions
std::unique_ptr<Pass> mlir::comet::createIndexTreeKernelFusionPass()
{
  return std::make_unique<IndexTreeKernelFusion>();
}