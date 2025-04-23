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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/Passes.h"

#include <unordered_map>
#include "comet/Dialect/IndexTree/Patterns.h"

using namespace mlir;
using namespace mlir::indexTree;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
// *********** For debug purpose *********//

namespace mlir {
  namespace comet{
#define GEN_PASS_DEF_INDEXTREEKERNELFUSION
#include "comet/Dialect/IndexTree/Passes.h.inc"
  }
}

namespace {

using DimCompound = std::pair<uint32_t, Value>;  /// <i, Tensor>, i.e., Tensor's i-th dimension.

template<class T>
uint32_t getIdxInVector(const llvm::SmallVector<T> &array, T value)
{
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
  for (Value arg : itree->getResults()) {
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
        /// Find the true operand if it is one of block arguments
        Value operand = getRealLhsTensor(itree_op, indexToTensorDimOp.getTensor());
        dims_set.insert(std::make_pair(dim, operand));
      }
    }
  });
}


bool isSameDimension(const llvm::DenseSet<DimCompound> &dim1, const llvm::DenseSet<DimCompound> &dim2)
{
  /// Find the intersection
  if (std::any_of(dim1.begin(), dim1.end(), [&](const DimCompound &entry) { return dim2.contains(entry); })) {
    /// If any dimension in dim1 happens to be in dim2, dim1 and dim2 has intersection, thus they are representing the same dimension
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

  SmallVector<IndexTreeIndicesOp, 4> host_indices;
  itree_list[0]->walk([&host_indices] (IndexTreeIndicesOp indice) {
    host_indices.push_back(indice);
  });


  for (uint32_t tree_i = 1; tree_i < itree_list.size(); ++tree_i) {
    itree_to_common_indices[tree_i] = findCommonIndex(itree_to_DimCompound[0], itree_to_DimCompound[tree_i]);
    SmallVector<IndexTreeIndicesOp, 4> child_indices;
    itree_list[tree_i]->walk([&child_indices] (IndexTreeIndicesOp indice) {
      child_indices.push_back(indice);
    });
    for(size_t index: itree_to_common_indices[tree_i])
    {
      if(host_indices[index].getIsParallel() != child_indices[index].getIsParallel())
      {
        host_indices[index].setIsParallel(false);
      }
    }
//    {/// test
//      for (uint32_t idx : itree_to_common_indices[tree_i]) {
//        comet_debug() << idx << "\n";
//      }
//    }
  }

//  {/// test
//    comet_debug() << "itree_to_common_indices.size(): " << itree_to_common_indices.size() << "\n";
//    for (auto v : itree_to_num_indexOps) {
//      comet_debug() << v << "\n";
//    }
//  }

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

  itree.walk([&](IndexTreeComputeOp computeOp) {
    /// LHS
    Value lhs_OperandOp = computeOp.getLhs();
    {
      auto lhs_op = llvm::cast<IndexTreeLHSOperandOp>(lhs_OperandOp.getDefiningOp());
      /// Find the true operand if it is one of block arguments
      lhs_tensor = getRealLhsTensor(itree, lhs_op.getTensor());
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

uint32_t collectIntermediateIdx(Value prev_lhs_tensor,
                                const llvm::SmallVector<Value> &curr_rhs_tensors)
{
  uint32_t idx = getIdxInVector(curr_rhs_tensors, prev_lhs_tensor);
  assert(idx != (uint32_t) -1 && "Error: none of current rhs tensors is the previous lhs tensor.");
  return idx;
}

[[maybe_unused]] std::unordered_map<uint32_t, Value> createNewLhsTensors(
    uint32_t num_itrees,
    llvm::SmallVector<Value> &itree_to_lhs_tensors,
    std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices)
{
  std::unordered_map<uint32_t, Value> itree_to_new_tensors;
  for (uint32_t tree_i = 0; tree_i < num_itrees - 1; ++tree_i) {
    /// Old tensor example
    /// %10 = "ta.dense_tensor_decl"(%5) <{format = "Dense"}> : (index) -> tensor<?x4xf64>
    Value old_tensor = itree_to_lhs_tensors[tree_i];
    llvm::SmallVector<uint32_t> &common_indices = itree_to_common_indices[tree_i + 1];
    assert(llvm::isa<mlir::TensorType>(old_tensor.getType()) &&
        "Error: old_tensor is not a mlir::TensorType.");
    assert(!common_indices.empty() && "Error: no common indices.");
    auto old_tensor_tt = llvm::cast<mlir::TensorType>(old_tensor.getType());

    /// The start dim idx after those common indices
    uint32_t dim_base = common_indices.size();
//    dim_base = 1;  /// test
    if (dim_base < old_tensor_tt.getRank()) {
      /// The new tensor is still a tensor, with decreased dimensions.

      uint32_t count_of_dyn_dim = 0;  /// How many dynamic dimension is in common indices
      for (uint32_t dim_i = 0; dim_i < dim_base; ++dim_i) {
        if (old_tensor_tt.isDynamicDim(dim_i)) {
          ++count_of_dyn_dim;
        }
      }
      SmallVector<Value> operands;  /// The remaining dynamic dimensions after fusion.
      if (old_tensor.getDefiningOp()->getNumOperands() > count_of_dyn_dim) {
        operands.insert(operands.begin(),
                        old_tensor.getDefiningOp()->getOperands().begin() + count_of_dyn_dim,
                        old_tensor.getDefiningOp()->getOperands().end());
      }

      llvm::SmallVector<int64_t> shape;
      for (int64_t dim_i = dim_base; dim_i < old_tensor_tt.getRank(); ++dim_i) {
        if (old_tensor_tt.isDynamicDim(dim_i)) {
          shape.push_back(mlir::ShapedType::kDynamic);
        } else {
          shape.push_back(old_tensor_tt.getDimSize(dim_i));
        }
      }

      auto loc = old_tensor.getLoc();
      mlir::OpBuilder builder(old_tensor.getDefiningOp());
      Value new_tensor = builder.create<tensorAlgebra::DenseTensorDeclOp>(
          loc,
          mlir::RankedTensorType::get(shape, old_tensor_tt.getElementType()),
          operands);
      itree_to_new_tensors[tree_i] = new_tensor;
      mlir::TypedAttr zero = builder.getZeroAttr(old_tensor_tt.getElementType());
      builder.create<tensorAlgebra::TensorFillOp>(loc,
                                                  new_tensor,
                                                  zero);
      comet_vdump(new_tensor);
    } else {
      /// The new tensor is shrunk to a scalar.
      auto loc = old_tensor.getLoc();
      mlir::OpBuilder builder(old_tensor.getDefiningOp());
      mlir::TypedAttr zero = builder.getZeroAttr(old_tensor_tt.getElementType());
      Value new_scalar = builder.create<arith::ConstantOp>(loc,
                                                           old_tensor_tt.getElementType(),
                                                           zero);
      comet_vdump(new_scalar);
      itree_to_new_tensors[tree_i] = new_scalar;
    }
  }

  return itree_to_new_tensors;
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

/// 1) Generate new IndexToTensorDim for the new LHS operand.
/// 2) Replace the old LHS operand with the new one.
/// 3) Erase the old LHS operand and its IndexToTensorDim.
[[maybe_unused]] Value replaceOldLHSOperand(IndexTreeOp itree_op,
                           Value lhs_tensor,
                           llvm::SmallVector<llvm::SmallVector<DimCompound>> &itree_to_lhs_dims,
                           llvm::SmallVector<llvm::SmallVector<uint32_t>> &itree_to_lhs_index_idx,
                           const std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
                           MLIRContext *context,
                           mlir::IRRewriter &rewriter,
                           mlir::Location &loc)
{
  /// Get all IndexOp
  llvm::SmallVector<Value> index_ops;
  itree_op.walk([&](IndexTreeIndicesOp op) {
    index_ops.push_back(op);
  });
  /// Find the LHS operand
  IndexTreeLHSOperandOp old_lhs_operand;
  uint32_t lhs_count = 0;
  itree_op.walk([&](IndexTreeLHSOperandOp op) {
    old_lhs_operand = op;
    ++lhs_count;
  });
  assert(lhs_count == 1 && "Error: The kernel should only have one lhs operand.");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(old_lhs_operand);

  llvm::SmallVector<Value> pos;
  llvm::SmallVector<Value> crds;
  Value prev_dim = nullptr;
  auto access_type = rewriter.getIndexType();
  uint32_t tree_i = 0;  /// the host itree
  llvm::SmallVector<DimCompound> &lhs_dims = itree_to_lhs_dims[tree_i];
  llvm::SmallVector<uint32_t> &lhs_index_idx = itree_to_lhs_index_idx[tree_i];
  uint32_t dim_base = itree_to_common_indices.at(tree_i + 1).size();
//  dim_base = 1;  /// test
  for (uint32_t d_i = dim_base; d_i < lhs_dims.size(); ++d_i) {
    Value index_node = index_ops[lhs_index_idx[d_i]];
    uint32_t dim = lhs_dims[d_i].first - dim_base;
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
  Value new_lhs_operand = rewriter.create<indexTree::IndexTreeLHSOperandOp>(loc,
                                                                            operand_type,
                                                                            lhs_tensor,
                                                                            pos,
                                                                            crds);
  comet_vdump(new_lhs_operand);
  comet_vdump(new_lhs_operand.getDefiningOp()->getParentOfType<ModuleOp>());

  /// Replace and erase
  rewriter.replaceAllUsesWith(old_lhs_operand, new_lhs_operand);
  llvm::SmallVector<Value> old_index_to_tensor_dim;
  for (auto tensor_dim : old_lhs_operand.getPos()) {
    old_index_to_tensor_dim.push_back(tensor_dim);
  }
  std::reverse(old_index_to_tensor_dim.begin(), old_index_to_tensor_dim.end());
  rewriter.eraseOp(old_lhs_operand);
  for (auto tensor_dim : old_index_to_tensor_dim) {
    rewriter.eraseOp(tensor_dim.getDefiningOp());
  }

  comet_vdump(new_lhs_operand.getDefiningOp()->getParentOfType<ModuleOp>());
  return new_lhs_operand;
}


/// Create the new ComputeOp for the new LHS operand.
/// Only the return type needs change to the new LHS operand's type.
[[maybe_unused]] Value replaceOldComputeOp(IndexTreeOp itree_op,
                         const mlir::Type &tensor_type,
                         mlir::IRRewriter &rewriter,
                         mlir::Location &loc)
{
  IndexTreeComputeOp old_compute_op;
  uint32_t count_compute_op = 0;
  itree_op.walk([&](IndexTreeComputeOp op) {
    old_compute_op = op;
    ++count_compute_op;
  });
  assert(count_compute_op == 1 && "Error: expect only one ComputeOp");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(old_compute_op);
  Value new_compute_op = rewriter.create<indexTree::IndexTreeComputeOp>(
      loc,
      tensor_type,
      old_compute_op.getParent(),
      old_compute_op.getLhs(),
      old_compute_op.getRhs(),
      old_compute_op.getMask(),
      old_compute_op.getSemiringAttr(),
      old_compute_op.getComputeMissingAttr());

  comet_vdump(new_compute_op);
  rewriter.replaceAllUsesWith(old_compute_op, new_compute_op);
  rewriter.eraseOp(old_compute_op);
  comet_vdump(new_compute_op.getDefiningOp()->getParentOfType<ModuleOp>());

  return new_compute_op;
}


/// 1) Create the new itree based on the host itree (the 0th one).
/// 2) Replace the old LHS operand to the new one with shrunk dimensions.
/// 3) Replace the old ComputeOp with the new one with correct return type (shrunk tensor type).
IndexTreeOp createNewITree(IndexTreeOp host_itree,
                           const llvm::SmallVector<Type> &tree_types,
//                           const llvm::SmallVector<Value> &itree_arguments,
                           const llvm::SmallVector<Value> &itree_arguments_inputs,
                           const llvm::SmallVector<Value> &itree_arguments_intermediates,
                           llvm::SmallVector<llvm::SmallVector<DimCompound>> &itree_to_lhs_dims,
                           llvm::SmallVector<llvm::SmallVector<uint32_t>> &itree_to_lhs_index_idx,
//                           const std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
                           llvm::SmallVector<Value> &itree_to_new_compute_op /*out*/,
                           MLIRContext *context,
                           mlir::IRRewriter &rewriter,
                           mlir::Location &loc)
{
  OpBuilder::InsertionGuard guard(rewriter);
  /// Create the new itree
  llvm::SmallVector<Location> locs;
//  for (Value arg : itree_arguments) {
//    locs.push_back(arg.getLoc());
//  }
  for (Value arg : itree_arguments_inputs) {
    locs.push_back(arg.getLoc());
  }
  for (Value arg : itree_arguments_intermediates) {
    locs.push_back(arg.getLoc());
  }
  IndexTreeOp new_itree = rewriter.create<indexTree::IndexTreeOp>(loc, tree_types, itree_arguments_inputs,
                                                                  itree_arguments_intermediates);
  Region *body = &new_itree.getRegion();
  Block *block = rewriter.createBlock(body, {}, TypeRange(tree_types), locs);
  YieldOp dummy_yield = rewriter.create<indexTree::YieldOp>(loc, TypeRange(), itree_arguments_inputs);  /// create a dummy YieldOp to make the following inlineBlockBefore() work.
  comet_vdump(new_itree);
  /// Move the host itree's block to the new itree. The host itree's argument happens to be the first `intermediates` argument of new_itree.
  /// An itree's arguments are it.itree($inputs, $intermediates)
  rewriter.inlineBlockBefore(&host_itree.getRegion().front(),
                             block->getTerminator(),
                             {block->getArgument(itree_arguments_inputs.size())} /*argValues to replace host itree (the source)'s block arguments*/);

//  /// Replace the old LHS
//  replaceOldLHSOperand(new_itree,
//                       block->getArgument(tree_i),
//                       itree_to_lhs_dims,
//                       itree_to_lhs_index_idx,
//                       itree_to_common_indices,
//                       context,
//                       rewriter,
//                       loc);
//
//  /// Replace the old ComputeOp
//  Value new_compute_op = replaceOldComputeOp(new_itree,
//                                tree_types[tree_i],
//                                rewriter,
//                                loc);
//  itree_to_new_compute_op.push_back(new_compute_op);
  body->walk([&](IndexTreeComputeOp computOp) {
    itree_to_new_compute_op.push_back(computOp);
  });
  assert(itree_to_new_compute_op.size() == 1 && "Expect only one computeOp in the itree.");

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
    Value lhs_tensor,
    llvm::SmallVector<llvm::SmallVector<DimCompound>> &itree_to_lhs_dims,
    llvm::SmallVector<llvm::SmallVector<uint32_t>> &itree_to_lhs_index_idx,
//    const std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
    MLIRContext *context,
    mlir::IRRewriter &rewriter,
    mlir::Location &loc)
{
  llvm::SmallVector<Value> pos;
  llvm::SmallVector<Value> crds;
  Value prev_dim = nullptr;
  auto access_type = rewriter.getIndexType();
  llvm::SmallVector<DimCompound> &lhs_dims = itree_to_lhs_dims[tree_i];
  llvm::SmallVector<uint32_t> &lhs_index_idx = itree_to_lhs_index_idx[tree_i];
  uint32_t dim_base = 0;
//  if (itree_to_common_indices.find(tree_i + 1) != itree_to_common_indices.end()) {
//    dim_base = itree_to_common_indices.at(tree_i + 1).size();
//  }
  for (uint32_t d_i = dim_base; d_i < lhs_dims.size(); ++d_i)
  {
    Value index_node = index_ops[lhs_index_idx[d_i]];
    uint32_t dim = lhs_dims[d_i].first - dim_base;
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
//    const std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
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
    uint32_t dim_base = 0;
    if (rhs_i == intermediate_idx) {
      /// This RHS operand is an intermediate tensor
      rhs_tensor = prev_computeOp;
//      if (itree_to_common_indices.find(tree_i) != itree_to_common_indices.end()) {
//        dim_base = itree_to_common_indices.at(tree_i).size();
//      }
    }
    for (uint32_t d_i = dim_base; d_i < rhs_dims.size(); ++d_i) {
      Value index_node = index_ops[rhs_index_idx[d_i]];
      uint32_t dim = rhs_dims[d_i].first - dim_base;
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


Value createComputeOp(
    uint32_t tree_i,
    mlir::Type &tensor_type,
    llvm::SmallVector<Value> &index_ops,
    Value lhs_operand,
    llvm::SmallVector<Value> &rhs_operands,
    const llvm::SmallVector<mlir::StringRef> &itree_to_semiring,
    const llvm::SmallVector<bool> &itree_to_compute_missing,
    mlir::IRRewriter &rewriter,
    mlir::Location &loc)
{
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

  return compute_op;
}

[[maybe_unused]] Value createComputeOpReset(
    uint32_t tree_i,
    uint32_t intermediate_idx,
    const mlir::Type &tensor_type,
    const mlir::Type &element_type,
    const llvm::SmallVector<Value> &index_ops,
    Value lhs_tensor,
    const std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<DimCompound>>> &itree_to_rhs_dims,
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<uint32_t>>> &itree_to_rhs_index_idx,
    MLIRContext *context,
    mlir::IRRewriter &rewriter,
    mlir::Location &loc)
{
  indexTree::OperandType operand_type = indexTree::OperandType::get(context);
  /// Create LHS operand
  llvm::SmallVector<Value> pos;
  llvm::SmallVector<Value> crds;
  Value prev_dim = nullptr;
  auto access_type = rewriter.getIndexType();
  llvm::SmallVector<DimCompound> &lhs_dims = itree_to_rhs_dims[tree_i][intermediate_idx];
  llvm::SmallVector<uint32_t> &lhs_index_idx = itree_to_rhs_index_idx[tree_i][intermediate_idx];
  assert(itree_to_common_indices.find(tree_i) != itree_to_common_indices.end() && "Expect common indices.");
  uint32_t dim_base = itree_to_common_indices.at(tree_i).size();
  for (uint32_t d_i = dim_base; d_i < lhs_dims.size(); ++d_i) {
    Value index_node = index_ops[lhs_index_idx[d_i]];
    uint32_t dim = lhs_dims[d_i].first - dim_base;
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
  Value lhs_operand = rewriter.create<indexTree::IndexTreeLHSOperandOp>(loc,
                                                                        operand_type,
                                                                        lhs_tensor,
                                                                        pos,
                                                                        crds);
  /// Create RHS operand (constant 0)
  mlir::TypedAttr zero = rewriter.getZeroAttr(element_type);
  Value cst_0 = rewriter.create<arith::ConstantOp>(loc,
                                                   element_type,
                                                   zero);
  Value rhs_operand = rewriter.create<indexTree::IndexTreeOperandOp>(loc,
                                                                     operand_type,
                                                                     /*rhs_tensor*/cst_0,
                                                                     /*pos*/ValueRange{},
                                                                     /*crds*/ValueRange{});

  /// Create Compute Op for resetting
  Value parent;
  if (dim_base < lhs_dims.size()) {
    /// If the intermediate tensor is still a tensor, link to the last remaining dimension (index)
    /// For example, lhs_index_idx could be {1, 0}, but the innermost index should be 1. Thus we need to sort it at first.
    llvm::SmallVector<uint32_t> copy(lhs_index_idx);
    std::sort(copy.begin(), copy.end());
    parent = index_ops[copy.back()];
  } else {
    /// If the intermediate tensor is a scalar, link to the last common index
    parent = index_ops[itree_to_common_indices.at(tree_i).back()];
  }
  mlir::StringRef semiring("noop_times");
  bool compute_missing = false;
  Value compute_op = rewriter.create<indexTree::IndexTreeComputeOp>(
      loc,
      tensor_type,
      parent,
      lhs_operand,
      ValueRange{rhs_operand},
      /*mask_operand*/nullptr,
      rewriter.getStringAttr(semiring),
      rewriter.getBoolAttr(compute_missing));

  return compute_op;
}


void fuseITrees(IndexTreeOp new_itree,
                uint32_t num_itrees,
                const llvm::SmallVector<Type> &tree_types,
                const llvm::SmallVector<uint32_t> &itree_to_num_indexOps,
                std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
                const std::unordered_map<uint32_t, uint32_t> &itree_to_intermediate_idx,
                llvm::SmallVector<llvm::SmallVector<DimCompound>> &itree_to_lhs_dims,
                llvm::SmallVector<llvm::SmallVector<uint32_t>> &itree_to_lhs_index_idx,
                llvm::SmallVector<llvm::SmallVector<Value>> &itree_to_rhs_tensors,
                llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<DimCompound>>> &itree_to_rhs_dims,
                llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<uint32_t>>> &itree_to_rhs_index_idx,
                const llvm::SmallVector<mlir::StringRef> &itree_to_semiring,
                const llvm::SmallVector<bool> &itree_to_compute_missing,
                llvm::SmallVector<Value> &itree_to_new_compute_op,
                MLIRContext *context,
                mlir::IRRewriter &rewriter,
                mlir::Location &loc)
{
  /// Get all IndexOp of the new itree
  llvm::SmallVector<IndexTreeIndicesOp> host_index_ops;
  new_itree.walk([&](IndexTreeIndicesOp op) {
    host_index_ops.push_back(op);
  });
  

  indexTree::YieldOp yield_op = llvm::cast<indexTree::YieldOp>(new_itree.getRegion().getBlocks().front().getTerminator());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(yield_op);
  /// TODO: Parallel execution does not seem to work properly when fusion is in place...
  // for(auto indexOp : host_index_ops)
  // {
  //   indexOp.setIsParallel(false);
  // }
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
    /// An itree's arguments are it.itree($inputs, $intermediates)
//    Value lhs_tensor = new_itree.getRegion().getBlocks().front().getArgument(tree_i);
    Value lhs_tensor;
    if (tree_i == num_itrees - 1) {
      /// It is the final itree, generating the output
      lhs_tensor = new_itree.getRegion().getBlocks().front().getArgument(0);
    } else {
      /// It is to generate an intermediate
      uint32_t num_inputs = llvm::SmallVector<Value, 4>(new_itree.getInputs()).size();
      lhs_tensor = new_itree.getRegion().getBlocks().front().getArgument(num_inputs + tree_i);
    }
    Value lhs_operand = createLHSOperand(tree_i,
                                         index_ops,
                                         lhs_tensor,
                                         itree_to_lhs_dims,
                                         itree_to_lhs_index_idx,
//                                         itree_to_common_indices,
                                         context,
                                         rewriter,
                                         loc);
    comet_vdump(new_itree);

    /// Create RHS
    /// After fusion, one rhs operand should come from the intermediate variable.
    Value prev_computeOp = itree_to_new_compute_op[tree_i - 1];  /// the previous ComputeOp
    uint32_t intermediate_idx = itree_to_intermediate_idx.at(tree_i);  /// Which rhs operand is from the previous lhs operand
    llvm::SmallVector<Value> rhs_operands = createRHSOperands(tree_i,
                                                              intermediate_idx,
                                                              prev_computeOp,
                                                              index_ops,
                                                              itree_to_rhs_tensors,
                                                              itree_to_rhs_dims,
                                                              itree_to_rhs_index_idx,
//                                                              itree_to_common_indices,
                                                              context,
                                                              rewriter,
                                                              loc);
    comet_vdump(new_itree);

    /// Create Compute Ops
    auto tensor_type = tree_types[tree_i];
    Value compute_op = createComputeOp(tree_i,
                                       tensor_type,
                                       index_ops,
                                       lhs_operand,
                                       rhs_operands,
                                       itree_to_semiring,
                                       itree_to_compute_missing,
                                       rewriter,
                                       loc);
    itree_to_new_compute_op.push_back(compute_op);
    comet_vdump(new_itree);
    comet_debug() << "\n";

//    /// Create Compute Op to reset the intermediate tensor to 0
//    /// The element type of the intermediate tensor.
//    mlir::Type element_type =
//        llvm::cast<mlir::TensorType>(itree_to_rhs_tensors[tree_i][intermediate_idx].getType()).getElementType();
//    Value compute_op_reset = createComputeOpReset(
//        tree_i,
//        intermediate_idx,
//        /*tensor_type=*/tree_types[tree_i - 1],
//        /*element_type=*/element_type,
//        index_ops,
//        /*lhs_tensor=*/itree_to_new_compute_op[tree_i - 1],
//        itree_to_common_indices,
//        itree_to_rhs_dims,
//        itree_to_rhs_index_idx,
//        context,
//        rewriter,
//        loc);
//    itree_to_new_compute_op[tree_i - 1] = compute_op_reset;  /// Will be operands of the YieldOp.
  }

  /// Create yield op
  assert(itree_to_new_compute_op.size() == num_itrees && "Expect each itree has one ComputeOp.");
  llvm::SmallVector<Value> yieldOpArgs;
  /// Inputs go first
  yieldOpArgs.push_back(itree_to_new_compute_op.back());
  /// Then are intemediates
  yieldOpArgs.insert(yieldOpArgs.end(), itree_to_new_compute_op.begin(), itree_to_new_compute_op.begin() + num_itrees - 1);

  rewriter.create<indexTree::YieldOp>(loc, TypeRange(), yieldOpArgs);
  rewriter.eraseOp(yield_op);
  comet_vdump(new_itree);
}


void createITree(
    uint32_t num_itrees,
//    const llvm::SmallVector<Value> &itree_arguments,
    const llvm::SmallVector<Value> &itree_arguments_inputs,
    const llvm::SmallVector<Value> &itree_arguments_intermediates,
    llvm::SmallVector<IndexTreeOp> &itree_list,
    const llvm::SmallVector<uint32_t> &itree_to_num_indexOps,
    std::unordered_map<uint32_t, llvm::SmallVector<uint32_t>> &itree_to_common_indices,
    const std::unordered_map<uint32_t, uint32_t> &itree_to_intermediate_idx,
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
//  for (Value arg : itree_arguments) {
//    tree_types.push_back(arg.getType());
//  }
  for (Value arg : itree_arguments_inputs) {
    tree_types.push_back(arg.getType());
  }
  for (Value arg : itree_arguments_intermediates) {
    tree_types.push_back(arg.getType());
  }

  auto loc = host_itree->getLoc();

  /// Create the new itree.
  llvm::SmallVector<Value> itree_to_new_compute_op;
  rewriter.setInsertionPointAfter(itree_list.back());
  IndexTreeOp new_itree = createNewITree(host_itree,
                                         tree_types,
//                                         itree_arguments,
                                         itree_arguments_inputs,
                                         itree_arguments_intermediates,
                                         itree_to_lhs_dims,
                                         itree_to_lhs_index_idx,
//                                         itree_to_common_indices,
                                         itree_to_new_compute_op /*out*/,
                                         context,
                                         rewriter,
                                         loc);
  /// Fuse other itrees to the new itree
  fuseITrees(new_itree,
             num_itrees,
             tree_types,
             itree_to_num_indexOps,
             itree_to_common_indices,
             itree_to_intermediate_idx,
             itree_to_lhs_dims,
             itree_to_lhs_index_idx,
             itree_to_rhs_tensors,
             itree_to_rhs_dims,
             itree_to_rhs_index_idx,
             itree_to_semiring,
             itree_to_compute_missing,
             itree_to_new_compute_op,
             context,
             rewriter,
             loc);

  /// Update the usage of results of itree
  /// Erase uses of output of kernels other than the last kernel

  /// Replace the use of output of the last kernel
  // uint32_t new_r_i = 0;
  // uint32_t tree_i = num_itrees - 1;
  for(auto it : enumerate(ArrayRef(itree_list)))
  {
    auto index = itree_list.size() - 1 - it.index();
    auto itree_to_remove = it.value();
    rewriter.replaceUsesWithIf(
        itree_to_remove.getResults(),
        new_itree.getResults()[index],  
        [&](OpOperand &use) {
          auto user = use.getOwner();
          auto ancestor = itree_to_remove->getBlock()->findAncestorOpInBlock(*user);
          return (ancestor && new_itree->isBeforeInBlock(ancestor)); 
    });
        
    rewriter.replaceOp(itree_to_remove, new_itree->getOperand(index));
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

//  /// Collect all itrees' arguments
//  llvm::SmallVector<Value> itree_arguments;
//  for (IndexTreeOp &itree : itree_list) {
//    for (Value arg : itree.getInputs()) {
//      comet_vdump(arg);
//      itree_arguments.push_back(arg);
//    }
//  }
  /// Collect all itrees' arguments. The last itree's inputs will be the final `inputs`. Other itrees' inputs will be the
  /// final `intermediates`. The new fused itree's arguments will be it.itree($inputs, $intermediates).
  llvm::SmallVector<Value> itree_arguments_inputs;
  llvm::SmallVector<Value> itree_arguments_intermediates;
  for (uint32_t tree_i = 0; tree_i < num_itrees - 1; ++tree_i) {
    IndexTreeOp &itree = itree_list[tree_i];
    for (Value arg : itree.getInputs()) {
      itree_arguments_intermediates.push_back(arg);
    }
  }
  for (Value arg : itree_list.back().getInputs()) {
    itree_arguments_inputs.push_back(arg);
  }


  /// Collect all LHS and RHS operands

  /// `itree_to_lhs_dims`: which <dim, Tensor> is used as a LHSOperandOp's dimensions.
  /// `itree_to_lhs_index_idx`: which IndexOp is linked by a LHSOperandOp's dimensions. IndexOp are ordered starting from 0.
  /// For example,
  /*
    %11 = "it.itree"(%10) ({
    ^bb0(%arg0: tensor<?x4xf64>):
      %13 = "it.RootOp"() : () -> !it.index_tree
      %14 = "it.IndexOp"(%13) : (!it.index_tree) -> !it.index  /// %14 = h
      %15 = "it.IndexOp"(%14) : (!it.index) -> !it.index  /// %15 = i
      %16 = "it.IndexOp"(%15) : (!it.index) -> !it.index  /// %16 = k
      %crd, %pos = "it.IndexToTensorDim"(%arg0, %15) <{dim = 0 : ui32}> : (tensor<?x4xf64>, !it.index) -> (index, index)
      %crd_0, %pos_1 = "it.IndexToTensorDim"(%arg0, %14, %pos) <{dim = 1 : ui32}> : (tensor<?x4xf64>, !it.index, index) -> (index, index)
      %17 = "it.LHSOperandOp"(%arg0, %pos, %pos_1, %crd, %crd_0) : (tensor<?x4xf64>, index, index, index, index) -> !it.operand
      %crd_2, %pos_3 = "it.IndexToTensorDim"(%4, %15) <{dim = 0 : ui32}> : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>, !it.index) -> (index, index)

      // ...
    }) : (tensor<?x4xf64>) -> tensor<?x4xf64>
   */
  /// IndexOps %14, %15, and %16 are referred as 0, 1, and 2.
  /// itree_to_lhs_dims[0]: {<0, %10>, <1, %10>}. %10 is the real tensor, not %arg0.
  /// itree_to_lhs_index_idx[0]: {1, 0}. because %15 is the 1st IndexOp, and %14 is the 0th IndexOp.
  /// the rhs counterparts have the same behavior.
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

  /// Record which rhs operand is from the previous lhs operand
  /// For example,
  /*
    T[i, h] += B[i, k] * C[k, h];  // the 0th itree
    A[i, j] += T[i, h] * D[h, j];  // the 1st itree
   */
  /// itree_to_intermediate_idx[1] = 0, because the 1st itree has T as its 0st rhs operand that is the previous lhs operand.
  std::unordered_map<uint32_t, uint32_t> itree_to_intermediate_idx;
  for (uint32_t tree_i = 1; tree_i < num_itrees; ++tree_i) {
    itree_to_intermediate_idx[tree_i] = collectIntermediateIdx(itree_to_lhs_tensors[tree_i - 1] /*previous lhs*/,
                                                               itree_to_rhs_tensors[tree_i] /*current rhs*/);
  }

//  /// Create new lhs tensors with decreased dimensions
//  std::unordered_map<uint32_t, Value> itree_to_new_lhs_tensors =
//      createNewLhsTensors(num_itrees,
//                          itree_to_lhs_tensors,
//                          itree_to_common_indices);
//  /// Update itrees' arguments
//  for (uint32_t tree_i = 0; tree_i < num_itrees - 1; ++tree_i) {
//    if (itree_to_new_lhs_tensors.find(tree_i) != itree_to_new_lhs_tensors.end()) {
//      itree_arguments[tree_i] = itree_to_new_lhs_tensors[tree_i];
//    }
//  }

  /// Collect ComputeOp's information: semiring, and compute_missing.
  llvm::SmallVector<mlir::StringRef> itree_to_semiring(num_itrees);
  llvm::SmallVector<bool> itree_to_compute_missing(num_itrees, false);
  for (uint32_t tree_i = 0; tree_i < num_itrees; ++tree_i) {
    collectComputeOpInfo(itree_list[tree_i],
                         itree_to_semiring[tree_i] /*out*/,
                         itree_to_compute_missing[tree_i] /*out*/);
  }

  /// Build the new itree
  createITree(num_itrees,
//              itree_arguments,
              itree_arguments_inputs,
              itree_arguments_intermediates,
              itree_list,
              itree_to_num_indexOps,
              itree_to_common_indices,
              itree_to_intermediate_idx,
              itree_to_lhs_dims,
              itree_to_lhs_index_idx,
              itree_to_rhs_tensors,
              itree_to_rhs_dims,
              itree_to_rhs_index_idx,
              itree_to_semiring,
              itree_to_compute_missing);

//  /// Remove old lhs tensors
//  for (uint32_t tree_i = 0; tree_i < num_itrees - 1; ++tree_i) {
//    if (itree_to_new_lhs_tensors.find(tree_i) != itree_to_new_lhs_tensors.end()) {
//      Value old_tensor = itree_to_lhs_tensors[tree_i];
//      for (auto user : old_tensor.getUsers()) {
//        user->erase();
//      }
//      old_tensor.getDefiningOp()->erase();
//    }
//  }

  comet_vdump(funcOp->getParentOfType<ModuleOp>());
}


/// Apply the redundancy-aware kernel fusion on index tree dialect for some compound expressions
std::unique_ptr<Pass> mlir::comet::createIndexTreeKernelFusionPass()
{
  return std::make_unique<IndexTreeKernelFusion>();
}