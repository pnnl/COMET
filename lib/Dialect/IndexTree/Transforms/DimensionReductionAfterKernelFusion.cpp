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

#include <unordered_map>

using namespace mlir;
using namespace mlir::indexTree;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
// *********** For debug purpose *********//

namespace mlir {
  namespace comet{
#define GEN_PASS_DEF_INDEXTREEDIMENSIONREDUCTION
#include "comet/Dialect/IndexTree/Passes.h.inc"
  }
}

namespace {

struct ITreeInfo {
  IndexTreeOp itree_op;
  llvm::SmallVector<Value> args_inputs;
  llvm::SmallVector<Value> args_intermediates;
  llvm::SmallVector<Value> block_args;
  llvm::SmallVector<Type> args_types;
};

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

void dfs_it(IndexTreeIndicesOp node,
            llvm::SmallVector<Value> &path,
            llvm::DenseMap<Value, llvm::SmallVector<Value>> &all_paths)
{
  path.push_back(node);
  for (auto child : node->getUsers()) {
    if (auto computeOp = llvm::dyn_cast<IndexTreeComputeOp>(child)) {
      all_paths[computeOp] = path;
    } else if (auto indexOp = llvm::dyn_cast<IndexTreeIndicesOp>(child)) {
      dfs_it(indexOp, path, all_paths);
    }
  }
  path.pop_back();
}

llvm::DenseMap<Value, llvm::SmallVector<Value>> getAllPathsToComputeOp(func::FuncOp funcOp)
{
  IndexTreeRootOp rootOp;
  funcOp.walk([&](IndexTreeRootOp op) {
    rootOp = op;
  });

  uint32_t count_child = 0;
  IndexTreeIndicesOp child;
  for (auto user : rootOp->getUsers()) {
    ++count_child;
    child = llvm::cast<IndexTreeIndicesOp>(user);
  }
  assert(count_child == 1 && "No common IndexOp nodes are shared by ComputeOp nodes.");
  llvm::DenseMap<Value, llvm::SmallVector<Value>> all_paths;
  llvm::SmallVector<Value> curr_path;
  dfs_it(child, curr_path, all_paths);
//  {/// test
//    for (const auto &[computeOp, path] : all_paths) {
//      comet_debug() << "a path for \n";
//      comet_vdump(computeOp);
//      for (const auto index : path) {
//        comet_vdump(index);
//      }
//    }
//  }
  return all_paths;
}

llvm::SmallVector<Value> getCommonIndices(Value computeOp1,
                                          Value computeOp2,
                                          const llvm::DenseMap<Value, llvm::SmallVector<Value>> &all_paths)
{
  const llvm::SmallVector<Value> &path1 = all_paths.at(computeOp1);
  const llvm::SmallVector<Value> &path2 = all_paths.at(computeOp2);
  llvm::SmallVector<Value> common_indices;
  uint32_t num = 0;
  while (num < path1.size() && num < path2.size()) {
    if (path1[num] == path2[num]) {
      common_indices.push_back(path1[num]);
      ++num;
    } else {
      break;
    }
  }

  return common_indices;
}

std::unordered_map<uint32_t, llvm::SmallVector<Value>>
getAllComputeOpsCommonIndices(func::FuncOp funcOp,
                                   const llvm::SmallVector<Value> &computeOps)
{
  llvm::DenseMap<Value, llvm::SmallVector<Value>> computeOp_to_index_path = getAllPathsToComputeOp(funcOp);
  std::unordered_map<uint32_t, llvm::SmallVector<Value>> computeOp_to_common_indices;

  for (uint32_t computeOp_i = 0; computeOp_i < computeOps.size() - 1; ++computeOp_i) {
    computeOp_to_common_indices[computeOp_i + 1] = getCommonIndices(computeOps[computeOp_i],
                                                                    computeOps[computeOp_i + 1],
                                                                    computeOp_to_index_path);
  }
//  {/// test
//    for (const auto [computeOp_i, indices] : computeOp_to_common_indices) {
//      comet_debug() << "computeOp_i: " << computeOp_i << " num_common_indices: " << indices.size() << "\n";
//    }
//  }

  return computeOp_to_common_indices;
}

Value getOldLhsTensor(Value computeOp,
                      ITreeInfo &itreeInfo)
{
  IndexTreeComputeOp cmptOp = llvm::cast<IndexTreeComputeOp>(computeOp.getDefiningOp());
  Value lhs_tensor =  llvm::cast<IndexTreeLHSOperandOp>(cmptOp.getLhs().getDefiningOp()).getTensor();

  uint32_t idx = getIdxInVector(itreeInfo.block_args, lhs_tensor);
  if (idx != (uint32_t) -1) {
    if (idx < itreeInfo.args_inputs.size()) {
      lhs_tensor = itreeInfo.args_inputs[idx];
    } else {
      lhs_tensor = itreeInfo.args_intermediates[idx - itreeInfo.args_inputs.size()];
    }
  } else {
    assert(false && "Expect to find the lhs tensor in itree's block arguments.");
  }

  return lhs_tensor;
}


std::unordered_map<uint32_t, Value> createNewLhsTensors(
    llvm::SmallVector<Value> &computeOps,
    ITreeInfo &itreeInfo,
    std::unordered_map<uint32_t, llvm::SmallVector<Value>> &computeOp_to_common_indices,
    mlir::IRRewriter &rewriter,
    mlir::Location &loc)
{
  std::unordered_map<uint32_t, Value> computeOp_to_new_tensors;
  for (uint32_t computeOp_i = 0; computeOp_i < computeOps.size() - 1; ++computeOp_i) {
    Value computeOp = computeOps[computeOp_i];
    Value old_tensor = getOldLhsTensor(computeOp, itreeInfo);
    uint32_t num_common_indices = computeOp_to_common_indices[computeOp_i + 1].size();
    assert(llvm::isa<mlir::TensorType>(old_tensor.getType()) &&
        "Expect a mlir::TensorType.");
    assert(num_common_indices && "Error: number of common indices is zero");
    auto old_tensor_tt = llvm::cast<mlir::TensorType>(old_tensor.getType());

    /// The start dim idx after those common indices
    uint32_t dim_base = num_common_indices;
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

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(old_tensor.getDefiningOp());
      Value new_tensor = rewriter.create<tensorAlgebra::DenseTensorDeclOp>(
          loc,
          mlir::RankedTensorType::get(shape, old_tensor_tt.getElementType()),
          operands,
          llvm::cast<tensorAlgebra::DenseTensorDeclOp>(old_tensor.getDefiningOp()).getFormatAttr());
      computeOp_to_new_tensors[computeOp_i] = new_tensor;
      mlir::TypedAttr zero = rewriter.getZeroAttr(old_tensor_tt.getElementType());
      rewriter.create<tensorAlgebra::TensorFillOp>(loc,
                                                  new_tensor,
                                                  zero);
      comet_vdump(new_tensor);
    } else {
      /// The new tensor is shrunk to a scalar.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(old_tensor.getDefiningOp());
      mlir::TypedAttr zero = rewriter.getZeroAttr(old_tensor_tt.getElementType());
      Value new_scalar = rewriter.create<arith::ConstantOp>(loc,
                                                           old_tensor_tt.getElementType(),
                                                           zero);
      comet_vdump(new_scalar);
      computeOp_to_new_tensors[computeOp_i] = new_scalar;
    }
  }

  return computeOp_to_new_tensors;
}

ITreeInfo createNewItreeOp(uint32_t num_computeOps,
                           ITreeInfo &oldITreeInfo,
                           const std::unordered_map<uint32_t, Value> &computeOp_to_new_tensors,
                           mlir::IRRewriter &rewriter,
                           mlir::Location &loc)
{
  IndexTreeOp old_itree = oldITreeInfo.itree_op;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(old_itree);
  llvm::SmallVector<Value> args_inputs = oldITreeInfo.args_inputs;
  llvm::SmallVector<Value> args_intermediates;
  for (uint32_t computeOp_i = 0; computeOp_i < num_computeOps - 1; ++computeOp_i) {
    args_intermediates.push_back(computeOp_to_new_tensors.at(computeOp_i));
  }

  llvm::SmallVector<Location> locs;
  llvm::SmallVector<Type> args_types;
  for (Value arg : args_inputs) {
    locs.push_back(arg.getLoc());
    args_types.push_back(arg.getType());
  }
  for (Value arg : args_intermediates) {
    locs.push_back(arg.getLoc());
    args_types.push_back(arg.getType());
  }
  IndexTreeOp new_itree = rewriter.create<indexTree::IndexTreeOp>(loc,
                                                                  args_types,
                                                                  args_inputs,
                                                                  args_intermediates);
  Region *body = &new_itree.getRegion();
  Block *block = rewriter.createBlock(body, {}, TypeRange(args_types), locs);
  YieldOp dummy_yield = rewriter.create<indexTree::YieldOp>(loc, TypeRange(), args_inputs);  /// create a dummy YieldOp to make the following inlineBlockBefore() work.
  /// Move the old itree's block to the new itree.
  rewriter.inlineBlockBefore(&old_itree.getRegion().front(),
                             block->getTerminator(),
                             block->getArguments());
  rewriter.eraseOp(dummy_yield);
  assert(old_itree.getNumResults() == new_itree.getNumResults() &&
    "Expect old itree and new itree have the same number of inputs and outputs.");
  for (uint32_t r_i = 0; r_i < old_itree.getNumResults(); ++r_i) {
    rewriter.replaceAllUsesWith(old_itree.getResult(r_i), new_itree.getResult(r_i));
  }
  rewriter.eraseOp(old_itree);
  comet_vdump(new_itree);
  comet_vdump(new_itree->getParentOfType<ModuleOp>());

  ITreeInfo newITreeInfo;
  newITreeInfo.itree_op = new_itree;
  newITreeInfo.args_inputs = args_inputs;
  newITreeInfo.args_intermediates = args_intermediates;
  for (Value arg : block->getArguments()) {
    newITreeInfo.block_args.push_back(arg);
  }
  newITreeInfo.args_types = args_types;

  return newITreeInfo;
}

/// Create the new LHS operand and remove the old one.
Value createNewLHSOperandOp(Value computeOp,
                            uint32_t num_common_indices,
                            Value lhs_tensor,
                            mlir::IRRewriter &rewriter,
                            mlir::Location &loc)
{
  /// Find the old LHSOperandOp
  IndexTreeLHSOperandOp old_lhs_operand_op = llvm::cast<IndexTreeLHSOperandOp>(
      llvm::cast<IndexTreeComputeOp>(computeOp.getDefiningOp()).getLhs().getDefiningOp());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(old_lhs_operand_op);

  /// Create the new LHSOperandOp
  llvm::SmallVector<Value> pos;
  llvm::SmallVector<Value> crds;
  Value prev_dim = nullptr;
  auto access_type = rewriter.getIndexType();
  uint32_t dim_i = 0;
  for (Value posOp : old_lhs_operand_op.getPos()) {
    if (dim_i < num_common_indices) {
      ++dim_i;
      continue;
    }
    IndexTreeIndexToTensorOp tensorDimOp = llvm::cast<IndexTreeIndexToTensorOp>(posOp.getDefiningOp());
    Value index_node = tensorDimOp.getIndex();
    uint32_t dim = tensorDimOp.getDim() - num_common_indices;
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

  indexTree::OperandType operand_type = indexTree::OperandType::get(rewriter.getContext());
  Value new_lhs_operand = rewriter.create<indexTree::IndexTreeLHSOperandOp>(loc,
                                                                            operand_type,
                                                                            lhs_tensor,
                                                                            pos,
                                                                            crds);
  comet_vdump(new_lhs_operand);
  comet_vdump(new_lhs_operand.getDefiningOp()->getParentOfType<ModuleOp>());

  /// Replace and erase
  rewriter.replaceAllUsesWith(old_lhs_operand_op, new_lhs_operand);
  llvm::SmallVector<Value> old_index_to_tensor_dim;
  for (auto tensor_dim : old_lhs_operand_op.getPos()) {
    old_index_to_tensor_dim.push_back(tensor_dim);
  }
  std::reverse(old_index_to_tensor_dim.begin(), old_index_to_tensor_dim.end());
  rewriter.eraseOp(old_lhs_operand_op);
  for (auto tensor_dim : old_index_to_tensor_dim) {
    rewriter.eraseOp(tensor_dim.getDefiningOp());
  }

  comet_vdump(new_lhs_operand.getDefiningOp()->getParentOfType<ModuleOp>());
  return new_lhs_operand;
}

Value createNewComputeOp(Value computeOp,
                         const mlir::Type &tensor_type,
                         Value lhsOperandOp,
                         ValueRange rhsOperandOps,
                         mlir::IRRewriter &rewriter,
                         mlir::Location &loc)
{
  IndexTreeComputeOp old_compute_op = llvm::cast<IndexTreeComputeOp>(computeOp.getDefiningOp());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(computeOp.getDefiningOp());
  Value new_compute_op = rewriter.create<indexTree::IndexTreeComputeOp>(
      loc,
      tensor_type,
      old_compute_op.getParent(),
      lhsOperandOp,
      rhsOperandOps,
      old_compute_op.getMask(),
      old_compute_op.getSemiringAttr(),
      old_compute_op.getComputeMissingAttr());

//  rewriter.replaceAllUsesWith(old_compute_op, new_compute_op);
//  rewriter.eraseOp(old_compute_op);

  comet_vdump(new_compute_op);
  comet_vdump(new_compute_op.getDefiningOp()->getParentOfType<ModuleOp>());
  return new_compute_op;
}

/// Create the new rhs oprand for the new intermeidate tensor, then return it along with the other old rhs operand.
llvm::SmallVector<Value> createNewRHSOperandOps(Value prev_old_compute_op,
                                                Value curr_old_compute_op,
                                                uint32_t num_common_indices,
                                                Value intermediate_tensor,
                                                IndexTreeOperandOp &new_rhs_operand_op /*out*/,
                                                mlir::IRRewriter &rewriter,
                                                mlir::Location &loc)
{
  /// Find the intermediate idx
  llvm::SmallVector<Value> old_rhs_operand_ops =
      llvm::cast<IndexTreeComputeOp>(curr_old_compute_op.getDefiningOp()).getRhs();
  uint32_t intermediate_idx = 0;
  IndexTreeOperandOp old_rhs_operand_op = nullptr;
  while (intermediate_idx < old_rhs_operand_ops.size()) {
    IndexTreeOperandOp operandOp =
        llvm::cast<IndexTreeOperandOp>(old_rhs_operand_ops[intermediate_idx].getDefiningOp());
    if (operandOp.getTensor() == prev_old_compute_op) {
      old_rhs_operand_op = operandOp;
      break;  /// Found the intermediate tensor
    }
    ++intermediate_idx;
  }
  assert(intermediate_idx < old_rhs_operand_ops.size() && "Error: not found the intermediate tensor.");

  /// Create the new rhs operand
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(old_rhs_operand_op);
  llvm::SmallVector<Value> pos;
  llvm::SmallVector<Value> crds;
  Value prev_dim = nullptr;
  auto access_type = rewriter.getIndexType();
  uint32_t dim_i = 0;
  for (Value posOp : old_rhs_operand_op.getPos()) {
    if (dim_i < num_common_indices) {
      ++dim_i;
      continue;
    }
    IndexTreeIndexToTensorOp tensorDimOp = llvm::cast<IndexTreeIndexToTensorOp>(posOp.getDefiningOp());
    Value index_node = tensorDimOp.getIndex();
    uint32_t dim = tensorDimOp.getDim() - num_common_indices;
    auto access_op = rewriter.create<indexTree::IndexTreeIndexToTensorOp>(
        loc,
        TypeRange({access_type, access_type}),
        /*lhs_tensor=*/intermediate_tensor,
        index_node,
        rewriter.getUI32IntegerAttr(dim),
        prev_dim);
    pos.push_back(access_op.getPos());
    crds.push_back(access_op.getCrd());
    prev_dim = pos.back();
  }
  indexTree::OperandType operand_type = indexTree::OperandType::get(rewriter.getContext());
  new_rhs_operand_op = rewriter.create<indexTree::IndexTreeOperandOp>(loc,
                                                                            operand_type,
                                                                            intermediate_tensor,
                                                                            pos,
                                                                            crds);
  comet_vdump(new_rhs_operand_op);
  comet_vdump(new_rhs_operand_op->getParentOfType<ModuleOp>());

  /// Replace and erase
  rewriter.replaceAllUsesWith(old_rhs_operand_op, new_rhs_operand_op);
  llvm::SmallVector<Value> old_index_to_tensor_dim;
  for (auto tensor_dim : old_rhs_operand_op.getPos()) {
    old_index_to_tensor_dim.push_back(tensor_dim);
  }
  std::reverse(old_index_to_tensor_dim.begin(), old_index_to_tensor_dim.end());
  rewriter.eraseOp(old_rhs_operand_op);
  for (auto tensor_dim : old_index_to_tensor_dim) {
    rewriter.eraseOp(tensor_dim.getDefiningOp());
  }

  /// Collect all rhs operands
  llvm::SmallVector<Value> rhs_operand_ops;
  for (uint32_t r_i = 0; r_i < old_rhs_operand_ops.size(); ++r_i) {
    if (r_i == intermediate_idx) {
      rhs_operand_ops.push_back(new_rhs_operand_op);
    } else {
      rhs_operand_ops.push_back(old_rhs_operand_ops[r_i]);
    }
  }

  return rhs_operand_ops;
}


llvm::SmallVector<Value> createExtraIndicesOpsForReset(Value prev_old_compute_op,
                               Value parent_index_op,
                               mlir::IRRewriter &rewriter,
                               mlir::Location &loc)
{
  /// Get the domain from previous ComputeOP
  /// Domain: previous ComputeOp -> LHSOperandOp -> IndexToTensorDim -> IndexOp -> DenseDomainOp
  auto prev_compute_op = llvm::cast<IndexTreeComputeOp>(prev_old_compute_op.getDefiningOp());
  auto lhs_operand_op = llvm::cast<IndexTreeLHSOperandOp>(prev_compute_op.getLhs().getDefiningOp());
  llvm::SmallVector<IndexTreeIndexToTensorOp> index_to_tensor_dim_ops;
  for (Value pos : lhs_operand_op.getPos()) {
    index_to_tensor_dim_ops.push_back(llvm::cast<IndexTreeIndexToTensorOp>(pos.getDefiningOp()));
  }
  llvm::SmallVector<IndexTreeIndicesOp> indices_ops;
  for (auto index_to_tensor_dim : index_to_tensor_dim_ops) {
    indices_ops.push_back(llvm::cast<IndexTreeIndicesOp>(index_to_tensor_dim.getIndex().getDefiningOp()));
  }
  llvm::SmallVector<Value> domain_ops;
  for (auto index : indices_ops) {
    domain_ops.push_back(index.getDomain());
  }

  llvm::SmallVector<Value> new_indices_ops;
  Value parent = parent_index_op;
  for (Value domain : domain_ops) {
    Value new_index = rewriter.create<indexTree::IndexTreeIndicesOp>(loc,
                                                                     indexTree::IndexNodeType::get(rewriter.getContext()),
                                                                     parent,
                                                                     domain);
    new_indices_ops.push_back(new_index);
    parent = new_index;
    comet_vdump(new_index);
  }

  return new_indices_ops;
}

Value createComputeOpForReset(const llvm::SmallVector<Value> &common_indices,
                              Value prev_old_comput_op,
                              Value intermediate_tensor,
                              IndexTreeOperandOp new_rhs_operand_op,
                              mlir::Type element_type,
                              mlir::IRRewriter &rewriter,
                              mlir::Location &loc)
{
  uint32_t num_dims_new_tensor = 0;
  for (Value _ : new_rhs_operand_op.getPos()) {
    ++num_dims_new_tensor;
  }
  indexTree::OperandType operand_type = indexTree::OperandType::get(rewriter.getContext());
  Value parent;
  Value lhs_operand_op;
  if (num_dims_new_tensor)
  {
    /// If the intermediate tensor is still a tensor, generate extra indices (loops) to reset it.
    llvm::SmallVector<Value> new_indices_ops = createExtraIndicesOpsForReset(prev_old_comput_op,
                                                                        /*parent_index_op=*/common_indices.back(),
                                                                        rewriter,
                                                                        loc);
    assert(new_indices_ops.size() == num_dims_new_tensor && "Expect to reset the whole new tensor");
    parent = new_indices_ops.back();

    /// Create the new lhs operand to reset
    llvm::SmallVector<Value> pos;
    llvm::SmallVector<Value> crds;
    Value prev_dim = nullptr;
    auto access_type = rewriter.getIndexType();
    uint32_t index_i = 0;
    for (Value posOp : new_rhs_operand_op.getPos()) {
      IndexTreeIndexToTensorOp tensorDimOp = llvm::cast<IndexTreeIndexToTensorOp>(posOp.getDefiningOp());
//      Value index_node = tensorDimOp.getIndex();
      uint32_t dim = tensorDimOp.getDim();
      Value index_node = new_indices_ops[index_i++];
      auto access_op = rewriter.create<indexTree::IndexTreeIndexToTensorOp>(
          loc,
          TypeRange({access_type, access_type}),
          /*lhs_tensor=*/intermediate_tensor,
          index_node,
          rewriter.getUI32IntegerAttr(dim),
          prev_dim);
      pos.push_back(access_op.getPos());
      crds.push_back(access_op.getCrd());
      prev_dim = pos.back();
    }

    lhs_operand_op = rewriter.create<indexTree::IndexTreeLHSOperandOp>(
        loc,
        operand_type,
        intermediate_tensor,
        pos,
        crds);
  }
  else
  {
    /// If the intermediate tensor is a scalar, link to the last common index
    parent = common_indices.back();
    auto access_type = rewriter.getIndexType();
    lhs_operand_op = rewriter.create<indexTree::IndexTreeLHSOperandOp>(
        loc,
        operand_type,
        intermediate_tensor);
  }

  /// Create the rhs operand (i.e., constant 0)
  mlir::TypedAttr zero = rewriter.getZeroAttr(element_type);
  Value cst_0 = rewriter.create<arith::ConstantOp>(loc,
                                                   element_type,
                                                   zero);
  Value rhs_operand_op = rewriter.create<indexTree::IndexTreeOperandOp>(
      loc,
      operand_type,
      /*rhs_tensor*/cst_0,
      /*pos*/ValueRange{},
      /*crds*/ValueRange{});

  /// Create the Compute Op for resetting
  mlir::StringRef semiring("noop_times");
  bool compute_missing = false;
  Value compute_op = rewriter.create<indexTree::IndexTreeComputeOp>(
      loc,
      intermediate_tensor.getType(),
      parent,
      lhs_operand_op,
      ValueRange{rhs_operand_op},
      /*mask_operand*/nullptr,
      rewriter.getStringAttr(semiring),
      rewriter.getBoolAttr(compute_missing));

  return compute_op;
}

}  /// anonymous namespace

struct IndexTreeDimensionReduction : comet::impl::IndexTreeDimensionReductionBase<IndexTreeDimensionReduction> {
  using IndexTreeDimensionReductionBase::IndexTreeDimensionReductionBase;
  void runOnOperation() override;
};

void IndexTreeDimensionReduction::runOnOperation()
{
  comet_debug() << "IndexTreeDimensionReduction::runOnOperation()\n";
  func::FuncOp funcOp = getOperation();
  comet_vdump(funcOp->getParentOfType<ModuleOp>());

  /// Get all itree arguments
  ITreeInfo oldITreeInfo;
  uint32_t count_itrees = 0;
  funcOp.walk([&](IndexTreeOp itreeOp) {
    ++count_itrees;
    oldITreeInfo.itree_op = itreeOp;
    for (Value arg : itreeOp.getInputs()) {
      oldITreeInfo.args_inputs.push_back(arg);
    }
    for (Value arg : itreeOp.getIntermediates()) {
      oldITreeInfo.args_intermediates.push_back(arg);
    }
    for (Value arg : itreeOp.getRegion().getBlocks().front().getArguments()) {
      oldITreeInfo.block_args.push_back(arg);
    }
  });
  assert(count_itrees == 1 && "Expected one single fused itree.");

  /// Get all ComputeOp nodes
  llvm::SmallVector<Value> computeOps;
  funcOp.walk([&](IndexTreeComputeOp computeOp) {
    computeOps.push_back(computeOp);
    comet_vdump(computeOp);
  });
  uint32_t num_computeOps = computeOps.size();

  /// Get number of common indices for each two consecutive ComputeOp nodes.
  std::unordered_map<uint32_t, llvm::SmallVector<Value>> computeOp_to_common_indices =
      getAllComputeOpsCommonIndices(funcOp, computeOps);
  assert(computeOp_to_common_indices.size() == num_computeOps - 1 && "N ComputeOps expect N-1 intermediates.");

  /// Create new tensors with dimension reduction
  mlir::OpBuilder builder(oldITreeInfo.itree_op);
  mlir::IRRewriter rewriter(builder);
  Location loc = oldITreeInfo.itree_op.getLoc();
  std::unordered_map<uint32_t, Value> computeOp_to_new_tensors = createNewLhsTensors(computeOps,
                                                                            oldITreeInfo,
                                                                            computeOp_to_common_indices,
                                                                            rewriter,
                                                                            loc);

  /// Create a new itree to update the argument types, moving the body from old itree to the new itree
  ITreeInfo newITreeInfo = createNewItreeOp(num_computeOps,
                                            oldITreeInfo,
                                            computeOp_to_new_tensors,
                                            rewriter,
                                            loc);
  /// Dimension reduction
  llvm::SmallVector<Value> new_compute_ops;
  llvm::SmallVector<Value> new_compute_ops_for_reset;
  uint32_t intermediates_idx_base = newITreeInfo.args_inputs.size();
  for (uint32_t computeOp_i = 0; computeOp_i < num_computeOps; ++computeOp_i) {
    Value computeOp = computeOps[computeOp_i];
    if (computeOp_i == 0) {
      /// The first ComputeOp
      /// Only LHS needs dimension reduction
      Value lhs_operand_op = createNewLHSOperandOp(
          computeOp,
          /*num_common_indices=*/computeOp_to_common_indices[computeOp_i + 1].size(),
          /*lhs_tensor=*/newITreeInfo.block_args[intermediates_idx_base + computeOp_i],
          rewriter,
          loc);
      /// Create new ComputeOp
      auto rhs_operand_ops =
          llvm::cast<IndexTreeComputeOp>(computeOp.getDefiningOp()).getRhs();
      Value new_compute_op = createNewComputeOp(
          computeOp,
          /*output_tensor_type=*/newITreeInfo.args_types[intermediates_idx_base + computeOp_i],
          lhs_operand_op,
          rhs_operand_ops,
          rewriter,
          loc);
      new_compute_ops.push_back(new_compute_op);
    } else if (computeOp_i == num_computeOps - 1) {
      /// The last ComputeOp
      /// Only RHS needs dimension reduction
      uint32_t num_common_indices = computeOp_to_common_indices[computeOp_i].size();
      Value prev_new_compute_op = new_compute_ops.back();
      IndexTreeOperandOp new_rhs_operand_op;
      llvm::SmallVector<Value> rhs_operand_ops = createNewRHSOperandOps(
          /*prev_old_compute_op=*/computeOps[computeOp_i - 1],
          /*curr_old_compute_op=*/computeOps[computeOp_i],
          num_common_indices,
          /*intermediate_tensor=*/prev_new_compute_op,
          new_rhs_operand_op /*out*/,
          rewriter,
          loc);
      /// Create new ComputeOp
      auto lhs_operand_op =
          llvm::cast<IndexTreeComputeOp>(computeOp.getDefiningOp()).getLhs();
      Value new_compute_op = createNewComputeOp(
          computeOp,
          /*output_tensor_type=*/newITreeInfo.args_types[0]/*the input type*/,
          lhs_operand_op,
          rhs_operand_ops,
          rewriter,
          loc);
      new_compute_ops.push_back(new_compute_op);
      /// Create ComputeOp for resetting
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(new_compute_op.getDefiningOp());
      mlir::Type element_type = llvm::cast<mlir::TensorType>(computeOps[computeOp_i - 1].getType()).getElementType();
      Value compute_op_for_reset = createComputeOpForReset(
          /*common_indices=*/computeOp_to_common_indices[computeOp_i],
          /*prev_old_comput_op=*/computeOps[computeOp_i - 1],
          prev_new_compute_op,
          new_rhs_operand_op,
          element_type,
          rewriter,
          loc);
      new_compute_ops_for_reset.push_back(compute_op_for_reset);
    } else {
      /// The middle ComputeOp
      /// TODO: need a case to test this middle Computeop
      /// LHS dimension reduction
      Value lhs_operand_op = createNewLHSOperandOp(
          computeOp,
          /*num_common_indices=*/computeOp_to_common_indices[computeOp_i + 1].size(),
          /*lhs_tensor=*/newITreeInfo.block_args[intermediates_idx_base + computeOp_i],
          rewriter,
          loc);
      /// RHS dimension reduction
      uint32_t num_common_indices = computeOp_to_common_indices[computeOp_i].size();
      Value prev_new_compute_op = new_compute_ops.back();
      IndexTreeOperandOp new_rhs_operand_op;
      llvm::SmallVector<Value> rhs_operand_ops = createNewRHSOperandOps(
          /*prev_old_compute_op=*/computeOps[computeOp_i - 1],
          /*curr_old_compute_op=*/computeOps[computeOp_i],
                                  num_common_indices,
          /*intermediate_tensor=*/prev_new_compute_op,
                                  new_rhs_operand_op /*out*/,
                                  rewriter,
                                  loc);
      /// Create new ComputeOp
      Value new_compute_op = createNewComputeOp(
          computeOp,
          /*output_tensor_type=*/newITreeInfo.args_types[intermediates_idx_base + computeOp_i],
          lhs_operand_op,
          rhs_operand_ops,
          rewriter,
          loc);
      new_compute_ops.push_back(new_compute_op);
      /// Create ComputeOp for resetting
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(new_compute_op.getDefiningOp());
      mlir::Type element_type = llvm::cast<mlir::TensorType>(computeOps[computeOp_i - 1].getType()).getElementType();
      Value compute_op_for_reset = createComputeOpForReset(
          /*common_indices=*/computeOp_to_common_indices[computeOp_i],
          /*prev_old_comput_op=*/computeOps[computeOp_i - 1],
                            prev_new_compute_op,
                            new_rhs_operand_op,
                            element_type,
                            rewriter,
                            loc);
      new_compute_ops_for_reset.push_back(compute_op_for_reset);
    }
  }

  /// Update the YieldOp.
  Operation *old_yield_op = newITreeInfo.itree_op.getRegion().getBlocks().front().getTerminator();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(old_yield_op);
  llvm::SmallVector<Value> yield_op_args;
  yield_op_args.push_back(new_compute_ops.back());
  yield_op_args.insert(yield_op_args.end(), new_compute_ops_for_reset.begin(), new_compute_ops_for_reset.end());
  rewriter.create<indexTree::YieldOp>(loc, TypeRange(), yield_op_args);
  rewriter.eraseOp(old_yield_op);

  /// Erase old Compute Ops.
  for (uint32_t computeOp_i = 0; computeOp_i < num_computeOps; ++computeOp_i) {
    rewriter.replaceAllUsesWith(computeOps[computeOp_i], new_compute_ops[computeOp_i]);
    rewriter.eraseOp(computeOps[computeOp_i].getDefiningOp());
  }

  comet_vdump(funcOp->getParentOfType<ModuleOp>());

}

/// Apply the redundancy-aware kernel fusion on index tree dialect for some compound expressions
std::unique_ptr<Pass> mlir::comet::createIndexTreeDimensionReductionPass()
{
  return std::make_unique<IndexTreeDimensionReduction>();
}