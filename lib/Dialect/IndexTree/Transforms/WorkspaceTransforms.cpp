//===- WorkspaceTransforms.cpp  ------===//
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
//===----------------------------------------------------------------------===//
//
// This pass performs workspace transformations on index tree dialect for sparse-sparse computation
//===----------------------------------------------------------------------===//

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Types.h"

#include "llvm/Support/Debug.h"
#include <iostream>
#include <algorithm>
#include <vector>

#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <string>
#include <utility>
#include <queue>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::arith;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

using llvm::SmallVector;
using llvm::StringRef;

#define DEBUG_TYPE "workspace-transformations"

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

#define TENSOR_NUMS 3
#define INPUT_TENSOR_NUMS 2

const bool compressedworkspace = true;

struct dimInTensor
{
  int dim;
  int tensorId;
  int dimOrder;
};

///  Apply workspace transformation on the lhs
///  Consider CSR first
///  ikj Cij += Aik * Bkj ===> i (j Wj = 0; kj Wj += Aik * Bkj; kj Cij=Wj)
///  ij Cij = Aij * Bij =====> i (j Wj = 0; j Wj = Aij * Bij; j Cij=Wj)

///  Apply workspace transformation on the rhs
///  Consider CSR first
///  j Wj = Aij * Bij ===> j Vj = 0; j Vj = Bij; j Wj = Aij * Vj;

//===----------------------------------------------------------------------===//
/// WorkspaceTransforms Pass
//===----------------------------------------------------------------------===//

///  Apply workspace transformations on the ta.tc and tc.elews_mul
namespace
{
  struct WorkspaceTransformsPass
      : public PassWrapper<WorkspaceTransformsPass, OperationPass<mlir::func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WorkspaceTransformsPass)
    void runOnOperation() override;
    void WorkspaceTransforms(mlir::func::FuncOp function);
  };

  struct IndexTreeWorkspaceTransformationsPass
      : public PassWrapper<IndexTreeWorkspaceTransformationsPass, OperationPass<mlir::func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexTreeWorkspaceTransformationsPass)
    void runOnOperation() override;
    void CompressedWorkspaceTransforms(mlir::func::FuncOp function);
  };

} /// end anonymous namespace.

/// Need a function, dfs traverse the itree
/// get the sparse index that is sparse in the output
std::vector<int> getSparseDimsOutput(std::vector<std::vector<std::string>> opFormats, std::vector<std::vector<int>> opPerms)
{
  std::vector<int> sparseDimsOutput;
  assert(opFormats.size() > 0 && "opFormats.size() less than 0\n");
  std::vector<std::string> outputFormat = opFormats[opFormats.size() - 1];
  std::vector<int> outputPerm = opPerms[opPerms.size() - 1];
  for (unsigned int i = 0; i < outputFormat.size(); i++)
  {
    if (outputFormat[i].compare("D") != 0)
    { /// sparse dim
      sparseDimsOutput.push_back(outputPerm[i]);
      comet_debug() << "sparse dim in output: " << outputPerm[i] << "  with format: " << outputFormat[i] << "\n";
    }
  }
  return sparseDimsOutput;
}

/// get the sparse index that has sparse format in at least two input tensors
/// which tensor, which dimension.  use std::pair represent the information
std::vector<struct dimInTensor> getSparseDimsInput(std::vector<std::vector<std::string>> opFormats, std::vector<std::vector<int>> opPerms)
{
  std::vector<struct dimInTensor> sparseDimsInput;

  std::vector<std::vector<std::string>> inputFormats = {opFormats.begin(), opFormats.end() - 1};
  std::vector<std::vector<int>> inputPerms = {opPerms.begin(), opPerms.end() - 1};

  /// Get all dims in input tensors
  std::vector<int> allPermsInput = getUnionOf2Dvector(inputPerms);
  comet_debug() << " allPermsInput.size(): " << allPermsInput.size() << "\n";
  comet_debug() << "allPermsInput: ";
  for (auto n : allPermsInput)
  {
    comet_debug() << n << " ";
  }
  comet_debug() << "\n";

  for (unsigned int i = 0; i < allPermsInput.size(); i++)
  {
    int cur_index = allPermsInput[i];
    comet_debug() << " cur_index: " << cur_index << "\n";
    /// Get the format of cur_index from each input tensor
    std::vector<std::string> cur_formats;
    std::vector<int> tensor_ids;
    std::vector<int> dim_orders;
    for (unsigned int j = 0; j < inputPerms.size(); j++)
    {
      unsigned int whichFormat = findIndexInVector(inputPerms[j], cur_index);
      if (whichFormat < inputPerms[j].size())
      { /// found
        std::string format = inputFormats[j][whichFormat];
        cur_formats.push_back(format);
        tensor_ids.push_back(j);
        dim_orders.push_back(whichFormat);
      }
    }
    comet_debug() << " cur_formats.size(): " << cur_formats.size() << "\n";
    comet_debug() << "cur_formats: ";
    for (auto n : cur_formats)
    {
      comet_debug() << n << " ";
    }
    comet_debug() << "\n";

    /// check if there is sparse format in cur_formats vector
    std::vector<std::string> cur_sparse_formats;
    std::vector<int> sparse_tensor_ids;
    std::vector<int> sparse_dim_orders;
    for (unsigned int j = 0; j < cur_formats.size(); j++)
    {
      comet_debug() << " cur_formats[" << j << "]: " << cur_formats[j] << "\n";
      if (cur_formats[j].compare("D") != 0)
      { /// sparse format
        cur_sparse_formats.push_back(cur_formats[j]);
        sparse_tensor_ids.push_back(tensor_ids[j]);
        sparse_dim_orders.push_back(dim_orders[j]);
        comet_debug() << " sparse dim in format: " << cur_index << " with format: " << cur_formats[j] << "\n";
      }
    }

    if (cur_sparse_formats.size() > 1)
    { /// More than one sparse format
      struct dimInTensor dim_in_tensor;
      dim_in_tensor.dim = cur_index;
      dim_in_tensor.tensorId = sparse_tensor_ids[0]; /// Any sparse tensor is ok
      dim_in_tensor.dimOrder = sparse_dim_orders[0];
      sparseDimsInput.push_back(dim_in_tensor);
    }
  }

  comet_debug() << "sparseDimsInput: ";
  for (auto n : sparseDimsInput)
  {
    comet_debug() << "(" << n.dim << ", " << n.tensorId << ", " << n.dimOrder << ") ";
  }
  comet_debug() << "\n";
  return sparseDimsInput;
}

///  Split one indicesOp into several one, i.e. each computeOp has its own parent op
///  i -> j -> V=0;V=A;W=V*B ===> i -> j -> V=0;
///                                -> j -> V=A;
///                                -> j -> W=V*B
void splitIndicesOp(Operation *needSplitNode, Value denseIndicesOp, OpBuilder &builder, Location loc)
{
  while (isa<indexTree::IndexTreeIndicesOp>(needSplitNode))
  {

    comet_pdump(needSplitNode);
    /// check how many operands, split into many operands.
    indexTree::IndexTreeIndicesOp indicesOp = dyn_cast<indexTree::IndexTreeIndicesOp>(needSplitNode);

    comet_vdump(indicesOp);

    Operation *indicesOpFirstUsers = *(indicesOp.getOperation()->getResult(0).getUsers().begin());
    comet_pdump(indicesOpFirstUsers);

    builder.setInsertionPoint(indicesOpFirstUsers);
    comet_debug() << "\n";

    if (needSplitNode != denseIndicesOp.getDefiningOp())
    {
      ArrayAttr indices = indicesOp.getIndices();

      comet_debug() << " indicesOp.getOperation()->getNumOperands(): " << indicesOp.getOperation()->getNumOperands() << "\n";
      std::vector<mlir::Value> operands;
      std::vector<mlir::Value> newIndicesOp;
      for (unsigned int i = 0; i < indicesOp.getOperation()->getNumOperands(); i++)
      {
        operands.push_back(indicesOp.getOperation()->getOperand(i));

        comet_vdump(indicesOp.getOperation()->getOperand(i));
        comet_vdump(operands[i]);
        auto i64Type = builder.getI64Type();
        Value t1 = builder.create<indexTree::IndexTreeIndicesOp>(loc, i64Type, operands[i], indices);

        comet_debug() << "New IndexTreeIndicesOp added:\n";
        comet_vdump(t1);
        newIndicesOp.push_back(t1);
      }

      /// put it here
      comet_debug() << " finished calling replacereplaceOperands \n";
      /// This parentIndicesOp is the operation that need to be splitted next time

#ifdef DEBUG_MODE_WorkspaceTransformsPass
      comet_vdump(indicesOp.getOperation()->getResult(0));
      for (auto ppp : indicesOp.getOperation()->getResult(0).getUsers())
      {

        comet_pdump(ppp);
      }
#endif

      Operation *parentIndicesOp = *(indicesOp.getOperation()->getResult(0).getUsers().begin());

      comet_pdump(parentIndicesOp);

      replaceOperands(needSplitNode, newIndicesOp);
      needSplitNode = parentIndicesOp;

      comet_debug() << " plan to erase the following Op\n";
      comet_debug() << " Indices operations:\n";
      comet_vdump(indicesOp);
      comet_debug() << " Split Nodes:\n";
      comet_pdump(needSplitNode);
      comet_debug() << " Indices op first users:\n";
      comet_pdump(indicesOpFirstUsers);
      indicesOp.erase();
    }
    else
    {
      comet_debug() << "\n";
      break;
    }
  }
  comet_debug() << "\n";
}

void removeRedundantIndices(std::vector<Value> newComputeOps,
                            std::map<int, mlir::Value> indexValueMap,
                            int denseDimInOutput,
                            OpBuilder &builder,
                            Location loc)
{

  /// Check whether need to remove redundant indices or not
  /// Get the

  /// -------Remove redundant indices-------------
  /// For C, the 1st dim i is Dense, the second dim j is sparse.
  /// ---- the index including i and before i is not included
  mlir::Value denseIndicesOp = indexValueMap[denseDimInOutput];
  /// The indices after denseIndicesOp need to be splitted
  /// start from the computeOp,
  /// Finished one level
  /// Only one User, because it's a tree structure, the leaf only has one parent
  assert(newComputeOps[0].getDefiningOp()->getResult(0).hasOneUse() && " the computeOp has more than one users\n");
  /// Get the only one user
  Operation *onlyUser = *(newComputeOps[0].getDefiningOp()->getResult(0).getUsers().begin());
  comet_pdump(onlyUser);

  /// needSplitNode is the parent node of the "denseIndicesOp"
  Operation *needSplitNode = onlyUser;
  /// iterate until the
  /// call splitIndicesOp function to split indicesOp until latest "root"
  splitIndicesOp(needSplitNode, denseIndicesOp, builder, loc);
  comet_debug() << "\n";

  /// Remove the indices for each tensor
  /// iterate over all itComputeOps, get the indices for each tensor
  for (auto n : newComputeOps)
  {
    /// get allPerms, put all indices id into a vector,
    /// iterater up until reach the root noe, if the index of indicesOp is not in the vector
    ///      remove this one: set the operand of the parent of the indicesOp into current op

    comet_debug() << " current computeOp: \n";
    comet_vdump(n);
    ArrayAttr allperms_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(n.getDefiningOp()->getOperand(0).getDefiningOp()).getAllPerms();
    std::vector<std::vector<int>> allpermsInt_rhs = convertArrayAttrIntTo2DVector(allperms_rhs);
    std::vector<int> permsInt = getUnionOf2Dvector(allpermsInt_rhs);
    comet_debug() << " print permsInt: ";
    for (auto p : permsInt)
    {
      comet_debug() << p << " ";
    }
    comet_debug() << "\n";

    mlir::Value computeOp = n;

    /// iterate over the IndexTreeIndicesOp;
    mlir::Value computeOpParent; /// computeOpParent is IndexTreeIndicesOp

    comet_vdump(n);
    assert(n.getDefiningOp()->getResult(0).hasOneUse() && " indicesOp has more than one user\n");
    Operation *computeOpParentPointer = *(n.getDefiningOp()->getResult(0).getUsers().begin());
    computeOpParent = computeOpParentPointer->getResult(0);

    comet_pdump(computeOpParentPointer);

    comet_vdump(computeOpParent);

    while (!isRealRoot(computeOpParent.getDefiningOp()))
    {
      comet_vdump(computeOpParent);
      if (isa<indexTree::IndexTreeComputeOp>(computeOpParent.getDefiningOp()))
      {
        comet_debug() << " indicesOp's parent can not be computeOp\n";
      }
      else if (isa<indexTree::IndexTreeOp>(computeOpParent.getDefiningOp()))
      {
        comet_debug() << " indicesOp's parent is IndexTreeOp\n";
      }
      else if (isa<indexTree::IndexTreeIndicesOp>(computeOpParent.getDefiningOp()))
      {
        /// get the indices integer, to see if it is in permsInt
        /// if yes, don't remove
        /// if no, remove:
        indexTree::IndexTreeIndicesOp curIndicesOp = dyn_cast<indexTree::IndexTreeIndicesOp>(computeOpParent.getDefiningOp());
        comet_debug() << " \n";
        ArrayAttr idsArrayAttr = curIndicesOp.getIndices(); /// should be 1D vector
        std::vector<int> idsVec;
        for (auto n : idsArrayAttr)
        {
          idsVec.push_back(n.cast<mlir::IntegerAttr>().getInt());
        }
        comet_debug() << " print idsVec: ";
        for (auto p : idsVec)
        {
          comet_debug() << p << " ";
        }
        comet_debug() << "\n";

        assert(idsVec.size() == 1 && " indicesOp contain more than 1 index\n");
        bool isNeedRemove = false;
        for (auto n : idsVec)
        { /// only 1 index actually, because each indicesOp contain one index
          if (std::find(permsInt.begin(), permsInt.end(), n) != permsInt.end())
          {
            /// found
            isNeedRemove = false;
          }
          else
          { /// the index in curIndicesOp is not found in the computeOp indices
            isNeedRemove = true;
          }
        }

        /// if curIndicesOp is the "real root" of the index tree (has only one user)
        ///                 contain more than 1 index
        if (curIndicesOp.getOperation()->getNumOperands() > 1 && curIndicesOp.getOperation()->getResult(0).hasOneUse() && isa<indexTree::IndexTreeOp>(*(curIndicesOp.getOperation()->getResult(0).getUsers().begin())))
        {
          isNeedRemove = false;
        }
        comet_debug() << " isNeedRemove = " << isNeedRemove << "\n";

        if (isNeedRemove)
        {
          assert(curIndicesOp.getOperation()->getResult(0).hasOneUse() && " indicesOp has more than one user\n");
          Operation *curIndicesOpParent = *(curIndicesOp.getOperation()->getResult(0).getUsers().begin());

          comet_vdump(computeOpParent);
          comet_pdump(curIndicesOpParent);

          computeOpParent.replaceAllUsesWith(computeOp); /// replace all uses of the indexOp with the new indecesOp
          computeOp = computeOpParent;
          computeOpParent.getDefiningOp()->erase(); /// erase the previous  indecesOp
          computeOpParent = curIndicesOpParent->getResult(0);
        }
        else
        {
#ifdef DEBUG_MODE_WorkspaceTransformsPass
          comet_vdump(curIndicesOp);
          int count = 0;
          for (auto p : curIndicesOp.getOperation()->getResult(0).getUsers())
          {
            comet_pdump(p);
            count++;
          }
          comet_debug() << " count: " << count << "\n";
#endif
          assert(curIndicesOp.getOperation()->getResult(0).hasOneUse() && " indicesOp has more than one user\n");
          Operation *curIndicesOpParent = *(curIndicesOp.getOperation()->getResult(0).getUsers().begin());

          comet_pdump(curIndicesOpParent);

          computeOp = computeOpParent;
          computeOpParent = curIndicesOpParent->getResult(0);
        }
      }
    }

  } /// end for n
}

std::vector<Value> CompressedWorkspaceOutput(std::vector<int> sparseDimsOutput,
                                             indexTree::IndexTreeComputeOp itComputeOp,
                                             std::vector<std::vector<std::string>> opFormats,
                                             std::vector<std::vector<int>> opPerms,
                                             std::map<int, mlir::Value> indexValueMap,
                                             OpBuilder &builder, indexTree::IndexTreeOp op)
{
  Location loc = op.getLoc();
  auto comp_worksp_opt = builder.getBoolAttr(compressedworkspace);
  int sparseDimOutput = -1;
  int sparseDimOrderInOutput = -1;
  int denseDimInOutput = -1;
  auto i64Type = builder.getI64Type();

  for (unsigned int j = 0; j < opFormats[opFormats.size() - 1].size(); j++)
  {
    /// sparse dimension
    if (opFormats[opFormats.size() - 1][j].compare("D") != 0)
    {
      sparseDimOutput = opPerms[opPerms.size() - 1][j];
      sparseDimOrderInOutput = j;
    }
    else /// dense dimension
      denseDimInOutput = opPerms[opPerms.size() - 1][j];
  }
  comet_debug() << " " << sparseDimOutput << "\n";

  /// 3. Find the ta.itIndices op which represents sparseDimOutput
  /// Find its parent ...
  Value sparseIndicesOp = indexValueMap[sparseDimOutput];

  comet_vdump(sparseIndicesOp);
  comet_debug() << " sparseDimOrderInOutput: " << sparseDimOrderInOutput << "\n";
  Value sparseDimsPerent = indexValueMap[sparseDimOrderInOutput - 1];
  comet_debug() << " sparseDimsPerent: \n";
  comet_vdump(sparseDimsPerent);

  /// Cij = Aik * Bkj ==>
  ///    ComputeNode(c1): Wj = 0;
  ///    ComputeNode(c2): Wj += Aik * Bkj;
  ///    ComputeNode(c3): Cij = Wj
  std::vector<mlir::Value> tensors;
  getTensorsOfComputeOp(itComputeOp.getOperation()->getResult(0), tensors);

  /// 4. create W, j dim size of
  /// Value outputItComputeOp = itComputeOp.getOperation()->getOperand(itComputeOp.getOperation()->getNumOperands() - 1).getDefiningOp()->getOperand(0);
  ///  new version
  Value outputItComputeOp = tensors[tensors.size() - 1];
  comet_vdump(outputItComputeOp);

  std::vector<mlir::Value> w_lbls_value = {outputItComputeOp.getDefiningOp()->getOperand(sparseDimOrderInOutput)};

  comet_vdump(outputItComputeOp.getDefiningOp()->getOperand(sparseDimOrderInOutput));
  std::string w_format = "Dense"; /// tensor<?xf64>
  auto w_type = RankedTensorType::get({mlir::ShapedType::kDynamic}, builder.getF64Type());

  Operation *itComputeOpFirstUsers = *(itComputeOp.getOperation()->getUsers().begin());
  builder.setInsertionPoint(itComputeOpFirstUsers); /// Insert before itree Op

  mlir::Value w = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, w_type, w_lbls_value, w_format);
  comet_vdump(w);
  auto w_index_list_type = RankedTensorType::get({mlir::ShapedType::kDynamic}, builder.getIndexType()); /// tensor<?xindex>
  mlir::Value w_already_set = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, w_index_list_type, w_lbls_value, w_format);
  comet_vdump(w_already_set);
  mlir::Value w_index_list = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, w_index_list_type, w_lbls_value, w_format);
  comet_vdump(w_index_list);

  MemRefType w_index_list_size_type = MemRefType::get({1}, builder.getIndexType());                   /// tensor<1xindex>
  mlir::Value w_index_list_size_alloc = builder.create<memref::AllocOp>(loc, w_index_list_size_type); /// tensor<1xindex>
  Value w_index_list_size = builder.create<ToTensorOp>(loc, w_index_list_size_alloc);

  std::vector<Value> workspaceTensors = {w, w_already_set, w_index_list, w_index_list_size};
  tensors.push_back(w); /// {A, B, C, W}

  std::vector<std::vector<std::string>> formats = {opFormats[0], opFormats[1], opFormats[2], {"D"}};
  std::vector<std::vector<int>> perms = {opPerms[0], opPerms[1], opPerms[2], {sparseDimOutput}};

  /// Start building an IndexTreeCompute Operation to represent Wj = 0;
  std::vector<int> c1_perms_int_0;
  std::vector<int> c1_perms_int_1;
  std::vector<std::vector<int>> c1_perms_int = {c1_perms_int_0, c1_perms_int_1};
  std::vector<std::string> c1_formats_str_0;
  std::vector<std::string> c1_formats_str_1;
  std::vector<std::vector<std::string>> c1_formats_str = {c1_formats_str_0, c1_formats_str_1};

  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  std::vector<mlir::Value> c1_rhs = {const_index_0};
  mlir::Value c1_lhs = {w_index_list_size};
  std::string semiringName(itComputeOp.getSemiring().data());
  std::string maskNone = "none";
  std::string maskTypeName(itComputeOp.getMaskType().data());
  auto c1_semiring = builder.getStringAttr(semiringName);
  auto c1_maskType = builder.getStringAttr(maskNone); /// masking attribute

  /// for c1_rhs
  std::vector<std::vector<int>> c1_rhsop_perms_str = {c1_perms_int_0};
  ArrayAttr c1_rhsop_perms = convert2DVectorToArrayAttrInt(c1_rhsop_perms_str, builder);
  std::vector<std::vector<std::string>> c1_rhsop_formats_str = {c1_formats_str_0};
  ArrayAttr c1_rhsop_formats = convert2DVectorToArrayAttrStr(c1_rhsop_formats_str, builder);
  mlir::Value c1_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getIndexType()), c1_rhs, c1_rhsop_perms, c1_rhsop_formats);
  comet_debug() << "IndexTreeComputeRHS Operation in Output (c1_rhs):\n";
  comet_vdump(c1_rhsop);

  /// for c1_lhs
  std::vector<std::vector<int>> c1_lhsop_perms_str = {c1_perms_int_1};
  ArrayAttr c1_lhsop_perms = convert2DVectorToArrayAttrInt(c1_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> c1_lhsop_formats_str = {c1_formats_str_1};
  ArrayAttr c1_lhsop_formats = convert2DVectorToArrayAttrStr(c1_lhsop_formats_str, builder);
  mlir::Value c1_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), c1_lhs, c1_lhsop_perms, c1_lhsop_formats);
  comet_debug() << "IndexTreeComputeLHS Operation in Output (c1_lhs):\n";
  comet_vdump(c1_lhsop);

  /// for c1 ==> Wj = 0;
  mlir::Value c1 = builder.create<indexTree::IndexTreeComputeOp>(loc, builder.getI64Type(), c1_rhsop, c1_lhsop, comp_worksp_opt, c1_semiring, c1_maskType);
  comet_debug() << "IndexTreeCompute Operation in Output (c1):\n";
  comet_vdump(c1);

  /// insert c1 to sparseDimsParent
  sparseDimsPerent.getDefiningOp()->insertOperands(0, c1);

  /// Start building an IndexTreeCompute Operation to represent Wj += Aik * Bkj;
  std::vector<mlir::Value> c2_tensors = {tensors[0], tensors[1], w};
  std::vector<int> c2_perms_int_0 = opPerms[0];
  std::vector<int> c2_perms_int_1 = opPerms[1];
  std::vector<int> c2_perms_int_2 = {sparseDimOutput};
  std::vector<std::vector<int>> c2_perms_int = {c2_perms_int_0, c2_perms_int_1, c2_perms_int_2};

  /// Convert formats string array into StrAttr
  std::vector<std::string> c2_formats_str_0 = opFormats[0];
  std::vector<std::string> c2_formats_str_1 = opFormats[1];
  std::vector<std::string> c2_formats_str_2 = {"D"};
  std::vector<std::vector<std::string>> c2_formats_str = {c2_formats_str_0, c2_formats_str_1, c2_formats_str_2};
  std::vector<mlir::Value> c2_rhs;
  std::vector<std::vector<std::string>> c2_rhsop_formats_str;
  std::vector<std::vector<int>> c2_rhsop_perms_str;
  if (tensors.size() > 4) /// masking input is available: tensors = {%op0, %op1, %mask, %out, %W}
  {
    c2_rhs = {c2_tensors[0], c2_tensors[1], tensors[2]};                       /// tensors[2] val is the mask
    c2_rhsop_formats_str = {c2_formats_str_0, c2_formats_str_1, opFormats[2]}; /// mask format is same as the output
    c2_rhsop_perms_str = {c2_perms_int_0, c2_perms_int_1, opPerms[2]};         /// perms of mask are same as the output
  }
  else /// no masking input is provided: tensors = {%op0, %op1, %out, %W}
  {
    c2_rhs = {c2_tensors[0], c2_tensors[1]};
    c2_rhsop_formats_str = {c2_formats_str_0, c2_formats_str_1};
    c2_rhsop_perms_str = {c2_perms_int_0, c2_perms_int_1};
  }
  std::vector<mlir::Value> c2_lhs = workspaceTensors;

  auto c2_semiring = builder.getStringAttr(semiringName);
  auto c2_maskType = builder.getStringAttr(maskTypeName); /// masking attribute

  /// for c2_rhsop
  ArrayAttr c2_rhsop_perms = convert2DVectorToArrayAttrInt(c2_rhsop_perms_str, builder);
  ArrayAttr c2_rhsop_formats = convert2DVectorToArrayAttrStr(c2_rhsop_formats_str, builder);
  mlir::Value c2_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), c2_rhs, c2_rhsop_perms, c2_rhsop_formats);
  comet_debug() << "IndexTreeComputeRHS Operation in Output (c2_rhs):\n";
  comet_vdump(c2_rhsop);

  /// for c2_lhsop
  std::vector<std::vector<int>> c2_lhsop_perms_str = {c2_perms_int_2};
  ArrayAttr c2_lhsop_perms = convert2DVectorToArrayAttrInt(c2_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> c2_lhsop_formats_str = {c2_formats_str_2};
  ArrayAttr c2_lhsop_formats = convert2DVectorToArrayAttrStr(c2_lhsop_formats_str, builder);
  mlir::Value c2_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), c2_lhs, c2_lhsop_perms, c2_lhsop_formats);
  comet_debug() << "IndexTreeComputeLHS Operation in Output (c2_lhs):\n";
  comet_vdump(c2_lhsop);

  /// for c2
  mlir::Value c2 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, c2_rhsop, c2_lhsop, comp_worksp_opt, c2_semiring, c2_maskType);
  comet_debug() << "IndexTreeCompute Operation in Output (c2):\n";
  comet_vdump(c2);

  /// Start building an IndexTreeCompute Operation to represent Cij = Wj;
  std::vector<mlir::Value> c3_tensors;
  if (tensors.size() > 4) /// masking input is available: tensors = {%op0, %op1, %mask, %out, %W}
  {
    c3_tensors = {tensors[3]};
  }
  else /// masking input is NOT available: tensors = {%op0, %op1, %out, %W}
  {
    c3_tensors = {tensors[2]};
  }
  std::vector<int> c3_perms_int_0 = {sparseDimOutput};
  std::vector<int> c3_perms_int_1 = opPerms[2];
  std::vector<std::vector<int>> c3_perms_int = {c3_perms_int_0, c3_perms_int_1};

  /// Convert formats string array into StrAttr
  std::vector<std::string> c3_formats_str_0 = {"D"};
  std::vector<std::string> c3_formats_str_1 = opFormats[2];
  std::vector<std::vector<std::string>> c3_formats_str = {c3_formats_str_0, c3_formats_str_1};

  std::vector<mlir::Value> c3_rhs = workspaceTensors;
  mlir::Value c3_lhs = c3_tensors[0];
  auto c3_semiring = builder.getStringAttr(semiringName);
  auto c3_maskType = builder.getStringAttr(maskNone); /// masking attribute

  /// for c3_rhs
  std::vector<std::vector<int>> c3_rhsop_perms_str = {c3_perms_int_0};
  ArrayAttr c3_rhsop_perms = convert2DVectorToArrayAttrInt(c3_rhsop_perms_str, builder);
  std::vector<std::vector<std::string>> c3_rhsop_formats_str = {c3_formats_str_0};
  ArrayAttr c3_rhsop_formats = convert2DVectorToArrayAttrStr(c3_rhsop_formats_str, builder);
  mlir::Value c3_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), c3_rhs, c3_rhsop_perms, c3_rhsop_formats);
  comet_debug() << "IndexTreeComputeRHS Operation in Output (c3_rhs):\n";
  comet_vdump(c3_rhsop);

  /// for c3_lhs
  std::vector<std::vector<int>> c3_lhsop_perms_str = {c3_perms_int_1};
  ArrayAttr c3_lhsop_perms = convert2DVectorToArrayAttrInt(c3_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> c3_lhsop_formats_str = {c3_formats_str_1};
  ArrayAttr c3_lhsop_formats = convert2DVectorToArrayAttrStr(c3_lhsop_formats_str, builder);
  mlir::Value c3_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), c3_lhs, c3_lhsop_perms, c3_lhsop_formats);
  comet_debug() << "IndexTreeComputeLHS Operation in Output (c3_lhs):\n";
  comet_vdump(c3_lhsop);

  /// for c3 ==> Cij = Wj;
  mlir::Value c3 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, c3_rhsop, c3_lhsop, comp_worksp_opt, c3_semiring, c3_maskType);
  comet_debug() << "IndexTreeCompute Operation in Output (c3):\n";
  comet_vdump(c3);

  std::vector<mlir::Value> newComputeOps = {c2, c3};
  sparseIndicesOp.getDefiningOp()->setOperands(newComputeOps);

  /// remove redundant indices by calling a function
  /// in elementwise: not remove
  /// in spgemm: remove
  /// check if there is redundant index
  bool existRedundantIndex = false;
  for (auto n : newComputeOps)
  {
    std::vector<std::vector<int>> perms;
    getPermsOfComputeOp(n, perms);
    std::vector<int> allperms = getUnionOf2Dvector(perms);
    comet_debug() << " print allperms \n";
    print_vector<int>(allperms);

    std::vector<Value> ancestors;
    std::vector<mlir::Value> dfsOps;
    dfsRootOpTree(op.getChildren(), dfsOps);
    getAncestorsWp(n, ancestors, dfsOps);
    comet_debug() << " print ancestors \n";
    print_vector_value(ancestors);

    /// Iterate over every indicesOp
    for (auto ancestor : ancestors)
    {
      /// If indicesOp's index is in allperms, no redundant
      ///    the indicesOp is real root, no redundant
      /// Otherwise, redundant
      if (isa<indexTree::IndexTreeIndicesOp>(ancestor.getDefiningOp()))
      {
        indexTree::IndexTreeIndicesOp indicesOp = dyn_cast<indexTree::IndexTreeIndicesOp>(ancestor.getDefiningOp());

        ArrayAttr idsArrayAttr = indicesOp.getIndices(); /// should be 1D vector
        /// actually only one index for the indicesOp in our implementation
        for (auto m : idsArrayAttr)
        {
          int perm = m.cast<mlir::IntegerAttr>().getInt();
          comet_debug() << " perm: " << perm << "\n";

          if (findIndexInVector(allperms, perm) == allperms.size())
          { /// not exit
            comet_debug() << " perm not exist in allperms\n";
            if (!isRealRoot(indicesOp.getOperation()))
            {
              existRedundantIndex = true;
              comet_debug() << " existRedundantIndex: " << existRedundantIndex << "\n";
            }
          }
        }
      }
    }
  }

  if (existRedundantIndex)
  {
    comet_debug() << "There is loop invariant\n";
    removeRedundantIndices(newComputeOps, indexValueMap, denseDimInOutput, builder, loc);
  }

  return newComputeOps;
} /// end CompressedWorkspaceOutput()

void CompressedWorkspaceInput(std::vector<Value> computeOps, OpBuilder &builder, Location loc)
{
  auto comp_worksp_opt = builder.getBoolAttr(compressedworkspace);
  for (auto computeOp : computeOps)
  {
    ///  1. get the opFormats and opPerms of the computeOp
    std::vector<std::vector<std::string>> opFormats;
    std::vector<std::vector<int>> opPerms;
    std::vector<std::vector<bool>> inputOutputMapping;
    getFormatsPermsOfComputeOp(computeOp, opFormats, opPerms, inputOutputMapping);
    comet_debug() << " \n";
    for (auto n : opFormats)
    {

      print_vector<std::string>(n);
    }
    for (auto n : opPerms)
    {

      print_vector<int>(n);
    }
    std::vector<Value> tensors;
    getTensorsOfComputeOp(computeOp, tensors);
    comet_debug() << " tensors.size(): " << tensors.size() << "\n";
    std::vector<Value> tensors_rhs;
    getInputTensorsOfComputeOp(computeOp, tensors_rhs);
    comet_debug() << " tensors_rhs.size(): " << tensors_rhs.size() << "\n";
    std::vector<Value> tensors_lhs;
    getOutputTensorsOfComputeOp(computeOp, tensors_lhs);
    comet_debug() << " tensors_lhs.size(): " << tensors_lhs.size() << "\n";

    indexTree::IndexTreeComputeOp itComputeOp = dyn_cast<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp());
    std::string semiringName(itComputeOp.getSemiring().data());

    std::vector<int> sparseDimsOutput = getSparseDimsOutput(opFormats, opPerms);
    std::vector<struct dimInTensor> sparseDimsInput = getSparseDimsInput(opFormats, opPerms);
    comet_debug() << " sparseDimsInput.size(): " << sparseDimsInput.size() << "\n";

    if (sparseDimsInput.size() == 1)
    { /// solve only 1 sparseDimsInput
      /// No need to apply workspace transformation
      comet_debug() << " sparseDimsInput[0]: " << sparseDimsInput[0].dim << ", " << sparseDimsInput[0].tensorId << ", " << sparseDimsInput[0].dimOrder << "\n";

      /// Wj=Aij*Bij ==>
      ///    ComputeNode(c1): Vj=0;
      ///    ComputeNode(c2): Vj=Aij;
      ///    ComputeNode(c3): Wj=Vj*Bij

      Value sparseInput = tensors_rhs[sparseDimsInput[0].tensorId];
      comet_vdump(sparseInput);

      std::vector<mlir::Value> v_lbls_value = {sparseInput.getDefiningOp()->getOperand(sparseDimsInput[0].dimOrder)};
      comet_debug() << "Dumping v_lbls_value\n";
      comet_vdump(v_lbls_value[0]);
      comet_debug() << "Done\n";
      comet_vdump(sparseInput.getDefiningOp()->getOperand(sparseDimsInput[0].dimOrder));
      std::string v_format = "Dense"; /// tensor<?xf64>
      auto v_type = RankedTensorType::get({mlir::ShapedType::kDynamic}, builder.getF64Type());

      builder.setInsertionPoint(computeOp.getDefiningOp());
      mlir::Value v = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, v_type, v_lbls_value, v_format);
      comet_vdump(v);

      /// Start building an IndexTreeCompute Operation to represent Vj=0
      std::vector<int> c1_perms_int_0;
      std::vector<int> c1_perms_int_1 = {sparseDimsInput[0].dim};
      std::vector<std::vector<int>> c1_perms_int = {c1_perms_int_0, c1_perms_int_1};

      std::vector<std::string> c1_formats_str_0;
      std::vector<std::string> c1_formats_str_1 = {"D"};
      std::vector<std::vector<std::string>> c1_formats_str = {c1_formats_str_0, c1_formats_str_1};

      auto i64Type = builder.getI64Type();
      Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0.0));
      std::vector<mlir::Value> c1_rhs = {const_f64_0};
      mlir::Value c1_lhs = {v};
      std::string semiringName(itComputeOp.getSemiring().data());
      std::string maskNone = "none";
      auto c1_semiring = builder.getStringAttr(semiringName);
      auto c1_maskType = builder.getStringAttr(maskNone); /// masking attribute

      /// for c1_rhs
      std::vector<std::vector<int>> c1_rhsop_perms_str = {c1_perms_int_0};
      ArrayAttr c1_rhsop_perms = convert2DVectorToArrayAttrInt(c1_rhsop_perms_str, builder);
      std::vector<std::vector<std::string>> c1_rhsop_formats_str = {c1_formats_str_0};
      ArrayAttr c1_rhsop_formats = convert2DVectorToArrayAttrStr(c1_rhsop_formats_str, builder);
      mlir::Value c1_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc,
                                                                              mlir::UnrankedTensorType::get(builder.getF64Type()),
                                                                              c1_rhs,
                                                                              c1_rhsop_perms,
                                                                              c1_rhsop_formats);
      comet_debug() << "IndexTreeComputeRHS Operation in Input (c1_rhs):";
      comet_vdump(c1_rhsop);

      /// for c1_lhs
      std::vector<std::vector<int>> c1_lhsop_perms_str = {c1_perms_int_1};
      ArrayAttr c1_lhsop_perms = convert2DVectorToArrayAttrInt(c1_lhsop_perms_str, builder);
      std::vector<std::vector<std::string>> c1_lhsop_formats_str = {c1_formats_str_1};
      ArrayAttr c1_lhsop_formats = convert2DVectorToArrayAttrStr(c1_lhsop_formats_str, builder);
      mlir::Value c1_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc,
                                                                              mlir::UnrankedTensorType::get(builder.getF64Type()),
                                                                              c1_lhs,
                                                                              c1_lhsop_perms,
                                                                              c1_lhsop_formats);
      comet_debug() << "IndexTreeComputeLHS Operation in Input (c1_lhs):";
      comet_vdump(c1_lhsop);

      /// for c1
      mlir::Value c1 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type,
                                                                     c1_rhsop,
                                                                     c1_lhsop,
                                                                     comp_worksp_opt,
                                                                     c1_semiring,
                                                                     c1_maskType);
      comet_debug() << "IndexTreeCompute Operation in Input (c1): ";
      comet_vdump(c1);

      /// Start building an IndexTreeCompute Operation to represent Vj = Aij
      std::vector<int> c2_perms_int_0 = opPerms[sparseDimsInput[0].tensorId];
      std::vector<int> c2_perms_int_1 = {sparseDimsInput[0].dim};
      std::vector<std::vector<int>> c2_perms_int = {c2_perms_int_0, c2_perms_int_1};

      std::vector<std::string> c2_formats_str_0 = opFormats[sparseDimsInput[0].tensorId];
      std::vector<std::string> c2_formats_str_1 = {"D"};
      std::vector<std::vector<std::string>> c2_formats_str = {c2_formats_str_0, c2_formats_str_1};

      std::vector<mlir::Value> c2_rhs = {tensors_rhs[sparseDimsInput[0].tensorId]};

      mlir::Value c2_lhs = {v};
      auto c2_semiring = builder.getStringAttr(semiringName);
      auto c2_maskType = builder.getStringAttr(maskNone); /// masking attribute

      /// for c2_rhs
      std::vector<std::vector<int>> c2_rhsop_perms_str = {c2_perms_int_0};
      ArrayAttr c2_rhsop_perms = convert2DVectorToArrayAttrInt(c2_rhsop_perms_str, builder);
      std::vector<std::vector<std::string>> c2_rhsop_formats_str = {c2_formats_str_0};
      ArrayAttr c2_rhsop_formats = convert2DVectorToArrayAttrStr(c2_rhsop_formats_str, builder);
      mlir::Value c2_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc,
                                                                              mlir::UnrankedTensorType::get(builder.getF64Type()),
                                                                              c2_rhs,
                                                                              c2_rhsop_perms,
                                                                              c2_rhsop_formats);
      comet_debug() << "IndexTreeComputeRHS Operation in Input (c2_rhs):";
      comet_vdump(c2_rhsop);

      /// for c2_lhs
      std::vector<std::vector<int>> c2_lhsop_perms_str = {c2_perms_int_1};
      ArrayAttr c2_lhsop_perms = convert2DVectorToArrayAttrInt(c2_lhsop_perms_str, builder);
      std::vector<std::vector<std::string>> c2_lhsop_formats_str = {c2_formats_str_1};
      ArrayAttr c2_lhsop_formats = convert2DVectorToArrayAttrStr(c2_lhsop_formats_str, builder);
      mlir::Value c2_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc,
                                                                              mlir::UnrankedTensorType::get(builder.getF64Type()),
                                                                              c2_lhs,
                                                                              c2_lhsop_perms,
                                                                              c2_lhsop_formats);
      comet_debug() << "IndexTreeComputeLHS Operation in Input (c2_lhs):";
      comet_vdump(c2_lhsop);

      /// for c2
      mlir::Value c2 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type,
                                                                     c2_rhsop,
                                                                     c2_lhsop,
                                                                     comp_worksp_opt,
                                                                     c2_semiring,
                                                                     c2_maskType);
      comet_debug() << "IndexTreeCompute Operation in Input (c2): ";
      comet_vdump(c2);

      /// Start building an IndexTreeCompute Operation to represent Wj=Vj*Bij
      std::vector<int> c3_perms_int_0 = {sparseDimsInput[0].dim};
      std::vector<int> c3_perms_int_1 = opPerms[1];
      std::vector<int> c3_perms_int_2 = opPerms[opPerms.size() - 1];
      std::vector<std::vector<int>> c3_perms_int = {c3_perms_int_0, c3_perms_int_1, c3_perms_int_2};

      /// Convert formats string array into StrAttr
      std::vector<std::string> c3_formats_str_0 = {"D"};
      std::vector<std::string> c3_formats_str_1 = opFormats[1];
      std::vector<std::string> c3_formats_str_2 = opFormats[opFormats.size() - 1];

      std::vector<std::vector<std::string>> c3_formats_str = {c3_formats_str_0, c3_formats_str_1, c3_formats_str_2};
      std::vector<mlir::Value> c3_rhs = {v, tensors[1]};

      comet_debug() << " tensors.size(): " << tensors.size() << "\n";
      std::vector<mlir::Value> c3_lhs = tensors_lhs;

      auto c3_semiring = builder.getStringAttr(semiringName);
      auto c3_maskType = builder.getStringAttr(maskNone); /// masking attribute

      /// for c3_rhs
      std::vector<std::vector<int>> c3_rhsop_perms_str = {c3_perms_int_0, c3_perms_int_1};
      ArrayAttr c3_rhsop_perms = convert2DVectorToArrayAttrInt(c3_rhsop_perms_str, builder);
      std::vector<std::vector<std::string>> c3_rhsop_formats_str = {c3_formats_str_0, c3_formats_str_1};
      ArrayAttr c3_rhsop_formats = convert2DVectorToArrayAttrStr(c3_rhsop_formats_str, builder);
      mlir::Value c3_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc,
                                                                              mlir::UnrankedTensorType::get(builder.getF64Type()),
                                                                              c3_rhs,
                                                                              c3_rhsop_perms,
                                                                              c3_rhsop_formats);
      comet_debug() << "IndexTreeComputeRHS Operation in Input (c3_rhs):";
      comet_vdump(c3_rhsop);

      /// for c3_lhs
      std::vector<std::vector<int>> c3_lhsop_perms_str = {c3_perms_int_2};
      ArrayAttr c3_lhsop_perms = convert2DVectorToArrayAttrInt(c3_lhsop_perms_str, builder);
      std::vector<std::vector<std::string>> c3_lhsop_formats_str = {c3_formats_str_2};
      ArrayAttr c3_lhsop_formats = convert2DVectorToArrayAttrStr(c3_lhsop_formats_str, builder);
      mlir::Value c3_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc,
                                                                              mlir::UnrankedTensorType::get(builder.getF64Type()),
                                                                              c3_lhs,
                                                                              c3_lhsop_perms,
                                                                              c3_lhsop_formats);
      comet_debug() << "IndexTreeComputeLHS Operation in Input (c3_lhs):";
      comet_vdump(c3_lhsop);

      /// for c3
      mlir::Value c3 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type,
                                                                     c3_rhsop,
                                                                     c3_lhsop,
                                                                     comp_worksp_opt,
                                                                     c3_semiring,
                                                                     c3_maskType);
      comet_debug() << "IndexTreeCompute Operation in Input (t3): ";
      comet_vdump(c3);

      ///  old version for new children ops
      std::vector<mlir::Value> newComputeOps = {c1, c2, c3};
      replaceOperands(itComputeOp.getOperation(), newComputeOps);

      ///  Step 2: split j into 3.
      Operation *needSplitNode = *(newComputeOps[0].getDefiningOp()->getResult(0).getUsers().begin());
      Operation *parentSplitNode = *(needSplitNode->getResult(0).getUsers().begin());
      comet_debug() << " call splitIndicesOp for applying workspace in Input \n";
      comet_pdump(needSplitNode);
      splitIndicesOp(needSplitNode, parentSplitNode->getResult(0), builder, loc);
      comet_debug() << "\n";

    } /// end if(sparseDimsInput.size() == 1)
  }
}

void IndexTreeWorkspaceTransformationsPass::CompressedWorkspaceTransforms(mlir::func::FuncOp funcop)
{
  funcop.walk([](indexTree::IndexTreeOp op)
              {
                OpBuilder builder(op);
                comet_vdump(op);

                Location loc = op.getLoc();

                /// 1. Find its child, until reach the ta.itCompute op
                /// Get first user
                Value computeOp = op.getOperation()->getOperand(0);
                comet_vdump(computeOp);

                /// Only one child??
                /// Build a map, which index is in which IndexTreeIndicesOp
                /// ------ Notice: each index is only in one IndicesOp in original index tree here
                /// ------ TODO(gkestor): handle more complicate cases: one index is in more than one IndicesOp
                /// For an indexTree, the indices ids are
                std::map<int, mlir::Value> indexValueMap;

                while (!(isa<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp())))
                {
                  if (isa<indexTree::IndexTreeIndicesOp>(computeOp.getDefiningOp()))
                  {
                    auto indicesop = dyn_cast<indexTree::IndexTreeIndicesOp>(computeOp.getDefiningOp());
                    ArrayAttr idsArrayAttr = indicesop.getIndices();
                    for (auto n : idsArrayAttr)
                    {
                      int ids = n.cast<mlir::IntegerAttr>().getInt();
                      indexValueMap.emplace(ids, computeOp);
                    }
                  }
                  computeOp = computeOp.getDefiningOp()->getOperand(0); /// put here
                }
                comet_vdump(computeOp);

                /// 2. Check if there is sparse dim in the ta.itCompute op,
                std::vector<std::vector<std::string>> opFormats;
                std::vector<std::vector<int>> opPerms;
                std::vector<std::vector<bool>> inputOutputMapping;
                getFormatsPermsOfComputeOp(computeOp, opFormats, opPerms, inputOutputMapping);

#ifdef DEBUG_MODE_WorkspaceTransformsPass
                comet_debug() << "Print opFormats:\n";
                for (auto n : opFormats)
                {

                  print_vector<std::string>(n);
                }
#endif

                indexTree::IndexTreeComputeOp itComputeOp = dyn_cast<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp());

                /// Check the input tensors, and the output tensor, to see if it contains sparse dimensions
                /// get the dim ids
                std::vector<int> sparseDimsOutput = getSparseDimsOutput(opFormats, opPerms);

#ifdef DEBUG_MODE_WorkspaceTransformsPass
                comet_debug() << " Print sparseDimsOutput: ";
                for (auto p : sparseDimsOutput)
                {
                  comet_debug() << p << " ";
                }
                comet_debug() << "\n";
#endif

                std::vector<struct dimInTensor> sparseDimsInput = getSparseDimsInput(opFormats, opPerms);

                if (sparseDimsOutput.size() == 0 && sparseDimsInput.size() == 0)
                {
                  /// No need to apply workspace transformation
                  comet_debug() << __FILE__ << __LINE__ << " No need to apply workspace transformation\n";
                  return;
                }

                assert(sparseDimsOutput.size() == 1 && " More than one sparse index in the output, we are expecting to support it in the future\n");

                std::vector<Value> newComputeOps;
                /// create three IndexTreeComputeOp op
                /// sparse dim in output tensor
                if (sparseDimsOutput.size() == 1)
                {
                  newComputeOps = CompressedWorkspaceOutput(sparseDimsOutput, itComputeOp, opFormats, opPerms, indexValueMap, builder, op);
                }
    /// initially here workspaceOutput content

#ifdef DEBUG_MODE_WorkspaceTransformsPass
                /// Should notice, the itree has been the new itree already after call workspaceOutput
                for (auto n : newComputeOps)
                {

                  comet_vdump(n);
                }
#endif
                if (sparseDimsInput.size() == 1)
                {
                  comet_vdump(op);
                  /// Need the newComputeOps
                  CompressedWorkspaceInput(newComputeOps, builder, loc);
                }

                /// Also remove previous IndexTreeComputeOp's LHS and RHS.
                indexTree::IndexTreeComputeRHSOp itComputeOp_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(itComputeOp->getOperand(0).getDefiningOp());
                indexTree::IndexTreeComputeLHSOp itComputeOp_lhs = dyn_cast<indexTree::IndexTreeComputeLHSOp>(itComputeOp->getOperand(1).getDefiningOp());

                itComputeOp.erase();
                itComputeOp_rhs.erase();
                itComputeOp_lhs.erase(); }); /// end function traverse

  comet_debug() << __FILE__ << " " << __LINE__ << "CompressedWorkspaceTransforms pass is done\n";
}

void IndexTreeWorkspaceTransformationsPass::runOnOperation()
{
  comet_debug() << __FILE__ << " " << __LINE__ << " starting CompressedWorkspaceTransforms pass \n";
  func::FuncOp function = getOperation();
  /// Traverse the function, only handle ta.itree operation
  CompressedWorkspaceTransforms(function);
  comet_debug() << __FILE__ << " " << __LINE__ << " ending CompressedWorkspaceTransforms pass \n";
}

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeWorkspaceTransformationsPass()
{
  return std::make_unique<IndexTreeWorkspaceTransformationsPass>();
}