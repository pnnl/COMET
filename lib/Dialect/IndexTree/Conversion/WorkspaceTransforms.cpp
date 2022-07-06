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

#include "comet/Dialect/IndexTree/IR/ITDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::IndexTree;
using namespace mlir::tensorAlgebra;

using llvm::SmallVector;
using llvm::StringRef;

#define DEBUG_TYPE "lowering-ta-to-it"

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_WorkspaceTransformsPass
// #define DEBUG_MODE_WorkspaceTransformsPass
// #endif

#ifdef DEBUG_MODE_WorkspaceTransformsPass
#define comet_errs() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n) n->dump()
#define comet_vdump(n) n.dump()
#else
#define comet_errs() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

#define TENSOR_NUMS 3
#define INPUT_TENSOR_NUMS 2

enum OperatorName
{
  eq = 0, // =
  mul_eq, //*=
  add_eq  //+=
};

std::string eq_str = "=";
std::string mul_eq_str = "*=";
std::string add_eq_str = "+=";

const bool workspace = false;
const bool compressedworkspace = true;

struct dimInTensor
{
  int dim;
  int tensorId;
  int dimOrder;
};

/// Apply workspace transformation on the lhs
/// Consider CSR first
/// ikj Cij += Aik * Bkj ===> i (j Wj = 0; kj Wj += Aik * Bkj; kj Cij=Wj)
/// ij Cij = Aij * Bij =====> i (j Wj = 0; j Wj = Aij * Bij; j Cij=Wj)

/// Apply workspace transformation on the rhs
/// Consider CSR first
/// j Wj = Aij * Bij ===> j Vj = 0; j Vj = Bij; j Wj = Aij * Vj;

//===----------------------------------------------------------------------===//
// WorkspaceTransforms PASS
//===----------------------------------------------------------------------===//

/// Apply workspace transformations on the ta.tc and tc.elews_mul
namespace
{
  struct WorkspaceTransformsPass
      : public PassWrapper<WorkspaceTransformsPass, FunctionPass>
  {
    void runOnFunction() final;
    void WorkspaceTransforms(mlir::FuncOp function);
  };

  struct CompressedWorkspaceTransformsPass
      : public PassWrapper<CompressedWorkspaceTransformsPass, FunctionPass>
  {
    void runOnFunction() final;
    void CompressedWorkspaceTransforms(mlir::FuncOp function);
  };

} // end anonymous namespace.

// Need a function, dfs traverse the itree
// get the sparse index that is sparse in the output
std::vector<int> getSparseDimsOutput(std::vector<std::vector<std::string>> opFormats, std::vector<std::vector<int>> opPerms)
{
  std::vector<int> sparseDimsOutput;
  assert(opFormats.size() > 0 && "opFormats.size() less than 0\n");
  std::vector<std::string> outputFormat = opFormats[opFormats.size() - 1];
  std::vector<int> outputPerm = opPerms[opPerms.size() - 1];
  for (unsigned int i = 0; i < outputFormat.size(); i++)
  {
    if (outputFormat[i].compare("D") != 0)
    { // sparse dim
      sparseDimsOutput.push_back(outputPerm[i]);
      comet_errs() << " sparse dim in output: " << outputPerm[i] << "  with format: " << outputFormat[i] << "\n";
    }
  }
  return sparseDimsOutput;
}

// get the sparse index that has sparse format in at least two input tensors
// which tensor, which dimension.  use std::pair represent the information
std::vector<struct dimInTensor> getSparseDimsInput(std::vector<std::vector<std::string>> opFormats, std::vector<std::vector<int>> opPerms)
{
  std::vector<struct dimInTensor> sparseDimsInput;

  std::vector<std::vector<std::string>> inputFormats = {opFormats.begin(), opFormats.end() - 1};
  std::vector<std::vector<int>> inputPerms = {opPerms.begin(), opPerms.end() - 1};

  // Get all dims in input tensors
  std::vector<int> allPermsInput = getUnionOf2Dvector(inputPerms);
  comet_errs() << " allPermsInput.size(): " << allPermsInput.size() << "\n";
  comet_errs() << "allPermsInput: ";
  for (auto n : allPermsInput)
  {
    comet_errs() << n << " ";
  }
  comet_errs() << "\n";

  for (unsigned int i = 0; i < allPermsInput.size(); i++)
  {
    int cur_index = allPermsInput[i];
    comet_errs() << " cur_index: " << cur_index << "\n";
    // Get the format of cur_index from each input tensor
    std::vector<std::string> cur_formats;
    std::vector<int> tensor_ids;
    std::vector<int> dim_orders;
    for (unsigned int j = 0; j < inputPerms.size(); j++)
    {
      unsigned int whichFormat = findIndexInVector(inputPerms[j], cur_index);
      if (whichFormat < inputPerms[j].size())
      { // found
        std::string format = inputFormats[j][whichFormat];
        cur_formats.push_back(format);
        tensor_ids.push_back(j);
        dim_orders.push_back(whichFormat);
      }
    }
    comet_errs() << " cur_formats.size(): " << cur_formats.size() << "\n";
    comet_errs() << "cur_formats: ";
    for (auto n : cur_formats)
    {
      comet_errs() << n << " ";
    }
    comet_errs() << "\n";

    // check if there is sparse format in cur_formats vector
    std::vector<std::string> cur_sparse_formats;
    std::vector<int> sparse_tensor_ids;
    std::vector<int> sparse_dim_orders;
    for (unsigned int j = 0; j < cur_formats.size(); j++)
    {
      comet_errs() << " cur_formats[" << j << "]: " << cur_formats[j] << "\n";
      if (cur_formats[j].compare("D") != 0)
      { // sparse format
        cur_sparse_formats.push_back(cur_formats[j]);
        sparse_tensor_ids.push_back(tensor_ids[j]);
        sparse_dim_orders.push_back(dim_orders[j]);
        comet_errs() << " sparse dim in format: " << cur_index << " with format: " << cur_formats[j] << "\n";
      }
    }

    if (cur_sparse_formats.size() > 1)
    { // More than one sparse format
      // sparseDimsInput.push_back(cur_index);

      struct dimInTensor dim_in_tensor;
      dim_in_tensor.dim = cur_index;
      dim_in_tensor.tensorId = sparse_tensor_ids[0]; // Any sparse tensor is ok
      dim_in_tensor.dimOrder = sparse_dim_orders[0];
      sparseDimsInput.push_back(dim_in_tensor);
    }
  }

  comet_errs() << "sparseDimsInput: ";
  for (auto n : sparseDimsInput)
  {
    // comet_errs() << n << " ";
    comet_errs() << "(" << n.dim << ", " << n.tensorId << ", " << n.dimOrder << ") ";
  }
  comet_errs() << "\n";
  return sparseDimsInput;
}

/// Split one indicesOp into several one, i.e. each computeOp has its own parent op
// i -> j -> V=0;V=A;W=V*B ===> i -> j -> V=0;
//                                -> j -> V=A;
//                                -> j -> W=V*B
void splitIndicesOp(Operation *needSplitNode, Value denseIndicesOp, OpBuilder &builder, Location loc)
{
  while (isa<indexTree::IndexTreeIndicesOp>(needSplitNode))
  {

    comet_pdump(needSplitNode);
    // check how many operands, split into many operands.
    indexTree::IndexTreeIndicesOp indicesOp = dyn_cast<indexTree::IndexTreeIndicesOp>(needSplitNode);

    comet_vdump(indicesOp);

    Operation *indicesOpFirstUsers = *(indicesOp.getOperation()->getResult(0).getUsers().begin());
    // for(auto n : indicesOp.getOperation()->getResult(0).getUsers()){
    //   indicesOpFirstUsers = n;
    // }

    comet_pdump(indicesOpFirstUsers);

    builder.setInsertionPoint(indicesOpFirstUsers);
    comet_errs() << "\n";

    if (needSplitNode != denseIndicesOp.getDefiningOp())
    {
      ArrayAttr indices = indicesOp.indices();

      comet_errs() << " indicesOp.getOperation()->getNumOperands(): " << indicesOp.getOperation()->getNumOperands() << "\n";
      std::vector<mlir::Value> operands;
      std::vector<mlir::Value> newIndicesOp;
      for (unsigned int i = 0; i < indicesOp.getOperation()->getNumOperands(); i++)
      {
        operands.push_back(indicesOp.getOperation()->getOperand(i));

        comet_vdump(indicesOp.getOperation()->getOperand(i));

        comet_vdump(operands[i]);
        auto i64Type = builder.getI64Type();
        Value t1 = builder.create<indexTree::IndexTreeIndicesOp>(loc, i64Type, operands[i], indices);

        comet_vdump(t1);
        newIndicesOp.push_back(t1);
      }

      // put it here
      comet_errs() << " finished calling replacereplaceOperands \n";
      // This parentIndicesOp is the operation that need to be splitted next time

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

      comet_errs() << " plan to erase ";
      comet_vdump(indicesOp);

      comet_pdump(needSplitNode);

      comet_pdump(indicesOpFirstUsers);
      indicesOp.erase();
    }
    else
    {
      comet_errs() << "\n";
      break;
    }
  }
  comet_errs() << "\n";
}

void removeRedundantIndices(std::vector<Value> newComputeOps, std::map<int, mlir::Value> indexValueMap, int denseDimInOutput, OpBuilder &builder, Location loc)
{

  // Check whether need to remove redundant indices or not
  // Get the

  // -------Remove redundant indices-------------
  // For C, the 1st dim i is Dense, the second dim j is sparse.
  // ---- the index including i and before i is not included
  mlir::Value denseIndicesOp = indexValueMap[denseDimInOutput];
  // The indices after denseIndicesOp need to be splitted
  // // start from the computeOp,
  // // Finished one level
  // Only one User, because it's a tree structure, the leaf only has one parent
  assert(newComputeOps[0].getDefiningOp()->getResult(0).hasOneUse() && " the computeOp has more than one users\n");
  // Get the only one user
  Operation *onlyUser = *(newComputeOps[0].getDefiningOp()->getResult(0).getUsers().begin());

  // for(auto u : newComputeOps[0].getDefiningOp()->getResult(0).getUsers()){

  comet_pdump(onlyUser);

  // needSplitNode is the parent node of the "denseIndicesOp"
  Operation *needSplitNode = onlyUser;
  // iterate until the
  // call splitIndicesOp function to split indicesOp until latest "root"
  splitIndicesOp(needSplitNode, denseIndicesOp, builder, loc);
  comet_errs() << "\n";
  // }

  // Remove the indices for each tensor
  // iterate over all itComputeOps, get the indices for each tensor
  // std::vector<mlir::Value> pp = {newComputeOps[0]};
  // for(auto n : pp) {
  for (auto n : newComputeOps)
  {
    // mlir::Value n = newComputeOps[0];
    // get allPerms, put all indices id into a vector,
    // iterater up until reach the root noe, if the index of indicesOp is not in the vector
    //      remove this one: set the operand of the parent of the indicesOp into current op

    comet_errs() << " current computeOp: ";
    comet_vdump(n);
    ArrayAttr allperms_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(n.getDefiningOp()->getOperand(0).getDefiningOp()).allPerms();
    std::vector<std::vector<int>> allpermsInt_rhs = convertArrayAttrIntTo2DVector(allperms_rhs); 
    std::vector<int> permsInt = getUnionOf2Dvector(allpermsInt_rhs);                            
    comet_errs() << " print permsInt: ";
    for (auto p : permsInt)
    {
      comet_errs() << p << " ";
    }
    comet_errs() << "\n";
 
    mlir::Value computeOp = n;

    // iterate over the IndexTreeIndicesOp;

    mlir::Value computeOpParent;

    comet_vdump(n);
    assert(n.getDefiningOp()->getResult(0).hasOneUse() && " indicesOp has more than one user\n");
    Operation *computeOpParentPointer = *(n.getDefiningOp()->getResult(0).getUsers().begin());
    computeOpParent = computeOpParentPointer->getResult(0);

    comet_pdump(computeOpParentPointer);

    comet_vdump(computeOpParent);

    // while(!isa<indexTree::IndexTreeOp>(computeOpParent.getDefiningOp())){
    while (!isRealRoot(computeOpParent.getDefiningOp()))
    {

      comet_vdump(computeOpParent);
      if (isa<indexTree::IndexTreeComputeOp>(computeOpParent.getDefiningOp()))
      {
        comet_errs() << " indicesOp's parent can not be computeOp\n";
      }
      else if (isa<indexTree::IndexTreeOp>(computeOpParent.getDefiningOp()))
      {
        comet_errs() << " indicesOp's parent is IndexTreeOp\n";
      }
      else if (isa<indexTree::IndexTreeIndicesOp>(computeOpParent.getDefiningOp()))
      {

        // get the indices integer, to see if it is in permsInt
        // if yes, don't remove
        // if no, remove:
        indexTree::IndexTreeIndicesOp curIndicesOp = dyn_cast<indexTree::IndexTreeIndicesOp>(computeOpParent.getDefiningOp());
        comet_errs() << " \n";
        ArrayAttr idsArrayAttr = curIndicesOp.indices(); // should be 1D vector
        std::vector<int> idsVec;
        for (auto n : idsArrayAttr)
        {
          idsVec.push_back(n.cast<mlir::IntegerAttr>().getInt());
        }
        comet_errs() << " print idsVec: ";
        for (auto p : idsVec)
        {
          comet_errs() << p << " ";
        }
        comet_errs() << "\n";

        assert(idsVec.size() == 1 && " indicesOp contain more than 1 index\n");
        bool isNeedRemove = false;
        for (auto n : idsVec)
        { // only 1 index actually, because each indicesOp contain one index
          if (std::find(permsInt.begin(), permsInt.end(), n) != permsInt.end())
          {
            // found
            isNeedRemove = false;
          }
          else
          { // the index in curIndicesOp is not found in the computeOp indices
            isNeedRemove = true;
          }
        }

        // if curIndicesOp is the "real root" of the index tree (has only one user)
        //                 contain more than 1 index
        if (curIndicesOp.getOperation()->getNumOperands() > 1 && curIndicesOp.getOperation()->getResult(0).hasOneUse() && isa<indexTree::IndexTreeOp>(*(curIndicesOp.getOperation()->getResult(0).getUsers().begin())))
        {
          isNeedRemove = false;
        }
        comet_errs() << " isNeedRemove = " << isNeedRemove << "\n";

        if (isNeedRemove)
        {
          assert(curIndicesOp.getOperation()->getResult(0).hasOneUse() && " indicesOp has more than one user\n");
          Operation *curIndicesOpParent = *(curIndicesOp.getOperation()->getResult(0).getUsers().begin());

          comet_pdump(curIndicesOpParent);
          comet_vdump(computeOpParent);
          computeOpParent.replaceAllUsesWith(computeOp);
          comet_vdump(computeOpParent);
          comet_pdump(curIndicesOpParent);
          computeOp = computeOpParent;
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
          comet_errs() << " count: " << count << "\n";
          #endif
          assert(curIndicesOp.getOperation()->getResult(0).hasOneUse() && " indicesOp has more than one user\n");
          Operation *curIndicesOpParent = *(curIndicesOp.getOperation()->getResult(0).getUsers().begin());

          comet_pdump(curIndicesOpParent);

          computeOp = computeOpParent;
          computeOpParent = curIndicesOpParent->getResult(0);
        }
      }
    }

  } // end for n
}

std::vector<Value> workspaceOutput(std::vector<int> sparseDimsOutput, indexTree::IndexTreeComputeOp itComputeOp, std::vector<std::vector<std::string>> opFormats, std::vector<std::vector<int>> opPerms, std::string optype, std::map<int, mlir::Value> indexValueMap, OpBuilder &builder, indexTree::IndexTreeOp op)
{
  Location loc = op.getLoc();

  auto opt_type = builder.getBoolAttr(workspace);

  int sparseDimOutput = -1;
  int sparseDimOrderInOutput = -1;
  int denseDimInOutput = -1;
  for (unsigned int j = 0; j < opFormats[opFormats.size() - 1].size(); j++)
  {
    // sparse dimension
    if (opFormats[opFormats.size() - 1][j].compare("D") != 0)
    {
      sparseDimOutput = opPerms[opPerms.size() - 1][j];
      sparseDimOrderInOutput = j;
    }
    else // dense dimension
      denseDimInOutput = opPerms[opPerms.size() - 1][j];
  }
  comet_errs() << " " << sparseDimOutput << "\n";

  // 3. Find the ta.itIndices op which represents sparseDimOutput
  // Find its parent ...
  Value sparseIndicesOp = indexValueMap[sparseDimOutput];

  comet_vdump(sparseIndicesOp);
  // Value denseIndicesOp = indexValueMap[denseDimInOutput];
  // comet_vdump(sparseIndicesOp);

  // Cij = Aik * Bkj ==> Wj = 0; Wj += Aik * Bkj; Cij = Wj
  std::vector<mlir::Value> tensors;
  // for(unsigned int i = 0; i < itComputeOp.getOperation()->getNumOperands(); i++){
  //   tensors.push_back(itComputeOp.getOperation()->getOperand(i));
  // }
  getTensorsOfComputeOp(itComputeOp.getOperation()->getResult(0), tensors); 
  // tensors.push_back(w);// {A, B, C, W}
  //

  // 4. create W, j dim size of
  // Value outputItComputeOp = itComputeOp.getOperation()->getOperand(itComputeOp.getOperation()->getNumOperands() - 1).getDefiningOp()->getOperand(0);
  /// new version
  Value outputItComputeOp = tensors[tensors.size() - 1];

  comet_vdump(outputItComputeOp);

  std::vector<mlir::Value> w_lbls_value = {outputItComputeOp.getDefiningOp()->getOperand(sparseDimOrderInOutput)};

  comet_vdump(outputItComputeOp.getDefiningOp()->getOperand(sparseDimOrderInOutput));
  std::string semiringName(itComputeOp.semiring().data());

  // %w="ta.sparse_tensor_decl"(%2) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?xf64>
  std::string w_format = "Dense"; // tensor<?xf64>
  auto w_type = RankedTensorType::get({mlir::ShapedType::kDynamicSize}, builder.getF64Type());

  // Operation * itComputeOpFirstUsers;
  // for(auto n : computeOp.getUsers()){
  //   itComputeOpFirstUsers = n;
  // }

  Operation *itComputeOpFirstUsers = *(itComputeOp.getOperation()->getUsers().begin());
  builder.setInsertionPoint(itComputeOpFirstUsers); // Insert before itree Op

  mlir::Value w = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, w_type, w_lbls_value, w_format);

  tensors.push_back(w); // {A, B, C, W}

  std::vector<std::vector<std::string>> formats = {opFormats[0], opFormats[1], opFormats[2], {"D"}};
  std::vector<std::vector<int>> perms = {opPerms[0], opPerms[1], opPerms[2], {sparseDimOutput}};

  Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0.0));
  std::vector<mlir::Value> t1_tensors = {const_f64_0, w};
  std::vector<mlir::Value> t2_tensors = {tensors[0], tensors[1], w};
  std::vector<mlir::Value> t3_tensors = {w, tensors[2]};

  std::vector<int> t1_perms_int_0;
  std::vector<int> t1_perms_int_1 = {sparseDimOutput};
  std::vector<std::vector<int>> t1_perms_int = {t1_perms_int_0, t1_perms_int_1};
  //ArrayAttr t1_perms = convert2DVectorToArrayAttrInt(t1_perms_int, builder); 

  std::vector<std::string> t1_formats_str_0;
  std::vector<std::string> t1_formats_str_1 = {"D"};
  std::vector<std::vector<std::string>> t1_formats_str = {t1_formats_str_0, t1_formats_str_1};
  //ArrayAttr t1_formats = convert2DVectorToArrayAttrStr(t1_formats_str, builder); 

  auto i64Type = builder.getI64Type();
  std::vector<mlir::Value> t1_rhs = {t1_tensors[0]};
  mlir::Value t1_lhs = {t1_tensors[1]};
  // auto t1_optype = builder.getIntegerAttr(builder.getIntegerType(64), 0);
  auto t1_optype = builder.getStringAttr(eq_str);
  auto t1_semiring = builder.getStringAttr(semiringName);
  // for t1_rhs
  std::vector<std::vector<int>> t1_rhsop_perms_str = {t1_perms_int_0};
  ArrayAttr t1_rhsop_perms = convert2DVectorToArrayAttrInt(t1_rhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t1_rhsop_formats_str = {t1_formats_str_0};
  ArrayAttr t1_rhsop_formats = convert2DVectorToArrayAttrStr(t1_rhsop_formats_str, builder);
  mlir::Value t1_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t1_rhs, t1_rhsop_perms, t1_rhsop_formats);

  // for t1_lhs
  std::vector<std::vector<int>> t1_lhsop_perms_str = {t1_perms_int_1};
  ArrayAttr t1_lhsop_perms = convert2DVectorToArrayAttrInt(t1_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t1_lhsop_formats_str = {t1_formats_str_1};
  ArrayAttr t1_lhsop_formats = convert2DVectorToArrayAttrStr(t1_lhsop_formats_str, builder);
  mlir::Value t1_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t1_lhs, t1_lhsop_perms, t1_lhsop_formats);

  // for t1
  mlir::Value t1 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t1_rhsop, t1_lhsop, t1_optype, opt_type, t1_semiring);

  std::vector<int> t2_perms_int_0 = opPerms[0];
  std::vector<int> t2_perms_int_1 = opPerms[1];
  std::vector<int> t2_perms_int_2 = {sparseDimOutput};
  std::vector<std::vector<int>> t2_perms_int = {t2_perms_int_0, t2_perms_int_1, t2_perms_int_2};
  //ArrayAttr t2_perms = convert2DVectorToArrayAttrInt(t2_perms_int, builder);

  // Convert formats string array into StrAttr
  std::vector<std::string> t2_formats_str_0 = opFormats[0];
  std::vector<std::string> t2_formats_str_1 = opFormats[1];
  std::vector<std::string> t2_formats_str_2 = {"D"};
  std::vector<std::vector<std::string>> t2_formats_str = {t2_formats_str_0, t2_formats_str_1, t2_formats_str_2};
  //ArrayAttr t2_formats = convert2DVectorToArrayAttrStr(t2_formats_str, builder);
  std::vector<mlir::Value> t2_rhs = {t2_tensors[0], t2_tensors[1]};
  mlir::Value t2_lhs = w;

  auto t2_optype = builder.getStringAttr(optype);
  auto t2_semiring = builder.getStringAttr(semiringName);

  // for t2_rhsop
  std::vector<std::vector<int>> t2_rhsop_perms_str = {t2_perms_int_0, t2_perms_int_1};
  ArrayAttr t2_rhsop_perms = convert2DVectorToArrayAttrInt(t2_rhsop_perms_str, builder); 
  std::vector<std::vector<std::string>> t2_rhsop_formats_str = {t2_formats_str_0, t2_formats_str_1};
  ArrayAttr t2_rhsop_formats = convert2DVectorToArrayAttrStr(t2_rhsop_formats_str, builder);
  mlir::Value t2_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t2_rhs, t2_rhsop_perms, t2_rhsop_formats);

  // for t2_lhsop
  std::vector<std::vector<int>> t2_lhsop_perms_str = {t2_perms_int_2};
  ArrayAttr t2_lhsop_perms = convert2DVectorToArrayAttrInt(t2_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t2_lhsop_formats_str = {t2_formats_str_2};
  ArrayAttr t2_lhsop_formats = convert2DVectorToArrayAttrStr(t2_lhsop_formats_str, builder);
  mlir::Value t2_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t2_lhs, t2_lhsop_perms, t2_lhsop_formats);

  // for t2
  mlir::Value t2 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t2_rhsop, t2_lhsop, t2_optype, opt_type, t2_semiring);

  //  --------t3
  std::vector<int> t3_perms_int_0 = {sparseDimOutput};
  std::vector<int> t3_perms_int_1 = opPerms[2];
  std::vector<std::vector<int>> t3_perms_int = {t3_perms_int_0, t3_perms_int_1};
  //ArrayAttr t3_perms = convert2DVectorToArrayAttrInt(t3_perms_int, builder);

  // Convert formats string array into StrAttr
  std::vector<std::string> t3_formats_str_0 = {"D"};
  std::vector<std::string> t3_formats_str_1 = opFormats[2];
  std::vector<std::vector<std::string>> t3_formats_str = {t3_formats_str_0, t3_formats_str_1};

  //ArrayAttr t3_formats = convert2DVectorToArrayAttrStr(t3_formats_str, builder);
  std::vector<mlir::Value> t3_rhs = {w};
  mlir::Value t3_lhs = t3_tensors[1];
  // auto t3_optype = builder.getIntegerAttr(builder.getIntegerType(64), 0);
  auto t3_optype = builder.getStringAttr(eq_str);
  auto t3_semiring = builder.getStringAttr(semiringName);

  // for t3_rhs
  std::vector<std::vector<int>> t3_rhsop_perms_str = {t3_perms_int_0};
  ArrayAttr t3_rhsop_perms = convert2DVectorToArrayAttrInt(t3_rhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t3_rhsop_formats_str = {t3_formats_str_0};
  ArrayAttr t3_rhsop_formats = convert2DVectorToArrayAttrStr(t3_rhsop_formats_str, builder);
  mlir::Value t3_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t3_rhs, t3_rhsop_perms, t3_rhsop_formats);

  // for t3_lhs
  std::vector<std::vector<int>> t3_lhsop_perms_str = {t3_perms_int_1};
  ArrayAttr t3_lhsop_perms = convert2DVectorToArrayAttrInt(t3_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t3_lhsop_formats_str = {t3_formats_str_1};
  ArrayAttr t3_lhsop_formats = convert2DVectorToArrayAttrStr(t3_lhsop_formats_str, builder);
  mlir::Value t3_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t3_lhs, t3_lhsop_perms, t3_lhsop_formats);

  // for t3
  mlir::Value t3 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t3_rhsop, t3_lhsop, t3_optype, opt_type, t3_semiring);

  std::vector<mlir::Value> newComputeOps = {t1, t2, t3};
  sparseIndicesOp.getDefiningOp()->setOperands(newComputeOps);

  // builder.setInsertionPointAfter(newComputeOps[newComputeOps.size() -1].getDefiningOp());
  // itComputeOp.replaceAllUsesWith(itComputeOp
  // builder.eraseOp(itComputeOp);
  itComputeOp.erase();

  // remove redundant indices by calling a function
  // if elementwise: not remove
  // if spgemm: remove
  // check if there is redundant index
  bool existRedundantIndex = false;
  for (auto n : newComputeOps)
  {
    std::vector<std::vector<int>> perms;
    getPermsOfComputeOp(n, perms);
    std::vector<int> allperms = getUnionOf2Dvector(perms);
    comet_errs() << " print allperms \n";
    print_vector<int>(allperms);

    std::vector<Value> ancestors;
    std::vector<mlir::Value> dfsOps;
    // comet_errs() << "dfsRootOpTree \n";
    dfsRootOpTree(op.children(), dfsOps);
    getAncestorsWp(n, ancestors, dfsOps);
    comet_errs() << " print ancestors \n";
    print_vector_value(ancestors);

    // Iterate over every indicesOp
    for (auto ancestor : ancestors)
    {
      // If indicesOp's index is in allperms, no redundant
      //    the indicesOp is real root, no redundant
      // Otherwise, redundant
      if (isa<indexTree::IndexTreeIndicesOp>(ancestor.getDefiningOp()))
      {
        indexTree::IndexTreeIndicesOp indicesOp = dyn_cast<indexTree::IndexTreeIndicesOp>(ancestor.getDefiningOp());

        ArrayAttr idsArrayAttr = indicesOp.indices(); // should be 1D vector
        // actually only one index for the indicesOp in our implementation
        for (auto m : idsArrayAttr)
        {
          // idsVec.push_back(m.cast<mlir::IntegerAttr>().getInt());
          int perm = m.cast<mlir::IntegerAttr>().getInt();
          comet_errs() << " perm: " << perm << "\n";

          if (findIndexInVector(allperms, perm) == allperms.size())
          { // not exit
            comet_errs() << " perm not exist in allperms\n";
            if (!isRealRoot(indicesOp.getOperation()))
            {
              existRedundantIndex = true;
              comet_errs() << " existRedundantIndex: " << existRedundantIndex << "\n";
            }
          }
        }
      }
    }
  }
  comet_errs() << " existRedundantIndex: " << existRedundantIndex << "\n";

  if (existRedundantIndex)
  {
    removeRedundantIndices(newComputeOps, indexValueMap, denseDimInOutput, builder, loc);
  }

  return newComputeOps;
} // end workspaceOutput()

std::vector<Value> CompressedWorkspaceOutput(std::vector<int> sparseDimsOutput,
                                             indexTree::IndexTreeComputeOp itComputeOp,
                                             std::vector<std::vector<std::string>> opFormats,
                                             std::vector<std::vector<int>> opPerms, std::string optype,
                                             std::map<int, mlir::Value> indexValueMap,
                                             OpBuilder &builder, indexTree::IndexTreeOp op)
{
  Location loc = op.getLoc();

  auto opt_type = builder.getBoolAttr(compressedworkspace);

  int sparseDimOutput = -1;
  int sparseDimOrderInOutput = -1;
  int denseDimInOutput = -1;
  for (unsigned int j = 0; j < opFormats[opFormats.size() - 1].size(); j++)
  {
    // sparse dimension
    if (opFormats[opFormats.size() - 1][j].compare("D") != 0)
    {
      sparseDimOutput = opPerms[opPerms.size() - 1][j];
      sparseDimOrderInOutput = j;
    }
    else // dense dimension
      denseDimInOutput = opPerms[opPerms.size() - 1][j];
  }
  comet_errs() << " " << sparseDimOutput << "\n";

  // 3. Find the ta.itIndices op which represents sparseDimOutput
  // Find its parent ...
  Value sparseIndicesOp = indexValueMap[sparseDimOutput];

  comet_vdump(sparseIndicesOp);
  comet_errs() << " sparseDimOrderInOutput: " << sparseDimOrderInOutput << "\n";
  Value sparseDimsPerent = indexValueMap[sparseDimOrderInOutput - 1];
  comet_errs() << " sparseDimsPerent: ";
  comet_vdump(sparseDimsPerent);

  // Cij = Aik * Bkj ==> Wj = 0; Wj += Aik * Bkj; Cij = Wj
  std::vector<mlir::Value> tensors;
  getTensorsOfComputeOp(itComputeOp.getOperation()->getResult(0), tensors);

  // 4. create W, j dim size of
  // Value outputItComputeOp = itComputeOp.getOperation()->getOperand(itComputeOp.getOperation()->getNumOperands() - 1).getDefiningOp()->getOperand(0);
  /// new version
  Value outputItComputeOp = tensors[tensors.size() - 1];

  comet_vdump(outputItComputeOp);

  std::vector<mlir::Value> w_lbls_value = {outputItComputeOp.getDefiningOp()->getOperand(sparseDimOrderInOutput)};

  comet_vdump(outputItComputeOp.getDefiningOp()->getOperand(sparseDimOrderInOutput));
  // %w="ta.sparse_tensor_decl"(%2) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?xf64>
  std::string w_format = "Dense"; // tensor<?xf64>
  auto w_type = RankedTensorType::get({mlir::ShapedType::kDynamicSize}, builder.getF64Type());

  // Operation * itComputeOpFirstUsers;
  // for(auto n : computeOp.getUsers()){
  //   itComputeOpFirstUsers = n;
  // }

  Operation *itComputeOpFirstUsers = *(itComputeOp.getOperation()->getUsers().begin());
  builder.setInsertionPoint(itComputeOpFirstUsers); // Insert before itree Op

  mlir::Value w = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, w_type, w_lbls_value, w_format);
  auto w_already_set_type = RankedTensorType::get({mlir::ShapedType::kDynamicSize}, builder.getI1Type()); // tensor<?xi1>
  mlir::Value w_already_set = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, w_already_set_type, w_lbls_value, w_format);
  auto w_index_list_type = RankedTensorType::get({mlir::ShapedType::kDynamicSize}, builder.getIndexType()); // tensor<?xindex>
  mlir::Value w_index_list = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, w_index_list_type, w_lbls_value, w_format);

  MemRefType w_index_list_size_type = MemRefType::get({1}, builder.getIndexType());                   // tensor<1xindex>
  mlir::Value w_index_list_size_alloc = builder.create<memref::AllocOp>(loc, w_index_list_size_type); // tensor<1xindex>
  Value w_index_list_size = builder.create<memref::TensorLoadOp>(loc, w_index_list_size_alloc);

  // auto w_index_list_size_type = RankedTensorType::get({1}, builder.getIndexType()); // tensor<1xindex>
  // Value cnst_1_index = builder.create<ConstantIndexOp>(loc, 1);
  // std::vector<Value> w_index_list_size_lbls = {};
  // // %w_index_list_size = "ta.dense_tensor_decl"() {format = "Dense"} : () -> tensor<1xindex>
  // mlir::Value w_index_list_size = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, w_index_list_size_type, w_index_list_size_lbls, w_format); // tensor<1xindex>

  // // ///// not working: use memref.alloc and memref.tensor_load to declare the workspace arrays
  // auto w_type = MemRefType::get({mlir::ShapedType::kDynamicSize}, builder.getF64Type());
  // Value w_alloc = builder.create<memref::AllocOp>(loc, w_type, ValueRange{w_lbls_value});
  // Value w = builder.create<memref::TensorLoadOp>(loc, w_alloc);

  // // %w = alloc() : memref<4xf64>
  // // %w = alloc(%size) : memref<?xf64>

  // auto w_already_set_type = MemRefType::get({mlir::ShapedType::kDynamicSize}, builder.getI1Type()); // tensor<?xi1>
  // Value w_already_set_alloc = builder.create<memref::AllocOp>(loc, w_already_set_type, ValueRange{w_lbls_value});
  // Value w_already_set = builder.create<memref::TensorLoadOp>(loc, w_already_set_alloc);

  // auto w_index_list_type = MemRefType::get({mlir::ShapedType::kDynamicSize}, builder.getIndexType()); // tensor<?xindex>
  // Value w_index_list_alloc = builder.create<memref::AllocOp>(loc, w_index_list_type, ValueRange{w_lbls_value});
  // Value w_index_list = builder.create<memref::TensorLoadOp>(loc, w_index_list_alloc);

  // auto w_index_list_size_type = MemRefType::get({1}, builder.getIndexType()); // tensor<1xindex>
  // Value w_index_list_size_alloc = builder.create<memref::AllocOp>(loc, w_index_list_size_type); // tensor<1xindex>
  // Value w_index_list_size = builder.create<memref::TensorLoadOp>(loc, w_index_list_size_alloc);

  //
  std::vector<Value> workspaceTensors = {w, w_already_set, w_index_list, w_index_list_size};

  tensors.push_back(w); // {A, B, C, W}

  std::vector<std::vector<std::string>> formats = {opFormats[0], opFormats[1], opFormats[2], {"D"}};
  std::vector<std::vector<int>> perms = {opPerms[0], opPerms[1], opPerms[2], {sparseDimOutput}};

  //////////  For t0: w_index_list_size = 0; // tensor<1xindex>
  std::vector<int> t0_perms_int_0;
  std::vector<int> t0_perms_int_1;
  std::vector<std::vector<int>> t0_perms_int = {t0_perms_int_0, t0_perms_int_1};
  //ArrayAttr t0_perms = convert2DVectorToArrayAttrInt(t0_perms_int, builder);

  std::vector<std::string> t0_formats_str_0;
  std::vector<std::string> t0_formats_str_1; // = {"D"};
  std::vector<std::vector<std::string>> t0_formats_str = {t0_formats_str_0, t0_formats_str_1};
  //ArrayAttr t0_formats = convert2DVectorToArrayAttrStr(t0_formats_str, builder);

  // auto i64Type = builder.getI64Type();
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  std::vector<mlir::Value> t0_rhs = {const_index_0};
  mlir::Value t0_lhs = {w_index_list_size};
  auto t0_optype = builder.getStringAttr("=");
  std::string semiringName(itComputeOp.semiring().data());
  auto t0_semiring = builder.getStringAttr(semiringName);

  // for t0_rhs
  std::vector<std::vector<int>> t0_rhsop_perms_str = {t0_perms_int_0};
  ArrayAttr t0_rhsop_perms = convert2DVectorToArrayAttrInt(t0_rhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t0_rhsop_formats_str = {t0_formats_str_0};
  ArrayAttr t0_rhsop_formats = convert2DVectorToArrayAttrStr(t0_rhsop_formats_str, builder);
  mlir::Value t0_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getIndexType()), t0_rhs, t0_rhsop_perms, t0_rhsop_formats);

  // for t0_lhs
  std::vector<std::vector<int>> t0_lhsop_perms_str = {t0_perms_int_1};
  ArrayAttr t0_lhsop_perms = convert2DVectorToArrayAttrInt(t0_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t0_lhsop_formats_str = {t0_formats_str_1};
  ArrayAttr t0_lhsop_formats = convert2DVectorToArrayAttrStr(t0_lhsop_formats_str, builder);
  mlir::Value t0_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t0_lhs, t0_lhsop_perms, t0_lhsop_formats);

  // for t1
  mlir::Value t0 = builder.create<indexTree::IndexTreeComputeOp>(loc, builder.getI64Type(), t0_rhsop, t0_lhsop, t0_optype, opt_type, t0_semiring);

  // insert t0 to sparseDimsParent
  sparseDimsPerent.getDefiningOp()->insertOperands(0, t0);
  //////////  for t0 end

  Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0.0));
  std::vector<mlir::Value> t1_tensors = {const_f64_0, w};
  std::vector<mlir::Value> t2_tensors = {tensors[0], tensors[1], w};
  std::vector<mlir::Value> t3_tensors = {w, tensors[2]};

  std::vector<int> t1_perms_int_0;
  std::vector<int> t1_perms_int_1 = {sparseDimOutput};
  std::vector<std::vector<int>> t1_perms_int = {t1_perms_int_0, t1_perms_int_1};
  //ArrayAttr t1_perms = convert2DVectorToArrayAttrInt(t1_perms_int, builder);

  std::vector<std::string> t1_formats_str_0;
  std::vector<std::string> t1_formats_str_1 = {"D"};
  std::vector<std::vector<std::string>> t1_formats_str = {t1_formats_str_0, t1_formats_str_1};
  //ArrayAttr t1_formats = convert2DVectorToArrayAttrStr(t1_formats_str, builder);

  auto i64Type = builder.getI64Type();
  std::vector<mlir::Value> t1_rhs = {t1_tensors[0]};
  // mlir::Value t1_lhs = {t1_tensors[1]};
  std::vector<mlir::Value> t1_lhs = workspaceTensors;
  auto t1_optype = builder.getStringAttr(eq_str);
  auto t1_semiring = builder.getStringAttr(semiringName);

  // for t1_rhs
  std::vector<std::vector<int>> t1_rhsop_perms_str = {t1_perms_int_0};
  ArrayAttr t1_rhsop_perms = convert2DVectorToArrayAttrInt(t1_rhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t1_rhsop_formats_str = {t1_formats_str_0};
  ArrayAttr t1_rhsop_formats = convert2DVectorToArrayAttrStr(t1_rhsop_formats_str, builder);
  mlir::Value t1_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t1_rhs, t1_rhsop_perms, t1_rhsop_formats);

  // for t1_lhs
  std::vector<std::vector<int>> t1_lhsop_perms_str = {t1_perms_int_1};
  ArrayAttr t1_lhsop_perms = convert2DVectorToArrayAttrInt(t1_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t1_lhsop_formats_str = {t1_formats_str_1};
  ArrayAttr t1_lhsop_formats = convert2DVectorToArrayAttrStr(t1_lhsop_formats_str, builder);
  mlir::Value t1_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t1_lhs, t1_lhsop_perms, t1_lhsop_formats);

  // for t1
  mlir::Value t1 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t1_rhsop, t1_lhsop, t1_optype, opt_type, t1_semiring);

  std::vector<int> t2_perms_int_0 = opPerms[0];
  std::vector<int> t2_perms_int_1 = opPerms[1];
  std::vector<int> t2_perms_int_2 = {sparseDimOutput};
  std::vector<std::vector<int>> t2_perms_int = {t2_perms_int_0, t2_perms_int_1, t2_perms_int_2};
  //ArrayAttr t2_perms = convert2DVectorToArrayAttrInt(t2_perms_int, builder);

  // Convert formats string array into StrAttr
  std::vector<std::string> t2_formats_str_0 = opFormats[0];
  std::vector<std::string> t2_formats_str_1 = opFormats[1];
  std::vector<std::string> t2_formats_str_2 = {"D"};
  std::vector<std::vector<std::string>> t2_formats_str = {t2_formats_str_0, t2_formats_str_1, t2_formats_str_2};
  //ArrayAttr t2_formats = convert2DVectorToArrayAttrStr(t2_formats_str, builder);
  std::vector<mlir::Value> t2_rhs = {t2_tensors[0], t2_tensors[1]};
  // mlir::Value t2_lhs = w;
  std::vector<mlir::Value> t2_lhs = workspaceTensors;

  auto t2_optype = builder.getStringAttr(optype);
  auto t2_semiring = builder.getStringAttr(semiringName);

  // for t2_rhsop
  std::vector<std::vector<int>> t2_rhsop_perms_str = {t2_perms_int_0, t2_perms_int_1};
  ArrayAttr t2_rhsop_perms = convert2DVectorToArrayAttrInt(t2_rhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t2_rhsop_formats_str = {t2_formats_str_0, t2_formats_str_1};
  ArrayAttr t2_rhsop_formats = convert2DVectorToArrayAttrStr(t2_rhsop_formats_str, builder);
  mlir::Value t2_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t2_rhs, t2_rhsop_perms, t2_rhsop_formats);

  // for t2_lhsop
  std::vector<std::vector<int>> t2_lhsop_perms_str = {t2_perms_int_2};
  ArrayAttr t2_lhsop_perms = convert2DVectorToArrayAttrInt(t2_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t2_lhsop_formats_str = {t2_formats_str_2};
  ArrayAttr t2_lhsop_formats = convert2DVectorToArrayAttrStr(t2_lhsop_formats_str, builder);
  mlir::Value t2_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t2_lhs, t2_lhsop_perms, t2_lhsop_formats);

  // for t2
  mlir::Value t2 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t2_rhsop, t2_lhsop, t2_optype, opt_type, t2_semiring);

  //  --------t3
  std::vector<int> t3_perms_int_0 = {sparseDimOutput};
  std::vector<int> t3_perms_int_1 = opPerms[2];
  std::vector<std::vector<int>> t3_perms_int = {t3_perms_int_0, t3_perms_int_1};
  //ArrayAttr t3_perms = convert2DVectorToArrayAttrInt(t3_perms_int, builder);

  // Convert formats string array into StrAttr
  std::vector<std::string> t3_formats_str_0 = {"D"};
  std::vector<std::string> t3_formats_str_1 = opFormats[2];
  std::vector<std::vector<std::string>> t3_formats_str = {t3_formats_str_0, t3_formats_str_1};

  //ArrayAttr t3_formats = convert2DVectorToArrayAttrStr(t3_formats_str, builder);
  // std::vector<mlir::Value> t3_rhs = {w};
  std::vector<mlir::Value> t3_rhs = workspaceTensors;
  mlir::Value t3_lhs = t3_tensors[1];
  auto t3_optype = builder.getStringAttr(eq_str);
  auto t3_semiring = builder.getStringAttr(semiringName);

  // for t3_rhs
  std::vector<std::vector<int>> t3_rhsop_perms_str = {t3_perms_int_0};
  ArrayAttr t3_rhsop_perms = convert2DVectorToArrayAttrInt(t3_rhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t3_rhsop_formats_str = {t3_formats_str_0};
  ArrayAttr t3_rhsop_formats = convert2DVectorToArrayAttrStr(t3_rhsop_formats_str, builder);
  mlir::Value t3_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t3_rhs, t3_rhsop_perms, t3_rhsop_formats);

  // for t3_lhs
  std::vector<std::vector<int>> t3_lhsop_perms_str = {t3_perms_int_1};
  ArrayAttr t3_lhsop_perms = convert2DVectorToArrayAttrInt(t3_lhsop_perms_str, builder);
  std::vector<std::vector<std::string>> t3_lhsop_formats_str = {t3_formats_str_1};
  ArrayAttr t3_lhsop_formats = convert2DVectorToArrayAttrStr(t3_lhsop_formats_str, builder);
  mlir::Value t3_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t3_lhs, t3_lhsop_perms, t3_lhsop_formats);

  // for t3
  mlir::Value t3 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t3_rhsop, t3_lhsop, t3_optype, opt_type, t3_semiring);
  std::vector<mlir::Value> newComputeOps = {t1, t2, t3};
  sparseIndicesOp.getDefiningOp()->setOperands(newComputeOps);

  // builder.setInsertionPointAfter(newComputeOps[newComputeOps.size() -1].getDefiningOp());
  // itComputeOp.replaceAllUsesWith(itComputeOp
  // builder.eraseOp(itComputeOp);
  itComputeOp.erase();

  // remove redundant indices by calling a function
  // in elementwise: not remove
  // in spgemm: remove
  // check if there is redundant index
  bool existRedundantIndex = false;
  for (auto n : newComputeOps)
  {
    std::vector<std::vector<int>> perms;
    getPermsOfComputeOp(n, perms);
    std::vector<int> allperms = getUnionOf2Dvector(perms);
    comet_errs() << " print allperms \n";
    print_vector<int>(allperms);

    std::vector<Value> ancestors;
    std::vector<mlir::Value> dfsOps;
    // comet_errs() << "dfsRootOpTree \n";
    dfsRootOpTree(op.children(), dfsOps);
    getAncestorsWp(n, ancestors, dfsOps);
    comet_errs() << " print ancestors \n";
    print_vector_value(ancestors);

    // Iterate over every indicesOp
    for (auto ancestor : ancestors)
    {
      // If indicesOp's index is in allperms, no redundant
      //    the indicesOp is real root, no redundant
      // Otherwise, redundant
      if (isa<indexTree::IndexTreeIndicesOp>(ancestor.getDefiningOp()))
      {
        indexTree::IndexTreeIndicesOp indicesOp = dyn_cast<indexTree::IndexTreeIndicesOp>(ancestor.getDefiningOp());

        ArrayAttr idsArrayAttr = indicesOp.indices(); // should be 1D vector
        // actually only one index for the indicesOp in our implementation
        for (auto m : idsArrayAttr)
        {
          // idsVec.push_back(m.cast<mlir::IntegerAttr>().getInt());
          int perm = m.cast<mlir::IntegerAttr>().getInt();
          comet_errs() << " perm: " << perm << "\n";

          if (findIndexInVector(allperms, perm) == allperms.size())
          { // not exit
            comet_errs() << " perm not exist in allperms\n";
            if (!isRealRoot(indicesOp.getOperation()))
            {
              existRedundantIndex = true;
              comet_errs() << " existRedundantIndex: " << existRedundantIndex << "\n";
            }
          }
        }
      }
    }
  }
  comet_errs() << " existRedundantIndex: " << existRedundantIndex << "\n";

  if (existRedundantIndex)
  {
    removeRedundantIndices(newComputeOps, indexValueMap, denseDimInOutput, builder, loc);
  }

  return newComputeOps;
} // end CompressedWorkspaceOutput()

void workspaceInput(std::vector<Value> computeOps, OpBuilder &builder, Location loc)
{
  auto opt_type = builder.getBoolAttr(workspace);
  for (auto computeOp : computeOps)
  {

    /// 1. get the opFormats and opPerms of the computeOp
    std::vector<std::vector<std::string>> opFormats;
    std::vector<std::vector<int>> opPerms;
    std::vector<std::vector<bool>> inputOutputMapping;
    getFormatsPermsOfComputeOp(computeOp, opFormats, opPerms, inputOutputMapping);
    comet_errs() << " \n";
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
    indexTree::IndexTreeComputeOp itComputeOp = dyn_cast<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp());
    // int optype = itComputeOp.op_type();
    std::string optype(itComputeOp.op_type().data());
    std::vector<int> sparseDimsOutput = getSparseDimsOutput(opFormats, opPerms);
    std::vector<struct dimInTensor> sparseDimsInput = getSparseDimsInput(opFormats, opPerms);
    comet_errs() << " sparseDimsInput.size(): " << sparseDimsInput.size() << "\n";
    // // get sparseDimsInWhichTensor
    // std::vector<Value> sparseDimsWhichInput = getSparseDimsWhichInput(opFormats, opPerms);;

    comet_vdump(computeOp);
    if (sparseDimsInput.size() == 1)
    { // solve only 1 sparseDimsInput
      // No need to apply workspace transformation
      // comet_errs() << " sparseDimsInput[0]: " << sparseDimsInput[0] << "\n";
      comet_errs() << " sparseDimsInput[0]: " << sparseDimsInput[0].dim << ", " << sparseDimsInput[0].tensorId << ", " << sparseDimsInput[0].dimOrder << "\n";
      // return;
      // Wj=Aij*Bij => Vj=0; Vj=Aij; Wj=Vj*Bij
      // Value sparseInput = itComputeOp.getOperation()->getOperand(sparseDimsInput[0].tensorId);
      /// new version
      Value sparseInput = tensors[sparseDimsInput[0].tensorId];

      comet_vdump(sparseInput);

      std::vector<mlir::Value> v_lbls_value = {sparseInput.getDefiningOp()->getOperand(sparseDimsInput[0].dimOrder)};

      comet_vdump(sparseInput.getDefiningOp()->getOperand(sparseDimsInput[0].dimOrder));
      // %v="ta.dense_tensor_decl"(%1) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
      std::string v_format = "Dense"; // tensor<?xf64>
      auto v_type = RankedTensorType::get({mlir::ShapedType::kDynamicSize}, builder.getF64Type());

      builder.setInsertionPoint(computeOp.getDefiningOp());
      mlir::Value v = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, v_type, v_lbls_value, v_format);

      comet_vdump(v);

      // Vj=0
      std::vector<int> t1_perms_int_0;
      std::vector<int> t1_perms_int_1 = {sparseDimsInput[0].dim};
      std::vector<std::vector<int>> t1_perms_int = {t1_perms_int_0, t1_perms_int_1};
      //ArrayAttr t1_perms = convert2DVectorToArrayAttrInt(t1_perms_int, builder);

      std::vector<std::string> t1_formats_str_0;
      std::vector<std::string> t1_formats_str_1 = {"D"};
      std::vector<std::vector<std::string>> t1_formats_str = {t1_formats_str_0, t1_formats_str_1};
      //ArrayAttr t1_formats = convert2DVectorToArrayAttrStr(t1_formats_str, builder);

      auto i64Type = builder.getI64Type();
      Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0.0));
      std::vector<mlir::Value> t1_rhs = {const_f64_0};
      mlir::Value t1_lhs = {v};
      auto t1_optype = builder.getStringAttr(eq_str);
      std::string semiringName(itComputeOp.semiring().data());
      auto t1_semiring = builder.getStringAttr(semiringName);

      // for t1_rhs
      std::vector<std::vector<int>> t1_rhsop_perms_str = {t1_perms_int_0};
      ArrayAttr t1_rhsop_perms = convert2DVectorToArrayAttrInt(t1_rhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t1_rhsop_formats_str = {t1_formats_str_0};
      ArrayAttr t1_rhsop_formats = convert2DVectorToArrayAttrStr(t1_rhsop_formats_str, builder);
      mlir::Value t1_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t1_rhs, t1_rhsop_perms, t1_rhsop_formats);

      // for t1_lhs
      std::vector<std::vector<int>> t1_lhsop_perms_str = {t1_perms_int_1};
      ArrayAttr t1_lhsop_perms = convert2DVectorToArrayAttrInt(t1_lhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t1_lhsop_formats_str = {t1_formats_str_1};
      ArrayAttr t1_lhsop_formats = convert2DVectorToArrayAttrStr(t1_lhsop_formats_str, builder);
      mlir::Value t1_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t1_lhs, t1_lhsop_perms, t1_lhsop_formats);

      // for t1
      mlir::Value t1 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t1_rhsop, t1_lhsop, t1_optype, opt_type, t1_semiring);

      // Vj = Aij
      std::vector<int> t2_perms_int_0 = opPerms[sparseDimsInput[0].tensorId];
      std::vector<int> t2_perms_int_1 = {sparseDimsInput[0].dim};
      std::vector<std::vector<int>> t2_perms_int = {t2_perms_int_0, t2_perms_int_1};
      //ArrayAttr t2_perms = convert2DVectorToArrayAttrInt(t2_perms_int, builder);

      std::vector<std::string> t2_formats_str_0 = opFormats[sparseDimsInput[0].tensorId];
      std::vector<std::string> t2_formats_str_1 = {"D"};
      std::vector<std::vector<std::string>> t2_formats_str = {t2_formats_str_0, t2_formats_str_1};
      //ArrayAttr t2_formats = convert2DVectorToArrayAttrStr(t2_formats_str, builder);

      std::vector<mlir::Value> t2_rhs = {tensors[sparseDimsInput[0].tensorId]};

      mlir::Value t2_lhs = {v};
      auto t2_optype = builder.getStringAttr(eq_str);
      auto t2_semiring = builder.getStringAttr(semiringName);

      // for t2_rhs
      std::vector<std::vector<int>> t2_rhsop_perms_str = {t2_perms_int_0};
      ArrayAttr t2_rhsop_perms = convert2DVectorToArrayAttrInt(t2_rhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t2_rhsop_formats_str = {t2_formats_str_0};
      ArrayAttr t2_rhsop_formats = convert2DVectorToArrayAttrStr(t2_rhsop_formats_str, builder);
      mlir::Value t2_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t2_rhs, t2_rhsop_perms, t2_rhsop_formats);

      // for t2_lhs
      std::vector<std::vector<int>> t2_lhsop_perms_str = {t2_perms_int_1};
      ArrayAttr t2_lhsop_perms = convert2DVectorToArrayAttrInt(t2_lhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t2_lhsop_formats_str = {t2_formats_str_1};
      ArrayAttr t2_lhsop_formats = convert2DVectorToArrayAttrStr(t2_lhsop_formats_str, builder);
      mlir::Value t2_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t2_lhs, t2_lhsop_perms, t2_lhsop_formats);

      // for t2
      mlir::Value t2 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t2_rhsop, t2_lhsop, t2_optype, opt_type, t2_semiring);

      // Wj=Vj*Bij
      std::vector<int> t3_perms_int_0 = {sparseDimsInput[0].dim};
      std::vector<int> t3_perms_int_1 = opPerms[1]; 
      std::vector<int> t3_perms_int_2 = opPerms[opPerms.size() - 1];
      std::vector<std::vector<int>> t3_perms_int = {t3_perms_int_0, t3_perms_int_1, t3_perms_int_2};
      //ArrayAttr t3_perms = convert2DVectorToArrayAttrInt(t3_perms_int, builder);

      // Convert formats string array into StrAttr
      std::vector<std::string> t3_formats_str_0 = {"D"};
      std::vector<std::string> t3_formats_str_1 = opFormats[1];
      std::vector<std::string> t3_formats_str_2 = opFormats[opFormats.size() - 1];
      ;
      std::vector<std::vector<std::string>> t3_formats_str = {t3_formats_str_0, t3_formats_str_1, t3_formats_str_2};
      //ArrayAttr t3_formats = convert2DVectorToArrayAttrStr(t3_formats_str, builder);
      std::vector<mlir::Value> t3_rhs = {v, tensors[1]};

      mlir::Value t3_lhs = {tensors[tensors.size() - 1]};

      auto t3_optype = builder.getStringAttr(optype);
      auto t3_semiring = builder.getStringAttr(semiringName);

      // for t3_rhs
      std::vector<std::vector<int>> t3_rhsop_perms_str = {t3_perms_int_0, t3_perms_int_1};
      ArrayAttr t3_rhsop_perms = convert2DVectorToArrayAttrInt(t3_rhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t3_rhsop_formats_str = {t3_formats_str_0, t3_formats_str_1};
      ArrayAttr t3_rhsop_formats = convert2DVectorToArrayAttrStr(t3_rhsop_formats_str, builder);
      mlir::Value t3_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t3_rhs, t3_rhsop_perms, t3_rhsop_formats);

      // for t3_lhs
      std::vector<std::vector<int>> t3_lhsop_perms_str = {t3_perms_int_2};
      ArrayAttr t3_lhsop_perms = convert2DVectorToArrayAttrInt(t3_lhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t3_lhsop_formats_str = {t3_formats_str_2};
      ArrayAttr t3_lhsop_formats = convert2DVectorToArrayAttrStr(t3_lhsop_formats_str, builder);
      mlir::Value t3_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t3_lhs, t3_lhsop_perms, t3_lhsop_formats);

      // for t3
      mlir::Value t3 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t3_rhsop, t3_lhsop, t3_optype, opt_type, t3_semiring);

      std::vector<mlir::Value> newComputeOps = {t1, t2, t3};

      replaceOperands(itComputeOp.getOperation(), newComputeOps); 

      /// Step 2: split j into 3.
      Operation *needSplitNode = *(newComputeOps[0].getDefiningOp()->getResult(0).getUsers().begin());
      Operation *parentSplitNode = *(needSplitNode->getResult(0).getUsers().begin());
      comet_errs() << " call splitIndicesOp for applying workspace in Input \n";
      comet_pdump(needSplitNode);
      splitIndicesOp(needSplitNode, parentSplitNode->getResult(0), builder, loc);
      comet_errs() << "\n";

    } // end if(sparseDimsInput.size() == 1)
  }
}

void CompressedWorkspaceInput(std::vector<Value> computeOps, OpBuilder &builder, Location loc)
{
  auto opt_type = builder.getBoolAttr(workspace);
  for (auto computeOp : computeOps)
  {

    /// 1. get the opFormats and opPerms of the computeOp
    std::vector<std::vector<std::string>> opFormats;
    std::vector<std::vector<int>> opPerms;
    std::vector<std::vector<bool>> inputOutputMapping;
    getFormatsPermsOfComputeOp(computeOp, opFormats, opPerms, inputOutputMapping);
    comet_errs() << " \n";
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
    comet_errs() << " tensors.size(): " << tensors.size() << "\n";
    std::vector<Value> tensors_rhs;
    getInputTensorsOfComputeOp(computeOp, tensors_rhs);
    comet_errs() << " tensors_rhs.size(): " << tensors_rhs.size() << "\n";
    std::vector<Value> tensors_lhs;
    getOutputTensorsOfComputeOp(computeOp, tensors_lhs);
    comet_errs() << " tensors_lhs.size(): " << tensors_lhs.size() << "\n";

    indexTree::IndexTreeComputeOp itComputeOp = dyn_cast<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp());
    // int optype = itComputeOp.op_type();
    std::string optype(itComputeOp.op_type().data());
    std::string semiringName(itComputeOp.semiring().data());

    std::vector<int> sparseDimsOutput = getSparseDimsOutput(opFormats, opPerms);
    // std::vector<int>
    std::vector<struct dimInTensor> sparseDimsInput = getSparseDimsInput(opFormats, opPerms);
    comet_errs() << " sparseDimsInput.size(): " << sparseDimsInput.size() << "\n";
    // // get sparseDimsInWhichTensor
    // std::vector<Value> sparseDimsWhichInput = getSparseDimsWhichInput(opFormats, opPerms);;
    if (sparseDimsInput.size() == 1)
    { // solve only 1 sparseDimsInput
      // No need to apply workspace transformation
      // comet_errs() << " sparseDimsInput[0]: " << sparseDimsInput[0] << "\n";
      comet_errs() << " sparseDimsInput[0]: " << sparseDimsInput[0].dim << ", " << sparseDimsInput[0].tensorId << ", " << sparseDimsInput[0].dimOrder << "\n";
      // return;
      // Wj=Aij*Bij => Vj=0; Vj=Aij; Wj=Vj*Bij
      // Value sparseInput = itComputeOp.getOperation()->getOperand(sparseDimsInput[0].tensorId);
      /// new version
      // Value sparseInput = tensors[sparseDimsInput[0].tensorId];
      Value sparseInput = tensors_rhs[sparseDimsInput[0].tensorId];

      comet_vdump(sparseInput);

      std::vector<mlir::Value> v_lbls_value = {sparseInput.getDefiningOp()->getOperand(sparseDimsInput[0].dimOrder)};

      comet_vdump(sparseInput.getDefiningOp()->getOperand(sparseDimsInput[0].dimOrder));
      // %v="ta.dense_tensor_decl"(%1) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
      std::string v_format = "Dense"; // tensor<?xf64>
      auto v_type = RankedTensorType::get({mlir::ShapedType::kDynamicSize}, builder.getF64Type());

      builder.setInsertionPoint(computeOp.getDefiningOp());
      mlir::Value v = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc, v_type, v_lbls_value, v_format);

      comet_vdump(v);

      // Vj=0
      std::vector<int> t1_perms_int_0;
      std::vector<int> t1_perms_int_1 = {sparseDimsInput[0].dim};
      std::vector<std::vector<int>> t1_perms_int = {t1_perms_int_0, t1_perms_int_1};
      //ArrayAttr t1_perms = convert2DVectorToArrayAttrInt(t1_perms_int, builder);

      std::vector<std::string> t1_formats_str_0;
      std::vector<std::string> t1_formats_str_1 = {"D"};
      std::vector<std::vector<std::string>> t1_formats_str = {t1_formats_str_0, t1_formats_str_1};
      //ArrayAttr t1_formats = convert2DVectorToArrayAttrStr(t1_formats_str, builder);

      auto i64Type = builder.getI64Type();
      Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0.0));
      std::vector<mlir::Value> t1_rhs = {const_f64_0};
      mlir::Value t1_lhs = {v};
      auto t1_optype = builder.getStringAttr(eq_str);
      std::string semiringName(itComputeOp.semiring().data());
      auto t1_semiring = builder.getStringAttr(semiringName);

      // for t1_rhs
      std::vector<std::vector<int>> t1_rhsop_perms_str = {t1_perms_int_0};
      ArrayAttr t1_rhsop_perms = convert2DVectorToArrayAttrInt(t1_rhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t1_rhsop_formats_str = {t1_formats_str_0};
      ArrayAttr t1_rhsop_formats = convert2DVectorToArrayAttrStr(t1_rhsop_formats_str, builder);
      mlir::Value t1_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t1_rhs, t1_rhsop_perms, t1_rhsop_formats);

      // for t1_lhs
      std::vector<std::vector<int>> t1_lhsop_perms_str = {t1_perms_int_1};
      ArrayAttr t1_lhsop_perms = convert2DVectorToArrayAttrInt(t1_lhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t1_lhsop_formats_str = {t1_formats_str_1};
      ArrayAttr t1_lhsop_formats = convert2DVectorToArrayAttrStr(t1_lhsop_formats_str, builder);
      mlir::Value t1_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t1_lhs, t1_lhsop_perms, t1_lhsop_formats);

      // for t1
      mlir::Value t1 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t1_rhsop, t1_lhsop, t1_optype, opt_type, t1_semiring);

      // Vj = Aij
      std::vector<int> t2_perms_int_0 = opPerms[sparseDimsInput[0].tensorId];
      std::vector<int> t2_perms_int_1 = {sparseDimsInput[0].dim};
      std::vector<std::vector<int>> t2_perms_int = {t2_perms_int_0, t2_perms_int_1};
      //ArrayAttr t2_perms = convert2DVectorToArrayAttrInt(t2_perms_int, builder);

      std::vector<std::string> t2_formats_str_0 = opFormats[sparseDimsInput[0].tensorId];
      std::vector<std::string> t2_formats_str_1 = {"D"};
      std::vector<std::vector<std::string>> t2_formats_str = {t2_formats_str_0, t2_formats_str_1};
      //ArrayAttr t2_formats = convert2DVectorToArrayAttrStr(t2_formats_str, builder);

      std::vector<mlir::Value> t2_rhs = {tensors_rhs[sparseDimsInput[0].tensorId]};

      mlir::Value t2_lhs = {v};
      auto t2_optype = builder.getStringAttr(eq_str);
      auto t2_semiring = builder.getStringAttr(semiringName);
      // for t2_rhs
      std::vector<std::vector<int>> t2_rhsop_perms_str = {t2_perms_int_0};
      ArrayAttr t2_rhsop_perms = convert2DVectorToArrayAttrInt(t2_rhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t2_rhsop_formats_str = {t2_formats_str_0};
      ArrayAttr t2_rhsop_formats = convert2DVectorToArrayAttrStr(t2_rhsop_formats_str, builder);
      mlir::Value t2_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t2_rhs, t2_rhsop_perms, t2_rhsop_formats);

      // for t2_lhs
      std::vector<std::vector<int>> t2_lhsop_perms_str = {t2_perms_int_1};
      ArrayAttr t2_lhsop_perms = convert2DVectorToArrayAttrInt(t2_lhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t2_lhsop_formats_str = {t2_formats_str_1};
      ArrayAttr t2_lhsop_formats = convert2DVectorToArrayAttrStr(t2_lhsop_formats_str, builder);
      mlir::Value t2_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t2_lhs, t2_lhsop_perms, t2_lhsop_formats);

      // for t2
      mlir::Value t2 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t2_rhsop, t2_lhsop, t2_optype, opt_type, t2_semiring);

      // Wj=Vj*Bij
      std::vector<int> t3_perms_int_0 = {sparseDimsInput[0].dim};
      std::vector<int> t3_perms_int_1 = opPerms[1]; 
      std::vector<int> t3_perms_int_2 = opPerms[opPerms.size() - 1];
      std::vector<std::vector<int>> t3_perms_int = {t3_perms_int_0, t3_perms_int_1, t3_perms_int_2};
      //ArrayAttr t3_perms = convert2DVectorToArrayAttrInt(t3_perms_int, builder);

      // Convert formats string array into StrAttr
      std::vector<std::string> t3_formats_str_0 = {"D"};
      std::vector<std::string> t3_formats_str_1 = opFormats[1];
      std::vector<std::string> t3_formats_str_2 = opFormats[opFormats.size() - 1];

      std::vector<std::vector<std::string>> t3_formats_str = {t3_formats_str_0, t3_formats_str_1, t3_formats_str_2};
      //ArrayAttr t3_formats = convert2DVectorToArrayAttrStr(t3_formats_str, builder);
      std::vector<mlir::Value> t3_rhs = {v, tensors[1]};

      comet_errs() << " tensors.size(): " << tensors.size() << "\n";
      std::vector<mlir::Value> t3_lhs = tensors_lhs;

      auto t3_optype = builder.getStringAttr(optype);
      auto t3_semiring = builder.getStringAttr(semiringName);

      // for t3_rhs
      std::vector<std::vector<int>> t3_rhsop_perms_str = {t3_perms_int_0, t3_perms_int_1};
      ArrayAttr t3_rhsop_perms = convert2DVectorToArrayAttrInt(t3_rhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t3_rhsop_formats_str = {t3_formats_str_0, t3_formats_str_1};
      ArrayAttr t3_rhsop_formats = convert2DVectorToArrayAttrStr(t3_rhsop_formats_str, builder);
      mlir::Value t3_rhsop = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t3_rhs, t3_rhsop_perms, t3_rhsop_formats);

      // for t3_lhs
      std::vector<std::vector<int>> t3_lhsop_perms_str = {t3_perms_int_2};
      ArrayAttr t3_lhsop_perms = convert2DVectorToArrayAttrInt(t3_lhsop_perms_str, builder);
      std::vector<std::vector<std::string>> t3_lhsop_formats_str = {t3_formats_str_2};
      ArrayAttr t3_lhsop_formats = convert2DVectorToArrayAttrStr(t3_lhsop_formats_str, builder);
      mlir::Value t3_lhsop = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()), t3_lhs, t3_lhsop_perms, t3_lhsop_formats);

      // for t3
      mlir::Value t3 = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, t3_rhsop, t3_lhsop, t3_optype, opt_type, t3_semiring);

      /// old version for new children ops
      std::vector<mlir::Value> newComputeOps = {t1, t2, t3};
      // Operation* parentOp = *(itComputeOp.getOperation()->getUsers().begin());
      // parentOp->setOperands(newComputeOps);
      // comet_pdump(parentOp);
      // itComputeOp.erase();

      replaceOperands(itComputeOp.getOperation(), newComputeOps);

      /// Step 2: split j into 3.
      Operation *needSplitNode = *(newComputeOps[0].getDefiningOp()->getResult(0).getUsers().begin());
      Operation *parentSplitNode = *(needSplitNode->getResult(0).getUsers().begin());
      comet_errs() << " call splitIndicesOp for applying workspace in Input \n";
      comet_pdump(needSplitNode);
      splitIndicesOp(needSplitNode, parentSplitNode->getResult(0), builder, loc);
      comet_errs() << "\n";

    } // end if(sparseDimsInput.size() == 1)
  }
}

void WorkspaceTransformsPass::WorkspaceTransforms(mlir::FuncOp funcop)
{

  comet_vdump(funcop);

  funcop.walk([](indexTree::IndexTreeOp op)
              {
    OpBuilder builder(op);
    comet_vdump(op);

    Location loc = op.getLoc();

    // 1. Find its child, until reach the ta.itCompute op
    // Get first user
    Value computeOp = op.getOperation()->getOperand(0);
    comet_vdump(computeOp);
    // auto v = computeOp.getDefiningOp(); //->getOperand(0);
    // comet_vdump(v);
    
    // mlir::Value computeOp = op.getOperation()->getOperand(0).getUsers().begin(); 
    // Only one child??
    // Build a map, which index is in which IndexTreeIndicesOp
    // ------ Notice: each index is only in one IndicesOp in original index tree here
    // ------ TODO(ruiqin): handle more complicate cases: one index is in more than one IndicesOp
    // For an indexTree, the indices ids are
    std::map<int, mlir::Value> indexValueMap;
    
    while(!(isa<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp()))){
      if(isa<indexTree::IndexTreeIndicesOp>(computeOp.getDefiningOp())){
        auto indicesop = dyn_cast<indexTree::IndexTreeIndicesOp>(computeOp.getDefiningOp());
        ArrayAttr idsArrayAttr = indicesop.indices();
        for(auto n : idsArrayAttr){
          int ids = n.cast<mlir::IntegerAttr>().getInt();
          indexValueMap.emplace(ids, computeOp);
        }      
      }
      computeOp = computeOp.getDefiningOp()->getOperand(0); // put here
    }
    comet_vdump(computeOp);

    // if (!isa<SparseTensorDeclOp>(computeOp.getDefiningOp()->getOperand(0).getDefiningOp()) || !isa<SparseTensorDeclOp>(computeOp.getDefiningOp()->getOperand(1).getDefiningOp())){
    //     comet_errs() <<  __FILE__ << __LINE__ << " Dense/Mixed mode computation. No need to apply workspace transformation\n";
    //     return;
    // }

    // 2. Check if there is sparse dim in the ta.itCompute op, 
    std::vector<std::vector<std::string>> opFormats;
    std::vector<std::vector<int>> opPerms;
    std::vector<std::vector<bool> > inputOutputMapping;
    getFormatsPermsOfComputeOp(computeOp, opFormats, opPerms, inputOutputMapping);
    indexTree::IndexTreeComputeOp itComputeOp = dyn_cast<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp());
    // int optype = itComputeOp.op_type();
    std::string optype(itComputeOp.op_type().data());
    

    // Check the input tensors, and the output tensor, to see if it contains sparse dimensions
    // get the dim ids
    std::vector<int> sparseDimsOutput = getSparseDimsOutput(opFormats, opPerms);
    // std::vector<int> 
    std::vector<struct dimInTensor> sparseDimsInput = getSparseDimsInput(opFormats, opPerms);

    if(sparseDimsOutput.size() == 0 && sparseDimsInput.size() == 0){
      // No need to apply workspace transformation
      comet_errs() <<  __FILE__ << __LINE__ << "No need to apply workspace transformation\n";
      comet_errs() <<  __FILE__ << __LINE__ << "Size of sparseDimsOutput:" << sparseDimsOutput.size() << "\n";
      comet_errs() <<  __FILE__ << __LINE__ << "Size of sparseDimsInput:" << sparseDimsInput.size() << "\n";
      return;
    }
    
    assert(sparseDimsOutput.size() == 1 && " More than one sparse index in the output, we are expecting to support it in the future\n"); 

    std::vector<Value> newComputeOps;
    // create three IndexTreeComputeOp op
    // sparse dim in output tensor   
    if(sparseDimsOutput.size() == 1){
      // newComputeOps = workspaceOutput(sparseDimsOutput, itComputeOp, opFormats, opPerms, optype, indexValueMap, builder, loc);
      newComputeOps = workspaceOutput(sparseDimsOutput, itComputeOp, opFormats, opPerms, optype, indexValueMap, builder, op);
    }
    // initially here workspaceOutput content 

    #ifdef DEBUG_MODE_WorkspaceTransformsPass
    // Should notice, the itree has been the new itree already after call workspaceOutput
    for(auto n : newComputeOps){
      comet_vdump(n);
    }
    #endif
    // assert(sparseDimsInput.size() >= 1 && " More than one sparse index in the output, we are expecting to support it in the future\n"); 
    if(sparseDimsInput.size() == 1){
      comet_vdump(op);
      // Need the newComputeOps
      workspaceInput(newComputeOps, builder, loc);
    } }); // end function traverse
}

void WorkspaceTransformsPass::runOnFunction()
{
  LLVM_DEBUG(llvm::dbgs() << "start WorkspaceTransformsPass\n");
  comet_errs() << __FILE__ << " " << __LINE__ << " start WorkspaceTransforms pass \n";
  auto function = getFunction();

  // Traverse the function, only handle ta.itree operation
  WorkspaceTransforms(function);
}

void CompressedWorkspaceTransformsPass::CompressedWorkspaceTransforms(mlir::FuncOp funcop)
{
  // comet_vdump(funcop);

  funcop.walk([](indexTree::IndexTreeOp op)
              {
    OpBuilder builder(op);
    comet_vdump(op);

    Location loc = op.getLoc();

    // 1. Find its child, until reach the ta.itCompute op
    // Get first user
    Value computeOp = op.getOperation()->getOperand(0);
    comet_vdump(computeOp);
    // auto v = computeOp.getDefiningOp(); //->getOperand(0);
    // comet_vdump(v);
    
    // Only one child??
    // Build a map, which index is in which IndexTreeIndicesOp
    // ------ Notice: each index is only in one IndicesOp in original index tree here
    // ------ TODO(ruiqin): handle more complicate cases: one index is in more than one IndicesOp
    // For an indexTree, the indices ids are
    std::map<int, mlir::Value> indexValueMap;
    
    while(!(isa<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp()))){
      if(isa<indexTree::IndexTreeIndicesOp>(computeOp.getDefiningOp())){
        auto indicesop = dyn_cast<indexTree::IndexTreeIndicesOp>(computeOp.getDefiningOp());
        ArrayAttr idsArrayAttr = indicesop.indices();
        for(auto n : idsArrayAttr){
          int ids = n.cast<mlir::IntegerAttr>().getInt();
          indexValueMap.emplace(ids, computeOp);
        }      
      }
      computeOp = computeOp.getDefiningOp()->getOperand(0); // put here
    }
    comet_vdump(computeOp);
    
    // 2. Check if there is sparse dim in the ta.itCompute op, 
    std::vector<std::vector<std::string>> opFormats;
    std::vector<std::vector<int>> opPerms;
    std::vector<std::vector<bool> > inputOutputMapping;
    getFormatsPermsOfComputeOp(computeOp, opFormats, opPerms, inputOutputMapping);

    #ifdef DEBUG_MODE_WorkspaceTransformsPass
    comet_errs() << "Print opFormats:\n";
    for (auto n : opFormats)
    {
      
      print_vector<std::string>(n);
    }
    #endif
    
    indexTree::IndexTreeComputeOp itComputeOp = dyn_cast<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp());
    std::string optype(itComputeOp.op_type().data());
    
    //Check the input tensors, and the output tensor, to see if it contains sparse dimensions
    //get the dim ids
    std::vector<int> sparseDimsOutput = getSparseDimsOutput(opFormats, opPerms);
    
    #ifdef DEBUG_MODE_WorkspaceTransformsPass
    comet_errs() << " Print sparseDimsOutput: ";
    for(auto p : sparseDimsOutput){
      comet_errs() << p << " ";
    }
    comet_errs() << "\n";
    #endif

    std::vector<struct dimInTensor> sparseDimsInput = getSparseDimsInput(opFormats, opPerms);

    if(sparseDimsOutput.size() == 0 && sparseDimsInput.size() == 0){
      // No need to apply workspace transformation
      comet_errs() <<  __FILE__ << __LINE__ << " No need to apply workspace transformation\n";
      return;
    }
    
    assert(sparseDimsOutput.size() == 1 && " More than one sparse index in the output, we are expecting to support it in the future\n"); 

    std::vector<Value> newComputeOps;
    // create three IndexTreeComputeOp op
    // sparse dim in output tensor   
    if(sparseDimsOutput.size() == 1){
      
      newComputeOps = CompressedWorkspaceOutput(sparseDimsOutput, itComputeOp, opFormats, opPerms, optype, indexValueMap, builder, op);
    }
    // initially here workspaceOutput content 

    #ifdef DEBUG_MODE_WorkspaceTransformsPass
    // Should notice, the itree has been the new itree already after call workspaceOutput
    for(auto n : newComputeOps){
      
      comet_vdump(n);
    }
    #endif
    // assert(sparseDimsInput.size() >= 1 && " More than one sparse index in the output, we are expecting to support it in the future\n"); 
    if(sparseDimsInput.size() == 1){
      comet_vdump(op);
      // Need the newComputeOps
      CompressedWorkspaceInput(newComputeOps, builder, loc);
      // workspaceInput(newComputeOps, builder, loc);
    } }); // end function traverse
}

void CompressedWorkspaceTransformsPass::runOnFunction()
{
  LLVM_DEBUG(llvm::dbgs() << "start CompressedWorkspaceTransformsPass\n");
  comet_errs() << __FILE__ << " " << __LINE__ << " start CompressedWorkspaceTransforms pass \n";
  auto function = getFunction();
  
  // Traverse the function, only handle ta.itree operation

  CompressedWorkspaceTransforms(function);
}

// Apply the initial workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::IndexTree::createWorkspaceTransformsPass()
{
  return std::make_unique<WorkspaceTransformsPass>();
}

// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::IndexTree::createCompressedWorkspaceTransformsPass()
{
  return std::make_unique<CompressedWorkspaceTransformsPass>();
}