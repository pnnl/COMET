//===- LowerIndexTreeIRToSCF.cpp  ------===//
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
// This file implements a lowering of index tree dialect to SCF dialect
//===----------------------------------------------------------------------===//

#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
// #include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "llvm/Support/Debug.h"
#include "llvm/ADT/StringSet.h"
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
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

using llvm::SmallVector;
using llvm::StringRef;

#define DEBUG_TYPE "lowering-it-to-scf"

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_LowerIndexTreeToSCFPass
// #define DEBUG_MODE_LowerIndexTreeToSCFPass
// #endif

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
#define comet_debug() llvm::errs() << __FILE__ << ":" << __LINE__ << " "
#define comet_pdump(n)                                \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " "; \
  n->dump()
#define comet_vdump(n)                                \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " "; \
  n.dump()
#else
#define comet_debug() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

namespace comet {
#define GEN_PASS_DEF_CONVERTINDEXTREETOSCF
#include "comet/Conversion/Passes.h.inc"
} // namespace comet

#define TENSOR_NUMS 3
#define INPUT_TENSOR_NUMS 2

// Valid semiring operators.
static const llvm::StringSet<> Semiring_ops{
    "atan2", "div", "eq", "first", "ge", "gt", "hypot",
    "land", "le", "lor", "lt", "max", "minxy", "minus",
    "ne", "pairxy", "plusxy", "pow", "rdiv", "second", "times"};

// List of valid semiring operands for reduce op
static const llvm::StringSet<> Semiring_reduceOps{"any", "land", "lor", "max",
                                                  "minxy", "plusxy", "times", "noop"}; // noop is for monoid op support

class OpsTree
{
  // private:
public:
  std::vector<scf::ForOp> forOps; // The (nested) for loops
  std::vector<Value> accessIdx;   // The coordinate of accessing that dimension
  std::vector<Value> cmptOps;     // The computation ops
  std::vector<OpsTree *> children;
  OpsTree *parent;
  int id;

public:
  OpsTree() {}

  OpsTree(std::vector<scf::ForOp> &forOps, std::vector<Value> &accessIdx, std::vector<Value> &cmptOps, OpsTree *parent, int id)
  {
    this->forOps = forOps;
    this->accessIdx = accessIdx;
    this->cmptOps = cmptOps;
    this->parent = parent;
    this->id = id;
  }

  OpsTree(std::vector<scf::ForOp> &forOps, std::vector<Value> &accessIdx, std::vector<Value> &cmptOps, OpsTree *parent)
  {
    this->forOps = forOps;
    this->accessIdx = accessIdx;
    this->cmptOps = cmptOps;
    this->parent = parent;
  }

  ~OpsTree() {}

  void addChild(OpsTree *tree)
  { // const T& node
    this->children.push_back(tree);
  }

  std::vector<scf::ForOp> &getForOps()
  {
    return this->forOps;
  }

  OpsTree *getParent()
  {
    return this->parent;
  }

  void setForOps(std::vector<scf::ForOp> &forOps)
  {
    this->forOps = forOps;
  }

  std::vector<OpsTree *> &getChildren()
  {
    return this->children;
  }
};

unsigned int findIndexInVector_OpsTree(std::vector<OpsTree *> vec, OpsTree *e)
{
  // Check if element e exists in vector
  auto it = std::find(vec.begin(), vec.end(), e);

  // It accepts a range and an element to search in the given range. If element is found then it returns an iterator to the first element in the given range that’s equal to given element, else it returns an end of the list.
  unsigned int ret = 0;
  if (it != vec.end())
  {
    // Get index of element from iterator
    ret = std::distance(vec.begin(), it);
  }
  else
  {
    ret = vec.size();
  }
  return ret;
}

struct indexInTensor
{
  Value tensor;
  unsigned int id;
  std::string format;
};

Value findCorrespondingAlloc(Value iOp)
{
  comet_debug() << "findCorrespondingAlloc for loop upper bound\n";
  comet_vdump(iOp);
  auto init_alloc = iOp.getDefiningOp()->getOperand(0);
  comet_vdump(init_alloc);

  while (true)
  {
    if (isa<memref::AllocOp>(init_alloc.getDefiningOp()))
    {
      if (init_alloc.getType().dyn_cast<MemRefType>().getDimSize(0) != ShapedType::kDynamic)
      {
        return init_alloc;
      }
    }
    if (init_alloc.getDefiningOp()->getNumOperands() > 0)
    {
      init_alloc = init_alloc.getDefiningOp()->getOperand(0);
    }
    else
    {
      // Alloc related to another sparse tensor construct such as coming from sparse transpose
      comet_debug() << "Return alloc op - comes from sptensor_construct\n";
      comet_vdump(init_alloc);
      return init_alloc;
    }
  }
}

/// Get allocs for a tensor (sparse or dense)
std::vector<Value> getAllocs(Value tensor)
{
  std::vector<Value> allocs;
  if (tensor.getType().isa<mlir::TensorType>())
  { // Dense tensor
    comet_debug() << " getAllocs() -  it is dense\n";
    if (isa<ToTensorOp>(tensor.getDefiningOp()))
    {
      Operation *tensorload = cast<ToTensorOp>(tensor.getDefiningOp());
      auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
      comet_vdump(alloc_op);
      allocs.push_back(alloc_op);
    }
    else
    {
      for (unsigned int i = 0; i < tensor.getDefiningOp()->getNumOperands(); i++)
      {
        if (isa<ToTensorOp>(tensor.getDefiningOp()->getOperand(i).getDefiningOp()))
        {
          Operation *tensorload = cast<ToTensorOp>(tensor.getDefiningOp()->getOperand(i).getDefiningOp());
          auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
          comet_vdump(alloc_op);
          allocs.push_back(alloc_op);
        }
      }
    }
  }
  else if (tensor.getType().isa<tensorAlgebra::SparseTensorType>())
  { // nSparse tensor
    comet_debug() << " getAllocs() -  it is sparse\n";
    auto defop = tensor.getDefiningOp<tensorAlgebra::SparseTensorConstructOp>();

    // TODO(gkestor): get tensor ranks by functions
    unsigned int ranks = defop.getTensorRank();
    for (unsigned int n = 0; n < defop.getTotalDimArrayCount(); n++)
    {
      comet_vdump(defop.getIndices()[n]);
      Operation *tensorload = defop.getIndices()[n].getDefiningOp<ToTensorOp>();
      auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
      allocs.push_back(alloc_op);
      comet_vdump(alloc_op);
    }

    comet_debug() << "------\n";
  }
  else if (dyn_cast<ConstantOp>(tensor.getDefiningOp()))
  { // ConstantOp
    allocs.push_back(tensor);
  }
  return allocs;
}

std::vector<std::vector<Value>> getAllAllocs(std::vector<Value> tensors)
{
  std::vector<std::vector<Value>> allAllocs(tensors.size());
  for (unsigned int i = 0; i < tensors.size(); i++)
  {
    allAllocs[i] = getAllocs(tensors[i]);
  }
  return allAllocs;
}

/// while until parent == null
void getAncestorsOps(OpsTree *opstree, std::vector<OpsTree *> &ret)
{

  while (opstree->parent != NULL)
  {
    ret.push_back(opstree->parent);
    opstree = opstree->parent;
  }
}

/// Generate scf.for op for indices
/// The index is the "idx"th index of "tensor"
void genForOps(std::vector<Value> tensors,
               std::vector<unsigned int> ids,
               std::vector<std::string> formats,
               indexTree::IndexTreeOp rootOp,
               PatternRewriter &rewriter,
               OpsTree *opstree, std::vector<Value> ancestors_wp)
{
  comet_debug() << " genForOps indexTreeOp\n";
  comet_vdump(rootOp);
  Location loc = rootOp.getLoc();
  /// The insertion location should be "the end of the body of parent loop"
  std::vector<OpsTree *> ancestorsOps;
  getAncestorsOps(opstree, ancestorsOps);
  comet_debug() << " genForOps ancestorsOps.size(): " << ancestorsOps.size() << "\n";
  for (unsigned int i = 0; i < ancestorsOps.size(); i++)
  {
    comet_debug() << " ancestorsOps[" << i << "]->forOps.size(): " << ancestorsOps[i]->forOps.size() << ", ancestorsOps->id: "
                  << ancestorsOps[i]->id << "\n";
  }

  comet_debug() << "\n";
  /// If parent is for loop, insert into the body, How to get end of body?
  if (ancestorsOps.size() > 0)
  {
    // ancestorsOps[0] stores the closest parent
    scf::ForOp parent_forop = NULL;
    comet_debug() << "\n";
    std::vector<scf::ForOp> parent_forops = ancestorsOps[0]->forOps;
    comet_debug() << " parent_forops.size(): " << parent_forops.size() << " \n";

    parent_forop = parent_forops[parent_forops.size() - 1];

    comet_debug() << " reset the insertion point\n";
    comet_vdump(parent_forop);

    unsigned int order = findIndexInVector_OpsTree(ancestorsOps[0]->getChildren(), opstree);
    comet_debug() << " order: " << order << "\n";
    if (order == ancestorsOps[0]->getChildren().size())
    {
      llvm::errs() << __LINE__ << "Not belong to parent's children\n";
    }
    else
    {
      // Get the children of the parent_forop
      comet_debug() << " number of children: " << parent_forops.size() << "\n";
      if (order == 0)
      {
        // rewriter.setInsertionPointToStart(parent_forop.getBody());
        comet_debug() << "Insertion point order == 0\n";
        rewriter.setInsertionPoint(parent_forop.getBody()->getTerminator());
      }
      else
      {
        comet_debug() << "\n";
        std::vector<scf::ForOp> brother_forops = ancestorsOps[0]->getChildren()[order - 1]->forOps;
        if (brother_forops.size() > 0)
        {
          comet_debug() << " brother_forops.size(): " << brother_forops.size() << "\n";
          if (opstree->forOps.size() == 0)
          {
            comet_debug() << "\n";
            comet_vdump(brother_forops[0]);
            comet_debug() << "Insertion point (brother_forops.size() > 0 &&  opstree->forOps.size() == 0)\n";
            rewriter.setInsertionPointAfter(brother_forops[0]);
          }
          else
          { // current opstree contains loops, insert in the body of the loops
            comet_debug() << " -------- current opstree contain loops --- impossible\n";
            comet_debug() << "Insertion point (brother_forops.size() > 0 &&  opstree->forOps.size() != 0)\n";
            rewriter.setInsertionPoint(opstree->forOps[opstree->forOps.size() - 1].getBody()->getTerminator());
          }
        }
      }
    }
    comet_debug() << " reset the insertion point\n";
  }
  comet_debug() << "\n";

  Value const_index_1 = rewriter.create<ConstantIndexOp>(loc, 1);

  comet_debug() << "Tensor size: " << tensors.size() << "\n";
  std::vector<std::vector<Value>> allAllocs = getAllAllocs(tensors);

  comet_debug() << "Tensors:\n";
  for (unsigned int i = 0; i < tensors.size(); i++)
  {
    comet_vdump(tensors[i]);
  }
  // Dense, get dimension size --> loop upperbound
  // If the index is in rhs1, get it from rhs1; Otherwise, get it from rhs2

  Value upperBound, lowerBound;
  for (unsigned int i = 0; i < tensors.size(); i++)
  {
    if (i > 0)
    {
      // insertion point: the body of the previous i's loop body
      comet_debug() << " -------- current opstree contain loops\n";
      rewriter.setInsertionPoint(opstree->forOps[opstree->forOps.size() - 1].getBody()->getTerminator());
    }

    Value tensor = tensors[i];
    std::string format = formats[i];
    unsigned int id = ids[i];

    comet_debug() << " current index format: " << format << "\n";
    if (format.compare(0, 1, "D") == 0)
    {
      // Check which tensor is sparse, which is dense;
      // Since this function only handles mixed sparse/dense, then "D" only occurs in one tensor
      // Both the dense and sparse tensor contain the dim size; But they are different. Use one.
      int64_t maxSize = 0;
      comet_debug() << " ";
      comet_vdump(tensor);
      if (tensor.getType().isa<mlir::RankedTensorType>())
      { // Dense tensor

        auto tensorTy = tensor.getType().cast<mlir::TensorType>();
        maxSize = tensorTy.getDimSize(id);

        // Check if dynamic size
        // Check upperBoundsize
        if (maxSize == ShapedType::kDynamic)
        {
          // Find defOp allocOp, check the parameter
          comet_debug() << " Dynamic size ";
          comet_pdump(tensor.getDefiningOp());                // tensor_load
          comet_vdump(tensor.getDefiningOp()->getOperand(0)); // alloc <?x32xf64>
          // Check the order of the current dynamic size
          auto rhs1_alloc = tensor.getDefiningOp()->getOperand(0);
          std::vector<unsigned int> dyn_dims_vec;
          for (unsigned i = 0; i < tensorTy.getRank(); i++)
          {
            if (tensorTy.isDynamicDim(i))
            {
              dyn_dims_vec.push_back(i);
            }
          } // ? x ? x 20 x ?
          auto rhs1_loc_dyn = findIndexInVector<unsigned int>(dyn_dims_vec, id);
          comet_vdump(rhs1_alloc.getDefiningOp()->getOperand(rhs1_loc_dyn));

          upperBound = rhs1_alloc.getDefiningOp()->getOperand(rhs1_loc_dyn);
        }
        else
        {
          upperBound = rewriter.create<ConstantIndexOp>(loc, maxSize);
        }

        lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        comet_debug() << " D Loop\n";
        comet_vdump(loop);

        opstree->forOps.push_back(loop);
        opstree->accessIdx.push_back(loop.getInductionVar());
      }
      else if (tensor.getType().isa<mlir::UnrankedTensorType>())
      {
        comet_debug() << " \n";
        comet_pdump(tensor.getDefiningOp());
        if (indexTree::IndexTreeComputeRHSOp rhsop = dyn_cast<indexTree::IndexTreeComputeRHSOp>(tensor.getDefiningOp()))
        {
          comet_debug() << " \n";
        }
      }
      else if (tensor.getType().cast<tensorAlgebra::SparseTensorType>())
      {
        comet_debug() << "cur_idx is in tensor " << i << "\n";

        lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto index_0 = rewriter.create<ConstantIndexOp>(loc, 0);
        std::vector<Value> upper_indices = {index_0};
        upperBound = rewriter.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices);

        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        comet_debug() << " D Loop\n";
        comet_vdump(loop);

        opstree->forOps.push_back(loop);
        opstree->accessIdx.push_back(loop.getInductionVar());
      }
      // }
    }
    // mix sparse dense tensor contraction, only one sparse tensor
    else if (format.compare(0, 2, "CU") == 0)
    {
      // Generate for(int m = pos[0]; m < pos[1]; m++){int i = crd[m];}
      // if i = 0, index is [0,1]
      // if parent loop and child loop is accessing the same sparse tensor (CSF), index is [m, m+1], m is the nearest loop induction variable
      // Otherwise, the m comes from load operation of the input sparse tensor such as
      // j = crd[i];
      // for (int m = pos[j]; m < pos[j+1]; m++)

      comet_debug() << " format is CU id: " << id << "\n";
      comet_debug() << " Tensor: \n";
      comet_vdump(tensor);
      Value index_lower;
      Value index_upper;
      if (tensor.getType().cast<tensorAlgebra::SparseTensorType>())
      {
        comet_debug() << " Tensor type is sparse\n";
        // cur_idx is in ith input tensor, which is sparse
        if (id == 0)
        { // The first index in the tensor
          index_lower = rewriter.create<ConstantIndexOp>(loc, 0);
          comet_debug() << " index_lower is set here, id == 0 " << opstree->forOps.size() << "\n";
          comet_vdump(index_lower);
        }
        else
        {
          if (opstree->parent != NULL)
          {
            comet_debug() << " opstree->parent is not NULL\n";
            // Parent loop is
            scf::ForOp parent_forop = opstree->parent->forOps[opstree->parent->forOps.size() - 1];
            comet_debug() << " parent forop\n";
            comet_vdump(parent_forop);
            auto parent_UpperBound = parent_forop.getUpperBound();
            comet_debug() << " parent upperBound:\n";
            comet_vdump(parent_UpperBound);

            //  check if parent's and child's upper bounds come from the same sparse tensor
            auto alloc_parent_bounds = findCorrespondingAlloc(parent_UpperBound);
            comet_debug() << " parent upperBound alloc\n";
            comet_vdump(alloc_parent_bounds);

            comet_debug() << " child upperBound:\n";
            comet_vdump(allAllocs[i][4 * id]);
            auto alloc_child_bounds = findCorrespondingAlloc(allAllocs[i][4 * id]);
            comet_debug() << " child upperBound alloc\n";
            comet_vdump(alloc_child_bounds);

            if (alloc_child_bounds == alloc_parent_bounds) // m is the nearest loop induction variable
            {
              comet_debug() << " THESAME: Parent and Child has the same alloc\n";
              index_lower = parent_forop.getInductionVar();
            }
            else
            { // m comes from the load
              comet_debug() << " DIFFERENT:Parent and Child has the different alloc\n";
              comet_vdump(alloc_parent_bounds);
              comet_vdump(alloc_child_bounds);
              index_lower = opstree->parent->accessIdx[opstree->parent->forOps.size() - 1];
            }
          }
          else
            assert(false && "Unexpected condition\n");
        }

        comet_debug() << " index_lower:";
        comet_vdump(index_lower);
        comet_vdump(const_index_1);
        index_upper = rewriter.create<AddIOp>(loc, index_lower, const_index_1);
        comet_debug() << " AddIOps (index_upper):";
        comet_vdump(index_upper);

        std::vector<Value> lower_indices = {index_lower};
        lowerBound = rewriter.create<memref::LoadOp>(loc, allAllocs[i][4 * id], lower_indices);

        std::vector<Value> upper_indices = {index_upper};
        upperBound = rewriter.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        comet_debug() << " CU Loop\n";
        comet_vdump(loop);

        rewriter.setInsertionPoint(loop.getBody()->getTerminator());

        std::vector<Value> crd_indices = {loop.getInductionVar()};
        auto get_index = rewriter.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);

        comet_debug() << "CU loop generated\n";
        comet_vdump(loop);
        opstree->forOps.push_back(loop);
        opstree->accessIdx.push_back(get_index);
      }
    }
    else if (format.compare(0, 2, "CN") == 0)
    {
      // Generate for(int m = pos[0]; m < pos[1]; m++){int i = crd[m];}
      if (tensor.getType().cast<tensorAlgebra::SparseTensorType>())
      {
        auto index_0 = rewriter.create<ConstantIndexOp>(loc, 0);
        std::vector<Value> lower_indices = {index_0};
        lowerBound = rewriter.create<memref::LoadOp>(loc, allAllocs[i][4 * id], lower_indices);

        auto index_1 = rewriter.create<ConstantIndexOp>(loc, 1);
        std::vector<Value> upper_indices = {index_1};
        upperBound = rewriter.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        comet_debug() << " CN Loop\n";
        comet_vdump(loop);

        rewriter.setInsertionPoint(loop.getBody()->getTerminator());

        std::vector<Value> crd_indices = {loop.getInductionVar()};
        auto get_index = rewriter.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);

        opstree->forOps.push_back(loop);
        opstree->accessIdx.push_back(get_index);
      }
      // DVSpM_CSR .. here .. why there is tensorIsSparse[i] == true?
    }
    else if (format.compare(0, 1, "S") == 0)
    {
      // Currently supported formats, Singleton is not the format of first dimension
      // and it doesn't produce a loop
      // Generate: int j = A2crd[m];

      if (tensor.getType().cast<tensorAlgebra::SparseTensorType>())
      {
        comet_debug() << "cur_idx is in tensor " << i << "\n";
        // Accesing the last level loop info
        scf::ForOp last_forop;
        if (opstree->forOps.size() > 0)
        { // current node contain at least 1 level loop
          last_forop = opstree->forOps[opstree->forOps.size() - 1];
        }
        else
        {
          if (opstree->parent != NULL)
            last_forop = opstree->parent->forOps[opstree->parent->forOps.size() - 1];
        }

        std::vector<Value> crd_indices = {last_forop.getInductionVar()};
        auto get_index = rewriter.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);

        /// Adding one iteration loop to provide consistency with the corresponding index tree.
        /// Index tree includes an index node for the dimension but "S" format for this dimension
        /// doesn't produce a loop.
        lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        upperBound = rewriter.create<ConstantIndexOp>(loc, 1);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
        comet_debug() << " S Loop\n";
        comet_vdump(loop);
        opstree->forOps.push_back(loop);

        opstree->accessIdx.push_back(get_index);
      }
      else
      {
        llvm::errs() << "Not supported tensor type\n";
      }
    }
    else
    {
      llvm::errs() << "Not supported format: " << format << "\n";
    }

    comet_debug() << " UpperBound: (i) " << i << " ";
    comet_vdump(upperBound);

    comet_debug() << "finish generate loops for current index format: " << format << "\n";
  }
}

Value getSemiringSecondVal(PatternRewriter &rewriter, Location loc,
                           llvm::StringRef &semiringSecond, Value &Input0, Value &Input1,
                           bool compressedWorkspace)
{

  Value elementWiseResult;
  if (semiringSecond == "times")
  {
    elementWiseResult = rewriter.create<MulFOp>(loc, Input0, Input1);
  }
  else if (semiringSecond == "first")
  {
    elementWiseResult = Input0;
  }
  else if (semiringSecond == "second")
  {
    elementWiseResult = Input1;
  }
  else if (semiringSecond == "atan2")
  {
    elementWiseResult = rewriter.create<math::Atan2Op>(loc, Input0, Input1);
  }
  else if (semiringSecond == "div")
  {
    elementWiseResult = rewriter.create<DivFOp>(loc, Input0, Input1);
  }
  else if (semiringSecond == "eq")
  {
    elementWiseResult = rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, Input0, Input1);
  }
  else if (semiringSecond == "ge")
  {
    elementWiseResult = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGE, Input0, Input1);
  }
  else if (semiringSecond == "gt")
  {
    elementWiseResult = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
  }
  else if (semiringSecond == "le")
  {
    elementWiseResult = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLE, Input0, Input1);
  }
  else if (semiringSecond == "lt")
  {
    elementWiseResult = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
  }
  else if (semiringSecond == "land")
  {
    // land requires integer type input
    llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                 << "land"
                 << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    exit(1);
  }
  else if (semiringSecond == "lor")
  {
    // lor requires integer type input
    llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                 << "lor"
                 << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    exit(1);
  }
  else if (semiringSecond == "lxor")
  {
    // lxor requires integer type input
    llvm::errs() << "Not supported semiring operator: "
                 << "lxor"
                 << "\n";
  }
  else if (semiringSecond == "minxy")
  {
    Value cmp = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
    elementWiseResult = rewriter.create<SelectOp>(loc, cmp, Input0, Input1);
  }
  else if (semiringSecond == "max")
  {
    Value cmp = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
    elementWiseResult = rewriter.create<SelectOp>(loc, cmp, Input0, Input1);
  }
  else if (semiringSecond == "ne")
  {
    elementWiseResult = rewriter.create<CmpFOp>(loc, CmpFPredicate::ONE, Input0, Input1);
  }
  else if (semiringSecond == "minus")
  {
    elementWiseResult = rewriter.create<SubFOp>(loc, Input0, Input1);
  }
  else if (semiringSecond == "plusxy")
  {
    elementWiseResult = rewriter.create<AddFOp>(loc, Input0, Input1);
  }
  else if (semiringSecond == "pairxy")
  {
    elementWiseResult = rewriter.create<ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));
  }
  else if (semiringSecond == "pow")
  {
    elementWiseResult = rewriter.create<math::PowFOp>(loc, Input0, Input1);
  }
  else
  {
    llvm::errs() << "Not supported semiring operator: " << semiringSecond << "\n";
    assert(false);
    // we should not proceed forward from this point to avoid faulty behavior.
  }

  return elementWiseResult;
}

Value getSemiringFirstVal(PatternRewriter &rewriter, Location loc,
                          llvm::StringRef &semiringFirst, Value &Input0, Value &Input1,
                          bool compressedWorkspace)
{

  Value reduceResult;
  if (semiringFirst == "times")
  {
    reduceResult = rewriter.create<MulFOp>(loc, Input0, Input1);
  }
  else if (semiringFirst == "plusxy")
  {
    reduceResult = rewriter.create<AddFOp>(loc, Input0, Input1);
  }
  else if (semiringFirst == "minxy")
  {
    if (!compressedWorkspace)
    {
      llvm::errs() << "Not supported semiring operator "
                      "(please use compressed workspace optimization or opt-comp-workspace "
                      "where this operation is known to work): "
                   << "min"
                   << "\n";
      // we should not proceed forward from this point to avoid in-correct results from generated code.
      assert(false);
    }
    Value cmp = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
    reduceResult = rewriter.create<SelectOp>(loc, cmp, Input0, Input1);
  }
  else if (semiringFirst == "max")
  {
    if (!compressedWorkspace)
    {
      llvm::errs() << "Not supported semiring operator "
                      "(please use compressed workspace optimization or opt-comp-workspace "
                      "where this operation is known to work): "
                   << "max"
                   << "\n";
      // we should not proceed forward from this point to avoid in-correct results from generated code.
      assert(false);
    }
    Value cmp = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
    reduceResult = rewriter.create<SelectOp>(loc, cmp, Input0, Input1);
  }
  else if (semiringFirst == "land")
  {
    // land requires integer type input
    llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                 << "land"
                 << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    assert(false);
  }
  else if (semiringFirst == "lor")
  {
    // lor requires integer type input
    llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                 << "lor"
                 << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    assert(false);
  }
  else if (semiringFirst == "any")
  {
    reduceResult = Input1;
  }
  else if (semiringFirst == "noop")
  {
    reduceResult = Input1;
  }
  else
  {
    llvm::errs() << "Not supported semiring operator: " << semiringFirst << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    assert(false);
  }

  return reduceResult;
}

void formSemiringLoopBody(bool comp_worksp_opt, llvm::StringRef &semiringFirst,
                          llvm::StringRef &semiringSecond,
                          PatternRewriter &rewriter, Location loc, int lhs_loc,
                          std::vector<std::vector<Value>> main_tensors_all_Allocs,
                          std::vector<std::vector<Value>> tensors_lhs_Allocs,
                          std::vector<std::vector<Value>> tensors_rhs_Allocs,
                          std::vector<std::vector<Value>> allValueAccessIdx,
                          std::vector<std::vector<Value>> allAccessIdx,
                          std::vector<scf::ForOp> forLoops,
                          std::vector<std::vector<int>> rhsPerms,
                          std::vector<std::vector<std::string>> rhsFormats,
                          std::vector<std::vector<std::string>> lhsFormats)
{
  bool isMixedMode = checkIsMixedMode(rhsFormats);
  bool isElementwise = checkIsElementwise(rhsPerms);
  comet_debug() << " isElementwise:" << isElementwise << " isMixedMode: " << isMixedMode << "\n";
  auto ctx = rewriter.getContext();
  IndexType indexType = IndexType::get(ctx);

  if ((semiringFirst.size() == 0) | (semiringSecond.size() == 0))
    llvm::errs() << "Error during semiring parsing!"
                 << "\n";

  if (main_tensors_all_Allocs.size() != allValueAccessIdx.size())
    llvm::errs() << "DEBUG ONLY: issue with main_tensor_nums size"
                 << "\n";

  Value const_i1_0 = rewriter.create<ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(0));
  Value const_i1_1 = rewriter.create<ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(1));
  Value const_index_0 = rewriter.create<ConstantIndexOp>(loc, 0);
  auto f64Type = rewriter.getF64Type();
  auto const_f64_0 = rewriter.create<ConstantOp>(loc, f64Type, rewriter.getF64FloatAttr(0));

  int main_tensor_nums = main_tensors_all_Allocs.size();
  bool compressedWorkspace = false;

  if (comp_worksp_opt) // always lhs is dense after workspace transformations
  {
    compressedWorkspace = true;

    // Workspace tensors are on the lhs
    comet_debug() << " lhs_loc: " << lhs_loc << "\n";
    Value checkAlreadySet = rewriter.create<memref::LoadOp>(loc, tensors_lhs_Allocs[1][0], allValueAccessIdx[lhs_loc]);
    comet_debug() << " ";
    comet_vdump(checkAlreadySet);
    comet_debug() << " ";
    comet_vdump(checkAlreadySet.getType());
    comet_debug() << " ";
    comet_vdump(const_i1_0.getType());

    Value notAlreadySet = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, checkAlreadySet, const_i1_0);
    comet_debug() << " ";
    comet_vdump(notAlreadySet);
    auto if_notAlreadySet = rewriter.create<scf::IfOp>(loc, notAlreadySet, /*WithElseRegion*/ true);
    comet_debug() << " If branch:\n";
    comet_vdump(if_notAlreadySet);

    // if-then region corresponding to if_notAlreadySet instruction.
    // if (&if_notAlreadySet. getThenRegion())
    if (!if_notAlreadySet.getThenRegion().empty())
    {
      rewriter.setInsertionPointToStart(&if_notAlreadySet.getThenRegion().front());

      // Wj = Aik * Bkj          // computation wj, outer has k, so +=/= need if/else
      // W_already_set[j] = 1
      // W_index_list[W_index_list_size] = j
      // W_index_list_size++

      std::vector<Value> allLoadsIf(main_tensor_nums);
      for (int m = 0; m < main_tensor_nums; m++)
      {
        Value s = rewriter.create<memref::LoadOp>(loc, main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1], allValueAccessIdx[m]);
        allLoadsIf[m] = s;
        comet_debug() << " ";
        comet_vdump(s);
      }
      comet_debug() << " allLoadsIf.size(): " << allLoadsIf.size() << "\n";

      comet_debug() << "calculate elementWise operation only\n";
      Value elementWiseResult = getSemiringSecondVal(rewriter, loc, semiringSecond, allLoadsIf[0], allLoadsIf[1], compressedWorkspace);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      auto store_sum = rewriter.create<memref::StoreOp>(loc, elementWiseResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], allValueAccessIdx[2]);
      comet_debug() << " ";
      comet_vdump(elementWiseResult);
      comet_vdump(store_sum);
#else
      rewriter.create<memref::StoreOp>(loc, elementWiseResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], allValueAccessIdx[2]);
#endif

      rewriter.create<memref::StoreOp>(loc, const_i1_1, tensors_lhs_Allocs[1][0], allValueAccessIdx[lhs_loc]);

      Value W_index_list_size_old = rewriter.create<memref::LoadOp>(loc, tensors_lhs_Allocs[3][0], ValueRange{const_index_0});

      assert(allValueAccessIdx[lhs_loc].size() == 1 && " more than one access id for auxiliary array\n");
      rewriter.create<memref::StoreOp>(loc, allValueAccessIdx[lhs_loc][0], tensors_lhs_Allocs[2][0], ValueRange{W_index_list_size_old});

      Value const_index_1 = rewriter.create<ConstantIndexOp>(loc, 1);
      Value W_index_list_size_new = rewriter.create<AddIOp>(loc, W_index_list_size_old, const_index_1);
      comet_debug() << " AddIOps (W_index_list_size_new)";
      comet_vdump(W_index_list_size_new);

      rewriter.create<memref::StoreOp>(loc, W_index_list_size_new, tensors_lhs_Allocs[3][0], ValueRange{const_index_0});
    }

    // if-else region corresponding to if_notAlreadySet instruction.
    // if (&if_notAlreadySet.getElseRegion())
    if (!if_notAlreadySet.getElseRegion().empty())
    {
      rewriter.setInsertionPointToStart(&if_notAlreadySet.getElseRegion().front());

      std::vector<Value> allLoadsElse(main_tensor_nums);
      for (auto m = 0; m < main_tensor_nums; m++)
      {
        Value s = rewriter.create<memref::LoadOp>(loc, main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1], allValueAccessIdx[m]);
        allLoadsElse[m] = s;
        comet_debug() << " ";
        comet_vdump(s);
      }
      comet_debug() << " allLoadsElse.size(): " << allLoadsElse.size() << "\n";

      comet_debug() << "calculate elementWise operation and reduction\n";
      Value elementWiseResult = getSemiringSecondVal(rewriter, loc, semiringSecond, allLoadsElse[0], allLoadsElse[1], compressedWorkspace);
      Value reduceResult = getSemiringFirstVal(rewriter, loc, semiringFirst, allLoadsElse[2], elementWiseResult, compressedWorkspace);
      rewriter.create<memref::StoreOp>(loc, reduceResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], allValueAccessIdx[2]);
    }
  }
  else
  { // general dense or mixed mode computation, no need workspace transformations
    std::vector<Value> allLoads(main_tensor_nums);
    for (auto m = 0; m < main_tensor_nums; m++)
    {
      Value load_op = rewriter.create<memref::LoadOp>(loc, main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1], allValueAccessIdx[m]);
      allLoads[m] = load_op;
      comet_debug() << " ";
      comet_vdump(load_op);
    }
    comet_debug() << " allLoads.size(): " << allLoads.size() << "\n";

    // if computeOp is elementwise mixed mode operation, the output is sparse
    if (isMixedMode && isElementwise && !checkIsDense(lhsFormats[0]))
    {

      int dense_inputtensor_id = 0;
      for (unsigned int i = 0; i < rhsFormats.size(); i++)
      {
        if (checkIsDense(rhsFormats[i]))
        {
          dense_inputtensor_id = i;
          break;
        }
      }

      int sparse_inputtensor_id = dense_inputtensor_id ? 0 : 1;
      std::string sparse_format = getTensorFormat(rhsFormats, sparse_inputtensor_id);

      auto last_insertionPoint = rewriter.saveInsertionPoint();

      /// Need to initialize some memory accesses outside the nested loop
      /// Reset the insertion point: the body of the innermost loop
      comet_debug() << "LoopSize: " << forLoops.size() << " Loop:\n";
      comet_vdump(forLoops[forLoops.size() - 1]);
      rewriter.setInsertionPoint(forLoops[forLoops.size() - 1]);

      Value const_index_0 = rewriter.create<ConstantIndexOp>(loc, 0);
      MemRefType memTy_alloc_Cnnz = MemRefType::get({1}, indexType);
      Value alloc_Cnnz = rewriter.create<memref::AllocOp>(loc, memTy_alloc_Cnnz);
      comet_debug() << " AllocOp for Cnnz: ";
      comet_vdump(alloc_Cnnz);

      std::vector<Value> alloc_Cnnz_insert_loc = {const_index_0};
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      auto store_Cnnz = rewriter.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz, alloc_Cnnz_insert_loc);
      comet_debug() << " StoreOp: ";
      comet_vdump(store_Cnnz);
#else
      rewriter.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz, alloc_Cnnz_insert_loc);
#endif

      // The following code block is needed to update Update C2pos in the case of output tensor is in DCSR
      Value Cnnz_index_old;
      Value alloc_Cnnz_row;
      if (sparse_format.compare("DCSR") == 0)
      {
        alloc_Cnnz_row = rewriter.create<memref::AllocOp>(loc, memTy_alloc_Cnnz);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
        auto store_Cnnz_row = rewriter.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
        comet_debug() << " StoreOp DCSR: ";
        comet_vdump(store_Cnnz_row);
#else
        rewriter.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
#endif
        // Get Cnnz_old
        Cnnz_index_old = rewriter.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
      }

      rewriter.restoreInsertionPoint(last_insertionPoint);

      comet_debug() << " dense_inputtensor_id: " << dense_inputtensor_id << "\n";
      comet_debug() << " sparse_inputtensor_id: " << sparse_inputtensor_id << "\n";
      Value denseInput_is_nonzero = rewriter.create<CmpFOp>(loc, CmpFPredicate::ONE, allLoads[dense_inputtensor_id], const_f64_0);
      auto if_nonzero = rewriter.create<scf::IfOp>(loc, denseInput_is_nonzero, /*WithElseRegion*/ false);
      comet_debug() << " If branch:\n";
      comet_vdump(if_nonzero);

      if (!if_nonzero.getThenRegion().empty())
      {

        rewriter.setInsertionPointToStart(&if_nonzero.getThenRegion().front());

        comet_debug() << "calculate product and sum in \n";
        Value elementWiseResult = getSemiringSecondVal(rewriter, loc, semiringSecond, allLoads[0], allLoads[1], compressedWorkspace);

        // Get Cnnz
        Value Cnnz_index = rewriter.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);

// Store product to Cval
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
        comet_debug() << "Store product to Cval\n";
        auto store_Cval = rewriter.create<memref::StoreOp>(loc, elementWiseResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], Cnnz_index);
        comet_debug() << " StoreOp: ";
        comet_vdump(store_Cval);

        // Update C1crd, C2crd
        comet_debug() << "Getting A1crd\n";
        comet_debug() << "allValueAccessIdx[" << sparse_inputtensor_id << "].size(): " << allAccessIdx[sparse_inputtensor_id].size() << "\n";
        comet_vdump(allAccessIdx[sparse_inputtensor_id][0]);

        for (unsigned int i = 0; i < allAccessIdx.size(); i++)
        {
          comet_debug() << "allAccessIdx[" << i << "].size(): " << allAccessIdx[i].size() << "\n";
          for (auto n : allAccessIdx[i])
          {
            comet_vdump(n);
          }
        }
#else
        rewriter.create<memref::StoreOp>(loc, elementWiseResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], Cnnz_index);
#endif

        comet_debug() << "Store C1crd\n";
        /// Branch out COO... CSR... DCSR...
        if (sparse_format.compare("COO") == 0)
        {
          comet_debug() << "COO format for Elementwise MulOp, update all coordinates\n";
          for (unsigned d = 0; d < rhsPerms[sparse_inputtensor_id].size(); d++)
          {
            Value crd = allAccessIdx[sparse_inputtensor_id][d];
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
            auto store_coo_crd = rewriter.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][2 * d + 1], Cnnz_index);
            comet_debug() << " COO StoreOp: ";
            comet_vdump(store_coo_crd);
#else
            rewriter.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1], Cnnz_index);
#endif
          }
        }
        else if (sparse_format.compare("CSR") == 0 || sparse_format.compare("DCSR") == 0)
        {
          for (unsigned int d = forLoops.size() - 1; d < rhsPerms[sparse_inputtensor_id].size(); d++)
          {
            Value crd = allAccessIdx[sparse_inputtensor_id][d];
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
            auto store_csr_crd = rewriter.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][2 * d + 1], Cnnz_index);
            comet_debug() << " CSR or DCSR StoreOp: ";
            comet_vdump(store_csr_crd);
#else
            rewriter.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1], Cnnz_index);
#endif
          }
        }

        // Update Cnnz
        comet_debug() << "Update Cnnz\n";
        Value const_index_1 = rewriter.create<ConstantIndexOp>(loc, 1);
        Value new_Cnnz_index = rewriter.create<AddIOp>(loc, Cnnz_index, const_index_1);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
        comet_debug() << "AddIOps (new_Cnnz_index): ";
        comet_vdump(new_Cnnz_index);
        auto store_updated_cnnz = rewriter.create<memref::StoreOp>(loc, new_Cnnz_index, alloc_Cnnz, alloc_Cnnz_insert_loc);
        comet_debug() << " Update Cnnz (store new value) StoreOp: ";
        comet_vdump(store_updated_cnnz);
#else
        rewriter.create<memref::StoreOp>(loc, new_Cnnz_index, alloc_Cnnz, alloc_Cnnz_insert_loc);
#endif
      }

      // Need to identify dense tensor upperbound to be able to update Cpos and Csize arrays
      std::vector<Value> denseAllocs = tensors_rhs_Allocs[dense_inputtensor_id];
      assert(denseAllocs.size() == 1);

      comet_debug() << " DenseAllocs: ";
      auto inputType = denseAllocs[0].getType();
      std::vector<Value> denseDimsSize;
      for (unsigned rank = 0; rank < inputType.cast<mlir::MemRefType>().getRank(); rank++)
      {
        auto dimSize = inputType.cast<mlir::MemRefType>().getDimSize(rank);
        Value upperBound;
        if (dimSize == ShapedType::kDynamic)
        {
          comet_debug() << " This dimension is a dynamic size:\n";
          unsigned dynamicDimPos = inputType.dyn_cast<MemRefType>().getDynamicDimIndex(rank);
          comet_debug() << " DynamicDimPos: " << dynamicDimPos << "\n";
          upperBound = denseAllocs[0].getDefiningOp()->getOperand(dynamicDimPos);
          comet_vdump(upperBound);
        }
        else
        {
          comet_debug() << " This dimension is a static size\n";
          upperBound = rewriter.create<ConstantIndexOp>(loc, dimSize);
          comet_vdump(upperBound);
        }
        denseDimsSize.push_back(upperBound);
      }

      // To update Cpos
      if (sparse_format.compare("CSR") == 0)
      {
        rewriter.setInsertionPointAfter(forLoops[0]);
        Value const_index_1 = rewriter.create<ConstantIndexOp>(loc, 1);
        Value arg0_next = rewriter.create<AddIOp>(loc, forLoops[1].getInductionVar(), const_index_1);
        comet_debug() << "AddIOp (arg0_next): ";
        comet_vdump(arg0_next);

        Value Cnnz_index_final = rewriter.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
        rewriter.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][2], arg0_next);

        rewriter.setInsertionPointAfter(forLoops[1]);
        // Update C2pos[0]
        comet_debug() << "Update C2pos[0]\n";
        std::vector<Value> insert_loc_0 = {const_index_0};
        rewriter.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][2], insert_loc_0);

        // Update C1pos[0]
        comet_debug() << "Update C1pos[0]\n";
        Value dim0_index = denseDimsSize[0];
        rewriter.create<memref::StoreOp>(loc, dim0_index, main_tensors_all_Allocs[2][0], insert_loc_0);
      }
      else
      {
        if (sparse_format.compare("DCSR") == 0)
        {
          // Update C2pos
          comet_debug() << "Update DCSR C2pos\n";
          rewriter.setInsertionPointAfter(forLoops[0]);
          auto Cnnz_index_new = rewriter.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
          auto has_nnz_row = rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, Cnnz_index_new, Cnnz_index_old);
          auto has_nnz_row_ifOp = rewriter.create<scf::IfOp>(loc, has_nnz_row, /*WithElseRegion*/ false);
          comet_debug() << " If branch:\n";
          comet_vdump(has_nnz_row_ifOp);

          if (!has_nnz_row_ifOp.getThenRegion().empty())
          {
            rewriter.setInsertionPointToStart(&has_nnz_row_ifOp.getThenRegion().front());

            Value const_index_1 = rewriter.create<ConstantIndexOp>(loc, 1);
            Value arg0_next = rewriter.create<AddIOp>(loc, forLoops[1].getInductionVar(), const_index_1);
            comet_debug() << "AddIOp (arg0_next): ";
            comet_vdump(arg0_next);

            Value Cnnz_index_final = rewriter.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
            rewriter.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][2], arg0_next); // C2pos
            Value Cnnz_row_index = rewriter.create<memref::LoadOp>(loc, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
            Value idx_i = allAccessIdx[sparse_inputtensor_id][0];
            rewriter.create<memref::StoreOp>(loc, /*i*/ idx_i, main_tensors_all_Allocs[2][1], Cnnz_row_index); // C1crd
            Value Cnnz_row_index_new = rewriter.create<AddIOp>(loc, Cnnz_row_index, const_index_1);
            comet_debug() << "AddIOp (Cnnz_row_index_new): ";
            comet_vdump(Cnnz_row_index_new);
            rewriter.create<memref::StoreOp>(loc, Cnnz_row_index_new, alloc_Cnnz_row, alloc_Cnnz_insert_loc); // Update Cnnz_row
          }

          rewriter.setInsertionPointAfter(forLoops[1]);
          Value const_index_1 = rewriter.create<ConstantIndexOp>(loc, 1);
          std::vector<Value> insert_loc_1 = {const_index_1};

          // Update C2pos[0]
          std::vector<Value> insert_loc_0 = {const_index_0};
          rewriter.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][2], insert_loc_0);

          // Update C1pos[0], C1pos[1]
          Value Cnnz_row_index = rewriter.create<memref::LoadOp>(loc, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
          rewriter.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][0], insert_loc_0);
          rewriter.create<memref::StoreOp>(loc, Cnnz_row_index, main_tensors_all_Allocs[2][0], insert_loc_1);
        }
        else
        {
          if (sparse_format.compare("COO") == 0)
          {
            // Finally, Update C1pos
            comet_debug() << "Update C1pos\n";
            rewriter.setInsertionPointAfter(forLoops[0]);
            Value Cnnz_index_final = rewriter.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
            Value const_index_1 = rewriter.create<ConstantIndexOp>(loc, 1);
            rewriter.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][0], const_index_0);
            rewriter.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][0], const_index_1);
          }
          else
            llvm::errs() << "// Coordinate values are not updated for output sparse tensor in " << sparse_format << " format\n";
        }
      }

    } // end if (isMixedMode && isElementwise)
    else
    {
      // calculate elementWise operation and reduction for general dense or mix mode computation (which has dense output)
      comet_debug() << "calculate elementWise operation and reduction for general dense or mix mode computation (which has dense output)\n";
      Value elementWiseResult = getSemiringSecondVal(rewriter, loc, semiringSecond, allLoads[0], allLoads[1], compressedWorkspace);
      Value reduceResult = getSemiringFirstVal(rewriter, loc, semiringFirst, allLoads[2], elementWiseResult, compressedWorkspace);
      rewriter.create<memref::StoreOp>(loc, reduceResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], allValueAccessIdx[2]);
    }
  }
}

/// 1. Get the nested loops
/// ---1.1 the nested loops corresponding indices can be infered from ancestors_wp
/// 2. get lhs and rhs. if only 1 rhs, then it's a fill op; otherwise, binary op
/// Note: 1. The auxiliary arrays does not contain the perms/formats information
///       2. We only apply the compressed workspace on the output of the tensor, then in this case, the workspace tensors will not be in the same side with the main tensors.
///         (main tensors: such as A, B, C, w;  auxiliary tensors: such as w_index_list ...)
void genCmptOps(indexTree::IndexTreeComputeOp cur_op,
                indexTree::IndexTreeOp rootOp,
                PatternRewriter &rewriter,
                OpsTree *opstree,
                std::vector<Value> ancestorsWps)
{
  comet_debug() << " calling genCmptOps\n";
  Location loc = rootOp.getLoc();
  comet_debug() << " \n";

  comet_debug() << " Current IndexTreeComputeOp:";
  comet_vdump(cur_op);

  const bool comp_worksp_opt(cur_op.getCompWorkspOpt());
  comet_debug() << " comp_worksp_opt (bool: true is compressed): " << comp_worksp_opt << "\n";

  // Two cases:
  // 1. for the initial workspace, only 1 auxiliary vector w
  // 2. for the compressed workspace, there are 4 auxiliaty vectors, w, w_already_set, w_index_list, w_index_list_size

  /// The insertion location should be "the end of the body of parent loop"
  std::vector<OpsTree *> ancestorsOps;
  getAncestorsOps(opstree, ancestorsOps);
  comet_debug() << " ancestorsOps.size(): " << ancestorsOps.size() << "\n";
  for (unsigned int i = 0; i < ancestorsOps.size(); i++)
  {
    comet_debug() << " ancestorsOps[i]->id:" << ancestorsOps[i]->id << "\n";
  }

  /// 1. get the nested loops, from innermost to outermost order
  std::vector<scf::ForOp> nested_forops;
  std::vector<Value> nested_AccessIdx;

  for (unsigned int i = 0; i < ancestorsOps.size(); i++)
  {
    comet_debug() << " ancestorsOps[" << i << "]->forOps.size(): " << ancestorsOps[i]->forOps.size() << ", ancestorsOps->id: "
                  << ancestorsOps[i]->id << "\n";
    if (ancestorsOps[i]->forOps.size() > 0)
    { // for loops OpsTree node
      for (int j = ancestorsOps[i]->forOps.size() - 1; j >= 0; j--)
      {
        comet_debug() << " j: " << j << "\n";
        nested_forops.push_back(ancestorsOps[i]->forOps[j]);
        comet_debug() << "AccessIdx: " << ancestorsOps[i]->accessIdx[j] << "\n";
        nested_AccessIdx.push_back(ancestorsOps[i]->accessIdx[j]);
      }
    }
  }
  comet_debug() << " nested_forops.size(): " << nested_forops.size() << "\n";
  std::vector<int64_t> nested_forops_indices;
  for (unsigned int i = 0; i < ancestorsWps.size(); i++)
  {
    comet_debug() << " ";
    comet_vdump(ancestorsWps[i]);

    if (indexTree::IndexTreeIndicesOp cur_op = dyn_cast<mlir::indexTree::IndexTreeIndicesOp>(ancestorsWps[i].getDefiningOp()))
    {
      // Get indices
      ArrayAttr op_indices = cur_op.getIndices();

      if (op_indices.size() > 0)
      { // for loops OpsTree node
        for (int j = op_indices.size() - 1; j >= 0; j--)
        {
          // Get the indices;
          int64_t idx = op_indices[j].cast<IntegerAttr>().getInt();
          nested_forops_indices.push_back(idx);
        }
      }
    }
  }
  comet_debug() << " nested_forops_indices.size(): " << nested_forops_indices.size() << "\n";

  assert(nested_forops.size() == nested_forops_indices.size() && "nested_forops.size() != nested_forops_indices.size()");

  /// Reset the insertion point: the body of the innermost loop
  assert(nested_forops.size() > 0 && "No loops\n");
  comet_debug() << " ";
  comet_pdump(nested_forops[0].getBody());
  comet_debug() << " ";
  comet_pdump(nested_forops[0].getBody()->getTerminator());
  rewriter.setInsertionPoint(nested_forops[0].getBody()->getTerminator());

  auto f64Type = rewriter.getF64Type();
  auto indexType = IndexType::get(rootOp.getContext());

  Value const_f64_0 = rewriter.create<ConstantOp>(loc, f64Type, rewriter.getF64FloatAttr(0));
  Value const_i1_0 = rewriter.create<ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(0));
  Type unrankedMemrefType_index = UnrankedMemRefType::get(indexType, 0);

  /// Analyze the leafop, Get the tensors, rhs, lhs, and operator_type
  /// --- only one rhs, it will be a fill op; if two, check op_type (+, +=, *=)
  /// Check the indices contained in each tensor
  /// Generate loadOp, compute ops, StoreOp.
  comet_debug() << " \n";
  // New version
  Value lhs = cur_op.getLhs().getDefiningOp()->getOperand(0);
  comet_debug() << " ";
  comet_vdump(lhs);
// lhs is TensorLoadOp
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  Value lhs_alloc = (lhs.getDefiningOp())->getOperand(0);
  comet_debug() << " ";
  comet_vdump(lhs_alloc);
#endif

  comet_debug() << " ";
  comet_vdump(cur_op);
  std::vector<Value> tensors_rhs;
  for (auto n : cur_op.getRhs())
  {
    comet_debug() << " ";
    comet_vdump(n);
    for (unsigned i = 0; i < n.getDefiningOp()->getNumOperands(); i++)
    {
      comet_debug() << " ";
      comet_vdump(n.getDefiningOp()->getOperand(i));
      tensors_rhs.push_back(n.getDefiningOp()->getOperand(i));
    }
  }

  std::vector<Value> tensors_lhs;
  for (unsigned i = 0; i < cur_op.getLhs().getDefiningOp()->getNumOperands(); i++)
  {
    comet_debug() << " ";
    comet_vdump(cur_op.getLhs().getDefiningOp()->getOperand(i));
    tensors_lhs.push_back(cur_op.getLhs().getDefiningOp()->getOperand(i));
  }

  // Currently, only one case, the rhs is constant. Wj = 0.0;
  std::vector<Value> tensors_all = tensors_rhs; // Including ConstantOp
  tensors_all.insert(tensors_all.end(), tensors_lhs.begin(), tensors_lhs.end());
  comet_debug() << " tensors_all.size(): " << tensors_all.size() << "\n";

  std::vector<std::vector<Value>> tensors_lhs_Allocs = getAllAllocs(tensors_lhs);
  comet_debug() << " tensors_lhs_Allocs.size(): " << tensors_lhs_Allocs.size() << "\n";
  std::vector<std::vector<Value>> tensors_rhs_Allocs = getAllAllocs(tensors_rhs);
  comet_debug() << " tensors_rhs_Allocs.size(): " << tensors_rhs_Allocs.size() << "\n";

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  comet_debug() << " tensors_rhs_Allocs: \n";
  for (auto m : tensors_rhs_Allocs)
  {
    comet_debug() << " ";
    for (auto n : m)
    {
      comet_vdump(n);
    }
    comet_debug() << "\n";
  }
#endif

  OpBuilder builder(cur_op);
  std::vector<std::vector<int>> allPerms;
  getPermsOfComputeOp(cur_op.getOperation()->getResult(0), allPerms);

  comet_debug() << " allPerms: \n";
  for (auto m : allPerms)
  {
    comet_debug() << " "; // print_vector(m);
    for (auto n : m)
    {
      comet_debug() << n << " ";
    }
    comet_debug() << "\n";
  }

  std::vector<std::vector<std::string>> allFormats;
  getFormatsOfComputeOp(cur_op.getOperation()->getResult(0), allFormats);
  comet_debug() << " allFormats: \n";
  for (auto m : allFormats)
  {
    comet_debug() << " ";
    for (auto n : m)
    {
      comet_debug() << n << " ";
    }
    comet_debug() << "\n";
  }

  comet_debug() << " ";
  comet_vdump(cur_op);

  std::vector<std::vector<std::string>> rhsFormats;
  getRHSFormatsOfComputeOp(cur_op.getOperation()->getResult(0), rhsFormats);

  std::vector<std::vector<std::string>> lhsFormats;
  getLHSFormatsOfComputeOp(cur_op.getOperation()->getResult(0), lhsFormats);

  assert(allPerms.size() == allFormats.size() && "allPerms.size() != allFormats.size()\n");
  for (unsigned int m = 0; m < allPerms.size(); m++)
  {
    assert(allPerms[m].size() == allFormats[m].size() && "allPerms[m].size() != allFormats[m].size()\n");
  }
  comet_debug() << " allPerms.size(): " << allPerms.size() << "\n";
  // tensor_nums means the actual tensors except the auxiliary tensors
  // Suppose for LHSOp, there are "n" real tensors, then allPerms[m].size()

  std::vector<std::vector<int>> allPerms_rhs;
  getRHSPermsOfComputeOp(cur_op.getOperation()->getResult(0), allPerms_rhs);
  comet_debug() << " allPerms_rhs.size(): " << allPerms_rhs.size() << "\n";
  std::vector<std::vector<int>> allPerms_lhs;
  getLHSPermsOfComputeOp(cur_op.getOperation()->getResult(0), allPerms_lhs);

  comet_debug() << " allPerms_lhs.size(): " << allPerms_lhs.size() << "\n";
  std::vector<Value> main_tensors_all;
  std::vector<Value> main_tensors_rhs;
  std::vector<Value> main_tensors_lhs;
  if (tensors_rhs.size() == allPerms_rhs.size())
  { // all are "main" tensors
    main_tensors_rhs.insert(main_tensors_rhs.end(), tensors_rhs.begin(), tensors_rhs.end());
  }
  else
  {                                                                                                                                   // the rhs contains the auxiliary tensors
    assert(allPerms_rhs.size() == 1 && " rhs contains auxiliary tensors and main tensors at the same time, not support currently\n"); // only 1 main tensor on rhs
    main_tensors_rhs.push_back(tensors_rhs[0]);
  }
  comet_debug() << " main_tensors_rhs.size(): " << main_tensors_rhs.size() << "\n";

  if (tensors_lhs.size() == allPerms_lhs.size())
  { // all are "main" tensors
    main_tensors_lhs.insert(main_tensors_lhs.end(), tensors_lhs.begin(), tensors_lhs.end());
  }
  else
  {                                                                                                                                   // the lhs contains the auxiliary tensors
    assert(allPerms_lhs.size() == 1 && " lhs contains auxiliary tensors and main tensors at the same time, not support currently\n"); // only 1 main tensor on lhs
    main_tensors_lhs.push_back(tensors_lhs[0]);
  }
  comet_debug() << " main_tensors_lhs.size(): " << main_tensors_lhs.size() << "\n";

  main_tensors_all = main_tensors_rhs;
  main_tensors_all.insert(main_tensors_all.end(), main_tensors_lhs.begin(), main_tensors_lhs.end());
  comet_debug() << " main_tensors_all.size(): " << main_tensors_all.size() << "\n";

  int main_tensor_nums = main_tensors_all.size();
  comet_debug() << " main_tensor_nums: " << main_tensor_nums << "\n";

  // Check the loop arg in each tensor
  std::vector<std::vector<Value>> main_tensors_all_Allocs = getAllAllocs(main_tensors_all);
  comet_debug() << " main_tensors_all_Allocs.size(): " << main_tensors_all_Allocs.size() << "\n";

  std::vector<std::vector<Value>> allLoopsArg(main_tensor_nums);
  std::vector<std::vector<Value>> allAccessIdx(main_tensor_nums);
  for (unsigned int i = 0; i < main_tensors_all.size(); i++)
  {
    for (unsigned int j = 0; j < allPerms[i].size(); j++)
    {
      unsigned int index_loc = findIndexInVector<int64_t>(nested_forops_indices, allPerms[i][j]);
      comet_debug() << " index_loc " << index_loc << "\n";
      comet_debug() << " Perm: " << allPerms[i][j] << "\n";
      comet_debug() << " Format: " << allFormats[i][j] << "\n";
      assert(index_loc < nested_forops.size() && "index_loc < nested_forops.size(), i.e. the index not exist in nested for loop\n");
      allLoopsArg[i].push_back(nested_forops[index_loc].getInductionVar());
      allAccessIdx[i].push_back(nested_AccessIdx[index_loc]);
    }
    // Consider for the case w_index_list_size
    // if allPerms[i].size() == 0
  }

  std::vector<std::vector<Value>> allValueAccessIdx(main_tensor_nums);
  for (int i = 0; i < main_tensor_nums; i++)
  { // If constantOp, do not consider it
    comet_debug() << " ";
    comet_vdump(main_tensors_all[i]);
    if (main_tensors_all[i].getType().isa<tensorAlgebra::SparseTensorType>())
    { // sparse tensor

      // Find the last sparse index m, then loop_arg * all dense loop args
      unsigned lastSparseIndexLoc = allPerms[i].size();
      for (int d = (int)allPerms[i].size() - 1; d >= 0; d--)
      {
        if (allFormats[i][d].compare(0, 1, "D") != 0 && allFormats[i][d].compare(0, 1, "S") != 0)
        { // sparse dimension and has a loop, i.e. "CU" or "CN"
          lastSparseIndexLoc = d;
          break;
        }
      }
      // Calculate for ModeGeneric style format: [CN, S, D (, ... ) ]
      auto valueAccessIdx_part = allLoopsArg[i][lastSparseIndexLoc];
      if (lastSparseIndexLoc < allPerms[i].size() - 1)
      { // There is dense index after the sparse index
        unsigned int last_d = lastSparseIndexLoc + 1;
        for (unsigned int d = lastSparseIndexLoc + 1; d < allPerms[i].size(); d++)
        { // i=0
          if (allFormats[i][d].compare(0, 1, "D") == 0)
          {
            // Get dense dim size
            auto index_0 = rewriter.create<ConstantIndexOp>(loc, 0);
            std::vector<Value> upper_indices = {index_0};
            auto upperBound = rewriter.create<memref::LoadOp>(loc, main_tensors_all_Allocs[i][4 * d], upper_indices);
            comet_vdump(upperBound);
            valueAccessIdx_part = rewriter.create<MulIOp>(loc, upperBound, valueAccessIdx_part);
            last_d = d;
          }
        }
        if (allFormats[i][last_d].compare(0, 1, "D") == 0)
        {
          comet_debug() << " ";
          comet_vdump(allLoopsArg[i][allLoopsArg[i].size() - 1]);
          comet_vdump(valueAccessIdx_part);
          valueAccessIdx_part = rewriter.create<AddIOp>(loc, allLoopsArg[i][allLoopsArg[i].size() - 1], valueAccessIdx_part);
          comet_debug() << " AddIOps (valueAccessIdx_part): ";
          comet_vdump(valueAccessIdx_part);
        }
      }

      allValueAccessIdx[i].push_back(valueAccessIdx_part);
    }
    else if (main_tensors_all[i].getType().isa<TensorType>())
    { // dense tensor
      allValueAccessIdx[i] = allAccessIdx[i];
    }
  }

  for (unsigned int i = 0; i < allValueAccessIdx.size(); i++)
  {
    comet_debug() << "allValueAccessIdx[" << i << "].size(): " << allValueAccessIdx[i].size() << ", main_tensors_all_Allocs[" << i << "].size()-1: " << main_tensors_all_Allocs[i].size() - 1 << "\n";
  }

  int rhs_loc = 0;
  int lhs_loc = main_tensors_rhs.size();
  if (main_tensors_rhs.size() == 1)
  { // Generate "a = b"
    if (ConstantOp cstop = dyn_cast<ConstantOp>(main_tensors_rhs[0].getDefiningOp()))
    { // "a = 1.0"
      comet_debug() << " ";
      comet_vdump(cstop);
      if (comp_worksp_opt) // true attr means compressed workspace
      {

        comet_debug() << " compressed_workspace ComputeOp\n";
        std::vector<MemRefType> tensors_lhs_Allocs_type;
        for (unsigned i = 0; i < tensors_lhs_Allocs.size(); i++)
        {
          // only 1 alloc op for dense type tensor or constant
          assert(tensors_lhs_Allocs[i].size() == 1 && " more than 1 alloc op for dense type tensor or constant\n");
          MemRefType resultType = tensors_lhs_Allocs[i][0].getDefiningOp()->getResult(0).getType().cast<MemRefType>();
          comet_debug() << " ";
          comet_vdump(resultType);
          tensors_lhs_Allocs_type.push_back(resultType);
        }

        // Generate Store 1.0, A[...]  this op
        // this case: allPerms[0] is empty, allFormats[0] is empty
        comet_debug() << " cstop.getValue(): " << cstop.getValue() << "\n";
        comet_debug() << " ";
        comet_vdump(main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1]);
        comet_debug() << " tensors_lhs_Allocs.size(): " << tensors_lhs_Allocs.size() << "\n";
        comet_debug() << " ";

        insertInitialize(loc, cstop, main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1], rewriter);
        comet_debug() << " ";
      }
      else
      { // initial workspace
        // Generate Store 1.0, A[...]  this op
        // this case: allPerms[0] is empty, allFormats[0] is empty
        comet_debug() << " cstop.getValue(): " << cstop.getValue() << "\n";
        comet_debug() << " ";
        comet_vdump(main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1]);

        if (allValueAccessIdx[lhs_loc].size() > 0)
        {
          rewriter.create<memref::StoreOp>(loc, cstop, main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1], allValueAccessIdx[lhs_loc]);
        }
        else
        {
          insertInitialize(loc, cstop, main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1], rewriter);
        }
      }
    }
    else if (main_tensors_rhs[0].getType().isa<mlir::TensorType>())
    { // Cij = Wj
      // When Cij is dense type
      if (lhs.getType().isa<mlir::TensorType>())
      {
        // %1 = load b[...]
        // store %1, a[...]
        comet_debug() << " main_tensors_all_Allocs[" << rhs_loc << "].size(): " << main_tensors_all_Allocs[rhs_loc].size() << ", allValueAccessIdx[" << rhs_loc << "].size(): " << allValueAccessIdx[rhs_loc].size() << "\n";

        Value rhs_value = rewriter.create<memref::LoadOp>(loc, main_tensors_all_Allocs[rhs_loc][main_tensors_all_Allocs[rhs_loc].size() - 1], allValueAccessIdx[rhs_loc]);
        comet_debug() << " ";
        comet_vdump(rhs_value);

        comet_debug() << " ";
        comet_vdump(main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1]);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
        auto s1 = rewriter.create<memref::StoreOp>(loc, rhs_value, main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1], allValueAccessIdx[lhs_loc]);
        comet_debug() << " ";
        comet_vdump(s1);
#else
        rewriter.create<memref::StoreOp>(loc, rhs_value, main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1], allValueAccessIdx[lhs_loc]);
#endif
      }
      // Cij = Wj
      else if (lhs.getType().isa<tensorAlgebra::SparseTensorType>())
      {
        // TODO(patrick): This needs to be fixed
        // TODO(gkestor): get tensor ranks by functions
        //unsigned int lhs_ranks = (lhs.getDefiningOp()->getNumOperands() - 2) / 5;
        unsigned int lhs_ranks = 2;

        //[0...2d,2d+1...4d+1,4d+2...5d+1]
        //unsigned int lhs_val_size_loc = 4 * lhs_ranks + 1;
        //unsigned int lhs_2crd_size_loc = 4 * lhs_ranks;
        //unsigned int lhs_2pos_size_loc = 4 * lhs_ranks - 1;
        unsigned int lhs_val_size_loc = 15;
        unsigned int lhs_2crd_size_loc = 12;
        unsigned int lhs_2pos_size_loc = 11;

        // [0...2d, 2d+1...4d+1, 4d+2...5d+1]
        comet_debug() << " ";
        comet_pdump(lhs.getDefiningOp());
        comet_pdump(lhs.getDefiningOp()->getParentOp());
        comet_debug() << " ";
        comet_vdump(lhs.getDefiningOp()->getOperand(lhs_val_size_loc));

        Value lhs_nnz_operand = lhs.getDefiningOp()->getOperand(lhs_val_size_loc);
        Value lhs_nnz_op;
        comet_debug() << " ";
        comet_vdump(lhs_nnz_operand);
        if (isa<IndexCastOp>(lhs_nnz_operand.getDefiningOp()))
        {
          lhs_nnz_op = lhs_nnz_operand.getDefiningOp()->getOperand(0);
        }
        else
        {
          lhs_nnz_op = lhs_nnz_operand;
        }
        comet_debug() << " ";
        comet_vdump(lhs_nnz_op);
        auto lhs_nnz_load = cast<memref::LoadOp>(lhs_nnz_op.getDefiningOp());                  // index
        Value lhs_nnz_alloc = cast<memref::AllocOp>(lhs_nnz_load.getMemRef().getDefiningOp()); // index

        Value cst_0_index = rewriter.create<ConstantIndexOp>(loc, 0);
        Value lhs_nnz = rewriter.create<memref::LoadOp>(loc, lhs_nnz_alloc, ValueRange{cst_0_index});

        std::vector<Value> lhs_accessIndex = {lhs_nnz};
        
        Value lhs_val = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1];
        comet_debug() << " ";
        comet_vdump(lhs_val);

        if (comp_worksp_opt) // true attr means compressed workspace
        {
          // Get the parent for op, change the upperbound as w_index_list_size
          auto last_insertionPoint = rewriter.saveInsertionPoint();

          Value const_index_0 = rewriter.create<ConstantIndexOp>(loc, 0);
          scf::ForOp theForop = dyn_cast<scf::ForOp>(const_index_0.getDefiningOp()->getParentOp());
          comet_debug() << " ";
          comet_vdump(theForop);

          rewriter.setInsertionPoint(theForop);

          Value const_index_00 = rewriter.create<ConstantIndexOp>(loc, 0);
          Value w_index_list_size = rewriter.create<memref::LoadOp>(loc, tensors_rhs_Allocs[3][0], const_index_00);

          std::string quick_sort_Str = "quick_sort";
          Value w_index_list_cast = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_index, tensors_rhs_Allocs[2][0]);
          rewriter.create<func::CallOp>(loc, quick_sort_Str, SmallVector<Type, 2>{}, ValueRange{w_index_list_cast, w_index_list_size});

          theForop.setUpperBound(w_index_list_size);
          comet_debug() << " ";
          comet_vdump(theForop);

          rewriter.restoreInsertionPoint(last_insertionPoint);
          Value crd_index = rewriter.create<memref::LoadOp>(loc, tensors_rhs_Allocs[2][0], theForop.getInductionVar());
          Value c_value = rewriter.create<memref::LoadOp>(loc, tensors_rhs_Allocs[0][0], crd_index);
          // Fill CVal
          rewriter.create<memref::StoreOp>(loc, c_value, lhs_val, ValueRange{lhs_nnz});

          // w_already_set[crd_j] = 0
          rewriter.create<memref::StoreOp>(loc, const_i1_0, tensors_rhs_Allocs[1][0], ValueRange{crd_index});

          comet_debug() << " lhs_loc: " << lhs_loc << "\n";
          comet_debug() << " format: " << allFormats[lhs_loc][allFormats[lhs_loc].size() - 1] << "\n";
          if (allFormats[lhs_loc][allFormats[lhs_loc].size() - 1].compare(0, 2, "CU") == 0)
          {
            Value lhs_2crd = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 4];
            comet_debug() << " ";
            comet_vdump(lhs_2crd);

            rewriter.create<memref::StoreOp>(loc, crd_index, lhs_2crd, ValueRange{lhs_nnz});
          }

          comet_debug() << "\n";
          Value cst_1_index = rewriter.create<ConstantIndexOp>(loc, 1);
          comet_debug() << " ";
          comet_vdump(lhs_nnz);
          Value lhs_nnz_new = rewriter.create<AddIOp>(loc, lhs_nnz, cst_1_index);
          comet_debug() << " AddIOps (lhs_nnz_new): ";
          comet_vdump(lhs_nnz_new);
          comet_debug() << " ";
          comet_vdump(lhs_nnz_alloc);

          rewriter.create<memref::StoreOp>(loc, lhs_nnz_new, lhs_nnz_alloc, ValueRange{cst_0_index});

          Value lhs_2crd = lhs.getDefiningOp()->getOperand(lhs_2crd_size_loc);
          Value lhs_2crd_op;
          comet_vdump(lhs_2crd);
          if (isa<IndexCastOp>(lhs_2crd.getDefiningOp()))
          {
            lhs_2crd_op = lhs_2crd.getDefiningOp()->getOperand(0);
          }
          else
          {
            lhs_2crd_op = lhs_2crd;
          }
          comet_debug() << " ";
          comet_vdump(lhs_2crd_op);
          auto c2crd_size_load = cast<memref::LoadOp>(lhs_2crd_op.getDefiningOp());                    // index
          Value c2crd_size_alloc = cast<memref::AllocOp>(c2crd_size_load.getMemRef().getDefiningOp()); // index
          comet_debug() << " ";
          comet_vdump(c2crd_size_alloc);

          rewriter.create<memref::StoreOp>(loc, lhs_nnz_new, c2crd_size_alloc, ValueRange{cst_0_index});

          // Fill C2pos
          comet_debug() << " \n";
          auto prev_forop = nested_forops[nested_forops.size() - 1 - 1];
          rewriter.setInsertionPointAfter(prev_forop);

          Value lhs_2pos_0 = lhs.getDefiningOp()->getOperand(lhs_2pos_size_loc);
          Value lhs_2pos_op;
          comet_debug() << " ";
          comet_vdump(lhs_2pos_0);
          if (isa<IndexCastOp>(lhs_2pos_0.getDefiningOp()))
          {
            lhs_2pos_op = lhs_2pos_0.getDefiningOp()->getOperand(0);
          }
          else
          {
            lhs_2pos_op = lhs_2pos_0;
          }
          comet_debug() << " ";
          comet_vdump(lhs_2pos_op);
          auto c2pos_size_load = cast<memref::LoadOp>(lhs_2pos_op.getDefiningOp());                    // index
          Value c2pos_size_alloc = cast<memref::AllocOp>(c2pos_size_load.getMemRef().getDefiningOp()); // index
          Value cst_index_000 = rewriter.create<ConstantIndexOp>(loc, 0);
          Value c2pos_size_value = rewriter.create<memref::LoadOp>(loc, c2pos_size_alloc, ValueRange{cst_index_000});

          Value c2crd_size_nnz = rewriter.create<memref::LoadOp>(loc, c2crd_size_alloc, ValueRange{cst_index_000});

          // store crd_size into pos
          Value lhs_2pos = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 5];
          comet_debug() << " ";
          comet_vdump(lhs_2pos);
          rewriter.create<memref::StoreOp>(loc, c2crd_size_nnz, lhs_2pos, ValueRange{c2pos_size_value});

          Value cst_index_1 = rewriter.create<ConstantIndexOp>(loc, 1);
          comet_debug() << " ";
          comet_vdump(c2pos_size_value);
          Value c2pos_size_value_new = rewriter.create<AddIOp>(loc, c2pos_size_value, cst_index_1);
          comet_debug() << " AddIOps (c2pos_size_value_new): ";
          comet_vdump(c2pos_size_value_new);

          rewriter.create<memref::StoreOp>(loc, c2pos_size_value_new, c2pos_size_alloc, ValueRange{cst_index_000});
        }
        else
        {

          // %1 = load b[...]
          // if(%1 != 0) {
          //    Cnnz = load Cop.operand(4d+1)
          //    store %1, cval[Cnnz]
          //    store Cnnz+1, Cop.operand(4d+1)
          // }
          comet_debug() << " main_tensors_all_Allocs[" << rhs_loc << "].size(): " << main_tensors_all_Allocs[rhs_loc].size() << ", allValueAccessIdx[" << rhs_loc << "].size(): " << allValueAccessIdx[rhs_loc].size() << "\n";
          Value rhs_value = rewriter.create<memref::LoadOp>(loc, main_tensors_all_Allocs[rhs_loc][main_tensors_all_Allocs[rhs_loc].size() - 1], allValueAccessIdx[rhs_loc]);
          comet_debug() << " ";
          comet_vdump(rhs_value);
          Value isNonzero = rewriter.create<CmpFOp>(loc, CmpFPredicate::ONE, rhs_value, const_f64_0);
          comet_debug() << " ";
          comet_vdump(isNonzero);
          auto if_nonzero = rewriter.create<scf::IfOp>(loc, isNonzero, /*WithElseRegion*/ false);
          comet_debug() << " If branch:\n";
          comet_vdump(if_nonzero);

          if (!if_nonzero.getThenRegion().empty())
          {
            auto last_insertionPoint = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointToStart(&if_nonzero.getThenRegion().front());

            rewriter.create<memref::StoreOp>(loc, rhs_value, lhs_val, lhs_accessIndex);

            /// update pos/crd arrays
            // Fill C2crd in CSR format, parent loop's accessIdx
            /// Check format j in the output
            if (allFormats[lhs_loc][allFormats[lhs_loc].size() - 1].compare(0, 2, "CU") == 0)
            {
              Value crd_index = allAccessIdx[allAccessIdx.size() - 1][allAccessIdx[allAccessIdx.size() - 1].size() - 1];
              comet_debug() << " ";
              comet_vdump(crd_index);
              Value lhs_2crd = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 2];
              comet_debug() << " ";
              comet_vdump(lhs_2crd);

              rewriter.create<memref::StoreOp>(loc, crd_index, lhs_2crd, lhs_accessIndex);
            }

            comet_debug() << "\n";
            Value cst_1_index = rewriter.create<ConstantIndexOp>(loc, 1);
            comet_debug() << " ";
            comet_vdump(lhs_nnz);
            Value lhs_nnz_new = rewriter.create<AddIOp>(loc, lhs_nnz, cst_1_index);
            comet_debug() << " AddIOps: (lhs_nnz_new)";
            comet_vdump(lhs_nnz_new);
            comet_debug() << " ";
            comet_vdump(lhs_nnz_alloc);

            rewriter.create<memref::StoreOp>(loc, lhs_nnz_new, lhs_nnz_alloc, ValueRange{cst_0_index});

            comet_debug() << "\n";
            Value lhs_2crd = lhs.getDefiningOp()->getOperand(lhs_2crd_size_loc);
            Value lhs_2crd_op;
            comet_vdump(lhs_2crd);
            if (isa<IndexCastOp>(lhs_2crd.getDefiningOp()))
            {
              lhs_2crd_op = lhs_2crd.getDefiningOp()->getOperand(0);
            }
            else
            {
              lhs_2crd_op = lhs_2crd;
            }
            comet_debug() << " ";
            comet_vdump(lhs_2crd_op);
            // unsigned int lhs_2crd_size_loc = 4*lhs_ranks;
            auto c2crd_size_load = cast<memref::LoadOp>(lhs_2crd_op.getDefiningOp());                    // index
            Value c2crd_size_alloc = cast<memref::AllocOp>(c2crd_size_load.getMemRef().getDefiningOp()); // index
            comet_debug() << " ";
            comet_vdump(c2crd_size_alloc);

            rewriter.create<memref::StoreOp>(loc, lhs_nnz_new, c2crd_size_alloc, ValueRange{cst_0_index});

            comet_debug() << " \n";
            rewriter.restoreInsertionPoint(last_insertionPoint);
          }

          comet_debug() << " \n";
          auto prev_forop = nested_forops[nested_forops.size() - 1 - 1];
          rewriter.setInsertionPointAfter(prev_forop);

          comet_debug() << " ";
          comet_vdump(lhs.getDefiningOp()->getOperand(lhs_2pos_size_loc));
          Value lhs_2pos_0 = lhs.getDefiningOp()->getOperand(lhs_2pos_size_loc);
          Value lhs_2pos_op;
          comet_vdump(lhs_2pos_0);
          if (isa<IndexCastOp>(lhs_2pos_0.getDefiningOp()))
          {
            lhs_2pos_op = lhs_2pos_0.getDefiningOp()->getOperand(0);
          }
          else
          {
            lhs_2pos_op = lhs_2pos_0;
          }
          comet_debug() << " ";
          comet_vdump(lhs_2pos_op);
          auto c2pos_size_load = cast<memref::LoadOp>(lhs_2pos_op.getDefiningOp());                    // index
          Value c2pos_size_alloc = cast<memref::AllocOp>(c2pos_size_load.getMemRef().getDefiningOp()); // index
          Value cst_0_index = rewriter.create<ConstantIndexOp>(loc, 0);
          Value c2pos_size_value = rewriter.create<memref::LoadOp>(loc, c2pos_size_alloc, ValueRange{cst_0_index});

          Value lhs_2crd = lhs.getDefiningOp()->getOperand(lhs_2crd_size_loc);
          Value lhs_2crd_op;
          comet_vdump(lhs_2crd);
          if (isa<IndexCastOp>(lhs_2crd.getDefiningOp()))
          {
            lhs_2crd_op = lhs_2crd.getDefiningOp()->getOperand(0);
          }
          else
          {
            lhs_2crd_op = lhs_2crd;
          }
          comet_debug() << " ";
          comet_vdump(lhs_2crd_op);
          auto c2crd_size_load = cast<memref::LoadOp>(lhs_2crd_op.getDefiningOp());                    // index
          Value c2crd_size_alloc = cast<memref::AllocOp>(c2crd_size_load.getMemRef().getDefiningOp()); // index
          Value c2crd_size_nnz = rewriter.create<memref::LoadOp>(loc, c2crd_size_alloc, ValueRange{cst_0_index});

          // store crd_size into pos
          Value lhs_2pos = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 3];
          comet_debug() << " ";
          comet_vdump(lhs_2pos);

          rewriter.create<memref::StoreOp>(loc, c2crd_size_nnz, lhs_2pos, ValueRange{c2pos_size_value});

          Value cst_1_index = rewriter.create<ConstantIndexOp>(loc, 1);
          comet_debug() << " ";
          comet_vdump(c2pos_size_value);
          Value c2pos_size_value_new = rewriter.create<AddIOp>(loc, c2pos_size_value, cst_1_index);
          comet_debug() << " AddIOps (c2pos_size_value_new): ";
          comet_vdump(c2pos_size_value_new);

          rewriter.create<memref::StoreOp>(loc, c2pos_size_value_new, c2pos_size_alloc, ValueRange{cst_0_index});
        }
      }
    }
    // Vj = Bij
    else if (main_tensors_rhs[0].getType().isa<tensorAlgebra::SparseTensorType>())
    {
      // %Bvalue = load %Bval[..]
      // store %Bvalue, %v[%j]
      std::vector<Value> allLoads(main_tensor_nums);
      for (auto m = 0; m < main_tensor_nums; m++)
      {
        Value s = rewriter.create<memref::LoadOp>(loc, main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1], allValueAccessIdx[m]);
        allLoads[m] = s;
        comet_debug() << " ";
        comet_vdump(s);
      }
      comet_debug() << " allLoads.size(): " << allLoads.size() << "\n";

      rewriter.create<memref::StoreOp>(loc, allLoads[0], main_tensors_all_Allocs[1][main_tensors_all_Allocs[1].size() - 1], allValueAccessIdx[1]);
    }
  }
  else if (main_tensors_rhs.size() == 2)
  { // Generate " a = b * c" binary op

    auto semiringParts = cur_op.getSemiring().split('_');
    // check validity of semiring provided by user.
    if (!Semiring_reduceOps.contains(semiringParts.first) || !Semiring_ops.contains(semiringParts.second))
    {
      llvm::errs() << "Not supported semiring operator: "
                   << semiringParts.first << " or " << semiringParts.second << " \n";
      llvm::errs() << "Please report this error to the developers!\n";
      // we should not proceed forward from this point to avoid faults.
      assert(false && "Not supported semiring operator");
    }

    formSemiringLoopBody(comp_worksp_opt,
                         semiringParts.first, semiringParts.second,
                         rewriter, loc, lhs_loc,
                         main_tensors_all_Allocs,
                         tensors_lhs_Allocs,
                         tensors_rhs_Allocs,
                         allValueAccessIdx,
                         allAccessIdx,
                         nested_forops,
                         allPerms_rhs,
                         rhsFormats,
                         lhsFormats);
  }
  else
  {
    llvm::errs() << "No support for operation with greater than two operands in workspace transforms!"
                 << "\n";
  }
}

//===----------------------------------------------------------------------===//
// LowerIndexTreeIRToSCF PASS
//===----------------------------------------------------------------------===//

/// Lower the ta.tc (tensor contraction operation in TA dialect) into scf dialect.
namespace
{
  //===----------------------------------------------------------------------===//
  // LowerIndexTreeIRToSCF RewritePatterns: SparseTensor Constant operations
  //===----------------------------------------------------------------------===//

  struct IndexTreeIRLowering : public OpRewritePattern<indexTree::IndexTreeOp>
  {
    using OpRewritePattern<indexTree::IndexTreeOp>::OpRewritePattern;
    /**
     * @brief :
     * Goal: IndexTreeOp(i.e. a tree structure), convert into OpsTree(also tree structure)
     * Steps: 1.Iterate over IndexTreeOptree
     *        2.pass info to opsGen(), including tensors, current workspacetreeop, parent OpsTree node
     *          -- the parent of "current workspacetreeop" can get from getUser(). Only one user(tree structure)
     *          -- DFS traverse the workspacetreeop. How?
     * */
    LogicalResult matchAndRewrite(indexTree::IndexTreeOp rootOp,
                                  PatternRewriter &rewriter) const final
    {
      // auto module = rootOp->getParentOfType<ModuleOp>();

      assert(isa<indexTree::IndexTreeOp>(rootOp));
      comet_debug() << "\nIndexTreeIRLowering in LowerIndexTreeIRToSCF\n";
      comet_pdump(rootOp.getOperation()->getParentOp());
      // Here, should check the operands, at least one operand should be sparse;
      // Otherwise, if all dense operands, just return.
      // rootOp only contains one workspace child, no indices

      std::vector<mlir::Value> wp_ops;
      dfsRootOpTree(rootOp.getChildren(), wp_ops);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      comet_debug() << " wp_ops.size(): " << wp_ops.size() << "\n";
      for (auto n : wp_ops)
      {
        comet_debug() << " ";
        comet_vdump(n);
        // Declare opsTree
      }
#endif

      // In ops vector, for each op, the parent of each op can get from getUsers()
      // Since it's a tree structure, only one user ==> which is the parent
      // We can initialize the OpsTree structure with this relationship.
      // Search the location of the parent of current op, if rootOp, return ops.size;
      // Otherwise, return the location index.
      std::vector<unsigned int> parent_idx;
      for (unsigned int i = 0; i < wp_ops.size(); i++)
      {
        mlir::Value wp_op = wp_ops[i];
        mlir::Value wp_parent;

        for (auto n : wp_op.getDefiningOp()->getUsers())
        {
          comet_debug() << " " << i << " ";
          comet_pdump(n);
          wp_parent = n->getResult(0);

          comet_debug() << " parent: " << findIndexInVector_Value(wp_ops, wp_parent) << "\n";
          bool isInTree = false;
          if (findIndexInVector_Value(wp_ops, wp_parent) < wp_ops.size())
          {
            isInTree = true;
          }

          if (isInTree || isRealRoot(wp_op.getDefiningOp()))
            parent_idx.push_back(findIndexInVector_Value(wp_ops, wp_parent));
        }
      }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      comet_debug() << " parent_idx: " << parent_idx.size() << "\n";
      for (auto n : parent_idx)
      {
        comet_debug() << " " << n << " \n";
        // Declare opsTree
      }
#endif

      std::vector<OpsTree *> opstree_vec;
      for (unsigned int i = 0; i < wp_ops.size(); i++)
      {
        std::vector<scf::ForOp> forOps;
        std::vector<Value> accessIdx;
        std::vector<Value> cmptOps;

        OpsTree *parent = NULL;
        if (i >= 1)
        { // Not rootop
          parent = opstree_vec[parent_idx[i]];
        }
        comet_debug() << " \n";
        OpsTree *ops = new OpsTree(forOps, accessIdx, cmptOps, parent, i);
        if (parent != NULL)
        { // add child to the parent
          parent->addChild(ops);
        }

        opstree_vec.push_back(ops);
      }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      // Check the parent node
      for (auto n : opstree_vec)
      {
        comet_debug() << " " << n->id << " ";
        if (n->parent != NULL)
        {
          comet_debug() << "parent: " << n->parent->id << "\n";
        }
        else
          comet_debug() << "parent: null \n";
      }
#endif

      for (unsigned int i = 0; i < wp_ops.size(); i++)
      {
        comet_debug() << " i: " << i << "\n";
        comet_vdump(wp_ops[i]);
        if (indexTree::IndexTreeIndicesOp cur_op = dyn_cast<mlir::indexTree::IndexTreeIndicesOp>(wp_ops[i].getDefiningOp()))
        {
          // Get indices
          ArrayAttr op_indices = cur_op.getIndices();
          comet_debug() << "curOp is IndexTreeIndicesOp\n";
          comet_vdump(cur_op);

          std::vector<int> indices;
          for (unsigned int j = 0; j < op_indices.size(); j++)
          {
            // Get the indices;
            int idx = op_indices[j].cast<IntegerAttr>().getInt();
            indices.push_back(idx);
          }
          comet_debug() << " indices.size(): " << indices.size() << "\n";

          std::vector<Value> leafs;
          findLeafs(cur_op, indices, wp_ops, leafs);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
          comet_debug() << " leafs.size(): " << leafs.size() << "\n";
          for (auto n : leafs)
          {
            comet_debug() << " ";
            comet_vdump(n);
          }
#endif

          std::vector<Value> tensors;
          std::vector<unsigned int> ids;
          std::vector<std::string> formats;

          comet_debug() << " ";
          comet_vdump(cur_op);

          getFormatsInfo(cur_op, indices, leafs, tensors, ids, formats);

          comet_debug() << " indices.size(): " << indices.size() << " tensors.size(): " << tensors.size() << "\n";
          for (unsigned int m = 0; m < tensors.size(); m++)
          {
            comet_debug() << " Formats:" << formats[m] << " " << ids[m] << " ";
            comet_vdump(tensors[m]);
          }

          /// Generate loops
          std::vector<Value> ancestors_wp; // workspace tree ancestor
          getAncestorsWp(cur_op, ancestors_wp, wp_ops);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
          comet_debug() << " Current Op (IndexTreeIndicesOp): ";
          comet_vdump(cur_op);
          for (auto n : ancestors_wp)
          {
            comet_debug() << " ancestors_wp:";
            comet_vdump(n);
          }
#endif

          comet_debug() << " call genForOps, i = " << i << "\n";
          genForOps(tensors, ids, formats, rootOp, rewriter, opstree_vec[i], ancestors_wp);
          comet_debug() << " finished call genForOps, i = " << i << "\n";
        }
        else if (indexTree::IndexTreeComputeOp cur_op = dyn_cast<mlir::indexTree::IndexTreeComputeOp>(wp_ops[i].getDefiningOp()))
        {
          // Generate computation ops.
          std::vector<Value> ancestors_wp; // workspace tree ancestor
          getAncestorsWp(cur_op, ancestors_wp, wp_ops);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
          comet_debug() << " Current Op (IndexTreeComputeOp):";
          comet_vdump(cur_op);
          for (auto n : ancestors_wp)
          {
            comet_debug() << " ";
            comet_vdump(n);
          }
#endif

          comet_debug() << " call genCmptOps, i = " << i << "\n";
          // ancestors_wp can give all the indices of the nested loops
          genCmptOps(cur_op, rootOp, rewriter, opstree_vec[i], ancestors_wp);
          comet_debug() << " finished call genCmptOps, i = " << i << "\n";
        }
      }

      comet_debug() << "Cleaning up IndexTree Operations\n";
      comet_vdump(rootOp);
      rewriter.eraseOp(rootOp);
      for (auto itOp : wp_ops)
      {
        if (indexTree::IndexTreeComputeOp cur_op = dyn_cast<mlir::indexTree::IndexTreeComputeOp>(itOp.getDefiningOp()))
        {
          comet_pdump(itOp.getDefiningOp()->getOperand(0).getDefiningOp());
          comet_pdump(itOp.getDefiningOp()->getOperand(1).getDefiningOp());
          rewriter.eraseOp(itOp.getDefiningOp()->getOperand(0).getDefiningOp()); //RHS
          rewriter.eraseOp(itOp.getDefiningOp()->getOperand(1).getDefiningOp()); //LHS
        }
        comet_pdump(itOp.getDefiningOp());
        rewriter.eraseOp(itOp.getDefiningOp());
      }

      return success();
    }
  }; // IndexTreeIRLowering

  struct LowerIndexTreeToSCFPass
      : public PassWrapper<LowerIndexTreeToSCFPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerIndexTreeToSCFPass)
    void runOnOperation() override;
  };

} // end anonymous namespace.

void LowerIndexTreeToSCFPass::runOnOperation()
{
  comet_debug() << "LowerIndexTreeToSCFPass\n";
  func::FuncOp function = getOperation();
  auto module = function.getOperation()->getParentOfType<ModuleOp>();
  auto *ctx = &getContext();

  IndexType indexType = IndexType::get(ctx);
  auto quickSortFunc = FunctionType::get(ctx, {mlir::UnrankedMemRefType::get(indexType, 0), indexType}, {});

  if (!hasFuncDeclaration(module, "quick_sort"))
  {
    mlir::func::FuncOp func1 = mlir::func::FuncOp::create(function.getLoc(), "quick_sort",
                                                          quickSortFunc, ArrayRef<NamedAttribute>{});
    func1.setPrivate();
    module.push_back(func1);
  }

  ConversionTarget target(getContext());
  target.addLegalDialect<LinalgDialect,
                         ArithDialect,
                         scf::SCFDialect,
                         memref::MemRefDialect,
                         bufferization::BufferizationDialect>();

  target.addIllegalDialect<tensorAlgebra::TADialect>();
  target.addLegalOp<tensorAlgebra::PrintOp,
                    tensorAlgebra::TAReturnOp,
                    tensorAlgebra::ReduceOp,
                    tensorAlgebra::TransposeOp,
                    tensorAlgebra::TensorFillOp,
                    tensorAlgebra::GetTimeOp,
                    tensorAlgebra::PrintElapsedTimeOp,
                    tensorAlgebra::SparseTensorConstructOp,
                    tensorAlgebra::TensorSetOp,
                    tensorAlgebra::IndexLabelStaticOp,
                    tensorAlgebra::IndexLabelDynamicOp,
                    tensorAlgebra::LabeledTensorOp,
                    func::CallOp>();

  RewritePatternSet patterns(&getContext());
  patterns.insert<IndexTreeIRLowering>(&getContext());

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    llvm::errs() << "Failed to Lower LowerIndexTreeToSCFPass\n";
  }
}

// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createLowerIndexTreeToSCFPass()
{
  return std::make_unique<LowerIndexTreeToSCFPass>();
}
