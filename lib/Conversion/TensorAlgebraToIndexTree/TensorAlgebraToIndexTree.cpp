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

#include <cassert>

#include "comet/Conversion/TensorAlgebraToIndexTree/TensorAlgebraToIndexTree.h"

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/Transforms/UnitExpression.h"
#include "comet/Dialect/IndexTree/IR/IndexTree.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_LowerTensorAlgebraToIndexTreePass
// #define DEBUG_MODE_LowerTensorAlgebraToIndexTreePass
// #endif

#ifdef DEBUG_MODE_LowerTensorAlgebraToIndexTreePass
#define comet_debug() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n)                                \
  llvm::errs() << __FILE__ << " " << __LINE__ << " "; \
  n->dump()
#define comet_vdump(n)                                \
  llvm::errs() << __FILE__ << " " << __LINE__ << " "; \
  n.dump()
#else
#define comet_debug() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

using namespace mlir;

unique_ptr<Index_Tree> tree;
namespace
{

  struct LowerTensorAlgebraToIndexTreePass
      : public PassWrapper<LowerTensorAlgebraToIndexTreePass, OperationPass<func::FuncOp>>
  {
     MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorAlgebraToIndexTreePass)
    void runOnOperation() override;
  };
} // namespace

Value getRealLhs(Operation *op)
{
  assert(isa<TensorMultOp>(op) || isa<TensorElewsMultOp>(op) || isa<TensorAddOp>(op) || isa<TensorSubtractOp>(op));
  Operation *firstUser;
  for (auto user : op->getResult(0).getUsers())
  {
    firstUser = user;
    break;
  }

  comet_pdump(firstUser);
  assert(isa<TensorSetOp>(firstUser));
  TensorSetOp setOp = cast<TensorSetOp>(firstUser);
  return setOp.getOperand(1);
}

Value getRealRhs(Operation *op)
{
  Operation *firstUser = op->getNextNode(); // this will return set_op for transpose, but messes up getUsers() or subsequent calls to it.
  comet_pdump(firstUser);
  // TODO: need to find out why user set_op is not showing up in users of TransposeOp
  //       from the for loop below. once resolved, remove getNextNode().
  // Operation *firstUser;
  // for (auto user : op->getResult(0).getUsers())
  //{
  //  firstUser = user;
  //  break;
  //}

  if (isa<tensorAlgebra::TransposeOp>(op))
  {
    if (isa<TensorSetOp>(firstUser))
    {
      TensorSetOp setOp = cast<TensorSetOp>(firstUser);
      return setOp.getOperand(1);
    }
    else
    {
      assert(false && "Transpose has no set_op after it!");
    }
  }
  else
  {
    // do nothing
    return op->getResult(0);
  }
  return op->getResult(0);
}

void buildDefUseInfo(UnitExpression *e)
{
  auto lhs = e->getLHS();
  lhs->setDefiningExpr(e);
  for (auto operand : e->getOperands())
  {
    if (auto def = operand->getDefiningExpr())
    {
      def->addUser(e);
    }
  }
}

IndicesType getUnion(IndicesType indices1, IndicesType indices2)
{

  sort(indices1.begin(), indices1.end());
  sort(indices2.begin(), indices2.end());

  IndicesType allIndices(indices1.size() * 2);

  IndicesType::iterator it = set_union(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(), allIndices.begin());
  allIndices.resize(it - allIndices.begin());
  return allIndices;
}

void doTensorMultOp(TensorMultOp op)
{
  Value rhs1_tensor = getRealRhs(op.getRhs1().getDefiningOp());
  Value rhs2_tensor = getRealRhs(op.getRhs2().getDefiningOp());
  Value lhs_tensor = getRealLhs(op);

  comet_debug() << "LowerTensorAlgebraToIndexTreePass: doTensorMultOp\n";
  comet_debug() << "rhs1-tensor\n";
  comet_vdump(rhs1_tensor);
  comet_debug() << "rhs2-tensor\n";
  comet_vdump(rhs2_tensor);
  comet_debug() << "lhs-tensor\n";
  comet_vdump(lhs_tensor);

  auto allPerms = getAllPerms(op.getIndexingMaps());
  auto allFormats = getAllFormats(op.getFormatsAttr(), allPerms);
  auto SemiringOp = op.getSemiringAttr();

  assert(allPerms.size() == 3);

  auto B = tree->getOrCreateTensor(rhs1_tensor, allFormats[0]);
  auto C = tree->getOrCreateTensor(rhs2_tensor, allFormats[1]);
  auto A = tree->getOrCreateTensor(lhs_tensor, allFormats[2]);

  auto e = make_unique<UnitExpression>(A, B, C, "*");
  e->setSemiring(SemiringOp.cast<mlir::StringAttr>().getValue());

  e->setOperation(op);
  buildDefUseInfo(e.get());

  auto inputDomains = e->computeInputIterDomains();
  auto outputDomains = e->computeOutputIterDomains();

  IndicesType rhs1_indices = tree->getIndices(rhs1_tensor);
  IndicesType rhs2_indices = tree->getIndices(rhs2_tensor);
  IndicesType allIndices = getUnion(rhs1_indices, rhs2_indices);

  auto lhsIndices = A->getIndices();

  TreeNode *parent = tree->getRoot();
  for (unsigned long i = 0; i < allIndices.size(); i++)
  {
    int index = allIndices[i];
    auto &idomain = inputDomains.at(index);

    auto node = tree->addIndexNode(index, parent, idomain);

    // If this index appears on the lhs too, set output domain for the index node
    if (std::find(lhsIndices.begin(), lhsIndices.end(), index) != lhsIndices.end())
    {
      auto &odomain = outputDomains.at(index);
      node->setOutputDomain(odomain);
    }

    parent = node;
  }

  tree->addComputeNode(std::move(e), parent);
}

template <typename T>
void doElementWiseOp(T op)
{
  Value rhs1_tensor = getRealRhs(op.getRhs1().getDefiningOp());
  Value rhs2_tensor = getRealRhs(op.getRhs2().getDefiningOp());
  Value lhs_tensor = getRealLhs(op);

  comet_debug() << "LowerTensorAlgebraToIndexTreePass: doElementWiseMultOp\n";
  comet_debug() << "rhs1-tensor\n";
  comet_vdump(rhs1_tensor);
  comet_debug() << "rhs2-tensor\n";
  comet_vdump(rhs2_tensor);
  comet_debug() << "lhs-tensor\n";
  comet_vdump(lhs_tensor);

  auto allPerms = getAllPerms(op.getIndexingMaps());
  auto allFormats = getAllFormats(op.getFormatsAttr(), allPerms);
  auto SemiringOp = op.getSemiringAttr();

  assert(allPerms.size() == 3);

  auto B = tree->getOrCreateTensor(rhs1_tensor, allFormats[0]);
  auto C = tree->getOrCreateTensor(rhs2_tensor, allFormats[1]);
  auto A = tree->getOrCreateTensor(lhs_tensor, allFormats[2]);

  auto e = make_unique<UnitExpression>(A, B, C, "*");

  e->setOperation(op);
  e->setSemiring(SemiringOp.template cast<mlir::StringAttr>().getValue()); // for element-wise multiplication
  buildDefUseInfo(e.get());

  auto inputDomains = e->computeInputIterDomains();
  auto outputDomains = e->computeOutputIterDomains();

  // RHS and LHS indices must be the same for elementwise multiplication
  IndicesType allIndices = tree->getIndices(rhs1_tensor);

  auto lhsIndices = A->getIndices();
  TreeNode *parent = tree->getRoot();
  for (unsigned long i = 0; i < allIndices.size(); i++)
  {
    int index = allIndices[i];
    auto &idomain = inputDomains.at(index);

    auto node = tree->addIndexNode(index, parent, idomain);

    // If this index appears on the lhs too, set output domain for the index node
    if (std::find(lhsIndices.begin(), lhsIndices.end(), index) != lhsIndices.end())
    {
      auto &odomain = outputDomains.at(index);
      node->setOutputDomain(odomain);
    }

    parent = node;
  }
  tree->addComputeNode(std::move(e), parent);
  // cout << "print tree after tc\n";
  // tree->print();
}

// helper for treeToDialect()
Operation *getSetOpForTC(Operation *op)
{
  assert(isa<TensorMultOp>(op) || isa<TensorElewsMultOp>(op) || isa<TensorAddOp>(op) || isa<TensorSubtractOp>(op));
  // TODO: fix the issue with getUsers() after getRealRhs().
  comet_debug() << "The following loop may cause issue!\n";
  Operation *firstUser;
  for (auto user : op->getResult(0).getUsers())
  {
    firstUser = user;
    break;
  }

  assert(isa<TensorSetOp>(firstUser));
  return firstUser;
}

// helper for treeToDialect()
IndexTreeComputeOp createComputeNodeOp(OpBuilder &builder, TreeNode *node, Location &loc)
{
  auto context = builder.getContext();
  IntegerType i64Type = IntegerType::get(context, 64);
  auto expr = node->getExpression();

  SmallVector<Attribute, 8> allIndices_rhs;
  for (auto t : expr->getOperands())
  {
    SmallVector<int64_t, 8> indices;
    for (auto index : t->getIndices())
    {
      indices.push_back(index);
    }
    allIndices_rhs.push_back(builder.getI64ArrayAttr(indices));
  }
  SmallVector<Attribute, 8> allIndices_lhs;
  for (auto t : expr->getResults())
  {
    SmallVector<int64_t, 8> indices;
    for (auto index : t->getIndices())
    {
      indices.push_back(index);
    }
    allIndices_lhs.push_back(builder.getI64ArrayAttr(indices));
  }

  SmallVector<Attribute, 8> allFormats_rhs;
  for (auto t : expr->getOperands())
  {
    SmallVector<StringRef, 8> formats;
    for (auto &f : t->getFormats())
    {
      formats.push_back(f);
    }
    allFormats_rhs.push_back(builder.getStrArrayAttr(formats));
  }
  SmallVector<Attribute, 8> allFormats_lhs;
  for (auto t : expr->getResults())
  {
    SmallVector<StringRef, 8> formats;
    for (auto &f : t->getFormats())
    {
      formats.push_back(f);
    }
    allFormats_lhs.push_back(builder.getStrArrayAttr(formats));
  }

  std::vector<Value> t_rhs;
  Value t_lhs = expr->getLHS()->getValue();
  for (auto o : expr->getOperands())
  {
    t_rhs.push_back(o->getValue());
  }

  Value leafop_rhs = builder.create<indexTree::IndexTreeComputeRHSOp>(loc,
                                                                      mlir::UnrankedTensorType::get(builder.getF64Type()), t_rhs,
                                                                      builder.getArrayAttr(allIndices_rhs),
                                                                      builder.getArrayAttr(allFormats_rhs));
  Value leafop_lhs = builder.create<indexTree::IndexTreeComputeLHSOp>(loc,
                                                                      mlir::UnrankedTensorType::get(builder.getF64Type()), t_lhs,
                                                                      builder.getArrayAttr(allIndices_lhs),
                                                                      builder.getArrayAttr(allFormats_lhs));
  bool comp_worksp_opt = false; // non-compressed workspace, this is a place-holder and it is updated in workspace transform pass.
  llvm::StringRef semiring = expr->getSemiring();
  auto leafop = builder.create<IndexTreeComputeOp>(loc, i64Type, leafop_rhs, leafop_lhs, builder.getBoolAttr(comp_worksp_opt), builder.getStringAttr(semiring));

  comet_pdump(leafop);
  return leafop;
}

/**
 * This function performs the actual removal of the ta operations in the tree,
 * and add corresponding ta.itree operations.›
 * @param tree
 */
void treeToDialect(Index_Tree *tree)
{
  vector<mlir::Operation *> TAOps = tree->getContainingTAOps();
  unsigned int TAOpsID = 0;
  OpBuilder builder(TAOps[TAOpsID]);
  auto loc = TAOps[TAOpsID]->getLoc();
  auto context = builder.getContext();

  std::map<TreeNode *, Value> nodeToOp;

  IntegerType i64Type = IntegerType::get(context, 64);

  for (auto &node : tree->getNodesInReverseTopoOrder())
  {
    if (node->isComputeNode())
    {
      assert(nodeToOp.count(node) == 0);
      builder.setInsertionPoint(TAOps[TAOpsID]);
      nodeToOp[node] = createComputeNodeOp(builder, node, loc);
      TAOpsID++;
    }
    else if (node->isRealIndexNode())
    {
      if (node->getChildren().empty())
      {
        continue; // to skip nodes that become childless after fusion
      }
      SmallVector<Value, 8> children;
      for (auto c : node->getChildren())
      {
        assert(nodeToOp.count(c) > 0);
        children.push_back(nodeToOp[c]);
      }
      // assert(!children.empty());
      SmallVector<int64_t, 1> indices;
      indices.push_back(node->getIndex());
      auto indicesAttr = builder.getI64ArrayAttr(indices);

      SmallVector<int64_t, 1> ids;
      ids.push_back(node->getId());

      Value indexNodeOp = builder.create<IndexTreeIndicesOp>(loc,
                                                             i64Type,
                                                             children,
                                                             indicesAttr);

      nodeToOp[node] = indexNodeOp;

      if (node->getParent() != nullptr && node->getParent()->isFillerIndexNode())
      {
#ifdef DEBUG_MODE_LowerTensorAlgebraToIndexTreePass
        Value op = builder.create<indexTree::IndexTreeOp>(loc, i64Type, indexNodeOp);
        comet_vdump(op);
#else
        builder.create<indexTree::IndexTreeOp>(loc, i64Type, indexNodeOp);
#endif
      }
    }
  }

  for (auto op : tree->getContainingTAOps())
  {
    auto setOp = getSetOpForTC(op);
    setOp->erase();
    op->erase();
  }
}

void LowerTensorAlgebraToIndexTreePass::runOnOperation()
{
  assert(tree == nullptr);
  func::FuncOp func = getOperation();

  tree = Index_Tree::createTreeWithRoot();
  bool formIndexTreeDialect = false;

  comet_debug() << "IndexTree pass running on Function\n";
  for (Block &B : func.getBody())
  {
    for (Operation &op : B)
    {
      if (isa<TensorMultOp>(&op))
      {
        doTensorMultOp(cast<TensorMultOp>(&op));
        formIndexTreeDialect = true;
      }
      else if (isa<TensorElewsMultOp>(&op))
      {
        doElementWiseOp<TensorElewsMultOp>(cast<TensorElewsMultOp>(&op));
        formIndexTreeDialect = true;
      }
      else if (isa<TensorAddOp>(&op) || isa<TensorSubtractOp>(&op))
      {
        // elementwise addition and subtraction
        if (isa<TensorAddOp>(&op))
        {
          doElementWiseOp<TensorAddOp>(cast<TensorAddOp>(&op));
        }

        if (isa<TensorSubtractOp>(&op))
        {
          doElementWiseOp<TensorSubtractOp>(cast<TensorSubtractOp>(&op));
        }
        formIndexTreeDialect = true;
      }
    }
  }

  if (formIndexTreeDialect)
  {
    comet_debug() << " Dumping Index tree IR\n";
    // only do this for TensorMultOp or TensorElewsMultOp
    treeToDialect(tree.get());
  }
}

// create all the passes.
//
std::unique_ptr<Pass> mlir::comet::createLowerTensorAlgebraToIndexTreePass()
{
  comet_debug() << " Calling createLowerTensorAlgebraToIndexTreePass\n";
  return std::make_unique<LowerTensorAlgebraToIndexTreePass>();
}
