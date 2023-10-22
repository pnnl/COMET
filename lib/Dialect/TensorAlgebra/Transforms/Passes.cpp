//===- Passes.cpp ---------------------------===//
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

// TODO(gkestor): check these header files
#include "mlir/Pass/Pass.h"

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <algorithm>
#include <map>
#include <set>
#include <stack>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#define DEBUG_TYPE "comet-passes"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace tensorAlgebra;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_PASSES
// #define DEBUG_MODE_PASSES
// #endif

#ifdef DEBUG_MODE_PASSES
#define comet_debug() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n)                                \
  llvm::errs() << __FILE__ << " " << __LINE__ << " "; \
  n->dump()
#define comet_vdump(n)                                \
  llvm::errs() << __FILE__ << " " << __LINE__ << " "; \
  n.dump()
#else
#define comet_debug() if(true){}else llvm::errs()
#define comet_pdump(n)
#define comet_vdump(n)
#endif


namespace
{
  class FindOptimalTCFactorizationPass
      : public mlir::PassWrapper<FindOptimalTCFactorizationPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FindOptimalTCFactorizationPass)
    void runOnOperation() override;

    void FindOptimalTCFactorization(tensorAlgebra::TensorSetOp op);
  }; // class FindOptimalTCFactorizationPass
} // End anonymous namespace

namespace
{

  struct LowerTAMulChainPass
      : public PassWrapper<LowerTAMulChainPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTAMulChainPass)
    void runOnOperation() override;
  };

  struct OptDenseTransposePass
      : public PassWrapper<OptDenseTransposePass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptDenseTransposePass)
    void runOnOperation() override;
  };

  struct STCRemoveDeadOpsPass
      : public PassWrapper<STCRemoveDeadOpsPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(STCRemoveDeadOpsPass)
    void runOnOperation() override;
  };

} // end anonymous namespace.

void removeAllUsers(Operation *op)
{
  for (auto u : op->getUsers())
  {
    comet_debug() << "Users\n";
    comet_pdump(u);
    removeAllUsers(u);
  }
  comet_debug() << "Deleting started\n";
  comet_pdump(op);
  op->erase();
  comet_debug() << "Deleting ends\n";
}

std::vector<int64_t> getTensorShape(const std::vector<Operation *> &labels,
                                    const std::map<Operation *, int64_t> &lblSizes)
{
  std::vector<int64_t> result;

  for (auto lbl : labels)
  {
    result.push_back(lblSizes.at(lbl));
  }

  return result;
}

IndexVector getLabelPerm(const std::vector<Operation *> &labels,
                         const std::map<Operation *, unsigned> &labelIdMap)
{
  IndexVector result;
  for (auto lbl : labels)
  {
    result.push_back(labelIdMap.at(lbl));
  }
  return result;
}

std::vector<Operation *> findOutput(const std::vector<Operation *> &rhs1Labels,
                                    const std::vector<Operation *> &rhs2Labels,
                                    const std::set<Operation *> &outLabels)
{
  std::vector<Operation *> result_labels;

  std::set<Operation *> inLabels(rhs1Labels.begin(), rhs1Labels.end());
  inLabels.insert(rhs2Labels.begin(), rhs2Labels.end());

  std::set_intersection(inLabels.begin(), inLabels.end(), outLabels.begin(),
                        outLabels.end(),
                        std::back_inserter(result_labels));

  return result_labels;
}

std::tuple<IndexVector, std::vector<std::vector<Operation *>>, std::vector<std::vector<int64_t>>>
optimalOrder(ArrayRef<Operation *> inLTOps, Operation *outLTOp,
             const std::map<Operation *, int64_t> &lblSizes,
             const std::map<Operation *, std::vector<Operation *>> &lblMaps)
{
  IndexVector result;
  for (size_t i = 0; i < inLTOps.size(); i++)
  {
    result.push_back(i);
  }

  double minCost = std::numeric_limits<double>::max();
  std::vector<unsigned> minResult;
  std::vector<std::vector<Operation *>> minSumLabels;
  std::vector<std::vector<int64_t>> minLHSTensorShapes;
  double totalCost = 0;
  std::map<Operation *, unsigned> labelIdMap;

  unsigned id = 0;
  for (auto op_size_pair : lblSizes)
  {
    auto op = op_size_pair.first;
    labelIdMap[op] = id++;
  }

  // go through each and every permutation of result vector.
  do
  {
    totalCost = 0;

    std::vector<std::vector<Operation *>> sumLabels;
    std::vector<std::vector<int64_t>> lhsTensorShapes;
    std::vector<Operation *> rhs1Labels = lblMaps.at(inLTOps[result[0]]);
    for (size_t i = 1; i < result.size(); i++)
    {
      std::vector<Operation *> rhs2Labels = lblMaps.at(inLTOps[result[i]]);

      std::set<Operation *> remainingLabels(lblMaps.at(outLTOp).begin(),
                                            lblMaps.at(outLTOp).end());

      for (size_t j = i + 1; j < result.size(); j++)
      {
        auto lblSet = lblMaps.at(inLTOps[result[j]]);
        remainingLabels.insert(lblSet.begin(), lblSet.end());
      }

      // find intersection of {rhs1Labels, rhs2Labels}, {remainingLabels}
      auto lhsLabels = findOutput(rhs1Labels, rhs2Labels, remainingLabels);
      // find difference of {rhs1Labels, rhs2Labels}, {lhsLabels}
      sumLabels.push_back(getSumLabels(rhs1Labels, rhs2Labels, lhsLabels));
      auto permA = getLabelPerm(rhs1Labels, labelIdMap);
      auto permB = getLabelPerm(rhs2Labels, labelIdMap);
      auto permC = getLabelPerm(lhsLabels, labelIdMap);

      auto tensorShapeA = getTensorShape(rhs1Labels, lblSizes);
      auto tensorShapeB = getTensorShape(rhs2Labels, lblSizes);
      auto tensorShapeC = getTensorShape(lhsLabels, lblSizes);
      lhsTensorShapes.push_back(tensorShapeC);

      ContractionPlan plan{permA, tensorShapeA,
                           permB, tensorShapeB,
                           permC, tensorShapeC};
      // make getTotal optional to include only the operation count or
      // plus the cost  operation transpose
      totalCost += plan.getTotalTime();

      rhs1Labels = lhsLabels;
    }

    // update global
    if (totalCost <= minCost)
    {
      minCost = totalCost;
      minResult = result;
      minSumLabels = sumLabels;
      minLHSTensorShapes = lhsTensorShapes;
    }

  } while (std::next_permutation(result.begin(), result.end()));

  return std::make_tuple(minResult, minSumLabels, minLHSTensorShapes);
}

void FindOptimalTCFactorizationPass::FindOptimalTCFactorization(tensorAlgebra::TensorSetOp op)
{
  OpBuilder builder(op);
  comet_pdump(op);
  auto operands = op->getOperands();
  // auto module = op->getParentOfType<ModuleOp>();
  auto loc = op->getLoc();
  auto lhsOp = operands[0].getDefiningOp(); // TensorMultOp
  auto rhsOp = operands[1].getDefiningOp();

  std::vector<Operation *> MultOpsToRemove;
  std::vector<Operation *> LTOpsToRemove;

  std::vector<Operation *> inLTOps;
  std::map<Operation *, Value> inLTValues;

  comet_debug() << "Chain Multiplication Factorization begin...\n";

  // collect all operands from series of ta.tc ops
  if (isa<tensorAlgebra::TensorMultOp>(lhsOp))
  {
    std::stack<Operation *> stack;
    Value currValue = operands[0];
    comet_vdump(currValue);
    Operation *curr = currValue.getDefiningOp();
    while (isa<tensorAlgebra::TensorMultOp>(curr) || !stack.empty())
    {
      while (isa<tensorAlgebra::TensorMultOp>(curr))
      {
        stack.push(curr);
        MultOpsToRemove.push_back(cast<tensorAlgebra::TensorMultOp>(curr).getOperation());
        currValue = cast<tensorAlgebra::TensorMultOp>(curr).getOperation()->getOperand(1);
        curr = currValue.getDefiningOp();
      }

      inLTOps.push_back(curr);
      inLTValues[curr] = currValue;

      curr = stack.top();
      stack.pop();
      currValue = cast<tensorAlgebra::TensorMultOp>(curr).getOperation()->getOperand(0);
      curr = currValue.getDefiningOp();
    }
    inLTOps.push_back(curr);
    inLTValues[curr] = currValue;
  }

  std::map<Operation *, Value> labelValues;
  // IndexLabelStaticOp to size map
  std::map<Operation *, int64_t> lblSizes;
  // LabeledTensorOp to label set map
  std::map<Operation *, std::vector<Operation *>> lblMaps;

  for (auto op : inLTOps)
  {
    auto labels = cast<tensorAlgebra::DenseTensorDeclOp>(op).getLabels();
    std::vector<Operation *> labelVec;
    for (auto lbl : labels)
    {
      auto lblOp = lbl.getDefiningOp();
      if (lblSizes.count(lblOp) == 0)
      {
        lblSizes[lblOp] = mlir::tensorAlgebra::labelSize(lblOp);
        labelValues[lblOp] = lbl;
      }
      labelVec.push_back(lblOp);
    }
    lblMaps[op] = labelVec;
  }

  auto outLabels = cast<tensorAlgebra::LabeledTensorOp>(rhsOp).getLabels();
  LTOpsToRemove.push_back(cast<tensorAlgebra::LabeledTensorOp>(rhsOp).getOperation());
  std::vector<Operation *> outLabelVec;
  for (auto lbl : outLabels)
  {
    auto lblOp = lbl.getDefiningOp();
    // comet_debug() << " outlabels: " << lbl << "\n";
    outLabelVec.push_back(lblOp);
  }
  lblMaps[lhsOp] = outLabelVec;

  IndexVector order;
  std::vector<std::vector<Operation *>> sumLabels;
  std::vector<std::vector<int64_t>> lhsTensorShapes;
  std::tie(order, sumLabels, lhsTensorShapes) = optimalOrder(inLTOps, lhsOp, lblSizes, lblMaps);

  bool same_order = hasSameOrder(getIdentityPermutation(order.size()), order);

  // updated ta dialect generation.
  if (!same_order)
  {

    Value newRhs1, newRhs2;
    std::vector<Value> ___newSumLabels; // needed for intermediate tensor information

    newRhs1 = inLTValues[inLTOps[order[0]]]; // updated later for subsequent ta.tc ops in the chain
    for (size_t i = 1; i < order.size(); i++)
    {
      newRhs2 = inLTValues[inLTOps[order[i]]];
      auto elType = newRhs1.getType().dyn_cast<RankedTensorType>().getElementType();
      auto newType = RankedTensorType::get(lhsTensorShapes[i - 1], elType);
      std::vector<Value> newSumLabels;

      std::vector<Operation *> rhs1Labels = lblMaps.at(inLTOps[order[i - 1]]);
      std::vector<Operation *> rhs2Labels = lblMaps.at(inLTOps[order[i]]);
      std::set<Operation *> remainingLabels(lblMaps.at(lhsOp).begin(),
                                            lblMaps.at(lhsOp).end());

      for (size_t j = i + 1; j < order.size(); j++)
      {
        auto lblSet = lblMaps.at(inLTOps[order[j]]);
        remainingLabels.insert(lblSet.begin(), lblSet.end());
      }
      auto lhsLabels = findOutput(rhs1Labels, rhs2Labels, remainingLabels);
      // find difference of {rhs1Labels, rhs2Labels}, {lhsLabels}
      auto tempLabelsOps = getSumLabels(rhs1Labels, rhs2Labels, lhsLabels);

      for (auto lbl : lhsLabels)
      {
        newSumLabels.push_back(labelValues[lbl]);
        // comet_vdump(labelValues[lbl]);
      }
      if (!isa<tensorAlgebra::TensorMultOp>(newRhs1.getDefiningOp()))
      { // store the output label values for subsequent ta.tc ops
        ___newSumLabels = newSumLabels;
      }

      std::vector<Value> new_all_lbls_value;
      std::vector<Value> new_lhs_lbls_value;
      std::vector<Value> new_rhs_lbls_value;
      for (auto lbl : rhs2Labels)
      {
        new_lhs_lbls_value.push_back(labelValues[lbl]);
        new_all_lbls_value.push_back(labelValues[lbl]);
        // comet_vdump(labelValues[lbl]);
      }
      if (isa<tensorAlgebra::TensorMultOp>(newRhs1.getDefiningOp()))
      { // retrieve the labels from prev iteration.
        new_rhs_lbls_value = ___newSumLabels;
        for (auto lbl : new_rhs_lbls_value)
        {
          auto result1 = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), lbl);
          if (result1 == new_all_lbls_value.end())
          {
            new_all_lbls_value.push_back(lbl);
          }
          // comet_vdump(lbl);
        }
      }
      else
      { // retrieve the labels from lblMaps
        for (auto lbl : rhs1Labels)
        {
          new_rhs_lbls_value.push_back(labelValues[lbl]);
          auto result1 = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), labelValues[lbl]);
          if (result1 == new_all_lbls_value.end())
          {
            new_all_lbls_value.push_back(labelValues[lbl]);
          }
          // comet_vdump(labelValues[lbl]);
        }
      }

      // formats
      SmallVector<mlir::StringRef, 8> formats;
      if (isa<DenseTensorDeclOp>(newRhs2.getDefiningOp()))
      {
        auto lhs_format = dyn_cast<DenseTensorDeclOp>(newRhs2.getDefiningOp()).getFormat();
        formats.push_back(lhs_format);
      }
      if (isa<DenseTensorDeclOp>(newRhs1.getDefiningOp()))
      {
        auto rhs_format = dyn_cast<DenseTensorDeclOp>(newRhs1.getDefiningOp()).getFormat();
        formats.push_back(rhs_format);
      }
      if (isa<DenseTensorDeclOp>(newRhs1.getDefiningOp()) &&
          isa<DenseTensorDeclOp>(newRhs2.getDefiningOp()))
      {
        auto rhs_format = dyn_cast<DenseTensorDeclOp>(newRhs1.getDefiningOp()).getFormat();
        formats.push_back(rhs_format);
      }
      if (isa<tensorAlgebra::TensorMultOp>(newRhs1.getDefiningOp()))
      { // for series of ta.tc case
        auto lhs_format = dyn_cast<DenseTensorDeclOp>(newRhs2.getDefiningOp()).getFormat();
        formats.push_back(lhs_format);
        formats.push_back(lhs_format);
      }
      auto strAttr = builder.getStrArrayAttr(formats);

      std::vector<int> lhs_lbls;
      std::vector<int> rhs_lbls;
      for (unsigned int i = 0; i < new_all_lbls_value.size(); i++)
      {
        auto result1 = std::find(new_lhs_lbls_value.begin(), new_lhs_lbls_value.end(), new_all_lbls_value[i]);
        if (result1 != new_lhs_lbls_value.end())
        {
          lhs_lbls.push_back(i);
        }

        auto result2 = std::find(new_rhs_lbls_value.begin(), new_rhs_lbls_value.end(), new_all_lbls_value[i]);
        if (result2 != new_rhs_lbls_value.end())
        {
          rhs_lbls.push_back(i);
        }
      }

      std::vector<int> sum_lbls;
      std::set_intersection(lhs_lbls.begin(), lhs_lbls.end(), rhs_lbls.begin(), rhs_lbls.end(), std::back_inserter(sum_lbls));
      std::vector<int> all_lbls;
      std::set_union(lhs_lbls.begin(), lhs_lbls.end(), rhs_lbls.begin(), rhs_lbls.end(), std::back_inserter(all_lbls));
      std::vector<int> ret_lbls;
      std::set_difference(all_lbls.begin(), all_lbls.end(), sum_lbls.begin(), sum_lbls.end(), std::back_inserter(ret_lbls));

      std::map<int, mlir::AffineExpr> expr_map;
      unsigned dim = 0;
      for (const auto &lbl : all_lbls)
      {
        expr_map[lbl] = getAffineDimExpr(dim++, builder.getContext());
      }

      std::vector<mlir::AffineExpr> lhs_exprs;
      std::vector<mlir::AffineExpr> rhs_exprs;
      std::vector<mlir::AffineExpr> ret_exprs;
      for (const auto &lbl : lhs_lbls)
      {
        lhs_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : rhs_lbls)
      {
        rhs_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : ret_lbls)
      {
        ret_exprs.push_back(expr_map[lbl]);
      }

      auto context = builder.getContext();
      SmallVector<mlir::AffineMap, 8> affine_maps{
          mlir::AffineMap::get(dim, 0, rhs_exprs, context),
          mlir::AffineMap::get(dim, 0, lhs_exprs, context),
          mlir::AffineMap::get(dim, 0, ret_exprs, context)};
      auto affineMapArrayAttr = builder.getAffineMapArrayAttr(affine_maps);

      auto SemiringAttr = builder.getStringAttr("plusxy_times");
      auto MaskingAttr = builder.getStringAttr("none");
      Value tcop = builder.create<tensorAlgebra::TensorMultOp>(loc, newType, newRhs1, newRhs2,
                                                               newSumLabels, affineMapArrayAttr, strAttr, SemiringAttr,
                                                               MaskingAttr, nullptr);  //TODO: masking is an optional operand
      tcop.getDefiningOp()->setAttr("__alpha__", builder.getF64FloatAttr(1.0));
      tcop.getDefiningOp()->setAttr("__beta__", builder.getF64FloatAttr(0.0));

      newRhs1 = tcop;
    }

    mlir::tensorAlgebra::TensorSetOp newSetOp = builder.create<tensorAlgebra::TensorSetOp>(loc, newRhs1, operands[1].getDefiningOp()->getOperand(0));
    newSetOp->setAttr("__beta__", builder.getF64FloatAttr(0.0));

    comet_debug() << "are they previous multop\n";
    for (auto oldTcOp : MultOpsToRemove)
    {
      comet_debug() << "Calling removeAllUsers\n";
      removeAllUsers(oldTcOp);
    }
  }
  comet_debug() << "MulOpFactorization end\n";
  return;
}

void FindOptimalTCFactorizationPass::runOnOperation()
{
  comet_debug() << " start FindOptimalTCFactorizationPass pass \n";
  func::FuncOp func = getOperation();

  func.walk([&](tensorAlgebra::TensorSetOp op)
            { FindOptimalTCFactorization(op); });
}

void LowerTAMulChainPass::runOnOperation()
{
  func::FuncOp function = getOperation();
  RewritePatternSet patterns(&getContext());
  populateLowerTAMulChainPatterns(patterns, &getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<ArithDialect>();

  // target.addIllegalDialect<tensorAlgebra::TADialect>();
  target.addLegalOp<tensorAlgebra::PrintOp,
                    tensorAlgebra::TAReturnOp,
                    tensorAlgebra::ReduceOp,
                    tensorAlgebra::TransposeOp,
                    tensorAlgebra::TensorFillOp,
                    tensorAlgebra::TensorFillFromFileOp,
                    tensorAlgebra::GetTimeOp,
                    tensorAlgebra::PrintElapsedTimeOp,
                    tensorAlgebra::TensorMultOp,
                    tensorAlgebra::TensorElewsMultOp,
                    tensorAlgebra::TensorSetOp,
                    tensorAlgebra::ChainMulOp,
                    tensorAlgebra::TensorCopyOp,
                    tensorAlgebra::IndexLabelDynamicOp,
                    tensorAlgebra::IndexLabelStaticOp,
                    tensorAlgebra::SparseTensorDeclOp,
                    tensorAlgebra::DenseTensorDeclOp,
                    tensorAlgebra::SparseTensorConstructOp>();

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    llvm::errs() << "Failed to applyPartialConversion in LowerTAMulChainPass\n";
    signalPassFailure();
  }
}

void STCRemoveDeadOpsPass::runOnOperation()
{
  comet_debug() << " start STCRemoveDeadOpsPass \n";
  ConversionTarget target(getContext());

  func::FuncOp func = getOperation();
  target.addLegalDialect<mlir::linalg::LinalgDialect, 
                          ArithDialect, 
                          scf::SCFDialect, 
                          AffineDialect, memref::MemRefDialect, 
                          bufferization::BufferizationDialect>();
  
  target.addLegalOp<tensorAlgebra::TensorMultOp>();
  RewritePatternSet patterns(&getContext());
  populateSTCRemoveDeadOpsPatterns(patterns, &getContext());
  if (failed(applyPartialConversion(func, target, std::move(patterns))))
  {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::comet::createFindOptimalTCFactorizationPass()
{
  return std::make_unique<FindOptimalTCFactorizationPass>();
}

std::unique_ptr<Pass> mlir::comet::createLowerTAMulChainPass()
{
  return std::make_unique<LowerTAMulChainPass>();
}

// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createSTCRemoveDeadOpsPass()
{
  return std::make_unique<STCRemoveDeadOpsPass>();
}
