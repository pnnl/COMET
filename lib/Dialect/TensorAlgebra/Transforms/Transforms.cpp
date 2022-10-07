//===- Transforms.cpp - Tensor Algebra High Level Optimizer --------------------------===//
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
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the TA dialect.
//
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/IndexTree/IR/ITDialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"

#include <limits>
#include <map>
#include <stack>
#include <set>
#include <unordered_map>
#include <numeric>

#define DEBUG_TYPE "comet-transforms"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using namespace mlir::tensorAlgebra;
using namespace mlir::indexTree;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_TRANSFORMS
// #define DEBUG_MODE_TRANSFORMS
// #endif

#ifdef DEBUG_MODE_TRANSFORMS
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

std::vector<Value> dim_format;

namespace
{
  template <typename TAOp>
  struct RemoveTAOpLowering : public ConversionPattern
  {
    RemoveTAOpLowering(MLIRContext *ctx)
        : ConversionPattern(TAOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      rewriter.eraseOp(op);

      return success();
    }
  };

  struct MulOpFactorization : public ConversionPattern
  {
    MulOpFactorization(MLIRContext *ctx)
          : ConversionPattern(tensorAlgebra::TensorSetOp::getOperationName(), 2,
                            ctx) {}
    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      comet_pdump(op);
      auto loc = op->getLoc();
      auto lhsOp = operands[0].getDefiningOp();  // TensorMultOp
      auto rhsOp = operands[1].getDefiningOp();

      std::set<Operation *> MultOpsToRemove; 
      std::set<Operation *> LTOpsToRemove;

      std::vector<Operation *> inLTOps;
      std::map<Operation *, Value> inLTValues;

      comet_debug() << "MulOpFactorization begin...\n";
      
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
            MultOpsToRemove.insert(cast<tensorAlgebra::TensorMultOp>(curr).getOperation());
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
        auto labels = cast<tensorAlgebra::DenseTensorDeclOp>(op).labels();
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

      auto outLabels = cast<tensorAlgebra::LabeledTensorOp>(rhsOp).labels();
      LTOpsToRemove.insert(cast<tensorAlgebra::LabeledTensorOp>(rhsOp).getOperation());
      std::vector<Operation *> outLabelVec;
      for (auto lbl : outLabels)
      {
        auto lblOp = lbl.getDefiningOp();
        //comet_debug() << " outlabels: " << lbl << "\n";
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

        newRhs1 = inLTValues[inLTOps[order[0]]];  // updated later for subsequent ta.tc ops in the chain
        for (size_t i = 1; i < order.size(); i++)
        {
          newRhs2 = inLTValues[inLTOps[order[i]]];
          auto elType = newRhs1.getType().dyn_cast<RankedTensorType>().getElementType();
          auto newType = RankedTensorType::get(lhsTensorShapes[i - 1], elType);
          std::vector<Value> newSumLabels;

          std::vector<Operation *> rhs1Labels = lblMaps.at(inLTOps[order[i-1]]);
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
            //comet_vdump(labelValues[lbl]);
          }
          if (! isa<tensorAlgebra::TensorMultOp>(newRhs1.getDefiningOp()) ) { // store the output label values for subsequent ta.tc ops
            ___newSumLabels = newSumLabels;
          } 

          std::vector<Value> new_all_lbls_value;
          std::vector<Value> new_lhs_lbls_value;
          std::vector<Value> new_rhs_lbls_value;
          for (auto lbl : rhs2Labels) 
          {
            new_lhs_lbls_value.push_back(labelValues[lbl]);
            new_all_lbls_value.push_back(labelValues[lbl]);
            //comet_vdump(labelValues[lbl]);
          }
          if ( isa<tensorAlgebra::TensorMultOp>(newRhs1.getDefiningOp()) ) { // retrieve the labels from prev iteration. 
            new_rhs_lbls_value = ___newSumLabels;
            for (auto lbl : new_rhs_lbls_value) {
              auto result1 = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), lbl);
              if (result1 == new_all_lbls_value.end())
              {
                new_all_lbls_value.push_back(lbl);
              }
              //comet_vdump(lbl);
            }
          } else {  // retrieve the labels from lblMaps
              for (auto lbl : rhs1Labels) {
                new_rhs_lbls_value.push_back(labelValues[lbl]);
                auto result1 = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), labelValues[lbl]);
                if (result1 == new_all_lbls_value.end())
                {
                  new_all_lbls_value.push_back(labelValues[lbl]);
                }
                //comet_vdump(labelValues[lbl]);
              }
          }

          // formats
          SmallVector<mlir::StringRef, 8> formats;
          if (isa<DenseTensorDeclOp>(newRhs2.getDefiningOp())) {
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(newRhs2.getDefiningOp()).format();
            //auto lhs_lbls = dyn_cast<DenseTensorDeclOp>(newRhs2.getDefiningOp()).labels();
            formats.push_back(lhs_format);
          }
          if (isa<DenseTensorDeclOp>(newRhs1.getDefiningOp()) ) {
            auto rhs_format = dyn_cast<DenseTensorDeclOp>(newRhs1.getDefiningOp()).format();
            formats.push_back(rhs_format);
          }
          if ( isa<DenseTensorDeclOp>(newRhs1.getDefiningOp()) && 
                isa<DenseTensorDeclOp>(newRhs2.getDefiningOp()) ) {
            auto rhs_format = dyn_cast<DenseTensorDeclOp>(newRhs1.getDefiningOp()).format();
            formats.push_back(rhs_format);
          }
          if ( isa<tensorAlgebra::TensorMultOp>(newRhs1.getDefiningOp()) ) {  // for series of ta.tc case
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(newRhs2.getDefiningOp()).format();
            formats.push_back(lhs_format);
            formats.push_back(lhs_format); 
          }
          auto strAttr = rewriter.getStrArrayAttr(formats);
          
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
            expr_map[lbl] = getAffineDimExpr(dim++, rewriter.getContext());
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

          auto context = rewriter.getContext();
          SmallVector<mlir::AffineMap, 8> affine_maps{
            mlir::AffineMap::get(dim, 0, rhs_exprs, context),
            mlir::AffineMap::get(dim, 0, lhs_exprs, context),
            mlir::AffineMap::get(dim, 0, ret_exprs, context)};
          auto affineMapArrayAttr = rewriter.getAffineMapArrayAttr(affine_maps);

          auto SemiringAttr = rewriter.getStringAttr("plusxy_times");          
          Value tcop = rewriter.create<tensorAlgebra::TensorMultOp>(loc, newType, newRhs1, newRhs2,
                                                             newSumLabels, affineMapArrayAttr, strAttr, SemiringAttr); 
          tcop.getDefiningOp()->setAttr("__alpha__", rewriter.getF64FloatAttr(1.0));
          tcop.getDefiningOp()->setAttr("__beta__", rewriter.getF64FloatAttr(0.0));
          
          newRhs1 = tcop;
        }

        //mlir::tensorAlgebra::TensorSetOp newSetOp = rewriter.replaceOpWithNewOp<tensorAlgebra::TensorSetOp>(op, newRhs1, operands[1].getDefiningOp()->getOperand(0));
        mlir::tensorAlgebra::TensorSetOp newSetOp = rewriter.create<tensorAlgebra::TensorSetOp>(loc, newRhs1, operands[1].getDefiningOp()->getOperand(0)); 
        newSetOp->setAttr("__beta__", rewriter.getF64FloatAttr(0.0));
        // Remove the old TensorMultOps, the LabeledTensorOps (since they are not used), and the old setOp
        // TODO: the following erase does not work.
        comet_debug() << "Finding the LabeledTensorOps that need to be removed\n";
        for (auto LT : inLTOps) 
        {
          Operation *firstUser;
          for (auto user : LT->getResult(0).getUsers())
          {
            firstUser = user;
            if (isa<tensorAlgebra::LabeledTensorOp>(firstUser))
              break;
          }
          assert(isa<tensorAlgebra::LabeledTensorOp>(firstUser));
          comet_pdump(firstUser);
          LTOpsToRemove.insert(firstUser);
        }

        //rewriter.eraseOp(op);  // remove the old setOp
        comet_debug() << "Removing the LabelTensorOps\n";
        int uses = 0;
        for (auto elem : LTOpsToRemove)
        {
          comet_pdump(elem);
          comet_debug() << "uses: ";
          uses = 0;
          for (auto u: elem->getUsers()) {
            uses++;
            comet_pdump(u);
            rewriter.eraseOp(u);
            //u->erase();
          }
          if (uses == 0)
            //rewriter.eraseOp(elem);
            elem->erase();
          comet_debug() << "\n";
        }
      }
      else
      {
        auto resultType = op->getResultTypes()[0];
        rewriter.replaceOpWithNewOp<tensorAlgebra::TensorSetOp>(op, resultType, operands[0], operands[1]);
      }

      comet_debug() << "MulOpFactorization end\n";
      return success();
    }

    std::vector<int64_t>
    getTensorShape(const std::vector<Operation *> &labels,
                   const std::map<Operation *, int64_t> &lblSizes) const
    {
      std::vector<int64_t> result;

      for (auto lbl : labels)
      {
        result.push_back(lblSizes.at(lbl));
      }

      return result;
    }

    IndexVector
    getLabelPerm(const std::vector<Operation *> &labels,
                 const std::map<Operation *, unsigned> &labelIdMap) const
    {
      IndexVector result;
      for (auto lbl : labels)
      {
        result.push_back(labelIdMap.at(lbl));
      }
      return result;
    }

    std::vector<Operation *>
    findOutput(const std::vector<Operation *> &rhs1Labels,
               const std::vector<Operation *> &rhs2Labels,
               const std::set<Operation *> &outLabels) const
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
                 const std::map<Operation *, std::vector<Operation *>> &lblMaps) const
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
  };

  // TODO: verify the use of the SetOpLowering Pass.
  //       need to check the usage of TensorChainSetOp op.
  struct SetOpLowering : public ConversionPattern
  {
    SetOpLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::TensorChainSetOp::getOperationName(), 1,
                            ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::TensorChainSetOp>(op));

      comet_debug() << "SetOpLowering begin\n";
      comet_pdump(op);
      auto ctx = rewriter.getContext();
      auto loc = op->getLoc();

      auto lhs = operands[0].getDefiningOp();
      auto rhs = operands[1].getDefiningOp();

      assert(isa<tensorAlgebra::LabeledTensorOp>(lhs));
      auto lhsLT = cast<tensorAlgebra::LabeledTensorOp>(lhs);

      Operation *rhsOp;
      if (isa<tensorAlgebra::MulOp>(rhs))
      {
        comet_debug() << "\n";
        rhsOp = cast<tensorAlgebra::MulOp>(rhs);
      }
      else if (isa<tensorAlgebra::AddOp>(rhs))
      {
        comet_debug() << "\n";
        rhsOp = cast<tensorAlgebra::AddOp>(rhs);
      }
      else if (isa<tensorAlgebra::LabeledTensorOp>(rhs))
      {
        comet_debug() << "\n";
        auto rhsLT = cast<tensorAlgebra::LabeledTensorOp>(rhs);

        auto lhsLabels = lhsLT.labels();
        auto rhsLabels = rhsLT.labels();
        std::vector<Operation *> lhsLabelOps, rhsLabelOps;
        for (const auto lbl : lhsLabels)
        {
          lhsLabelOps.push_back(lbl.getDefiningOp());
        }
        for (const auto lbl : rhsLabels)
        {
          rhsLabelOps.push_back(lbl.getDefiningOp());
        }
        auto outPerm = constructPermutationMapAttr(rhsLabelOps, lhsLabelOps);
        auto inPerm = constructPermutationMapAttr(rhsLabelOps, rhsLabelOps);

        auto inPermAttr = AffineMapAttr::get(AffineMap::getPermutationMap(inPerm, ctx));
        auto outPermAttr = AffineMapAttr::get(AffineMap::getPermutationMap(outPerm, ctx));

        auto new_op = rewriter.create<tensorAlgebra::TensorCopyOp>(loc, lhsLT.tensor(), rhsLT.tensor(), inPermAttr, outPermAttr);
        comet_debug() << "\n";
        comet_vdump(new_op);

        double alpha = 1.0;
        auto betaAttr = op->getAttr("__beta__");
        double beta = betaAttr.cast<FloatAttr>().getValueAsDouble();

        new_op.getOperation()->setAttr("__alpha__", rewriter.getF64FloatAttr(alpha));
        new_op.getOperation()->setAttr("__beta__", rewriter.getF64FloatAttr(beta));

        rewriter.eraseOp(op);
        return success();
      }
      else
      {
        comet_debug() << "Neither MulOp, AddOp, nor LabeledTensorOp, it is: ";
        comet_pdump(rhs);
        // return failure();
        comet_debug() << "SetOpLowering end\n";
        return success();
      }

      comet_debug() << "\n";
      auto lhsTensor = lhsLT.tensor();
      comet_debug() << "\n";
      auto labels = lhsLT.labels();
      comet_debug() << "\n";
      std::vector<Value> lhsLabels(labels.begin(), labels.end());

      comet_pdump(rhsOp);
      comet_debug() << "\n";
      replaceSetOp(rhsOp, lhsTensor, lhsLabels, loc, rewriter);
      rewriter.eraseOp(op);
      comet_debug() << "SetOpLowering end\n";
      return success();
    }
  };

  struct TensorCopyLowering : public ConversionPattern
  {
    TensorCopyLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::TensorCopyOp::getOperationName(), 1,
                            ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::TensorCopyOp>(op));

      auto loc = op->getLoc();
      auto tensorCopyOp = cast<tensorAlgebra::TensorCopyOp>(op);

      auto lhsTensorOperand = operands[0];
      auto lhsTensorLoadOp = cast<memref::TensorLoadOp>(lhsTensorOperand.getDefiningOp());
      auto lhsMemref = lhsTensorLoadOp.memref();

      auto rhsTensorOperand = operands[1];
      auto rhsTensorLoadOp = cast<memref::TensorLoadOp>(rhsTensorOperand.getDefiningOp());
      auto rhsMemref = rhsTensorLoadOp.memref();

      auto inPermMap = tensorCopyOp.inputPerm();
      auto outPermMap = tensorCopyOp.outputPerm();

      auto copyOp = rewriter.create<linalg::CopyOp>(loc, rhsMemref, lhsMemref, inPermMap, outPermMap);

      auto alphaAttr = tensorCopyOp.getOperation()->getAttr("__alpha__");
      auto betaAttr = tensorCopyOp.getOperation()->getAttr("__beta__");

      copyOp.getOperation()->setAttr("__alpha__", alphaAttr);
      copyOp.getOperation()->setAttr("__beta__", betaAttr);

      rewriter.eraseOp(op);

      return success();
    }
  };

  //===----------------------------------------------------------------------===//
  // STCRemoveDeadOps RewritePatterns: SparseTensor Constant operations
  //===----------------------------------------------------------------------===//

  struct RemoveDeadOpLowering : public OpRewritePattern<tensorAlgebra::TensorMultOp>
  {
    using OpRewritePattern<tensorAlgebra::TensorMultOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tensorAlgebra::TensorMultOp op,
                                  PatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::TensorMultOp>(op));
      comet_debug() << " erase TensorMultOp \n";
      comet_debug() << "--------------TensorContractionLowering in format\n";
      // Here, should check the operands, at least one operand should be sparse;
      // Otherwise, if all dense operands, just return.
      return success();
    }
  }; // TensorContractionLowering

  template <typename TAOp>
  struct RemoveDeadTAOpLowering : public ConversionPattern
  {
    RemoveDeadTAOpLowering(MLIRContext *ctx)
        : ConversionPattern(TAOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      comet_debug() << " erase op \n";
      rewriter.eraseOp(op);

      return success();
    }
  };

}

// =============================================================================
//
// These patterns implements tensor factorization optimization for multiple operands.
// It identifies optimal ordering of tensor contractions in a given chain of tensor
// contractions.
//===----------------------------------------------------------------------===//
void mlir::tensorAlgebra::populateMultiOpFactorizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context)
{
  //patterns.insert<MulOpFactorization, SetOpLowering>(context);
  patterns.insert<MulOpFactorization>(context);  // not sure why setOpLowering pass is needed if I modified ta.mult -> ta.tc
}

// =============================================================================
//
// These patterns lowers tensor multiplication chains into s series of ta.tc operations.
//
//===----------------------------------------------------------------------===//
void mlir::tensorAlgebra::populateLowerTAMulChainPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context)
{
  patterns.insert<SetOpLowering>(context);
}

void mlir::tensorAlgebra::populateSTCRemoveDeadOpsPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context)
{
  patterns.insert<RemoveDeadTAOpLowering<tensorAlgebra::MulOp>>(context);
  patterns.insert<RemoveDeadTAOpLowering<tensorAlgebra::IndexLabelDynamicOp>>(context);
}