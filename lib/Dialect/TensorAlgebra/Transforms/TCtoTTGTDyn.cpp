//===- TCtoTTGT.cpp ------===//
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
// This file implements reformulation of tensor contraction operations as Transpose-Transpose-GEMM-Transpose
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::arith;
using namespace mlir::bufferization;

using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

const StringLiteral kLinalgTransformMarker = "__with_tiling__";

template <typename T>
static bool arePermutations(const std::vector<T> &vec1,
                            const std::vector<T> &vec2)
{
  if (vec1.size() != vec2.size())
  {
    return false;
  }
  std::vector<bool> taken(vec1.size(), false);
  for (size_t i = 0; i < vec1.size(); i++)
  {
    auto it = std::find(vec2.begin(), vec2.end(), vec1[i]);
    if (it == vec2.end())
    {
      return false;
    }
    if (taken[std::distance(vec2.begin(), it)] == true)
    {
      return false;
    }
    taken[std::distance(vec2.begin(), it)] = true;
  }
  return true;
}

/// Detect whether memref dims [dim, dim + extent) can be reshaped without
/// copies.
[[maybe_unused]] static bool isReshapableDimBand(unsigned dim, unsigned extent,
                                                 ArrayRef<int64_t> sizes,
                                                 ArrayRef<AffineExpr> strides)
{
  assert(sizes.size() == strides.size() && "mismatched ranks");
  /// off by 1 indexing to avoid out of bounds
  for (auto idx = dim, e = dim + extent; idx + 1 < e; ++idx)
  {
    /// Only bands of static shapes are reshapable. This is due to the fact that
    /// there is no relation between dynamic sizes and dynamic strides: we do not
    /// have enough information to know whether a "-1" size corresponds to the
    /// proper symbol in the AffineExpr of a stride.
    if (ShapedType::isDynamic(sizes[dim + 1]))
      return false;
    /// simplify on the fly and catch more reshapable cases.
    if (strides[idx] != strides[idx + 1] * sizes[idx + 1])
      return false;
  }
  return true;
}

static IndexVector getIndexRange(unsigned lo, unsigned hi, unsigned step = 1)
{
  IndexVector result;
  for (unsigned i = lo; i < hi; i += step)
  {
    result.push_back(i);
  }
  return result;
}

//===----------------------------------------------------------------------===//
/// TAEarlyLoweringTTGTPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to linear algebra of the tensor algebra
/// operations that are computationally intensive (like matmul for example...)
/// while keeping the rest of the code in the TA dialect.
namespace
{

  struct TensorContractionOpLoweringTTGT : public ConversionPattern
  {
    TensorContractionOpLoweringTTGT(MLIRContext *ctx, int whatPerm, bool printFlops)
        : ConversionPattern(tensorAlgebra::TensorMultOp::getOperationName(), 1, ctx),
          whatPerm(whatPerm), printFlops{printFlops} {}

    /**
     * @brief Latest implementation with following optimizations:
     *        - if no transpose is required there won't be any copy operations
     *        - if any operand is 2 dimensional no reshape
     *        - does not copy C
     */
    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      comet_pdump(op);
      assert(isa<tensorAlgebra::TensorMultOp>(op));

      auto ctx = rewriter.getContext();
      auto loc = op->getLoc();
      auto multop = cast<tensorAlgebra::TensorMultOp>(op);
      auto alphaAttr = multop.getOperation()->getAttr("__alpha__");
      auto betaAttr = multop.getOperation()->getAttr("__beta__");

      Operation *startTime = nullptr;
      std::string getTimeStr = "getTime";
      auto f64Type = rewriter.getF64Type();
      if (printFlops)
      {
        startTime = rewriter.create<func::CallOp>(
            op->getLoc(), getTimeStr, SmallVector<Type, 2>{f64Type});
      }

      ArrayAttr indexMaps = multop.getIndexingMaps();
      std::vector<std::vector<unsigned>> allPerms;

      /// Find summation indices
      for (const auto &map : indexMaps)
      {
        auto affineMap = cast<AffineMapAttr>(map).getValue();
        std::vector<unsigned> perm;
        for (size_t i = 0; i < affineMap.getNumResults(); i++)
        {
          auto expr = affineMap.getResult(i);
          perm.push_back(llvm::cast<AffineDimExpr>(expr).getPosition());
        }

        allPerms.push_back(perm);
      }


      comet_vdump(setnewop);
      comet_debug() << "\n";

      Value rhs1Tensor = operands[0], rhs2Tensor = operands[1], lhsTensor;

      auto rhs1TensorType = cast<TensorType>(rhs1Tensor.getType());
      auto rhs2TensorType = cast<TensorType>(rhs2Tensor.getType());

      SmallVector<Value, 4> dims; 
      auto shapeT = cast<ShapedType>(multop.getResult().getType());
      ArrayAttr indexing_maps = cast<ArrayAttr>(multop.getIndexingMaps());
      for(auto [index, v]: enumerate(cast<AffineMapAttr>(indexing_maps[2]).getValue().getResults()))
      {
        if(!cast<ShapedType>(rhs1Tensor.getType()).isDynamicDim(index))
        {
          continue;
        }
    
        AffineMap map = cast<AffineMapAttr>(indexing_maps[0]).getValue();
        if (auto pos = map.getResultPosition(v))
        {
          auto dim = rewriter.create<TensorDimOp>(loc, rhs1Tensor, *pos);
          dims.push_back(dim);
          continue;
        }
    
        map = cast<AffineMapAttr>(indexing_maps[1]).getValue(); // try the second map (rhs2)
        if (auto pos = map.getResultPosition(v))
        {
          auto dim = rewriter.create<TensorDimOp>(loc, rhs2Tensor, *pos);
          dims.push_back(dim);
          continue;
        }
      }
      
      auto zero = rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(shapeT.getElementType(), 0.0));
      lhsTensor = rewriter.create<tensor::SplatOp>(loc, shapeT, zero, ValueRange(dims));

      auto lhsTensorType =  cast<TensorType>(lhsTensor.getType());

      std::vector<Value> allShapes{rhs1Tensor,
        rhs2Tensor,
        lhsTensor};

      ContractionPlanDyn plan(rewriter, loc, allPerms[0], allShapes[0], allPerms[1],
                           allShapes[1], allPerms[2], allShapes[2]);

      /// computeBestPermutations identifies the optimal index permutation for TTGT
      /// it should enable and disable to heuristic
      // IndexVector rhs1OutPerm, rhs2OutPerm, lhsOutPerm;
      plan.computePermutations(rewriter, loc);

      comet_debug() << "Best permutation : " << plan.bestPermStr_ << "\n";

      std::set<unsigned> rhsIndices(allPerms[0].begin(), allPerms[0].end());
      rhsIndices.insert(allPerms[1].begin(), allPerms[1].end());
      std::set<unsigned> lhsIndices(allPerms[2].begin(), allPerms[2].end());

      std::vector<unsigned> sumIndices;

      std::set_difference(rhsIndices.begin(), rhsIndices.end(),
                          lhsIndices.begin(), lhsIndices.end(),
                          std::inserter(sumIndices, sumIndices.begin()));

      std::vector<unsigned int> rhs1InPerm = getIdentityPermutation(allPerms[0].size());
      std::vector<unsigned int> rhs2InPerm = getIdentityPermutation(allPerms[1].size());
      std::vector<unsigned int> lhsInPerm = getIdentityPermutation(allPerms[2].size());
      // auto contractionTimes = rewriter.create<tensor::FromElementsOp>(loc, plan.m_contraction_time);

      // Value minContractionTime =  rewriter.create<arith::ConstantFloatOp>(loc, llvm::APFloat(std::numeric_limits<double>::max()), FloatType::getF64(ctx));
      std::vector<int64_t> m_contraction_time_indices;
      for (size_t i = 0; i < plan.m_contraction_time.size(); ++i)
      {
        m_contraction_time_indices.push_back(i);
      }
      scf::IndexSwitchOp switchOp;
      Value permutation; 
      if(whatPerm == -1) // -1 means select the best permutation based on contraction time
      {
        Value minTime = plan.m_contraction_time.front();
        Value minIndex = rewriter.create<ConstantIndexOp>(loc, 0);
        
        for(size_t i = 1; i < plan.m_contraction_time.size(); i++)
        {
          Value thisIndex = rewriter.create<ConstantIndexOp>(loc, i);
          auto foundMin = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT,  plan.m_contraction_time[i], minTime);
          minIndex = rewriter.create<arith::SelectOp>(loc, foundMin, thisIndex, minIndex);
          minTime = rewriter.create<arith::SelectOp>(loc, foundMin, plan.m_contraction_time[i], minTime);
        }
        permutation = minIndex;
      }
      else
      {
        permutation = rewriter.create<arith::ConstantIndexOp>(loc, whatPerm);  // use the specified permutation from the command line
      }

      switchOp = rewriter.create<scf::IndexSwitchOp>(loc, TypeRange(lhsTensor.getType()), permutation,  ArrayRef(m_contraction_time_indices), plan.m_contraction_time.size());

      auto& defaultCaseRegion = switchOp.getDefaultRegion();
      auto& defaultBlock = defaultCaseRegion.emplaceBlock(); // ensure the default case has a block
      rewriter.setInsertionPointToStart(&defaultBlock);
      rewriter.create<scf::YieldOp>(loc, ValueRange(lhsTensor));

      auto caseRegions = switchOp.getCaseRegions();
      for(size_t i = 0; i < plan.m_contraction_time.size(); ++i)
      {
        bool useLHSTranspose = false;
        Value rhs1Final = rhs1Tensor; // default to the original tensor
        Value rhs2Final = rhs2Tensor; // default to the original tensor
        Value lhsFinal = lhsTensor;   // default to the original tensor
        auto& caseBlock = caseRegions[i].emplaceBlock();
        rewriter.setInsertionPointToStart(&caseBlock);
        if(plan.m_transposeA[i])
        {
          auto shape = rhs1TensorType.getShape();
          std::vector<Value> operands;
          std::vector<int64_t> rhs1Dims;
          for (auto idx :plan.m_contraction_permutations[i][0]) // rhs1OutPerm
          {
            rhs1Dims.push_back(shape[idx]);
            if (rhs1TensorType.isDynamicDim(idx))
            {
              operands.push_back(rewriter.create<tensor::DimOp>(loc, rhs1Tensor, idx));
            }
          }

          auto outputShape = rewriter.create<tensor::EmptyOp>(loc, rhs1Dims, rhs1TensorType.getElementType(), operands);
  
          std::vector<int64_t> rhs1OutPerm_int64(plan.m_contraction_permutations[i][0].begin(), plan.m_contraction_permutations[i][0].end());
          rhs1Final = rewriter.create<linalg::TransposeOp>(loc, rhs1Tensor, outputShape, llvm::ArrayRef<int64_t>(rhs1OutPerm_int64)).getResults()[0];
          comet_debug() << "\n";
          comet_vdump(rhs1Final);
        }
        if(plan.m_transposeB[i])
        {
          std::vector<Value> operands;
          std::vector<int64_t> rhs2Dims;
          auto shape = rhs2TensorType.getShape();
          for (auto idx : plan.m_contraction_permutations[i][1]) // use the perm for rhs2
          {
            rhs2Dims.push_back(shape[idx]);
            if (rhs2TensorType.isDynamicDim(idx))
            {
              operands.push_back(rewriter.create<tensor::DimOp>(loc, rhs2Tensor, idx));
            }
          }

          std::vector<int64_t> rhs2OutPerm_int64(plan.m_contraction_permutations[i][1].begin(), plan.m_contraction_permutations[i][1].end());

          auto outputShape = rewriter.create<tensor::EmptyOp>(loc, rhs2Dims, rhs2TensorType.getElementType(), operands);
          rhs2Final = rewriter.create<linalg::TransposeOp>(loc, rhs2Tensor, outputShape, llvm::ArrayRef<int64_t>(rhs2OutPerm_int64)).getResults()[0];
          comet_debug() << " rhs2Transpose op: " << __LINE__ << "\n";
          comet_vdump(rhs2Final);
        }

        if(plan.m_transposeC[i])
        {
          std::vector<Value> operands;
          std::vector<int64_t> lhsDims;
          auto shape = lhsTensorType.getShape();
          for (auto idx : plan.m_contraction_permutations[i][2]) // lhsOutPerm
          {
            lhsDims.push_back(shape[idx]);
            if (lhsTensorType.isDynamicDim(idx))
            {
              operands.push_back(rewriter.create<tensor::DimOp>(loc, lhsTensor, idx));
            }
          }

          useLHSTranspose = true;
          double beta_val = cast<FloatAttr>(betaAttr).getValueAsDouble();
          auto outputShape = rewriter.create<tensor::EmptyOp>(loc, lhsDims, lhsTensorType.getElementType(), operands);
  
          if (beta_val == 0)
          {
            Value constantOp = rewriter.create<ConstantOp>(loc, rewriter.getF64FloatAttr(0.0));
            lhsFinal = rewriter.create<linalg::FillOp>(loc, constantOp, ValueRange(outputShape)).getResults()[0];
          }
          else
          {
            std::vector<int64_t> lhsOutPerm_int64(plan.m_contraction_permutations[i][2].begin(), plan.m_contraction_permutations[i][2].end()); 
            lhsFinal = rewriter.create<linalg::TransposeOp>(loc, lhsTensor, outputShape, llvm::ArrayRef<int64_t>(lhsOutPerm_int64)).getResults()[0];
          }
        }

        Value rhs1Reshape = rhs1Final;
        Value rhs2Reshape = rhs2Final;
        Value lhsReshape = lhsFinal;

      unsigned mIdxSize = plan.m_indices_.size();
      unsigned nIdxSize = plan.n_indices_.size();
      unsigned kIdxSize = plan.k_indices_.size();

      bool isRHS1SumPermutation = arePermutations(allPerms[0], sumIndices);
      bool isRHS2SumPermutation = arePermutations(allPerms[1], sumIndices);

      comet_debug() << __LINE__ << "mIdxSize, nIdxSize, kIdxSize: "
                    << mIdxSize << ", "
                    << nIdxSize << ", "
                    << kIdxSize << " isRHS1SumPermutation, isRHS2SumPermutation: "
                    << isRHS1SumPermutation << ", "
                    << isRHS2SumPermutation << "\n";

      /// Do reshape if needed
      if (isRHS1SumPermutation)
      {
        auto resultShape = rhs1TensorType.getShape();

        auto rhs1AffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);

        SmallVector<AffineMap, 2> rhs1IndexingMap{rhs1AffineMap};

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(rhs1IndexingMap);

        comet_debug() << "\n";
        rhs1Reshape = rewriter.create<tensor::CollapseShapeOp>(
            loc, rhs1Final, reassociationIndices);
        comet_vdump(rhs1Reshape);
      }
      else if (rhs1TensorType.getShape().size() != 2)
      {
        auto resultShape = rhs1TensorType.getShape();
        /// Construct combined shape of 2D memrefc
        std::vector<unsigned> rhs1_0, rhs1_1;

        if (plan.m_swapAB[i])
        {
          rhs1_0 = getIndexRange(0, kIdxSize);
          rhs1_1 = getIndexRange(kIdxSize, kIdxSize + mIdxSize);
        }
        else
        {
          rhs1_0 = getIndexRange(0, mIdxSize);
          rhs1_1 = getIndexRange(mIdxSize, mIdxSize + kIdxSize);
        }

        auto rhs1AffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);
        auto rhs1Subset0 = rhs1AffineMap.getSubMap(rhs1_0);
        auto rhs1Subset1 = rhs1AffineMap.getSubMap(rhs1_1);

        SmallVector<AffineMap, 2> rhs1IndexingMap;

        rhs1IndexingMap.push_back(rhs1Subset0);
        rhs1IndexingMap.push_back(rhs1Subset1);

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(rhs1IndexingMap);

        comet_debug() << "\n";
        comet_debug() << " rhs1Alloc: \n";
        comet_vdump(rhs1Alloc);
        comet_vdump(rhs1MemrefType);

        rhs1Reshape = rewriter.create<tensor::CollapseShapeOp>(
            loc, rhs1Final, reassociationIndices);
        comet_debug() << " Before rhs1Reshape: \n";
        comet_vdump(rhs1Reshape);
        comet_debug() << " After rhs1Reshape: \n";
      }

      if (isRHS2SumPermutation && rhs2TensorType.getShape().size() != 1)
      {
        auto resultShape = rhs2TensorType.getShape();

        auto rhs2AffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);

        SmallVector<AffineMap, 2> rhs2IndexingMap{rhs2AffineMap};

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(rhs2IndexingMap);

        rhs2Reshape = rewriter.create<tensor::CollapseShapeOp>(
            loc, rhs2Final, reassociationIndices);

        comet_debug() << "\n";
        comet_vdump(rhs2Reshape);
      }
      else if (rhs2TensorType.getShape().size() != 2 && rhs2TensorType.getShape().size() != 1)
      {
        auto resultShape = rhs2TensorType.getShape();

        /// Construct combined shape of 2D memref
        std::vector<unsigned> rhs2_0, rhs2_1;

        if (plan.m_swapAB[i])
        {
          rhs2_0 = getIndexRange(0, nIdxSize);
          rhs2_1 = getIndexRange(nIdxSize, nIdxSize + kIdxSize);
        }
        else
        {
          rhs2_0 = getIndexRange(0, kIdxSize);
          rhs2_1 = getIndexRange(kIdxSize, kIdxSize + nIdxSize);
        }

        auto rhs2AffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);
        auto rhs2Subset0 = rhs2AffineMap.getSubMap(rhs2_0);
        auto rhs2Subset1 = rhs2AffineMap.getSubMap(rhs2_1);

        SmallVector<AffineMap, 2> rhs2IndexingMap;

        rhs2IndexingMap.push_back(rhs2Subset0);
        rhs2IndexingMap.push_back(rhs2Subset1);

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(rhs2IndexingMap);

        rhs2Reshape = rewriter.create<tensor::CollapseShapeOp>(
            loc, rhs2Final, reassociationIndices);

        comet_debug() << "\n";
        comet_vdump(rhs2Reshape);
      }

      bool expandLHS = false;
      /// Keep the reassociation indices that will be used for collapsing the LHS tensor
      /// The exact same indices can be used to re-expand it back to its original rank (after the potential transpose operation)
      SmallVector<ReassociationIndices> lhsReassociationIndices;

      comet_debug() << "\n";
      if (isRHS1SumPermutation || (isRHS2SumPermutation && rhs2TensorType.getShape().size() != 1))
      {
        comet_debug() << "\n";
        auto resultShape = lhsTensorType.getShape();

        auto lhsAffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);

        SmallVector<AffineMap, 2> lhsIndexingMap{lhsAffineMap};

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(lhsIndexingMap);

        /// TODO(gkestor): should it be expandop?
        lhsReshape = rewriter.create<tensor::CollapseShapeOp>(
            loc, lhsFinal, reassociationIndices);

        comet_debug() << "\n";
        comet_vdump(lhsReshape);
        expandLHS = true;
        lhsReassociationIndices = reassociationIndices;
      }
      else if (lhsTensorType.getShape().size() != 2 && lhsTensorType.getShape().size() != 1)
      {
        comet_debug() << "\n";
        auto resultShape = lhsTensorType.getShape();
        /// Construct combined shape of 2D memref
        std::vector<unsigned> lhs_0, lhs_1;
        if (plan.m_swapAB[i]) // swap A and B in the matmul
        {
          lhs_0 = getIndexRange(0, nIdxSize);
          lhs_1 = getIndexRange(nIdxSize, nIdxSize + mIdxSize);
        }
        else
        {
          lhs_0 = getIndexRange(0, mIdxSize);
          lhs_1 = getIndexRange(mIdxSize, mIdxSize + nIdxSize);
        }

        auto lhsAffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);
        auto lhsSubset0 = lhsAffineMap.getSubMap(lhs_0);
        auto lhsSubset1 = lhsAffineMap.getSubMap(lhs_1);

        SmallVector<AffineMap, 2> lhsIndexingMap;

        lhsIndexingMap.push_back(lhsSubset0);
        lhsIndexingMap.push_back(lhsSubset1);

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(lhsIndexingMap);

        lhsReshape = rewriter.create<tensor::CollapseShapeOp>(
            loc, lhsFinal, reassociationIndices);
        comet_debug() << "\n";
        comet_vdump(lhsReshape);

        expandLHS = true;
        lhsReassociationIndices = reassociationIndices;
      }

      comet_debug() << "\n";
      /// Create linalg matmul op
      linalg::MatmulOp matmulop;
      linalg::MatvecOp matvecop;

      if (isRHS1SumPermutation)
      {
        comet_debug() << "\n";
        matvecop = rewriter.create<linalg::MatvecOp>(
            loc, ValueRange{rhs2Reshape, rhs1Reshape},
            ValueRange{lhsReshape});
        comet_debug() << "\n";
        comet_vdump(matvecop);

        matvecop.getOperation()->setAttr("__alpha__", alphaAttr);
        matvecop.getOperation()->setAttr("__beta__", betaAttr);
        lhsReshape = matvecop.getResults()[0]; 
        /// TODO(gkestor): Add attribute to the linalg.matvec operations
        /// matvecop.setAttr(kLinalgTransformMarker, rewriter.getStringAttr(kLinalgTransformMarker));
      }
      else if (isRHS2SumPermutation)
      {
        comet_debug() << "\n";
        matvecop = rewriter.create<linalg::MatvecOp>(
            loc, ValueRange{rhs1Reshape, rhs2Reshape},
            ValueRange{lhsReshape});
        comet_debug() << "\n";
        comet_vdump(rhs1Reshape);
        comet_vdump(rhs2Reshape);
        comet_vdump(lhsReshape);
        comet_vdump(matvecop);

        matvecop.getOperation()->setAttr("__alpha__", alphaAttr);
        matvecop.getOperation()->setAttr("__beta__", betaAttr);
        lhsReshape = matvecop.getResults()[0]; 

        /// TODO(gkestor): Add attribute to the linalg.matvec operations
        /// matvecop.setAttr(kLinalgTransformMarker, rewriter.getStringAttr(kLinalgTransformMarker));
      }
      else
      {
        comet_debug() << "\n";

        if (plan.m_swapAB[i])
        {
          std::swap(rhs1Reshape, rhs2Reshape); // swap the operands for matmul when A and B are swapped in the contraction
          comet_debug() << "Swapping rhs1 and rhs2 for matmul due to contraction swap\n";
        }

        matmulop = rewriter.create<linalg::MatmulOp>(
            loc, ValueRange{rhs1Reshape, rhs2Reshape},
            ValueRange{lhsReshape});
        comet_debug() << "\n";
        comet_vdump(rhs1Reshape);
        comet_vdump(rhs2Reshape);
        comet_vdump(lhsReshape);
        comet_vdump(matmulop);
        comet_debug() << "\n";
        /// Add attribute to the linalg.matmul operations
        auto iterator_types = matmulop.getIteratorTypesArray();
        
        matmulop.getOperation()->setAttr(kLinalgTransformMarker,
                                         rewriter.getStringAttr(kLinalgTransformMarker));
        matmulop.getOperation()->setAttr("__alpha__", alphaAttr);
        matmulop.getOperation()->setAttr("__beta__", betaAttr);
        lhsReshape = matmulop.getResults()[0]; 
      }

      Value lhsExpand = lhsReshape;
      if (expandLHS) /// LHS tensor was collapsed and now needs to be re-expanded using the same reassociation indices
      {
        auto expandedTensorType = RankedTensorType::get(cast<RankedTensorType>(lhsFinal.getType()).getShape(), cast<RankedTensorType>(lhsFinal.getType()).getElementType());
        SmallVector<OpFoldResult, 4> dims;
        for(int64_t i = 0; i < expandedTensorType.getRank(); i++)
        {
          if(expandedTensorType.isDynamicDim(i))
          {
            Value dim = rewriter.create<arith::ConstantIndexOp>(loc, i);
            dims.push_back(rewriter.create<tensor::DimOp>(loc, lhsFinal, dim).getResult());
          }
          else
          {
            Value dim = rewriter.create<arith::ConstantIndexOp>(loc, expandedTensorType.getDimSize(i));
            dims.push_back(dim);
          }
        }
        
        

        comet_debug() << "\nExpanded:\n";
        // lhsExpand = rewriter.create<tensor::ExpandShapeOp>(
        //     loc, expandedTensorType, lhsReshape, getReassociationIndicesAttribute(rewriter, lhsReassociationIndices), dims, expandedTensorType.getShape());
        lhsExpand = rewriter.create<tensor::ExpandShapeOp>(
            loc, expandedTensorType, lhsReshape, lhsReassociationIndices, dims);
        comet_debug() << "\n";
        comet_vdump(lhsExpand);
      }

      /// Copy back the result if needed
      if (lhsFinal != lhsTensor && useLHSTranspose)
      {
        std::vector<int64_t> revLhsOutPerm(plan.m_contraction_permutations[i][2].size());
        for (size_t j = 0; j < revLhsOutPerm.size(); j++)
          revLhsOutPerm[plan.m_contraction_permutations[i][2][j]] = j;

        lhsExpand = rewriter.create<linalg::TransposeOp>(loc, lhsExpand, lhsTensor, llvm::ArrayRef<int64_t>(revLhsOutPerm)).getResults()[0];
        comet_vdump(lhsExpand);
        comet_debug() << "\n";

      }
      rewriter.create<scf::YieldOp>(loc, ValueRange({lhsExpand}));
      }
      rewriter.setInsertionPointAfter(switchOp);

      // auto argInt = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIntegerType(64), switchOp.getArg());
      // auto argFloat =
      //     rewriter.create<arith::SIToFPOp>(loc, f64Type, argInt);
      // rewriter.create<tensorAlgebra::PrintOp>(loc, forOp.getResult(0)); // for debugging purposes, print the minimum time found
      // rewriter.create<tensorAlgebra::PrintOp>(loc, argFloat); // for debugging purposes, print the minimum permutation index found

      if (printFlops)
      {
        auto endTime = rewriter.create<func::CallOp>(
            loc, getTimeStr, SmallVector<Type, 2>{f64Type});

        auto start = startTime->getResult(0);
        auto end = endTime.getResult(0);

        Value totalTimeValue =
            rewriter.create<SubFOp>(loc, f64Type, end, start);
        
        Value opNums = rewriter.create<ConstantIndexOp>(loc, 2);
        opNums = rewriter.create<MulIOp>(loc, opNums,
          plan.m_size_); // m size
        opNums = rewriter.create<MulIOp>(loc, opNums,
          plan.n_size_); // n size
        opNums = rewriter.create<MulIOp>(loc, opNums,
          plan.k_size_); // k size
        opNums = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIntegerType(64), opNums);
        opNums = rewriter.create<arith::SIToFPOp>(loc, f64Type, opNums); // convert to f64 for division

        Value flopsOp =
            rewriter.create<DivFOp>(loc, f64Type, opNums, totalTimeValue);

        ///   call @print_flops(%flops) : (f64) -> ()
        std::string printFlopsStr = "print_flops";
        /// auto printFlopsCall =
        rewriter.create<func::CallOp>(
            loc, printFlopsStr, SmallVector<Type, 2>{}, ValueRange{flopsOp});
      }

      rewriter.replaceOp(op, switchOp);
      // rewriter.replaceAllUsesWith(
      //     op->getResults(), switchOp.getResults()); // Replace the original op with the final result of the matmul or matvec
      // rewriter.replaceUsesWithIf(setnewop->getOperand(1), switchOp.getResult(0), [&](OpOperand& use) { 
      //   auto user = use.getOwner();
      //   auto ancestor = switchOp->getBlock()->findAncestorOpInBlock(*user);
      //   return (ancestor && switchOp->isBeforeInBlock(ancestor)); 
      // });
      // op->replaceAllUsesWith(switchOp);
      // rewriter.eraseOp(setnewop);
      // rewriter.eraseOp(op);
      return success();
    }

  private:
    int whatPerm;
    bool printFlops;
  }; /// namespace

  struct TALoweringTTGTDynPass
      : public PassWrapper<TALoweringTTGTDynPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TALoweringTTGTDynPass)
    TALoweringTTGTDynPass(int whatPerm, bool printFlops) : whatPerm(whatPerm), printFlops{printFlops} {};
    void runOnOperation() override;

  private:
    int whatPerm;
    bool printFlops;
  };

} /// end anonymous namespace.

void TALoweringTTGTDynPass::runOnOperation()
{
  func::FuncOp function = getOperation();
  auto module = function.getOperation()->getParentOfType<ModuleOp>();
  auto *ctx = &getContext();

  auto getTimeFunc = FunctionType::get(ctx, {}, {FloatType::getF64(ctx)});
  auto printFlopFunc = FunctionType::get(ctx, {FloatType::getF64(ctx)}, {});

  /// func @getTime() -> f64
  if (this->printFlops && !hasFuncDeclaration(module, "getTime"))
  {
    mlir::func::FuncOp func1 = mlir::func::FuncOp::create(function.getLoc(), "getTime", getTimeFunc,
                                                          ArrayRef<NamedAttribute>{});
    func1.setPrivate();
    module.push_back(func1);
  }

  /// func @print_flops(%flops) : (f64) -> ()
  if (this->printFlops && !hasFuncDeclaration(module, "print_flops"))
  {
    mlir::func::FuncOp func1 = mlir::func::FuncOp::create(function.getLoc(), "print_flops",
                                                          printFlopFunc, ArrayRef<NamedAttribute>{});
    func1.setPrivate();
    module.push_back(func1);
  }

  RewritePatternSet patterns(&getContext());
  patterns.insert<TensorContractionOpLoweringTTGT>(&getContext(), whatPerm, printFlops);

  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, LinalgDialect, ArithDialect, bufferization::BufferizationDialect, scf::SCFDialect, tensor::TensorDialect, tensorAlgebra::TADialect>();
  target.addIllegalOp<tensorAlgebra::TensorMultOp>();

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    llvm::errs() << "Failed to applyPartialConversion in TALoweringTTGTDynPass\n";
    signalPassFailure();
  }
}

/// Create a pass for lowering operations in the `LinAlg` and `Std` dialects,
/// for a subset of the TA IR (e.g. matmul).
/// ordering of permutation starts with one
std::unique_ptr<Pass> mlir::comet::createLoweringTTGTDynPass(int whatPerm, bool printFlops)
{
  return std::make_unique<TALoweringTTGTDynPass>(whatPerm, printFlops);
}
