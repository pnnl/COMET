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
//

#include "mlir/Pass/Pass.h"

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Analysis/AffineAnalysis.h"
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

#include <algorithm>
#include <map>
#include <set>
#include <iostream>

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
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "comet-passes"

using namespace mlir;
using namespace tensorAlgebra;

// *********** For debug purpose *********//
//#ifndef DEBUG_MODE_PASSES
//#define DEBUG_MODE_PASSES
//#endif

#ifdef DEBUG_MODE_PASSES
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

/// Fold simple cast operations that return the same type as the input.
OpFoldResult CastOp::fold(ArrayRef<Attribute> operands)
{
  return mlir::impl::foldCastOp(*this);
}

namespace
{
  struct TAOptimalTCFactorizationPass
      : public PassWrapper<TAOptimalTCFactorizationPass, FunctionPass>
  {
    void runOnFunction() final;
  };

  struct LowerTAMulChainPass
      : public PassWrapper<LowerTAMulChainPass, FunctionPass>
  {
    void runOnFunction() final;
  };

  struct OptDenseTransposePass
      : public PassWrapper<OptDenseTransposePass, FunctionPass>
  {
    void runOnFunction() final;
  };

  struct STCRemoveDeadOpsPass
      : public PassWrapper<STCRemoveDeadOpsPass, FunctionPass>
  {
    void runOnFunction() final;
  };

} // end anonymous namespace.

namespace
{
  class SimulationAnalysisPass
      : public mlir::PassWrapper<SimulationAnalysisPass, mlir::FunctionPass>
  {

  public:
    void runOnFunction() final;

    void SimulationAnalysis(tensorAlgebra::TensorMultOp op);
  }; // class SimulationAnalysisPass
} // End anonymous namespace

// determine the information needed for hardware simulation analysis.
std::tuple<std::map<int, long long>, std::vector<std::vector<int>>>
DetermineIndexVals (Operation *rhsOp0, Operation *rhsOp1)
{
  std::map<int, long long> IndexToSizes; // i = 128; j = 128; k = 128;
  std::vector<std::vector<int>> IndicesAll;  // input0 = {i, j}; input1 = {j, k}; output = {i, k}; 

  std::vector<Value> all_lbls_value;
  std::vector<Value> rhs0_lbls_value;
  std::vector<Value> rhs1_lbls_value;

  if (isa<tensorAlgebra::TensorMultOp>(rhsOp0))
  {
    auto numOperands = cast<tensorAlgebra::TensorMultOp>(rhsOp0).getNumOperands();  // get the upper-bound.
    for (unsigned int i = 2; i < numOperands; i++) // the label info for output of TensorMultOp
    {
      auto lbl = cast<tensorAlgebra::TensorMultOp>(rhsOp0).getOperand(i);
      // comet_vdump(lbl);
      rhs0_lbls_value.push_back(lbl);
      all_lbls_value.push_back(lbl);
    }
  } // end-mult-op
  else if (isa<tensorAlgebra::DenseTensorDeclOp>(rhsOp0))
  {
    auto rhs0Labels = cast<tensorAlgebra::DenseTensorDeclOp>(rhsOp0).labels();
    for (auto lbl : rhs0Labels)
    {
      rhs0_lbls_value.push_back(lbl);
      all_lbls_value.push_back(lbl);
      //comet_vdump(lbl);
    }
  }
  else if (isa<tensorAlgebra::SparseTensorDeclOp>(rhsOp0))
  {
    auto rhs0Labels = cast<tensorAlgebra::SparseTensorDeclOp>(rhsOp0).labels();
    for (auto lbl : rhs0Labels)
    {
      // check if user provided dynamic index labels for sparse input.
      // these are not supported due to limitations of the simulation framework.
      if (isa<tensorAlgebra::IndexLabelDynamicOp>(lbl.getDefiningOp())) 
      {
        assert (false && "Dynamic index labels are not supported in the sim-analysis pass!");
      }
      rhs0_lbls_value.push_back(lbl);
      all_lbls_value.push_back(lbl);
      //comet_vdump(lbl);
    }
  }
  else
  {
    assert (false && "TO BE SUPPORTED!");
  }

  if (isa<tensorAlgebra::DenseTensorDeclOp>(rhsOp1))
  {
    auto rhs1Labels = cast<tensorAlgebra::DenseTensorDeclOp>(rhsOp1).labels();
    for (auto lbl : rhs1Labels)
    {
      rhs1_lbls_value.push_back(lbl);
      auto result1 = std::find(all_lbls_value.begin(), all_lbls_value.end(), lbl);
      if (result1 == all_lbls_value.end())
      {
        all_lbls_value.push_back(lbl);
      }
      // comet_vdump(lbl);
    }
  } // end-dense-decl
  else if (isa<tensorAlgebra::SparseTensorDeclOp>(rhsOp0))
  {
    auto rhs1Labels = cast<tensorAlgebra::SparseTensorDeclOp>(rhsOp1).labels();
    for (auto lbl : rhs1Labels)
    {
      if (isa<tensorAlgebra::IndexLabelDynamicOp>(lbl.getDefiningOp())) 
      {
        assert (false && "Dynamic index labels are not supported in the sim-analysis pass!");
      }
      rhs1_lbls_value.push_back(lbl);
      auto result1 = std::find(all_lbls_value.begin(), all_lbls_value.end(), lbl);
      if (result1 == all_lbls_value.end())
      {
        all_lbls_value.push_back(lbl);
      }
      // comet_vdump(lbl);
    }
  } // end-sparse-decl
  else 
  {
    assert (false && "TO BE SUPPORTED!");
  }

  std::vector<int> rhs0_lbls;
  std::vector<int> rhs1_lbls;
  for (unsigned int i = 0; i < all_lbls_value.size(); i++)
  {
    auto result1 = std::find(rhs0_lbls_value.begin(), rhs0_lbls_value.end(), all_lbls_value[i]);
    if (result1 != rhs0_lbls_value.end())
    {
      rhs0_lbls.push_back(i);
    }

    auto result2 = std::find(rhs1_lbls_value.begin(), rhs1_lbls_value.end(), all_lbls_value[i]);
    if (result2 != rhs1_lbls_value.end())
    {
      rhs1_lbls.push_back(i);
    }
    
    // populate map for output
    auto indx = cast<tensorAlgebra::IndexLabelStaticOp>(all_lbls_value[i].getDefiningOp()).getOperand(1);  // get the upper-bound.
    auto indxVal = cast<mlir::ConstantIndexOp>(indx.getDefiningOp()).getValue();
    if (IndexToSizes.find(i) == IndexToSizes.end())
    {
      IndexToSizes[i] = indxVal;
    }
  }

  std::vector<int> sum_lbls;
  std::set_intersection(rhs0_lbls.begin(), rhs0_lbls.end(), rhs1_lbls.begin(), rhs1_lbls.end(), std::back_inserter(sum_lbls));
  std::vector<int> all_lbls;
  std::set_union(rhs0_lbls.begin(), rhs0_lbls.end(), rhs1_lbls.begin(), rhs1_lbls.end(), std::back_inserter(all_lbls));
  std::vector<int> ret_lbls;
  std::set_difference(all_lbls.begin(), all_lbls.end(), sum_lbls.begin(), sum_lbls.end(), std::back_inserter(ret_lbls));

  
  // populate index-labels for output
  IndicesAll.push_back(rhs0_lbls);
  IndicesAll.push_back(rhs1_lbls);
  IndicesAll.push_back(ret_lbls);

  return std::make_tuple(IndexToSizes, IndicesAll);
}

bool DetermineTensorType(Operation *op)
{
  bool result = false;  // default: sparse

  if (isa<tensorAlgebra::DenseTensorDeclOp>(op)) 
  {
    //comet_debug() << "found dense\n";
    result = true;
  } 
  else if (isa<tensorAlgebra::SparseTensorDeclOp>(op)) 
  {
    comet_debug() << "found sparse\n";
  } 
  else if (isa<tensorAlgebra::TensorMultOp>(op)) // chain operation
  {
    auto operands = op->getOperands();
    Operation* rhsOp0 = operands[0].getDefiningOp();
    Operation* rhsOp1 = operands[1].getDefiningOp();

    bool MultOp0 = DetermineTensorType(rhsOp0);
    bool MultOp1 = DetermineTensorType(rhsOp1);
    if (MultOp0 && MultOp1)  // if any input is sparse, assume output is sparse
      result = true;
  }
  else
  {
    comet_debug() << "ERROR: undetermined case in Simulation-Analysis Pass\n";
  }

  return result;
}


void SimulationAnalysisPass::SimulationAnalysis(tensorAlgebra::TensorMultOp op)
{
  comet_debug() << "Checking the following op: \n";
  comet_pdump(op);

  // determine the format of inputs
  auto operands = op->getOperands();
  Operation* rhsOp0 = operands[0].getDefiningOp();
  Operation* rhsOp1 = operands[1].getDefiningOp();

  comet_debug() << "\tFollowing are the two operands of op: \n";
  comet_pdump(rhsOp0);
  comet_pdump(rhsOp1);

  // Determine whether the Tensor inputs are Dense or Sparse
  bool IsformatDenseOp0 = DetermineTensorType (rhsOp0);
  bool IsformatDenseOp1 = DetermineTensorType (rhsOp1);

  // determine the indexing maps
  std::map<int, long long> IndexToSizes; 
  std::vector<std::vector<int>> IndicesAll;

  std::tie(IndexToSizes, IndicesAll) = DetermineIndexVals (rhsOp0, rhsOp1);
  
  // print the output to std-out
  std::vector<char> alphabets = {'A', 'B', 'C'};
  std::vector<char> indexAlphabets = {'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'};
  assert(IndexToSizes.size() < indexAlphabets.size()  && "Sim-analysis pass: too many indices.\n");

  unsigned i = 0;
  comet_debug() << "starting config file...\n";
  std::cout << "problem:" << "\n"; 
  std::cout << "  shape:" << "\n"; 
  std::cout << "    name: " << "\"tc\"" << "\n";
  std::cout << "    dimensions: " << "[ ";
  for (auto elem : IndexToSizes) 
  {
    if ((unsigned) elem.first < indexAlphabets.size()-1)
    {
      std::cout << indexAlphabets[elem.first] << ", ";
    }
    else
    {
      assert(false && "Sim-analysis pass: index out of bounds.\n");
    }
  }
  std::cout << " ]" << "\n";

  std::cout << "    data-spaces:" << "\n";
  for (auto outer : IndicesAll) 
  {
    std::cout << "    - name: " << alphabets[i++] << "\n";
    std::cout << "      projection:" << "\n";
    for (auto inner : outer)
    {
      // making this element access in the indexAlphabets vector is safe
      std::cout << "        - - " << "[" << indexAlphabets[inner] << "]" << "\n";
    }

    // the output
    if (i == alphabets.size())
      std::cout << "      read-write: True" << "\n";
  }

  std::cout << "  instance: {";
  for (auto elem : IndexToSizes)
  {
    std::cout << indexAlphabets[elem.first] << ": " <<
            elem.second << ", ";
  }
  std::cout << "}\n";

  if (!IsformatDenseOp0) // for sparse input 0
  {
    std::cout << "    densities:\n";
    std::cout << "      " << alphabets[0] << ":\n";
    std::cout << "        distribution: " << "fixed-structured" << "\n";
    std::cout << "        density: " << 0.25 << "\n";
  }
  if (!IsformatDenseOp1) // for sparse input 1
  {
    if (IsformatDenseOp0) // header only if Op0 was not sparse
    {
      std::cout << "    densities:\n";
    }
    std::cout << "      " << alphabets[1] << ":\n";
    std::cout << "        distribution: " << "fixed-structured" << "\n";
    std::cout << "        density: " << 0.5 << "\n";

  }
  comet_debug() << "done with config file...\n";

}

void TAOptimalTCFactorizationPass::runOnFunction()
{
  auto function = getFunction();
  OwningRewritePatternList patterns(&getContext());
  populateMultiOpFactorizationPatterns(patterns, &getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();

  target.addIllegalDialect<tensorAlgebra::TADialect>();
  target.addLegalOp<tensorAlgebra::TensorMultOp,
                    tensorAlgebra::TensorFillOp,
                    tensorAlgebra::PrintOp,
                    tensorAlgebra::TAReturnOp,
                    ConstantOp, tensorAlgebra::MulOp,
                    tensorAlgebra::TensorCopyOp,
                    tensorAlgebra::SparseTensorDeclOp,
                    tensorAlgebra::DenseTensorDeclOp>();

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    llvm::errs() << "Failed to applyPartialConversion in TAOptimalTCFactorizationPass\n";
    signalPassFailure();
  }
}

void SimulationAnalysisPass::runOnFunction()
{
  comet_debug() << " start SimulationAnalysisPass pass \n";
  auto function = getFunction();

  function.walk([&](tensorAlgebra::TensorMultOp op)
              { SimulationAnalysis(op); });

  comet_debug() << " end SimulationAnalysisPass pass \n";  
}

void LowerTAMulChainPass::runOnFunction()
{
  auto function = getFunction();
  OwningRewritePatternList patterns(&getContext());
  populateLowerTAMulChainPatterns(patterns, &getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();

  target.addIllegalDialect<tensorAlgebra::TADialect>();
  target.addLegalOp<tensorAlgebra::PrintOp,
                    tensorAlgebra::TAReturnOp,
                    tensorAlgebra::SUMOp,
                    tensorAlgebra::TransposeOp,
                    tensorAlgebra::TensorFillOp,
                    tensorAlgebra::TensorFillFromFileOp,
                    tensorAlgebra::GetTimeOp,
                    tensorAlgebra::PrintElapsedTimeOp,
                    tensorAlgebra::TensorMultOp,
                    tensorAlgebra::TensorElewsMultOp,
                    tensorAlgebra::TensorSetOp,
                    tensorAlgebra::MulOp,
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

void STCRemoveDeadOpsPass::runOnFunction()
{
  comet_debug() << " start STCRemoveDeadOpsPass \n";
  ConversionTarget target(getContext());

  target.addLegalDialect<mlir::linalg::LinalgDialect, StandardOpsDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect>();
  target.addLegalOp<tensorAlgebra::TensorMultOp>();
  OwningRewritePatternList patterns(&getContext());
  populateSTCRemoveDeadOpsPatterns(patterns, &getContext());
  if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
  {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::tensorAlgebra::createFindOptimalTCFactorizationPass()
{
  return std::make_unique<TAOptimalTCFactorizationPass>();
}

std::unique_ptr<Pass> mlir::tensorAlgebra::createLowerTAMulChainPass()
{
  return std::make_unique<LowerTAMulChainPass>();
}

// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::tensorAlgebra::createSTCRemoveDeadOpsPass()
{
  return std::make_unique<STCRemoveDeadOpsPass>();
}

// This pass enables hw/sw co-design. 
std::unique_ptr<Pass> mlir::tensorAlgebra::createSimulationAnalysisPass()
{
  return std::make_unique<SimulationAnalysisPass>();
}