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
#define comet_errs() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n) n->dump()
#define comet_vdump(n) n.dump()
#else
#define comet_errs() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif

/// Fold simple cast operations that return the same type as the input.
OpFoldResult CastOp::fold(ArrayRef<Attribute> operands)
{
  return mlir::impl::foldCastOp(*this);
}

/// Fold struct constants.
OpFoldResult SparseTensorConstantOp::fold(ArrayRef<Attribute> operands)
{
  return value();
}

/// Fold struct constants.
OpFoldResult SparseTensorVarOp::fold(ArrayRef<Attribute> operands)
{
  return value();
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

void TAOptimalTCFactorizationPass::runOnFunction()
{
  auto function = getFunction();
  OwningRewritePatternList patterns(&getContext());
  populateMultiOpFactorizationPatterns(patterns, &getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<tensorAlgebra::TensorMultOp, tensorAlgebra::TensorFillOp,
                    tensorAlgebra::PrintOp, tensorAlgebra::TAReturnOp,
                    ConstantOp, tensorAlgebra::MulOp, tensorAlgebra::TensorDeclOp,
                    tensorAlgebra::TensorCopyOp>();
  target.addLegalOp<tensorAlgebra::SparseTensorDeclOp, tensorAlgebra::DenseTensorDeclOp>();

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    llvm::errs() << "Failed to Convert\n";
    signalPassFailure();
  }
}

void LowerTAMulChainPass::runOnFunction()
{
  auto function = getFunction();
  OwningRewritePatternList patterns(&getContext());
  populateLowerTAMulChainPatterns(patterns, &getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<tensorAlgebra::TensorMultOp, tensorAlgebra::TensorFillOp,
                    tensorAlgebra::PrintOp, tensorAlgebra::TAReturnOp,
                    ConstantOp, tensorAlgebra::MulOp, tensorAlgebra::TensorDeclOp,
                    tensorAlgebra::TensorCopyOp>();
  target.addLegalOp<tensorAlgebra::SparseTensorDeclOp, tensorAlgebra::DenseTensorDeclOp>();

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    llvm::errs() << "Failed to Convert\n";
    signalPassFailure();
  }
}

void STCRemoveDeadOpsPass::runOnFunction()
{
  comet_errs() << " start STCRemoveDeadOpsPass \n";
  auto function = getFunction();
  auto module = function.getOperation()->getParentOfType<ModuleOp>();
  auto *ctx = &getContext();

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
