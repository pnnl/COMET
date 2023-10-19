//===- LateLowering.cpp------===//
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
// This file implements the lowering of some TA operations such as print, gettime, etc.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

//===----------------------------------------------------------------------===//
/// Late lowering RewritePatterns
//===----------------------------------------------------------------------===//

namespace
{
  /// Lowers `ta.print` to a loop nest calling `printf` on each of the individual
  /// elements of the array.
  class PrintOpLowering : public ConversionPattern
  {
  public:
    explicit PrintOpLowering(MLIRContext *context)
        : ConversionPattern(tensorAlgebra::PrintOp::getOperationName(), 1, context) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
      comet_debug() << "PrintOpLowering starts\n";
      Location loc = op->getLoc();
      auto module = op->getParentOfType<ModuleOp>();
      auto *ctx = op->getContext();
      FloatType f64Type = FloatType::getF64(ctx);
      IndexType indexType = IndexType::get(ctx);
      Type unrankedMemrefType_f64 = UnrankedMemRefType::get(f64Type, 0);

      auto printTensorF64Func = FunctionType::get(ctx, {mlir::UnrankedMemRefType::get(f64Type, 0)}, {});
      auto printTensorIndexFunc = FunctionType::get(ctx, {mlir::UnrankedMemRefType::get(indexType, 0)}, {});
      auto printScalarFunc = FunctionType::get(ctx, {FloatType::getF64(ctx)}, {});

      func::FuncOp print_func;
      auto inputType = op->getOperand(0).getType();

      /// If the Input type is scalar (F64)
      if (inputType.isa<FloatType>())
      {
        std::string print_scalar_f64Str = "printF64";
        std::string print_newline_Str = "printNewline";
        if (!hasFuncDeclaration(module, "printF64"))
        {
          print_func = func::FuncOp::create(loc, print_scalar_f64Str, printScalarFunc, ArrayRef<NamedAttribute>{});
          print_func.setPrivate();
          module.push_back(print_func);

          if (!hasFuncDeclaration(module, "printNewline"))
          {
            auto printNewLineFunc = FunctionType::get(ctx, {}, {});
            func::FuncOp print_newline = func::FuncOp::create(loc, print_newline_Str, printNewLineFunc, ArrayRef<NamedAttribute>{});
            print_newline.setPrivate();
            module.push_back(print_newline);
          }
        }
        rewriter.create<func::CallOp>(loc, print_scalar_f64Str, SmallVector<Type, 2>{}, ValueRange{op->getOperand(0)});
        rewriter.create<func::CallOp>(loc, print_newline_Str, SmallVector<Type, 2>{}, ValueRange{});
      }
      else
      {
        std::string comet_print_f64Str = "comet_print_memref_f64";
        if (!hasFuncDeclaration(module, comet_print_f64Str))
        {
          print_func = func::FuncOp::create(loc, comet_print_f64Str, printTensorF64Func, ArrayRef<NamedAttribute>{});
          print_func.setPrivate();
          module.push_back(print_func);
        }

        if (inputType.isa<MemRefType>())
        {
          auto alloc_op = cast<memref::AllocOp>(op->getOperand(0).getDefiningOp());
          comet_vdump(alloc_op);
          auto u = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_f64, alloc_op);
          rewriter.create<func::CallOp>(loc, comet_print_f64Str, SmallVector<Type, 2>{}, ValueRange{u});
        }
        else
        {
          /// If the Input type is tensor
          if (inputType.isa<TensorType>())
          {
            auto rhs = op->getOperand(0).getDefiningOp();
            auto alloc_op = cast<memref::AllocOp>(rhs->getOperand(0).getDefiningOp());
            comet_vdump(alloc_op);
            auto u = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_f64, alloc_op);
            rewriter.create<func::CallOp>(loc, comet_print_f64Str, SmallVector<Type, 2>{}, ValueRange{u});
          }
          else if (inputType.isa<SparseTensorType>())
          {
            std::string comet_print_i64Str = "comet_print_memref_i64";
            if (!hasFuncDeclaration(module, comet_print_i64Str))
            {
              print_func = func::FuncOp::create(loc, comet_print_i64Str, printTensorIndexFunc, ArrayRef<NamedAttribute>{});
              print_func.setPrivate();
              module.push_back(print_func);
            }

            auto sp_op = cast<tensorAlgebra::SparseTensorConstructOp>(op->getOperand(0).getDefiningOp());
            Type unrankedMemref_index = mlir::UnrankedMemRefType::get(indexType, 0);

            auto rhs = op->getOperand(0).getDefiningOp();
            for (int rsize = 0; rsize < sp_op.getDimArrayCount(); rsize += 2)
            {
              /// accessing xD_pos array and creating cast op for its alloc
              auto xD_pos = rhs->getOperand(rsize).getDefiningOp();
              auto alloc_rhs = cast<memref::AllocOp>(xD_pos->getOperand(0).getDefiningOp());
              auto u = rewriter.create<memref::CastOp>(loc, unrankedMemref_index, alloc_rhs);
              rewriter.create<func::CallOp>(loc, comet_print_i64Str, SmallVector<Type, 2>{}, ValueRange{u});

              /// accessing xD_crd array and creating cast op for its alloc
              auto xD_crd = rhs->getOperand(rsize + 1).getDefiningOp();
              alloc_rhs = cast<memref::AllocOp>(xD_crd->getOperand(0).getDefiningOp());
              u = rewriter.create<memref::CastOp>(loc, unrankedMemref_index, alloc_rhs);
              rewriter.create<func::CallOp>(loc, comet_print_i64Str, SmallVector<Type, 2>{}, ValueRange{u});
            }

            auto xD_value = rhs->getOperand(sp_op.getValueArrayPos()).getDefiningOp();
            auto alloc_rhs = cast<memref::AllocOp>(xD_value->getOperand(0).getDefiningOp());
            auto u = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_f64, alloc_rhs);
            rewriter.create<func::CallOp>(loc, comet_print_f64Str, SmallVector<Type, 2>{}, ValueRange{u});
          }
          else
            llvm::errs() << __FILE__ << " " << __LINE__ << "Unknown Data type\n";
        }
      }

      /// Notify the rewriter that this operation has been removed.
      comet_pdump(op);
      rewriter.eraseOp(op);
      return success();
    }
  };

  class GetTimeLowering : public ConversionPattern
  {
  public:
    explicit GetTimeLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::GetTimeOp::getOperationName(), 1,
                            ctx) {}
    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      auto ctx = rewriter.getContext();
      auto module = op->getParentOfType<ModuleOp>();
      auto f64Type = rewriter.getF64Type();
      std::string getTimeStr = "getTime";

      if (!hasFuncDeclaration(module, getTimeStr))
      {
        auto getTimeFunc = FunctionType::get(ctx, {}, {FloatType::getF64(ctx)});
        /// func @getTime() -> f64
        func::FuncOp func1 = func::FuncOp::create(op->getLoc(), getTimeStr,
                                                  getTimeFunc, ArrayRef<NamedAttribute>{});
        func1.setPrivate();
        module.push_back(func1);
      }

      rewriter.replaceOpWithNewOp<func::CallOp>(op, getTimeStr, SmallVector<Type, 2>{f64Type});

      return success();
    }
  };

  class PrintElapsedTimeLowering : public ConversionPattern
  {
  public:
    explicit PrintElapsedTimeLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::PrintElapsedTimeOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
      auto ctx = rewriter.getContext();
      auto module = op->getParentOfType<ModuleOp>();

      auto start = operands[0];
      auto end = operands[1];
      std::string printElapsedTimeStr = "printElapsedTime";
      auto f64Type = rewriter.getF64Type();

      if (!hasFuncDeclaration(module, printElapsedTimeStr))
      {
        auto printElapsedTimeFunc = FunctionType::get(ctx, {f64Type, f64Type}, {});
        /// func @printElapsedTime(f64, f64) -> ()
        func::FuncOp func1 = func::FuncOp::create(op->getLoc(), printElapsedTimeStr,
                                                  printElapsedTimeFunc, ArrayRef<NamedAttribute>{});
        func1.setPrivate();
        module.push_back(func1);
      }

      rewriter.replaceOpWithNewOp<func::CallOp>(op, printElapsedTimeStr, SmallVector<Type, 2>{}, ValueRange{start, end});

      return success();
    }
  };

  class ReturnOpLowering : public ConversionPattern
  {
  public:
    explicit ReturnOpLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::PrintElapsedTimeOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
      auto ctx = rewriter.getContext();
      auto module = op->getParentOfType<ModuleOp>();

      auto start = operands[0];
      auto end = operands[1];
      std::string printElapsedTimeStr = "printElapsedTime";
      auto f64Type = rewriter.getF64Type();

      if (!hasFuncDeclaration(module, printElapsedTimeStr))
      {
        auto printElapsedTimeFunc = FunctionType::get(ctx, {f64Type, f64Type}, {});
        /// func @printElapsedTime(f64, f64) -> ()
        func::FuncOp func1 = func::FuncOp::create(op->getLoc(), printElapsedTimeStr,
                                                  printElapsedTimeFunc, ArrayRef<NamedAttribute>{});
        func1.setPrivate();
        module.push_back(func1);
      }

      rewriter.replaceOpWithNewOp<func::CallOp>(op, printElapsedTimeStr, SmallVector<Type, 2>{}, ValueRange{start, end});

      return success();
    }
  };

} /// end anonymous namespace.

/// This is a partial lowering to linear algebra of the tensor algebra operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the TA dialect.
namespace
{
  struct LateLoweringPass
      : public PassWrapper<LateLoweringPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LateLoweringPass)
    void runOnOperation() override;
  };
} /// end anonymous namespace.

void LateLoweringPass::runOnOperation()
{
  func::FuncOp function = getOperation();

  /// Removing this check COMET will fail to lower source with functions other than
  /// main. However, this is in contrast with the assumption in EarlyLowering.cpp
  /// that all functions have been inlined.
  if (function.getName() == "main")
  {
    ///  Verify that the given main has no inputs and results.
    if (function.getNumArguments() || function.getFunctionType().getNumResults())
    {
      function.emitError("expected 'main' to have 0 inputs and 0 results");
      return signalPassFailure();
    }
  }

  /// The first thing to define is the conversion target. This will define the
  /// final target for this lowering.
  ConversionTarget target(getContext());

  target.addIllegalDialect<tensorAlgebra::TADialect>();

  /// We define the specific operations, or dialects, that are legal targets for
  /// this lowering.
  target.addLegalDialect<AffineDialect,
                         scf::SCFDialect,
                         ArithDialect,
                         memref::MemRefDialect,
                         bufferization::BufferizationDialect>();

  /// PrintOp Lowering insert function call, so mark some operations as a legal Operation
  target.addLegalOp<func::CallOp,                           /// for function calls
                    tensorAlgebra::SparseTensorConstructOp, /// in the case printing sparse tensor
                    tensorAlgebra::TensorSetOp              /// in the case assigning the result of the operation to the final output
                    >();

  /// Now that the conversion target has been defined, we just need to provide
  /// the set of patterns that will lower the TA operations.
  RewritePatternSet patterns(&getContext());
  patterns.insert<PrintOpLowering,
                  GetTimeLowering,
                  PrintElapsedTimeLowering>(&getContext());

  /// With the target and rewrite patterns defined, we can now attempt the
  /// conversion. The conversion will signal failure if any of our `illegal`
  /// operations were not converted successfully.
  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    signalPassFailure();
  }
}

/// Create a pass for lowering utility operations in tensor algebra to lower level dialects
std::unique_ptr<Pass> mlir::comet::createLateLoweringPass()
{
  return std::make_unique<LateLoweringPass>();
}
