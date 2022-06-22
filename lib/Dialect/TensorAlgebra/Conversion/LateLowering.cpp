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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/EDSC/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_TALateLoweringPass
// #define DEBUG_MODE_TALateLoweringPass
// #endif

#ifdef DEBUG_MODE_TALateLoweringPass
#define comet_errs() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n) n->dump()
#define comet_vdump(n) n.dump()
#else
#define comet_errs() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

//===----------------------------------------------------------------------===//
// Late lowering RewritePatterns
//===----------------------------------------------------------------------===//

namespace
{
  //===----------------------------------------------------------------------===//
  // Late Lowering to Standard Dialect RewritePatterns: Constant operations
  //===----------------------------------------------------------------------===//
  struct ConstantOpLowering : public OpRewritePattern<tensorAlgebra::DenseConstantOp>
  {

    using OpRewritePattern<tensorAlgebra::DenseConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensorAlgebra::DenseConstantOp op,
                                  PatternRewriter &rewriter) const final
    {
      DenseElementsAttr constantValue = op.value();
      Location loc = op.getLoc();

      tensorAlgebra::TensorSetOp setnewop;
      for (auto u : op.getOperation()->getResult(0).getUsers())
      {
        if (isa<tensorAlgebra::TensorSetOp>(u))
        {
          setnewop = cast<tensorAlgebra::TensorSetOp>(u);
        }
        else
        {
          llvm::errs() << __FILE__ << " " << __LINE__ << "Dense constant op is used in ops besides SetOp\n";
        }
      }

      mlir::Value rhs = setnewop.rhs();
      mlir::Value lhs = setnewop.lhs();

      auto LabeledTensoroperands = rhs.getDefiningOp()->getOperands();

      auto tensorload = cast<memref::TensorLoadOp>(LabeledTensoroperands[0].getDefiningOp());
      auto alloc_rhs = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());

      // When lowering the constant operation, we allocate and assign the constant
      // values to a corresponding memref allocation.
      auto tensorType = op.getType().cast<TensorType>();
      auto memRefType = convertTensorToMemRef(tensorType);
      // auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

      // We will be generating constant indices up-to the largest dimension.
      // Create these constants up-front to avoid large amounts of redundant
      // operations.
      auto valueShape = memRefType.getShape();
      SmallVector<Value, 8> constantIndices;
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));

      // The constant operation represents a multi-dimensional constant, so we
      // will need to generate a store for each of the elements. The following
      // functor recursively walks the dimensions of the constant shape,
      // generating a store when the recursion hits the base case.

      SmallVector<Value, 2> indices;
      auto valueIt = constantValue.getValues<FloatAttr>().begin();
      std::function<void(uint64_t)> storeElements = [&](uint64_t dimension)
      {
        // The last dimension is the base case of the recursion, at this point
        // we store the element at the given index.
        if (dimension == valueShape.size())
        {
          rewriter.create<memref::StoreOp>(
              loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc_rhs,
              llvm::makeArrayRef(indices));

          return;
        }

        // Otherwise, iterate over the current dimension and add the indices to
        // the list.
        for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i)
        {
          indices.push_back(constantIndices[i]);
          storeElements(dimension + 1);
          indices.pop_back();
        }
      };

      // Start the element storing recursion from the first dimension.
      storeElements(/*dimension=*/0);

      // Replace this operation with the generated alloc.
      rewriter.eraseOp(setnewop);
      rewriter.eraseOp(op);

      return success();
    }
  };

  //===----------------------------------------------------------------------===//
  // Late Lowering to Standard Dialect RewritePatterns: Return operations
  //===----------------------------------------------------------------------===//
  struct ReturnOpLowering : public OpRewritePattern<tensorAlgebra::TAReturnOp>
  {
    using OpRewritePattern<tensorAlgebra::TAReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensorAlgebra::TAReturnOp op,
                                  PatternRewriter &rewriter) const final
    {
      // During this lowering, we expect that all function calls have been
      // inlined.
      if (op.hasOperand())
        return failure();

      // We lower "ta.return" directly to "std.return".
      rewriter.replaceOpWithNewOp<ReturnOp>(op);
      return success();
    }
  };

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

      Location loc = op->getLoc();
      auto module = op->getParentOfType<ModuleOp>();
      auto *ctx = op->getContext();
      FloatType f64Type = FloatType::getF64(ctx);
      IndexType indexType = IndexType::get(ctx);
      Type unrankedMemrefType_f64 = UnrankedMemRefType::get(f64Type, 0);

      auto printTensorF64Func = FunctionType::get(ctx, {mlir::UnrankedMemRefType::get(f64Type, 0)}, {});
      auto printTensorIndexFunc = FunctionType::get(ctx, {mlir::UnrankedMemRefType::get(indexType, 0)}, {});
      auto printScalarFunc = FunctionType::get(ctx, {FloatType::getF64(ctx)}, {});

      FuncOp print_func;
      auto inputType = op->getOperand(0).getType();

      // If the Input type is scalar (F64)
      if (inputType.isa<FloatType>())
      {
        std::string print_scalar_f64Str = "printF64";
        std::string print_newline_Str = "printNewline";
        if (isFuncInMod("printF64", module) == false)
        {
          print_func = FuncOp::create(loc, print_scalar_f64Str, printScalarFunc, ArrayRef<NamedAttribute>{});
          print_func.setPrivate();
          module.push_back(print_func);

          if (isFuncInMod("printNewline", module) == false)
          {
            auto printNewLineFunc = FunctionType::get(ctx, {}, {});
            FuncOp print_newline = FuncOp::create(loc, print_newline_Str, printNewLineFunc, ArrayRef<NamedAttribute>{});
            print_newline.setPrivate();
            module.push_back(print_newline);
          }
        }
        rewriter.create<mlir::CallOp>(loc, print_scalar_f64Str, SmallVector<Type, 2>{}, ValueRange{op->getOperand(0)});
        rewriter.create<mlir::CallOp>(loc, print_newline_Str, SmallVector<Type, 2>{}, ValueRange{});
      }
      else
      {
        std::string comet_print_f64Str = "comet_print_memref_f64";
        if (isFuncInMod(comet_print_f64Str, module) == false)
        {
          print_func = FuncOp::create(loc, comet_print_f64Str, printTensorF64Func, ArrayRef<NamedAttribute>{});
          print_func.setPrivate();
          module.push_back(print_func);
        }

        if (inputType.isa<MemRefType>())
        {
          auto alloc_op = cast<memref::AllocOp>(op->getOperand(0).getDefiningOp());
          auto u = rewriter.create<memref::CastOp>(loc, alloc_op, unrankedMemrefType_f64);
          rewriter.create<mlir::CallOp>(loc, comet_print_f64Str, SmallVector<Type, 2>{}, ValueRange{u});
        }
        else
        {
          // If the Input type is tensor
          if (inputType.isa<TensorType>())
          {
            auto rhs = op->getOperand(0).getDefiningOp();
            auto alloc_op = cast<memref::AllocOp>(rhs->getOperand(0).getDefiningOp());

            auto u = rewriter.create<memref::CastOp>(loc, alloc_op, unrankedMemrefType_f64);
            rewriter.create<mlir::CallOp>(loc, comet_print_f64Str, SmallVector<Type, 2>{}, ValueRange{u});
          }
          else if (inputType.isa<SparseTensorType>())
          {
            std::string comet_print_i64Str = "comet_print_memref_i64";

            if (isFuncInMod(comet_print_i64Str, module) == false)
            {
              print_func = FuncOp::create(loc, comet_print_i64Str, printTensorIndexFunc, ArrayRef<NamedAttribute>{});
              print_func.setPrivate();
              module.push_back(print_func);
            }

            // SparseTensorType includes 5 metadata per dimension. Additionally, 2 elements for value array, value array size.
            int tensorRanks = (op->getOperand(0).getDefiningOp()->getNumOperands() - 2) / 5;
            Type unrankedMemref_index = mlir::UnrankedMemRefType::get(indexType, 0);

            auto rhs = op->getOperand(0).getDefiningOp();
            for (int rsize = 0; rsize < tensorRanks; rsize++)
            {
              // accessing xD_pos array and creating cast op for its alloc
              auto xD_pos = rhs->getOperand(rsize * 2).getDefiningOp();
              auto alloc_rhs = cast<memref::AllocOp>(xD_pos->getOperand(0).getDefiningOp());
              auto u = rewriter.create<memref::CastOp>(loc, alloc_rhs, unrankedMemref_index);
              rewriter.create<mlir::CallOp>(loc, comet_print_i64Str, SmallVector<Type, 2>{}, ValueRange{u});

              // accessing xD_crd array and creating cast op for its alloc
              auto xD_crd = rhs->getOperand((rsize * 2) + 1).getDefiningOp();
              alloc_rhs = cast<memref::AllocOp>(xD_crd->getOperand(0).getDefiningOp());
              u = rewriter.create<memref::CastOp>(loc, alloc_rhs, unrankedMemref_index);
              rewriter.create<mlir::CallOp>(loc, comet_print_i64Str, SmallVector<Type, 2>{}, ValueRange{u});
            }

            auto xD_value = rhs->getOperand(tensorRanks * 2).getDefiningOp();
            auto alloc_rhs = cast<memref::AllocOp>(xD_value->getOperand(0).getDefiningOp());
            auto u = rewriter.create<memref::CastOp>(loc, alloc_rhs, unrankedMemrefType_f64);
            rewriter.create<mlir::CallOp>(loc, comet_print_f64Str, SmallVector<Type, 2>{}, ValueRange{u});
          }
          else
            llvm::errs() << __FILE__ << " " << __LINE__ << "Unknown Data type\n";
        }
      }

      // Notify the rewriter that this operation has been removed.
      rewriter.eraseOp(op);
      return success();
    }
  };

  /*
    Lower elementwise tensor addition.
  */
  struct TensorAdditionLowering : public OpRewritePattern<tensorAlgebra::AddOp>
  {
    using OpRewritePattern<tensorAlgebra::AddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensorAlgebra::AddOp op,
                                  PatternRewriter &rewriter) const final
    {
      Location loc = op.getLoc();
      Value lhs = op.lhs();
      Value rhs = op.rhs();

      auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
      auto step = rewriter.create<ConstantIndexOp>(loc, 1);

      auto rhsTy = rhs.getType().cast<mlir::TensorType>();
      std::vector<scf::ForOp> forloops;
      Value upperBound;
      std::vector<Value> InductionVars;
      for (int i = 0; i < rhsTy.getRank(); i++)
      {

        upperBound = rewriter.create<ConstantIndexOp>(loc, rhsTy.getDimSize(i));
        auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        forloops.push_back(loop);
        InductionVars.push_back(loop.getInductionVar());
        rewriter.setInsertionPointToStart(loop.getBody());
      }
      auto lhs_def_op = lhs.getDefiningOp();
      memref::AllocOp alloc_lhs;
      if (isa<tensorAlgebra::LabeledTensorOp>(lhs_def_op))
      {

        auto LabeledTensoroperands_lhs = lhs.getDefiningOp()->getOperands();
        auto tensorload_lhs = cast<memref::TensorLoadOp>(LabeledTensoroperands_lhs[0].getDefiningOp());
        alloc_lhs = cast<memref::AllocOp>(tensorload_lhs->getOperand(0).getDefiningOp());
      }
      else
      {
        Operation *tensorload_lhs = cast<memref::TensorLoadOp>(lhs.getDefiningOp());
        alloc_lhs = cast<memref::AllocOp>(tensorload_lhs->getOperand(0).getDefiningOp());
      }

      Operation *tensorload_rhs = cast<memref::TensorLoadOp>(rhs.getDefiningOp());
      auto alloc_rhs = cast<memref::AllocOp>(tensorload_rhs->getOperand(0).getDefiningOp());

      auto load_lhs = rewriter.create<memref::LoadOp>(loc, alloc_lhs, InductionVars);
      auto load_rhs = rewriter.create<memref::LoadOp>(loc, alloc_rhs, InductionVars);
      auto sum = rewriter.create<mlir::AddFOp>(loc, load_lhs, load_rhs);

      tensorAlgebra::TensorSetOp setnewop;
      for (auto u : op.getOperation()->getResult(0).getUsers())
      {
        if (isa<tensorAlgebra::TensorSetOp>(u))
        {
          setnewop = cast<tensorAlgebra::TensorSetOp>(u);
        }
        else
        {
          llvm::errs() << __FILE__ << " " << __LINE__ << "Dense constant op is used in ops besides SetOp\n";
        }
      }

      mlir::Value setOp_rhs = setnewop.rhs();
      mlir::Value setOp_lhs = setnewop.lhs();

      auto def_setOp_rhs = setOp_rhs.getDefiningOp();

      memref::AllocOp SetOp_rhs_alloc;

      if (isa<tensorAlgebra::LabeledTensorOp>(def_setOp_rhs))
      {
        auto LTSetOp_rhs = setOp_rhs.getDefiningOp()->getOperands();
        auto tensorload_SetOp_rhs = cast<memref::TensorLoadOp>(LTSetOp_rhs[0].getDefiningOp());
        SetOp_rhs_alloc = cast<memref::AllocOp>(tensorload_SetOp_rhs->getOperand(0).getDefiningOp());
      }
      else
      {
        Operation *tensorload_setOp = cast<memref::TensorLoadOp>(setOp_rhs.getDefiningOp());
        SetOp_rhs_alloc = cast<memref::AllocOp>(tensorload_setOp->getOperand(0).getDefiningOp());
      }

      auto store_sum = rewriter.create<memref::StoreOp>(loc, sum, SetOp_rhs_alloc, InductionVars);
      rewriter.setInsertionPointAfter(forloops[0]);
      rewriter.eraseOp(setnewop);
      rewriter.eraseOp(op);
      return success();
    }
  };

  struct TensorSubtractionLowering : public OpRewritePattern<tensorAlgebra::SubstractOp>
  {
    using OpRewritePattern<tensorAlgebra::SubstractOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensorAlgebra::SubstractOp op,
                                  PatternRewriter &rewriter) const final
    {

      Location loc = op.getLoc();
      Value lhs = op.lhs();
      Value rhs = op.rhs();

      auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
      auto step = rewriter.create<ConstantIndexOp>(loc, 1);

      auto rhsTy = rhs.getType().cast<mlir::TensorType>();
      std::vector<scf::ForOp> forloops;
      Value upperBound;
      std::vector<Value> InductionVars;
      for (int i = 0; i < rhsTy.getRank(); i++)
      {

        upperBound = rewriter.create<ConstantIndexOp>(loc, rhsTy.getDimSize(i));
        auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        forloops.push_back(loop);
        InductionVars.push_back(loop.getInductionVar());
        rewriter.setInsertionPointToStart(loop.getBody());
      }

      auto lhs_def_op = lhs.getDefiningOp();
      memref::AllocOp alloc_lhs;
      if (isa<tensorAlgebra::LabeledTensorOp>(lhs_def_op))
      {

        auto LabeledTensoroperands_lhs = lhs.getDefiningOp()->getOperands();
        auto tensorload_lhs = cast<memref::TensorLoadOp>(LabeledTensoroperands_lhs[0].getDefiningOp());
        alloc_lhs = cast<memref::AllocOp>(tensorload_lhs->getOperand(0).getDefiningOp());
      }
      else
      {
        Operation *tensorload_lhs = cast<memref::TensorLoadOp>(lhs.getDefiningOp());
        alloc_lhs = cast<memref::AllocOp>(tensorload_lhs->getOperand(0).getDefiningOp());
      }

      Operation *tensorload_rhs = cast<memref::TensorLoadOp>(rhs.getDefiningOp());
      auto alloc_rhs = cast<memref::AllocOp>(tensorload_rhs->getOperand(0).getDefiningOp());

      auto load_lhs = rewriter.create<memref::LoadOp>(loc, alloc_lhs, InductionVars);
      auto load_rhs = rewriter.create<memref::LoadOp>(loc, alloc_rhs, InductionVars);
      auto subtract = rewriter.create<mlir::SubFOp>(loc, load_lhs, load_rhs);

      tensorAlgebra::TensorSetOp setnewop;
      for (auto u : op.getOperation()->getResult(0).getUsers())
      {
        if (isa<tensorAlgebra::TensorSetOp>(u))
        {
          setnewop = cast<tensorAlgebra::TensorSetOp>(u);
        }
        else
        {
          llvm::errs() << __FILE__ << " " << __LINE__ << "Dense constant op is used in ops besides SetOp\n";
        }
      }

      mlir::Value setOp_rhs = setnewop.rhs();
      mlir::Value setOp_lhs = setnewop.lhs();

      auto def_setOp_rhs = setOp_rhs.getDefiningOp();

      memref::AllocOp SetOp_rhs_alloc;

      if (isa<tensorAlgebra::LabeledTensorOp>(def_setOp_rhs))
      {
        auto LTSetOp_rhs = setOp_rhs.getDefiningOp()->getOperands();
        auto tensorload_SetOp_rhs = cast<memref::TensorLoadOp>(LTSetOp_rhs[0].getDefiningOp());
        SetOp_rhs_alloc = cast<memref::AllocOp>(tensorload_SetOp_rhs->getOperand(0).getDefiningOp());
      }
      else
      {
        Operation *tensorload_setOp = cast<memref::TensorLoadOp>(setOp_rhs.getDefiningOp());
        SetOp_rhs_alloc = cast<memref::AllocOp>(tensorload_setOp->getOperand(0).getDefiningOp());
      }

      auto store_sum = rewriter.create<memref::StoreOp>(loc, subtract, SetOp_rhs_alloc, InductionVars);

      rewriter.setInsertionPointAfter(forloops[0]);
      rewriter.eraseOp(setnewop);
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
      std::string getTimeStr = "getTime";
      auto f64Type = rewriter.getF64Type();

      if (!hasFuncDeclaration(module, getTimeStr))
      {
        auto getTimeFunc = FunctionType::get(ctx, {}, {FloatType::getF64(ctx)});
        // func @getTime() -> f64
        FuncOp func1 = FuncOp::create(op->getLoc(), getTimeStr,
                                      getTimeFunc, ArrayRef<NamedAttribute>{});
        func1.setPrivate();
        module.push_back(func1);
      }

      rewriter.replaceOpWithNewOp<mlir::CallOp>(op, getTimeStr, SmallVector<Type, 2>{f64Type});

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
        // func @printElapsedTime(f64, f64) -> ()
        FuncOp func1 = FuncOp::create(op->getLoc(), printElapsedTimeStr,
                                      printElapsedTimeFunc, ArrayRef<NamedAttribute>{});
        func1.setPrivate();
        module.push_back(func1);
      }

      rewriter.replaceOpWithNewOp<mlir::CallOp>(op, printElapsedTimeStr, SmallVector<Type, 2>{}, ValueRange{start, end});

      return success();
    }
  };

} // end anonymous namespace.

/// This is a partial lowering to linear algebra of the tensor algebra operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the TA dialect.
namespace
{
  struct TALateLoweringPass
      : public PassWrapper<TALateLoweringPass, FunctionPass>
  {
    void runOnFunction() final;
  };
} // end anonymous namespace.

void TALateLoweringPass::runOnFunction()
{

  auto function = getFunction();

  // llvm::outs() << "Late lower input:\n" <<  function << "\n";
  //  Verify that the given main has no inputs and results.
  if (function.getNumArguments() || function.getType().getNumResults())
  {
    function.emitError("expected 'main' to have 0 inputs and 0 results");
    return signalPassFailure();
  }

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `LinAlg` and `Standard` dialects.
  target.addLegalDialect<AffineDialect, LinalgDialect, scf::SCFDialect, StandardOpsDialect, memref::MemRefDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the TA operations.
  // OwningRewritePatternList patterns;
  OwningRewritePatternList patterns(&getContext());
  patterns.insert<ConstantOpLowering,
                  ReturnOpLowering,
                  PrintOpLowering,
                  GetTimeLowering,
                  PrintElapsedTimeLowering>(&getContext());

  patterns.insert<TensorAdditionLowering, TensorSubtractionLowering>(&getContext());
  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.

  if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
  {
    signalPassFailure();
  }

  // llvm::outs() << "Late lower output: \n" << function << "\n";
}

/// Create a pass for lowering operations in the `LinAlg` and `Std` dialects,
/// for a subset of the TA IR (e.g. matmul).
std::unique_ptr<Pass> mlir::tensorAlgebra::createLateLoweringPass()
{
  return std::make_unique<TALateLoweringPass>();
}
