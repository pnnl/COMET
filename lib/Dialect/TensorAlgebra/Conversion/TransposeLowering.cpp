//===- TransposeLowering.cpp -----//
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

// This file implements a lowering of tensor transpose operation in
// which one of the inputs might be sparse tensor to runtime sorting functions
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
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

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::tensorAlgebra;

#define DEBUG_TYPE "transpose-lowering"

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_TransposeLoweringPass
// #define DEBUG_MODE_TransposeLoweringPass
// #endif

#ifdef DEBUG_MODE_TransposeLoweringPass
#define comet_errs() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n) n->dump()
#define comet_vdump(n) n.dump()
#else
#define comet_errs() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

unsigned perm2num(std::vector<unsigned> vec)
{
  unsigned psum = 0;
  unsigned sz = vec.size();
  unsigned decimal = pow(10, (sz - 1));

  for (auto n : vec)
  {
    psum += n * decimal;
    decimal /= 10;
  }

  return psum;
}

//===----------------------------------------------------------------------===//
// TransposeLowering PASS
//===----------------------------------------------------------------------===//
/// Lowering the ta.transpose (tensor transpose operation in TA dialect) to SCF or runtime utility functions
namespace
{

  struct TensorTransposeLowering : public OpRewritePattern<tensorAlgebra::TransposeOp>
  {
    using OpRewritePattern<tensorAlgebra::TransposeOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Identify the tensor sparsity property - dense or sparse
     * Step 2: Write auto gen mlir code algorithm
     *        - If it is dense, lower transposeOp to SCF
     *        - If input sparse tensor format CSR and the output tensor format CSC - generate code in SCF
     *        - For the rest of the cases, we need a sorting algorithm to reorder coordinate after transpose.
     *          In this case, make a call to utility library for sorting.
     */
    LogicalResult matchAndRewrite(tensorAlgebra::TransposeOp op,
                                  PatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::TransposeOp>(op));

      auto module = op->getParentOfType<ModuleOp>();
      // module->dump();
      Location loc = op.getLoc();
      comet_errs() << " Transpose lowering\n";
      comet_vdump(op);
      auto *ctx = op->getContext();
      auto inputType = op->getOperand(0).getType();

      // Get tensor contraction expression through analyzing the index map
      ArrayAttr indexMaps = op.indexing_maps();
      std::vector<std::vector<unsigned int>> allPerms = getAllPerms(indexMaps);

      // There are tensors for transpose operation: input and output tensors
      unsigned int tensors_num = 2;
      tensorAlgebra::TensorSetOp setOp;
      Value lhs;

      if (inputType.isa<TensorType>())
      { // for dense
        AffineMap inputPermuation = AffineMap::getPermutationMap(allPerms[0], ctx);
        AffineMap outputPermuation = AffineMap::getPermutationMap(allPerms[1], ctx);

        auto inputTensorLoadOp = cast<memref::TensorLoadOp>(op->getOperand(0).getDefiningOp());
        auto inputMemref = inputTensorLoadOp.memref();

        for (auto u : op.getOperation()->getResult(0).getUsers())
        {
          if (isa<tensorAlgebra::TensorSetOp>(u))
          {
            setOp = cast<tensorAlgebra::TensorSetOp>(u);
            Value dstTensor = u->getOperand(1);
            if (isa<tensorAlgebra::LabeledTensorOp>(dstTensor.getDefiningOp()))
            {
              Value dstTensor_labeledTensor = cast<tensorAlgebra::LabeledTensorOp>(dstTensor.getDefiningOp());
              lhs = dstTensor_labeledTensor.getDefiningOp()->getOperand(0);
            }
            else
            {
              lhs = dstTensor;
            }
          }
        }

        auto outputMemref = lhs.getDefiningOp()->getOperand(0);
        rewriter.create<linalg::CopyOp>(loc, inputMemref, outputMemref, inputPermuation, outputPermuation);
        Value res_value = rewriter.create<memref::TensorLoadOp>(loc, outputMemref);

        op.replaceAllUsesWith(res_value);
        rewriter.eraseOp(op);

        return success();
      }
      else
      { // for sparse tensors
        unsigned pnum[2];
        // print allPerms
        int i = 0;
        for (auto perm : allPerms)
        { // lhs, rhs: from left to right order
          pnum[i] = perm2num(perm);
          i++;
        }

        ArrayAttr opFormatsArrayAttr = op.formats();
        std::string formats_str(opFormatsArrayAttr[0].cast<mlir::StringAttr>().getValue());
        IntegerType i32Type = IntegerType::get(ctx, 32);
        IndexType indexType = IndexType::get(ctx);
        FloatType f64Type = FloatType::getF64(ctx);

        Value input_perm_num = rewriter.create<mlir::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(pnum[0]));
        Value output_perm_num = rewriter.create<mlir::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(pnum[1]));

        Type unrankedMemrefType_f64 = UnrankedMemRefType::get(f64Type, 0);
        Type unrankedMemrefType_i32 = UnrankedMemRefType::get(i32Type, 0);
        Type unrankedMemrefType_index = UnrankedMemRefType::get(indexType, 0);

        FuncOp transpose_func; // runtime call

        std::vector<std::vector<mlir::Value>> alloc_sizes_cast_vecs{tensors_num};
        std::vector<std::vector<mlir::Value>> allocs_for_sparse_tensors{tensors_num};
        std::vector<mlir::Value> tensors = {op.getOperation()->getOperand(0)};

        for (auto u : op.getOperation()->getResult(0).getUsers())
        {
          if (isa<tensorAlgebra::TensorSetOp>(u))
          {
            setOp = cast<tensorAlgebra::TensorSetOp>(u);
            mlir::Value lhs = setOp->getOperand(1); // dest tensor is the 2nd
            tensors.push_back(lhs);
          }
        }

        for (unsigned int n = 0; n < tensors_num; n++)
        {
          unsigned int tensor_rank = (tensors[n].getDefiningOp()->getNumOperands() - 2) / 5;
          comet_errs() << " tensor_rank: " << tensor_rank << "\n";
          comet_errs() << " tensor[n]: "
                       << "\n";
          comet_pdump(tensors[n].getDefiningOp());

          for (unsigned int i = 0; i < 2 * tensor_rank + 1; i++)
          {
            auto tensorload_op = tensors[n].getDefiningOp()->getOperand(i);
            comet_errs() << " tensorload_op "
                         << "\n";
            comet_vdump(tensorload_op);

            auto alloc_op = tensorload_op.getDefiningOp()->getOperand(0);
            comet_errs() << " alloc_op "
                         << "\n";
            comet_vdump(alloc_op);

            if (i < 2 * tensor_rank)
            {
              // indexes crd's
              mlir::Value v = rewriter.create<memref::CastOp>(loc, alloc_op, unrankedMemrefType_index);
              alloc_sizes_cast_vecs[n].push_back(v);
            }
            else
            {
              // NNZ vals
              mlir::Value v = rewriter.create<memref::CastOp>(loc, alloc_op, unrankedMemrefType_f64);
              alloc_sizes_cast_vecs[n].push_back(v);
            }
          }

          auto memrefload_op = tensors[n].getDefiningOp()->getOperand(tensors[n].getDefiningOp()->getNumOperands() - 1);
          allocs_for_sparse_tensors[n].push_back(memrefload_op);
          comet_errs() << " memrefload_op "
                       << "\n";
          comet_vdump(memrefload_op);
        }

        Value last_dim_size_alloc = allocs_for_sparse_tensors[0][0].getDefiningOp()->getOperand(0);
        comet_errs() << "Alloc for last dim size:\n";
        comet_vdump(last_dim_size_alloc);

        mlir::Value sparse_tensor_desc = rewriter.create<memref::CastOp>(loc, last_dim_size_alloc, unrankedMemrefType_index);
        comet_errs() << "Sparse tensor descriptive to extract row/col values:\n";
        comet_vdump(sparse_tensor_desc);

        auto rank_size = (tensors[0].getDefiningOp()->getNumOperands() - 2) / 5;
        std::vector<Value>
            dim_format = mlir::tensorAlgebra::getFormatsValueInt(formats_str, rank_size, rewriter, loc, i32Type);

        if (rank_size == 2)
        { // 2D
          auto transpose2DF64Func = FunctionType::get(ctx, {i32Type, i32Type, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_f64, i32Type, i32Type, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_f64, unrankedMemrefType_index},
                                                      {});

          std::string print_transpose_2D_f64Str = "transpose_2D_f64";

          if (isFuncInMod(print_transpose_2D_f64Str, module) == false)
          {
            transpose_func = FuncOp::create(loc, print_transpose_2D_f64Str, transpose2DF64Func, ArrayRef<NamedAttribute>{});
            transpose_func.setPrivate();
            module.push_back(transpose_func);
          }

          rewriter.create<mlir::CallOp>(loc, print_transpose_2D_f64Str, SmallVector<Type, 2>{},
                                        ValueRange{dim_format[0], dim_format[1], alloc_sizes_cast_vecs[0][0],
                                                   alloc_sizes_cast_vecs[0][1], alloc_sizes_cast_vecs[0][2],
                                                   alloc_sizes_cast_vecs[0][3], alloc_sizes_cast_vecs[0][4],
                                                   dim_format[0], dim_format[1], alloc_sizes_cast_vecs[1][0],
                                                   alloc_sizes_cast_vecs[1][1], alloc_sizes_cast_vecs[1][2],
                                                   alloc_sizes_cast_vecs[1][3], alloc_sizes_cast_vecs[1][4],
                                                   sparse_tensor_desc});
        }
        else if (rank_size == 3)
        { // 3D
          auto transpose3DF64Func = FunctionType::get(ctx, {i32Type, i32Type, i32Type, i32Type, i32Type, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_f64, i32Type, i32Type, i32Type, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_index, unrankedMemrefType_f64, unrankedMemrefType_index},
                                                      {});

          std::string print_transpose_3D_f64Str = "transpose_3D_f64";
          if (isFuncInMod(print_transpose_3D_f64Str, module) == false)
          {
            transpose_func = FuncOp::create(loc, print_transpose_3D_f64Str, transpose3DF64Func, ArrayRef<NamedAttribute>{});
            transpose_func.setPrivate();
            module.push_back(transpose_func);
          }

          rewriter.create<mlir::CallOp>(loc, print_transpose_3D_f64Str, SmallVector<Type, 2>{},
                                        ValueRange{input_perm_num, output_perm_num, dim_format[0], dim_format[1], dim_format[2],
                                                   alloc_sizes_cast_vecs[0][0], alloc_sizes_cast_vecs[0][1],
                                                   alloc_sizes_cast_vecs[0][2], alloc_sizes_cast_vecs[0][3],
                                                   alloc_sizes_cast_vecs[0][4], alloc_sizes_cast_vecs[0][5],
                                                   alloc_sizes_cast_vecs[0][6], dim_format[0], dim_format[1],
                                                   dim_format[2], alloc_sizes_cast_vecs[1][0], alloc_sizes_cast_vecs[1][1],
                                                   alloc_sizes_cast_vecs[1][2], alloc_sizes_cast_vecs[1][3],
                                                   alloc_sizes_cast_vecs[1][4], alloc_sizes_cast_vecs[1][5], alloc_sizes_cast_vecs[1][6],
                                                   sparse_tensor_desc});
        }
        else
        {
          assert(false && "ERROR: Tensors greater than 3 are not currently supported.\n");
        }

        rewriter.eraseOp(setOp);
        rewriter.eraseOp(op);
        return success();

      } // end else sparse tensor

    }; // Tensor TransposeLowering

    struct TransposeLoweringPass
        : public PassWrapper<TransposeLoweringPass, FunctionPass>
    {
      void runOnFunction() final;
    };
  };
}; // end anonymous namespace.

void TensorTransposeLowering::TransposeLoweringPass::runOnFunction()
{
  ConversionTarget target(getContext());
  target.addLegalDialect<LinalgDialect, StandardOpsDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect>();
  OwningRewritePatternList patterns(&getContext());
  patterns.insert<TensorTransposeLowering>(&getContext());

  if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
  {
    signalPassFailure();
  }
}

// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::tensorAlgebra::createTransposeLoweringPass()
{
  return std::make_unique<TensorTransposeLowering::TransposeLoweringPass>();
}