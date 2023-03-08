//===- TensorOpsLowering.cpp------===//
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
// This file implements the lowering of some TA operations such as elementwise tensor addition, substract, reduce, etc.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"
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
// #ifndef DEBUG_MODE_TensorOpsLoweringPass
// #define DEBUG_MODE_TensorOpsLoweringPass
// #endif

#ifdef DEBUG_MODE_TensorOpsLoweringPass
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
// TensorOps lowering RewritePatterns
//===----------------------------------------------------------------------===//
namespace
{
  struct ConstantOpLowering : public OpRewritePattern<tensorAlgebra::DenseConstantOp>
  {
    using OpRewritePattern<tensorAlgebra::DenseConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensorAlgebra::DenseConstantOp op,
                                  PatternRewriter &rewriter) const final
    {
      comet_debug() << "ConstantOpLowering starts\n";
      // auto module = op->getParentOfType<ModuleOp>();
      DenseElementsAttr constantValue = op.value();
      Location loc = op.getLoc();

      tensorAlgebra::TensorSetOp setnewop;
      bool user_setOp = false;
      for (auto u : op.getOperation()->getResult(0).getUsers())
      {
        if (isa<tensorAlgebra::TensorSetOp>(u))
        {
          setnewop = cast<tensorAlgebra::TensorSetOp>(u);
          user_setOp = true;
        }
      }

      // When lowering the constant operation, we allocate and assign the constant
      // values to a corresponding memref allocation.
      auto tensorType = op.getType().cast<TensorType>();
      auto memRefType = convertTensorToMemRef(tensorType);

      comet_debug() << "User_setop: " << user_setOp << "/n";
      Value alloc;
      // TODO(gkestor): the following code should be checked with tensor operations
      //  if (user_setOp)
      //  {
      //    mlir::Value rhs = setnewop.rhs();
      //    comet_debug() << "RHS:";
      //    comet_vdump(rhs);
      //    auto LabeledTensoroperands = rhs.getDefiningOp()->getOperands();
      //    comet_vdump(LabeledTensoroperands[0]);

      //   auto tensorload = cast<memref::TensorLoadOp>(LabeledTensoroperands[0].getDefiningOp());
      //   alloc = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
      // }
      // else
      // {
      alloc = rewriter.create<memref::AllocOp>(loc, memRefType);
      //}

      // We will be generating constant indices up-to the largest dimension.
      // Create these constants up-front to avoid large amounts of redundant
      // operations.
      auto valueShape = memRefType.getShape();
      SmallVector<Value, 8> constantIndices;

      if (!valueShape.empty())
      {
        for (auto i : llvm::seq<int64_t>(
                 0, *std::max_element(valueShape.begin(), valueShape.end())))
          constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
      }
      else
      {
        // This is the case of a tensor of rank 0.
        constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
      }

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
              loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
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
      // rewriter.replaceOp(op, alloc);
      op.replaceAllUsesWith(alloc);
      rewriter.eraseOp(op);

      comet_debug() << "ConstantOpLowering ends\n";
      return success();
    }
  };

  // //===----------------------------------------------------------------------===//
  // // Lowering elementwise tensor addition
  // //===----------------------------------------------------------------------===//
  // struct TensorAdditionLowering : public OpRewritePattern<tensorAlgebra::TensorAddOp>
  // {
  //   using OpRewritePattern<tensorAlgebra::TensorAddOp>::OpRewritePattern;

  //   LogicalResult matchAndRewrite(tensorAlgebra::TensorAddOp op,
  //                                 PatternRewriter &rewriter) const final
  //   {
  //     comet_debug() << "TensorAdditionLowering starts\n";
  //     Location loc = op.getLoc();
  //     Value lhs = op.rhs1();
  //     Value rhs = op.rhs2();

  //     assert(rhs.getType().isa<mlir::TensorType> && rhs.getType().isa<mlir::TensorType> &&
  //            "No support for mixed/sparse tensor addition");

  //     auto rhsTy = rhs.getType().cast<mlir::TensorType>();

  //     std::vector<scf::ForOp> forloops;
  //     Value upperBound;
  //     std::vector<Value> InductionVars;
  //     for (int i = 0; i < rhsTy.getRank(); i++)
  //     {
  //       upperBound = rewriter.create<ConstantIndexOp>(loc, rhsTy.getDimSize(i));
  //       auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
  //       auto step = rewriter.create<ConstantIndexOp>(loc, 1);
  //       auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

  //       comet_vdump(upperBound);
  //       comet_vdump(lowerBound);
  //       comet_vdump(loop);

  //       forloops.push_back(loop);
  //       InductionVars.push_back(loop.getInductionVar());
  //       rewriter.setInsertionPointToStart(loop.getBody());
  //     }

  //     comet_vdump(rhs);
  //     comet_vdump(lhs);
  //     auto lhs_def_op = lhs.getDefiningOp();
  //     memref::AllocOp alloc_lhs;
  //     if (isa<tensorAlgebra::LabeledTensorOp>(lhs_def_op))
  //     {
  //       auto LabeledTensoroperands_lhs = lhs.getDefiningOp()->getOperands();
  //       auto tensorload_lhs = cast<memref::TensorLoadOp>(LabeledTensoroperands_lhs[0].getDefiningOp());
  //       alloc_lhs = cast<memref::AllocOp>(tensorload_lhs->getOperand(0).getDefiningOp());
  //     }
  //     else
  //     {
  //       Operation *tensorload_lhs = cast<memref::TensorLoadOp>(lhs.getDefiningOp());
  //       comet_pdump(tensorload_lhs);
  //       alloc_lhs = cast<memref::AllocOp>(tensorload_lhs->getOperand(0).getDefiningOp());
  //     }

  //     comet_debug() << "LHS alloc\n";
  //     comet_vdump(alloc_lhs);

  //     Operation *tensorload_rhs = cast<memref::TensorLoadOp>(rhs.getDefiningOp());
  //     auto alloc_rhs = cast<memref::AllocOp>(tensorload_rhs->getOperand(0).getDefiningOp());

  //     comet_debug() << "RHS alloc\n";
  //     comet_vdump(alloc_rhs);

  //     auto load_lhs = rewriter.create<memref::LoadOp>(loc, alloc_lhs, InductionVars);
  //     auto load_rhs = rewriter.create<memref::LoadOp>(loc, alloc_rhs, InductionVars);
  //     auto sum = rewriter.create<mlir::AddFOp>(loc, load_lhs, load_rhs);

  //     tensorAlgebra::TensorSetOp setnewop;
  //     for (auto u : op.getOperation()->getResult(0).getUsers())
  //     {
  //       if (isa<tensorAlgebra::TensorSetOp>(u))
  //       {
  //         setnewop = cast<tensorAlgebra::TensorSetOp>(u);
  //       }
  //       else
  //       {
  //         llvm::errs() << __FILE__ << " " << __LINE__ << "Dense constant op is used in ops besides SetOp\n";
  //       }
  //     }

  //     mlir::Value setOp_rhs = setnewop.rhs();
  //     auto def_setOp_rhs = setOp_rhs.getDefiningOp();

  //     memref::AllocOp SetOp_rhs_alloc;

  //     if (isa<tensorAlgebra::LabeledTensorOp>(def_setOp_rhs))
  //     {
  //       auto LTSetOp_rhs = setOp_rhs.getDefiningOp()->getOperands();
  //       auto tensorload_SetOp_rhs = cast<memref::TensorLoadOp>(LTSetOp_rhs[0].getDefiningOp());
  //       SetOp_rhs_alloc = cast<memref::AllocOp>(tensorload_SetOp_rhs->getOperand(0).getDefiningOp());
  //     }
  //     else
  //     {
  //       Operation *tensorload_setOp = cast<memref::TensorLoadOp>(setOp_rhs.getDefiningOp());
  //       SetOp_rhs_alloc = cast<memref::AllocOp>(tensorload_setOp->getOperand(0).getDefiningOp());
  //     }

  //     rewriter.create<memref::StoreOp>(loc, sum, SetOp_rhs_alloc, InductionVars);
  //     rewriter.setInsertionPointAfter(forloops[0]);
  //     rewriter.eraseOp(setnewop);
  //     rewriter.eraseOp(op);

  //     comet_debug() << "TensorAdditionLowering ends\n";
  //     return success();
  //   }
  // };

  // //===----------------------------------------------------------------------===//
  // // Lowering elementwise tensor substraction
  // //===----------------------------------------------------------------------===//
  // struct TensorSubtractionLowering : public OpRewritePattern<tensorAlgebra::TensorSubstractOp>
  // {
  //   using OpRewritePattern<tensorAlgebra::TensorSubstractOp>::OpRewritePattern;

  //   LogicalResult matchAndRewrite(tensorAlgebra::TensorSubstractOp op,
  //                                 PatternRewriter &rewriter) const final
  //   {

  //     Location loc = op.getLoc();
  //     Value lhs = op.rhs1();
  //     Value rhs = op.rhs2();

  //     assert(rhs.getType().isa<mlir::TensorType> && rhs.getType().isa<mlir::TensorType> &&
  //            "No support for mixed/sparse tensor substraction");

  //     auto rhsTy = rhs.getType().cast<mlir::TensorType>();
  //     std::vector<scf::ForOp> forloops;
  //     Value upperBound;
  //     std::vector<Value> InductionVars;
  //     for (int i = 0; i < rhsTy.getRank(); i++)
  //     {
  //       upperBound = rewriter.create<ConstantIndexOp>(loc, rhsTy.getDimSize(i));
  //       auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
  //       auto step = rewriter.create<ConstantIndexOp>(loc, 1);
  //       auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

  //       forloops.push_back(loop);
  //       InductionVars.push_back(loop.getInductionVar());
  //       rewriter.setInsertionPointToStart(loop.getBody());
  //     }

  //     auto lhs_def_op = lhs.getDefiningOp();
  //     memref::AllocOp alloc_lhs;
  //     if (isa<tensorAlgebra::LabeledTensorOp>(lhs_def_op))
  //     {

  //       auto LabeledTensoroperands_lhs = lhs.getDefiningOp()->getOperands();
  //       auto tensorload_lhs = cast<memref::TensorLoadOp>(LabeledTensoroperands_lhs[0].getDefiningOp());
  //       alloc_lhs = cast<memref::AllocOp>(tensorload_lhs->getOperand(0).getDefiningOp());
  //     }
  //     else
  //     {
  //       Operation *tensorload_lhs = cast<memref::TensorLoadOp>(lhs.getDefiningOp());
  //       alloc_lhs = cast<memref::AllocOp>(tensorload_lhs->getOperand(0).getDefiningOp());
  //     }

  //     Operation *tensorload_rhs = cast<memref::TensorLoadOp>(rhs.getDefiningOp());
  //     auto alloc_rhs = cast<memref::AllocOp>(tensorload_rhs->getOperand(0).getDefiningOp());

  //     auto load_lhs = rewriter.create<memref::LoadOp>(loc, alloc_lhs, InductionVars);
  //     auto load_rhs = rewriter.create<memref::LoadOp>(loc, alloc_rhs, InductionVars);
  //     auto subtract = rewriter.create<mlir::SubFOp>(loc, load_lhs, load_rhs);

  //     tensorAlgebra::TensorSetOp setnewop;
  //     for (auto u : op.getOperation()->getResult(0).getUsers())
  //     {
  //       if (isa<tensorAlgebra::TensorSetOp>(u))
  //       {
  //         setnewop = cast<tensorAlgebra::TensorSetOp>(u);
  //       }
  //       else
  //       {
  //         llvm::errs() << __FILE__ << " " << __LINE__ << "Dense constant op is used in ops besides SetOp\n";
  //       }
  //     }

  //     mlir::Value setOp_rhs = setnewop.rhs();
  //     auto def_setOp_rhs = setOp_rhs.getDefiningOp();

  //     memref::AllocOp SetOp_rhs_alloc;

  //     if (isa<tensorAlgebra::LabeledTensorOp>(def_setOp_rhs))
  //     {
  //       auto LTSetOp_rhs = setOp_rhs.getDefiningOp()->getOperands();
  //       auto tensorload_SetOp_rhs = cast<memref::TensorLoadOp>(LTSetOp_rhs[0].getDefiningOp());
  //       SetOp_rhs_alloc = cast<memref::AllocOp>(tensorload_SetOp_rhs->getOperand(0).getDefiningOp());
  //     }
  //     else
  //     {
  //       Operation *tensorload_setOp = cast<memref::TensorLoadOp>(setOp_rhs.getDefiningOp());
  //       SetOp_rhs_alloc = cast<memref::AllocOp>(tensorload_setOp->getOperand(0).getDefiningOp());
  //     }

  //     rewriter.create<memref::StoreOp>(loc, subtract, SetOp_rhs_alloc, InductionVars);

  //     rewriter.setInsertionPointAfter(forloops[0]);
  //     rewriter.eraseOp(setnewop);
  //     rewriter.eraseOp(op);
  //     return success();
  //   }
  // };

  //===----------------------------------------------------------------------===//
  // Lowering dense and sparse tensor transpose
  //===----------------------------------------------------------------------===//
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
      Location loc = op.getLoc();
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
        comet_debug() << "Dense transpose\n";
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
            // TODO(gkestor): this following code block might be needed if we reintroduce label_tensor
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

        comet_vdump(lhs);
        auto outputMemref = lhs.getDefiningOp()->getOperand(0);
        auto copyOp = rewriter.create<linalg::CopyOp>(loc, inputMemref, outputMemref, inputPermuation, outputPermuation);
        comet_vdump(copyOp);
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
        std::string formats_strIn(opFormatsArrayAttr[0].cast<mlir::StringAttr>().getValue());
        std::string formats_strOut(opFormatsArrayAttr[1].cast<mlir::StringAttr>().getValue());
        IntegerType i32Type = IntegerType::get(ctx, 32);
        IndexType indexType = IndexType::get(ctx);
        FloatType f64Type = FloatType::getF64(ctx);

        Value input_perm_num = rewriter.create<mlir::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(pnum[0]));
        Value output_perm_num = rewriter.create<mlir::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(pnum[1]));

        Type unrankedMemrefType_f64 = UnrankedMemRefType::get(f64Type, 0);
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
          // TODO(gkestor): get tensor ranks by functions
          unsigned int tensor_rank = (tensors[n].getDefiningOp()->getNumOperands() - 2) / 5;
          comet_debug() << " tensor_rank: " << tensor_rank << "\n";
          comet_debug() << " tensor[n]: "
                        << "\n";
          comet_pdump(tensors[n].getDefiningOp());

          for (unsigned int i = 0; i < 2 * tensor_rank + 1; i++)
          {
            auto tensorload_op = tensors[n].getDefiningOp()->getOperand(i);
            comet_debug() << " tensorload_op "
                          << "\n";
            comet_vdump(tensorload_op);

            auto alloc_op = tensorload_op.getDefiningOp()->getOperand(0);
            comet_debug() << " alloc_op "
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
          comet_debug() << " memrefload_op "
                        << "\n";
          comet_vdump(memrefload_op);
        }

        Value last_dim_size_alloc = allocs_for_sparse_tensors[0][0].getDefiningOp()->getOperand(0);
        comet_debug() << "Alloc for last dim size:\n";
        comet_vdump(last_dim_size_alloc);

        mlir::Value sparse_tensor_desc = rewriter.create<memref::CastOp>(loc, last_dim_size_alloc, unrankedMemrefType_index);
        comet_debug() << "Sparse tensor descriptive to extract row/col values:\n";
        comet_vdump(sparse_tensor_desc);

        // TODO(gkestor): get tensor ranks by functions
        auto rank_size = (tensors[0].getDefiningOp()->getNumOperands() - 2) / 5;

        // dim format of input tensor
        std::vector<Value>
            dim_formatIn = mlir::tensorAlgebra::getFormatsValueInt(formats_strIn, rank_size, rewriter, loc, i32Type);

        // dim format of output tensor
        std::vector<Value>
            dim_formatOut = mlir::tensorAlgebra::getFormatsValueInt(formats_strOut, rank_size, rewriter, loc, i32Type);

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
                                        ValueRange{dim_formatIn[0], dim_formatIn[1], alloc_sizes_cast_vecs[0][0],
                                                   alloc_sizes_cast_vecs[0][1], alloc_sizes_cast_vecs[0][2],
                                                   alloc_sizes_cast_vecs[0][3], alloc_sizes_cast_vecs[0][4],
                                                   dim_formatOut[0], dim_formatOut[1], alloc_sizes_cast_vecs[1][0],
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
                                        ValueRange{input_perm_num, output_perm_num, dim_formatIn[0], dim_formatIn[1], dim_formatIn[2],
                                                   alloc_sizes_cast_vecs[0][0], alloc_sizes_cast_vecs[0][1],
                                                   alloc_sizes_cast_vecs[0][2], alloc_sizes_cast_vecs[0][3],
                                                   alloc_sizes_cast_vecs[0][4], alloc_sizes_cast_vecs[0][5],
                                                   alloc_sizes_cast_vecs[0][6], dim_formatOut[0], dim_formatOut[1],
                                                   dim_formatOut[2], alloc_sizes_cast_vecs[1][0], alloc_sizes_cast_vecs[1][1],
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
    }   // Tensor TransposeLowering
  };

  //===----------------------------------------------------------------------===//
  // ReduceOptoSCF RewritePatterns: Reduction operation lowering for sparse and dense tensors
  //===----------------------------------------------------------------------===//
  struct ReduceOpLowering : public OpRewritePattern<tensorAlgebra::ReduceOp>
  {
    using OpRewritePattern<tensorAlgebra::ReduceOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tensorAlgebra::ReduceOp op,
                                  PatternRewriter &rewriter) const final
    {

      assert(isa<tensorAlgebra::ReduceOp>(op));
      comet_debug() << "Lowering Reduce operation to SCF\n";

      Location loc = op.getLoc();
      auto f64Type = rewriter.getF64Type();
      auto inputType = op->getOperand(0).getType();

      // Allocate memory for the result and initialized it
      auto cst_zero = rewriter.create<ConstantIndexOp>(loc, 0); // need to access res alloc
      MemRefType memTy_alloc_res = MemRefType::get({1}, f64Type);
      Value res = rewriter.create<memref::AllocOp>(loc, memTy_alloc_res);
      Value const_f64_0 = rewriter.create<mlir::ConstantOp>(loc, f64Type, rewriter.getF64FloatAttr(0));
      std::vector<Value> alloc_zero_loc = {cst_zero};
      rewriter.create<memref::StoreOp>(loc, const_f64_0,
                                       res, alloc_zero_loc);

      comet_vdump(res);

      if (inputType.isa<TensorType>())
      { // tensor is dense
        comet_debug() << "Input Tensor is dense\n";
        std::vector<Value> indices;
        auto alloc_op = op->getOperand(0).getDefiningOp()->getOperand(0);

        comet_vdump(alloc_op);

        for (unsigned rank = 0; rank < inputType.cast<mlir::TensorType>().getRank(); rank++)
        {
          auto dimSize = inputType.cast<mlir::TensorType>().getDimSize(rank);
          Value upperBound;
          if (dimSize == ShapedType::kDynamicSize)
          {
            comet_debug() << " This dimension is a dynamic size\n";

            comet_vdump(alloc_op);
            auto memRefType = alloc_op.getType().dyn_cast<MemRefType>();
            unsigned dynamicDimPos = memRefType.getDynamicDimIndex(rank);
            comet_debug() << " dynamicDimPos: " << dynamicDimPos << "\n";
            upperBound = alloc_op.getDefiningOp()->getOperand(dynamicDimPos);
          }
          else
          {
            upperBound = rewriter.create<ConstantIndexOp>(loc, dimSize);
          }
          auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
          auto step = rewriter.create<ConstantIndexOp>(loc, 1);
          // create for loops
          auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
          indices.push_back(loop.getInductionVar());
          rewriter.setInsertionPointToStart(loop.getBody());
        }

        // Build loop body
        auto load_rhs = rewriter.create<memref::LoadOp>(loc, alloc_op, indices);
        auto res_load = rewriter.create<memref::LoadOp>(loc, res, alloc_zero_loc);
        auto reduced = rewriter.create<mlir::AddFOp>(loc, load_rhs, res_load);
        rewriter.create<memref::StoreOp>(loc, reduced, res, alloc_zero_loc);
      }
      else
      { // sparse tensor type
        assert(inputType.isa<SparseTensorType>());
        comet_debug() << "Input Tensor is sparse\n";

        comet_pdump(op);
        assert(isa<tensorAlgebra::SparseTensorConstructOp>(op->getOperand(0).getDefiningOp()));

        // TODO(gkestor): get tensor ranks by functions
        int tensorRanks = (op->getOperand(0).getDefiningOp()->getNumOperands() - 2) / 5;
        comet_debug() << " tensorRank: " << tensorRanks << " \n";
        comet_debug() << "Tensor to reduce:\n";
        comet_pdump(op->getOperand(0).getDefiningOp());

        // TODO(gkestor): need a better way to acces information from SparseTensorConstructOp
        //  create the lowerBound, upperbound and step for loop
        int indexValueSize = (tensorRanks * 4) + 1; // 4 corresponding to pos, crd, crd_size, pos_size
        comet_debug() << "indexValueSize in SparseTensorConstructOp:" << indexValueSize << "\n";

        auto loadOpForNNZ = op->getOperand(0).getDefiningOp()->getOperand(indexValueSize);
        comet_debug() << "Corresponding AllocOp from SparseTensorConstructOp:\n";
        comet_vdump(loadOpForNNZ);
        auto memAllocForNNZ = loadOpForNNZ.getDefiningOp()->getOperand(0);
        comet_debug() << "Corresponding MemAllocOp for NNZ:\n";
        comet_vdump(memAllocForNNZ);

        MemRefType resultMemTy = memAllocForNNZ.getDefiningOp()->getResult(0).getType().cast<MemRefType>();
        auto memRefRank = resultMemTy.getRank();
        comet_debug() << "memRefRank for alloc: " << memRefRank << "\n";
        assert(memRefRank == 1); // Memref rank should be 1

        auto memRefDimSize = resultMemTy.getDimSize(memRefRank - 1);
        comet_debug() << "memRefDimSize for alloc: " << memRefDimSize << "\n";

        Value upperBound;
        if (memRefDimSize == 1) // size of value array comes from temporary sparse tensor and Dimsize of alloc is one
        {
          upperBound = rewriter.create<memref::LoadOp>(loc, memAllocForNNZ, alloc_zero_loc);
        }
        else
        {
          // size of value array comes from read_input_sizes_2D_f64, and alloc dimsize can be only expected size
          auto expectedMemRefSize = tensorRanks * 2 + tensorRanks + 1; //"2" is corresponding the pos, crd per dimension, "1" is for value array
          assert(memRefDimSize == expectedMemRefSize);
          upperBound = op->getOperand(0).getDefiningOp()->getOperand(indexValueSize);
        }
        comet_debug() << "Upper Bound:\n";
        comet_vdump(upperBound);
        auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);

        // create for loops
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        auto insertPt = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(loop.getBody());

        // Build loop body
        int indexValuePtr = (tensorRanks * 2); // 2 corresponding to pos, crd
        auto alloc_op = op->getOperand(0).getDefiningOp()->getOperand(indexValuePtr).getDefiningOp()->getOperand(0);
        comet_debug() << " ValueAllocOp";
        comet_vdump(alloc_op);
        std::vector<Value> indices = {loop.getInductionVar()};
        auto load_rhs = rewriter.create<memref::LoadOp>(loc, alloc_op, indices);
        auto res_load = rewriter.create<memref::LoadOp>(loc, res, alloc_zero_loc);
        auto reduce = rewriter.create<mlir::AddFOp>(loc, load_rhs, res_load);
        rewriter.create<memref::StoreOp>(loc, reduce, res, alloc_zero_loc);

        // need to restore the insertion point to the previous point
        rewriter.restoreInsertionPoint(insertPt);
        comet_vdump(loop);
      }

      // Important to replace all uses of this operation with the new one, otherwise, the current op won't be lowered.
      op.replaceAllUsesWith(res);
      rewriter.eraseOp(op);

      return success();
    }
  }; // ReduceOpLowering

  struct ScalarOpsLowering : public OpRewritePattern<tensorAlgebra::ScalarOp>
  {
    using OpRewritePattern<tensorAlgebra::ScalarOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tensorAlgebra::ScalarOp op,
                                  PatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::ScalarOp>(op));
      comet_debug() << "ScalarOpsLowering starts\n";

      // auto module = op->getParentOfType<ModuleOp>();

      // Scalar operation could be between
      //  1. two tensors (size of 1) and the output with a tensor of size 1
      //  2. two F64 values and the output will be F64
      Location loc = op.getLoc();
      Value rhs = op.rhs();
      Value lhs = op.lhs();

      comet_vdump(rhs);
      comet_vdump(lhs);

      auto rhsType = op->getOperand(0).getType();
      auto lhsType = op->getOperand(1).getType();

      auto f64Type = rewriter.getF64Type();
      Type unrankedMemrefType_f64 = UnrankedMemRefType::get(f64Type, 0);
      Value const_index_0 = rewriter.create<ConstantIndexOp>(loc, 0);
      comet_vdump(const_index_0);
      std::vector<Value> alloc_zero_loc = {const_index_0};

      if (rhsType.isa<MemRefType>())
      {
        comet_debug() << "RHS is a tensor\n";
        rhs = rewriter.create<memref::LoadOp>(loc, rhs, alloc_zero_loc);
        comet_vdump(rhs);
      }
      if (lhsType.isa<MemRefType>())
      {
        comet_debug() << "LHS is a tensor\n";
        lhs = rewriter.create<memref::LoadOp>(loc, lhs, alloc_zero_loc);
      }

      assert((rhsType.isF64() || rhsType.isa<MemRefType>()) && (lhsType.isF64() || lhsType.isa<MemRefType>()) && "Scalar Operands data type must be either F64 or memref");

      Value res;
      bool res_comes_from_setop = false;
      for (auto u : op.getOperation()->getResult(0).getUsers())
      {
        comet_debug() << "Users:\n";
        comet_pdump(u);
        if (isa<tensorAlgebra::TensorSetOp>(u))
        {
          res = cast<tensorAlgebra::TensorSetOp>(u).getOperation()->getOperand(1);
          comet_debug() << "Result from SetOp:\n";
          comet_vdump(res);
          res_comes_from_setop = true;
          break;
        }
      }

      assert(res_comes_from_setop && "SetOp is needed to assign the scalar operation result to final variable");

      comet_debug() << "Scalar lowering Final step\n";
      comet_debug() << "RHS, LHS and result:\n";
      comet_vdump(rhs);
      comet_vdump(lhs);
      comet_vdump(res);

      // Op rhs and lhs
      auto arith_op_attr = op.opAttr();
      std::string op_attr(arith_op_attr.getValue());
      comet_debug() << "aritmetic op: " << op_attr << "\n";
      Value res_val;
      if (op_attr.compare("+") == 0)
      {
        res_val = rewriter.create<mlir::AddFOp>(loc, rhs, lhs);
      }
      else if (op_attr.compare("-") == 0)
      {
        res_val = rewriter.create<mlir::SubFOp>(loc, lhs, rhs);
      }
      else if (op_attr.compare("*") == 0)
      {
        res_val = rewriter.create<mlir::MulFOp>(loc, rhs, lhs);
      }
      else if (op_attr.compare("/") == 0)
      {
        res_val = rewriter.create<mlir::DivFOp>(loc, lhs, rhs);
      }
      else
      {
        assert(false && "Unsuported Operation\n");
      }

      comet_vdump(res_val);
      // store res_val to res
      auto storeOp = rewriter.create<memref::StoreOp>(loc, res_val, res, alloc_zero_loc);
      comet_vdump(storeOp);

      op.replaceAllUsesWith(res);
      rewriter.eraseOp(op);
      return success();
    }
  }; // ScalarOpsLowering

} // end anonymous namespace.

/// This is a partial lowering to linear algebra of the tensor algebra operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the TA dialect.
namespace
{
  struct TensorOpsLoweringPass
      : public PassWrapper<TensorOpsLoweringPass, FunctionPass>
  {
    void runOnFunction() final;
  };
} // end anonymous namespace.

void TensorOpsLoweringPass::runOnFunction()
{
  auto function = getFunction();

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
  target.addLegalDialect<LinalgDialect,
                         AffineDialect,
                         scf::SCFDialect,
                         StandardOpsDialect,
                         memref::MemRefDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the TA operations.

  OwningRewritePatternList patterns(&getContext());
  patterns.insert<//TensorAdditionLowering,
                  //TensorSubtractionLowering,
                  TensorTransposeLowering,
                  ReduceOpLowering,
                  ScalarOpsLowering,
                  ConstantOpLowering>(&getContext());
  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.

  if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
  {
    signalPassFailure();
  }
}

/// Create a pass for lowering tensor operations in the TensorAlgebra dialect to other lower level dialects
std::unique_ptr<Pass> mlir::tensorAlgebra::createTensorOpsLoweringPass()
{
  return std::make_unique<TensorOpsLoweringPass>();
}
