//===- EarlyLowering.cpp ------===//
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
// This file implements a partial lowering of some of TA operations such as tensor_declaration, sparse tensor declaration, etc.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/IndexTree/IR/ITDialect.h"
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
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"

#include <limits>
#include <map>
#include <set>
#include <unordered_map>

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using namespace mlir::tensorAlgebra;
using namespace mlir::indexTree;

#define DEBUG_TYPE "early-lowering"

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_EarlyLoweringPass
// #define DEBUG_MODE_EarlyLoweringPass
// #endif

#ifdef DEBUG_MODE_EarlyLoweringPass
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
/// Early Lowering Passes for sparse/dense tensor declarations
//===----------------------------------------------------------------------===//
namespace
{

  void mixModeEltWiseMultSparseTensorOutputLowering(Value computeOp, Location loc,
                                                    std::vector<std::vector<int>> rshPerms,
                                                    std::vector<Value> &dimSizes,
                                                    std::vector<Value> &tensorload_sizes_vec,
                                                    std::vector<Value> &array_sizes_vec,
                                                    PatternRewriter &rewriter)
  {

    IndexType indexType = IndexType::get(computeOp.getContext());
    FloatType f64Type = FloatType::getF64(computeOp.getContext());
    auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamicSize}, indexType); // memref<?xindex>
    auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamicSize}, f64Type);     // memref<?xf64>

    comet_errs() << "mixModeEltWiseMultSparseTensorOutputLowering computeOp\n";
    comet_vdump(computeOp);

    //  elementwise mul op in mix sparse dense case
    //  If elementwise, copy sparse input arrays for elementwise mul
    int sparse_inputtensor_id = 0;

    auto rhsComputeOp = computeOp.getDefiningOp()->getOperand(0).getDefiningOp();
    comet_errs() << " SparseTensorConstructOp: ";
    comet_pdump(rhsComputeOp->getOperand(sparse_inputtensor_id).getDefiningOp());
    auto sptensor_construct_op = cast<tensorAlgebra::SparseTensorConstructOp>(rhsComputeOp->getOperand(sparse_inputtensor_id).getDefiningOp());

    for (unsigned int i = 0; i < 2 * (rshPerms[sparse_inputtensor_id].size()) + 1; i++)
    {
      comet_errs() << " in for loop\n";
      Value intput_tensorload_op = cast<memref::TensorLoadOp>(sptensor_construct_op.getOperand(i).getDefiningOp());
      Value input_alloc_op = cast<memref::AllocOp>(intput_tensorload_op.getDefiningOp()->getOperand(0).getDefiningOp());
      comet_errs() << " AllocOp: ";
      comet_vdump(input_alloc_op);

      comet_errs() << " ";
      Value input_alloc_op_param = input_alloc_op.getDefiningOp()->getOperand(0);
      comet_errs() << " ";

      Value output_alloc_op;
      if (i < 2 * (rshPerms[sparse_inputtensor_id].size()))
      {
        // Memory allocation for position and coordinate arrays in sparse tensor contractions
        output_alloc_op = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{input_alloc_op_param}, rewriter);
      }
      else
      {
        // Memory allocation for value array in sparse tensor contractions
        output_alloc_op = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{input_alloc_op_param}, rewriter); // Cval array
        comet_errs() << " AllocOp: ";
        comet_vdump(output_alloc_op);
      }

      Value output_tensorload_op = rewriter.create<memref::TensorLoadOp>(loc, output_alloc_op);
      tensorload_sizes_vec.push_back(output_tensorload_op);
    }
    comet_errs() << " ";

    // [0...2d, 2d+1...4d+1, 4d+2...5d+1]
    for (unsigned int i = 0; i < 2 * (rshPerms[sparse_inputtensor_id].size()) + 1; i++)
    {
      int sizes_i = i + 2 * (rshPerms[sparse_inputtensor_id].size()) + 1;
      comet_errs() << " ";
      comet_pdump(sptensor_construct_op.getOperand(sizes_i).getDefiningOp());

      Value input_load_op = sptensor_construct_op.getOperand(sizes_i);
      comet_errs() << " ";
      comet_vdump(input_load_op);
      array_sizes_vec.push_back(input_load_op);
    }

    for (unsigned int i = 0; i < rshPerms[sparse_inputtensor_id].size(); i++)
    {
      int sizes_i = i + 2 * (2 * (rshPerms[sparse_inputtensor_id].size()) + 1);
      comet_errs() << " ";
      comet_pdump(sptensor_construct_op.getOperand(sizes_i).getDefiningOp());

      Value input_load_op = sptensor_construct_op.getOperand(sizes_i);
      comet_errs() << " ";
      comet_vdump(input_load_op);
      dimSizes.push_back(input_load_op);
    }
  }

  void pureSparseMultSparseTensorOutputLowering(tensorAlgebra::SparseOutputTensorDeclOp op,
                                                Location loc,
                                                std::string sparseOutputFormat,
                                                std::vector<Value> &dimSizes,
                                                std::vector<Value> &tensorload_sizes_vec,
                                                std::vector<Value> &array_sizes_vec,
                                                PatternRewriter &rewriter)
  {
    comet_errs() << " sparse output is used in itComputeOp op\n";
    comet_errs() << " sparseOutputFormat: " << sparseOutputFormat << "\n";

    IndexType indexType = IndexType::get(op.getContext());
    FloatType f64Type = FloatType::getF64(op.getContext());
    auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamicSize}, indexType); // memref<?xindex>
    auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamicSize}, f64Type);     // memref<?xf64>

    Value cst_index_0 = rewriter.create<mlir::ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(0));
    Value cst_index_1 = rewriter.create<mlir::ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(1));

    unsigned int tensor_rank = op.getOperation()->getNumOperands();

    std::vector<mlir::Value> array_sizes;
    std::vector<mlir::Value> array_sizes_alloc_vec;
    std::vector<mlir::Value> initial_array_sizes;

    if (sparseOutputFormat.compare("CSR") == 0)
    { // CSR format
      comet_errs() << " 2D CSR format in sparse output decl op\n";
      // AllocOp, storeOp, LoadOp
      initial_array_sizes.push_back(cst_index_1);
      initial_array_sizes.push_back(cst_index_1);

      // The other three size information size..
      // get the dimension size from operand
      // std::vector<Value> dim_sizes;
      for (unsigned int i = 0; i < op.getOperation()->getNumOperands(); i++)
      {
        if (isa<tensorAlgebra::IndexLabelOp>(op.getOperation()->getOperand(i).getDefiningOp()))
        {
          Value indexlabelop = dyn_cast<tensorAlgebra::IndexLabelOp>(op.getOperation()->getOperand(i).getDefiningOp());
          dimSizes.push_back(indexlabelop.getDefiningOp()->getOperand(1));
          ;
        }
      }
      // The dim size is the second parameter of the
      Value dim2_posSize = rewriter.create<mlir::AddIOp>(loc, dimSizes[0], cst_index_1);
      comet_errs() << " ";
      comet_vdump(dim2_posSize);
      initial_array_sizes.push_back(dim2_posSize);

      Value dim2_crdSize = rewriter.create<mlir::MulIOp>(loc, dimSizes[0], dimSizes[1]);
      initial_array_sizes.push_back(dim2_crdSize);
      initial_array_sizes.push_back(dim2_crdSize);
      comet_errs() << " ";
      comet_vdump(dim2_crdSize);
    }
    else
    {
      assert(false && "Not supported format\n");
    }

    // same with transpose case
    comet_errs() << " array_sizes_vec.size(): " << array_sizes_vec.size() << "\n";
    comet_errs() << " tensor_rank: " << tensor_rank << "\n";
    std::vector<Value> array_alloc_vec;
    for (unsigned int i = 0; i < 2 * tensor_rank + 1; i++)
    {
      Value alloc_sizes;
      if (i < 2 * tensor_rank)
      {
        comet_errs() << " Inserting AllocOp: ";
        alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{initial_array_sizes[i]}, rewriter);
        comet_errs() << " AllocOp: ";
        comet_vdump(alloc_sizes);
      }
      else
      {
        alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{initial_array_sizes[i]}, rewriter);
        comet_errs() << " AllocOp: ";
        comet_vdump(alloc_sizes);
      }
      Value tensorload_sizes = rewriter.create<memref::TensorLoadOp>(loc, alloc_sizes);
      tensorload_sizes_vec.push_back(tensorload_sizes);
      array_alloc_vec.push_back(alloc_sizes);
    }

    /// Initialize the sizes of pos/crd/val arrays
    array_sizes.push_back(cst_index_1);
    array_sizes.push_back(cst_index_1);
    array_sizes.push_back(cst_index_1);
    array_sizes.push_back(cst_index_0);
    array_sizes.push_back(cst_index_0);
    // put the array sizes into alloc/store/loadOp
    for (auto size : array_sizes)
    {
      MemRefType memTy_alloc_sizes = MemRefType::get({1}, IndexType::get(op.getContext()));
      Value allocop = rewriter.create<memref::AllocOp>(loc, memTy_alloc_sizes);
      rewriter.create<memref::StoreOp>(loc, size, allocop, ValueRange{cst_index_0});
      Value loadop = rewriter.create<memref::LoadOp>(loc, allocop, ValueRange{cst_index_0});
      array_sizes_vec.push_back(loadop);
      array_sizes_alloc_vec.push_back(allocop);
    }

    // initialize C1pos[0] = Cdim1_size;
    rewriter.create<memref::StoreOp>(loc, dimSizes[0], array_alloc_vec[0], ValueRange{cst_index_0});
  }

  struct DenseTensorDeclOpLowering : public OpRewritePattern<tensorAlgebra::DenseTensorDeclOp>
  {
    using OpRewritePattern<tensorAlgebra::DenseTensorDeclOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions and ta.sptensor_construct operation.
     * Step 3: Remove the DenseTensorDeclOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::DenseTensorDeclOp op,
                                  PatternRewriter &rewriter) const final
    {

      comet_errs() << "--------------DenseTensorDeclarationLowering in format begin\n";
      comet_vdump(op);
      auto loc = op.getLoc();

      auto tensor_decl_value = cast<tensorAlgebra::DenseTensorDeclOp>(op);

      auto resultTensorType = op.getResult().getType();
      std::vector<Value> cur_indices;
      std::vector<int64_t> cur_memref;
      auto resultMemTy = convertTensorToMemRef(resultTensorType.cast<TensorType>());

      for (int i = 0; i < resultMemTy.getRank(); i++)
      {
        if (resultMemTy.isDynamicDim(i))
          cur_memref.push_back(ShapedType::kDynamicSize);
        else // The constant dim size must NOT comes from the sparse matrix
          cur_memref.push_back(resultMemTy.getDimSize(i));

        if (isa<tensorAlgebra::IndexLabelOp>(tensor_decl_value.labels()[i].getDefiningOp()))
        {
          comet_vdump(tensor_decl_value.labels()[i]);
          auto label_decl_value = cast<tensorAlgebra::IndexLabelOp>(tensor_decl_value.labels()[i].getDefiningOp());
          auto hi = label_decl_value.max();
          if (resultMemTy.isDynamicDim(i))
            cur_indices.push_back(hi); // IndexCastOp
        }
      }

      // auto alloc_sizes1 = rewriter.create<memref::AllocOp>(loc, resultMemTy, ValueRange(cur_indices));
      // comet_errs() << " ";
      // comet_vdump(alloc_sizes1);

      // alloc_sizes1.getOperation()->setAttr(memref::AllocOp::getAlignmentAttrName(), rewriter.getI64IntegerAttr(32));

      // Value tensorLoad = rewriter.create<memref::TensorLoadOp>(loc, alloc_sizes1);
      // comet_errs() << " ";
      // comet_vdump(tensorLoad);

      // op.replaceAllUsesWith(tensorLoad);
      // rewriter.replaceOp(op, tensorLoad);

      //Check if this tensor is explicitly initialized with ta.fill operation
      bool is_filled = false;
      for (auto u : op->getUsers())
      {
        if (isa<tensorAlgebra::TensorFillOp>(u) || isa<tensorAlgebra::TensorSetOp>(u))
          is_filled = true;
      }

      comet_errs() << " AllocOp for initialization";
      Value init_alloc;
      if (is_filled)
      {
        //if is_filled is true, only allocate memory and let ta.fill initializes tensors
        init_alloc = rewriter.create<memref::AllocOp>(loc, resultMemTy, ValueRange(cur_indices));
      }
      else
      {
        //if is_filled is false, allocate memory and initialize it
        init_alloc = insertAllocAndInitialize(loc, resultMemTy, ValueRange(cur_indices), rewriter);
      }
      init_alloc.getDefiningOp()->setAttr(memref::AllocOp::getAlignmentAttrName(), rewriter.getI64IntegerAttr(32));

      Value tensorLoad = rewriter.create<memref::TensorLoadOp>(loc, init_alloc);
      comet_errs() << " TensorLoad:";
      comet_vdump(tensorLoad);

      op.replaceAllUsesWith(tensorLoad);
      rewriter.replaceOp(op, tensorLoad);
      comet_errs() << "--------------DenseTensorDeclarationLowering in format end\n";
      return success();
    }
  };

  void insertReadFileLibCall(int rank_size, MLIRContext *ctx, ModuleOp &module, FuncOp function)
  {
    comet_errs() << "Inserting insertReadFileLibCall\n";
    FloatType f32Type, f64Type;
    if (VALUETYPE.compare("f32") == 0)
    {
      f32Type = FloatType::getF32(ctx);
    }
    else
    {
      f64Type = FloatType::getF64(ctx);
    }

    IndexType indexType = IndexType::get(function.getContext());
    IntegerType i32Type = IntegerType::get(ctx, 32);
    auto unrankedMemref_f64 = mlir::UnrankedMemRefType::get(f64Type, 0);
    // TODO(gkestor): there is an issue with F32 UnrankedMemRefType
    auto unrankedMemref_f32 = mlir::UnrankedMemRefType::get(f64Type, 0);
    auto unrankedMemref_index = mlir::UnrankedMemRefType::get(indexType, 0);

    if (rank_size == 2)
    {
      comet_errs() << " Rank Size is 2\n";
      auto readInput2DF32Func = FunctionType::get(ctx, {i32Type, indexType, indexType, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_f32}, {});
      auto readInput2DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_f64}, {});

      if (VALUETYPE.compare("f32") == 0)
      {
        if (isFuncInMod("read_input_2D_f32", module) == false)
        {
          FuncOp func1 = FuncOp::create(function.getLoc(), "read_input_2D_f32",
                                        readInput2DF32Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        if (isFuncInMod("read_input_2D_f64", module) == false)
        {
          comet_errs() << " Inserting read_input_2D_f64\n";
          FuncOp func1 = FuncOp::create(function.getLoc(), "read_input_2D_f64",
                                        readInput2DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }

      auto readInputSizes2DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, unrankedMemref_index}, {});

      if (VALUETYPE.compare("f32") == 0)
      {
        if (isFuncInMod("read_input_sizes_2D_f32", module) == false)
        {
          FuncOp func1 = FuncOp::create(function.getLoc(), "read_input_sizes_2D_f32",
                                        readInputSizes2DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        if (isFuncInMod("read_input_sizes_2D_f64", module) == false)
        {
          comet_errs() << " Inserting read_input_sizes_2D_f64\n";
          FuncOp func1 = FuncOp::create(function.getLoc(), "read_input_sizes_2D_f64",
                                        readInputSizes2DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
    }

    // 3D tensor
    else if (rank_size == 3)
    {
      auto readInput3DF32Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_f32}, {});

      auto readInput3DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_f64}, {});

      if (VALUETYPE.compare("f32") == 0)
      {
        if (isFuncInMod("read_input_3D_f32", module) == false)
        {
          FuncOp func1 = FuncOp::create(function.getLoc(), "read_input_3D_f32",
                                        readInput3DF32Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        if (isFuncInMod("read_input_3D_f64", module) == false)
        {
          comet_errs() << " Insert read_input_sizes_3D_f64 decl\n";
          FuncOp func1 = FuncOp::create(function.getLoc(), "read_input_3D_f64",
                                        readInput3DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
          comet_errs() << " Insert read_input_sizes_3D_f64 decl\n";
        }
      }

      auto readInputSizes3DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, unrankedMemref_index}, {});

      if (VALUETYPE.compare("f32") == 0)
      {
        if (isFuncInMod("read_input_sizes_3D_f32", module) == false)
        {
          FuncOp func1 = FuncOp::create(function.getLoc(), "read_input_sizes_3D_f32",
                                        readInputSizes3DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        if (isFuncInMod("read_input_sizes_3D_f64", module) == false)
        {
          comet_errs() << " Insert read_input_sizes_3D_f64 decl\n";
          FuncOp func1 = FuncOp::create(function.getLoc(), "read_input_sizes_3D_f64",
                                        readInputSizes3DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
          comet_errs() << " Insert read_input_sizes_3D_f64 decl finish\n";
        }
      }
    }
    else
    {
      llvm::errs() << __LINE__ << "Not supported dims\n";
    }
  }

  struct SparseInputTensorDeclOpLowering : public OpRewritePattern<tensorAlgebra::SparseTensorDeclOp>
  {
    using OpRewritePattern<tensorAlgebra::SparseTensorDeclOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions and ta.sptensor_construct operation.
     * Step 3: Remove the SparseTensorDeclOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::SparseTensorDeclOp op,
                                  PatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::SparseTensorDeclOp>(op));
      comet_errs() << " SparseInputTensorDeclOpLowering in format begin\n";
      comet_vdump(op);
      mlir::MLIRContext *ctx = rewriter.getContext();
      auto function = cast<FuncOp>(op->getParentOp());
      comet_vdump(function);
      auto module = function.getOperation()->getParentOfType<ModuleOp>();
      // module->dump();

      std::string op_str = dump2str(op);

      bool isOutputTensor = false;

      for (auto u1 : op.getOperation()->getUsers())
      {
        comet_errs() << "\nCheck the tensor is input or output\n";
        comet_pdump(u1);
        if (isa<tensorAlgebra::TensorMultOp>(u1))
        {
          comet_errs() << " used in ta.tc op\n";
          auto p = cast<tensorAlgebra::TensorMultOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            // comet_vdump(n);
            std::string n_str = dump2str(p->getOperand(i));
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_errs() << " FIND IT: " << i << "\n";
              if (i == 2)
              {
                isOutputTensor = true;
              }
            }
          }
        }
        else if (isa<tensorAlgebra::TensorSetOp>(u1))
        {
          comet_errs() << " used in ta.set op\n";
          auto p = cast<tensorAlgebra::TensorSetOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            // comet_vdump(n);
            comet_errs() << " the " << i << "th operand\n";
            std::string n_str = dump2str(p->getOperand(i));
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_errs() << " FIND IT: " << i << "\n";
              if (i == 1)
              {
                // The source tensor of the set op
                isOutputTensor = true;
              }
            }
          }
        }

        else if (isa<tensorAlgebra::TransposeOp>(u1))
        {
          comet_errs() << " used in transpose op\n";
          auto p = cast<tensorAlgebra::TransposeOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            std::string n_str = dump2str(p->getOperand(i));
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_errs() << " FIND IT: " << i << "\n";
              if (i == 1)
              {
                // (ruiqin): output of ta.elews_mul
                isOutputTensor = true;
              }
            }
          }
        }
        else if (isa<indexTree::IndexTreeComputeRHSOp>(u1))
        {
          comet_errs() << " used in ta.itComputeRHS op\n";
          isOutputTensor = false;
        }
        else if (isa<indexTree::IndexTreeComputeLHSOp>(u1))
        {
          comet_errs() << " used in ta.itComputeLHS op\n";
          isOutputTensor = true;
        }
        else if (isa<tensorAlgebra::TensorFillFromFileOp>(u1))
        {
          // do nothing
          comet_errs() << " the tensor is in fill_from_file op\n";
        }
        else if (isa<tensorAlgebra::PrintOp>(u1))
        {
          comet_errs() << " the tensor is in PrintOp\n";
        }
        else if (isa<tensorAlgebra::SUMOp>(u1))
        {
          comet_errs() << " the tensor is in SumOp\n";
        }
        else if (isa<tensorAlgebra::TensorElewsMultOp>(u1))
        {
          comet_errs() << " the tensor is in Elementwise multiplication\n";
        }
        else
        {
          u1->dump();
          assert(false && " the tensor is in not supported operation");
        }
      }

      comet_errs() << " isOutputTensor: " << isOutputTensor << "\n";

      auto loc = op.getLoc();
      StringRef formatsAttr = op.format();
      std::string formats_str(formatsAttr.data());
      comet_errs() << " --- " << formats_str << "\n";

      comet_errs() << " " << op.getNumOperands() << "\n";
      auto rank_size = op.getNumOperands();

      IndexType indexType = IndexType::get(op.getContext());
      FloatType f64Type = FloatType::getF64(op.getContext());
      if (VALUETYPE.compare(0, 3, "f32") == 0)
        f64Type = FloatType::getF32(op.getContext());

      // A1_pos ... A_value
      auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamicSize}, indexType); // memref<?xindex>
      auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamicSize}, f64Type);     // memref<?xf64>

      Type unrankedMemTy_index = UnrankedMemRefType::get(indexType, 0);
      Type unrankedMemTy_f64 = UnrankedMemRefType::get(f64Type, 0);

      comet_errs() << " " << formats_str << " isDense: " << isDense(formats_str, ", ") << "\n";

      if (isDense(formats_str, ", ") == false && isOutputTensor == false)
      {
        comet_errs() << " Sparse input tensor \n";
        // search read_from_file function call to get the input file name
        // Currently, has no filename
        std::string input_filename;
        for (auto u : op.getOperation()->getUsers())
        {
          // Used in LabeledTensorOp and then the LabeledTensorOp is used in TensorChainSetOp
          if (isa<tensorAlgebra::LabeledTensorOp>(u))
          {
            comet_errs() << "\n";
            // comet_pdump(u);
            auto labeledtensorop = cast<tensorAlgebra::LabeledTensorOp>(u);
            LLVM_DEBUG(comet_errs() << " labeled_tensor op\n");
            for (auto u1 : u->getUsers())
            {
              if (isa<tensorAlgebra::TensorChainSetOp>(u1))
              {
                LLVM_DEBUG(comet_errs() << " tensor set_op\n");

                auto setop = cast<tensorAlgebra::TensorChainSetOp>(u1);

                auto read_from_file_operand = setop.getOperand(1).getDefiningOp(); // funccall

                if (isa<tensorAlgebra::GenericCallOp>(read_from_file_operand))
                {
                  auto genericcallop = cast<tensorAlgebra::GenericCallOp>(read_from_file_operand);
                  // comet_vdump(genericcallop);
                  LLVM_DEBUG(comet_errs() << " read_from_file op\n");
                  std::string read_ref(genericcallop.callee().getLeafReference());
                  LLVM_DEBUG(comet_errs() << " read_ref: " << read_ref << "\n");
                  if (read_ref.compare(0, 14, "read_from_file") == 0)
                  {
                    LLVM_DEBUG(comet_errs() << " yes, read_from_file op\n");
                    // get filename through operand
                    comet_errs() << " genericcallop.getNumOperands(): " << genericcallop.getOperation()->getNumOperands() << "\n";

                    // Erase the useless ops
                    rewriter.eraseOp(setop);
                    rewriter.eraseOp(genericcallop);
                    rewriter.eraseOp(labeledtensorop);
                  }
                }
              }
            }
          }
          // Used in TensorFillFromFileOp
          else if (isa<tensorAlgebra::TensorFillFromFileOp>(u))
          {
            // comet_pdump(u);
            comet_errs() << "\n";
            auto fillfromfileop = cast<tensorAlgebra::TensorFillFromFileOp>(u);
            // Can get filename, from "filename" attribute of fillfromfileop

            StringAttr filename = fillfromfileop.filename().cast<StringAttr>();

            comet_errs() << " filename: " << filename.getValue() << "\n";

            std::string filename_str(filename.getValue());
            input_filename = filename_str;
            comet_errs() << " " << input_filename << "\n";

            rewriter.eraseOp(fillfromfileop);
          }
        }

        MemRefType memTy_alloc_sizes = MemRefType::get({3 * rank_size + 1}, indexType);
        Value alloc_sizes = rewriter.create<memref::AllocOp>(loc, memTy_alloc_sizes);
        comet_errs() << " ";
        comet_vdump(alloc_sizes);

        Value alloc_sizes_cast = rewriter.create<memref::CastOp>(loc, alloc_sizes, unrankedMemTy_index);

        std::vector<Value> dim_format = mlir::tensorAlgebra::getFormatsValue(formats_str, rank_size, rewriter, loc, indexType);
        comet_errs() << " Get the dim_format\n";

        // inform the runtime of what env var to use for parsing input file
        IntegerType i32Type = IntegerType::get(op.getContext(), 32);
        Value sparseFileID;
        std::size_t pos = input_filename.find("SPARSE_FILE_NAME");
        if (pos == std::string::npos) // not found
        {
          // currently, reading of file when path of file is provided as arg is not supported at runtime.
          sparseFileID = rewriter.create<mlir::ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, -1));
        }
        // 16 is the length of SPARSE_FILE_NAME
        std::string fileID = input_filename.substr(pos + 16, 1); // this will only catch 0..9
        if (fileID.empty())
        { // SPARSE_FILE_NAME
          sparseFileID = rewriter.create<mlir::ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, 9999));
        }
        else
        { // SPARSE_FILE_NAME{int}
          comet_errs() << " Parsed fileID: " << fileID << "\n";
          int intFileID = std::stoi(fileID);
          sparseFileID = rewriter.create<mlir::ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, intFileID));
        }

        // Now, setup the runtime calls
        if (rank_size == 2)
        { // 2D
          comet_errs() << " 2D\n";
          insertReadFileLibCall(rank_size, ctx, module, function);

          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            std::string read_input_sizes_str = "read_input_sizes_2D_f32";
            auto read_input_sizes_f64Call = rewriter.create<mlir::CallOp>(
                loc, read_input_sizes_str, SmallVector<Type, 2>{}, ValueRange{sparseFileID, dim_format[0], dim_format[1], alloc_sizes_cast});
            read_input_sizes_f64Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
          }
          else
          {
            std::string read_input_sizes_str = "read_input_sizes_2D_f64";
            auto read_input_sizes_f64Call = rewriter.create<mlir::CallOp>(
                loc, read_input_sizes_str, SmallVector<Type, 2>{}, ValueRange{sparseFileID, dim_format[0], dim_format[1], alloc_sizes_cast});
            read_input_sizes_f64Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
            comet_errs() << " \n";
            comet_vdump(read_input_sizes_f64Call);
          }
          comet_errs() << " 2D\n";
        }
        else if (rank_size == 3)
        { // 3D

          comet_errs() << " 3D\n";
          insertReadFileLibCall(rank_size, ctx, module, function);

          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            comet_errs() << "\n";
            std::string read_input_sizes_str = "read_input_sizes_3D_f32";
            auto read_input_sizes_f64Call = rewriter.create<mlir::CallOp>(
                loc, read_input_sizes_str, SmallVector<Type, 2>{}, ValueRange{sparseFileID, dim_format[0], dim_format[1], dim_format[2], alloc_sizes_cast});
            read_input_sizes_f64Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
            comet_errs() << "\n";
          }
          else
          { // default f64
            comet_errs() << "\n";
            std::string read_input_sizes_str = "read_input_sizes_3D_f64";
            comet_errs() << " "
                         << "dim_format.size(): " << dim_format.size() << " \n";
            auto read_input_sizes_f64Call = rewriter.create<mlir::CallOp>(
                loc, read_input_sizes_str, SmallVector<Type, 2>{}, ValueRange{sparseFileID, dim_format[0], dim_format[1], dim_format[2], alloc_sizes_cast});
            read_input_sizes_f64Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
            comet_errs() << "\n";
          }
          comet_errs() << "\n";
        }
        else
        {
          assert(false && " Utility functions to read sparse tensors are supported up to 3 dimensions\n");
        }

        std::vector<Value> array_sizes;
        for (unsigned int i = 0; i < 3 * rank_size + 1; i++)
        { // 2*rank_size + 1 + rank_size
          Value idx = rewriter.create<ConstantIndexOp>(loc, i);
          // sizes_idxes.push_back(idx);
          Value cor = rewriter.create<memref::LoadOp>(loc, alloc_sizes, idx);
          comet_errs() << " ";
          comet_vdump(cor);
          array_sizes.push_back(cor);
        }

        std::vector<Value> alloc_sizes_cast_vec;
        std::vector<Value> alloc_sizes_vec;
        for (unsigned int i = 0; i < 2 * rank_size; i++)
        {
          std::vector<Value> idxes;
          idxes.push_back(array_sizes[i]);
          comet_vdump(array_sizes[i]);
          Value alloc_size = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{idxes}, rewriter);
          comet_errs() << " ";
          comet_vdump(alloc_size);

          alloc_sizes_vec.push_back(alloc_size);
          Value alloc_size_cast = rewriter.create<memref::CastOp>(loc, alloc_size, unrankedMemTy_index);
          alloc_sizes_cast_vec.push_back(alloc_size_cast);
        }

        for (unsigned int i = 2 * rank_size; i < 2 * rank_size + 1; i++)
        {
          std::vector<Value> idxes;
          idxes.push_back(array_sizes[i]);
          Value alloc_size = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{idxes}, rewriter);
          comet_errs() << " ";
          comet_vdump(alloc_size);
          alloc_sizes_vec.push_back(alloc_size);
          Value alloc_size_cast = rewriter.create<memref::CastOp>(loc, alloc_size, unrankedMemTy_f64);
          alloc_sizes_cast_vec.push_back(alloc_size_cast);
        }

        if (rank_size == 2)
        { // 2D
          std::string read_input_str;
          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            read_input_str = "read_input_2D_f32";
          }
          else
          {
            read_input_str = "read_input_2D_f64";
          }
          auto read_input_f64Call = rewriter.create<mlir::CallOp>(
              loc, read_input_str, SmallVector<Type, 2>{}, ValueRange{sparseFileID, dim_format[0], dim_format[1], alloc_sizes_cast_vec[0], alloc_sizes_cast_vec[1], alloc_sizes_cast_vec[2], alloc_sizes_cast_vec[3], alloc_sizes_cast_vec[4]});
          read_input_f64Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
        }
        else if (rank_size == 3)
        { // 3D
          std::string read_input_str;
          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            read_input_str = "read_input_3D_f32";
          }
          else
          {
            read_input_str = "read_input_3D_f64";
          }
          auto read_input_f64Call = rewriter.create<mlir::CallOp>(
              loc, read_input_str, SmallVector<Type, 2>{}, ValueRange{sparseFileID, dim_format[0], dim_format[1], dim_format[2], alloc_sizes_cast_vec[0], alloc_sizes_cast_vec[1], alloc_sizes_cast_vec[2], alloc_sizes_cast_vec[3], alloc_sizes_cast_vec[4], alloc_sizes_cast_vec[5], alloc_sizes_cast_vec[6]});
          read_input_f64Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
        }
        else
        {
          llvm::errs() << __LINE__ << " more than 3D, not supported\n";
        }

        comet_errs() << " Generate read_input_2D or read_input_3D functions\n";
        std::vector<Value> alloc_tensor_vec;
        for (unsigned int i = 0; i < 2 * rank_size + 1; i++)
        {
          Value tensorLoad = rewriter.create<memref::TensorLoadOp>(loc, alloc_sizes_vec[i]);
          alloc_tensor_vec.push_back(tensorLoad);
        }

        // create sptensor_construct
        SmallVector<mlir::Type, 1> elementTypes;
        for (unsigned int i = 0; i < 2 * rank_size + 1; i++)
        {
          elementTypes.push_back(alloc_tensor_vec[i].getType());
        }
        // for(unsigned int i = 2*rank_size + 1; i < 3*rank_size + 1; i++){
        // [0 ... 2*rank_size, 2*rank_size+1 ... 4*rank_size+1, 4*rank_size+2 ... 5*rank_size + 1]
        // 2d+1 + 2d+1 + d => 5d+2
        // for(unsigned int i = 2*rank_size + 1; i < 5*rank_size + 2; i++){
        for (unsigned int i = 0; i < 3 * rank_size + 1; i++)
        {
          elementTypes.push_back(array_sizes[i].getType());
        }

        auto ty = tensorAlgebra::SparseTensorType::get(elementTypes);

        Value sptensor;

        if (rank_size == 2)
        {
          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty, ValueRange{alloc_tensor_vec[0], alloc_tensor_vec[1], alloc_tensor_vec[2], alloc_tensor_vec[3], alloc_tensor_vec[4], array_sizes[0], array_sizes[1], array_sizes[2], array_sizes[3], array_sizes[4], array_sizes[5], array_sizes[6]});
        }
        else if (rank_size == 3)
        {
          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty, ValueRange{alloc_tensor_vec[0], alloc_tensor_vec[1], alloc_tensor_vec[2], alloc_tensor_vec[3], alloc_tensor_vec[4], alloc_tensor_vec[5], alloc_tensor_vec[6], array_sizes[0], array_sizes[1], array_sizes[2], array_sizes[3], array_sizes[4], array_sizes[5], array_sizes[6], array_sizes[7], array_sizes[8], array_sizes[9]});
        }
        else
        {
          llvm::errs() << __LINE__ << " more than 3D, not supported\n";
        }

        comet_errs() << " sptensor: ";
        comet_vdump(sptensor);

        // create ta.index_label operation.
        comet_vdump(op);

        auto tensor_decl_value = cast<tensorAlgebra::SparseTensorDeclOp>(op);
        LLVM_DEBUG(comet_errs() << " " << tensor_decl_value.labels().size() << "\n");
        for (unsigned int i = 0; i < tensor_decl_value.labels().size(); i++)
        {
          comet_vdump(tensor_decl_value.labels()[i]);
          comet_pdump(tensor_decl_value.labels()[i].getDefiningOp());
          if (isa<tensorAlgebra::IndexLabelDynamicOp>(tensor_decl_value.labels()[i].getDefiningOp()))
          {
            // comet_vdump(tensor_decl_value.labels()[i]);
            auto label_decl_value = cast<tensorAlgebra::IndexLabelDynamicOp>(tensor_decl_value.labels()[i].getDefiningOp());
            auto lo = label_decl_value.min();
            auto step = label_decl_value.step();
            auto hi = array_sizes[2 * rank_size + 1 + i];
            // mlir::Value value =
            Value new_index = rewriter.create<IndexLabelOp>(loc, lo, hi, step);
            label_decl_value.replaceAllUsesWith(new_index);
          }
          else if (isa<tensorAlgebra::IndexLabelOp>(tensor_decl_value.labels()[i].getDefiningOp()))
          {
            comet_errs() << " isa<tensorAlgebra::IndexLabelOp\n";
          }
        }

        op.replaceAllUsesWith(sptensor);
        rewriter.replaceOp(op, sptensor);
      }
      // The tensor is sparse output
      else if (isDense(formats_str, ", ") == false && isOutputTensor == true)
      {
        // Is sparse output ,lower to ta.output_tensor_decl
        auto tensor_decl_value = cast<tensorAlgebra::SparseTensorDeclOp>(op);
        auto labels = tensor_decl_value.labels();
        auto tensor_format = tensor_decl_value.format(); 
        auto tensor_type = tensor_decl_value.getType();

        mlir::Value outputtensordecl = rewriter.create<SparseOutputTensorDeclOp>(loc,
                                                                           tensor_type, labels, tensor_format);
        op.replaceAllUsesWith(outputtensordecl);
        rewriter.replaceOp(op, outputtensordecl);
      }
      /**** The tensor is dense *****/
      else
      { // format == "Dense"
        comet_errs() << " it is dense tensor\n";
      }

      comet_errs() << " SparseInputTensorDeclOpLowering in format end\n";
      // module->dump();
      return success();
    }
  };

  struct SparseOutputTensorDeclOpLowering : public OpRewritePattern<tensorAlgebra::SparseOutputTensorDeclOp>
  {
    using OpRewritePattern<tensorAlgebra::SparseOutputTensorDeclOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions and ta.sptensor_construct operation.
     * Step 3: Remove the SparseOutputTensorDeclOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::SparseOutputTensorDeclOp op,
                                  PatternRewriter &rewriter) const final
    {
      // Sparse output tensor declaration happens after lowering to index tree dialect
      assert(isa<tensorAlgebra::SparseOutputTensorDeclOp>(op));
      comet_errs() << "SparseOutputTensorDeclOpLowering in format begin\n";
      comet_vdump(op);

      auto loc = op.getLoc();
      StringRef formatsAttr = op.format();
      std::string formats_str(formatsAttr.data());
      comet_errs() << " --- " << formats_str << "\n";

      comet_errs() << " " << op.getNumOperands() << "\n";
      auto rank_size = op.getNumOperands();

      IndexType indexType = IndexType::get(op.getContext());
      FloatType f64Type = FloatType::getF64(op.getContext());
      if (VALUETYPE.compare(0, 3, "f32") == 0)
        f64Type = FloatType::getF32(op.getContext());

      // A1_pos ... A_value
      auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamicSize}, indexType); // memref<?xindex>
      auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamicSize}, f64Type);     // memref<?xf64>

      comet_errs() << " " << formats_str << " isDense: " << isDense(formats_str, ", ") << "\n";

      // sparse output
      if (isDense(formats_str, ", ") == false)
      {
        // search read_from_file function call to get the input file name
        // Currently, has no filename

        std::vector<Value> tensorload_sizes_vec;
        std::vector<Value> array_sizes_vec; // Store the size of C1pos, C1crd,..., Cval,C_dim1_size, C_dim2_size....

        // No need to read from file
        // We need to fill this tensorload_sizes_vec operations with new ones.....
        // Some should get from sparse input, some are dense
        std::string input_filename;
        std::vector<std::vector<unsigned>> allPerms;

        std::vector<Value> dimSizes; // for dimSizes in sptensor_construct

        for (auto u : op.getOperation()->getUsers())
        {
          comet_errs() << " Get users before ";
          comet_pdump(u);
          if (isa<tensorAlgebra::LabeledTensorOp>(u))
          {
            comet_pdump(u);
            auto labeledtensorop = cast<tensorAlgebra::LabeledTensorOp>(u);
            comet_errs() << " labeled_tensor op\n";
            for (auto u1 : u->getUsers())
            {
              if (isa<tensorAlgebra::TensorChainSetOp>(u1))
              {
                comet_errs() << " tensor set_op\n";
                auto setop = cast<tensorAlgebra::TensorChainSetOp>(u1);
                auto read_from_file_operand = setop.getOperand(1).getDefiningOp(); // funccall
                if (isa<tensorAlgebra::GenericCallOp>(read_from_file_operand))
                {
                  auto genericcallop = cast<tensorAlgebra::GenericCallOp>(read_from_file_operand);
                  // comet_vdump(genericcallop);
                  comet_errs() << " read_from_file op\n";
                  std::string read_ref(genericcallop.callee().getLeafReference());
                  comet_errs() << " read_ref: " << read_ref << "\n";
                  if (read_ref.compare(0, 14, "read_from_file") == 0)
                  {
                    comet_errs() << " yes, read_from_file op\n";
                    // get filename through operand
                    comet_errs() << " genericcallop.getNumOperands(): " << genericcallop.getOperation()->getNumOperands() << "\n";

                    // Erase the useless ops
                    rewriter.eraseOp(setop);
                    rewriter.eraseOp(genericcallop);
                    rewriter.eraseOp(labeledtensorop);
                  }
                }
              }
            }
          }

          else if (isa<tensorAlgebra::TensorFillFromFileOp>(u))
          {
            comet_errs() << " Sparse output is used in TensorFillFromFileOp\n";
            auto fillfromfileop = cast<tensorAlgebra::TensorFillFromFileOp>(u);
            // Can get filename, from "filename" attribute of fillfromfileop
            rewriter.eraseOp(fillfromfileop);
          }

          else if (isa<tensorAlgebra::TransposeOp>(u) ||
                   (isa<tensorAlgebra::TensorSetOp>(u) &&
                    isa<tensorAlgebra::TransposeOp>(cast<tensorAlgebra::TensorSetOp>(u).getOperand(0).getDefiningOp())))
          {
            comet_errs() << " sparse output is used in transpose op\n";
            // Get the freeIndices of the sparse input tensor
            // Check the dimension size, if it is integer, format is dense and get dim_size
            // If it is ?, get the sparse input and get the definition, and the freeindex,
            // tensorAlgebra::TransposeOp transpose_op = cast<tensorAlgebra::TransposeOp>(u);
            tensorAlgebra::TransposeOp transpose_op;
            if (isa<tensorAlgebra::TransposeOp>(u))
            {
              transpose_op = cast<tensorAlgebra::TransposeOp>(u);
            }
            else
            {
              transpose_op = cast<tensorAlgebra::TransposeOp>(cast<tensorAlgebra::TensorSetOp>(u).getOperand(0).getDefiningOp());
            }

            ArrayAttr indexMaps = transpose_op.indexing_maps();
            comet_errs() << " we get the indexMaps\n";
            allPerms = getAllPerms(indexMaps);
            comet_errs() << " we get the permutations\n";

            // mlir::Value src_input = getOperand(0);
            // mlir::Value dst_input = transpose_op.getOperand(0);
            mlir::Value src_input = transpose_op.rhs();
            comet_errs() << " ";
            comet_vdump(src_input);
            mlir::Value dst_input;
            for (auto u : op.getOperation()->getResult(0).getUsers())
            {
              comet_errs() << " ";
              comet_pdump(u);
              if (isa<tensorAlgebra::TensorSetOp>(u))
              {
                dst_input = u->getOperand(1); // dest tensor is the 2nd
                comet_vdump(dst_input);
              }
            }

            /// If in COO format, for every dimension, different dimensions are
            std::vector<unsigned int> dstIndexLocInSrcVec;
            for (auto n : allPerms[1])
            { // In dst index
              unsigned int dstIndexLocInSrc = findIndexInVector(allPerms[0], n);
              assert(dstIndexLocInSrc < allPerms[0].size() && " the index in dest is not found in src for transpose op\n");
              dstIndexLocInSrcVec.push_back(dstIndexLocInSrc);
            }

            ArrayAttr allFormats = transpose_op.formats();
            std::vector<std::string> allFormatsStr;
            for (unsigned int i = 0; i < allFormats.size(); i++)
            {
              std::string formats_str(allFormats[i].cast<mlir::StringAttr>().getValue());
              allFormatsStr.push_back(formats_str);
            }
            std::string src_format = allFormatsStr[0];
            std::string dst_format = allFormatsStr[1];

            // ArrayAttr opFormatsArrayAttr = tc_op.formats();
            // std::vector<std::vector<std::string>> allFormats = getAllFormats(opFormatsArrayAttr, allPerms);

            // If in COO format, then the sizes are the same as the input
            // for A and B: 2x+1 + 2x+1 + x = 5x+2
            // for ith index in B: pos is 2*i, crd is 2*i + 1
            //                     pos_size is (2*rank+1) + 2*i, crd_size is (2*rank+1) + 2*i+1
            // unsigned int dst_rank = (dst_input.getDefiningOp()->getNumOperands() -2)/5;
            comet_errs() << " ";
            comet_vdump(dst_input);
            comet_errs() << " ";
            comet_pdump(dst_input.getDefiningOp());
            unsigned int dst_rank = dst_input.getDefiningOp()->getNumOperands();
            for (unsigned int i = 0; i < dst_rank; i++)
            {
              // 4*rank+2 + i
              dimSizes.push_back(src_input.getDefiningOp()->getOperand(4 * dst_rank + 2 + dstIndexLocInSrcVec[i]));
            }

            Value cst_index_1 = rewriter.create<mlir::ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(1));
            Value cst_index_2 = rewriter.create<mlir::ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(2));

            /// For COO format, 2D and 3D are the same
            // if src format is in COO format,
            if (src_format.compare("COO") == 0)
            {
              for (unsigned int i = 0; i < dst_rank; i++)
              {
                // 2*dst_rank+1
                unsigned int dstIndexLocInSrc = dstIndexLocInSrcVec[i];
                // src_rank = dst_rank
                unsigned int posLocInSrc = (2 * dst_rank + 1) + 2 * dstIndexLocInSrc;
                unsigned int crdLocInSrc = posLocInSrc + 1;

                array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(posLocInSrc));
                array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(crdLocInSrc));
              }
              // val array size
              array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(4 * dst_rank + 1));

              // set the pos array size, 1st dim as 2, all others as 1.
              for (unsigned int i = 0; i < dst_rank; i++)
              {
                if (i == 0)
                {
                  array_sizes_vec[2 * i] = cst_index_2;
                }
                else
                {
                  array_sizes_vec[2 * i] = cst_index_1;
                }
              }
            }
            // For 2D, consider CSR
            else if (dst_rank == 2)
            {
              if (src_format.compare("CSR") == 0)
              {
                comet_errs() << " 2D CSR transpose to 2D CSR\n";
                array_sizes_vec.push_back(cst_index_1);
                array_sizes_vec.push_back(cst_index_1);
                mlir::Value crd_size = rewriter.create<mlir::AddIOp>(loc, dimSizes[0], cst_index_1);
                array_sizes_vec.push_back(crd_size);
                // B2pos, Bval are the same size with A2pos, Aval
                array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(9));
                array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(9));
              }
            }
            // For 3D, consider CSF
            else if (dst_rank == 3)
            {
              if (src_format.compare("CSF") == 0)
              {
                comet_errs() << " 3D CSF transpose to 3D CSF\n";
                array_sizes_vec.push_back(cst_index_2);
                mlir::Value src_nnz = src_input.getDefiningOp()->getOperand(13);
                mlir::Value src_nnz_add1 = rewriter.create<mlir::AddIOp>(loc, src_nnz, cst_index_1);
                array_sizes_vec.push_back(src_nnz);
                array_sizes_vec.push_back(src_nnz_add1);
                array_sizes_vec.push_back(src_nnz);
                array_sizes_vec.push_back(src_nnz_add1);
                array_sizes_vec.push_back(src_nnz);
                array_sizes_vec.push_back(src_nnz_add1);
              }
            }

            comet_errs() << " array_sizes_vec.size(): " << array_sizes_vec.size() << "\n";
            comet_errs() << " dst_rank: " << dst_rank << "\n";
            for (unsigned int i = 0; i < 2 * dst_rank + 1; i++)
            {
              Value alloc_sizes;
              if (i < 2 * dst_rank)
              {
                alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{array_sizes_vec[i]}, rewriter);
                comet_errs() << " AllocOp: ";
                comet_vdump(alloc_sizes);
              }
              else
              {
                alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{array_sizes_vec[i]}, rewriter);
                comet_errs() << " AllocOp: ";
                comet_vdump(alloc_sizes);
              }
              Value tensorload_sizes = rewriter.create<memref::TensorLoadOp>(loc, alloc_sizes);
              tensorload_sizes_vec.push_back(tensorload_sizes);
            }
          }

          else if (isa<indexTree::IndexTreeComputeLHSOp>(u))
          {
            comet_errs() << " sparse output is used in itComputeOp op\n";
            comet_errs() << " formats_str: " << formats_str << "\n";

            indexTree::IndexTreeComputeLHSOp lhsOp = cast<indexTree::IndexTreeComputeLHSOp>(u);
            comet_errs() << " current Op: ";
            comet_vdump(lhsOp);

            bool completed = false;
            for (auto uLHS : lhsOp.getOperation()->getUsers())
            {
              if (isa<indexTree::IndexTreeComputeOp>(uLHS))
              {
                completed = true;
                auto computeOp = cast<indexTree::IndexTreeComputeOp>(uLHS);
                comet_errs() << " Get RHS op: ";
                comet_vdump(computeOp);

                std::vector<std::vector<int>> rhsPerms;
                getRHSPermsOfComputeOp(computeOp, rhsPerms);

                std::vector<std::vector<std::string>> rhsFormats;
                getRHSFormatsOfComputeOp(computeOp, rhsFormats);

                comet_errs() << " rhsPerms: \n";
                for (auto m : rhsPerms)
                {
                  comet_errs() << " ";
                  for (auto n : m)
                  {
                    comet_errs() << n << " ";
                  }
                  comet_errs() << "\n";
                }

                comet_errs() << " rhsFormats: \n";
                for (auto m : rhsFormats)
                {
                  comet_errs() << " ";
                  for (auto n : m)
                  {
                    comet_errs() << n << " ";
                  }
                  comet_errs() << "\n";
                }

                bool isElementwise = checkIsElementwise(rhsPerms);
                bool isMixedMode = checkIsMixedMode(rhsFormats);

                comet_errs() << "IsElementWise: " << isElementwise << " isMixedMode: " << isMixedMode << "\n";
                if (isElementwise && isMixedMode)
                {
                  comet_errs() << "It is an elementwise multiplication in mixed Mode sparse = sparse * dense\n";
                  if (isMixedMode)
                  {
                    comet_errs() << "It is an mix-mode elementwise multiplication in Mix Mode\n";
                    mixModeEltWiseMultSparseTensorOutputLowering(computeOp,
                                                                 loc,
                                                                 rhsPerms,
                                                                 dimSizes,
                                                                 tensorload_sizes_vec,
                                                                 array_sizes_vec, rewriter);
                  }
                  else
                  {
                    comet_errs() << "It is an pure-sparse elementwise multiplication\n";
                    pureSparseMultSparseTensorOutputLowering(op,
                                                             loc,
                                                             formats_str,
                                                             dimSizes,
                                                             tensorload_sizes_vec,
                                                             array_sizes_vec,
                                                             rewriter);
                  }
                }
                else
                {
                  if (!isMixedMode)
                  {
                    comet_errs() << "It is an pure-sparse multiplication or assigment from dense to sparse (produced after workspace transformations)\n";
                    pureSparseMultSparseTensorOutputLowering(op,
                                                             loc,
                                                             formats_str,
                                                             dimSizes,
                                                             tensorload_sizes_vec,
                                                             array_sizes_vec,
                                                             rewriter);
                  }
                  else
                  {
                    assert(false && "Mix-mode sparse computation with sparse output not yet supported such as TTM (tensor times matrix)");
                  }
                }
              }
            }

            if (!completed)
              assert(false && "Sparse tensor output tensor declaration was not completed\n");
          }
          else if (isa<indexTree::IndexTreeComputeRHSOp>(u))
          {
            comet_errs() << " the tensor is in IndexTreeComputeRHSOp\n";
            continue;
          }
          else if (isa<tensorAlgebra::PrintOp>(u))
          {
            comet_errs() << " the tensor is in print op\n";
            continue;
          }
          else
          {
            llvm::errs() << __FILE__ << __LINE__ << " tensor is used in the following unsupported op\n";
            comet_pdump(u);
          }

          comet_errs() << " Get users after ";
          // create sparse tensor construct after lowering each sparse tensor output users
          comet_errs() << " tensorload_sizes_vec.size(): " << tensorload_sizes_vec.size() << ", rank_size: " << rank_size << "\n";
          // create sptensor_construct
          SmallVector<mlir::Type, 1> elementTypes;
          for (unsigned int i = 0; i < 2 * rank_size + 1; i++)
          {
            assert(tensorload_sizes_vec.size() > 0 && "ERROR: Please report this error to the developers!");
            comet_errs() << " " << i << " ";
            comet_vdump(tensorload_sizes_vec[i]);
            elementTypes.push_back(tensorload_sizes_vec[i].getType());
          }
          comet_errs() << "\n ";
          // [0 ... 2*rank_size, 2*rank_size+1 ... 4*rank_size+1, 4*rank_size+2 ... 5*rank_size + 1]
          // 2d+1 + 2d+1 + d => 5d+2
          for (unsigned int i = 0; i < 2 * rank_size + 1; i++)
          {
            assert(array_sizes_vec.size() > 0 && "ERROR: Please report this error to the developers!");
            comet_errs() << " " << i << " ";
            comet_vdump(array_sizes_vec[i]);
            elementTypes.push_back(array_sizes_vec[i].getType());
          }
          comet_errs() << "\n ";
          for (unsigned int i = 0; i < rank_size; i++)
          {
            assert(dimSizes.size() > 0 && "ERROR: Please report this error to the developers!");
            elementTypes.push_back(dimSizes[i].getType());
          }
          comet_errs() << "\n ";

          auto ty = tensorAlgebra::SparseTensorType::get(elementTypes);

          Value sptensor;
          if (rank_size == 2)
          {
            sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty, ValueRange{tensorload_sizes_vec[0], tensorload_sizes_vec[1], tensorload_sizes_vec[2], tensorload_sizes_vec[3], tensorload_sizes_vec[4], array_sizes_vec[0], array_sizes_vec[1], array_sizes_vec[2], array_sizes_vec[3], array_sizes_vec[4], dimSizes[0], dimSizes[1]});
          }
          else if (rank_size == 3)
          {
            sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty, ValueRange{tensorload_sizes_vec[0], tensorload_sizes_vec[1], tensorload_sizes_vec[2], tensorload_sizes_vec[3], tensorload_sizes_vec[4], tensorload_sizes_vec[5], tensorload_sizes_vec[6], array_sizes_vec[0], array_sizes_vec[1], array_sizes_vec[2], array_sizes_vec[3], array_sizes_vec[4], array_sizes_vec[5], array_sizes_vec[6], dimSizes[0], dimSizes[1], dimSizes[2]});
          }
          else
          {
            assert(false && "Not supported format (Tensors of dimensions greater than 3 are currently not supported).\n");
          }

          comet_errs() << "sptensor: ";
          comet_vdump(sptensor);

          // create ta.index_label operation.
          comet_vdump(op);

          op.replaceAllUsesWith(sptensor);
          rewriter.replaceOp(op, sptensor);
        } // for (auto u : op.getOperation()->getUsers())
      }
      else
      { // format == "Dense"

        auto tensor_decl_value = cast<tensorAlgebra::SparseOutputTensorDeclOp>(op);

        // <?x32xf64>
        auto resultTensorType = op.getResult().getType();
        std::vector<Value> cur_indices;
        std::vector<int64_t> cur_memref;
        auto resultMemTy = convertTensorToMemRef(resultTensorType.cast<TensorType>());
        for (int i = 0; i < resultMemTy.getRank(); i++)
        {
          if (resultMemTy.isDynamicDim(i))
            cur_memref.push_back(ShapedType::kDynamicSize);
          else // The constant dim size must NOT comes from the sparse matrix
            cur_memref.push_back(resultMemTy.getDimSize(i));

          if (isa<tensorAlgebra::IndexLabelOp>(tensor_decl_value.labels()[i].getDefiningOp()))
          {
            // comet_vdump(tensor_decl_value.labels()[i]);
            auto label_decl_value = cast<tensorAlgebra::IndexLabelOp>(tensor_decl_value.labels()[i].getDefiningOp());
            auto hi = label_decl_value.max();
            if (resultMemTy.isDynamicDim(i))
              cur_indices.push_back(hi); // IndexCastOp
          }
        }
        llvm::ArrayRef<int64_t> cur_memref_arrayref = llvm::ArrayRef<int64_t>(cur_memref);

        MemRefType memrefType2 = MemRefType::get(cur_memref_arrayref, f64Type);
        Value alloc_sizes1 = insertAllocAndInitialize(loc, memrefType2, ValueRange(cur_indices), rewriter);
        comet_errs() << " AllocOp: ";
        comet_vdump(alloc_sizes1);

        Value tensorLoad = rewriter.create<memref::TensorLoadOp>(loc, alloc_sizes1);
        comet_vdump(tensorLoad);

        op.replaceAllUsesWith(tensorLoad);
        rewriter.replaceOp(op, tensorLoad);
      }
      comet_errs() << "--------------SparseOutputTensorDeclOpLowering in format end\n";
      return success();
    }
  };

  struct TensorFillLowering : public ConversionPattern
  {
    TensorFillLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::TensorFillOp::getOperationName(), 1,
                            ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::TensorFillOp>(op));

      auto loc = op->getLoc();
      auto tensorFillOp = cast<tensorAlgebra::TensorFillOp>(op);

      auto tensorOperand = operands[0];
      auto tensorLoadOp = cast<memref::TensorLoadOp>(tensorOperand.getDefiningOp());
      auto memref = tensorLoadOp.memref();
      auto valueAttr = tensorFillOp.value();
      auto constantOp = rewriter.create<ConstantOp>(loc, valueAttr);

      rewriter.create<linalg::FillOp>(loc, memref, constantOp);
      rewriter.eraseOp(op);

      return success();
    }
  };

}
//===----------------------------------------------------------------------===//
/// Early Lowering Passes end
//===----------------------------------------------------------------------===//

namespace
{
  struct DenseTensorDeclLoweringPass
      : public PassWrapper<DenseTensorDeclLoweringPass, FunctionPass>
  {
    void runOnFunction() final;
  };

  struct SparseInputTensorDeclLoweringPass
      : public PassWrapper<SparseInputTensorDeclLoweringPass, FunctionPass>
  {
    void runOnFunction() final;
  };

  struct SparseOutputTensorDeclLoweringPass
      : public PassWrapper<SparseOutputTensorDeclLoweringPass, FunctionPass>
  {
    void runOnFunction() final;
  };

  struct TensorFillLoweringPass
      : public PassWrapper<TensorFillLoweringPass, FunctionPass>
  {
    void runOnFunction() final;
  };

} // end anonymous namespace.

/// Dense tensor declaration lowering
void DenseTensorDeclLoweringPass::runOnFunction()
{
  comet_errs() << "DenseTensorDeclLoweringPass begin\n";
  auto function = getFunction();

  ConversionTarget target(getContext());
  target.addLegalDialect<LinalgDialect, scf::SCFDialect, 
                        StandardOpsDialect, memref::MemRefDialect,
                        ITDialect>();

  target.addIllegalDialect<tensorAlgebra::TADialect>();

  target.addLegalOp<tensorAlgebra::TensorMultOp,
                    tensorAlgebra::PrintOp,
                    tensorAlgebra::TAReturnOp,
                    tensorAlgebra::SUMOp,
                    tensorAlgebra::TransposeOp,
                    tensorAlgebra::TensorFillOp,
                    tensorAlgebra::GetTimeOp,
                    tensorAlgebra::PrintElapsedTimeOp,
                    tensorAlgebra::TensorSetOp,
                    tensorAlgebra::TensorElewsMultOp>();

  target.addLegalOp<tensorAlgebra::SparseOutputTensorDeclOp,
                    tensorAlgebra::IndexLabelDynamicOp,
                    tensorAlgebra::IndexLabelOp,
                    tensorAlgebra::SparseTensorConstructOp>();

  OwningRewritePatternList patterns(&getContext());
  patterns.insert<DenseTensorDeclOpLowering>(&getContext());

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    llvm::errs() << "Failed to Convert DenseTensorDeclLoweringPass\n";
    signalPassFailure();
  }

  comet_errs() << "Early lowering finished\n";
}

/**********************************************************************/
/********* SparseTCEarlyLowering pass: sparse input *******************/
/**********************************************************************/
// These patterns implements the early lowering of tensor algebra dialect
// it should be apply after high-level optimization such multi operations
// optimization
//===----------------------------------------------------------------===//
void SparseInputTensorDeclLoweringPass::runOnFunction()
{
  ConversionTarget target(getContext());

  target.addLegalDialect<LinalgDialect, StandardOpsDialect, 
                        scf::SCFDialect, AffineDialect, 
                        mlir::memref::MemRefDialect>();

  target.addLegalOp<tensorAlgebra::SparseTensorConstructOp>();
  target.addLegalOp<tensorAlgebra::IndexLabelOp>();
  target.addLegalOp<tensorAlgebra::GenericCallOp>();
  target.addLegalOp<tensorAlgebra::TensorMultOp,
                    tensorAlgebra::TensorSetOp,
                    tensorAlgebra::TensorElewsMultOp>();
  target.addLegalOp<tensorAlgebra::SparseOutputTensorDeclOp>();
  target.addLegalOp<tensorAlgebra::DenseTensorDeclOp>();
  target.addLegalOp<tensorAlgebra::IndexLabelDynamicOp>();
  target.addLegalOp<tensorAlgebra::TensorFillOp>();

  target.addLegalOp<indexTree::IndexTreeOp>();
  target.addLegalOp<indexTree::IndexTreeIndicesOp>();
  target.addLegalOp<indexTree::IndexTreeComputeOp>();

  OwningRewritePatternList patterns(&getContext());
  patterns.insert<SparseInputTensorDeclOpLowering>(&getContext());

  if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
  {
    llvm::errs() << "Failed to Lower SparseInputTensorDeclLoweringPass\n";
    signalPassFailure();
  }
}

/******************************************************************/
/********** SparseTCOutputLowering pass: sparse output ************/
/*******************************************************************/
void SparseOutputTensorDeclLoweringPass::runOnFunction()
{
  comet_errs() << "start SparseOutputTensorDeclLoweringPass\n";
  ConversionTarget target(getContext());

  target.addLegalDialect<LinalgDialect, StandardOpsDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect>();

  target.addLegalOp<tensorAlgebra::SparseTensorConstructOp>();
  target.addLegalOp<tensorAlgebra::IndexLabelOp>();
  target.addLegalOp<tensorAlgebra::GenericCallOp>();

  target.addLegalOp<tensorAlgebra::TensorMultOp, tensorAlgebra::TensorSetOp,
                    tensorAlgebra::TensorElewsMultOp>();
  target.addLegalOp<tensorAlgebra::IndexLabelDynamicOp>();

  target.addLegalOp<indexTree::IndexTreeOp>();
  target.addLegalOp<indexTree::IndexTreeIndicesOp>();
  target.addLegalOp<indexTree::IndexTreeComputeOp>();

  comet_errs() << " ";

  OwningRewritePatternList patterns(&getContext());
  patterns.insert<SparseOutputTensorDeclOpLowering>(&getContext());

  if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
  {
    llvm::errs() << "Failed to Lower STCOutputLowering\n";
    signalPassFailure();
  }
}

void TensorFillLoweringPass::runOnFunction()
{
  comet_errs() << "start TensorFillLoweringPass\n";

  ConversionTarget target(getContext());
  target.addLegalDialect<LinalgDialect, StandardOpsDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect>();
  OwningRewritePatternList patterns(&getContext());
  patterns.insert<TensorFillLowering>(&getContext());

  if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
  {
    llvm::errs() << "Failed to Lower STCOutputLowering\n";
    signalPassFailure();
  }
}

/// Create a pass for lowering dense tensor (inputs and utput) declaration operations in memref dialect
std::unique_ptr<Pass> mlir::tensorAlgebra::createDenseTensorDeclLoweringPass()
{
  return std::make_unique<DenseTensorDeclLoweringPass>();
}

// Create a pass for lowering dense input tensor declaration operations
std::unique_ptr<Pass> mlir::tensorAlgebra::createSparseInputTensorDeclLoweringPass()
{
  return std::make_unique<SparseInputTensorDeclLoweringPass>();
}

// Create a pass for lowering sparse output declaration
std::unique_ptr<Pass> mlir::tensorAlgebra::createSparseOutputTensorDeclLoweringPass()
{
  return std::make_unique<SparseOutputTensorDeclLoweringPass>();
}

// Create a pass for lowering tensor fill operation
std::unique_ptr<Pass> mlir::tensorAlgebra::createTensorFillLoweringPass()
{
  return std::make_unique<TensorFillLoweringPass>();
}