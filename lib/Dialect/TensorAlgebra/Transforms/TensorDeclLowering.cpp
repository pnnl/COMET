//===- TensorDeclLowering.cpp -- Lower dense and sparse tensor declarations for input and output tensors----===//
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
// This file implements a lowering of sparse and dense tensor declarations
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include <limits>
#include <map>
#include <set>
#include <unordered_map>

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;

using namespace mlir::tensorAlgebra;
using namespace mlir::indexTree;

#define DEBUG_TYPE "tensor-decl-lowering"

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_TENSORDECLLOWERING
// #define DEBUG_MODE_TENSORDECLLOWERING
// #endif

#ifdef DEBUG_MODE_TENSORDECLLOWERING
#define comet_debug() llvm::errs() << __FILE__ << ":" << __LINE__ << " "
#define comet_pdump(n)                                \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " "; \
  n->dump()
#define comet_vdump(n)                                \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " "; \
  n.dump()
#else
#define comet_debug() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

//===----------------------------------------------------------------------===//
/// Lowering Passes for sparse/dense tensor declarations
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
    auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); // memref<?xindex>
    auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamic}, f64Type);     // memref<?xf64>

    comet_debug() << "mixModeEltWiseMultSparseTensorOutputLowering computeOp\n";
    comet_vdump(computeOp);

    //  elementwise mul op in mix sparse dense case
    //  If elementwise, copy sparse input arrays for elementwise mul
    int sparse_inputtensor_id = -1;
    auto rhsComputeOp = computeOp.getDefiningOp()->getOperand(0).getDefiningOp();

    auto first_operand = rhsComputeOp->getOperand(0).getDefiningOp();
    auto second_operand = rhsComputeOp->getOperand(1).getDefiningOp();
    comet_debug() << "EltWiseMult Operands:\n";
    comet_pdump(first_operand);
    comet_pdump(second_operand);

    if (isa<tensorAlgebra::SparseTensorConstructOp>(first_operand))
    {
      sparse_inputtensor_id = 0;
    }
    else if (isa<tensorAlgebra::SparseTensorConstructOp>(second_operand))
    {
      sparse_inputtensor_id = 1;
    }
    else
    {
      assert(false && "SparseTensorConstructOp was not found as one of the operands for itCompute");
    }

    comet_debug() << " SparseTensorConstructOp for computeOp: \n";
    comet_pdump(rhsComputeOp->getOperand(sparse_inputtensor_id).getDefiningOp());
    auto sptensor_construct_op = cast<tensorAlgebra::SparseTensorConstructOp>(rhsComputeOp->getOperand(sparse_inputtensor_id).getDefiningOp());

    for (unsigned int i = 0; i < 2 * (rshPerms[sparse_inputtensor_id].size()) + 1; i++)
    {
      comet_debug() << " in for loop\n";
      Value intput_tensorload_op = cast<ToTensorOp>(sptensor_construct_op.getOperand(i).getDefiningOp());
      Value input_alloc_op = cast<memref::AllocOp>(intput_tensorload_op.getDefiningOp()->getOperand(0).getDefiningOp());
      comet_debug() << " AllocOp: ";
      comet_vdump(input_alloc_op);

      comet_debug() << " ";
      Value input_alloc_op_param = input_alloc_op.getDefiningOp()->getOperand(0);
      comet_debug() << " ";

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
        comet_debug() << " AllocOp: ";
        comet_vdump(output_alloc_op);
      }

      Value output_tensorload_op = rewriter.create<ToTensorOp>(loc, output_alloc_op);
      tensorload_sizes_vec.push_back(output_tensorload_op);
    }
    comet_debug() << " ";

    // [0...2d, 2d+1...4d+1, 4d+2...5d+1]
    for (unsigned int i = 0; i < 2 * (rshPerms[sparse_inputtensor_id].size()) + 1; i++)
    {
      int sizes_i = i + 2 * (rshPerms[sparse_inputtensor_id].size()) + 1;
      comet_debug() << " ";
      comet_pdump(sptensor_construct_op.getOperand(sizes_i).getDefiningOp());

      Value input_load_op = sptensor_construct_op.getOperand(sizes_i);
      comet_debug() << "Ops push_back for Sparse Tensor Construct Op for MixedMode elementwise multiplication (array_sizes_vec):\n";
      comet_vdump(input_load_op);
      array_sizes_vec.push_back(input_load_op);
    }

    for (unsigned int i = 0; i < rshPerms[sparse_inputtensor_id].size(); i++)
    {
      int sizes_i = i + 2 * (2 * (rshPerms[sparse_inputtensor_id].size()) + 1);
      comet_debug() << " ";
      comet_pdump(sptensor_construct_op.getOperand(sizes_i).getDefiningOp());

      Value input_load_op = sptensor_construct_op.getOperand(sizes_i);
      comet_debug() << "Ops push_back for Sparse Tensor Construct Op for MixedMode elementwise multiplication (dimSizes):\n";
      comet_vdump(input_load_op);
      dimSizes.push_back(input_load_op);
    }
  }

  template <typename T>
  void pureSparseMultSparseTensorOutputLowering(T op,
                                                Location loc,
                                                std::string sparseOutputFormat,
                                                std::vector<Value> &dimSizes,
                                                std::vector<Value> &tensorload_sizes_vec,
                                                std::vector<Value> &array_sizes_vec,
                                                PatternRewriter &rewriter)
  {
    comet_debug() << " sparse output is used in itComputeOp op\n";
    comet_debug() << " sparseOutputFormat: " << sparseOutputFormat << "\n";

    comet_vdump(op);

    IndexType indexType = IndexType::get(op.getContext());
    FloatType f64Type = FloatType::getF64(op.getContext());
    auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); // memref<?xindex>
    auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamic}, f64Type);     // memref<?xf64>

    Value cst_index_0 = rewriter.create<ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(0));
    comet_vdump(cst_index_0);
    Value cst_index_1 = rewriter.create<ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(1));
    comet_vdump(cst_index_1);

    unsigned int tensor_rank = op.getOperation()->getNumOperands();

    std::vector<mlir::Value> array_sizes;
    std::vector<mlir::Value> array_sizes_alloc_vec;
    std::vector<mlir::Value> initial_array_sizes;

    if (sparseOutputFormat.compare("CSR") == 0)
    { // CSR format
      comet_debug() << " 2D CSR format in sparse output decl op\n";
      // AllocOp, storeOp, LoadOp
      initial_array_sizes.push_back(cst_index_1);
      initial_array_sizes.push_back(cst_index_1);

      // The other three size information size..
      // get the dimension size from operand
      // std::vector<Value> dim_sizes;
      for (unsigned int i = 0; i < op.getOperation()->getNumOperands(); i++)
      {
        if (isa<tensorAlgebra::IndexLabelStaticOp>(op.getOperation()->getOperand(i).getDefiningOp()))
        {
          Value indexlabelop = dyn_cast<tensorAlgebra::IndexLabelStaticOp>(op.getOperation()->getOperand(i).getDefiningOp());
          dimSizes.push_back(indexlabelop.getDefiningOp()->getOperand(1));
        }
      }
      // The dim size is the second parameter of the
      Value dim2_posSize = rewriter.create<AddIOp>(loc, dimSizes[0], cst_index_1);
      comet_debug() << "AddIOp generated for dim2_posSize:\n";
      comet_vdump(dim2_posSize);
      initial_array_sizes.push_back(dim2_posSize);

      Value dim2_crdSize = rewriter.create<MulIOp>(loc, dimSizes[0], dimSizes[1]);
      initial_array_sizes.push_back(dim2_crdSize);
      initial_array_sizes.push_back(dim2_crdSize);
      comet_debug() << " ";
      comet_vdump(dim2_crdSize);
    }
    else
    {
      assert(false && "Not supported format\n");
    }

    // same with transpose case
    comet_debug() << " array_sizes_vec.size(): " << array_sizes_vec.size() << "\n";
    comet_debug() << " tensor_rank: " << tensor_rank << "\n";
    std::vector<Value> array_alloc_vec;
    for (unsigned int i = 0; i < 2 * tensor_rank + 1; i++)
    {
      Value alloc_sizes;
      if (i < 2 * tensor_rank)
      {
        comet_debug() << " Inserting AllocOp: ";
        alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{initial_array_sizes[i]}, rewriter);
        comet_debug() << " AllocOp: ";
        comet_vdump(alloc_sizes);
      }
      else
      {
        alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{initial_array_sizes[i]}, rewriter);
        comet_debug() << " AllocOp: ";
        comet_vdump(alloc_sizes);
      }
      Value tensorload_sizes = rewriter.create<ToTensorOp>(loc, alloc_sizes);
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

  void insertReadFileLibCall(int rank_size, MLIRContext *ctx, ModuleOp &module, func::FuncOp function)
  {
    comet_debug() << "Inserting insertReadFileLibCall\n";
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
      comet_debug() << " Rank Size is 2\n";
      auto readInput2DF32Func = FunctionType::get(ctx, {i32Type, indexType, indexType, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_f32, i32Type}, {}); // last arg (i32Type): readMode
      auto readInput2DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_f64, i32Type}, {});

      if (VALUETYPE.compare("f32") == 0)
      {
        std::string func_name = "read_input_2D_f32";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << "Adding read_input_2D_f32 to the module\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInput2DF32Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else // f64
      {
        std::string func_name = "read_input_2D_f64";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << "Adding read_input_2D_f64 to the module\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInput2DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }

      auto readInputSizes2DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, unrankedMemref_index, i32Type}, {}); // last arg (i32Type): readMode

      if (VALUETYPE.compare("f32") == 0)
      {
        std::string func_name = "read_input_sizes_2D_f32";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << "Adding read_input_sizes_2D_f32 to the module\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInputSizes2DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        std::string func_name = "read_input_sizes_2D_f64";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << "Adding read_input_sizes_2D_f64 to the module\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInputSizes2DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
    }
    // 3D tensor
    else if (rank_size == 3)
    {
      auto readInput3DF32Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_f32, i32Type}, {}); // last arg (i32Type): readMode
      auto readInput3DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_index, unrankedMemref_f64, i32Type}, {});

      if (VALUETYPE.compare("f32") == 0)
      {
        std::string func_name = "read_input_3D_f32";
        if (!hasFuncDeclaration(module, func_name))
        {
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInput3DF32Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        std::string func_name = "read_input_3D_f64";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << " Insert read_input_3D_f64 decl\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInput3DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }

      auto readInputSizes3DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, unrankedMemref_index, i32Type}, {}); // last arg (i32Type): readMode

      if (VALUETYPE.compare("f32") == 0)
      {

        std::string func_name = "read_input_sizes_3D_f32";
        if (!hasFuncDeclaration(module, func_name))
        {
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInputSizes3DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        std::string func_name = "read_input_sizes_3D_f64";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << " Insert read_input_sizes_3D_f64 decl\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInputSizes3DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
    }
    else
    {
      llvm::errs() << __LINE__ << "Not supported dims\n";
    }
  }

  // This a common lowering function used to lower SparseOutputTensorDeclOp and TempSparseOutputTensorDeclOp
  template <typename T>
  void lowerSparseOutputTensorDec(T op, PatternRewriter &rewriter)
  {
    if (isa<SparseOutputTensorDeclOp>(op))
    {
      comet_debug() << "lowerSparseOutputTensorDec::SparseOutputTensorDeclOp lowering\n";
    }
    else if (isa<TempSparseOutputTensorDeclOp>(op))
    {
      comet_debug() << "lowerSparseOutputTensorDec::TempSparseOutputTensorDeclOp lowering\n";
    }

    assert(isa<SparseOutputTensorDeclOp>(op) || isa<TempSparseOutputTensorDeclOp>(op) &&
                                                    "Op should be either SparseOutputTensorDeclOp or TempSparseOutputTensorDeclOp");

    comet_vdump(op);
    auto loc = op.getLoc();
    StringRef formatsAttr = op.getFormat();
    std::string formats_str(formatsAttr.data());
    comet_debug() << " --- " << formats_str << "\n";

    comet_debug() << " " << op.getNumOperands() << "\n";
    auto rank_size = op.getNumOperands();

    IndexType indexType = IndexType::get(op.getContext());
    FloatType f64Type = FloatType::getF64(op.getContext());
    if (VALUETYPE.compare(0, 3, "f32") == 0)
      f64Type = FloatType::getF32(op.getContext());

    // A1_pos ... A_value
    auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); // memref<?xindex>
    auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamic}, f64Type);     // memref<?xf64>

    comet_debug() << " " << formats_str << " isDense: " << isDense(formats_str, ", ") << "\n";

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
      std::vector<std::vector<int64_t>> allPerms;

      std::vector<Value> dimSizes; // for dimSizes in sptensor_construct

      for (auto u : op.getOperation()->getUsers())
      {
        comet_debug() << " Users:\n";
        comet_pdump(u);

        if (isa<tensorAlgebra::TransposeOp>(u) ||
            (isa<tensorAlgebra::TensorSetOp>(u) &&
             isa<tensorAlgebra::TransposeOp>(cast<tensorAlgebra::TensorSetOp>(u).getOperand(0).getDefiningOp())))
        {
          if (!isa<tensorAlgebra::TransposeOp>(u))
          {
            comet_debug() << "User of sparse tensor is a set Operation. Src of setOp is transpose\n";
            // Set the insertion point before its user
            rewriter.setInsertionPoint(cast<tensorAlgebra::TensorSetOp>(u).getOperand(0).getDefiningOp());
          }
          else
          {
            comet_debug() << "User of sparse tensor is transpose operation\n";
            // Set the insertion point before its user
            rewriter.setInsertionPoint(u);
          }

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

          ArrayAttr indexMaps = transpose_op.getIndexingMaps();
          comet_debug() << " we get the indexMaps\n";
          allPerms = getAllPerms(indexMaps);
          comet_debug() << " we get the permutations\n";

          // mlir::Value src_input = getOperand(0);
          // mlir::Value dst_input = transpose_op.getOperand(0);
          mlir::Value src_input = transpose_op.getRhs();
          comet_debug() << " ";
          comet_vdump(src_input);
          mlir::Value dst_input;
          for (auto u : op.getOperation()->getResult(0).getUsers())
          {
            comet_debug() << " ";
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

          ArrayAttr allFormats = transpose_op.getFormats();
          std::vector<std::string> allFormatsStr;
          for (unsigned int i = 0; i < allFormats.size(); i++)
          {
            std::string formats_str(allFormats[i].cast<mlir::StringAttr>().getValue());
            allFormatsStr.push_back(formats_str);
          }
          std::string src_format = allFormatsStr[0];
          std::string dst_format = allFormatsStr[1];

          // If in COO format, then the sizes are the same as the input
          // for A and B: 2x+1 + 2x+1 + x = 5x+2
          // for ith index in B: pos is 2*i, crd is 2*i + 1
          //                     pos_size is (2*rank+1) + 2*i, crd_size is (2*rank+1) + 2*i+1
          // unsigned int dst_rank = (dst_input.getDefiningOp()->getNumOperands() -2)/5;
          comet_debug() << " ";
          comet_vdump(dst_input);
          comet_debug() << " ";
          comet_pdump(dst_input.getDefiningOp());
          unsigned int dst_rank = dst_input.getDefiningOp()->getNumOperands();
          for (unsigned int i = 0; i < dst_rank; i++)
          {
            // 4*rank+2 + i
            dimSizes.push_back(src_input.getDefiningOp()->getOperand(4 * dst_rank + 2 + dstIndexLocInSrcVec[i]));
          }

          Value cst_index_1 = rewriter.create<ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(1));
          comet_vdump(cst_index_1);
          Value cst_index_2 = rewriter.create<ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(2));
          comet_vdump(cst_index_2);

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
              comet_debug() << " 2D CSR transpose to 2D CSR\n";
              array_sizes_vec.push_back(cst_index_1);
              array_sizes_vec.push_back(cst_index_1);
              mlir::Value crd_size = rewriter.create<AddIOp>(loc, dimSizes[0], cst_index_1);
              comet_debug() << "AddIOp generated for crd_size for CSR:\n";
              comet_vdump(crd_size);
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
              comet_debug() << " 3D CSF transpose to 3D CSF\n";
              array_sizes_vec.push_back(cst_index_2);
              mlir::Value src_nnz = src_input.getDefiningOp()->getOperand(13);
              mlir::Value src_nnz_add1 = rewriter.create<AddIOp>(loc, src_nnz, cst_index_1);
              comet_debug() << "AddIOp generated for nnz for CSF:\n";
              comet_vdump(src_nnz_add1);
              array_sizes_vec.push_back(src_nnz);
              array_sizes_vec.push_back(src_nnz_add1);
              array_sizes_vec.push_back(src_nnz);
              array_sizes_vec.push_back(src_nnz_add1);
              array_sizes_vec.push_back(src_nnz);
              array_sizes_vec.push_back(src_nnz_add1);
            }
          }

          comet_debug() << " array_sizes_vec.size(): " << array_sizes_vec.size() << "\n";
          comet_debug() << " dst_rank: " << dst_rank << "\n";
          for (unsigned int i = 0; i < 2 * dst_rank + 1; i++)
          {
            Value alloc_sizes;
            if (i < 2 * dst_rank)
            {
              alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{array_sizes_vec[i]}, rewriter);
              comet_debug() << " AllocOp: ";
              comet_vdump(alloc_sizes);
            }
            else
            {
              alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{array_sizes_vec[i]}, rewriter);
              comet_debug() << " AllocOp: ";
              comet_vdump(alloc_sizes);
            }
            Value tensorload_sizes = rewriter.create<ToTensorOp>(loc, alloc_sizes);
            tensorload_sizes_vec.push_back(tensorload_sizes);
          }
        }
        else if (isa<indexTree::IndexTreeComputeLHSOp>(u))
        {
          comet_debug() << " sparse output is used in itComputeOp op\n";

          // Set the insertion point before its user
          rewriter.setInsertionPoint(u);

          indexTree::IndexTreeComputeLHSOp lhsOp = cast<indexTree::IndexTreeComputeLHSOp>(u);
          comet_debug() << " formats_str: " << formats_str << "\n";
          comet_debug() << " current Op: ";
          comet_vdump(lhsOp);

          for (auto uLHS : lhsOp.getOperation()->getUsers())
          {
            assert(isa<indexTree::IndexTreeComputeOp>(uLHS) && "User of IndexTreeComputeLHSOp can only be IndexTreeComputeOp");

            comet_debug() << " lhsOp user: ";
            comet_pdump(uLHS);

            auto computeOp = cast<indexTree::IndexTreeComputeOp>(uLHS);
            comet_debug() << " Get RHS op: ";
            comet_vdump(computeOp);

            std::vector<std::vector<int>> rhsPerms;
            getRHSPermsOfComputeOp(computeOp, rhsPerms);

            std::vector<std::vector<std::string>> rhsFormats;
            getRHSFormatsOfComputeOp(computeOp, rhsFormats);

            comet_debug() << " rhsPerms: \n";
            for (auto m : rhsPerms)
            {
              comet_debug() << " \n";
              for (auto n : m)
              {
                comet_debug() << n << " \n";
              }
              comet_debug() << "\n";
            }

            comet_debug() << " rhsFormats: \n";
            for (auto m : rhsFormats)
            {
              comet_debug() << " \n";
              for (auto n : m)
              {
                comet_debug() << n << " \n";
              }
              comet_debug() << "\n";
            }

            bool isElementwise = checkIsElementwise(rhsPerms);

            comet_debug() << "Checking if it is mixed mode\n";
            bool isMixedMode = checkIsMixedMode(rhsFormats);

            comet_debug() << "IsElementWise: " << isElementwise << " isMixedMode: " << isMixedMode << "\n";
            if (isElementwise && isMixedMode)
            {
              comet_debug() << "It is an elementwise multiplication in mixed Mode sparse = sparse * dense\n";
              if (isMixedMode)
              {
                comet_debug() << "It is an mix-mode elementwise multiplication in Mix Mode\n";
                mixModeEltWiseMultSparseTensorOutputLowering(computeOp,
                                                             loc,
                                                             rhsPerms,
                                                             dimSizes,
                                                             tensorload_sizes_vec,
                                                             array_sizes_vec, rewriter);
              }
              else
              {
                comet_debug() << "It is an pure-sparse elementwise multiplication\n";
                pureSparseMultSparseTensorOutputLowering<>(op,
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
                comet_debug() << "It is an pure-sparse multiplication or assigment from dense to sparse (produced after workspace transformations)\n";
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
                // assert(false && "Mix-mode sparse computation with sparse output not yet supported such as TTM (tensor times matrix)");
                // TODO(gkestor): if the sparsity patterns is known
                comet_debug() << "It is an mix mode element-wise multiplication\n";
                mixModeEltWiseMultSparseTensorOutputLowering(computeOp,
                                                             loc,
                                                             rhsPerms,
                                                             dimSizes,
                                                             tensorload_sizes_vec,
                                                             array_sizes_vec, rewriter);
              }
            }
          }
        }
        else if (isa<tensorAlgebra::TensorFillFromFileOp>(u))
        {
          comet_debug() << " Sparse output is used in TensorFillFromFileOp\n";
          auto fillfromfileop = cast<tensorAlgebra::TensorFillFromFileOp>(u);
          // Can get filename, from "filename" attribute of fillfromfileop
          rewriter.eraseOp(fillfromfileop);
        }
        else if (isa<indexTree::IndexTreeComputeRHSOp>(u))
        {
          comet_debug() << "The tensor is in IndexTreeComputeRHSOp, no action taken\n";
          continue;
        }
        else if (isa<tensorAlgebra::PrintOp>(u))
        {
          comet_debug() << "The tensor is in print op,  no action taken\n";
          continue;
        }
        else if (isa<tensorAlgebra::ReduceOp>(u))
        {
          comet_debug() << "The tensor is in sum op,  no action taken\n";
          continue;
        }
        else if (isa<tensorAlgebra::LabeledTensorOp>(u))
        {
          // TODO(gkestor): LabeledTensorOp is not used in the current design, needs cleaning up.
          // Look at the generated code. We should not generate LabeledTensorOp
          continue;
        }
        else
        {
          comet_pdump(u);
          llvm::errs() << __FILE__ << __LINE__ << " tensor is used in the following unsupported op\n";
        }

        comet_debug() << " Get users after ";
        // create sparse tensor construct after lowering each sparse tensor output users
        comet_debug() << " tensorload_sizes_vec.size(): " << tensorload_sizes_vec.size() << ", rank_size: " << rank_size << "\n";
        // create sptensor_construct
        SmallVector<mlir::Type, 1> elementTypes;
        for (unsigned int i = 0; i < 2 * rank_size + 1; i++)
        {
          assert(tensorload_sizes_vec.size() > 0 && "ERROR: Please report this error to the developers!");
          comet_debug() << " " << i << " ";
          comet_vdump(tensorload_sizes_vec[i]);
          elementTypes.push_back(tensorload_sizes_vec[i].getType());
        }
        comet_debug() << "\n ";
        // [0 ... 2*rank_size, 2*rank_size+1 ... 4*rank_size+1, 4*rank_size+2 ... 5*rank_size + 1]
        // 2d+1 + 2d+1 + d => 5d+2
        for (unsigned int i = 0; i < 2 * rank_size + 1; i++)
        {
          assert(array_sizes_vec.size() > 0 && "ERROR: Please report this error to the developers!");
          comet_debug() << " " << i << " ";
          comet_vdump(array_sizes_vec[i]);
          elementTypes.push_back(array_sizes_vec[i].getType());
        }
        comet_debug() << "\n ";
        for (unsigned int i = 0; i < rank_size; i++)
        {
          assert(dimSizes.size() > 0 && "ERROR: Please report this error to the developers!");
          elementTypes.push_back(dimSizes[i].getType());
        }
        comet_debug() << "\n ";

        auto ty = tensorAlgebra::SparseTensorType::get(elementTypes);

        Value sptensor;
        if (rank_size == 2)
        {
          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty,
                                                                             ValueRange{tensorload_sizes_vec[0], // A1pos (each dimension consists of pos and crd arrays)
                                                                                        tensorload_sizes_vec[1], // A1crd
                                                                                        tensorload_sizes_vec[2], // A2pos
                                                                                        tensorload_sizes_vec[3], // A2crd
                                                                                        tensorload_sizes_vec[4], // Aval
                                                                                        array_sizes_vec[0],      // A1pos_size (size of each pos and crd arrays)
                                                                                        array_sizes_vec[1],      // A1crd_size
                                                                                        array_sizes_vec[2],      // A2pos_size
                                                                                        array_sizes_vec[3],      // A2crd_size
                                                                                        array_sizes_vec[4],      // Aval_size (size of value array)
                                                                                        dimSizes[0],             // dim1_size(size of each dimension in sparse tensor)
                                                                                        dimSizes[1]});           // dim2_size (size of each dimension in sparse tensor)
        }
        else if (rank_size == 3)
        {
          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty,
                                                                             ValueRange{tensorload_sizes_vec[0],
                                                                                        tensorload_sizes_vec[1],
                                                                                        tensorload_sizes_vec[2],
                                                                                        tensorload_sizes_vec[3],
                                                                                        tensorload_sizes_vec[4],
                                                                                        tensorload_sizes_vec[5],
                                                                                        tensorload_sizes_vec[6],
                                                                                        array_sizes_vec[0],
                                                                                        array_sizes_vec[1],
                                                                                        array_sizes_vec[2],
                                                                                        array_sizes_vec[3],
                                                                                        array_sizes_vec[4],
                                                                                        array_sizes_vec[5],
                                                                                        array_sizes_vec[6],
                                                                                        dimSizes[0],
                                                                                        dimSizes[1],
                                                                                        dimSizes[2]});
        }
        else
        {
          assert(false && "Not supported format (Tensors of dimensions greater than 3 are currently not supported).\n");
        }

        comet_debug() << "SparseTensorConstructOp generated for sparse output tensor:\n";
        comet_vdump(sptensor);

        // create ta.index_label operation.
        comet_vdump(op);

        op.replaceAllUsesWith(sptensor);
        rewriter.replaceOp(op, sptensor);
      } // for (auto u : op.getOperation()->getUsers())
    }
    else
    { // format == "Dense"

      // <?x32xf64>
      auto resultTensorType = op.getResult().getType().template cast<mlir::TensorType>();
      ;
      std::vector<Value> cur_indices;
      std::vector<int64_t> cur_memref;
      auto resultMemTy = convertTensorToMemRef(resultTensorType);
      for (int i = 0; i < resultMemTy.getRank(); i++)
      {
        if (resultMemTy.isDynamicDim(i))
          cur_memref.push_back(ShapedType::kDynamic);
        else // The constant dim size must NOT comes from the sparse matrix
          cur_memref.push_back(resultMemTy.getDimSize(i));

        if (isa<tensorAlgebra::IndexLabelStaticOp>(op.getLabels()[i].getDefiningOp()))
        {
          auto label_decl_value = cast<tensorAlgebra::IndexLabelStaticOp>(op.getLabels()[i].getDefiningOp());
          auto hi = label_decl_value.getMax();
          if (resultMemTy.isDynamicDim(i))
            cur_indices.push_back(hi); // IndexCastOp
        }
      }
      llvm::ArrayRef<int64_t> cur_memref_arrayref = llvm::ArrayRef<int64_t>(cur_memref);

      MemRefType memrefType2 = MemRefType::get(cur_memref_arrayref, f64Type);
      Value alloc_sizes1 = insertAllocAndInitialize(loc, memrefType2, ValueRange(cur_indices), rewriter);
      comet_debug() << " AllocOp: ";
      comet_vdump(alloc_sizes1);

      Value tensorLoad = rewriter.create<ToTensorOp>(loc, alloc_sizes1);
      comet_vdump(tensorLoad);

      op.replaceAllUsesWith(tensorLoad);
      rewriter.replaceOp(op, tensorLoad);
    }
  }

  struct DenseTensorDeclOpLowering : public OpRewritePattern<tensorAlgebra::DenseTensorDeclOp>
  {
    using OpRewritePattern<tensorAlgebra::DenseTensorDeclOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions for dense tensor declaration
     * Step 3: Remove the DenseTensorDeclOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::DenseTensorDeclOp op,
                                  PatternRewriter &rewriter) const final
    {

      comet_debug() << "--------------DenseTensorDeclarationLowering in format begin\n";
      auto module = op->getParentOfType<ModuleOp>(); // debuging purposes
      comet_debug() << "\n\n";

      comet_vdump(op);
      auto loc = op.getLoc();

      auto tensor_decl_value = cast<tensorAlgebra::DenseTensorDeclOp>(op);

      auto resultTensorType = op.getResult().getType();
      std::vector<Value> cur_indices;
      std::vector<int64_t> cur_memref;
      auto resultMemTy = convertTensorToMemRef(resultTensorType.cast<TensorType>());

      for (int i = 0; i < resultMemTy.getRank(); i++)
      {
        // if (resultMemTy.isDynamicDim(i))
        //   cur_memref.push_back(ShapedType::kDynamic);
        // else // The constant dim size must NOT come from the sparse matrix
        //   cur_memref.push_back(resultMemTy.getDimSize(i));

        if (isa<tensorAlgebra::IndexLabelStaticOp>(tensor_decl_value.getLabels()[i].getDefiningOp()))
        {
          comet_vdump(tensor_decl_value.getLabels()[i]);
          auto label_decl_value = cast<tensorAlgebra::IndexLabelStaticOp>(tensor_decl_value.getLabels()[i].getDefiningOp());
          auto hi = label_decl_value.getMax();
          if (resultMemTy.isDynamicDim(i))
          {
            cur_indices.push_back(hi); // IndexCastOp
            comet_vdump(hi);
          }
        }
      }

      // Check if this tensor is explicitly initialized with ta.fill operation
      bool is_filled = false;
      for (auto u : op->getUsers())
      {
        comet_pdump(u);
        if (isa<tensorAlgebra::TensorFillOp>(u) || isa<tensorAlgebra::TensorSetOp>(u))
          is_filled = true;
      }

      comet_debug() << " AllocOp for initialization is_filled: " << is_filled << " \n";
      Value init_alloc;
      if (is_filled)
      {
        // if is_filled is true, only allocate memory and let ta.fill initializes tensors
        init_alloc = rewriter.create<memref::AllocOp>(loc, resultMemTy, ValueRange(cur_indices));
        comet_vdump(init_alloc);
      }
      else
      {
        // if is_filled is false, allocate memory and initialize it
        init_alloc = insertAllocAndInitialize(loc, resultMemTy, ValueRange(cur_indices), rewriter);
        comet_vdump(init_alloc);
      }

      cast<memref::AllocOp>(init_alloc.getDefiningOp()).setAlignmentAttr(rewriter.getI64IntegerAttr(32));

      Value tensorLoad = rewriter.create<ToTensorOp>(loc, init_alloc);
      comet_debug() << " TensorLoad:\n";
      comet_vdump(tensorLoad);

      op.replaceAllUsesWith(tensorLoad);
      // rewriter.replaceOp(op, tensorLoad);
      rewriter.eraseOp(op);

      comet_debug() << "--------------DenseTensorDeclarationLowering in format end\n";
      comet_debug() << "\n\n";

      // module->dump(); //debugging purposes
      return success();
    }
  };

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
      comet_debug() << " SparseInputTensorDeclOpLowering in format begin\n";
      comet_vdump(op);
      mlir::MLIRContext *ctx = rewriter.getContext();
      auto function = cast<func::FuncOp>(op->getParentOp());
      auto module = function.getOperation()->getParentOfType<ModuleOp>();

      std::string op_str = dump2str(op);
      bool isOutputTensor = false;

      auto loc = op.getLoc();
      StringRef formatsAttr = op.getFormat();
      std::string formats_str(formatsAttr.data());
      comet_debug() << " --- " << formats_str << "\n";

      comet_debug() << " " << op.getNumOperands() << "\n";
      auto rank_size = op.getNumOperands();

      IndexType indexType = IndexType::get(op.getContext());
      FloatType f64Type = FloatType::getF64(op.getContext());
      if (VALUETYPE.compare(0, 3, "f32") == 0)
        f64Type = FloatType::getF32(op.getContext());

      for (auto u1 : op.getOperation()->getUsers())
      {
        comet_debug() << "\nCheck the tensor is input or output\n";
        comet_pdump(u1);
        if (isa<tensorAlgebra::TensorMultOp>(u1))
        {
          comet_debug() << " used in ta.tc op\n";
          auto p = cast<tensorAlgebra::TensorMultOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            // comet_vdump(n);
            std::string n_str = dump2str(p->getOperand(i));
            comet_debug() << "the operands: " << n_str << "\n";
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_debug() << " FIND IT: " << i << "\n";
              if (i == 2)
              {
                isOutputTensor = true;
              }
            }
          }
        }
        else if (isa<tensorAlgebra::TensorElewsMultOp>(u1))
        {
          comet_debug() << " used in ta.elews_mul op\n";
          auto p = cast<tensorAlgebra::TensorElewsMultOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            std::string n_str = dump2str(p->getOperand(i));
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_debug() << " FIND IT: " << i << "\n";
              if (i == 2)
              {
                isOutputTensor = true;
              }
            }
          }
        }
        else if (isa<tensorAlgebra::TensorSetOp>(u1))
        {
          comet_debug() << " used in ta.set op\n";
          auto p = cast<tensorAlgebra::TensorSetOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            // comet_vdump(n);
            comet_debug() << " the " << i << "th operand\n";
            std::string n_str = dump2str(p->getOperand(i));
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_debug() << " FIND IT: " << i << "\n";
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
          comet_debug() << " used in transpose op\n";
          auto p = cast<tensorAlgebra::TransposeOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            std::string n_str = dump2str(p->getOperand(i));
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_debug() << " FIND IT: " << i << "\n";
              if (i == 2)
              {
                // output of ta.elews_mul
                isOutputTensor = true;
              }
            }
          }
        }
        else if (isa<indexTree::IndexTreeComputeRHSOp>(u1))
        {
          comet_debug() << " used in ta.itComputeRHS op\n";
          isOutputTensor = false;
        }
        else if (isa<indexTree::IndexTreeComputeLHSOp>(u1))
        {
          comet_debug() << " used in ta.itComputeLHS op\n";
          isOutputTensor = true;
        }
        else if (isa<tensorAlgebra::TensorFillFromFileOp>(u1))
        {
          // do nothing
          comet_debug() << " the tensor is in fill_from_file op\n";
        }
        else if (isa<tensorAlgebra::PrintOp>(u1))
        {
          comet_debug() << " the tensor is in PrintOp\n";
        }
        else if (isa<tensorAlgebra::ReduceOp>(u1))
        {
          comet_debug() << " the tensor is in ReduceOp\n";
        }
        else if (isa<tensorAlgebra::TensorElewsMultOp>(u1))
        {
          comet_debug() << " the tensor is in Elementwise multiplication\n";
        }
        else if (isa<tensorAlgebra::TensorFillOp>(u1))
        {
          // TODO: should we add this warning for user?
          // assert(false && " the sparse input tensor is using fill-op. Please use read_from_file() for sparse tensor inputs.");
        }
        else if (isa<tensorAlgebra::LabeledTensorOp>(u1))
        {
          // do nothing!
          comet_debug() << " the tensor has use in LabeledTensorOp and this use will be ignored!\n";
        }
        else
        {
          u1->dump();
          assert(false && " the tensor is in not supported operation");
        }
      }

      comet_debug() << " isOutputTensor: " << isOutputTensor << "\n";

      // A1_pos ... A_value
      auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); // memref<?xindex>
      auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamic}, f64Type);     // memref<?xf64>

      Type unrankedMemTy_index = UnrankedMemRefType::get(indexType, 0);
      Type unrankedMemTy_f64 = UnrankedMemRefType::get(f64Type, 0);

      comet_debug() << " " << formats_str << " isDense: " << isDense(formats_str, ", ") << "\n";

      // tensor is sparse and input.
      if (isDense(formats_str, ", ") == false && isOutputTensor == false)
      {
        comet_debug() << " Sparse input tensor \n";

        // search read_from_file function call to get the input file name
        // Currently, has no filename
        std::string input_filename;
        int readModeVal = -1;
        for (auto u : op.getOperation()->getUsers())
        {
          // Used in LabeledTensorOp and then the LabeledTensorOp is used in ChainSetOp
          if (isa<tensorAlgebra::LabeledTensorOp>(u))
          {
            comet_debug() << "\n";
            // comet_pdump(u);
            auto labeledtensorop = cast<tensorAlgebra::LabeledTensorOp>(u);
            LLVM_DEBUG(comet_debug() << " labeled_tensor op\n");
            for (auto u1 : u->getUsers())
            {
              if (isa<tensorAlgebra::ChainSetOp>(u1))
              {
                LLVM_DEBUG(comet_debug() << " tensor set_op\n");

                auto setop = cast<tensorAlgebra::ChainSetOp>(u1);

                auto read_from_file_operand = setop.getOperand(1).getDefiningOp(); // funccall

                if (isa<tensorAlgebra::GenericCallOp>(read_from_file_operand))
                {
                  auto genericcallop = cast<tensorAlgebra::GenericCallOp>(read_from_file_operand);
                  LLVM_DEBUG(comet_debug() << " read_from_file op\n");
                  std::string read_ref(genericcallop.getCalleeAttr().getLeafReference().getValue());
                  LLVM_DEBUG(comet_debug() << " read_ref: " << read_ref << "\n");
                  if (read_ref.compare(0, 14, "read_from_file") == 0)
                  {
                    LLVM_DEBUG(comet_debug() << " yes, read_from_file op\n");
                    // get filename through operand
                    comet_debug() << " genericcallop.getNumOperands(): " << genericcallop.getOperation()->getNumOperands() << "\n";

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
            auto fillfromfileop = cast<tensorAlgebra::TensorFillFromFileOp>(u);
            // Can get filename, from "filename" attribute of fillfromfileop
            StringAttr filename = fillfromfileop.getFilename().cast<StringAttr>();
            IntegerAttr readModeAttr = fillfromfileop.getReadMode().cast<IntegerAttr>();
            rewriter.eraseOp(fillfromfileop);

            comet_debug() << " filename: " << filename.getValue() << "\n";

            std::string filename_str(filename.getValue());
            input_filename = filename_str;
            comet_debug() << " " << input_filename << "\n";

            readModeVal = readModeAttr.getInt();
            comet_debug() << " readMode: " << readModeVal << "\n";
          }
        }

        MemRefType memTy_alloc_sizes = MemRefType::get({3 * rank_size + 1}, indexType);
        Value alloc_sizes = rewriter.create<memref::AllocOp>(loc, memTy_alloc_sizes);
        comet_debug() << " ";
        comet_vdump(alloc_sizes);

        Value alloc_sizes_cast = rewriter.create<memref::CastOp>(loc, unrankedMemTy_index, alloc_sizes);

        std::vector<Value> dim_format = mlir::tensorAlgebra::getFormatsValue(formats_str, rank_size, rewriter, loc, indexType);
        comet_debug() << " Get the dim_format\n";

        // inform the runtime of what env var to use for parsing input file
        IntegerType i32Type = IntegerType::get(op.getContext(), 32);
        Value sparseFileID;
        std::size_t pos = input_filename.find("SPARSE_FILE_NAME");
        if (pos == std::string::npos) // not found
        {
          // currently, reading of file when path of file is provided as arg is not supported at runtime.
          sparseFileID = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, -1));
        }
        // 16 is the length of SPARSE_FILE_NAME
        std::string fileID = input_filename.substr(pos + 16, 1); // this will only catch 0..9
        if (fileID.empty())
        { // SPARSE_FILE_NAME
          sparseFileID = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, 9999));
        }
        else
        { // SPARSE_FILE_NAME{int}
          comet_debug() << " Parsed fileID: " << fileID << "\n";
          int intFileID = std::stoi(fileID);
          sparseFileID = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, intFileID));
        }

        Value readModeConst;
        if (readModeVal == -1) // none specified
        {                      // 1, Default: standard matrix read
          readModeConst = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, 1));
        }
        else
        { // readMode specified by user
          readModeConst = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, readModeVal));
        }

        // TODO(gkestor): refactor the code to insert read_input file functions
        //  Now, setup the runtime calls to read sizes related to the input matrices (e.g., read_input_sizes_2D_f32)
        if (rank_size == 2)
        { // 2D
          comet_debug() << " 2D\n";
          // Add function definition to the module
          insertReadFileLibCall(rank_size, ctx, module, function);

          std::string read_input_sizes_str;
          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            read_input_sizes_str = "read_input_sizes_2D_f32";
          }
          else
          {
            read_input_sizes_str = "read_input_sizes_2D_f64";
          }
          auto read_input_sizes_Call = rewriter.create<func::CallOp>(loc, read_input_sizes_str, SmallVector<Type, 2>{},
                                                                     ValueRange{sparseFileID, dim_format[0], dim_format[1],
                                                                                alloc_sizes_cast, readModeConst});
          read_input_sizes_Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
        }
        else if (rank_size == 3)
        { // 3D

          comet_debug() << " 3D\n";
          // Add function definition to the module
          insertReadFileLibCall(rank_size, ctx, module, function);

          std::string read_input_sizes_str;
          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            read_input_sizes_str = "read_input_sizes_3D_f32";
          }
          else
          { // default f64
            read_input_sizes_str = "read_input_sizes_3D_f64";
          }
          auto read_input_sizes_3D_Call = rewriter.create<func::CallOp>(loc, read_input_sizes_str, SmallVector<Type, 2>{},
                                                                        ValueRange{sparseFileID, dim_format[0], dim_format[1],
                                                                                   dim_format[2], alloc_sizes_cast, readModeConst});
          read_input_sizes_3D_Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
          comet_debug() << "\n";
        }
        else
        {
          assert(false && " Utility functions to read sparse tensors are supported up to 3 dimensions\n");
        }

        std::vector<Value> array_sizes;
        for (unsigned int i = 0; i < 3 * rank_size + 1; i++)
        { // 2*rank_size + 1 + rank_size
          Value idx = rewriter.create<ConstantIndexOp>(loc, i);
          Value cor = rewriter.create<memref::LoadOp>(loc, alloc_sizes, idx);
          comet_debug() << " ";
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
          comet_debug() << " ";
          comet_vdump(alloc_size);

          alloc_sizes_vec.push_back(alloc_size);
          Value alloc_size_cast = rewriter.create<memref::CastOp>(loc, unrankedMemTy_index, alloc_size);
          alloc_sizes_cast_vec.push_back(alloc_size_cast);
        }

        for (unsigned int i = 2 * rank_size; i < 2 * rank_size + 1; i++)
        {
          std::vector<Value> idxes;
          idxes.push_back(array_sizes[i]);
          Value alloc_size = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{idxes}, rewriter);
          comet_debug() << " ";
          comet_vdump(alloc_size);
          alloc_sizes_vec.push_back(alloc_size);
          Value alloc_size_cast = rewriter.create<memref::CastOp>(loc, unrankedMemTy_f64, alloc_size);
          alloc_sizes_cast_vec.push_back(alloc_size_cast);
        }

        // Now, setup the runtime calls to read the input matrices (e.g., read_input_3D_f64)
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
          auto read_input_f64Call = rewriter.create<func::CallOp>(loc, read_input_str, SmallVector<Type, 2>{},
                                                                  ValueRange{sparseFileID, dim_format[0], dim_format[1],
                                                                             alloc_sizes_cast_vec[0], alloc_sizes_cast_vec[1],
                                                                             alloc_sizes_cast_vec[2], alloc_sizes_cast_vec[3],
                                                                             alloc_sizes_cast_vec[4], readModeConst});
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
          auto read_input_f64Call = rewriter.create<func::CallOp>(loc, read_input_str, SmallVector<Type, 2>{},
                                                                  ValueRange{sparseFileID, dim_format[0], dim_format[1],
                                                                             dim_format[2], alloc_sizes_cast_vec[0], alloc_sizes_cast_vec[1],
                                                                             alloc_sizes_cast_vec[2], alloc_sizes_cast_vec[3],
                                                                             alloc_sizes_cast_vec[4], alloc_sizes_cast_vec[5],
                                                                             alloc_sizes_cast_vec[6], readModeConst});
          read_input_f64Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
        }
        else
        {
          llvm::errs() << __LINE__ << " more than 3D, not supported\n";
        }

        comet_debug() << " Generate read_input_2D or read_input_3D functions\n";
        std::vector<Value> alloc_tensor_vec;
        for (unsigned int i = 0; i < 2 * rank_size + 1; i++)
        {
          Value tensorLoad = rewriter.create<ToTensorOp>(loc, alloc_sizes_vec[i]);
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

        comet_debug() << "SparseTensorConstructOp generated for input sparse tensor:\n";
        comet_vdump(sptensor);

        // Dynamic indexlabel is propogated to the dense tensor declaration.
        // For example in the spmv example, dense vector index label comes from sparse input matrix.
        auto tensor_decl_value = cast<tensorAlgebra::SparseTensorDeclOp>(op);
        LLVM_DEBUG(comet_debug() << " " << tensor_decl_value.getLabels().size() << "\n");
        for (unsigned int i = 0; i < tensor_decl_value.getLabels().size(); i++)
        {
          comet_vdump(tensor_decl_value.getLabels()[i]);
          comet_pdump(tensor_decl_value.getLabels()[i].getDefiningOp());
          if (isa<tensorAlgebra::IndexLabelDynamicOp>(tensor_decl_value.getLabels()[i].getDefiningOp()))
          {
            auto label_decl_value = cast<tensorAlgebra::IndexLabelDynamicOp>(tensor_decl_value.getLabels()[i].getDefiningOp());
            auto lo = label_decl_value.getMin();
            auto step = label_decl_value.getStep();
            auto hi = array_sizes[2 * rank_size + 1 + i];

            Value new_index = rewriter.create<IndexLabelStaticOp>(loc, lo, hi, step);
            comet_vdump(new_index);
            label_decl_value.replaceAllUsesWith(new_index);
          }
          else if (isa<tensorAlgebra::IndexLabelStaticOp>(tensor_decl_value.getLabels()[i].getDefiningOp()))
          {
            comet_debug() << " isa<tensorAlgebra::IndexLabelStaticOp\n";
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
        auto labels = tensor_decl_value.getLabels();
        auto tensor_format = tensor_decl_value.getFormat();
        auto tensor_type = tensor_decl_value.getType();
        auto is_temporal_tensor = tensor_decl_value.getTemporalTensor();

        mlir::Value outputtensordecl;
        if (is_temporal_tensor)
        {
          // TempSparseOutputTensorDeclOp should be lowered before SparseOutputTensorDeclOp
          outputtensordecl = rewriter.create<TempSparseOutputTensorDeclOp>(loc,
                                                                           tensor_type, labels, tensor_format);
          comet_debug() << "Gokcen\n";
          comet_vdump(outputtensordecl);
        }
        else
          outputtensordecl = rewriter.create<SparseOutputTensorDeclOp>(loc,
                                                                       tensor_type, labels, tensor_format);
        comet_debug() << "SparseOutputTensorDecl or TempSparseOutputTensorDeclOp Operation is generated\n";
        comet_vdump(outputtensordecl);
        op.replaceAllUsesWith(outputtensordecl);
        rewriter.replaceOp(op, outputtensordecl);
      }
      /**** The tensor is dense *****/
      else
      { // format == "Dense"
        comet_debug() << " it is dense tensor\n";
      }

      comet_debug() << " SparseInputTensorDeclOpLowering in format end\n";
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
      comet_debug() << "SparseOutputTensorDeclOpLowering in format begin\n";
      comet_vdump(op);

      lowerSparseOutputTensorDec<tensorAlgebra::SparseOutputTensorDeclOp>(op, rewriter);

      comet_debug() << "--------------SparseOutputTensorDeclOpLowering in format end\n";
      return success();
    }
  };

  struct TempSparseOutputTensorDeclOpLowering : public OpRewritePattern<tensorAlgebra::TempSparseOutputTensorDeclOp>
  {
    using OpRewritePattern<tensorAlgebra::TempSparseOutputTensorDeclOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions and ta.sptensor_construct operation.
     * Step 3: Remove the TempSparseOutputTensorDeclOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::TempSparseOutputTensorDeclOp op,
                                  PatternRewriter &rewriter) const final
    {
      // Sparse output tensor declaration happens after lowering to index tree dialect
      assert(isa<tensorAlgebra::TempSparseOutputTensorDeclOp>(op));

      comet_debug() << "TempSparseOutputTensorDeclOpLowering in format begins\n";
      lowerSparseOutputTensorDec<tensorAlgebra::TempSparseOutputTensorDeclOp>(op, rewriter);
      comet_debug() << "TempSparseOutputTensorDeclOpLowering in format ends\n";

      return success();
    }
  };

  class DenseTensorDeclLoweringPass
      : public PassWrapper<DenseTensorDeclLoweringPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DenseTensorDeclLoweringPass)
    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      mlir::comet::populateDenseTensorDeclLoweringPatterns(patterns);

      func::FuncOp function = getOperation();
      ConversionTarget target(getContext());
      target.addLegalDialect<ArithDialect,
                             memref::MemRefDialect,
                             scf::SCFDialect,
                             bufferization::BufferizationDialect,
                             IndexTreeDialect>();

      target.addLegalOp<tensorAlgebra::PrintOp,
                        tensorAlgebra::ReduceOp,
                        tensorAlgebra::TransposeOp,
                        tensorAlgebra::TensorFillOp,
                        tensorAlgebra::GetTimeOp,
                        tensorAlgebra::PrintElapsedTimeOp,
                        tensorAlgebra::TensorSetOp,
                        tensorAlgebra::SparseOutputTensorDeclOp,
                        tensorAlgebra::TempSparseOutputTensorDeclOp,
                        tensorAlgebra::IndexLabelDynamicOp,
                        tensorAlgebra::IndexLabelStaticOp,
                        tensorAlgebra::SparseTensorConstructOp>();

      //function.dump();
      if (failed(applyPartialConversion(function, target, std::move(patterns))))
      {
        llvm::errs() << "Failed to applyPartialConversion in DenseTensorDeclLoweringPass\n";
        signalPassFailure();
      }
      //function.dump();
    }
  };

  class SparseTensorDeclLoweringPass
      : public PassWrapper<SparseTensorDeclLoweringPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseTensorDeclLoweringPass)
    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      patterns.insert<SparseInputTensorDeclOpLowering>(&getContext());

      func::FuncOp function = getOperation();
      ConversionTarget target(getContext());

      target.addLegalDialect<LinalgDialect,
                             ArithDialect,
                             scf::SCFDialect,
                             mlir::memref::MemRefDialect,
                             IndexTreeDialect,
                             bufferization::BufferizationDialect>();

      target.addLegalOp<tensorAlgebra::PrintOp,
                        tensorAlgebra::GetTimeOp,
                        tensorAlgebra::PrintElapsedTimeOp,
                        tensorAlgebra::ReduceOp,
                        tensorAlgebra::TransposeOp,
                        tensorAlgebra::TensorFillOp,
                        tensorAlgebra::SparseTensorConstructOp,
                        tensorAlgebra::SparseOutputTensorDeclOp,
                        tensorAlgebra::TempSparseOutputTensorDeclOp,
                        tensorAlgebra::TensorSetOp,
                        tensorAlgebra::DenseTensorDeclOp,
                        tensorAlgebra::IndexLabelStaticOp,
                        tensorAlgebra::IndexLabelDynamicOp,
                        func::CallOp
                        >();

      if (failed(applyPartialConversion(function, target, std::move(patterns))))
      {
        llvm::errs() << "Failed to Lower SparseTensorDeclLoweringPass\n";
        signalPassFailure();
      }
      comet_debug() << "---------------SparseTensorDeclLoweringPass end\n";
    }
  };

  class SparseOutputTensorDeclLoweringPass
      : public PassWrapper<SparseOutputTensorDeclLoweringPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseTensorDeclLoweringPass)
    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      mlir::comet::populateSparseOutputTensorDeclLoweringPatterns(patterns);

      func::FuncOp function = getOperation();
      ConversionTarget target(getContext());

      target.addLegalDialect<LinalgDialect,
                             ArithDialect,
                             scf::SCFDialect,
                             mlir::memref::MemRefDialect,
                             IndexTreeDialect,
                             bufferization::BufferizationDialect>();

      target.addLegalOp<tensorAlgebra::PrintOp,
                        tensorAlgebra::GetTimeOp,
                        tensorAlgebra::PrintElapsedTimeOp,
                        tensorAlgebra::ReduceOp,
                        tensorAlgebra::TransposeOp,
                        tensorAlgebra::TensorFillOp,
                        tensorAlgebra::SparseTensorConstructOp,
                        tensorAlgebra::TensorSetOp,
                        tensorAlgebra::DenseTensorDeclOp,
                        tensorAlgebra::IndexLabelStaticOp,
                        tensorAlgebra::IndexLabelDynamicOp,
                        func::CallOp>();

      if (failed(applyPartialConversion(function, target, std::move(patterns))))
      {
        llvm::errs() << "Failed to Lower SparseOutputTensorDeclLoweringPass\n";
        signalPassFailure();
      }
      comet_debug() << "---------------SparseOutputTensorDeclLoweringPass end\n";
    }
  };

}
//===----------------------------------------------------------------------===//
/// Early Lowering Passes end
//===----------------------------------------------------------------------===//

void mlir::comet::populateDenseTensorDeclLoweringPatterns(RewritePatternSet &patterns)
{
  patterns.insert<DenseTensorDeclOpLowering>(patterns.getContext());
}

void mlir::comet::populateSparseOutputTensorDeclLoweringPatterns(RewritePatternSet &patterns)
{
  patterns.add<TempSparseOutputTensorDeclOpLowering>(patterns.getContext(),
                                                  /*benefit=*/200);
  patterns.add<SparseOutputTensorDeclOpLowering>(patterns.getContext(),
                                                  /*benefit=*/100);
  // patterns.insert<SparseOutputTensorDeclOpLowering,
  //                 TempSparseOutputTensorDeclOpLowering>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::comet::createDenseTensorDeclLoweringPass()
{
  return std::make_unique<DenseTensorDeclLoweringPass>();
}

std::unique_ptr<Pass> mlir::comet::createSparseTensorDeclLoweringPass()
{
  return std::make_unique<SparseTensorDeclLoweringPass>();
}

std::unique_ptr<Pass> mlir::comet::createSparseOutputTensorDeclLoweringPass()
{
  return std::make_unique<SparseOutputTensorDeclLoweringPass>();
}