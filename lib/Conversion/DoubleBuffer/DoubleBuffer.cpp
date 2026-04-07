//===- DoubleBuffer.cpp - Double Buffer Pass Implementation ---------------===//
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
// This file implements the pass to convert dense GEMM operations to
// double-buffered parallel execution with compute and auxiliary workers.
//
//===----------------------------------------------------------------------===//

#include "comet/Conversion/DoubleBuffer/DoubleBuffer.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-to-double-buffer"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
// *********** For debug purpose *********//

using namespace mlir;
using namespace mlir::indexTree;

namespace mlir {
  namespace comet {
  #define GEN_PASS_DEF_CONVERTTODOUBLEBUFFER
  #include "comet/Conversion/Passes.h.inc"
  } // namespace comet
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Helper Structures
//===----------------------------------------------------------------------===//

/// Information about a GEMM operation extracted from the IR
struct GEMMInfo {
  Value A;           // Input tensor A
  Value B;           // Input tensor B
  Value C;           // Output tensor C
  Value A_memref;    // Backing memref for A
  Value B_memref;    // Backing memref for B
  Value C_memref;    // Backing memref for C
  int64_t M;         // Rows of A, rows of C
  int64_t K;         // Cols of A, rows of B
  int64_t N;         // Cols of B, cols of C
  Type elemType;     // Element type (e.g., f64)
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Forward declaration
void declareRuntimeFunctions(OpBuilder &builder, ModuleOp module);

/// Check if an it.itree contains a GEMM pattern (C = A * B)
static bool isGEMMPattern(indexTree::IndexTreeOp itreeOp) {
  bool foundGEMM = false;
  itreeOp.getBody()->walk([&](indexTree::IndexTreeComputeOp computeOp) {
    /// TODO: should not include sparse computation right now. Adding checkin for dense only.
    if (computeOp.getSemiring() == "plusxy_times") {
      foundGEMM = true;
    }
  });
  return foundGEMM;
}

/// Get the memref backing a tensor (assumes bufferization.to_tensor pattern)
static Value getMemRefFromTensor(Value tensor) {
  if (auto toTensorOp = tensor.getDefiningOp<bufferization::ToTensorOp>()) {
    return toTensorOp.getMemref();
  }
  return nullptr;
}

/// Try to extract GEMM information from an itree operation
static std::optional<GEMMInfo> extractGEMMInfo(indexTree::IndexTreeOp itreeOp) {
  GEMMInfo info;

  // The itree has one input (the output tensor C)
  if (itreeOp.getInputs().size() != 1)
    return std::nullopt;

  info.C = itreeOp.getInputs()[0];
  auto cType = dyn_cast<RankedTensorType>(info.C.getType());
  if (!cType || cType.getRank() != 2)
    return std::nullopt;

  info.elemType = cType.getElementType();
  info.M = cType.getDimSize(0);
  info.N = cType.getDimSize(1);

  // Find A and B by walking the compute op
  itreeOp.getBody()->walk([&](indexTree::IndexTreeComputeOp computeOp) {
    for (auto operand : computeOp.getRhs()) {
      if (auto operandOp = operand.getDefiningOp<indexTree::IndexTreeOperandOp>()) {
        Value tensor = operandOp.getTensor();
        auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
        if (!tensorType)
          continue;

        // A is MxK, B is KxN
        if (tensorType.getDimSize(0) == info.M) {
          info.A = tensor;
          info.K = tensorType.getDimSize(1);
        } else {
          info.B = tensor;
        }
      }
    }
  });

  if (!info.A || !info.B)
    return std::nullopt;

  // Get backing memrefs
  info.A_memref = getMemRefFromTensor(info.A);
  info.B_memref = getMemRefFromTensor(info.B);
  info.C_memref = getMemRefFromTensor(info.C);

  if (!info.A_memref || !info.B_memref || !info.C_memref)
    return std::nullopt;

  return info;
}

/// Declare an external runtime function if not already declared
static func::FuncOp getOrInsertFuncDecl(OpBuilder &builder, ModuleOp module,
                                        StringRef name, TypeRange inputs,
                                        TypeRange results) {
  if (auto funcOp = module.lookupSymbol<func::FuncOp>(name))
    return funcOp;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType(inputs, results);
  auto funcOp = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
  funcOp.setPrivate();
  return funcOp;
}

//===----------------------------------------------------------------------===//
// Double Buffer Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertToDoubleBufferPass
    : public PassWrapper<ConvertToDoubleBufferPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToDoubleBufferPass)

  ConvertToDoubleBufferPass() = default;
  ConvertToDoubleBufferPass(int32_t numComputeWorkers, int32_t numAuxWorkers,
                            int32_t tileSize) {
    this->numComputeWorkers = numComputeWorkers;
    this->numAuxWorkers = numAuxWorkers;
    this->tileSize = tileSize;
  }

  // Pass options (populated from TableGen)
  int32_t numComputeWorkers = 4;
  int32_t numAuxWorkers = 1;
  int32_t tileSize = 2;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    // Find all it.itree operations that contain GEMM patterns
    SmallVector<indexTree::IndexTreeOp> gemmOps;
    funcOp.walk([&](indexTree::IndexTreeOp itreeOp) {
      if (isGEMMPattern(itreeOp)) {
        gemmOps.push_back(itreeOp);
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Found " << gemmOps.size() << " GEMM patterns\n");

    // Transform each GEMM itree
    for (auto itreeOp : gemmOps) {
      if (failed(transformGEMM(itreeOp, builder))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to transform GEMM\n");
        return signalPassFailure();
      }
    }
  }

  LogicalResult transformGEMM(indexTree::IndexTreeOp itreeOp, OpBuilder &builder) {
    auto gemmInfoOpt = extractGEMMInfo(itreeOp);
    if (!gemmInfoOpt) {
      LLVM_DEBUG(llvm::dbgs() << "Could not extract GEMM info\n");
      return failure();
    }
    GEMMInfo &info = *gemmInfoOpt;

    Location loc = itreeOp.getLoc();
    builder.setInsertionPoint(itreeOp);
    MLIRContext *ctx = builder.getContext();

    // Get parent module for function declarations
    ModuleOp module = itreeOp->getParentOfType<ModuleOp>();

    // === Create constants ===
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value cst0_f64 = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(0.0), builder.getF64Type());
    Value char0 = builder.create<arith::ConstantIntOp>(loc, 0, 8);
    Value char1 = builder.create<arith::ConstantIntOp>(loc, 1, 8);

    Value numWorkersVal = builder.create<arith::ConstantIndexOp>(loc, numComputeWorkers);
    Value numAuxVal = builder.create<arith::ConstantIndexOp>(loc, numAuxWorkers);

    // Matrix dimensions
    Value A1 = builder.create<arith::ConstantIndexOp>(loc, info.M);
    Value A2 = builder.create<arith::ConstantIndexOp>(loc, info.K);
    Value B1 = builder.create<arith::ConstantIndexOp>(loc, info.K);
    Value B2 = builder.create<arith::ConstantIndexOp>(loc, info.N);
    Value C1 = builder.create<arith::ConstantIndexOp>(loc, info.M);
    Value C2 = builder.create<arith::ConstantIndexOp>(loc, info.N);

    // Tile sizes
    Value A1_tile = builder.create<arith::ConstantIndexOp>(loc, tileSize);
    Value A2_tile = builder.create<arith::ConstantIndexOp>(loc, tileSize);
    Value B1_tile = builder.create<arith::ConstantIndexOp>(loc, tileSize);
    Value B2_tile = builder.create<arith::ConstantIndexOp>(loc, tileSize);
    Value A_tile_size = builder.create<arith::MulIOp>(loc, A1_tile, A2_tile);
    Value B_tile_size = builder.create<arith::MulIOp>(loc, B1_tile, B2_tile);

    // === Cast matrices to unranked memrefs ===
    auto unrankedF64Type = UnrankedMemRefType::get(builder.getF64Type(), {});
    Value A_cast = builder.create<memref::CastOp>(loc, unrankedF64Type, info.A_memref);
    Value B_cast = builder.create<memref::CastOp>(loc, unrankedF64Type, info.B_memref);
    Value C_cast = builder.create<memref::CastOp>(loc, unrankedF64Type, info.C_memref);

    // === Create double buffer infrastructure ===
    // Type for array of unranked memrefs of unranked memrefs (double pointer)
    auto unrankedUnrankedF64Type = UnrankedMemRefType::get(unrankedF64Type, {});
    auto bufferArrayType = MemRefType::get({ShapedType::kDynamic}, unrankedUnrankedF64Type);

    // A_buffer1s, A_buffer2s
    Value A_buffer1s = builder.create<memref::AllocOp>(loc, bufferArrayType, numWorkersVal);
    Value A_buffer2s = builder.create<memref::AllocOp>(loc, bufferArrayType, numWorkersVal);

    // Initialize A_buffer1s
    initializeBufferArray(builder, loc, A_buffer1s, A_tile_size, numWorkersVal, c0, c1, info.elemType);
    initializeBufferArray(builder, loc, A_buffer2s, A_tile_size, numWorkersVal, c0, c1, info.elemType);

    // A_buffer_readys
    auto unrankedI8Type = UnrankedMemRefType::get(builder.getI8Type(), {});
    auto flagArrayType = MemRefType::get({ShapedType::kDynamic}, unrankedI8Type);
    Value A_buffer_readys = builder.create<memref::AllocOp>(loc, flagArrayType, numWorkersVal);
    initializeFlagArray(builder, loc, A_buffer_readys, numWorkersVal, c0, c1, char1);

    // A offset lists
    auto unrankedIndexType = UnrankedMemRefType::get(builder.getIndexType(), {});
    auto indexArrayType = MemRefType::get({ShapedType::kDynamic}, unrankedIndexType);
    Value A1_offset_list = builder.create<memref::AllocOp>(loc, indexArrayType, numWorkersVal);
    Value A2_offset_list = builder.create<memref::AllocOp>(loc, indexArrayType, numWorkersVal);
    Value A_block_rows_list = builder.create<memref::AllocOp>(loc, indexArrayType, numWorkersVal);
    Value A_block_cols_list = builder.create<memref::AllocOp>(loc, indexArrayType, numWorkersVal);
    initializeIndexArray(builder, loc, A1_offset_list, numWorkersVal, c0, c1);
    initializeIndexArray(builder, loc, A2_offset_list, numWorkersVal, c0, c1);
    initializeIndexArray(builder, loc, A_block_rows_list, numWorkersVal, c0, c1);
    initializeIndexArray(builder, loc, A_block_cols_list, numWorkersVal, c0, c1);

    // B_buffer1s, B_buffer2s
    Value B_buffer1s = builder.create<memref::AllocOp>(loc, bufferArrayType, numWorkersVal);
    Value B_buffer2s = builder.create<memref::AllocOp>(loc, bufferArrayType, numWorkersVal);
    initializeBufferArray(builder, loc, B_buffer1s, B_tile_size, numWorkersVal, c0, c1, info.elemType);
    initializeBufferArray(builder, loc, B_buffer2s, B_tile_size, numWorkersVal, c0, c1, info.elemType);

    // B_buffer_readys
    Value B_buffer_readys = builder.create<memref::AllocOp>(loc, flagArrayType, numWorkersVal);
    initializeFlagArray(builder, loc, B_buffer_readys, numWorkersVal, c0, c1, char1);

    // B offset lists
    Value B1_offset_list = builder.create<memref::AllocOp>(loc, indexArrayType, numWorkersVal);
    Value B2_offset_list = builder.create<memref::AllocOp>(loc, indexArrayType, numWorkersVal);
    Value B_block_rows_list = builder.create<memref::AllocOp>(loc, indexArrayType, numWorkersVal);
    Value B_block_cols_list = builder.create<memref::AllocOp>(loc, indexArrayType, numWorkersVal);
    initializeIndexArray(builder, loc, B1_offset_list, numWorkersVal, c0, c1);
    initializeIndexArray(builder, loc, B2_offset_list, numWorkersVal, c0, c1);
    initializeIndexArray(builder, loc, B_block_rows_list, numWorkersVal, c0, c1);
    initializeIndexArray(builder, loc, B_block_cols_list, numWorkersVal, c0, c1);

    // Compute workers finished flags
    Value compute_workers_finished = builder.create<memref::AllocOp>(loc, flagArrayType, numWorkersVal);
    initializeFlagArray(builder, loc, compute_workers_finished, numWorkersVal, c0, c1, char0);

    // === Calculate work distribution ===
    // tiles_per_worker_list
    auto tilesPerWorkerType = MemRefType::get({ShapedType::kDynamic}, builder.getIndexType());
    Value tiles_per_worker_list = builder.create<memref::AllocOp>(loc, tilesPerWorkerType, numWorkersVal);

    // Calculate number of i tiles: num_i_tiles = (A1 + A1_tile - 1) / A1_tile
    Value A1_tile_sub_one = builder.create<arith::SubIOp>(loc, A1_tile, c1);
    Value A1_ceiling = builder.create<arith::AddIOp>(loc, A1, A1_tile_sub_one);
    Value num_i_tiles = builder.create<arith::DivUIOp>(loc, A1_ceiling, A1_tile);

    // tiles_per_worker_base and remainder
    Value tiles_per_worker_base = builder.create<arith::DivUIOp>(loc, num_i_tiles, numWorkersVal);
    Value tiles_per_worker_remainder = builder.create<arith::RemUIOp>(loc, num_i_tiles, numWorkersVal);

    // Initialize tiles_per_worker_list with distribution
    builder.create<scf::ForOp>(
        loc, c0, numWorkersVal, c1, ValueRange{},
        [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
          Value is_remainder = b.create<arith::CmpIOp>(
              l, arith::CmpIPredicate::ult, iv, tiles_per_worker_remainder);
          auto ifOp = b.create<scf::IfOp>(
              l, is_remainder,
              [&](OpBuilder &thenB, Location thenL) {
                thenB.create<scf::YieldOp>(thenL, c1);
              },
              [&](OpBuilder &elseB, Location elseL) {
                elseB.create<scf::YieldOp>(elseL, c0);
              });
          Value plus = ifOp.getResult(0);
          Value tiles_per_worker = b.create<arith::AddIOp>(l, tiles_per_worker_base, plus);
          b.create<memref::StoreOp>(l, tiles_per_worker, tiles_per_worker_list, iv);
          b.create<scf::YieldOp>(l);
        });

    // === Declare runtime functions ===
    declareRuntimeFunctions(builder, module);

    // === Cast arrays to unranked for function calls ===
    auto unrankedBufferArrayType = UnrankedMemRefType::get(unrankedUnrankedF64Type, {});
    Value A_buffer1s_cast = builder.create<memref::CastOp>(loc, unrankedBufferArrayType, A_buffer1s);
    Value A_buffer2s_cast = builder.create<memref::CastOp>(loc, unrankedBufferArrayType, A_buffer2s);
    Value B_buffer1s_cast = builder.create<memref::CastOp>(loc, unrankedBufferArrayType, B_buffer1s);
    Value B_buffer2s_cast = builder.create<memref::CastOp>(loc, unrankedBufferArrayType, B_buffer2s);

    auto unrankedFlagArrayType = UnrankedMemRefType::get(unrankedI8Type, {});
    Value A_buffer_readys_cast = builder.create<memref::CastOp>(loc, unrankedFlagArrayType, A_buffer_readys);
    Value B_buffer_readys_cast = builder.create<memref::CastOp>(loc, unrankedFlagArrayType, B_buffer_readys);
    Value compute_workers_finished_cast = builder.create<memref::CastOp>(loc, unrankedFlagArrayType, compute_workers_finished);

    auto unrankedIndexArrayType = UnrankedMemRefType::get(unrankedIndexType, {});
    Value A1_offset_list_cast = builder.create<memref::CastOp>(loc, unrankedIndexArrayType, A1_offset_list);
    Value A2_offset_list_cast = builder.create<memref::CastOp>(loc, unrankedIndexArrayType, A2_offset_list);
    Value A_block_rows_list_cast = builder.create<memref::CastOp>(loc, unrankedIndexArrayType, A_block_rows_list);
    Value A_block_cols_list_cast = builder.create<memref::CastOp>(loc, unrankedIndexArrayType, A_block_cols_list);
    Value B1_offset_list_cast = builder.create<memref::CastOp>(loc, unrankedIndexArrayType, B1_offset_list);
    Value B2_offset_list_cast = builder.create<memref::CastOp>(loc, unrankedIndexArrayType, B2_offset_list);
    Value B_block_rows_list_cast = builder.create<memref::CastOp>(loc, unrankedIndexArrayType, B_block_rows_list);
    Value B_block_cols_list_cast = builder.create<memref::CastOp>(loc, unrankedIndexArrayType, B_block_cols_list);

    auto unrankedTilesListType = UnrankedMemRefType::get(builder.getIndexType(), {});
    Value tiles_per_worker_list_cast = builder.create<memref::CastOp>(loc, unrankedTilesListType, tiles_per_worker_list);

    // === Launch compute workers ===
    Value computeGroup = builder.create<async::CreateGroupOp>(
        loc, async::GroupType::get(ctx), numWorkersVal);

    builder.create<scf::ForOp>(
        loc, c0, numWorkersVal, c1, ValueRange{c0},
        [&](OpBuilder &b, Location l, Value w, ValueRange iterArgs) {
          Value i_start = iterArgs[0];

          // Load per-worker data
          Value A_buffer1_ptr_cast = b.create<memref::LoadOp>(l, A_buffer1s, w);
          Value B_buffer1_ptr_cast = b.create<memref::LoadOp>(l, B_buffer1s, w);
          Value A_buffer_is_ready_cast = b.create<memref::LoadOp>(l, A_buffer_readys, w);
          Value B_buffer_is_ready_cast = b.create<memref::LoadOp>(l, B_buffer_readys, w);
          Value num_local_i_tiles = b.create<memref::LoadOp>(l, tiles_per_worker_list, w);
          Value A1_offset_ptr = b.create<memref::LoadOp>(l, A1_offset_list, w);
          Value A2_offset_ptr = b.create<memref::LoadOp>(l, A2_offset_list, w);
          Value A_block_rows_ptr = b.create<memref::LoadOp>(l, A_block_rows_list, w);
          Value A_block_cols_ptr = b.create<memref::LoadOp>(l, A_block_cols_list, w);
          Value B1_offset_ptr = b.create<memref::LoadOp>(l, B1_offset_list, w);
          Value B2_offset_ptr = b.create<memref::LoadOp>(l, B2_offset_list, w);
          Value B_block_rows_ptr = b.create<memref::LoadOp>(l, B_block_rows_list, w);
          Value B_block_cols_ptr = b.create<memref::LoadOp>(l, B_block_cols_list, w);
          Value is_finished_ptr = b.create<memref::LoadOp>(l, compute_workers_finished, w);

          // Create async.execute for compute worker
          auto execOp = b.create<async::ExecuteOp>(
              l, TypeRange{}, ValueRange{}, ValueRange{},
              [&](OpBuilder &asyncB, Location asyncL, ValueRange asyncArgs) {
                asyncB.create<func::CallOp>(
                    asyncL, "comet_double_buffer_compute_worker_drive",
                    TypeRange{},
                    ValueRange{A1, A2, A1_tile, A2_tile,
                               A1_offset_ptr, A2_offset_ptr,
                               A_block_rows_ptr, A_block_cols_ptr,
                               A_buffer1_ptr_cast, A_buffer_is_ready_cast,
                               B1, B2, B1_tile, B2_tile,
                               B1_offset_ptr, B2_offset_ptr,
                               B_block_rows_ptr, B_block_cols_ptr,
                               B_buffer1_ptr_cast, B_buffer_is_ready_cast,
                               C_cast, i_start, num_local_i_tiles, is_finished_ptr});
                asyncB.create<async::YieldOp>(asyncL, ValueRange{});
              });

          b.create<async::AddToGroupOp>(l, execOp.getToken(), computeGroup);

          // Update i_start for next worker
          Value forward = b.create<arith::MulIOp>(l, num_local_i_tiles, A1_tile);
          Value i_next = b.create<arith::AddIOp>(l, i_start, forward);
          b.create<scf::YieldOp>(l, ValueRange{i_next});
        });

    // === Launch auxiliary workers ===
    Value auxGroup = builder.create<async::CreateGroupOp>(
        loc, async::GroupType::get(ctx), numAuxVal);

    // For now, launch a single aux worker (can extend to multiple)
    auto auxExecOp = builder.create<async::ExecuteOp>(
        loc, TypeRange{}, ValueRange{}, ValueRange{},
        [&](OpBuilder &asyncB, Location asyncL, ValueRange asyncArgs) {
          asyncB.create<func::CallOp>(
              asyncL, "comet_double_buffer_aux_worker_pull",
              TypeRange{},
              ValueRange{A_cast, A1, A2, A1_tile, A2_tile,
                         A_buffer1s_cast, A_buffer2s_cast,
                         A1_offset_list_cast, A2_offset_list_cast,
                         A_block_rows_list_cast, A_block_cols_list_cast,
                         A_buffer_readys_cast,
                         B_cast, B1, B2, B1_tile, B2_tile,
                         B_buffer1s_cast, B_buffer2s_cast,
                         B1_offset_list_cast, B2_offset_list_cast,
                         B_block_rows_list_cast, B_block_cols_list_cast,
                         B_buffer_readys_cast,
                         c0, numWorkersVal,
                         compute_workers_finished_cast, numWorkersVal});
          asyncB.create<async::YieldOp>(asyncL, ValueRange{});
        });
    builder.create<async::AddToGroupOp>(loc, auxExecOp.getToken(), auxGroup);

    // === Await all workers ===
    builder.create<async::AwaitAllOp>(loc, computeGroup);
    builder.create<async::AwaitAllOp>(loc, auxGroup);

    // === Cleanup: deallocate buffers ===
    // Note: In practice, you may want to add deallocation loops here
    // For brevity, we skip explicit deallocation (MLIR's bufferization can handle it)

    // Replace itree result with C tensor
    itreeOp.replaceAllUsesWith(ValueRange{info.C});
    itreeOp.erase();

    return success();
  }

private:
  /// Initialize an array of double-pointer buffers for each worker
  void initializeBufferArray(OpBuilder &builder, Location loc, Value bufferArray,
                             Value tileSize, Value numWorkers, Value c0, Value c1,
                             Type elemType) {
    auto dynamicMemRefType = MemRefType::get({ShapedType::kDynamic}, elemType);
    auto unrankedElemType = UnrankedMemRefType::get(elemType, {});
    auto wrapperType = MemRefType::get({1}, unrankedElemType);
    auto unrankedWrapperType = UnrankedMemRefType::get(unrankedElemType, {});

    builder.create<scf::ForOp>(
        loc, c0, numWorkers, c1, ValueRange{},
        [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
          // Allocate buffer
          Value buffer = b.create<memref::AllocOp>(l, dynamicMemRefType, tileSize);
          Value bufferCast = b.create<memref::CastOp>(l, unrankedElemType, buffer);

          // Create wrapper memref<1xmemref<*xf64>>
          Value wrapper = b.create<memref::AllocOp>(l, wrapperType);
          b.create<memref::StoreOp>(l, bufferCast, wrapper, c0);

          // Cast wrapper and store in array
          Value wrapperCast = b.create<memref::CastOp>(l, unrankedWrapperType, wrapper);
          b.create<memref::StoreOp>(l, wrapperCast, bufferArray, iv);

          b.create<scf::YieldOp>(l);
        });
  }

  /// Initialize an array of flag memrefs
  void initializeFlagArray(OpBuilder &builder, Location loc, Value flagArray,
                           Value numWorkers, Value c0, Value c1, Value initVal) {
    auto flagType = MemRefType::get({1}, builder.getI8Type());
    auto unrankedI8Type = UnrankedMemRefType::get(builder.getI8Type(), {});

    builder.create<scf::ForOp>(
        loc, c0, numWorkers, c1, ValueRange{},
        [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
          Value flag = b.create<memref::AllocOp>(l, flagType);
          b.create<memref::StoreOp>(l, initVal, flag, c0);
          Value flagCast = b.create<memref::CastOp>(l, unrankedI8Type, flag);
          b.create<memref::StoreOp>(l, flagCast, flagArray, iv);
          b.create<scf::YieldOp>(l);
        });
  }

  /// Initialize an array of index memrefs
  void initializeIndexArray(OpBuilder &builder, Location loc, Value indexArray,
                            Value numWorkers, Value c0, Value c1) {
    auto indexMemType = MemRefType::get({1}, builder.getIndexType());
    auto unrankedIndexType = UnrankedMemRefType::get(builder.getIndexType(), {});

    builder.create<scf::ForOp>(
        loc, c0, numWorkers, c1, ValueRange{},
        [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
          Value indexMem = b.create<memref::AllocOp>(l, indexMemType);
          Value indexMemCast = b.create<memref::CastOp>(l, unrankedIndexType, indexMem);
          b.create<memref::StoreOp>(l, indexMemCast, indexArray, iv);
          b.create<scf::YieldOp>(l);
        });
  }

  /// Declare runtime functions for double buffering
  void declareRuntimeFunctions(OpBuilder &builder, ModuleOp module) {
    MLIRContext *ctx = builder.getContext();

    // Types for function signatures
    auto indexType = IndexType::get(ctx);
    auto unrankedF64Type = UnrankedMemRefType::get(Float64Type::get(ctx), {});
    auto unrankedI8Type = UnrankedMemRefType::get(IntegerType::get(ctx, 8), {});
    auto unrankedIndexType = UnrankedMemRefType::get(indexType, {});
    auto unrankedUnrankedF64Type = UnrankedMemRefType::get(unrankedF64Type, {});
    auto unrankedUnrankedI8Type = UnrankedMemRefType::get(unrankedI8Type, {});
    auto unrankedUnrankedIndexType = UnrankedMemRefType::get(unrankedIndexType, {});

    // comet_double_buffer_compute_worker_drive
    SmallVector<Type> computeWorkerInputs = {
        indexType, indexType, indexType, indexType,  // A1, A2, A1_tile, A2_tile
        unrankedIndexType, unrankedIndexType,        // A1_offset_ptr, A2_offset_ptr
        unrankedIndexType, unrankedIndexType,        // A_block_rows_ptr, A_block_cols_ptr
        unrankedUnrankedF64Type, unrankedI8Type,     // A_buffer1, A_buffer_is_ready
        indexType, indexType, indexType, indexType,  // B1, B2, B1_tile, B2_tile
        unrankedIndexType, unrankedIndexType,        // B1_offset_ptr, B2_offset_ptr
        unrankedIndexType, unrankedIndexType,        // B_block_rows_ptr, B_block_cols_ptr
        unrankedUnrankedF64Type, unrankedI8Type,     // B_buffer1, B_buffer_is_ready
        unrankedF64Type,                              // C
        indexType, indexType,                         // i_start, num_local_i_tiles
        unrankedI8Type                                // is_finished
    };
    getOrInsertFuncDecl(builder, module, "comet_double_buffer_compute_worker_drive",
                        computeWorkerInputs, {});

    // comet_double_buffer_aux_worker_pull
    SmallVector<Type> auxWorkerInputs = {
        unrankedF64Type,                              // A
        indexType, indexType, indexType, indexType,  // A1, A2, A1_tile, A2_tile
        unrankedUnrankedF64Type,                      // A_buffer1s
        unrankedUnrankedF64Type,                      // A_buffer2s
        unrankedUnrankedIndexType,                    // A1_offset_list
        unrankedUnrankedIndexType,                    // A2_offset_list
        unrankedUnrankedIndexType,                    // A_block_rows_list
        unrankedUnrankedIndexType,                    // A_block_cols_list
        unrankedUnrankedI8Type,                       // A_buffer_readys
        unrankedF64Type,                              // B
        indexType, indexType, indexType, indexType,  // B1, B2, B1_tile, B2_tile
        unrankedUnrankedF64Type,                      // B_buffer1s
        unrankedUnrankedF64Type,                      // B_buffer2s
        unrankedUnrankedIndexType,                    // B1_offset_list
        unrankedUnrankedIndexType,                    // B2_offset_list
        unrankedUnrankedIndexType,                    // B_block_rows_list
        unrankedUnrankedIndexType,                    // B_block_cols_list
        unrankedUnrankedI8Type,                       // B_buffer_readys
        indexType, indexType,                         // compute_worker_start, num_local_compute_workers
        unrankedUnrankedI8Type,                       // compute_workers_finished
        indexType                                     // num_compute_workers
    };
    getOrInsertFuncDecl(builder, module, "comet_double_buffer_aux_worker_pull",
                        auxWorkerInputs, {});
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::comet::createConvertToDoubleBufferPass() {
  return std::make_unique<ConvertToDoubleBufferPass>();
}

std::unique_ptr<Pass> mlir::comet::createConvertToDoubleBufferPass(
    int32_t numComputeWorkers, int32_t numAuxWorkers, int32_t tileSize) {
  return std::make_unique<ConvertToDoubleBufferPass>(
      numComputeWorkers, numAuxWorkers, tileSize);
}