//===- GPUReduceBankConflicts.cpp------===//
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTICE: The source code has been modified for integration with COMET.
//===----------------------------------------------------------------------===//
//
// This file does padding to reduce shared memory bank conflicts.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Conversion/TensorAlgebraToSCF/TensorAlgebraToSCF.h"
#include "comet/Conversion/SCFToGPU/SCFToGPU.h"
#include "comet/Conversion/Utils/GPUUtils.h"
#include "comet/Conversion/TensorAlgebraToIndexTree/TensorAlgebraToIndexTree.h"
#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_GPUReduceBankConflicts
// #define DEBUG_MODE_GPUReduceBankConflicts
// #endif

#ifdef DEBUG_MODE_GPUReduceBankConflicts
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

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

/// Padd out the inner dimension of the allocOp in order reduce the chances to
/// have bank conflicts when reading 2D shapes within shared memory.
static void padAlloc(MLIRContext *context, memref::AllocOp allocOp,
                     int64_t paddingSizeBits) {
  
  comet_debug() << "padAlloc start\n";

  int64_t innerDim = allocOp.getType().getShape().back();
  if (ShapedType::isDynamic(innerDim)) return;
  Type elType = allocOp.getType().getElementType();
  unsigned bitwidth =
      mlir::DataLayout::closest(allocOp).getTypeSizeInBits(elType);
  // Pad with 128bits==16bytes so that accesses are still aligned on 16bytes.
  int64_t paddingSize = paddingSizeBits / bitwidth;
  SmallVector<int64_t> shape = llvm::to_vector(allocOp.getType().getShape());
  shape.back() = shape.back() + paddingSize;
  MemRefType allocType =
      MemRefType::get(shape, elType, MemRefLayoutAttrInterface{},
                      allocOp.getType().getMemorySpace());
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(allocOp);
  Location loc = allocOp.getLoc();
  Value paddedAlloc = rewriter.create<memref::AllocOp>(loc, allocType);
  SmallVector<int64_t> offsets(shape.size(), 0);
  SmallVector<int64_t> strides(shape.size(), 1);
  Value subview = rewriter.create<memref::SubViewOp>(
      loc, paddedAlloc, offsets, allocOp.getType().getShape(), strides);
  replaceMemrefUsesAndPropagateType(rewriter, loc, allocOp, subview);
  rewriter.eraseOp(allocOp);

  comet_debug() << "padAlloc end\n";
}

namespace {
struct GPUReduceBankConflictsPass
    : public PassWrapper<GPUReduceBankConflictsPass, OperationPass<func::FuncOp>> {


  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUReduceBankConflictsPass)

private:
  int64_t paddingSizeBits;

 public:
  GPUReduceBankConflictsPass(int64_t paddingSizeBits)
      : paddingSizeBits(paddingSizeBits) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    comet_debug() << "[DEBUG][GPU][Start] GPUReduceBankConflicts\n";
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    SmallVector<memref::AllocOp> sharedMemAllocs;
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocOp allocOp) {
      // TODO: need to update all shared memory checks as below
      if ((allocOp.getType().getMemorySpaceAsInt() == 3) && 
          allocOp.getType().hasStaticShape()) {
        sharedMemAllocs.push_back(allocOp);
      }
    });
    for (memref::AllocOp alloc : sharedMemAllocs)
      padAlloc(context, alloc, paddingSizeBits);

    comet_debug() << "[DEBUG][GPU][End] GPUReduceBankConflicts\n";
  }
};

}  // namespace


/// Create a pass to do padding
std::unique_ptr<Pass>
 mlir::comet::createGPUReduceSharedMemoryBankConflictsPass(int64_t paddingSizeBits) {
  return std::make_unique<GPUReduceBankConflictsPass>(paddingSizeBits);
}