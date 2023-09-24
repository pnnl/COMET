//===- GPUAllocateGlobalMem.cpp------===//
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
// Allocates global memory for GPU shared memory. 
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Conversion/TensorAlgebraToSCF/TensorAlgebraToSCF.h"
#include "comet/Conversion/SCFToGPU/SCFToGPU.h"
#include "comet/Conversion/Utils/GPUUtils.h"
#include "comet/Conversion/Utils/MarkerUtils.h"
#include "comet/Conversion/TensorAlgebraToIndexTree/TensorAlgebraToIndexTree.h"
#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"


#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::gpu;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_GPUAllocateGlobalMem
// #define DEBUG_MODE_GPUAllocateGlobalMem
// #endif

#ifdef DEBUG_MODE_GPUAllocateGlobalMem
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


namespace {

struct convertSharedMemAllocOp : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {

    if ((allocOp.getType().getMemorySpaceAsInt()) != 3) {
      comet_debug() << "found normal allocOp: \n";
      comet_pdump(allocOp);
      return failure();
    }

    comet_debug() << "found gpu alloc op\n";
    comet_pdump(allocOp);

    ArrayRef<int64_t> shape = allocOp.getType().getShape();
    if (llvm::any_of(shape,
                     [](int64_t dim) { return dim == ShapedType::kDynamic; })) {
      comet_debug() << "dynamic not supported!\n";
      return failure();
    }

    uint64_t alignement;
    if (std::optional<uint64_t> alignementInfo = allocOp.getAlignment()) {
      alignement = alignementInfo.value();
    } else {
      // If no alignment specified align at least to the size of an element.
       Type elType = allocOp.getType().getElementType();
       if (auto shapeType = llvm::dyn_cast<ShapedType>(elType))
         alignement =
             shapeType.getNumElements() * shapeType.getElementTypeBitWidth() / 8;
      // else if (elType.isIndex()) {
      //   auto mod = allocOp->getParentOfType<ModuleOp>();
      //   LowerToLLVMOptions options(mod.getContext(), DataLayout(mod));
      //   alignement = options.getIndexBitwidth() / 8;
      // } else
      else
         alignement = elType.getIntOrFloatBitWidth() / 8;
    }
    // In CUDA workgroup memory is represented by a global variable.
    MemRefType allocType = allocOp.getType();
    auto funcOp = allocOp->getParentOfType<func::FuncOp>();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(moduleOp);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&moduleOp.front());
    auto global = rewriter.create<memref::GlobalOp>(
        funcOp.getLoc(), "__shared_memory__",
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/allocType,
        /*initial_value=*/ElementsAttr(),
        /*constant=*/false,
        /*alignment=*/rewriter.getI64IntegerAttr(alignement));
    symbolTable.insert(global);

    rewriter.setInsertionPointToStart(&(*funcOp.getFunctionBody().begin()));
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(allocOp, global.getType(),
                                                     global.getName());

    comet_debug() << "I was able to successfully create globals and replace old ops with get-global op\n";
    return success();                              
  }

};

} // namespace


namespace {
struct GPUAllocateGlobalMemPass
    : public PassWrapper<GPUAllocateGlobalMemPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUAllocateGlobalMemPass)

private:
  
 public:
  GPUAllocateGlobalMemPass() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final;

};
}  // namespace

void GPUAllocateGlobalMemPass::runOnOperation() {

  comet_debug() << "[DEBUG][GPU][Start] GPUAllocateGlobalMemPass\n";
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  target.addLegalDialect<memref::MemRefDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<convertSharedMemAllocOp>(&getContext());

  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

  comet_debug() << "[DEBUG][GPU][End] GPUAllocateGlobalMemPass\n";
}

/// Create a pass to allocate global memory.
std::unique_ptr<Pass> mlir::comet::createGPUAllocateGlobalMemPass() {
  return std::make_unique<GPUAllocateGlobalMemPass>();
}