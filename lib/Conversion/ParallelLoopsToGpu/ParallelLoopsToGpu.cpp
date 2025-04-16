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


#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"


#include "comet/Conversion/ParallelLoopsToGpu/ParallelLoopsToGpu.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_CLASSES
#include "comet/Conversion/ParallelLoopsToGpu/Passes.h.inc"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

namespace {

bool contains_arg(mlir::Block& block, mlir::BlockArgument arg)
{
    auto store_ops = block.getOps<mlir::memref::StoreOp>();
    for(auto store_op: store_ops)
    {
        for(auto index: store_op.getIndices())
        {
            if (auto affine_expr = llvm::dyn_cast_or_null<mlir::affine::AffineApplyOp>(index.getDefiningOp()))
            {
                for(auto op: affine_expr.getOperands())
                {
                    if(op == arg)
                    {
                        // llvm::errs() << "operation: " << store_op << " contains: " << op << "\n";
                        return true;
                    }
                }
            }
            else if(auto block_arg = mlir::dyn_cast_if_present<mlir::BlockArgument>(index))
            {
                if(block_arg == arg)
                {
                    // llvm::errs() << "operation: " << store_op << " contains: " << op << "\n";
                    return true;
                }
            }
            else if(llvm::isa_and_present<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MinUIOp, mlir::arith::MaxUIOp>(index.getDefiningOp())) {
                for(auto op: index.getDefiningOp()->getOperands())
                {
                    if(op == arg)
                    {
                        return true;
                    }
                }
            }
            else {
                llvm::errs() << "Load operation without affine expression\n";
                block.dump();
                index.dump();
                store_op->dump();
                exit(1);
            }
        }
    }

    return false;
}

bool is_reduction_(mlir::Region& region, mlir::BlockArgument arg)
{
    for (mlir::Block &block : region.getBlocks()) 
    {
        if(contains_arg(block, arg))
        {
            return false;
        }

        for (mlir::Operation &innerOp : block) 
        {
            for (mlir::Region &innerRegion : innerOp.getRegions()) 
            {
                if(!is_reduction_(innerRegion, arg))
                {
                    return false;
                }
            }
        }
    }

    return true;
}

bool is_reduction(mlir::scf::ForOp forOp)
{
    return is_reduction_(forOp.getBodyRegion(), forOp.getBody()->getArgument(0));
}
using namespace mlir;


std::pair<scf::ParallelOp, llvm::SmallVector<scf::ForOp, 2>> tileParallelLoop(ConversionPatternRewriter& rewriter, scf::ParallelOp& op, ArrayRef<int64_t> tileSizes) {
    rewriter.setInsertionPoint(op);
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    SmallVector<Value, 2> tileSizeConstants;
    tileSizeConstants.reserve(op.getUpperBound().size());
    for (size_t i = 0, end = op.getUpperBound().size(); i != end; ++i) {
        if (i < tileSizes.size())
        tileSizeConstants.push_back(
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), tileSizes[i]));
        else
        // Just pick 1 for the remaining dimensions.
        tileSizeConstants.push_back(
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1));
    }

    // Create the outer loop with adjusted steps.
    SmallVector<Value, 2> newSteps;
    newSteps.reserve(op.getStep().size());
    for (auto step : llvm::zip(op.getStep(), tileSizeConstants)) {
        newSteps.push_back(rewriter.create<arith::MulIOp>(op.getLoc(), std::get<0>(step),
                                                std::get<1>(step)));
    }
    auto outerLoop = rewriter.create<scf::ParallelOp>(op.getLoc(), op.getLowerBound(),
                                            op.getUpperBound(), newSteps);
    rewriter.setInsertionPointToStart(outerLoop.getBody());



    // Create the inner loop with adjusted bounds.
    SmallVector<Value, 2> newBounds;
    newBounds.reserve(op.getUpperBound().size());
    bool needInboundCheck = false;
    for (auto [lowerBound, upperBound, newStep, iv, step, tileSizeConstant] :
        llvm::zip(outerLoop.getLowerBound(), outerLoop.getUpperBound(),
                    outerLoop.getStep(), outerLoop.getInductionVars(),
                    op.getStep(), tileSizeConstants)) 
        {
            auto tileSize =
                cast<arith::ConstantIndexOp>(tileSizeConstant.getDefiningOp()).value();
            // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
            auto minMap = AffineMap::get(
                /*dimCount=*/2, /*symbolCount=*/0,
                {getAffineConstantExpr(/*position=*/tileSize, rewriter.getContext()),
                getAffineDimExpr(/*position=*/0, rewriter.getContext()) -
                    getAffineDimExpr(/*position=*/1, rewriter.getContext())},
                rewriter.getContext());
            // Collect the statically known loop bounds
            auto lowerBoundConstant =
                dyn_cast_or_null<arith::ConstantIndexOp>(lowerBound.getDefiningOp());
            auto upperBoundConstant =
                dyn_cast_or_null<arith::ConstantIndexOp>(upperBound.getDefiningOp());
            auto stepConstant =
                dyn_cast_or_null<arith::ConstantIndexOp>(step.getDefiningOp());
            
            // If the loop bounds and the loop step are constant and if the number of
            // loop iterations is an integer multiple of the tile size, we use a static
            // bound for the inner loop.
            if (lowerBoundConstant && upperBoundConstant && stepConstant) {
            auto numIterations = llvm::divideCeil(upperBoundConstant.value() -
                                                        lowerBoundConstant.value(),
                                                    stepConstant.value());
            if (numIterations % tileSize == 0) {
                newBounds.push_back(newStep);
                continue;
            }
        }

        // Otherwise, we dynamically compute the bound for
        // each iteration of the outer loop.
        newBounds.push_back(
            rewriter.create<affine::AffineMinOp>(op.getLoc(), rewriter.getIndexType(), minMap,
                                        ValueRange{upperBound, iv}));
    }

    SmallVector<Value,2> newArgs;
    SmallVector<scf::ForOp, 2> innerLoops;
    SmallVector<Value, 2> inductionVars;
    for(size_t i = 0; i <  tileSizes.size(); i++)
    {
        auto innerLoop = rewriter.create<scf::ForOp>(op.getLoc(), zero, newBounds[i], op.getStep()[i]);
        innerLoop->setAttr("blockSize", rewriter.getUI32IntegerAttr(tileSizes[i]));
        innerLoops.push_back(innerLoop);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        auto new_arg = rewriter.create<arith::AddIOp>(op.getLoc(), innerLoop.getBody()->getArgument(0), outerLoop.getBody()->getArgument(i));
        newArgs.push_back(new_arg);
        inductionVars.push_back(innerLoop.getInductionVar());
    }

    rewriter.eraseOp(op.getBody()->getTerminator());
    rewriter.eraseOp(innerLoops.back().getBody()->getTerminator());
    rewriter.setInsertionPointToStart(innerLoops.back().getBody());

    rewriter.mergeBlocks(op.getBody(), innerLoops.back().getBody(), newArgs);
    rewriter.setInsertionPointToEnd(innerLoops.back().getBody());
    rewriter.create<scf::YieldOp>(op->getLoc());
    rewriter.eraseOp(op);

    return std::make_pair(outerLoop, innerLoops);
}

std::pair<scf::ForOp, scf::ForOp> tileForLoop(OpBuilder& builder, scf::ForOp& op, int64_t tileSize) {
    builder.setInsertionPoint(op);
    auto zero = builder.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value tileSizeConstant = builder.create<arith::ConstantIndexOp>(op.getLoc(), tileSize);

    // Create the outer loop with adjusted steps.
    Value newStep = builder.create<arith::MulIOp>(op.getLoc(), op.getStep(), tileSizeConstant);
    
    auto outerLoop = builder.create<scf::ForOp>(op.getLoc(), op.getLowerBound(), op.getUpperBound(), newStep, op.getInitArgs());
    outerLoop->setAttr("reduceDim", builder.getUnitAttr());
    builder.setInsertionPointToStart(outerLoop.getBody());

    // Create the inner loop with adjusted bounds.
    Value newBound;
    auto tileSizeNew =
        cast<arith::ConstantIndexOp>(tileSizeConstant.getDefiningOp()).value();
    // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
    auto minMap = AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {getAffineConstantExpr(/*position=*/tileSizeNew, builder.getContext()),
        getAffineDimExpr(/*position=*/0, builder.getContext()) -
            getAffineDimExpr(/*position=*/1, builder.getContext())},
        builder.getContext());
    // Collect the statically known loop bounds
    auto lowerBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(outerLoop.getLowerBound().getDefiningOp());
    auto upperBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(outerLoop.getUpperBound().getDefiningOp());
    auto stepConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(op.getStep().getDefiningOp());
        
    // If the loop bounds and the loop step are constant and if the number of
    // loop iterations is an integer multiple of the tile size, we use a static
    // bound for the inner loop.
    if (lowerBoundConstant && upperBoundConstant && stepConstant) {
        auto numIterations = llvm::divideCeil(upperBoundConstant.value() -
                                                    lowerBoundConstant.value(),
                                                stepConstant.value());
        if (numIterations % tileSize == 0) {
            newBound = outerLoop.getStep();
        }
    }

    // Otherwise, we dynamically compute the bound for
    // each iteration of the outer loop.
    newBound =  builder.create<affine::AffineMinOp>(op.getLoc(), builder.getIndexType(), minMap, ValueRange{outerLoop.getUpperBound(), outerLoop.getInductionVar()});

    SmallVector<Value,2> newArgs;
    SmallVector<scf::ForOp, 2> innerLoops;
    SmallVector<Value, 2> inductionVars;
    auto innerLoop = builder.create<scf::ForOp>(op.getLoc(), zero, newBound, op.getStep(), outerLoop.getRegionIterArgs());
    innerLoop->setAttr("blockSize", builder.getUI32IntegerAttr(tileSize));
    innerLoops.push_back(innerLoop);
    builder.setInsertionPointToStart(innerLoop.getBody());
    auto new_arg = builder.create<arith::AddIOp>(op.getLoc(), innerLoop.getBody()->getArgument(0), outerLoop.getBody()->getArgument(0));
    newArgs.push_back(new_arg);
    newArgs.insert(newArgs.end(), innerLoop.getRegionIterArgs().begin(), innerLoop.getRegionIterArgs().end());
    inductionVars.push_back(innerLoop.getInductionVar());
    builder.setInsertionPointToEnd(innerLoops.back().getBody());
    auto temp_yieldOp = builder.create<scf::YieldOp>(op->getLoc());
    
    builder.setInsertionPointToEnd(outerLoop.getBody());
    builder.create<scf::YieldOp>(op->getLoc(), innerLoop.getResults());

    for(auto& inner_op: llvm::make_early_inc_range(op.getBody()->getOperations()))
    {
        inner_op.moveBefore(innerLoops.back().getBody()->getTerminator());
    }
    temp_yieldOp->erase();
    for(auto [old_arg, new_arg] : llvm::zip(op.getBody()->getArguments(), newArgs) )
    {
        old_arg.replaceAllUsesWith(new_arg);
    }
    op->replaceAllUsesWith(outerLoop);
    op->erase();

    return std::make_pair(outerLoop, innerLoop);
}

class ParallelOpToGpu: public mlir::OpConversionPattern<mlir::scf::ParallelOp> {
private:
    int blockX, blockY, blockR;
public:
    using mlir::OpConversionPattern<mlir::scf::ParallelOp>::OpConversionPattern;
    ParallelOpToGpu(mlir::MLIRContext* ctx, int blockX, int blockY, int blockR) : mlir::OpConversionPattern<mlir::scf::ParallelOp>(ctx), blockX(blockX), blockY(blockY), blockR(blockR) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::scf::ParallelOp parOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        
        SmallVector<Operation*, 8> worklist;
        scf::ParallelOp otherParOp = NULL;
        for(auto& op: parOp.getBody()->without_terminator())
        {
            if(auto inner_parOp = mlir::dyn_cast<scf::ParallelOp>(op))
            {
                otherParOp = inner_parOp;
                break;
            }
            else 
            {
                worklist.push_back(&op);
            }
        }
        SmallVector<Operation*, 8> canMove;
        if(otherParOp != NULL)
        {
            for(auto op: worklist)
            {
                if(isMemoryEffectFree(op) && op->getRegions().size() == 0)
                {
                    canMove.push_back(op);
                }
            }
        }

        if(worklist.size() == canMove.size())
        {
            if(!canMove.empty())
            {
                for(auto op: canMove)
                {
                    rewriter.setInsertionPointToStart(otherParOp.getBody());
                    auto clone = rewriter.clone(*op);
                    rewriter.replaceAllOpUsesWith(op, clone);
                    rewriter.eraseOp(op);
                }
                rewriter.setInsertionPoint(parOp);
                auto combinedParOp = rewriter.create<scf::ParallelOp>(parOp->getLoc(), ValueRange({parOp.getLowerBound().front(), otherParOp.getLowerBound().front()}), ValueRange({parOp.getUpperBound().front(), otherParOp.getUpperBound().front()}), ValueRange({parOp.getStep().front(), otherParOp.getStep().front()}));
                Operation* terminator =  combinedParOp.getBody()->getTerminator();
                rewriter.eraseOp(terminator);
                terminator = parOp.getBody()->getTerminator();
                rewriter.mergeBlocks(parOp.getBody(), combinedParOp.getBody(), combinedParOp.getInductionVars().front());
                rewriter.eraseOp(parOp);
                rewriter.eraseOp(terminator);
                terminator = otherParOp.getBody()->getTerminator();
                rewriter.mergeBlocks(otherParOp.getBody(), combinedParOp.getBody(), combinedParOp.getInductionVars().back());
                rewriter.eraseOp(otherParOp);
                parOp = combinedParOp;
            }
        }

        llvm::SmallVector<int64_t, 3> allTileSizes = {blockY, blockX};
        llvm::SmallVector<int64_t, 3> tileSizes; 
        std::copy(allTileSizes.begin(), allTileSizes.begin() + parOp.getInductionVars().size(), std::back_inserter(tileSizes));
        SmallVector<Attribute, 2> allStringAttrs = {rewriter.getAttr<mlir::StringAttr>("dimY_grid"), rewriter.getAttr<mlir::StringAttr>("dimX_grid")};
        SmallVector<Attribute, 2> stringAttrs;
        std::copy(allStringAttrs.begin(), allStringAttrs.begin() + parOp.getInductionVars().size(), std::back_inserter(stringAttrs));
        
        auto dim0 = rewriter.getAffineDimExpr(0);
        auto dim1 = rewriter.getAffineDimExpr(0);
        auto yMap = mlir::AffineMap::get(1, 0, dim0);
        auto xMap = mlir::AffineMap::get(1, 0, dim1);
        SmallVector<Attribute, 2> allGpuAttrs = {mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockY, yMap, yMap), mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockX, xMap, xMap)};
        SmallVector<Attribute, 2> gpuAttrs;
        std::copy(allGpuAttrs.begin(), allGpuAttrs.begin() + parOp.getInductionVars().size(), std::back_inserter(gpuAttrs));
        auto tiledLoop = tileParallelLoop(rewriter, parOp, tileSizes);

        tiledLoop.first->setAttr("parallelDim", rewriter.getArrayAttr(stringAttrs));
        tiledLoop.first->setAttr("mapping", rewriter.getArrayAttr(gpuAttrs));

        return success();
    }
};
class ConvertParallelLoopsToGpu: public CometParallelLoopsToGpuBase<ConvertParallelLoopsToGpu> {
public:
    ConvertParallelLoopsToGpu() = default;
    ConvertParallelLoopsToGpu(int blockX, int blockY, int blockR) {
        this->blockX = blockX;
        this->blockY = blockY;
        this->blockR = blockR;
    }

    void runOnOperation() override {
        mlir::MLIRContext *context = &getContext();
        mlir::func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration())
        {
            return;
        }

        mlir::RewritePatternSet patterns(context);
        patterns.insert<ParallelOpToGpu>(context, blockX, blockY, blockR);
        
        mlir::ConversionTarget target(*context);
        target.addLegalDialect<mlir::memref::MemRefDialect, mlir::arith::ArithDialect,  mlir::affine::AffineDialect, mlir::scf::SCFDialect>();
        target.addLegalOp<mlir::scf::ReduceOp>();
        target.addDynamicallyLegalOp<mlir::scf::ParallelOp>([](mlir::scf::ParallelOp op) -> bool {
            return op->hasAttr("parallelDim");
        });

        if (mlir::failed(mlir::applyPartialConversion(funcOp, target, std::move(patterns))))
        {
            return signalPassFailure();
        }


        llvm::SmallVector<scf::ForOp, 2> toReduceForOps;
        funcOp->walk([&toReduceForOps](mlir::scf::ForOp forOp) {
            if(forOp->getParentOfType<scf::ParallelOp>())
            {
                if(is_reduction(forOp))
                {
                    toReduceForOps.push_back(forOp);
                }
            }
        });

        OpBuilder builder(funcOp);

        for(scf::ForOp forOp: llvm::make_early_inc_range(toReduceForOps))
        {
            llvm::SmallVector<mlir::Value, 4> inductionVars;
            llvm::SmallVector<mlir::Operation*, 4> loopInvMemOps;
            inductionVars.push_back(forOp.getInductionVar());

            forOp->walk([&loopInvMemOps](Operation* op){
                if(mlir::isa<mlir::memref::StoreOp,mlir::memref::LoadOp>(op))
                {
                    loopInvMemOps.push_back(op);
                }
            });


            for(auto inductionVar: inductionVars)
            {
                for(auto user: llvm::make_early_inc_range(inductionVar.getUsers()))
                {
                    if(mlir::isa<mlir::memref::StoreOp,mlir::memref::LoadOp>(user))
                    {
                        auto it = std::find(loopInvMemOps.begin(), loopInvMemOps.end(), user);
                        loopInvMemOps.erase(it);
                    }
                    for(auto res: user->getResults())
                    {
                        inductionVars.push_back(res);
                    }
                }
            }

            llvm::SmallMapVector<mlir::Value, llvm::SmallVector<mlir::Operation*, 2>, 4> loadStorePairs;
            for(auto memOp: loopInvMemOps)
            {
                if(memref::StoreOp storeOp = dyn_cast<memref::StoreOp>(memOp))
                {
                    auto it = loadStorePairs.find(storeOp.getMemRef());
                    assert(it!= loadStorePairs.end());
                    it->second.push_back(storeOp);
                }
                else if(memref::LoadOp loadOp = dyn_cast<memref::LoadOp>(memOp))
                {
                    auto it = loadStorePairs.find(loadOp.getMemRef());
                    assert(it == loadStorePairs.end());
                    loadStorePairs[loadOp.getMemRef()].push_back(loadOp);
                    // it->second.push_back(loadOp);
                }
                else 
                {
                    assert(false && "UNREACHABLE. Should vectore should only contain store or load operations");
                }
            }
            auto pairs = loadStorePairs.front().second;
            pairs[0]->moveBefore(forOp);
            pairs[1]->moveAfter(forOp);
            builder.setInsertionPoint(forOp);
            auto newForOp = builder.create<scf::ForOp>(forOp->getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), ValueRange(pairs[0]->getResult(0)));
            // newForOp->dump();
            builder.setInsertionPointToEnd(newForOp.getBody());
            builder.create<scf::YieldOp>(forOp->getLoc(), pairs[1]->getOperand(0));
            pairs[0]->getResult(0).replaceAllUsesExcept(newForOp.getRegionIterArg(0), newForOp);
            pairs[1]->setOperand(0, newForOp.getResult(0));
            forOp.getBody()->getTerminator()->erase();
            for(auto& op: llvm::make_early_inc_range(forOp.getBody()->getOperations()))
            {
                op.moveBefore(newForOp.getBody()->getTerminator());
            }
            forOp.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
            auto [outerLoop, innerLoop] = tileForLoop(builder, newForOp, blockR);
            forOp->erase();
        }
    }
}; 
}


std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::comet::createConvertParallelLoopsToGpuPass() {
    return std::make_unique<ConvertParallelLoopsToGpu>();
}


std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::comet::createConvertParallelLoopsToGpuPass(int blockX, int blockY, int blockR) {
    return std::make_unique<ConvertParallelLoopsToGpu>(blockX, blockY, blockR);
}
