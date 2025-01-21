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
#include <__fwd/get.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"


#include "comet/Dialect/Utils/Utils.h"
#include "comet/Conversion/ParallelLoopsToGpu/ParallelLoopsToGpu.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#define GEN_PASS_CLASSES
#include "comet/Conversion/ParallelLoopsToGpu/Passes.h.inc"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

namespace {

/// For memref::Load/StoreOp, changes indexing from >=2D to 1D by applying an affine expression.
/// For example, for  A[i,j] -> A[i*dim(A,0) + j]
template< typename T>
mlir::Value reassociateIndices(T op, mlir::OpBuilder& builder)
{
    int currDimPos = 0, currSymPos = 0;
    std::vector<mlir::Value> dimVals, symVals;

    /// Start from the right-most index as it does not need to be multiplied with anything
    mlir::Value newIndex = op.getIndices().back();
    mlir::AffineExpr expr;
    
    /// Next, iterate over the idices right to left (skipping the right-most)
    /// multiply them by the size of the next dimension and add them with the 
    /// running index. 
    for(int64_t i = op.getIndices().size()-2; i >= 0; i--)
    {
        mlir::Value stride;
        for(size_t j = i + 1; j < op.getIndices().size(); j++)
        {
            builder.setInsertionPointAfterValue(op.getMemref());
            auto dimSize = builder.create<mlir::memref::DimOp>(op->getLoc(), op.getMemRef(), j)->getResult(0);
            if(!stride)
            {   
                expr = mlir::getAffineSymbolExpr(currSymPos++, op->getContext());
                symVals.push_back(dimSize);
                // stride = dimSize;
            }
            else 
            {
                symVals.push_back(dimSize);
                expr = expr * mlir::getAffineSymbolExpr(currSymPos++, op->getContext());
                // stride = builder.create<mlir::arith::MulIOp>(op->getLoc(), dimSize, stride);
            }
        }

        if(mlir::isa<mlir::BlockArgument>(op.getIndices()[i])) 
        {
            dimVals.push_back(op.getIndices()[i]);
            expr =  expr * mlir::getAffineDimExpr(currDimPos++, op->getContext());
        }
        else
        {
            symVals.push_back(op.getIndices()[i]);
            expr =  expr * mlir::getAffineSymbolExpr(currSymPos++, op->getContext());
        }

        if(mlir::isa<mlir::BlockArgument>(newIndex)) 
        {
            dimVals.push_back(newIndex);
            expr =  mlir::getAffineDimExpr(currDimPos++, op->getContext()) + expr;
        }
        else
        {
            symVals.push_back(newIndex);
            expr =  mlir::getAffineSymbolExpr(currSymPos++, op->getContext()) + expr;
        }

        // expr =  mlir::getAffineSymbolExpr(currSymPos++, op->getContext()) + expr;
        // newIndex = builder.create<mlir::arith::AddIOp>(op->getLoc(), builder.create<mlir::arith::MulIOp>(op->getLoc(), op.getIndices()[i], stride), newIndex);
    }
    builder.setInsertionPoint(op);
    auto affineIndex = mlir::AffineMap::get(currDimPos, currSymPos, {expr}, op->getContext());
    dimVals.insert(dimVals.end(), symVals.begin(), symVals.end());
    auto index = builder.create<mlir::affine::AffineApplyOp>(op->getLoc(), affineIndex, dimVals);

    return index;
}

void collapseMemrefAndUsers(mlir::Value val, mlir::OpBuilder& builder)
{
    
    auto memref = mlir::cast<mlir::MemRefType>(val.getType());
    if (memref.getRank() == 1)
    {
        return;
    }

    llvm::SmallVector<llvm::SmallVector<int64_t,2>,1> indices;
    indices.push_back(llvm::SmallVector<int64_t,2>());
    for(int64_t i = 0; i < memref.getRank(); i++)
    {
        indices[0].push_back(i);
    }

    /// Collapse memref to 1D
    auto collapsedMemref = builder.create<mlir::memref::CollapseShapeOp>(val.getLoc(), val, mlir::ArrayRef(indices));

    for(auto u: llvm::make_early_inc_range(val.getUsers()))
    {
        if(mlir::isa<mlir::memref::LoadOp>(u))
        {
            auto oldOp = mlir::cast<mlir::memref::LoadOp>(u);
            builder.setInsertionPoint(oldOp);
            auto newIndex = reassociateIndices(oldOp, builder);
            auto newOp = builder.create<mlir::memref::LoadOp>(u->getLoc(), collapsedMemref, newIndex );
            u->replaceAllUsesWith(newOp);
            u->erase();
        }
        else if(mlir::isa<mlir::memref::StoreOp>(u))
        {
            auto oldOp = mlir::cast<mlir::memref::StoreOp>(u);
            builder.setInsertionPoint(oldOp);
            auto newIndex = reassociateIndices(mlir::cast<mlir::memref::StoreOp>(u), builder);
            auto newOp = builder.create<mlir::memref::StoreOp>(u->getLoc(), oldOp.getValueToStore(), collapsedMemref, newIndex );
            u->replaceAllUsesWith(newOp);
            u->erase();
        }
    }
}

bool contains_arg(mlir::Block& block, mlir::BlockArgument arg)
{
    auto store_ops = block.getOps<mlir::memref::StoreOp>();
    for(auto store_op: store_ops)
    {
        for(auto index: store_op.getIndices())
        {
            if (auto affine_expr = mlir::dyn_cast_if_present<mlir::affine::AffineApplyOp>(index.getDefiningOp()))
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
            else if(auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(index))
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
                // index.dump();
                // store_op->dump();
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
        innerLoop.getBody()->getArgument(0).dump();
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

class ParallelOpToGpu: public mlir::OpConversionPattern<mlir::scf::ParallelOp> {
private:
[[maybe_unused]]  int blockX, blockY, blockR;
    mlir::tensorAlgebra::TargetDevice target;
public:
    using mlir::OpConversionPattern<mlir::scf::ParallelOp>::OpConversionPattern;
    ParallelOpToGpu(mlir::MLIRContext* ctx, int blockX, int blockY, int blockR, mlir::tensorAlgebra::TargetDevice target) : mlir::OpConversionPattern<mlir::scf::ParallelOp>(ctx), blockX(blockX), blockY(blockY), blockR(blockR), target(target) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::scf::ParallelOp parOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        
        std::vector<int64_t> tileSizes = {blockY, blockX};
        auto tiledLoop = tileParallelLoop(rewriter, parOp, tileSizes);
        SmallVector<Attribute, 2> string_attrs = {rewriter.getAttr<mlir::StringAttr>("dimY_grid"), rewriter.getAttr<mlir::StringAttr>("dimX_grid")};
        auto dim0 = rewriter.getAffineDimExpr(0);
        auto dim1 = rewriter.getAffineDimExpr(0);
        auto yMap = mlir::AffineMap::get(1, 0, dim0);
        auto xMap = mlir::AffineMap::get(1, 0, dim1);
        SmallVector<Attribute, 2> gpu_attrs = {mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockY, yMap, yMap), mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockX, xMap, xMap)};
        for(size_t i= 0; i < tiledLoop.first.getInductionVars().size(); i++)
        {
            tiledLoop.first->setAttr("parallelDim", rewriter.getArrayAttr(string_attrs));
            tiledLoop.first->setAttr("mapping", rewriter.getArrayAttr(gpu_attrs));
        }
        // y_loop_grid->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_grid"));
        // newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockY, map, map);
        return success();
        // for()
        // if(parOp->getAttrOfType<mlir::StringAttr>("parallelDim"))
        // {
        //     rewriter.startRootUpdate(parOp);

        //     bool changed = false;
        //     for(auto iter_arg: llvm::zip(parOp.getBody()->getArguments(), parOp.getUpperBound()))
        //     {

        //         for(auto u: std::get<0>(iter_arg).getUsers())
        //         {
        //             bool needsChange = false;
        //             if(!llvm::dyn_cast<mlir::arith::MinUIOp>(u))
        //             {
        //                 for(auto uu: u->getUsers())
        //                 {
        //                     if(!llvm::dyn_cast<mlir::arith::MinUIOp>(uu))
        //                     {
        //                         needsChange = true;
        //                     }
        //                 }

        //                 if(needsChange)
        //                 {
        //                     rewriter.setInsertionPointToStart(parOp.getBody());
        //                     auto minOp = rewriter.create<mlir::arith::MinUIOp>(std::get<0>(iter_arg).getLoc(), std::get<0>(iter_arg), std::get<1>(iter_arg));
        //                     if(parOp->getAttrOfType<mlir::StringAttr>("parallelDim").strref() == "dimY_grid" || parOp->getAttrOfType<mlir::StringAttr>("parallelDim").strref() == "dimY_block")
        //                     {
        //                         minOp->setAttr("GuardY", rewriter.getUnitAttr());
        //                     }
        //                     else 
        //                     {
        //                         minOp->setAttr("GuardX", rewriter.getUnitAttr());
        //                     }
        //                     changed = true;
        //                     std::get<0>(iter_arg).replaceAllUsesExcept(minOp, minOp);
        //                     break;
        //                 }
        //             }
        //         }
        //     }
        //     if(changed)
        //     {
        //         rewriter.finalizeRootUpdate(parOp);
        //     }
        //     else 
        //     {
        //         rewriter.cancelRootUpdate(parOp);
        //     }

        //     return mlir::success(changed);
        // }
        if(parOp->getAttrOfType<mlir::StringAttr>("parallelDim") || parOp->getAttr("mapping"))
        {
            return mlir::failure();
        }

        auto map = rewriter.getDimIdentityMap();
        mlir::gpu::ParallelLoopDimMappingAttr newAttr;

        // bool no_inner_parallel = parOp.getBody()->getOps<mlir::scf::ParallelOp>().empty();

        if(parOp.getLowerBound().size() > 2)
        {
            return mlir::failure();
        }
        else if (parOp.getLowerBound().size() == 2)
        {
            auto block_size_y = rewriter.create<mlir::arith::ConstantIndexOp>(parOp->getLoc(), blockY );
            auto block_size_x = rewriter.create<mlir::arith::ConstantIndexOp>(parOp->getLoc(), blockX );
            auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(parOp->getLoc(), 1);
            auto upperBound0 = CeilDivUIOp(rewriter, parOp->getLoc(), parOp.getUpperBound().front(), block_size_y);
            auto upperBound1 = CeilDivUIOp(rewriter, parOp->getLoc(), parOp.getUpperBound().back(), block_size_x);
            
            comet_debug() << upperBound0;
            auto y_loop_grid = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound().front(), upperBound0->getResult(0), c1->getResult(0));
            rewriter.setInsertionPointToStart(y_loop_grid.getBody());
        
            y_loop_grid->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_grid"));
            newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockY, map, map);
            y_loop_grid->setAttr("mapping", mlir::ArrayAttr::get(parOp->getContext(),  newAttr) );

            auto y_loop_block = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound().front(), block_size_y->getResult(0), c1->getResult(0));
            
            y_loop_block->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_block"));
            // auto x_loop = rewriter.create<mlir::scf::ParallelOp>(forOp->getLoc(), forOp.getLowerBound(), upperBound1->getResult(0), c1->getResult(0));
            rewriter.setInsertionPointToStart(y_loop_block.getBody());
            newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::ThreadY, map, map);
            y_loop_block->setAttr("mapping", mlir::ArrayAttr::get(parOp->getContext(),  newAttr) );

            auto res = mlir::getAffineDimExpr(0, parOp->getContext()) * mlir::getAffineSymbolExpr(0, parOp->getContext())  + mlir::getAffineSymbolExpr(1, parOp->getContext());
            // auto res = mlir::getAffineDimExpr(0, forOp->getContext());
            // auto res1 = mlir::getAffineDimExpr(1, forOp->getContext());
            comet_debug() << res;
            auto affineIndex = mlir::AffineMap::get(1, 2, {res}, parOp->getContext());
            comet_debug() << affineIndex;
            std::vector<mlir::Value> range = { y_loop_grid.getBody()->getArgument(0), block_size_y->getResult(0),  y_loop_block.getBody()->getArgument(0)};
            mlir::Operation* newIndexY = rewriter.create<mlir::affine::AffineApplyOp>(parOp->getLoc(), affineIndex, range);
            if(target == mlir::tensorAlgebra::TargetDevice::GPU)
            {
                newIndexY = rewriter.create<mlir::arith::MinUIOp>(parOp->getLoc(), newIndexY->getResult(0), parOp.getUpperBound().front());
                newIndexY->setAttr("GuardY", rewriter.getUnitAttr());
                // auto newIndexY = rewriter.create<mlir::affine::AffineApplyOp>(forOp->getLoc(), affineIndex, range);
                // auto newIndexY = rewriter.create<mlir::arith::AddIOp>(forOp->getLoc(),rewriter.create<mlir::arith::MulIOp>(forOp->getLoc(), y_loop_grid.getBody()->getArgument(0), block_size_y), y_loop_block.getBody()->getArgument(0));
                // rewriter.setInsertionPoint(newIndexY);
            }
            
            rewriter.replaceAllUsesWith(parOp.getBody()->getArgument(0), newIndexY->getResult(0));

            // auto upperBound0 = rewriter.create<mlir::arith::CeilDivUIOp>(parOp->getLoc(), parOp.getUpperBound(), block_size_y);
            // auto y_loop_grid = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound(), upperBound0->getResult(0), c1->getResult(0));
            auto x_loop_grid = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound().back(), upperBound1->getResult(0), c1->getResult(0));
            x_loop_grid->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimX_grid"));
            newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockX, map, map);
            x_loop_grid->setAttr("mapping", mlir::ArrayAttr::get(parOp->getContext(),  newAttr) );

            rewriter.setInsertionPointToStart(x_loop_grid.getBody());
            // auto y_loop_block = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound(), block_size_y->getResult(0), c1->getResult(0));
            auto x_loop_block = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound().back(), block_size_x->getResult(0), c1->getResult(0));
            x_loop_block->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimX_block"));
            newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::ThreadX, map, map);
            x_loop_block->setAttr("mapping", mlir::ArrayAttr::get(parOp->getContext(),  newAttr) );
            rewriter.setInsertionPointToStart(x_loop_block.getBody());
            
            res = mlir::getAffineDimExpr(0, parOp->getContext()) * mlir::getAffineSymbolExpr(0, parOp->getContext())  + mlir::getAffineSymbolExpr(1, parOp->getContext());
            affineIndex = mlir::AffineMap::get(1, 2, {res}, parOp->getContext());
            range = { x_loop_grid.getBody()->getArgument(0), block_size_x->getResult(0),  x_loop_block.getBody()->getArgument(0)};
            // auto newIndexX = rewriter.create<mlir::affine::AffineApplyOp>(parOp->getLoc(), affineIndex, range);
            mlir::Operation* newIndexX = rewriter.create<mlir::affine::AffineApplyOp>(parOp->getLoc(), affineIndex, range);
            if(target == mlir::tensorAlgebra::TargetDevice::GPU)
            {
                newIndexX = rewriter.create<mlir::arith::MinUIOp>(parOp->getLoc(), newIndexX->getResult(0), parOp.getUpperBound().back());
                newIndexX->setAttr("GuardX", rewriter.getUnitAttr());
            }
            
            rewriter.replaceAllUsesWith(parOp.getBody()->getArgument(1), newIndexX->getResult(0));
            
            // auto newIndexX = rewriter.create<mlir::arith::AddIOp>(parOp->getLoc(),rewriter.create<mlir::arith::MulIOp>(parOp->getLoc(), x_loop_grid.getBody()->getArgument(0), block_size_x), x_loop_block.getBody()->getArgument(0));
            rewriter.setInsertionPointAfter(newIndexX);
            rewriter.eraseOp(parOp.getBody()->getTerminator());
            if(target == mlir::tensorAlgebra::TargetDevice::GPU)
            {
                rewriter.inlineBlockBefore(parOp.getBody(), x_loop_block.getBody()->getTerminator(), {newIndexY->getResult(0), newIndexX->getResult(0)});
            }
            else 
            {
                auto withinY = rewriter.create<mlir::arith::CmpIOp>(parOp.getLoc(), mlir::arith::CmpIPredicate::slt, newIndexY->getResult(0), parOp.getUpperBound().front());
                auto withinX = rewriter.create<mlir::arith::CmpIOp>(parOp.getLoc(), mlir::arith::CmpIPredicate::slt, newIndexX->getResult(0), parOp.getUpperBound().back());
                auto withinXandY = rewriter.create<mlir::arith::AndIOp>(parOp.getLoc(), withinY, withinX);

                auto ifOp = rewriter.create<mlir::scf::IfOp>(parOp->getLoc(), withinXandY);
                rewriter.inlineBlockBefore(parOp.getBody(), ifOp.getBody()->getTerminator(), {newIndexY->getResult(0), newIndexX->getResult(0)});
            }


            rewriter.eraseOp(parOp);
            return mlir::success();
        }
        else if (!mlir::isa<mlir::scf::ForOp,mlir::scf::ParallelOp>(parOp->getParentOp()) && !(parOp->getParentOp() && parOp->getParentOp()->hasAttrOfType<mlir::UnitAttr>("GuardY"))) // Y level loop
        {
            // std::vector<int64_t> tileSizes = {blockY};
            // std::pair<scf::ParallelOp, scf::ForOp> tiledLoop = tileParallelLoop(rewriter, parOp, tileSizes);
            // scf::ParallelOp tiledLoop = tileParallelLoop(rewriter, parOp, tileSizes);
            // tiledLoop->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_grid"));
            
            // tiledLoop.first->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_grid"));
            // tiledLoop.second->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_block"));
            
            // newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockY, map, map);
            // tiledLoop.first->setAttr("mapping",  mlir::ArrayAttr::get(tiledLoop.first->getContext(),  newAttr));
            
            // newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::ThreadY, map, map);
            // tiledLoop.second->setAttr("mapping",  mlir::ArrayAttr::get(tiledLoop.first->getContext(),  newAttr));
            // tiledLoop.first.dump();

            // rewriter.modifyOpInPlace(parOp, [&]{
            //     std::pair<mlir::scf::ParallelOp, mlir::scf::ParallelOp> parOps =  mlir::scf::tileParallelLoop(parOp, tileSizes, false);
            //     parOps.first->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_grid"));
            //     parOps.second->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_block"));
            // });
            // rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOps.first);

            // // auto block_size_x = rewriter.create<mlir::arith::ConstantOp>(forOp->getLoc(), rewriter.getIndexType() , rewriter.getIndexAttr(blockX) );
            // auto block_size_y = rewriter.create<mlir::arith::ConstantIndexOp>(parOp->getLoc(), blockY );
            // auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(parOp->getLoc(), 1);
            // // auto upperBound0 = rewriter.create<mlir::arith::CeilDivUIOp>(parOp->getLoc(), parOp.getUpperBound().front(), block_size_y);

            // auto canonicalized_upper_bound = rewriter.create<mlir::arith::SubIOp>(parOp->getLoc(), parOp.getUpperBound()[0], parOp.getLowerBound()[0]);

            // auto canonicalized_upper_bound_for_ceil = rewriter.create<mlir::arith::AddIOp>(parOp->getLoc(), canonicalized_upper_bound, rewriter.create<mlir::arith::SubIOp>(parOp->getLoc(), block_size_y, c1));
            // auto outer_upper_bound = rewriter.create<mlir::arith::DivSIOp>(parOp->getLoc(), canonicalized_upper_bound_for_ceil, block_size_y);
            
            // comet_debug() << outer_upper_bound;
            // // auto upperBound1 = rewriter.create<mlir::arith::CeilDivUIOp>(forOp->getLoc(), forOp.getUpperBound(), block_size_x);
            // // auto y_loop_grid = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound().front(), upperBound0->getResult(0), c1->getResult(0));
            // auto y_loop_grid = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound().front(), outer_upper_bound.getResult(), c1->getResult(0));
            // rewriter.setInsertionPointToStart(y_loop_grid.getBody());
        
            // y_loop_grid->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_grid"));
            // newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockY, map, map);
            // y_loop_grid->setAttr("mapping", mlir::ArrayAttr::get(parOp->getContext(),  newAttr) );

            // auto y_loop_block = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound().front(), block_size_y->getResult(0), c1->getResult(0));
            
            // y_loop_block->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimY_block"));
            // // auto x_loop = rewriter.create<mlir::scf::ParallelOp>(forOp->getLoc(), forOp.getLowerBound(), upperBound1->getResult(0), c1->getResult(0));
            // rewriter.setInsertionPointToStart(y_loop_block.getBody());
            // newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::ThreadY, map, map);
            // y_loop_block->setAttr("mapping", mlir::ArrayAttr::get(parOp->getContext(),  newAttr) );

            // auto res = mlir::getAffineDimExpr(0, parOp->getContext()) * mlir::getAffineSymbolExpr(0, parOp->getContext())  + mlir::getAffineSymbolExpr(1, parOp->getContext()) + mlir::getAffineSymbolExpr(2, parOp->getContext());
            // // auto res = mlir::getAffineDimExpr(0, forOp->getContext());
            // // auto res1 = mlir::getAffineDimExpr(1, forOp->getContext());
            // comet_debug() << res;
            // auto affineIndex = mlir::AffineMap::get(1, 3, {res}, parOp->getContext());
            // comet_debug() << affineIndex;
            // std::vector<mlir::Value> range = { y_loop_grid.getBody()->getArgument(0), block_size_y->getResult(0),  y_loop_block.getBody()->getArgument(0), parOp.getLowerBound()[0]};
            // auto newIndexY = rewriter.create<mlir::arith::MinUIOp>(parOp->getLoc(), rewriter.create<mlir::affine::AffineApplyOp>(parOp->getLoc(), affineIndex, range), parOp.getUpperBound().front());
            // newIndexY->setAttr("GuardY", rewriter.getUnitAttr());
            // // auto newIndexY = rewriter.create<mlir::affine::AffineApplyOp>(forOp->getLoc(), affineIndex, range);
            // // auto newIndexY = rewriter.create<mlir::arith::AddIOp>(forOp->getLoc(),rewriter.create<mlir::arith::MulIOp>(forOp->getLoc(), y_loop_grid.getBody()->getArgument(0), block_size_y), y_loop_block.getBody()->getArgument(0));
            // rewriter.setInsertionPoint(newIndexY);

            // rewriter.replaceAllUsesWith(parOp.getBody()->getArgument(0), newIndexY);
            // rewriter.eraseOp(parOp.getBody()->getTerminator());
            // rewriter.inlineBlockBefore(parOp.getBody(), y_loop_block.getBody()->getTerminator(), newIndexY->getResult(0));
            // rewriter.eraseOp(parOp);
            return mlir::success();
        }
        else if ((mlir::isa<mlir::scf::ForOp>(parOp->getParentOp()) && (parOp->getParentOp()->getAttrOfType<mlir::StringAttr>("parallelDim").getValue().compare("dimY_block") == 0 || mlir::cast<mlir::scf::ParallelOp>(parOp->getParentOp())->getAttrOfType<mlir::StringAttr>("parallelDim").getValue().compare("dimY_grid") == 0) ))  // X level loop
        { 
            // std::vector<int64_t> tileSizes = {blockX};
            // std::pair<scf::ParallelOp, scf::ForOp> tiledLoop = tileParallelLoop(rewriter, parOp, tileSizes);
            // tiledLoop.first->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimX_grid"));
            // tiledLoop.second->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimX_block"));
            
            // newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockX, map, map);
            // tiledLoop.first->setAttr("mapping",  mlir::ArrayAttr::get(tiledLoop.first->getContext(),  newAttr));
            
            // newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::ThreadX, map, map);
            // tiledLoop.second->setAttr("mapping",  mlir::ArrayAttr::get(tiledLoop.first->getContext(),  newAttr));
            // tiledLoop.first.dump();

            // std::vector<int64_t> tileSizes = {blockX};
            // std::pair<mlir::scf::ParallelOp, mlir::scf::ParallelOp> parOps =  mlir::scf::tileParallelLoop(parOp, tileSizes, false);
            // parOps.first->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimX_grid"));
            // parOps.second->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimX_block"));

            // auto block_size_x = rewriter.create<mlir::arith::ConstantIndexOp>(parOp->getLoc(), blockX );
            // auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(parOp->getLoc(), 1);
            // // auto upperBound0 = rewriter.create<mlir::arith::CeilDivUIOp>(parOp->getLoc(), parOp.getUpperBound(), block_size_y);
            // // auto upperBound1 = rewriter.create<mlir::arith::CeilDivUIOp>(parOp->getLoc(), parOp.getUpperBound().front(), block_size_x);
            // auto canonicalized_upper_bound = rewriter.create<mlir::arith::SubIOp>(parOp->getLoc(), parOp.getUpperBound()[0], parOp.getLowerBound()[0]);

            // auto canonicalized_upper_bound_for_ceil = rewriter.create<mlir::arith::AddIOp>(parOp->getLoc(), canonicalized_upper_bound, rewriter.create<mlir::arith::SubIOp>(parOp->getLoc(), block_size_x, c1));
            // auto outer_upper_bound = rewriter.create<mlir::arith::DivSIOp>(parOp->getLoc(), canonicalized_upper_bound_for_ceil, block_size_x);


            // // auto y_loop_grid = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound(), upperBound0->getResult(0), c1->getResult(0));
            // auto x_loop_grid = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound()[0], outer_upper_bound.getResult(), c1->getResult(0));
            // x_loop_grid->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimX_grid"));
            // newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::BlockX, map, map);
            // x_loop_grid->setAttr("mapping", mlir::ArrayAttr::get(parOp->getContext(),  newAttr) );

            // rewriter.setInsertionPointToStart(x_loop_grid.getBody());
            // // auto y_loop_block = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound(), block_size_y->getResult(0), c1->getResult(0));
            // auto x_loop_block = rewriter.create<mlir::scf::ParallelOp>(parOp->getLoc(), parOp.getLowerBound()[0], block_size_x->getResult(0), c1->getResult(0));
            // x_loop_block->setAttr("parallelDim", rewriter.getAttr<mlir::StringAttr>("dimX_block"));
            // newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(rewriter.getContext(), ::mlir::gpu::Processor::ThreadX, map, map);
            // x_loop_block->setAttr("mapping", mlir::ArrayAttr::get(parOp->getContext(),  newAttr) );
            // rewriter.setInsertionPointToStart(x_loop_block.getBody());
            
            // auto res = mlir::getAffineDimExpr(0, parOp->getContext()) * mlir::getAffineSymbolExpr(0, parOp->getContext())  + mlir::getAffineSymbolExpr(1, parOp->getContext()) + mlir::getAffineSymbolExpr(2, parOp->getContext());
            // auto affineIndex = mlir::AffineMap::get(1, 3, {res}, parOp->getContext());
            // std::vector<mlir::Value> range = { x_loop_grid.getBody()->getArgument(0), block_size_x->getResult(0),  x_loop_block.getBody()->getArgument(0), parOp.getLowerBound()[0]};
            // // auto newIndexX = rewriter.create<mlir::affine::AffineApplyOp>(parOp->getLoc(), affineIndex, range);
            // auto newIndexX = rewriter.create<mlir::arith::MinUIOp>(parOp->getLoc(), rewriter.create<mlir::affine::AffineApplyOp>(parOp->getLoc(), affineIndex, range), parOp.getUpperBound()[0]);
            // newIndexX->setAttr("GuardX", rewriter.getUnitAttr());

            
            // // auto newIndexX = rewriter.create<mlir::arith::AddIOp>(parOp->getLoc(),rewriter.create<mlir::arith::MulIOp>(parOp->getLoc(), x_loop_grid.getBody()->getArgument(0), block_size_x), x_loop_block.getBody()->getArgument(0));
            // rewriter.setInsertionPoint(newIndexX);
            // rewriter.replaceAllUsesWith(parOp.getBody()->getArgument(0), newIndexX);

            // rewriter.eraseOp(parOp.getBody()->getTerminator());
            // rewriter.inlineBlockBefore(parOp.getBody(), x_loop_block.getBody()->getTerminator(), newIndexX->getResult(0));
            // rewriter.eraseOp(parOp);

            return mlir::success();
        }
        else 
        {
            std::cout << "Failed to match parallel Op\n";
            return mlir::failure();
        }
    }
};

struct DetectReduction
    : public mlir::OpConversionPattern<mlir::scf::ForOp> {
    DetectReduction(mlir::MLIRContext* ctx, int blockX, int blockY, int blockR) : mlir::OpConversionPattern<mlir::scf::ForOp>(ctx), blockX(blockX), blockY(blockY), blockR(blockR) {}
    private:
    [[maybe_unused]] int blockX, blockY, blockR;
    mlir::LogicalResult
    matchAndRewrite(mlir::scf::ForOp forOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        bool no_inner_loops = forOp.getBody()->getOps<mlir::scf::ForOp>().empty();
        if(forOp->hasAttr("parallelDim") || forOp->hasAttr("reduceDim") )
        {
            return mlir::failure();
        }


        bool reduction = is_reduction(forOp);
        if (mlir::scf::ParallelOp parent = llvm::dyn_cast_or_null<mlir::scf::ParallelOp>(forOp->getParentOp()); parent && parent->hasAttr("parallelDim") && no_inner_loops && reduction)
        {
            // assert(parent && parent->getAttrOfType<mlir::StringAttr>("parallelDim").getValue().equals("dimX_block") && !forOp->hasAttr("reduceDim"));

            auto block_size_r = rewriter.create<mlir::arith::ConstantIndexOp>(forOp->getLoc(), blockR );
            auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(forOp->getLoc(), 1);
            // auto upperBound0 = rewriter.create<mlir::arith::CeilDivUIOp>(forOp->getLoc(), forOp.getUpperBound(), block_size_y);
            auto outer_lower_bound = rewriter.create<mlir::arith::ConstantIndexOp>(forOp->getLoc(), 0);
            auto outer_upper_bound = CeilDivUIOp(rewriter,forOp->getLoc(), rewriter.create<mlir::arith::SubIOp>(forOp->getLoc(), forOp.getUpperBound(), forOp.getLowerBound()), block_size_r);

            // auto upperBound1 = rewriter.create<mlir::arith::CeilDivUIOp>(forOp->getLoc(), forOp.getUpperBound(), block_size_r);
            // auto y_loop_grid = rewriter.create<mlir::scf::ParallelOp>(forOp->getLoc(), forOp.getLowerBound(), upperBound0->getResult(0), c1->getResult(0));
            auto r_loop_grid = rewriter.create<mlir::scf::ForOp>(forOp->getLoc(), outer_lower_bound, outer_upper_bound->getResult(0), c1.getResult());
            r_loop_grid->setAttr("reduceDim", rewriter.getAttr<mlir::StringAttr>("dimR_grid"));

            rewriter.setInsertionPointToStart(r_loop_grid.getBody());
            auto inner_lower_bound = outer_lower_bound;
            // auto y_loop_block = rewriter.create<mlir::scf::ParallelOp>(forOp->getLoc(), forOp.getLowerBound(), block_size_y->getResult(0), c1->getResult(0));
            auto r_loop_block = rewriter.create<mlir::scf::ForOp>(forOp->getLoc(), inner_lower_bound.getResult(), block_size_r->getResult(0), c1.getResult());
            r_loop_block->setAttr("reduceDim", rewriter.getAttr<mlir::StringAttr>("dimR_block"));
            
            rewriter.setInsertionPointToStart(r_loop_block.getBody());
            
            auto res = mlir::getAffineDimExpr(0, forOp->getContext()) * mlir::getAffineSymbolExpr(0, forOp->getContext())  + mlir::getAffineSymbolExpr(1, forOp->getContext()) + mlir::getAffineSymbolExpr(2, forOp->getContext());
            auto affineIndex = mlir::AffineMap::get(1, 3, {res}, forOp->getContext());
            std::vector<mlir::Value> range = { r_loop_grid.getBody()->getArgument(0), block_size_r->getResult(0),  r_loop_block.getBody()->getArgument(0), forOp.getLowerBound()};
            // auto newIndexX = rewriter.create<mlir::affine::AffineApplyOp>(forOp->getLoc(), affineIndex, range);
            auto newIndexX = rewriter.create<mlir::arith::MinUIOp>(forOp->getLoc(), rewriter.create<mlir::affine::AffineApplyOp>(forOp->getLoc(), affineIndex, range), forOp.getUpperBound());
            newIndexX->setAttr("GuardR", rewriter.getUnitAttr());
            // auto newIndexX = rewriter.create<mlir::arith::AddIOp>(forOp->getLoc(),rewriter.create<mlir::arith::MulIOp>(forOp->getLoc(), x_loop_grid.getBody()->getArgument(0), block_size_x), x_loop_block.getBody()->getArgument(0));
            rewriter.setInsertionPoint(newIndexX);

            rewriter.replaceAllUsesWith(forOp.getBody()->getArgument(0), newIndexX);

            rewriter.eraseOp(forOp.getBody()->getTerminator());
            rewriter.inlineBlockBefore(forOp.getBody(), r_loop_block.getBody()->getTerminator(), newIndexX->getResult(0));
            rewriter.eraseOp(forOp);
            return mlir::success(); 
        }
        else 
        {
            return mlir::failure();
        }
    }
};

class ConvertParallelLoopsToGpu: public CometParallelLoopsToGpuBase<ConvertParallelLoopsToGpu> {
public:
    ConvertParallelLoopsToGpu() = default;
    ConvertParallelLoopsToGpu(int blockX, int blockY, int blockR, mlir::tensorAlgebra::TargetDevice target_device) {
        this->blockX = blockX;
        this->blockY = blockY;
        this->blockR = blockR;
        this->target_device = target_device;
    }

    void runOnOperation() override {
        mlir::MLIRContext *context = &getContext();
        mlir::func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration())
        {
            return;
        }

        // funcOp->walk([](mlir::scf::ForOp forOp) {
        //     mlir::OpBuilder builder(forOp);
        //     auto old_lower = forOp.getLowerBound();
        //     auto old_upper = forOp.getUpperBound();
        //     auto new_upper_bound =  builder.create<mlir::arith::SubIOp>(forOp->getLoc(), old_upper, old_lower);
        //     auto new_lower_bound = builder.create<mlir::arith::ConstantIndexOp>(forOp->getLoc(), 0);
        //     auto res = mlir::getAffineDimExpr(0, forOp->getContext()) + mlir::getAffineSymbolExpr(0, forOp->getContext());
        //     forOp.setLowerBound(new_lower_bound);
        //     forOp.setUpperBound(new_upper_bound);
        //     builder.setInsertionPointToStart(forOp.getBody());
        //     auto affineIndex = mlir::AffineMap::get(1, 1, {res}, forOp->getContext());
        //     std::vector<mlir::Value> range = { forOp.getBody()->getArgument(0), old_lower};
        //     auto newIndex = builder.create<mlir::affine::AffineApplyOp>(forOp->getLoc(), affineIndex, range);
        //     forOp.getBody()->getArgument(0).replaceAllUsesExcept(newIndex.getResult(), newIndex);
        // });

        mlir::RewritePatternSet patterns(context);
        patterns.insert<ParallelOpToGpu>(context, blockX, blockY, blockR, this->target_device);
        
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

        // funcOp->walk([](mlir::scf::ParallelOp par_for) {
        //     mlir::OpBuilder builder(par_for);
        //     auto map = builder.getDimIdentityMap();
        //     mlir::gpu::ParallelLoopDimMappingAttr newAttr;
        //     if(par_for->hasAttr("parallelDim") && !par_for->hasAttr("mapping"))
        //     {
        //         if(par_for->getAttrOfType<mlir::StringAttr>("parallelDim").str() == "dimY_grid")
        //         {
        //             newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(builder.getContext(), ::mlir::gpu::Processor::BlockY, map, map);
        //             par_for->setAttr("mapping", mlir::ArrayAttr::get(par_for->getContext(),  newAttr) );
        //         }
        //         else if(par_for->getAttrOfType<mlir::StringAttr>("parallelDim").str() == "dimX_grid")
        //         {
        //             newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(builder.getContext(), ::mlir::gpu::Processor::BlockX, map, map);
        //             par_for->setAttr("mapping", mlir::ArrayAttr::get(par_for->getContext(),  newAttr) );
        //         }
        //         // else if(par_for->getAttrOfType<mlir::StringAttr>("parallelDim").str() == "dimX_block")
        //         // {
        //         //     newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(builder.getContext(), ::mlir::gpu::Processor::ThreadX, map, map);
        //         // }
        //         // else if(par_for->getAttrOfType<mlir::StringAttr>("parallelDim").str() == "dimY_block")
        //         // {
        //         //     newAttr = mlir::gpu::ParallelLoopDimMappingAttr::get(builder.getContext(), ::mlir::gpu::Processor::ThreadY, map, map);
        //         // }
        //         // assert(newAttr);
        //     }
        // });

        // mlir::RewritePatternSet patterns2(context);
        // mlir::ConversionTarget target2(*context);

        // target2.addLegalDialect<mlir::memref::MemRefDialect, mlir::arith::ArithDialect,  mlir::affine::AffineDialect, mlir::scf::SCFDialect>();

        // target2.addLegalOp<mlir::scf::YieldOp>();
        // patterns2.insert<DetectReduction>(context, blockX, blockY, blockR);
        // target2.addDynamicallyLegalOp<mlir::scf::ForOp>([](mlir::scf::ForOp op) -> bool {
        //     mlir::scf::ParallelOp parent = llvm::dyn_cast_or_null<mlir::scf::ParallelOp>(op->getParentOp());
        //     if(parent && !op->hasAttr("reduceDim"))
        //     {
        //         return false;
        //     }
        //     else
        //     {
        //         return true;
        //     }
        // });

        // if (mlir::failed(mlir::applyPartialConversion(funcOp, target2, std::move(patterns2))))
        // {
        //     signalPassFailure();
        // }
    }
}; 
}


std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::comet::createConvertParallelLoopsToGpuPass() {
    return std::make_unique<ConvertParallelLoopsToGpu>();
}


std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::comet::createConvertParallelLoopsToGpuPass(int blockX, int blockY, int blockR, mlir::tensorAlgebra::TargetDevice target_device) {
    return std::make_unique<ConvertParallelLoopsToGpu>(blockX, blockY, blockR, target_device);
}

