#include "comet/Conversion/BlockedGpuToTriton/BlockedGpuToTriton.h"
#include "comet/Conversion/BlockedGpuToTriton/BlockedGpuToTritonConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <utility>

#define GEN_PASS_CLASSES
#include "comet/Conversion/BlockedGpuToTriton/Passes.h.inc"
using namespace mlir;


class ConvertGpuFuncToTritonFunc : public OpConversionPattern<mlir::gpu::GPUFuncOp>{
    public:
    using mlir::OpConversionPattern<mlir::gpu::GPUFuncOp>::OpConversionPattern;
    ConvertGpuFuncToTritonFunc(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::gpu::GPUFuncOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::gpu::GPUFuncOp gpuFunc, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        
        auto converter = getTypeConverter();
        llvm::SmallVector<Value, 4> materializedValues;
        gpu::GPUModuleOp gpuModuleOp = gpuFunc->getParentOfType<gpu::GPUModuleOp>();
        rewriter.setInsertionPointToStart(gpuModuleOp.getBody());
        
        llvm::SmallVector<Type, 4> tritonFuncTypes;
        llvm::SmallVector<size_t, 4> sizes;
        for(auto argType: gpuFunc.getArgumentTypes())
        {
            llvm::SmallVector<Type, 4> currentTritonFuncTypes;
            if(failed(converter->convertTypes(argType, currentTritonFuncTypes)))
            {
                return failure();
            }
            tritonFuncTypes.insert(tritonFuncTypes.end(), currentTritonFuncTypes.begin(), currentTritonFuncTypes.end());
            sizes.push_back(currentTritonFuncTypes.size());
        }

        auto tritonFuncType = rewriter.getFunctionType(TypeRange(tritonFuncTypes), gpuFunc.getFunctionType().getResults());
        auto tritonFunc = rewriter.create<triton::FuncOp>(gpuFunc->getLoc(), "tt_"+ gpuModuleOp.getName().str() +gpuFunc.getName().str(), tritonFuncType);
        auto ttFuncBlock = tritonFunc.addEntryBlock();
        rewriter.setInsertionPointToStart(ttFuncBlock);
        size_t prev = 0;
        for(size_t i = 0; i <  gpuFunc.getBody().getArguments().size(); i++)
        {
            if(!converter->isLegal( gpuFunc.getBody().getArguments()[i].getType()))
            {
                auto conveted = converter->materializeArgumentConversion(rewriter, gpuFunc.getBody().getArguments()[i].getLoc(),  gpuFunc.getBody().getArguments()[i].getType(), ttFuncBlock->getArguments().slice(prev, sizes[i]));
                materializedValues.push_back(conveted);
            }
            else
            {
                materializedValues.push_back(ttFuncBlock->getArguments()[prev]); 
            }
            rewriter.replaceAllUsesWith(gpuFunc.getBody().getArguments()[i], materializedValues.back());
            prev += sizes[i];
        }

        rewriter.eraseOp(gpuFunc.getBody().front().getTerminator());
        rewriter.mergeBlocks(&gpuFunc.getBody().front(), ttFuncBlock, materializedValues);
        rewriter.setInsertionPointToEnd(ttFuncBlock);
        rewriter.create<triton::ReturnOp>(tritonFunc->getLoc());
        rewriter.setInsertionPointToEnd(gpuModuleOp.getBody());
        func::FuncOp newFunc = rewriter.replaceOpWithNewOp<func::FuncOp>(gpuFunc, gpuFunc.getName(), gpuFunc.getFunctionType());
        auto entryBlock = newFunc.addEntryBlock();
        rewriter.setInsertionPointToEnd(entryBlock);
        rewriter.create<func::ReturnOp>(gpuFunc.getLoc());
        newFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                           rewriter.getUnitAttr());
        
        return success();
    }

};

class ConvertForOp : public OpConversionPattern<mlir::scf::ForOp>{
    public:
    using mlir::OpConversionPattern<mlir::scf::ForOp>::OpConversionPattern;
    ConvertForOp(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::scf::ForOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::scf::ForOp forOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        
        llvm::SmallVector<Value, 4> materializedValues;
        
        llvm::SmallVector<Value, 4> newForValues;
        llvm::SmallVector<size_t, 4> sizes;
        for(auto arg: forOp->getOperands())
        {

            if(arg.getType().isIndex())
            {
                auto cast_index = rewriter.create<arith::IndexCastOp>(forOp->getLoc(), IntegerType::get(getContext(), 32),  arg);
                newForValues.push_back(cast_index);
            }
            else
            {
                newForValues.push_back(arg);
            }
        }

        auto newForOp = rewriter.create<scf::ForOp>(forOp->getLoc(), newForValues[0], newForValues[1], newForValues[2], ValueRange(newForValues).drop_front(3));
        rewriter.setInsertionPointToStart(newForOp.getBody());
        llvm::SmallVector<Value, 4> newForArgs;

        for(auto arg: newForOp.getBody()->getArguments())
        {
            if(arg.getType().isInteger())
            {
                auto cast_index = rewriter.create<arith::IndexCastOp>(forOp->getLoc(), rewriter.getIndexType(),  arg);
                newForArgs.push_back(cast_index);
            }
            else
            {
                newForArgs.push_back(arg);
            }
        }

        rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(), newForArgs);
        rewriter.replaceAllOpUsesWith(forOp, newForOp);
        rewriter.eraseOp(forOp);
        
        return success();
    }

};


class ConvertBlockId : public OpConversionPattern<mlir::gpu::BlockIdOp>{
    public:
    using mlir::OpConversionPattern<mlir::gpu::BlockIdOp>::OpConversionPattern;
    ConvertBlockId(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::gpu::BlockIdOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::gpu::BlockIdOp blockIdOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        
        triton::ProgramIDDim axis;
        switch (blockIdOp.getDimension()) 
        {
            case mlir::gpu::Dimension::x: 
            {
                axis = triton::ProgramIDDim::X;
                break;
            }
            case mlir::gpu::Dimension::y:
            {
                axis = triton::ProgramIDDim::Y;
                break;
            }
            case mlir::gpu::Dimension::z:
            {
                axis = triton::ProgramIDDim::Z;
                break;
            }
        }

        auto getProgram = rewriter.create<triton::GetProgramIdOp>(blockIdOp->getLoc(), axis);
        auto cast = rewriter.create<arith::IndexCastOp>(blockIdOp->getLoc(), rewriter.getIndexType(), getProgram);
        rewriter.replaceOp(blockIdOp, cast);

        return success();
    }
};

class ConvertBlockDim : public OpConversionPattern<mlir::gpu::BlockDimOp>{
    public:
    using mlir::OpConversionPattern<mlir::gpu::BlockDimOp>::OpConversionPattern;
    ConvertBlockDim(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::gpu::BlockDimOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::gpu::BlockDimOp BlockDimOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        
        triton::ProgramIDDim axis;
        switch (BlockDimOp.getDimension()) 
        {
            case mlir::gpu::Dimension::x: 
            {
                axis = triton::ProgramIDDim::X;
                break;
            }
            case mlir::gpu::Dimension::y:
            {
                axis = triton::ProgramIDDim::Y;
                break;
            }
            case mlir::gpu::Dimension::z:
            {
                axis = triton::ProgramIDDim::Z;
                break;
            }
        }

        auto getProgram = rewriter.create<triton::GetNumProgramsOp>(BlockDimOp->getLoc(), axis);
        auto cast = rewriter.create<arith::IndexCastOp>(BlockDimOp->getLoc(), rewriter.getIndexType(), getProgram);
        rewriter.replaceOp(BlockDimOp, cast);

        return success();
    }
};

class ConvertTensorSplatOp : public OpConversionPattern<mlir::tensor::SplatOp>{
    public:
    using mlir::OpConversionPattern<mlir::tensor::SplatOp>::OpConversionPattern;
    ConvertTensorSplatOp(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::tensor::SplatOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::tensor::SplatOp SplatOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {

        MLIRContext* ctx = rewriter.getContext();
        RankedTensorType outTensorType; 
        Value input;
        if(isa<mlir::IndexType>(adaptor.getInput().getType()))
        {
            IntegerType i32 = IntegerType::get(ctx, 32);
            input =  rewriter.create<arith::IndexCastOp>(SplatOp->getLoc(), i32, adaptor.getInput());
            outTensorType = RankedTensorType::get(SplatOp.getResult().getType().getShape(), i32);
        }
        else 
        {
            input = adaptor.getInput();
            outTensorType = SplatOp.getResult().getType();
        }

        Value ttSplatOp = rewriter.create<triton::SplatOp>(SplatOp->getLoc(),  outTensorType, input);
        rewriter.replaceOpWithNewOp<arith::IndexCastOp>(SplatOp, SplatOp.getType(), ttSplatOp);

        return success();
    }
};

template<typename T>
std::pair<Value, Value> generate_triton_blocked_bounds_and_offsets(ConversionPatternRewriter& rewriter, T sliceOp, RankedTensorType sliceT, SmallVector<ReassociationIndices, 2> collapsed_indices={})
{
    llvm::SmallVector<Value, 2> blockedOffsets;
    llvm::SmallVector<Value, 2> blockedBounds;
    SmallVector<Value, 4> tritonOffsets, tritonBounds;
    llvm::SmallMapVector<size_t, Value, 4> dimToValue;
    size_t dimIndex = 0;
    for(size_t i = 0; i < sliceT.getRank(); i++)
    {
        if(sliceOp.getStaticSizes()[i] == ShapedType::kDynamic)
        {
            dimToValue[i] = sliceOp.getSizes()[dimIndex];
            dimIndex ++;
        }
    }

    for(size_t i = 0; i < sliceOp.getOffsets().size(); i++)
    {
        auto offset = sliceOp.getOffsets()[i];
        if(auto castOp = mlir::dyn_cast<UnrealizedConversionCastOp>(offset.getDefiningOp()))
        {
            Value cast;
            if(auto shaped = mlir::dyn_cast<ShapedType>(castOp.getInputs().front().getType()))
            {
                cast = rewriter.create<arith::IndexCastOp>(sliceOp->getLoc(), RankedTensorType::get(shaped.getShape(), rewriter.getIntegerType(32)), castOp.getInputs().front());
            }
            else 
            {
                cast = rewriter.create<arith::IndexCastOp>(sliceOp->getLoc(), rewriter.getIntegerType(32), castOp.getInputs().front());
            }
            blockedOffsets.push_back(cast);
        }
        else if(mlir::isa<IndexType>(offset.getType()))
        {
            auto cast = rewriter.create<arith::IndexCastOp>(sliceOp->getLoc(), rewriter.getIntegerType(32), offset);
            blockedOffsets.push_back(cast);
        }

        auto blockSize = sliceT.getDimSize(i);
        if(sliceOp.getStaticSizes()[i] == ShapedType::kDynamic)
        {
            
            Value bound = dimToValue[i];
            if(auto min = mlir::dyn_cast_if_present<arith::MinSIOp>(bound.getDefiningOp()))
            {
                bound = min.getRhs();
            }
            Value cast = rewriter.create<arith::IndexCastOp>(bound.getLoc(), rewriter.getIntegerType(32), bound);
            
            auto blockedBound =  rewriter.create<triton::SplatOp>(bound.getLoc(), RankedTensorType::get({blockSize}, rewriter.getIntegerType(32)),  cast);
            auto blockedBlockSize =  rewriter.create<triton::MakeRangeOp>(bound.getLoc(), RankedTensorType::get({blockSize}, rewriter.getIntegerType(32)), 0, blockSize);
            auto cmp = rewriter.create<arith::CmpIOp>(bound.getLoc(), arith::CmpIPredicate::slt, blockedBlockSize, blockedBound);
            blockedBounds.push_back(cmp);
            tritonBounds.push_back(cmp);
        }
    }


    SmallVector<int64_t, 4> resultShape;

    for(size_t i = 0; i < blockedOffsets.size(); i++)
    {
        auto blockedOffset = blockedOffsets[i];
        Value strideVal = rewriter.create<arith::IndexCastOp>(sliceOp->getLoc(), rewriter.getIntegerType(32), sliceOp.getStrides()[i]);
        Value stridesBlocked = strideVal;
        if(mlir::isa<ShapedType>(blockedOffset.getType()))
        {
            stridesBlocked = rewriter.create<triton::SplatOp>(sliceOp->getLoc(), blockedOffset.getType(), strideVal);
        }
        
        Value offsetBlocked = rewriter.create<arith::MulIOp>(sliceOp->getLoc(), blockedOffset, stridesBlocked);
        tritonOffsets.push_back(offsetBlocked);
        if(auto shaped = mlir::dyn_cast<ShapedType>(blockedOffset.getType()))
        {
            resultShape.push_back(shaped.getDimSize(0));
        }
        else
        {
            resultShape.push_back(1);
        }
    }

    SmallVector<Value, 4> collapsed_offsets;
    SmallVector<Value, 4> collapsed_bounds;
    SmallVector<int64_t, 4> collapsed_shape;
    if(collapsed_indices.empty())
    {
        collapsed_offsets = tritonOffsets;
        for(auto d: resultShape)
        {
            if (d != 1)
            {
                collapsed_shape.push_back(d);        
            }
        }
        // collapsed_shape = resultShape;
        
    }
    collapsed_bounds = tritonBounds;
    for(auto collapsed: collapsed_indices)
    {
        size_t shaped_i = 0;
        for(size_t i = 0; i < collapsed.size(); i++)
        {
            if(mlir::isa<RankedTensorType>(tritonOffsets[collapsed[i]].getType()))
            {
                shaped_i = i;
            }
        }
        collapsed_offsets.push_back(tritonOffsets[shaped_i]);
        for(size_t i = 0; i < collapsed.size(); i++)
        {
            if(i == shaped_i)
                continue; // skip the shaped index
            
            Value offset = tritonOffsets[collapsed[i]];
            if(tritonOffsets[collapsed[i]].getType() != collapsed_offsets.back().getType())
            {
                offset = rewriter.create<triton::SplatOp>(sliceOp->getLoc(), collapsed_offsets.back().getType(), tritonOffsets[collapsed[i]]);
            }
            auto res = rewriter.create<arith::AddIOp>(sliceOp->getLoc(), collapsed_offsets.back(), offset);
            collapsed_offsets.back() = res;
        }
        if(resultShape[shaped_i] != 1)
        {
            collapsed_shape.push_back(resultShape[shaped_i]);
        }
    }

    for(size_t i = 0; i < collapsed_offsets.size(); i++)
    {
        for(size_t j = 0; j <collapsed_offsets.size(); j++  )
        {
            if(i != j)
            {
                if(!mlir::isa<ShapedType>(collapsed_offsets[i].getType()))
                {
                    collapsed_offsets[i] = rewriter.create<triton::SplatOp>(sliceOp->getLoc(), RankedTensorType::get(1, rewriter.getIntegerType(32)), collapsed_offsets[i]);    
                }
                collapsed_offsets[i] = rewriter.create<triton::ExpandDimsOp>(sliceOp->getLoc(), collapsed_offsets[i], j);
                
            }
        }
        
        if(!collapsed_shape.empty())
        {
            collapsed_offsets[i] = rewriter.create<triton::BroadcastOp>(sliceOp->getLoc(), RankedTensorType::get(collapsed_shape, rewriter.getIntegerType(32)), collapsed_offsets[i]);
        }
    }
    
    auto resultType = RankedTensorType::get(collapsed_shape, rewriter.getIntegerType(1));
    size_t k = 0;
    for(size_t i = 0; i < collapsed_bounds.size(); i++)
    {
        for(size_t j = 0; j <collapsed_bounds.size(); j++  )
        {
            if(i != j)
            {
                collapsed_bounds[i] = rewriter.create<triton::ExpandDimsOp>(sliceOp->getLoc(), collapsed_bounds[i], j);
                
            }
        }
        if(mlir::cast<ShapedType>(collapsed_bounds[i].getType()).getRank() != resultType.getRank())
        {
            for(size_t j = 0; j < resultType.getRank(); j++)
            {
                if(resultType.getDimSize(j) != mlir::cast<ShapedType>(collapsed_bounds[i].getType()).getDimSize(k))
                {
                    collapsed_bounds[i] = rewriter.create<triton::ExpandDimsOp>(sliceOp->getLoc(), collapsed_bounds[i], k++);
                }
                else
                {
                    k++;
                }
            }
        }
        collapsed_bounds[i] = rewriter.create<triton::BroadcastOp>(sliceOp->getLoc(), resultType, collapsed_bounds[i]);
    }

    Value combinedOffsetBlocked, combinedBoundBlocked = nullptr;
    combinedOffsetBlocked = collapsed_offsets.front();
    if(!collapsed_bounds.empty())
    {
        combinedBoundBlocked = collapsed_bounds.front();
    }
    for(size_t i = 1; i <  collapsed_offsets.size(); i++)
    {
        combinedOffsetBlocked = rewriter.create<arith::AddIOp>(sliceOp->getLoc(), combinedOffsetBlocked, collapsed_offsets[i]);
    }

    for(size_t i = 1; i <  collapsed_bounds.size(); i++)
    {
        combinedBoundBlocked = rewriter.create<arith::AndIOp>(sliceOp->getLoc(), combinedBoundBlocked, collapsed_bounds[i]);
    }

    return std::make_pair(combinedOffsetBlocked, combinedBoundBlocked);    
}

// class ConvertTensorInsert: public OpConversionPattern<mlir::tensor::InsertOp>{
//     public:
//     using mlir::OpConversionPattern<mlir::tensor::InsertSliceOp>::OpConversionPattern;
//     ConvertInsertSlice(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::tensor::InsertSliceOp>(ctx) {}
//     mlir::LogicalResult
//     matchAndRewrite(mlir::tensor::InsertSliceOp insertSliceOp, OpAdaptor adaptor,
//                     mlir::ConversionPatternRewriter &rewriter) const override {
//     }

// };



class ConvertInsertSlice : public OpConversionPattern<mlir::tensor::InsertSliceOp>{
    public:
    using mlir::OpConversionPattern<mlir::tensor::InsertSliceOp>::OpConversionPattern;
    ConvertInsertSlice(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::tensor::InsertSliceOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::tensor::InsertSliceOp insertSliceOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {

        Value dest = adaptor.getDest();
        Value ttSource = adaptor.getSource();
        Value ttDest;
        bufferization::ToTensorOp toTensor = mlir::dyn_cast<bufferization::ToTensorOp>(dest.getDefiningOp());
        UnrealizedConversionCastOp castOp;
        if(toTensor)
        {
            if((castOp = mlir::dyn_cast<UnrealizedConversionCastOp>(toTensor.getMemref().getDefiningOp())))
            {
                if(triton::PointerType ttPtr =  mlir::dyn_cast<triton::PointerType>(castOp->getOperand(0).getType()))
                {
                    ttDest = castOp->getOperand(0);
                }
            }
        }
        else if((castOp = mlir::dyn_cast<UnrealizedConversionCastOp>(toTensor.getMemref().getDefiningOp())))
        {
            if(triton::PointerType ttPtr =  mlir::dyn_cast<triton::PointerType>(castOp->getOperand(0).getType()))
            {
                ttDest = castOp->getOperand(0);
            }
        }

        if(!ttDest)
        {
            return failure();
        }

        RankedTensorType sliceT;
        Operation* toReplace = insertSliceOp;
        SmallVector<ReassociationIndices, 2> expanded_indices;

        if(auto tensorCastOp = mlir::dyn_cast_if_present<mlir::tensor::CastOp>(insertSliceOp.getSource().getDefiningOp()))
        {
            sliceT = mlir::cast<RankedTensorType>(tensorCastOp.getSource().getType());
            if(auto expand_op =  mlir::dyn_cast_if_present<mlir::tensor::ExpandShapeOp>(tensorCastOp.getSource().getDefiningOp()))
            {
                expanded_indices = expand_op.getReassociationIndices();
                ttSource = expand_op.getSrc();
            }
            else if(auto expand_op =  mlir::dyn_cast_if_present<mlir::tensor::ExpandShapeOp>(insertSliceOp.getSource().getDefiningOp()))
            {
                expanded_indices = expand_op.getReassociationIndices();
                ttSource = expand_op.getSrc();
            }
        }
        else
        {
            sliceT = mlir::cast<RankedTensorType>(insertSliceOp.getSource().getType());
            if(auto expand_op =  mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(insertSliceOp.getSource().getDefiningOp()))
            {
                expanded_indices = expand_op.getReassociationIndices();
                ttSource = expand_op.getSrc();
            }
        }

        auto [combinedOffsetBlocked, combinedBoundBlocked] = generate_triton_blocked_bounds_and_offsets(rewriter, insertSliceOp, sliceT, expanded_indices);
        std::vector<int64_t> resultShape;
        if(auto offset_shaped = dyn_cast<ShapedType>(combinedOffsetBlocked.getType()))
        {
            resultShape = offset_shaped.getShape();
        }
        
        Value blockedPtr = ttDest;
        if(!resultShape.empty()) 
        {
            blockedPtr = rewriter.create<triton::SplatOp>(toReplace->getLoc(), RankedTensorType::get(resultShape, ttDest.getType()), ttDest);
        }

        Value ptr;
        if(!resultShape.empty()) 
        {
            ptr = rewriter.create<triton::AddPtrOp>(toReplace->getLoc(),  RankedTensorType::get(resultShape, ttDest.getType()), blockedPtr, combinedOffsetBlocked);
        }
        else 
        {
            ptr = rewriter.create<triton::AddPtrOp>(toReplace->getLoc(),  ttDest.getType(), blockedPtr, combinedOffsetBlocked);
        }

        if(ShapedType sourceShaped = mlir::dyn_cast<ShapedType>(ttSource.getType());  sourceShaped && !sourceShaped.hasStaticShape())
        {
            auto castOp = rewriter.create<tensor::CastOp>(toReplace->getLoc(), RankedTensorType::get(resultShape, insertSliceOp.getDestType().getElementType()), ttSource);
            if(combinedBoundBlocked)
            {
                rewriter.create<triton::StoreOp>( insertSliceOp->getLoc(), ptr, castOp, combinedBoundBlocked, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
            }
            else
            {
                if(auto ranked_source = dyn_cast<RankedTensorType>(castOp.getType()))
                {
                    if(!isa<RankedTensorType>(ptr.getType()))
                    {
                        if(llvm::all_of(ranked_source.getShape(), [](int64_t d){
                            return d == 1;
                        }))
                        {
                            ptr = rewriter.create<triton::SplatOp>(ptr.getLoc(), RankedTensorType::get(ranked_source.getShape(), ptr.getType()), ptr);
                        }
                    }
                }
                rewriter.create<triton::StoreOp>( toReplace->getLoc(), ptr, castOp, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
            }
            auto placeholder = rewriter.create<UnrealizedConversionCastOp>(toReplace->getLoc(), insertSliceOp.getResultType(), ValueRange(ptr));
            rewriter.replaceOp(insertSliceOp, placeholder->getResult(0));
        }
        else 
        {
            if(combinedBoundBlocked)
            {
                rewriter.create<triton::StoreOp>( insertSliceOp->getLoc(), ptr, ttSource, combinedBoundBlocked, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
            }
            else
            {
                if(auto ranked_source = dyn_cast<RankedTensorType>(ttSource.getType()))
                {
                    if(!isa<RankedTensorType>(ptr.getType()))
                    {
                        if(llvm::all_of(ranked_source.getShape(), [](int64_t d){
                            return d == 1;
                        }))
                        {
                            ptr = rewriter.create<triton::SplatOp>(ptr.getLoc(), RankedTensorType::get(ranked_source.getShape(), ptr.getType()), ptr);
                        }
                    }
                }
                rewriter.create<triton::StoreOp>( insertSliceOp->getLoc(), ptr, ttSource, combinedBoundBlocked, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
            }

            auto placeholder = rewriter.create<UnrealizedConversionCastOp>(insertSliceOp->getLoc(), insertSliceOp.getResultType(), ValueRange(ptr));
            rewriter.replaceOp(insertSliceOp, placeholder->getResult(0));
        }
        
        if(toTensor->getUsers().empty())
        {
            rewriter.eraseOp(toTensor);
        }
        if(castOp->getUsers().empty())
        {
            rewriter.eraseOp(castOp);
        }
        return success();
    }
};

class ConvertExtractSlice : public OpConversionPattern<mlir::tensor::ExtractSliceOp>{
    public:
    using mlir::OpConversionPattern<mlir::tensor::ExtractSliceOp>::OpConversionPattern;
    ConvertExtractSlice(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::tensor::ExtractSliceOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::tensor::ExtractSliceOp extractSliceOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        Value source = adaptor.getSource();
        Value ttSource;
        bufferization::ToTensorOp toTensor = mlir::dyn_cast<bufferization::ToTensorOp>(source.getDefiningOp());
        UnrealizedConversionCastOp castOp;
        if(toTensor)
        {
            if((castOp = mlir::dyn_cast_if_present<UnrealizedConversionCastOp>(toTensor.getMemref().getDefiningOp())))
            {
                if(triton::PointerType ttPtr =  mlir::dyn_cast<triton::PointerType>(castOp->getOperand(0).getType()))
                {
                    ttSource = castOp->getOperand(0);
                }
            }
        }
        else if((castOp = mlir::dyn_cast_if_present<UnrealizedConversionCastOp>(toTensor.getMemref().getDefiningOp())))
        {
            if(triton::PointerType ttPtr =  mlir::dyn_cast<triton::PointerType>(castOp->getOperand(0).getType()))
            {
                ttSource = castOp->getOperand(0);
            }
        }

        if(!ttSource)
        {
            return failure();
        }

        RankedTensorType sliceT;
        SmallVector<ReassociationIndices, 2> collapsed_indices;
        Operation* toReplace = extractSliceOp;
        auto tensorCastOp = mlir::dyn_cast<mlir::tensor::CastOp>(*extractSliceOp->getUsers().begin());
        Value extractOp = nullptr;
        Value collapseOp = nullptr;
        if(tensorCastOp)
        {
            sliceT = mlir::cast<RankedTensorType>(tensorCastOp.getResult().getType());
            if(auto collapse_op =  mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(*tensorCastOp->getUsers().begin()))
            {
                collapsed_indices = collapse_op.getReassociationIndices();
                toReplace = collapse_op;
                collapseOp = collapse_op;
                if(auto extract_op = mlir::dyn_cast<mlir::tensor::ExtractOp>(*collapse_op->getUsers().begin()))
                {
                    extractOp = extract_op;
                    toReplace = extract_op;
                }
            }
            else if(auto extract_op = mlir::dyn_cast<mlir::tensor::ExtractOp>(*tensorCastOp->getUsers().begin()))
            {
                extractOp = extract_op;
                toReplace = extract_op;
            }
        }
        else
        {
            sliceT = mlir::cast<RankedTensorType>(extractSliceOp.getResultType());
            if(auto collapse_op =  mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(*extractSliceOp->getUsers().begin()))
            {
                collapsed_indices = collapse_op.getReassociationIndices();
                toReplace = collapse_op;
                collapseOp = collapse_op;

                if(auto extract_op = mlir::dyn_cast<mlir::tensor::ExtractOp>(*collapse_op->getUsers().begin()))
                {
                    extractOp = extract_op;
                    toReplace = extract_op;
                }
            }
            else if(auto extract_op = mlir::dyn_cast<mlir::tensor::ExtractOp>(*extractSliceOp->getUsers().begin()))
            {
                extractOp = extract_op;
                toReplace = extract_op;
            }
        }


        // RankedTensorType sliceT = mlir::cast<RankedTensorType>(tensorCastOp.getResult().getType());
        auto [combinedOffsetBlocked, combinedBoundBlocked] = generate_triton_blocked_bounds_and_offsets(rewriter, extractSliceOp, sliceT, collapsed_indices);

        std::vector<int64_t> resultShape;
        if(auto ranked_offset_type = mlir::dyn_cast<RankedTensorType>(combinedOffsetBlocked.getType()))
        {
            resultShape = ranked_offset_type.getShape().vec();
        }

        Value blockedPtr = ttSource;
        
        if(!resultShape.empty()) 
        {
            blockedPtr = rewriter.create<triton::SplatOp>(extractSliceOp->getLoc(), RankedTensorType::get(resultShape, ttSource.getType()), ttSource);
        }
        
        Value ptr;
        if(!resultShape.empty())  
        {
            ptr = rewriter.create<triton::AddPtrOp>(extractSliceOp->getLoc(),  RankedTensorType::get(resultShape, ttSource.getType()), blockedPtr, combinedOffsetBlocked);
        }
        else 
        {
            ptr = rewriter.create<triton::AddPtrOp>(extractSliceOp->getLoc(),  ttSource.getType(), blockedPtr, combinedOffsetBlocked);
        }



        triton::LoadOp loadOp;
        if(combinedBoundBlocked)
        {
            loadOp = rewriter.create<triton::LoadOp>(extractSliceOp->getLoc(), ptr, combinedBoundBlocked, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
        }
        else
        {
            loadOp = rewriter.create<triton::LoadOp>(extractSliceOp->getLoc(), ptr, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

        }
        if(auto shaped = mlir::dyn_cast<ShapedType>(loadOp.getType()))
        {
            SmallVector<int64_t, 2> rank;
            for(int64_t i = 0; i < shaped.getRank(); i ++)
            {
                rank.push_back(ShapedType::kDynamic);
            }
            auto cast = rewriter.create<tensor::CastOp>(loadOp->getLoc(), RankedTensorType::get(rank, shaped.getElementType()), loadOp);
            rewriter.replaceOpUsesWithIf(extractSliceOp, cast->getResults(), 
                [&](OpOperand& opOperand) {
                    return opOperand.get().getType() != loadOp.getType();
                });
            rewriter.replaceOpUsesWithIf(extractSliceOp, loadOp->getResults(), 
                [&](OpOperand& opOperand) {
                    return opOperand.get().getType() == loadOp.getType();
                });
            
            rewriter.eraseOp(extractSliceOp);
        }
        else 
        {
            rewriter.replaceOp(extractSliceOp, loadOp);
        }
        
        if(toTensor->getUsers().empty())
        {
            rewriter.eraseOp(toTensor);
        }
        if(castOp->getUsers().empty())
        {
            rewriter.eraseOp(castOp);
        }
        if(toReplace != extractSliceOp)
        {
            rewriter.replaceOp(toReplace, loadOp);
            if(tensorCastOp)
            {
                rewriter.eraseOp(tensorCastOp);
            }
            if(extractOp && collapseOp)
            {
                rewriter.eraseOp(collapseOp.getDefiningOp());
            }
        }

        return success();
    }
};

class ConvertToTensor : public OpConversionPattern<mlir::bufferization::ToTensorOp>{
    public:
    using mlir::OpConversionPattern<mlir::bufferization::ToTensorOp>::OpConversionPattern;
    ConvertToTensor(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::bufferization::ToTensorOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::bufferization::ToTensorOp toTensor, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {

        if(toTensor->getUsers().empty())
        {
            rewriter.eraseOp(toTensor);
            return success();
        }
        else
        {
            // for(auto user: toTensor->getUsers())
            // {
            //     user->dump();
            // }
        }

        return failure();
    }
};

class ConvertUnrealizedCast : public OpConversionPattern<mlir::UnrealizedConversionCastOp>{
    public:
    using mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp>::OpConversionPattern;
    ConvertUnrealizedCast(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::UnrealizedConversionCastOp cast, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        if(cast->getUsers().empty())
        {
            rewriter.eraseOp(cast);
            return success();
        }
        else
        {
            if(adaptor.getInputs().size() == 1)
            {
                if(auto index = mlir::dyn_cast_if_present<mlir::index::ConstantOp>(adaptor.getInputs()[0].getDefiningOp()))
                {
                    ShapedType shape = mlir::cast<ShapedType>(cast->getResultTypes()[0]);
                    auto range_op = rewriter.create<triton::MakeRangeOp>(cast->getLoc(), RankedTensorType::get(shape.getShape(), rewriter.getIntegerType(32)), 0, shape.getShape()[0]);
                    auto indexCast = rewriter.create<arith::IndexCastOp>(cast->getLoc(), shape, range_op);
                    rewriter.replaceOp(cast, indexCast);
                    return success();
                }
            }
        }

        return failure();
    }
};

template<typename T>
class ConvertArithIndex : public OpConversionPattern<T> {
    public:
    using mlir::OpConversionPattern<T>::OpConversionPattern;
    ConvertArithIndex(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<T>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(T arith_op,  typename T::Adaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {

        Operation* op = arith_op.getOperation();
        auto converter = mlir::OpConversionPattern<T>::getTypeConverter();
        llvm::SmallVector<Type, 4> convertedTypes;
        llvm::SmallVector<Value, 4> newOperands;
        if(failed(converter->convertTypes(op->getOperandTypes(), convertedTypes)))
        {
            return failure();
        }

        for(auto operand: llvm::zip(convertedTypes,op->getOperands()))
        {
            newOperands.push_back(converter->materializeTargetConversion(rewriter, op->getLoc(), std::get<0>(operand), std::get<1>(operand)));
        }

        Value new_arith_op = rewriter.create<T>(op->getLoc(), newOperands);
        rewriter.replaceOp(arith_op, converter->materializeSourceConversion(rewriter, op->getLoc(), op->getResultTypes()[0], new_arith_op));

        return success();
    }
};

class ConvertLinalgReduceOp : public OpConversionPattern<mlir::linalg::ReduceOp> {
    public:
    using mlir::OpConversionPattern<linalg::ReduceOp>::OpConversionPattern;
    ConvertLinalgReduceOp(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<linalg::ReduceOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(linalg::ReduceOp reduceOp,  OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {

        assert(adaptor.getDimensions().size() == 1);
        triton::ReduceOp ttReduceOp = rewriter.create<triton::ReduceOp>(reduceOp->getLoc(), adaptor.getInputs(), static_cast<int>(adaptor.getDimensions().front()));
        auto arg_types = reduceOp.getBody()->getArgumentTypes();
        std::vector locs = {reduceOp->getOperand(0).getLoc(), reduceOp->getOperand(1).getLoc()};
        // rewriter.setInsertionPointToStart(ttReduceOp.getBody());
        Block* reduceBody = rewriter.createBlock(&ttReduceOp.getBodyRegion(),{}, arg_types, locs);

        auto yieldOp = reduceOp.getBody()->getTerminator();
        rewriter.mergeBlocks(reduceOp.getBody(), reduceBody, reduceBody->getArguments());
        rewriter.setInsertionPointToEnd(ttReduceOp.getBody());
        rewriter.replaceOpWithNewOp<triton::ReduceReturnOp>(yieldOp, yieldOp->getOperands());
        rewriter.replaceAllUsesWith(reduceOp.getResult(0), ttReduceOp.getResult());
        rewriter.eraseOp(reduceOp);
        if(!isa<RankedTensorType>(ttReduceOp->getResultTypes()[0]))
        {
            
            if(auto extractOp = dyn_cast<tensor::ExtractOp>(*ttReduceOp->getUsers().begin())) 
            {
                rewriter.replaceAllUsesWith(extractOp, ttReduceOp->getResult(0));
                rewriter.eraseOp(extractOp);
            }
        }
        
        return success();
    }
};

class ConvertLinalgBroadcastOp : public OpConversionPattern<mlir::linalg::BroadcastOp> {
    public:
    using mlir::OpConversionPattern<linalg::BroadcastOp>::OpConversionPattern;
    ConvertLinalgBroadcastOp(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::linalg::BroadcastOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(linalg::BroadcastOp bcastOp,  OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {

        assert(adaptor.getDimensions().size() == 1);
        auto expandDimsOp = rewriter.create<triton::ExpandDimsOp>(bcastOp->getLoc(), adaptor.getInput(), static_cast<int>(adaptor.getDimensions().front()));
        rewriter.replaceOpWithNewOp<triton::BroadcastOp>(bcastOp, adaptor.getInit().getType(), expandDimsOp);
        return success();
    }
};

class ConvertLinalgMatmulOp : public OpConversionPattern<mlir::linalg::MatmulOp> {
    public:
    using mlir::OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;
    ConvertLinalgMatmulOp(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::linalg::MatmulOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(linalg::MatmulOp matmulOp,  OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {

        assert(matmulOp->hasOneUse());
        arith::AddFOp addOp = mlir::dyn_cast<arith::AddFOp>(*matmulOp->getUsers().begin()); 
        assert(addOp);
        Value c;
        if(matmulOp.getResult(0) == addOp.getLhs())
        {
            c = addOp.getRhs();
        }
        else
        {
            c = addOp.getLhs();
        }


        triton::DotOp dot = rewriter.create<triton::DotOp>(matmulOp->getLoc(), matmulOp.getInputs()[0], matmulOp.getInputs()[1], c);
        rewriter.eraseOp(matmulOp);
        rewriter.replaceOp(*matmulOp->getUsers().begin(), dot);
        // rewriter.eraseOp(matmulOp->getUsers())
        return success();
    }
};



template<typename T>
class ConvertArithDynamicShape : public OpConversionPattern<T> {
    public:
    using mlir::OpConversionPattern<T>::OpConversionPattern;
    ConvertArithDynamicShape(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<T>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(T arith_op,  typename T::Adaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {

        Operation* op = arith_op.getOperation();
        llvm::SmallVector<Value, 4> newOperands;
        for(auto operand: op->getOperands())
        {
            if(auto shapedOperand = mlir::dyn_cast<RankedTensorType>(operand.getType()))
            {
                if(shapedOperand.hasStaticShape())
                {
                    newOperands.push_back(operand);
                }
                else
                {
                    if(auto castOp = mlir::dyn_cast_if_present<tensor::CastOp>(operand.getDefiningOp()))
                    {
                        newOperands.push_back(castOp.getSource());
                    }
                    else
                    {
                        return failure();
                    }
                }
            }
            else 
            {
                newOperands.push_back(operand);
            }
        }

        auto new_arith_op = rewriter.create<T>(op->getLoc(), newOperands);
        rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op->getResultTypes().front(), new_arith_op.getResult());
        return success();
    }
};

class ConvertIndexConstant : public OpConversionPattern<index::ConstantOp> {
    public:
    using mlir::OpConversionPattern<index::ConstantOp>::OpConversionPattern;
    ConvertIndexConstant(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<index::ConstantOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(index::ConstantOp constantOp,  OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        
        Value i32_const = rewriter.create<arith::ConstantIntOp>(constantOp->getLoc(), constantOp.getValue().getSExtValue(), 32);
        rewriter.replaceOpWithNewOp<arith::IndexCastOp>(constantOp, IndexType::get(getContext()), i32_const);

        return success();
    }
};

class ConvertExtractStridedMetadata : public OpConversionPattern<mlir::memref::ExtractStridedMetadataOp> {
    public:
    using mlir::OpConversionPattern<mlir::memref::ExtractStridedMetadataOp>::OpConversionPattern;
    ConvertExtractStridedMetadata(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::memref::ExtractStridedMetadataOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::memref::ExtractStridedMetadataOp metadaOp,  OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        
        if(UnrealizedConversionCastOp castOp = mlir::cast_if_present<UnrealizedConversionCastOp>(metadaOp.getSource().getDefiningOp()))
        {
            rewriter.replaceAllUsesWith(metadaOp.getResults(), castOp.getOperands());
            rewriter.eraseOp(metadaOp);
        }
        else 
        {
            return failure();
        }

        return success();
    }
};

class ConvertBlockedGpuToTriton: public CometBlockedGpuToTritonBase<ConvertBlockedGpuToTriton> {

    public:
    ConvertBlockedGpuToTriton() = default;

    void runOnOperation() override 
    {
        for(gpu::GPUModuleOp gpuModule : getOperation().getOps<gpu::GPUModuleOp>())
        {
            RewritePatternSet patterns(&getContext());

            mlir::comet::TritonTypeConverter converter(&getContext());
            mlir::comet::TritonConversionTarget target(getContext(), converter);
            patterns.insert<ConvertGpuFuncToTritonFunc, ConvertForOp>( converter, &getContext());
            patterns.insert<
                            ConvertExtractSlice, 
                            ConvertInsertSlice, 
                            ConvertBlockDim, 
                            ConvertBlockId, 
                            ConvertToTensor,
                            ConvertTensorSplatOp, 
                            ConvertExtractStridedMetadata, 
                            ConvertLinalgReduceOp, 
                            ConvertLinalgBroadcastOp,
                            ConvertLinalgMatmulOp
            >( &getContext());

            if (failed(applyPartialConversion(gpuModule, target, std::move(patterns))))
            {
                return signalPassFailure();
            }
            
            auto ttFuncs = getOperation().getOps<triton::FuncOp>();
            target.addIllegalOp<bufferization::ToTensorOp>();
            RewritePatternSet patternsClean(&getContext());
            patternsClean.insert<ConvertToTensor>(&getContext());
            
            for(auto func: ttFuncs)
            {
                if (failed(applyPatternsAndFoldGreedily(func, std::move(patternsClean))))
                {
                    return signalPassFailure();
                }

            }
            PassManager pm(&getContext());
            pm.addPass(mlir::createCanonicalizerPass());
            if (failed(pm.run(gpuModule))) {
                signalPassFailure();
                return;
            }
            
            RewritePatternSet patternsClean2(&getContext());
            target.addIllegalOp<UnrealizedConversionCastOp>();
            patternsClean2.insert<ConvertUnrealizedCast>(&getContext());

            if (failed(applyPartialConversion(gpuModule, target, std::move(patternsClean2))))
            {
                return signalPassFailure();
            }
            
            RewritePatternSet patternsClean3(&getContext());
            target.addDynamicallyLegalOp<index::ConstantOp, arith::AddIOp, arith::MulIOp, arith::SubIOp>([&](Operation *op) {
                return llvm::all_of(op->getOperandTypes(),
                                    [&](Type type) { return converter.isLegal(type); }) &&
                    llvm::all_of(op->getResultTypes(),
                                    [&](Type type) { return converter.isLegal(type); });
            });

            patternsClean3.insert<ConvertIndexConstant, ConvertArithIndex<arith::AddIOp>, ConvertArithIndex<arith::MulIOp>, ConvertArithIndex<arith::SubIOp> >(converter, &getContext());

            if (failed(applyPartialConversion(gpuModule, target, std::move(patternsClean3))))
            {
                return signalPassFailure();
            }

            RewritePatternSet patternsClean4(&getContext());
            target.addDynamicallyLegalOp<arith::AddIOp, arith::MulIOp, arith::SubIOp, arith::AddFOp, arith::MulFOp, arith::SubFOp>([&](Operation *op) {
                return llvm::all_of(op->getOperandTypes(),
                                    [&](Type type) { if(RankedTensorType shapedType = mlir::dyn_cast<RankedTensorType>(type)){return shapedType.hasStaticShape(); } return true; }) &&
                    llvm::all_of(op->getResultTypes(),
                                    [&](Type type) { if(RankedTensorType shapedType = mlir::dyn_cast<RankedTensorType>(type)){return shapedType.hasStaticShape(); } return true; });
            });

            patternsClean4.insert<ConvertArithDynamicShape<arith::AddIOp>, ConvertArithDynamicShape<arith::MulIOp>, ConvertArithDynamicShape<arith::SubIOp>, ConvertArithDynamicShape<arith::AddFOp>, ConvertArithDynamicShape<arith::MulFOp>, ConvertArithDynamicShape<arith::SubFOp> >(&getContext());

            if (failed(applyPartialConversion(gpuModule, target, std::move(patternsClean4))))
            {
                return signalPassFailure();
            }
        }

    }
};



std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::comet::createConvertBlockedGpuToTritonPass() {
    return std::make_unique<ConvertBlockedGpuToTriton>();
}