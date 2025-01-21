#include "comet/Conversion/BlockedGpuToTriton/BlockedGpuToTriton.h"
#include "comet/Conversion/BlockedGpuToTriton/BlockedGpuToTritonConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

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
        llvm::SmallVector<Type, 4> tritonFuncTypes;
        llvm::SmallVector<Value, 4> materializedValues;
        rewriter.setInsertionPoint(gpuFunc->getParentOfType<gpu::GPUModuleOp>());
        for(auto argType: gpuFunc.getArgumentTypes())
        {
            tritonFuncTypes.push_back(converter->convertType(argType));
        }

        auto tritonFuncType = rewriter.getFunctionType(TypeRange(tritonFuncTypes), gpuFunc.getFunctionType().getResults());
        auto tritonFunc = rewriter.create<triton::FuncOp>(gpuFunc->getLoc(), gpuFunc.getName(), tritonFuncType);
        auto funcBlock = tritonFunc.addEntryBlock();
        rewriter.setInsertionPointToStart(funcBlock);
        for(auto arg: llvm::zip(funcBlock->getArguments(),gpuFunc.getBody().getArguments()))
        {
            if(!converter->isLegal(std::get<1>(arg).getType()))
            {
                auto conveted = converter->materializeArgumentConversion(rewriter, std::get<0>(arg).getLoc(), std::get<1>(arg).getType(), std::get<0>(arg));
                materializedValues.push_back(conveted);
            }
            else
            {
                materializedValues.push_back(std::get<0>(arg)); 
            }
            rewriter.replaceAllUsesWith(std::get<1>(arg), materializedValues.back());
        }

        rewriter.eraseOp(gpuFunc.getBody().front().getTerminator());
        rewriter.mergeBlocks(&gpuFunc.getBody().front(), funcBlock, materializedValues);
        rewriter.setInsertionPointToEnd(funcBlock);
        rewriter.create<triton::ReturnOp>(tritonFunc->getLoc());
        rewriter.eraseOp(gpuFunc);
        llvm::errs() <<"RAN ConvertGPUFUNC\n";

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

class ConvertInsertSlice : public OpConversionPattern<mlir::tensor::InsertSliceOp>{
    public:
    using mlir::OpConversionPattern<mlir::tensor::InsertSliceOp>::OpConversionPattern;
    ConvertInsertSlice(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::tensor::InsertSliceOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::tensor::InsertSliceOp insertSliceOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {

        llvm::errs() <<"RUNNING InsertSlice\n";
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

        llvm::SmallVector<Value, 2> blockedOffsets;
        llvm::SmallVector<Value, 2> blockedBounds;
        for(size_t i = 0; i < insertSliceOp.getOffsets().size(); i++)
        {
            auto offset = insertSliceOp.getOffsets()[i];
            if(auto castOp = mlir::dyn_cast<UnrealizedConversionCastOp>(offset.getDefiningOp()))
            {
                Value cast;
                if(auto shaped = mlir::dyn_cast<ShapedType>(castOp.getInputs().front().getType()))
                {
                    cast = rewriter.create<arith::IndexCastOp>(insertSliceOp->getLoc(), RankedTensorType::get(shaped.getShape(), rewriter.getIntegerType(32)), castOp.getInputs().front());
                }
                else 
                {
                    cast = rewriter.create<arith::IndexCastOp>(insertSliceOp->getLoc(), rewriter.getIntegerType(32), castOp.getInputs().front());
                }
                blockedOffsets.push_back(cast);
            }

            auto bound = insertSliceOp.getSizes()[i];
            auto min = mlir::cast<arith::MinSIOp>(bound.getDefiningOp());
            auto blockSize =
                cast<arith::ConstantIndexOp>(min.getLhs().getDefiningOp()).value();
            Value cast = rewriter.create<arith::IndexCastOp>(min->getLoc(), rewriter.getIntegerType(32), min.getRhs());

            auto blockedBound =  rewriter.create<triton::SplatOp>(min->getLoc(), RankedTensorType::get({blockSize}, rewriter.getIntegerType(32)),  cast);
            auto blockedBlockSize =  rewriter.create<triton::MakeRangeOp>(min.getLhs().getLoc(), RankedTensorType::get({blockSize}, rewriter.getIntegerType(32)), 0, blockSize);
            auto cmp = rewriter.create<arith::CmpIOp>(min->getLoc(), arith::CmpIPredicate::slt, blockedBlockSize, blockedBound);
            blockedBounds.push_back(cmp);
        }

        if(blockedOffsets.size() == 1)
        {
            auto shaped = mlir::cast<ShapedType>(blockedOffsets[0].getType());
            auto blockedPtr = rewriter.create<triton::SplatOp>(insertSliceOp->getLoc(), RankedTensorType::get(shaped.getShape(), rewriter.getIntegerType(32)), ttDest);
            auto ptr = rewriter.create<triton::AddPtrOp>(insertSliceOp->getLoc(),  RankedTensorType::get({mlir::cast<ShapedType>(blockedOffsets[0].getType()).getShape()}, rewriter.getIntegerType(32)), blockedPtr, blockedOffsets[0]);
            if(ShapedType sourceShaped = mlir::dyn_cast<ShapedType>(ttSource.getType());  sourceShaped && !sourceShaped.hasStaticShape())
            {
                auto castOp = rewriter.create<tensor::CastOp>(insertSliceOp->getLoc(), RankedTensorType::get(shaped.getShape(), insertSliceOp.getDestType().getElementType()), ttSource);
                rewriter.create<triton::StoreOp>( insertSliceOp->getLoc(), ptr, castOp, blockedBounds[0], mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
            }
            else 
            {
                rewriter.create<triton::StoreOp>( insertSliceOp->getLoc(), ptr, ttSource, blockedBounds[0], mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
            }
            rewriter.eraseOp(insertSliceOp);
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
        else if(blockedOffsets.size() == 2)
        {
            auto shaped0 = mlir::cast<ShapedType>(blockedOffsets[0].getType());
            auto shaped1 = mlir::cast<ShapedType>(blockedOffsets[1].getType());
            auto resultShape = RankedTensorType::get({shaped0.getDimSize(0), shaped1.getDimSize(0)}, rewriter.getIntegerType(32));
            auto boundShape = RankedTensorType::get({shaped0.getDimSize(0), shaped1.getDimSize(0)}, rewriter.getIntegerType(1));
            Value ybound = rewriter.create<triton::ExpandDimsOp>(insertSliceOp->getLoc(), blockedBounds[0], 1);
            ybound = rewriter.create<triton::BroadcastOp>(insertSliceOp->getLoc(), boundShape, ybound);
            Value xbound = rewriter.create<triton::ExpandDimsOp>(insertSliceOp->getLoc(), blockedBounds[1], 0);
            xbound = rewriter.create<triton::BroadcastOp>(insertSliceOp->getLoc(), boundShape, xbound);
            Value combinedBlockedBound = rewriter.create<arith::AndIOp>(insertSliceOp->getLoc(), ybound, xbound);

            Value yOffsets = rewriter.create<triton::ExpandDimsOp>(insertSliceOp->getLoc(), blockedOffsets[0], 1);
            Value cast = rewriter.create<arith::IndexCastOp>(insertSliceOp->getLoc(), rewriter.getIntegerType(32), insertSliceOp.getStrides()[0]);

            Value yStrides = rewriter.create<triton::SplatOp>(insertSliceOp->getLoc(), yOffsets.getType(), cast);
            yOffsets = rewriter.create<arith::MulIOp>(insertSliceOp->getLoc(), yOffsets, yStrides);
            yOffsets = rewriter.create<triton::BroadcastOp>(insertSliceOp->getLoc(), resultShape, yOffsets);

            Value xOffsets = rewriter.create<triton::ExpandDimsOp>(insertSliceOp->getLoc(), blockedOffsets[1], 0);
            xOffsets = rewriter.create<triton::BroadcastOp>(insertSliceOp->getLoc(), resultShape, xOffsets);
            Value combinedBlockedOffsets = rewriter.create<arith::AddIOp>(insertSliceOp->getLoc(), yOffsets, xOffsets);
            auto blockedPtr = rewriter.create<triton::SplatOp>(insertSliceOp->getLoc(), RankedTensorType::get(resultShape.getShape(), ttDest.getType()), ttDest);
            auto ptr = rewriter.create<triton::AddPtrOp>(insertSliceOp->getLoc(),  RankedTensorType::get(resultShape.getShape(), ttDest.getType()), blockedPtr, combinedBlockedOffsets);
            if(ShapedType sourceShaped = mlir::dyn_cast<ShapedType>(ttSource.getType());  sourceShaped && !sourceShaped.hasStaticShape())
            {
                auto castOp = rewriter.create<tensor::CastOp>(insertSliceOp->getLoc(), RankedTensorType::get(resultShape.getShape(), insertSliceOp.getDestType().getElementType()), ttSource);
                rewriter.create<triton::StoreOp>( insertSliceOp->getLoc(), ptr, castOp, combinedBlockedBound, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
                auto placeholder = rewriter.create<UnrealizedConversionCastOp>(insertSliceOp->getLoc(), insertSliceOp.getResultType(), ValueRange(ptr));
                rewriter.replaceOp(insertSliceOp, placeholder->getResult(0));

            }
            else 
            {
                rewriter.create<triton::StoreOp>( insertSliceOp->getLoc(), ptr, ttSource, combinedBlockedBound, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
                auto placeholder = rewriter.create<UnrealizedConversionCastOp>(insertSliceOp->getLoc(), insertSliceOp.getResultType(), ValueRange(ptr));
                rewriter.replaceOp(insertSliceOp, placeholder->getResult(0));
            }
            // rewriter.eraseOp(insertSliceOp);
            // rewriter.create<triton::StoreOp>(insertSliceOp->getLoc(), ptr, ttSource, combinedBlockedBound, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
            
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
        else {
            assert(false && "blocks of dimension > 2 are not currently supported");
        }

        return failure();
    }
};

class ConvertExtractSlice : public OpConversionPattern<mlir::tensor::ExtractSliceOp>{
    public:
    using mlir::OpConversionPattern<mlir::tensor::ExtractSliceOp>::OpConversionPattern;
    ConvertExtractSlice(mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::tensor::ExtractSliceOp>(ctx) {}
    mlir::LogicalResult
    matchAndRewrite(mlir::tensor::ExtractSliceOp extractSliceOp, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        llvm::errs() <<"RUNNING ExtractSlice\n";
        Value source = adaptor.getSource();
        Value ttSource;
        bufferization::ToTensorOp toTensor = mlir::dyn_cast<bufferization::ToTensorOp>(source.getDefiningOp());
        UnrealizedConversionCastOp castOp;
        if(toTensor)
        {
            if((castOp = mlir::dyn_cast<UnrealizedConversionCastOp>(toTensor.getMemref().getDefiningOp())))
            {
                if(triton::PointerType ttPtr =  mlir::dyn_cast<triton::PointerType>(castOp->getOperand(0).getType()))
                {
                    ttSource = castOp->getOperand(0);
                }
            }
        }
        else if((castOp = mlir::dyn_cast<UnrealizedConversionCastOp>(toTensor.getMemref().getDefiningOp())))
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

        llvm::SmallVector<Value, 2> blockedOffsets;
        llvm::SmallVector<Value, 2> blockedBounds;
        for(size_t i = 0; i < adaptor.getOffsets().size(); i++)
        {
            auto offset = extractSliceOp.getOffsets()[i];
            if(auto castOp = mlir::dyn_cast<UnrealizedConversionCastOp>(offset.getDefiningOp()))
            {
                Value cast;
                if(auto shaped = mlir::dyn_cast<ShapedType>(castOp.getInputs().front().getType()))
                {
                    cast = rewriter.create<arith::IndexCastOp>(extractSliceOp->getLoc(), RankedTensorType::get(shaped.getShape(), rewriter.getIntegerType(32)), castOp.getInputs().front());
                }
                else 
                {
                    cast = rewriter.create<arith::IndexCastOp>(extractSliceOp->getLoc(), rewriter.getIntegerType(32), castOp.getInputs().front());
                }
                blockedOffsets.push_back(cast);
            }

            auto bound = adaptor.getSizes()[i];
            auto min = mlir::cast<arith::MinSIOp>(bound.getDefiningOp());
            auto blockSize =
                cast<arith::ConstantIndexOp>(min.getLhs().getDefiningOp()).value();
            Value cast = rewriter.create<arith::IndexCastOp>(min->getLoc(), rewriter.getIntegerType(32), min.getRhs());

            auto blockedBound =  rewriter.create<triton::SplatOp>(min->getLoc(), RankedTensorType::get({blockSize}, rewriter.getIntegerType(32)),  cast);
            auto blockedBlockSize =  rewriter.create<triton::MakeRangeOp>(min.getLhs().getLoc(), RankedTensorType::get({blockSize}, rewriter.getIntegerType(32)), 0, blockSize);
            auto cmp = rewriter.create<arith::CmpIOp>(min->getLoc(), arith::CmpIPredicate::slt, blockedBlockSize, blockedBound);
            blockedBounds.push_back(cmp);
        }

        if(blockedOffsets.size() == 1)
        {
            auto shaped = mlir::cast<ShapedType>(blockedOffsets[0].getType());
            auto blockedPtr = rewriter.create<triton::SplatOp>(extractSliceOp->getLoc(), RankedTensorType::get(shaped.getShape(), ttSource.getType()), ttSource);
            auto ptr = rewriter.create<triton::AddPtrOp>(extractSliceOp->getLoc(),  RankedTensorType::get({mlir::cast<ShapedType>(blockedOffsets[0].getType()).getShape()}, ttSource.getType()), blockedPtr, blockedOffsets[0]);
            auto loadOp = rewriter.create<triton::LoadOp>(extractSliceOp->getLoc(), ptr, blockedBounds[0], mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
            if(auto shaped = mlir::dyn_cast<ShapedType>(loadOp.getType()))
            {
                SmallVector<int64_t, 2> rank;
                for(int64_t i = 0; i < shaped.getRank(); i ++)
                {
                    rank.push_back(ShapedType::kDynamic);
                }
                auto cast = rewriter.create<tensor::CastOp>(loadOp->getLoc(), RankedTensorType::get(rank, shaped.getElementType()), loadOp);
                rewriter.replaceOp(extractSliceOp, cast);
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
        
            return success();
        }
        else if(blockedOffsets.size() == 2)
        {
            auto shaped0 = mlir::cast<ShapedType>(blockedOffsets[0].getType());
            auto shaped1 = mlir::cast<ShapedType>(blockedOffsets[1].getType());
            auto boundShape = RankedTensorType::get({shaped0.getDimSize(0), shaped1.getDimSize(0)}, rewriter.getIntegerType(1));

            auto resultShape = RankedTensorType::get({shaped0.getDimSize(0), shaped1.getDimSize(0)}, rewriter.getIntegerType(32));
            Value ybound = rewriter.create<triton::ExpandDimsOp>(extractSliceOp->getLoc(), blockedBounds[0], 1);
            ybound = rewriter.create<triton::BroadcastOp>(extractSliceOp->getLoc(), boundShape, ybound);
            Value xbound = rewriter.create<triton::ExpandDimsOp>(extractSliceOp->getLoc(), blockedBounds[1], 0);
            xbound = rewriter.create<triton::BroadcastOp>(extractSliceOp->getLoc(), boundShape, xbound);
            Value combinedBlockedBound = rewriter.create<arith::AndIOp>(extractSliceOp->getLoc(), ybound, xbound);

            Value yOffsets = rewriter.create<triton::ExpandDimsOp>(extractSliceOp->getLoc(), blockedOffsets[0], 1);
            Value cast = rewriter.create<arith::IndexCastOp>(extractSliceOp->getLoc(), rewriter.getIntegerType(32), adaptor.getStrides()[0]);
            Value yStrides = rewriter.create<triton::SplatOp>(extractSliceOp->getLoc(), yOffsets.getType(), cast);
            yOffsets = rewriter.create<arith::MulIOp>(extractSliceOp->getLoc(), yOffsets, yStrides);
            yOffsets = rewriter.create<triton::BroadcastOp>(extractSliceOp->getLoc(), resultShape, yOffsets);

            Value xOffsets = rewriter.create<triton::ExpandDimsOp>(extractSliceOp->getLoc(), blockedOffsets[1], 0);
            xOffsets = rewriter.create<triton::BroadcastOp>(extractSliceOp->getLoc(), resultShape, xOffsets);
            Value combinedBlockedOffsets = rewriter.create<arith::AddIOp>(extractSliceOp->getLoc(), yOffsets, xOffsets);
            auto blockedPtr = rewriter.create<triton::SplatOp>(extractSliceOp->getLoc(), RankedTensorType::get(resultShape.getShape(), ttSource.getType()), ttSource);
            auto ptr = rewriter.create<triton::AddPtrOp>(extractSliceOp->getLoc(),  RankedTensorType::get(resultShape.getShape(), ttSource.getType()), blockedPtr, combinedBlockedOffsets);
            
            auto loadOp = rewriter.create<triton::LoadOp>(extractSliceOp->getLoc(), ptr, combinedBlockedBound, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
            if(auto shaped = mlir::dyn_cast<ShapedType>(loadOp.getType()))
            {
                SmallVector<int64_t, 2> rank;
                for(int64_t i = 0; i < shaped.getRank(); i ++)
                {
                    rank.push_back(ShapedType::kDynamic);
                }
                auto cast = rewriter.create<tensor::CastOp>(loadOp->getLoc(), RankedTensorType::get(rank, shaped.getElementType()), loadOp);
                rewriter.replaceOp(extractSliceOp, cast);
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
            return success();
        }
        else {
            assert(false && "blocks of dimension > 2 are not currently supported");
        }

        return failure();
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
            for(auto user: toTensor->getUsers())
            {
                user->dump();
            }
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
        cast->dump();
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
                    auto range_op = rewriter.create<triton::MakeRangeOp>(cast->getLoc(), RankedTensorType::get(shape.getShape(), rewriter.getIntegerType(32)), rewriter.getI32IntegerAttr(0), index.getValueAttr());
                    auto indexCast = rewriter.create<tensor::CastOp>(cast->getLoc(), shape, range_op);
                    rewriter.replaceOp(cast, indexCast);
                }
            }
        }

        return failure();
    }
};

class ConvertBlockedGpuToTriton: public CometBlockedGpuToTritonBase<ConvertBlockedGpuToTriton> {

    public:
    ConvertBlockedGpuToTriton() = default;

    void runOnOperation() override 
    {
        llvm::errs() << "Running ConvertBlockedGpuToTriton\n";
        gpu::GPUModuleOp gpuModule = *getOperation().getOps<gpu::GPUModuleOp>().begin();
        
        RewritePatternSet patterns(&getContext());

        mlir::comet::TritonTypeConverter converter(&getContext());
        mlir::comet::TritonConversionTarget target(getContext(), converter);
        patterns.insert<ConvertGpuFuncToTritonFunc>( converter, &getContext());
        patterns.insert<ConvertExtractSlice, ConvertInsertSlice, ConvertBlockDim, ConvertBlockId, ConvertToTensor>( &getContext());

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
        
        RewritePatternSet patternsClean2(&getContext());
        for(auto func: ttFuncs)
        {
            patternsClean.insert<ConvertUnrealizedCast>(&getContext());

            if (failed(applyPartialConversion(func, target, std::move(patternsClean2))))
            {
                return signalPassFailure();
            }

        }


        llvm::errs() << "Completed!!\n";
    }
};



std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::comet::createConvertBlockedGpuToTritonPass() {
    return std::make_unique<ConvertBlockedGpuToTriton>();
}