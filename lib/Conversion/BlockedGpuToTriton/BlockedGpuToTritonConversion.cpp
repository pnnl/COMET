#include "comet//Conversion/BlockedGpuToTriton/BlockedGpuToTritonConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/BuiltinDialect.h"

#include <optional>


mlir::comet::TritonTypeConverter::TritonTypeConverter(MLIRContext *context) 
: context(context)
{

    addConversion([](mlir::Type t) {
        return t;
    });

    addConversion([this] (mlir::MemRefType memrefType, SmallVectorImpl<Type> & convertedTypes){
        Type i32 = mlir::IntegerType::get(this->context, 32);
        if(mlir::isa<IndexType>(memrefType.getElementType()))
        {
            convertedTypes.push_back(mlir::triton::PointerType::get(i32, 1));
        }
        else 
        {
            convertedTypes.push_back(mlir::triton::PointerType::get(memrefType.getElementType(), 1));
        }

        convertedTypes.push_back(i32); //  offset
        for(int64_t i = 0; i < memrefType.getRank(); i++) // sizes
        {
            convertedTypes.push_back(i32);
        }
        for(int64_t i = 0; i < memrefType.getRank(); i++) // strides
        {
            convertedTypes.push_back(i32);
        }

        return success();
    }); 

    addConversion([this] (mlir::IndexType indexType){
                {
            return IntegerType::get(this->context, 32);
        }
    }); 

    addConversion([this] (mlir::RankedTensorType rankedType){
        {
            if(rankedType.getElementType().isIndex())
            {
                return RankedTensorType::get(rankedType.getShape(), IntegerType::get(this->context, 32));
            }
            else 
            {
                return rankedType;
            }
        }
    }); 

    addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> std::optional<Value>   {
        llvm::errs() << "Called\n";
        if (inputs.size() != 1)
            return std::nullopt;
        return builder.create<arith::IndexCastOp>(loc, type, inputs)->getResult(0);
    });

    addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> std::optional<Value>   {
        llvm::errs() << "Called\n";
        if (inputs.size() != 1)
            return std::nullopt;
        return builder.create<arith::IndexCastOp>(loc, type, inputs)->getResult(0);
    });

    addArgumentMaterialization([](OpBuilder &builder, MemRefType memrefType, ValueRange values, Location loc)-> std::optional<Value> {
        assert(values.size() == memrefType.getRank() * 2 + 2);
        return builder.create<UnrealizedConversionCastOp>(loc, memrefType, values)->getResult(0);
    });

    addArgumentMaterialization([](OpBuilder &builder, IndexType indexType, ValueRange values, Location loc)-> std::optional<Value> {
        return builder.create<arith::IndexCastOp>(loc, indexType, values)->getResult(0);
    });
}

mlir::comet::TritonConversionTarget::TritonConversionTarget(MLIRContext &context, TritonTypeConverter &typeConverter)
    : ConversionTarget(context) 
{
    addLegalDialect<arith::ArithDialect, scf::SCFDialect, triton::TritonDialect, mlir::BuiltinDialect, func::FuncDialect>();
    addLegalOp<tensor::CastOp, bufferization::ToTensorOp, gpu::GPUModuleOp>();
    addIllegalDialect<gpu::GPUDialect>();
    addIllegalOp<tensor::InsertSliceOp, tensor::ExtractSliceOp, tensor::SplatOp>();
}