#include "comet//Conversion/BlockedGpuToTriton/BlockedGpuToTritonConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

    addConversion([] (mlir::MemRefType memrefType){
        // if(mlir::isa<IndexType>(memrefType.getElementType()))
        // {
        //     return mlir::triton::PointerType::get(mlir::IntegerType::get(this->context, 32), 1);
        // }
        // else 
        {
            return mlir::triton::PointerType::get(memrefType.getElementType(), 1);
        }
    }); 

    addArgumentMaterialization([](OpBuilder &builder, MemRefType memrefType, ValueRange values, Location loc)-> std::optional<Value> {
        return builder.create<UnrealizedConversionCastOp>(loc, memrefType, values)->getResult(0);
    });
}

mlir::comet::TritonConversionTarget::TritonConversionTarget(MLIRContext &context, TritonTypeConverter &typeConverter)
    : ConversionTarget(context) 
{
    addLegalDialect<arith::ArithDialect, scf::SCFDialect, triton::TritonDialect, mlir::BuiltinDialect>();
    addLegalOp<tensor::CastOp, bufferization::ToTensorOp, gpu::GPUModuleOp>();
    addIllegalDialect<gpu::GPUDialect>();
    addLegalOp<gpu::GPUFuncOp, gpu::GPUModuleOp>();
    addIllegalOp<tensor::InsertSliceOp, tensor::ExtractSliceOp>();
}