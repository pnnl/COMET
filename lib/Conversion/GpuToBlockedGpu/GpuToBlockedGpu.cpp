#include "comet/Conversion/GpuToBlockedGpu/GpuToBlockedGpu.h"
#include "comet/Conversion/ParallelLoopsToGpu/ParallelLoopsToGpu.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "comet/Conversion/GpuToBlockedGpu/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <map>
using namespace::mlir;

#define GEN_PASS_CLASSES
#include "comet/Conversion/GpuToBlockedGpu/Passes.h.inc"

struct BlockInfo {
    BlockInfo(size_t& argIndex, uint64_t& blockSize, mlir::Value index, mlir::Value max) : argIndex(argIndex), blockSize(blockSize), index(index), max(max) {}
    size_t argIndex;
    uint64_t blockSize;
    mlir::Value index;
    mlir::Value max;
};

bool checkOperandsSameShape(Operation* op) 
{
    if(op->getNumOperands() == 1) 
    {
        return true;
    }
    else if(op->getNumOperands() == 2)  
    {
        ShapedType shape0 = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
        ShapedType shape1 = mlir::dyn_cast<ShapedType>(op->getOperand(1).getType());
        if(shape0 == nullptr && shape1 == nullptr)
        {
            return true;
        }
        else if(shape0 == nullptr || shape1 == nullptr)
        {
            return false;
        }
        else 
        {
            if(shape0.getRank() == shape1.getRank())
            {
                for(int64_t i = 0; i < shape0.getRank(); i++)
                {
                    if(shape0.getDimSize(i) != shape1.getDimSize(i))
                    {
                        return false;
                    }
                }
            }
            else 
            {
                return false;
            }

            return true;
        }
    }
    else 
    {
        assert(false && "Operations with more than two operands are not handled currently");
    }
}
bool checkOperandsSameType(Operation* op) 
{
    if(op->getNumOperands() == 1) 
    {
        return true;
    }
    else if(op->getNumOperands() == 2)  
    {
        return op->getOperandTypes()[0] == op->getOperandTypes()[1];
    }
    else 
    {
        assert(false && "Operations with more than two operands are not handled currently");
    }
}

class ConvertGpuToBlockedGpu: public CometGpuToBlockedGpuBase<ConvertGpuToBlockedGpu> {
    public:
    ConvertGpuToBlockedGpu() = default;

    void runOnOperation() override 
    {
        mlir::gpu::GPUFuncOp funcOp = getOperation();
        mlir::OpBuilder builder(funcOp);

        for(auto arg: funcOp.getArguments())
        {
            if(mlir::isa<mlir::MemRefType>(arg.getType()))
            {
                builder.setInsertionPointToStart(&funcOp.getBody().front());   
                auto tensor = builder.create<mlir::bufferization::ToTensorOp>(arg.getLoc(), arg);
                for(auto user: llvm::make_early_inc_range(arg.getUsers()))
                {
                    if(mlir::memref::StoreOp storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(user))
                    {
                        
                        builder.setInsertionPoint(storeOp);
                        mlir::tensor::InsertOp insertOp = builder.create<mlir::tensor::InsertOp>(storeOp->getLoc(), storeOp.getValueToStore(), tensor, storeOp.getIndices());
                        storeOp->erase();
                        // storeOp->replaceAllUsesWith(insertOp);
                    }
                    else if(mlir::memref::LoadOp loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(user))
                    {
                        builder.setInsertionPoint(loadOp);
                        mlir::tensor::ExtractOp extractOp = builder.create<mlir::tensor::ExtractOp>(loadOp->getLoc(), tensor, loadOp.getIndices());
                        loadOp->replaceAllUsesWith(extractOp);
                        loadOp->erase();
                    }
                }
            }
        }

        llvm::MapVector<mlir::tensor::ExtractOp, llvm::SmallVector<BlockInfo, 2>> extracts;
        llvm::MapVector<mlir::tensor::InsertOp, llvm::SmallVector<BlockInfo, 2>> inserts;
        llvm::SmallVector<scf::ForOp, 2> forOps;
        llvm::SmallSet<mlir::Operation*, 4> needCheck;

        funcOp->walk([&](mlir::scf::ForOp forOp) {
            if(forOp->hasAttr("blockSize"))
            {
                uint64_t blockSize = forOp->getAttrOfType<mlir::IntegerAttr>("blockSize").getUInt();
                builder.setInsertionPointToStart(forOp.getBody());
                auto blockedIndex = builder.create<UnrealizedConversionCastOp>(forOp.getInductionVar().getLoc(), TypeRange({RankedTensorType::get({static_cast<int64_t>(blockSize)}, builder.getIndexType())}), forOp.getInductionVar());
                forOp.getInductionVar().replaceAllUsesExcept(blockedIndex.getResult(0), blockedIndex);
                needCheck.insert(blockedIndex);
                llvm::SmallVector<mlir::Value, 4> inductionVars;
                forOps.push_back(forOp);
                inductionVars.push_back(blockedIndex->getResult(0));
                for(size_t i = 0; i < inductionVars.size(); i++)
                {
                    mlir::Value inductionVar = inductionVars[i];
                    for(auto user: llvm::make_early_inc_range(inductionVar.getUsers()))
                    {
                        if(mlir::tensor::InsertOp insertOp = mlir::dyn_cast<mlir::tensor::InsertOp>(user))
                        {
                            if (inserts.find(insertOp) == inserts.end())
                            {
                                llvm::SmallVector<BlockInfo, 2> vec;
                                inserts[insertOp] = vec;
                            }
                            for(size_t i = 0; i < insertOp.getIndices().size(); i++)
                            {
                                if(insertOp.getIndices()[i] == inductionVar)
                                {
                                    builder.setInsertionPoint(insertOp);
                                    auto cast = builder.create<UnrealizedConversionCastOp>(inductionVar.getLoc(), builder.getIndexType(), inductionVar);
                                    inserts[insertOp].push_back(BlockInfo(i, blockSize, cast->getResult(0), forOp.getUpperBound()));
                                }
                            }
                        }
                        else if(mlir::tensor::ExtractOp extractOp = mlir::dyn_cast<mlir::tensor::ExtractOp>(user))
                        {
                            if (extracts.find(extractOp) == extracts.end())
                            {
                                llvm::SmallVector<BlockInfo, 2> vec;
                                extracts[extractOp] = vec;
                            }
                            for(size_t i = 0; i < extractOp.getIndices().size(); i++)
                            {
                                if(extractOp.getIndices()[i] == inductionVar)
                                {
                                    builder.setInsertionPoint(extractOp);
                                    auto cast = builder.create<UnrealizedConversionCastOp>(inductionVar.getLoc(), builder.getIndexType(), inductionVar);
                                    extracts[extractOp].push_back(BlockInfo(i, blockSize, cast->getResult(0), forOp.getUpperBound()));
                                }
                            }
                        }
                        else 
                        {
                            for(auto res: user->getResults())
                            {
                                inductionVars.push_back(res);
                            }
                        }
                    }
                }
            }
        });

        // for(std::pair<Value, int64_t> forOpIndex: forOpIndices)
        // {
        //     builder.setInsertionPointToStart(std::get<0>(forOpIndex).getParentBlock());
        //     auto blockedIndex = builder.create<UnrealizedConversionCastOp>(std::get<0>(forOpIndex).getLoc(), TypeRange({RankedTensorType::get({static_cast<int64_t>(std::get<1>(forOpIndex))}, builder.getIndexType())}), ValueRange(std::get<0>(forOpIndex)));
        //     std::get<0>(forOpIndex).replaceAllUsesExcept(blockedIndex.getResult(0), blockedIndex);
        //     needCheck.insert(blockedIndex);
        // }


        for(auto extract: extracts)
        {
            mlir::SmallVector<Value, 2> blockSizes;
            mlir::SmallVector<int64_t, 2> blockSizesLiteral;
            mlir::SmallVector<int64_t, 2> static_offsets;
            mlir::SmallVector<mlir::Value, 2> offsets;
            mlir::SmallVector<int64_t, 2> stridesLiteral;
            mlir::SmallVector<Value, 2> strides;
            std::sort(extract.second.begin(), extract.second.end(), [](BlockInfo& a, BlockInfo& b) {return a.argIndex < b.argIndex;} );
            int64_t rank = extract.first.getTensor().getType().getRank();
            bufferization::ToTensorOp toTensorOp = mlir::cast<bufferization::ToTensorOp>(extract.first.getTensor().getDefiningOp());
            builder.setInsertionPoint(extract.first);

            auto metaData = builder.create<memref::ExtractStridedMetadataOp>(extract.first->getLoc(), toTensorOp.getMemref());
            // UnrealizedConversionCastOp unrealized_cast = mlir::cast<UnrealizedConversionCastOp>(toTensorOp.getMemref().getDefiningOp());
            for(int64_t i = 0; i < rank; i++)
            {
                blockSizes.push_back(extract.second[i].max);
                blockSizesLiteral.push_back(ShapedType::kDynamic);
                offsets.push_back(extract.second[i].index);
                stridesLiteral.push_back(ShapedType::kDynamic);
                static_offsets.push_back(ShapedType::kDynamic);
                Value stride = metaData->getResult(2 + rank + i); // tt.ptr + offset + (size) * rank + (stride) * rank 
                strides.push_back(stride);
                // auto affineApply = mlir::cast<affine::AffineMinOp>(extract.second[i].max.getDefiningOp());
                // if(i < rank-1)
                // {
                //     strides.push_back(affineApply.getDimOperands()[0]);
                // }
            }


            auto type = mlir::RankedTensorType::get(blockSizesLiteral, extract.first.getTensor().getType().getElementType());
            // strides.push_back(builder.create<arith::ConstantIndexOp>(extract.first->getLoc(), 1));
            auto extractSlice = builder.create<mlir::tensor::ExtractSliceOp>(extract.first->getLoc(), type, extract.first.getTensor(), mlir::ValueRange(offsets), mlir::ValueRange(blockSizes), mlir::ValueRange(strides), static_offsets, mlir::ArrayRef(blockSizesLiteral), mlir::ArrayRef(stridesLiteral));
            SmallVector<int64_t, 2> static_shape;
            for(auto block: extract.second)
            {
                static_shape.push_back(block.blockSize);
            }
            auto castSliceShape = builder.create<mlir::tensor::CastOp>(extract.first->getLoc(), mlir::RankedTensorType::get(static_shape, extract.first.getTensor().getType().getElementType()), extractSlice);
            extract.first.replaceAllUsesWith(castSliceShape.getResult());
            extract.first->erase();
            needCheck.insert(castSliceShape);
        }

        funcOp.dump();
        size_t size_before;
        SmallVector<Operation*, 4> newOpsToCheck;
        do 
        {
            size_before = needCheck.size();

            for(auto op: needCheck)
            {
                for(mlir::Operation* user: llvm::make_early_inc_range(op->getUsers()))
                {
                    user->dump();
                    // If inputs and outputs should have the same shape
                    // This includes most of `arith` operations. Instead of handling each operation
                    // explicitly, we handle them based on whether they have the specific trait
                    if(user->hasTrait<mlir::OpTrait::SameOperandsAndResultType>())
                    {
                        llvm::errs() << "Has SameOperandsAmdResultTypeType trait\n";
                        bool operandsSameShape = checkOperandsSameType(user);
                        // If input operands do not have the same type/shape
                        if(!operandsSameShape)
                        {
                            auto shaped0 = mlir::dyn_cast<ShapedType>(user->getOperandTypes()[0]);
                            auto shaped1 = mlir::dyn_cast<ShapedType>(user->getOperandTypes()[1]);
                            // If LHS is shaped and RHS is scalar, simply "splat"/expand the value to 
                            // a tensor of same shape as the LHS
                            if(shaped0 && !shaped1)
                            {
                                builder.setInsertionPoint(user);
                                auto splatOp = builder.create<tensor::SplatOp>(user->getLoc(), shaped0, user->getOperands()[1]);
                                user->getOpOperands()[1].set(splatOp);
                            }
                            // Same as above with LHS being scalar and RHS being shaped
                            else if(!shaped0 && shaped1 )
                            {
                                builder.setInsertionPoint(user);
                                auto splatOp = builder.create<tensor::SplatOp>(user->getLoc(), shaped1, user->getOperands()[0]);
                                user->getOpOperands()[0].set(splatOp);
                            }
                            // Both are shaped, but the shapes do not match.
                            // We need to infer how to reshape them based on their indexing
                            else
                            {
                                assert(false && "Different shapes are note currently handled");
                                //[TODO] Both shaped, by different shapes/ element types handle it.
                            }

                        }

                        // For operations we have already "blocked" their input operands
                        // their outputs would still have remained scalar.
                        // This can be easily fixed by re-creating the same operation with
                        // the new "blocked" inputs, which will automatically update their output
                        // type
                        if(user->getOperandTypes()[0] != user->getResultTypes()[0])
                        {
                            builder.setInsertionPoint(user);
                            auto blockedOp = builder.create(user->getLoc(), user->getName().getIdentifier(), user->getOperands(), TypeRange(user->getOperandTypes()[0]));
                            user->replaceAllUsesWith(blockedOp);
                            newOpsToCheck.push_back(blockedOp);
                            user->erase();
                        }
                    }
                }
            }

            for(auto op: newOpsToCheck)
            {
                needCheck.insert(op);
            }
            newOpsToCheck.clear();
        }while(needCheck.size() != size_before);

        funcOp.dump();

        for(auto insert: inserts)
        {
            mlir::SmallVector<Value, 2> blockSizes;
            mlir::SmallVector<int64_t, 2> blockSizesLiteral;

            mlir::SmallVector<int64_t, 2> static_offsets;
            mlir::SmallVector<mlir::Value, 2> offsets;
            mlir::SmallVector<int64_t, 2> stridesLiteral;
            mlir::SmallVector<Value, 2> strides;
            std::sort(insert.second.begin(), insert.second.end(), [](BlockInfo& a, BlockInfo& b) {return a.argIndex < b.argIndex;} );
            bufferization::ToTensorOp toTensorOp = mlir::cast<bufferization::ToTensorOp>(insert.first.getDest().getDefiningOp());
            builder.setInsertionPoint(insert.first);
            auto metaData = builder.create<memref::ExtractStridedMetadataOp>(insert.first->getLoc(), toTensorOp.getMemref());

            // UnrealizedConversionCastOp unrealized_cast = mlir::cast<UnrealizedConversionCastOp>(toTensorOp.getMemref().getDefiningOp());
            int64_t rank = insert.first.getResult().getType().getRank();
            for(int64_t i = 0; i < rank; i++)
            {
                blockSizes.push_back(insert.second[i].max);
                blockSizesLiteral.push_back(ShapedType::kDynamic);
                offsets.push_back(insert.second[i].index);
                stridesLiteral.push_back(ShapedType::kDynamic);
                static_offsets.push_back(ShapedType::kDynamic);
                Value stride = metaData.getResult(2 + rank + i); // tt.ptr + offset + (size) * rank + (stride) * rank 
                strides.push_back(stride);
                // auto affineApply = mlir::cast<affine::AffineMinOp>(insert.second[i].max.getDefiningOp());
                // if(i < rank-1)
                // {
                //     strides.push_back(affineApply.getDimOperands()[0]);
                // }
            }
            auto type = mlir::RankedTensorType::get(blockSizesLiteral, insert.first.getResult().getType().getElementType());
            builder.setInsertionPoint(insert.first);
            // strides.push_back(builder.create<arith::ConstantIndexOp>(insert.first->getLoc(), 1));
            auto castSliceShape = builder.create<mlir::tensor::CastOp>(insert.first->getLoc(), type, insert.first.getScalar());

            auto insertSlice = builder.create<mlir::tensor::InsertSliceOp>(insert.first->getLoc(), castSliceShape, insert.first.getDest(), mlir::ValueRange(offsets), mlir::ValueRange(blockSizes), mlir::ValueRange(strides), static_offsets, mlir::ArrayRef(blockSizesLiteral), mlir::ArrayRef(stridesLiteral));
            insert.first->erase();
            // insert.first.replaceAllUsesWith(insertSlice.getResult());
        }

        for(scf::ForOp forOp: forOps)
        {
            builder.setInsertionPoint(forOp);
            auto c0 = builder.create<mlir::index::ConstantOp>(forOp->getLoc(), 0);
            forOp.getInductionVar().replaceAllUsesWith(c0);
            auto yieldOp = *forOp.getOps<scf::YieldOp>().begin();
            yieldOp->erase();
            forOp->getBlock()->getOperations().splice(forOp->getIterator(), forOp.getBody()->getOperations());
            forOp->erase();
        }


        PassManager pm(funcOp.getContext());


        pm.addPass(mlir::createLowerAffinePass());
        if (failed(pm.run(funcOp))) {
            signalPassFailure();
            return;
        }
    }
};

std::unique_ptr<mlir::OperationPass<mlir::gpu::GPUFuncOp>> mlir::comet::createConvertGpuToBlockedGpuPass() {
    return std::make_unique<ConvertGpuToBlockedGpu>();
}