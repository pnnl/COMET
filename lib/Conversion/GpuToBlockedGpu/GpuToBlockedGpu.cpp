#include "comet/Conversion/GpuToBlockedGpu/GpuToBlockedGpu.h"
#include "comet/Conversion/ParallelLoopsToGpu/ParallelLoopsToGpu.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <utility>
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
                        // If someone accesses the same memref (load) within the same block, replace its use with the output of this operation
                        storeOp.getMemRef().replaceUsesWithIf(insertOp.getResult(), [&](OpOperand& op) {
                            
                            return op.getOwner()->getBlock() == storeOp->getBlock() && insertOp->isBeforeInBlock(op.getOwner());
                        });
                        storeOp->erase();
                        // StoreOp has no uses
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
        auto comp = [](Value a, Value b) {return a.getAsOpaquePointer() < b.getAsOpaquePointer();};
        std::set<mlir::Value, decltype(comp)> needCheck(comp);

        funcOp->walk([&](mlir::scf::ForOp forOp) {
            // forOps with "blockSize"   attribute induction variables need to be expanded to blocked indices
            // i.e., for i : 0 -> M must be converted to a vector [0, 1, ..., M-1] that represents the indices taken by i
            if(forOp->hasAttr("blockSize"))
            {
                forOps.push_back(forOp);
                uint64_t blockSize = forOp->getAttrOfType<mlir::IntegerAttr>("blockSize").getUInt();
                builder.setInsertionPointToStart(forOp.getBody());
                
                // Since there is no built-in op that creates a range, we use unrealized casts to represent the expansion of an IV to a vector
                auto blockedIndex = builder.create<UnrealizedConversionCastOp>(forOp.getInductionVar().getLoc(), TypeRange({RankedTensorType::get({static_cast<int64_t>(blockSize)}, builder.getIndexType())}), forOp.getInductionVar());
                forOp.getInductionVar().replaceAllUsesExcept(blockedIndex.getResult(0), blockedIndex);
                
                // We have replaced indices with blocked indices, we need to make sure their users 
                // will be converted to blocks as well
                for(auto res: blockedIndex->getResults())
                {
                    needCheck.insert(res);    
                }
                llvm::SmallVector<mlir::Value, 4> inductionVars;
                inductionVars.push_back(blockedIndex->getResult(0));
                
                // If blocked indices (or their users) are used in a tensor insert/extract op, we cast them to a singl index
                // so that the respective operation remains valid
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

        /// TODO: Indices need to be the initial "block Indices" i.e. what will later be blockId, reduction Id or any other loop induction variable
        /// Since we only care about the "direction" of iteration, not the actual index
        llvm::MapVector<mlir::Value, llvm::SmallVector<Value, 2>> useToIndices;
        
        // Convert extractOp to extractSliceOp as it is a closer representation of triton's blocked memory loads
        // We also keep the information regarding the boolean masks that need to be generated in order to avoid
        // reading beyond memory boundaries
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

            // Get information regarding the size and stride of the memref
            auto metaData = builder.create<memref::ExtractStridedMetadataOp>(extract.first->getLoc(), toTensorOp.getMemref());
            for(int64_t i = 0; i < rank; i++)
            {
                blockSizes.push_back(extract.second[i].max);
                blockSizesLiteral.push_back(ShapedType::kDynamic);
                offsets.push_back(extract.second[i].index);
                stridesLiteral.push_back(ShapedType::kDynamic);
                static_offsets.push_back(ShapedType::kDynamic);
                Value stride = metaData->getResult(2 + rank + i); // tt.ptr + offset + (size) * rank + (stride) * rank 
                strides.push_back(stride);
            }


            auto type = mlir::RankedTensorType::get(blockSizesLiteral, extract.first.getTensor().getType().getElementType());
            auto extractSlice = builder.create<mlir::tensor::ExtractSliceOp>(extract.first->getLoc(), type, extract.first.getTensor(), mlir::ValueRange(offsets), mlir::ValueRange(blockSizes), mlir::ValueRange(strides), static_offsets, mlir::ArrayRef(blockSizesLiteral), mlir::ArrayRef(stridesLiteral));
            SmallVector<int64_t, 2> static_shape;
            for(auto block: extract.second)
            {
                static_shape.push_back(block.blockSize);
            }
            auto castSliceShape = builder.create<mlir::tensor::CastOp>(extract.first->getLoc(), mlir::RankedTensorType::get(static_shape, extract.first.getTensor().getType().getElementType()), extractSlice);
            extract.first.replaceAllUsesWith(castSliceShape.getResult());
            extract.first->erase();
            auto [already_there, inserted] = needCheck.insert(castSliceShape);
            if(inserted)
            {
                SmallVector<Value, 2> indices;
                for(auto offset: offsets)
                {
                    if(auto index = dyn_cast_if_present<UnrealizedConversionCastOp>(offset.getDefiningOp()))
                    {
                        indices.push_back(index->getOperand(0));
                    }
                    else
                    {
                        indices.push_back(offset);
                    }
                }
                assert(!indices.empty());
                bool isDone = useToIndices.insert(std::make_pair(castSliceShape,indices)).second;
                assert(isDone);
                // llvm::errs() << "FOUND\n";
                // castSliceShape->dump();
                // llvm::errs() << castSliceShape->getResult(0).getAsOpaquePointer() << "\n";

                // llvm::errs() << "======= START ===========\n";
                // for(auto index: useToIndices[castSliceShape->getResult(0)])
                // {
                //     llvm::errs() << index.getAsOpaquePointer() << "\n";
                // }
                // llvm::errs() << "======= END ===========\n";
            }
            else
            {
                assert(false && "Already there");
                llvm::errs() << "Already there : \n";
                already_there->dump();
                llvm::errs() << "Trying to insert : \n";
                castSliceShape->dump();
            }
        }

        size_t size_before;
        SmallVector<Value, 4> newOpsToCheck;
        do 
        {
            size_before = needCheck.size();

            for(auto op: llvm::make_early_inc_range(needCheck))
            {
                for(mlir::Operation* user: llvm::make_early_inc_range(op.getUsers()))
                {
                    bool isNotPartOfReduction = llvm::any_of(user->getOperands(), [](Value val) {
                        if(BlockArgument block_arg = mlir::dyn_cast<BlockArgument>(val))
                        {
                            // block_arg.dump();
                            if(auto owner_for =  mlir::dyn_cast<scf::ForOp>(block_arg.getOwner()->getParentOp()))
                            {
                                if(std::find(owner_for.getRegionIterArgs().begin(), owner_for.getRegionIterArgs().end(), block_arg) != owner_for.getRegionIterArgs().end())
                                {
                                    return true;
                                }
                            }
                        }
                        else if(val.getDefiningOp()->hasAttr("notPartOfReduction"))
                        {
                            return true;
                        }
                        return false;
                    });
                    if(isNotPartOfReduction && !isa<scf::ForOp>(user))
                    {
                        user->setAttr("notPartOfReduction", builder.getUnitAttr());
                    }
                    if(isNotPartOfReduction)
                    {
                        // user->dump();
                    }

                    // If inputs and outputs should have the same shape
                    // This includes most of `arith` operations. Instead of handling each operation
                    // explicitly, we handle them based on whether they have the specific trait
                    if(user->hasTrait<mlir::OpTrait::SameOperandsAndResultType>())
                    {
                        // llvm::errs() << "Has SameOperandsAmdResultTypeType trait\n";
                        bool operandsSameShape = checkOperandsSameType(user);
                        // If input operands do not have the same type/shape
                        if(!operandsSameShape)
                        {
                            auto shaped0 = mlir::dyn_cast<ShapedType>(user->getOperandTypes()[0]);
                            auto shaped1 = mlir::dyn_cast<ShapedType>(user->getOperandTypes()[1]);
                            // If LHS is shaped and RHS is scalar, simply "splat"/expand the value to 
                            // a tensor of same shape as the LHS
                            if(shaped0 && !shaped1 && ! isNotPartOfReduction)
                            {
                                builder.setInsertionPoint(user);
                                auto splatOp = builder.create<tensor::SplatOp>(user->getLoc(), shaped0, user->getOperands()[1]);
                                user->getOpOperands()[1].set(splatOp);
                            }
                            // Same as above with LHS being scalar and RHS being shaped
                            else if(!shaped0 && shaped1 && !isNotPartOfReduction)
                            {
                                builder.setInsertionPoint(user);
                                auto splatOp = builder.create<tensor::SplatOp>(user->getLoc(), shaped1, user->getOperands()[0]);
                                user->getOpOperands()[0].set(splatOp);
                            }
                            // Both are shaped, but the shapes do not match.
                            // We need to infer how to reshape them based on their indexing
                            else if (!isNotPartOfReduction)
                            {
                                assert(shaped0.getElementType() == shaped1.getElementType() && "Inputs need to have same element types");

                                // Assuming first operand is the one to potentially broadcast (input)
                                OpOperand& input = user->getOpOperand(0);
                                OpOperand& output = user->getOpOperand(1);
                                ShapedType initShape = shaped1;
                                
                                // If not, swap the input, output
                                if(shaped0.getRank() > shaped1.getRank())
                                {
                                    std::swap(input, output);
                                    initShape = shaped0;
                                }

                                /// TODO: Shapes same rank but different possibly mean matrix multiplication
                                assert(shaped0.getRank() != shaped1.getRank() && "Shapes have same rank but are different");


                                // Collect the induction variable indices used to form these values from load operations
                                // Note that there should be no case where the input or output do not have indices
                                // as those would be handled by the previous conditions of the if-else blocks
                                Value v0 = output.get(), v1 = input.get();
                                assert(useToIndices.find(v0)!= useToIndices.end());
                                assert(useToIndices.find(v1)!= useToIndices.end());
                                SmallVector<Value, 2>& indices0 = useToIndices[v0];
                                SmallVector<Value, 2>& indices1 = useToIndices[v1];
                                SmallVector<int64_t, 2> dimensions;

                                // Find the dimension(s) "missing" from the lower rank tensor
                                /// TODO: Handle the case where dimensions are missing from both tensors, e.g., adding a row and a column vector
                                for(size_t i = 0; i < indices0.size(); i++)
                                {
                                    if(std::find(indices1.begin(), indices1.end(), indices0[i]) == indices1.end())
                                    {
                                        dimensions.push_back(i);
                                    }
                                }

                                // Broadcast lower rank tensor to match the higher-rank one.
                                builder.setInsertionPoint(user);
                                Value init = builder.create<tensor::EmptyOp>(user->getLoc(), initShape.getShape(), initShape.getElementType());

                                Operation* bcastOp = builder.create<mlir::linalg::BroadcastOp>(user->getLoc(), input.get(), init, dimensions);
                                assert(bcastOp->getResults().size() == 1);
                                input.set(bcastOp->getResults().front());
                                
                                // newOpsToCheck.push_back(bcastOp->getResult(0));
                                assert(useToIndices.find(v0)!=useToIndices.end());
                                assert(!useToIndices[v0].empty());
                                bool isDone = useToIndices.insert(std::make_pair(bcastOp->getResult(0), useToIndices[v0])).second;
                                assert(isDone);
                                // bcastOp->dump();
                                // funcOp->dump();
                                // llvm::errs() << "FOUND\n";
                                // bcastOp->dump();
                                // v0.dump();
                                // for(auto index: useToIndices[v0])
                                // {
                                //     llvm::errs() << index.getAsOpaquePointer() << "\n";

                                // }
                                // llvm::errs() << "Output ptr: " << v0.getAsOpaquePointer() << "\n";
                                // funcOp->dump();
                                // llvm::errs() << bcastOp->getResult(0).getAsOpaquePointer() << "\n";

                                // llvm::errs() << "======= START ===========\n";
                                // for(auto index: useToIndices[bcastOp->getResult(0)])
                                // {
                                //     llvm::errs() << index.getAsOpaquePointer() << "\n";
                                // }
                                // llvm::errs() << "======= END ===========\n";
                            }

                        }

                        // For operations we have already "blocked" their input operands
                        // their outputs would still have remained scalar.
                        // This can be easily fixed by re-creating the same operation with
                        // the new "blocked" inputs, which will automatically update their output
                        // type
                        if(user->getOperandTypes()[0] != user->getResultTypes()[0] && ! isNotPartOfReduction)
                        {
                            builder.setInsertionPoint(user);
                            auto blockedOp = builder.create(user->getLoc(), user->getName().getIdentifier(), user->getOperands(), TypeRange(user->getOperandTypes()[0]));
                            user->replaceAllUsesWith(blockedOp);
                            newOpsToCheck.insert(newOpsToCheck.end(), blockedOp->getResults().begin(), blockedOp->getResults().end());
                            if(useToIndices.find(user->getOperands()[1])!=useToIndices.end())
                            {
                                bool isDone = useToIndices.insert(std::make_pair(blockedOp->getResult(0), useToIndices[user->getOperands()[1]])).second;
                                assert(isDone);
                            }
                            else if(useToIndices.find(user->getOperands()[0])!=useToIndices.end())
                            {
                                bool isDone = useToIndices.insert(std::make_pair(blockedOp->getResult(0), useToIndices[user->getOperands()[0]])).second;
                                assert(isDone);
                            }

                            // llvm::errs() << "FOUND\n";
                            // blockedOp->dump();
                            // llvm::errs() << blockedOp->getResult(0).getAsOpaquePointer() << "\n";
                            // llvm::errs() << "Operand 1\n";
                            // llvm::errs() << user->getOperands()[1].getAsOpaquePointer() << "\n";
                            // llvm::errs() << "======= START ===========\n";
                            // for(auto index: useToIndices[blockedOp->getResult(0)])
                            // {
                            //     llvm::errs() << index.getAsOpaquePointer() << "\n";
                            // }
                            // llvm::errs() << "======= END ===========\n";
                            user->erase();
                        }
                    }
                    else if(scf::ForOp forOp = mlir::dyn_cast<scf::ForOp>(user))
                    {
                        // If one of the changed operations has a use in a ForOp operation's initArgs
                        // we need to change the forOp accordingly and update its inner uses as well as its results
                        bool needsFixes = false;
                        for(size_t i = 0; i < forOp.getInitArgs().size(); i++)
                        {
                            if(forOp.getInitArgs()[i].getType() != forOp.getRegionIterArgs()[i].getType())
                            {
                                needsFixes = true;
                            }
                        }
                        if(!needsFixes)
                        {
                            continue;
                        }
                        // op.dump();
                        builder.setInsertionPoint(user);
                        auto newForOp = builder.create<scf::ForOp>(user->getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), forOp.getInitArgs());
                        if(forOp->hasAttr("blockSize"))
                        {
                            newForOp->setAttr("blockSize", forOp->getAttr("blockSize"));
                        }
                        else if(forOp->hasAttr("reduceDim"))
                        {
                            newForOp->setAttr("reduceDim", forOp->getAttr("reduceDim"));

                        }
                        builder.setInsertionPointToEnd(newForOp.getBody());
                        builder.create<scf::YieldOp>(forOp->getLoc(), forOp.getBody()->getTerminator()->getOperands());
                        forOp.getBody()->getTerminator()->erase();
                        for(auto& inner_op: llvm::make_early_inc_range(forOp.getBody()->getOperations()))
                        {
                            inner_op.moveBefore(newForOp.getBody()->getTerminator());
                        }
                        // newForOp.getBody()->getTerminator()->erase();
                        for(auto inner_arg: zip(forOp.getBody()->getArguments(), newForOp.getBody()->getArguments()))
                        {
                            std::get<0>(inner_arg).replaceAllUsesWith(std::get<1>(inner_arg));
                            newOpsToCheck.push_back(std::get<1>(inner_arg));
                        }
                        for(auto [old_arg, new_arg]: zip(forOp.getInits(), newForOp.getRegionIterArgs()))
                        {
                            assert(useToIndices.find(old_arg) != useToIndices.end());
                            bool isDone = useToIndices.insert(std::make_pair(new_arg, useToIndices[old_arg])).second;
                            assert(isDone);

                        }
                        forOp->replaceAllUsesWith(newForOp);
                        newOpsToCheck.insert(newOpsToCheck.end() , newForOp->getResults().begin(), newForOp->getResults().end());
                        forOp->erase();
                    }
                }
            }

            for(auto op: newOpsToCheck)
            {
                needCheck.insert(op);
            }
            newOpsToCheck.clear();
        }while(needCheck.size() != size_before);

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
            }
            auto type = mlir::RankedTensorType::get(blockSizesLiteral, insert.first.getResult().getType().getElementType());
            builder.setInsertionPoint(insert.first);
            auto castSliceShape = builder.create<mlir::tensor::CastOp>(insert.first->getLoc(), type, insert.first.getScalar());

            auto insertSlice = builder.create<mlir::tensor::InsertSliceOp>(insert.first->getLoc(), castSliceShape, insert.first.getDest(), mlir::ValueRange(offsets), mlir::ValueRange(blockSizes), mlir::ValueRange(strides), static_offsets, mlir::ArrayRef(blockSizesLiteral), mlir::ArrayRef(stridesLiteral));
            insert.first->erase();
            /// TODO: It might make sense to replaceAll uses at some point later
            // insert.first.replaceAllUsesWith(insertSlice.getResult());
        }

        // Since we might have replaced forOps if they were using initArgs
        // we need to collect forOps with "blockSize" again
        forOps.clear();
        funcOp->walk([&](mlir::scf::ForOp forOp) {
            if(forOp->hasAttr("blockSize"))
            {
                forOps.push_back(forOp);
        }});
        
        for(scf::ForOp forOp: llvm::make_early_inc_range(forOps))
        {
            // Since for loops with blockSize attribute are replaced with blocked operations
            // we can safely remove them 
            if(forOp.getInitArgs().empty())
            {
                builder.setInsertionPoint(forOp);
                auto c0 = builder.create<mlir::index::ConstantOp>(forOp->getLoc(), 0);
                forOp.getInductionVar().replaceAllUsesWith(c0);
                auto yieldOp = *forOp.getOps<scf::YieldOp>().begin();
                yieldOp->erase();
                forOp->getBlock()->getOperations().splice(forOp->getIterator(), forOp.getBody()->getOperations());
                forOp->erase();
            }
            else // Reduction loops need special handling 
            {
                SmallVector<Operation*> notPartOfReduction;
                forOp->walk([&notPartOfReduction](Operation* op) {
                    if(op->hasAttr("notPartOfReduction"))
                    {
                        notPartOfReduction.push_back(op);
                    }
                });

                for(auto op: notPartOfReduction)
                {
                    
                    // If one of the, possibly, two operands is not operating as part of the reduction, i.e., has the attribute notPartOfReduction
                    // this operand is either the accumulating target buffer (init) or an operation that is operating on it
                    Value init, toReduced;
                    if(!op->getOperands()[0].getDefiningOp() || op->getOperands()[0].getDefiningOp()->hasAttr("notPartOfReduction") )
                    {
                        init = op->getOperands()[0];
                        toReduced = op->getOperands()[1];
                    }
                    else if(!op->getOperands()[1].getDefiningOp() || op->getOperands()[1].getDefiningOp()->hasAttr("notPartOfReduction"))
                    {
                        init = op->getOperands()[1];
                        toReduced = op->getOperands()[0];
                    }
                    else
                    {
                        /// TODO: Handle cases where both are "notPartOfReduction"
                        assert(false && "TODO: Handle cases where both are notPartOfReduction");
                    }

                    builder.setInsertionPoint(op);
                    Value v0 = op->getOperand(0);
                    Value v1 = op->getOperand(1);


                    // v0.dump();
                    // v1.dump();


                    assert(useToIndices.find(v0) != useToIndices.end());
                    assert(useToIndices.find(v1) != useToIndices.end());
                    
                    // Collect the induction variable indices used to form these values from load operations
                    /// TODO: Handle the case where a scalar is used, i.e., A[i] + 1, 1 will not have indices
                    /// but it is valid to be expanded to any direction 
                    SmallVector<Value, 2>& indices0 = useToIndices[v0];
                    SmallVector<Value, 2>& indices1 = useToIndices[v1];
                    if(indices0.size() < indices1.size())
                    {
                        std::swap(indices0, indices1);
                        std::swap(v0, v1);
                    }

                    // op->getOpOperand(0).get().dump();
                    // op->getOpOperand(1).get().dump();
                    // op->dump();
                    SmallVector<int64_t, 2> dimensions;
                    for(size_t i = 0; i < indices0.size(); i++)
                    {
                        // auto it = std::find(indices1.begin(), indices1.end(), indices0[i]);
                        if(std::find(indices1.begin(), indices1.end(), indices0[i]) == indices1.end())
                        {
                            dimensions.push_back(i);
                        }
                    }
                    /// TODO: Handle cases where the init and reduced tensors are of the same rank (e.g., matrix multiplication, reducing a row vector against a column vector etc)

                    auto reduceOp = builder.create<mlir::linalg::ReduceOp>(op->getLoc(), toReduced, init, dimensions, [&](OpBuilder builder, Location loc, ValueRange values){
                        auto res = builder.create(loc, op->getName().getIdentifier(), values, op->getResultTypes());
                        builder.create<mlir::linalg::YieldOp>(loc, res->getResults());
                    });

                    op->replaceAllUsesWith(reduceOp);
                    op->erase();
                }

                builder.setInsertionPoint(forOp);
                auto c0 = builder.create<mlir::index::ConstantOp>(forOp->getLoc(), 0);
                forOp.getInductionVar().replaceAllUsesWith(c0);
                for(auto [index ,iter_arg]: enumerate(forOp.getRegionIterArgs()))
                {
                    iter_arg.replaceAllUsesWith(forOp.getInitArgs()[index]);
                }
                forOp->replaceAllUsesWith(forOp.getBody()->getTerminator()->getOperands());
                forOp.getBody()->getTerminator()->erase();
                forOp->getBlock()->getOperations().splice(forOp->getIterator(), forOp.getBody()->getOperations());
                forOp->erase();

            }
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