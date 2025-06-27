#include "comet/Conversion/PrepareGpuHost/PrepareGpuHostPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "comet/Conversion/PrepareGpuHost/Passes.h"

class PrepareGpuHost
    : public mlir::comet::PrepareGpuHostBase<PrepareGpuHost> {
public:
  PrepareGpuHost() = default;

  PrepareGpuHost(bool generateAllocsAndTransfers) {
    this->generateAllocsAndTransfers = generateAllocsAndTransfers;
  }

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    OpBuilder builder(modOp);

    std::map<std::string, Value> funcs;
    std::map<std::string, std::string> gpu_to_triton_kernel;
    std::map<std::string, func::FuncOp> gpu_name_to_funcOp;
    std::map<std::string, triton::FuncOp> triton_name_to_triton_func_op;
    auto gpuModules = modOp.getOps<gpu::GPUModuleOp>();
    for(auto gpuModuleOp: gpuModules)
    {
        auto funcOps = gpuModuleOp.getOps<mlir::func::FuncOp>();
        for(func::FuncOp funcOp: llvm::make_early_inc_range(funcOps))
        {
            std::map<size_t, std::vector<Attribute>> argsToSet;
            if(!funcOp->hasAttr(gpu::GPUDialect::getKernelFuncAttrName()))
            {
                continue;
            }
            builder.setInsertionPoint(funcOp);
            SmallVector<Type, 4> newTypes;
            for(auto arg: funcOp.getArguments())
            {
                auto argType = arg.getType();
                newTypes.push_back(argType);
                if(MemRefType rankedType = dyn_cast<mlir::MemRefType>(argType))
                {
                    argsToSet[newTypes.size() - 1] = {funcOp.getArgAttr(arg.getArgNumber(), "gpu.read"), funcOp.getArgAttr(arg.getArgNumber(), "gpu.write")};
                    if(rankedType.hasRank())
                    {
                        newTypes.push_back(builder.getIndexType());
                        for(int64_t i = 0; i < rankedType.getRank(); i++)
                        {
                            newTypes.push_back(builder.getIndexType());
                        }
                        for(int64_t i = 0; i < rankedType.getRank(); i++)
                        {
                            newTypes.push_back(builder.getIndexType());
                        }   
                    }
                    else
                    {
                        llvm::errs() << "ERROR! Unranked memrefs are not supported currently\n";
                        return signalPassFailure();
                    }
                }
            }

            auto funcType = builder.getFunctionType({newTypes}, funcOp->getResultTypes());
            auto newFunc = builder.create<func::FuncOp>(funcOp->getLoc(), funcOp.getName(), funcType);
            auto entryBlock = newFunc.addEntryBlock();
            builder.setInsertionPointToEnd(entryBlock);
            builder.create<func::ReturnOp>(funcOp.getLoc());
            newFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                                builder.getUnitAttr());
            for(auto [argNumber, attrs]: argsToSet)
            {
                if(attrs[0])
                {
                    newFunc.setArgAttr(argNumber, "gpu.read", attrs[0]);
                }
                if(attrs[1])
                {
                    newFunc.setArgAttr(argNumber, "gpu.write", attrs[1]);
                }
            }
            gpu_name_to_funcOp[funcOp.getName().str()] = newFunc;
            funcOp->erase();
        }
    }

    modOp->walk([&gpu_to_triton_kernel, &triton_name_to_triton_func_op](mlir::triton::FuncOp TTFuncOp) {
        gpu::GPUModuleOp gpuModuleOp = TTFuncOp->getParentOfType<gpu::GPUModuleOp>();
        gpu_to_triton_kernel[gpuModuleOp.getName().str() +"::"+ TTFuncOp.getName().substr(3 + gpuModuleOp.getName().size()).str()] = TTFuncOp.getName().str();
        triton_name_to_triton_func_op[TTFuncOp.getName().str()] = TTFuncOp;
    });

    modOp.walk([&](gpu::LaunchFuncOp launchOp){
        builder.setInsertionPoint(launchOp);
        SmallVector<mlir::Value, 8> newValues;
        for(auto& operand: llvm::make_early_inc_range(launchOp.getKernelOperandsMutable()))
        {
            newValues.push_back(operand.get());
            if(MemRefType memref = mlir::dyn_cast<MemRefType>(operand.get().getType()))
            {
                auto metadata = builder.create<memref::ExtractStridedMetadataOp>(launchOp->getLoc(), operand.get());
                newValues.insert(newValues.end(), metadata.getResults().begin()+1, metadata.getResults().end()); // Skip the memref
            }
        }
        builder.create<gpu::LaunchFuncOp>(launchOp->getLoc(), launchOp.getKernel(), launchOp.getGridSizeOperandValues(), launchOp.getBlockSizeOperandValues(), launchOp.getDynamicSharedMemorySize(), newValues);
        launchOp->erase();
    });

    std::vector<mlir::gpu::LaunchFuncOp> launchOps;

    modOp->walk([&launchOps](mlir::gpu::LaunchFuncOp launchOp) {
        launchOps.push_back(launchOp);
    });

    std::set<mlir::func::FuncOp> initFuncs;
    std::vector<Value> toAlloc;

    for (auto launchOp : launchOps) {
        auto name =
            gpu_to_triton_kernel[(launchOp.getKernelModuleName().strref() +
                                "::" + launchOp.getKernelName().strref())
                                    .str()];
        builder.setInsertionPoint(launchOp);
        auto ttFunc = triton_name_to_triton_func_op[name];
        Value sharedMem = builder.create<arith::ConstantIntOp>(
            launchOp->getLoc(), ttFunc->getAttrOfType<IntegerAttr>("triton_gpu.shared").getInt(), 32);
        launchOp.getDynamicSharedMemorySizeMutable().assign(sharedMem);

        Value numWarps = builder.create<arith::ConstantIndexOp>(
            launchOp->getLoc(),
            ttFunc->getAttrOfType<IntegerAttr>("triton_gpu.num-warps").getInt());
        Value threadsPerWarp = builder.create<arith::ConstantIndexOp>(
            launchOp->getLoc(),
            ttFunc->getAttrOfType<IntegerAttr>("triton_gpu.threads-per-warp")
                .getInt());
        Value numThreads = builder.create<arith::MulIOp>(launchOp->getLoc(), numWarps, threadsPerWarp);
        Value one = builder.create<arith::ConstantIndexOp>(launchOp->getLoc(), 1);
        launchOp.getBlockSizeXMutable().assign(numThreads);
        launchOp.getBlockSizeYMutable().assign(one);
        launchOp.getBlockSizeZMutable().assign(one);

        for (auto &operand : launchOp->getOpOperands()) {
            if (isa<MemRefType>(operand.get().getType())) {
                toAlloc.push_back(operand.get());
            }
        }
    }

    auto cmp = [](Value a, Value b) {
        return a.getAsOpaquePointer() < b.getAsOpaquePointer();
    };
    std::set<Value, bool (*)(Value, Value)> uniqueGpuAllocs(toAlloc.begin(),
                                                            toAlloc.end(), cmp);
    for (Value alloc : uniqueGpuAllocs) {
        bool isInDevice = false;
        MemRefType allocType = mlir::cast<MemRefType>(alloc.getType());
        mlir::gpu::AllocOp gpuAlloc;
        if (auto defOp = alloc.getDefiningOp()) {
            builder.setInsertionPointAfter(defOp);
        } else if (mlir::isa<BlockArgument>(alloc)) {
            if(mlir::cast<mlir::func::FuncOp>(mlir::cast<BlockArgument>(alloc).getOwner()->getParentOp()).getArgAttr(mlir::cast<BlockArgument>(alloc).getArgNumber(), "gpu.indevice"))
            {
                isInDevice = true;
            }
            builder.setInsertionPointToStart(alloc.getParentBlock());
        } else {
            assert(false &&
                    "Value has not defining Op and is not a block argument.");
        }

        if(!isInDevice)
        {
            if (allocType.hasStaticShape()) {
                gpuAlloc = builder.create<mlir::gpu::AllocOp>(
                    alloc.getLoc(), allocType, ValueRange(), ValueRange(),
                    ValueRange());
            } else {
                std::vector<Value> dynDims;
                for (size_t i = 0; i < allocType.getShape().size(); i++) {
                    if (allocType.isDynamicDim(i)) {
                    dynDims.push_back(
                        builder.create<memref::DimOp>(alloc.getLoc(), alloc, i));
                    }
                }
                gpuAlloc = builder.create<mlir::gpu::AllocOp>(
                    alloc.getLoc(), allocType, ValueRange(), dynDims, ValueRange());
            }

            
            auto op = alloc.getDefiningOp() != NULL
                        ? alloc.getDefiningOp()->getResult(0)
                        : alloc;
            for (auto &use : llvm::make_early_inc_range(op.getUses())) {
                if (mlir::gpu::LaunchFuncOp launchOp =
                        dyn_cast<mlir::gpu::LaunchFuncOp>(use.getOwner())) {
                    builder.setInsertionPoint(launchOp);
                    int offset = launchOp->getNumOperands() - launchOp.getNumKernelOperands();
                    int operNum = use.getOperandNumber() - offset;
                    if(gpu_name_to_funcOp[launchOp.getKernelName().str()].getArgAttr(operNum, "gpu.read"))
                    {
                        auto gpuMemCpy = builder.create<mlir::gpu::MemcpyOp>(launchOp->getLoc(), TypeRange(),
                                                            ValueRange(),
                                                            gpuAlloc.getMemref(), alloc);
                        gpuMemCpy->setAttr("gpu.read", builder.getUnitAttr());
                    }
                    use.set(gpuAlloc.getMemref());
                    builder.setInsertionPointAfter(launchOp);
                    if(gpu_name_to_funcOp[launchOp.getKernelName().str()].getArgAttr(operNum, "gpu.write"))
                    {
                        auto gpuMemCpy = builder.create<mlir::gpu::MemcpyOp>(launchOp->getLoc(), TypeRange(),
                                                            ValueRange(), alloc,
                                                            gpuAlloc.getMemref());
                        gpuMemCpy->setAttr("gpu.write", builder.getUnitAttr());
                    }
                }
            }
        }
        else
        {
            for (auto &use : llvm::make_early_inc_range(alloc.getUses())) {
                if (mlir::gpu::LaunchFuncOp launchOp =
                        dyn_cast<mlir::gpu::LaunchFuncOp>(use.getOwner())) {
                    builder.setInsertionPoint(launchOp);
                    auto ptr = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(launchOp->getLoc(), alloc);
                    use.set(ptr);
                }
            }
        }
    }

    std::vector<mlir::gpu::AllocOp> gpuAllocs;
    modOp->walk([&gpuAllocs](mlir::gpu::AllocOp gpuAllocOp) {
        gpuAllocs.push_back(gpuAllocOp);
    });


    std::map<void*, std::vector<Operation*>> memEffects;
    modOp->walk([&uniqueGpuAllocs, &memEffects](Operation* memEffect) {
        for(auto op: memEffect->getOperands())
        {
            if(uniqueGpuAllocs.find(op) != uniqueGpuAllocs.end())
            {
                memEffects[op.getAsOpaquePointer()].push_back(memEffect);
            }
        }
    });

    for(auto& [memref, effects]: memEffects)
    {
        bool copyIn = true;
        std::vector<Operation*> copyDelete;
        for(size_t i = 0; i < effects.size(); i++)
        {
            if(mlir::gpu::MemcpyOp gpuCopy = mlir::dyn_cast<mlir::gpu::MemcpyOp>(effects[i]))
            {
                if(!copyIn & gpuCopy->hasAttr("gpu.read"))
                {
                    // gpuCopy.dump();
                    gpuCopy->erase();
                }
                else if(gpuCopy->hasAttr("gpu.read"))
                {
                    copyIn = false;
                }
                else if(gpuCopy->hasAttr("gpu.write"))
                {
                    copyIn = false;
                    copyDelete.push_back(gpuCopy);
                }
            }
            else if(mlir::memref::StoreOp store = mlir::dyn_cast<mlir::memref::StoreOp>(effects[i]))
            {
                copyIn = true;
                if(!copyDelete.empty())
                {
                    copyDelete.pop_back();
                }
            }
            else if(mlir::memref::CopyOp copy = mlir::dyn_cast<mlir::memref::CopyOp>(effects[i]))
            {
                if(copy.getTarget().getAsOpaquePointer() == memref)
                {
                    copyIn = true;
                }
            }
            else if(mlir::memref::LoadOp load = mlir::dyn_cast<mlir::memref::LoadOp>(effects[i]))
            {
                if(!copyDelete.empty())
                {
                    copyDelete.pop_back();
                }
            }
            else if(isa<memref::ExtractStridedMetadataOp, memref::DimOp, memref::RankOp>(effects[i]))
            {
                continue;
            }
            else // Unknown operation, be conservative
            {
                effects[i]->dump();
                copyIn = true;
                if(!copyDelete.empty())
                {
                    copyDelete.pop_back();
                }  
            }
        }

        for(auto toDelete: copyDelete)
        {
            // toDelete->dump();
            toDelete->erase();
        }
    }

  }
};


std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createPrepareGpuHostPass() {
  // std::cout << "Running createPrepareGpuHostPass\n";

  return std::make_unique<::PrepareGpuHost>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createPrepareGpuHostPass(bool generateAllocsAndTransfers) {
  // std::cout << "Running createPrepareGpuHostPass\n";

  return std::make_unique<::PrepareGpuHost>(generateAllocsAndTransfers);
}