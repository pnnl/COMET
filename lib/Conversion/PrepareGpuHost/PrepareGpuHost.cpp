#include "comet/Conversion/PrepareGpuHost/PrepareGpuHostPass.h"
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

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    OpBuilder builder(modOp);

    std::map<std::string, Value> funcs;
    std::map<std::string, std::string> gpu_to_triton_kernel;
    std::map<std::string, triton::FuncOp> triton_name_to_triton_func_op;
    auto gpuModules = modOp.getOps<gpu::GPUModuleOp>();
    for(auto gpuModuleOp: gpuModules)
    {
        auto funcOps = gpuModuleOp.getOps<mlir::func::FuncOp>();
        for(func::FuncOp funcOp: llvm::make_early_inc_range(funcOps))
        {
            if(!funcOp->hasAttr(gpu::GPUDialect::getKernelFuncAttrName()))
            {
                continue;
            }
            builder.setInsertionPoint(funcOp);
            SmallVector<Type, 4> newTypes;
            for(auto argType: funcOp.getArgumentTypes())
            {
                newTypes.push_back(argType);
                if(MemRefType rankedType = dyn_cast<mlir::MemRefType>(argType))
                {
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
        MemRefType allocType = mlir::cast<MemRefType>(alloc.getType());
        mlir::gpu::AllocOp gpuAlloc;
        if (auto defOp = alloc.getDefiningOp()) {
            builder.setInsertionPointAfter(defOp);
        } else if (mlir::isa<BlockArgument>(alloc)) {
            builder.setInsertionPointToStart(alloc.getParentBlock());
        } else {
            assert(false &&
                    "Value has not defining Op and is not a block argument.");
        }

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
                builder.create<mlir::gpu::MemcpyOp>(launchOp->getLoc(), TypeRange(),
                                                    ValueRange(),
                                                    gpuAlloc.getMemref(), alloc);
                use.set(gpuAlloc.getMemref());
                builder.setInsertionPointAfter(launchOp);
                builder.create<mlir::gpu::MemcpyOp>(launchOp->getLoc(), TypeRange(),
                                                    ValueRange(), alloc,
                                                    gpuAlloc.getMemref());
            }
        }
    }

    std::vector<mlir::gpu::AllocOp> gpuAllocs;
    modOp->walk([&gpuAllocs](mlir::gpu::AllocOp gpuAllocOp) {
        gpuAllocs.push_back(gpuAllocOp);
    });

    std::vector<mlir::gpu::MemcpyOp> gpuCopies;
    modOp->walk([&gpuCopies](mlir::gpu::MemcpyOp gpuCopy) {
        gpuCopies.push_back(gpuCopy);
    });

  }
};


std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createPrepareGpuHostPass() {
  // std::cout << "Running createPrepareGpuHostPass\n";

  return std::make_unique<::PrepareGpuHost>();
}