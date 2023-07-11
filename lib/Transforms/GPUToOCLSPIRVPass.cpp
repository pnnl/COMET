#include "comet/Transforms/Passes.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace mlir {
#define GEN_PASS_DEF_GPUTOOCLSPIRV
#include "comet/Transforms/CometTransforms.h.inc"
}

using namespace mlir;

spirv::TargetEnvAttr getKernelEnv(Operation *op) {
    auto triple = spirv::VerCapExtAttr::get(spirv::Version::V_1_0,
                                            {spirv::Capability::Kernel, spirv::Capability::Addresses, spirv::Capability::Int64, spirv::Capability::Float64},
                                            ArrayRef<spirv::Extension>(), op->getContext());
    return spirv::TargetEnvAttr::get(triple, spirv::getDefaultResourceLimits(op->getContext()));
}

namespace {
    struct GPUToOCLSPIRVPass : mlir::impl::GPUToOCLSPIRVBase<GPUToOCLSPIRVPass> {
        void runOnOperation() override {
            MLIRContext *context = &getContext();
            ModuleOp module = getOperation();
            module->setAttr(spirv::getTargetEnvAttrName(), getKernelEnv(module));
            SmallVector<Operation *, 1> gpuModules;
            OpBuilder builder(context);
            module.walk([&, context](gpu::GPUModuleOp gpuModule) {
                // from https://github.com/llvm/llvm-project/blob/main/mlir/test/lib/Dialect/SPIRV/TestEntryPointAbi.cpp
                StringRef attrName = spirv::getEntryPointABIAttrName();
                for (gpu::GPUFuncOp gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
                    if (!gpu::GPUDialect::isKernel(gpuFunc) || gpuFunc->getAttr(attrName))
                        continue;
                    SmallVector<int32_t, 3> workgroupSizeVec(/* TODO: This should not be hardcoded in the future */ {1});
                    workgroupSizeVec.resize(3, 1);
                    gpuFunc->setAttr(attrName,
                                     spirv::getEntryPointABIAttr(context, workgroupSizeVec));
                }
                // end from

            // from https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/GPUToSPIRV/GPUToSPIRVPass.cpp
                // Clone each GPU kernel module for conversion, given that the GPU
                // launch op still needs the original GPU kernel module.
                builder.setInsertionPoint(gpuModule.getOperation());
                gpuModules.push_back(builder.clone(*gpuModule.getOperation()));
            });
            for (Operation *gpuModule : gpuModules) {
                std::unique_ptr<ConversionTarget> memTarget =
                        spirv::getMemorySpaceToStorageClassTarget(*context);
                spirv::MemorySpaceToStorageClassMap memorySpaceMap =
                        spirv::mapMemorySpaceToOpenCLStorageClass;
                spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);
                RewritePatternSet memPatterns(context);
                spirv::populateMemorySpaceToStorageClassPatterns(converter, memPatterns);

                if (failed(applyFullConversion(gpuModule, *memTarget, std::move(memPatterns))))
                    return signalPassFailure();

                auto targetAttr = spirv::lookupTargetEnvOrDefault(gpuModule);
                std::unique_ptr<ConversionTarget> target =
                        SPIRVConversionTarget::get(targetAttr);

                SPIRVConversionOptions options;
                options.use64bitIndex = true;
                SPIRVTypeConverter typeConverter(targetAttr, options);
                typeConverter.addConversion([&](gpu::MMAMatrixType type) -> Type {
                    return convertMMAToSPIRVType(type);
                });
                ScfToSPIRVContext scfContext;
                RewritePatternSet patterns(context);
                populateGPUToSPIRVPatterns(typeConverter, patterns);
                populateGpuWMMAToSPIRVConversionPatterns(typeConverter, patterns);
                mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
                populateMemRefToSPIRVPatterns(typeConverter, patterns);
                populateFuncToSPIRVPatterns(typeConverter, patterns);
                populateSCFToSPIRVPatterns(typeConverter, scfContext, patterns);
                if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
                    return signalPassFailure();
            }
        }
    };
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGPUToOCLSPIRVPass() {
    return std::make_unique<GPUToOCLSPIRVPass>();
}