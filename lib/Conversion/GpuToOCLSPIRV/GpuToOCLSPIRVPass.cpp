#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include <cstdint>
#include <fstream>
#include "comet/Conversion/GpuToOCLSPIRV/GpuToOCLSPIRVPass.h"

#define GEN_PASS_CLASSES
#include "comet/Conversion/GpuToOCLSPIRV/Passes.h.inc"

mlir::spirv::TargetEnvAttr getKernelEnv(mlir::Operation *op) {
  auto triple = mlir::spirv::VerCapExtAttr::get(mlir::spirv::Version::V_1_0,
                                          {mlir::spirv::Capability::Kernel, mlir::spirv::Capability::Addresses, mlir::spirv::Capability::Int64, mlir::spirv::Capability::Float64, },
                                          mlir::ArrayRef<mlir::spirv::Extension>(), op->getContext());
  return mlir::spirv::TargetEnvAttr::get(triple, mlir::spirv::getDefaultResourceLimits(op->getContext()));
}
class ConvertGpuToOCLSPIRV
    : public ConvertGpuKernelToOCLSPIRVPassBase<ConvertGpuToOCLSPIRV> {
public:
  ConvertGpuToOCLSPIRV() = default;
  ConvertGpuToOCLSPIRV(int block_size_x, int block_size_y, int block_size_z) {
        this->blockX = block_size_x;
        this->blockY = block_size_y;
        this->blockZ = block_size_z;
    }
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext
    ();
    mlir::ModuleOp module = getOperation();
    module->setAttr(mlir::spirv::getTargetEnvAttrName(), getKernelEnv(module));
    module->walk([context, this](mlir::gpu::GPUFuncOp gpuFunc) {
    mlir::StringRef attrName = mlir::spirv::getEntryPointABIAttrName();
    if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) || gpuFunc->getAttr(attrName))
    {
      return;
    }
    mlir::SmallVector<int32_t, 3> workgroupSizeVec = {this->blockX , this->blockY , this->blockZ};
    gpuFunc->setAttr(attrName,
      mlir::spirv::getEntryPointABIAttr(context));
    });

    mlir::PassManager pm(context);
    pm.addPass(mlir::createConvertGPUToSPIRVPass(true));
    
    if (failed(pm.run(module))) {
      signalPassFailure();
      return;
    }


    mlir::SmallVector<mlir::spirv::ModuleOp, 3> spirvModules;
    module->walk([&spirvModules](mlir::spirv::ModuleOp module) {
      spirvModules.push_back(module);
    });
    
    for(mlir::spirv::ModuleOp module: spirvModules)
    {
      mlir::PassManager pm(module.getContext());
      pm.addPass(mlir::spirv::createSPIRVLowerABIAttributesPass());
      pm.addPass(mlir::spirv::createSPIRVUpdateVCEPass());
      
      if (failed(pm.run(module))) {
        signalPassFailure();
        return;
      }

      mlir::SmallVector<uint32_t, 0> binary;
      if (!failed(mlir::spirv::serialize(module, binary)))
      {
        std::string pathname("spirv_comet_");
        pathname += module.getName()->data();
        pathname += ".bin";
        std::ofstream binout(pathname, std::ios::binary);
        binout.write(reinterpret_cast<char*>(binary.data()), binary.size() * sizeof(uint32_t));
      }
    }
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::comet::createConvertGPUKernelToOCLSPIRVPass(int block_size_x, int block_size_y, int block_size_z) {
  return std::make_unique<::ConvertGpuToOCLSPIRV>(block_size_x, block_size_y, block_size_z);
}