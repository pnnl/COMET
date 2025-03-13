
#include "comet/Conversion/TritonToHIP/TritonToHIPPass.h"
#include "comet/Conversion/GpuUtils/GpuUtils.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <cstddef>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#define GEN_PASS_CLASSES
#include "comet/Conversion/TritonToHIP/Passes.h"

using namespace mlir;

namespace hip_device
{
  bool add_ttir_passes(ModuleOp &mod) {
    PassManager pm(mod.getContext());

    pm.addPass(createInlinerPass());
    pm.addPass(triton::createRewriteTensorPointerPass());
    pm.addPass(triton::createCombineOpsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(triton::createReorderBroadcastPass());
    pm.addPass(createCSEPass());
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(triton::createLoopUnrollPass());
    if (failed(pm.run(mod))) {
      return false;
    }
    return true;
  }

  bool add_ttgir_passes(ModuleOp &mod, int32_t numWarps, int32_t threadsPerWarp, int32_t numStages, int32_t numCTAs, std::string computeCapability)  {
    {
      PassManager pm(mod.getContext());
      pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
          "hip:" + computeCapability, numWarps,
          threadsPerWarp, numCTAs));
      if (failed(pm.run(mod))) {
        return false;
      }
    }
    PassManager pm(mod.getContext());
    pm.addPass(triton::gpu::createTritonGPUCoalesce());

    pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(triton::gpu::createTritonGPUOptimizeThreadLocality());
    // TODO: This one takes options
    pm.addPass(createTritonAMDGPUAccelerateMatmulPass(computeCapability));
    pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(createTritonAMDGPUOptimizeEpiloguePass());
    mlir::triton::gpu::TritonGPUOptimizeDotOperandsOptions options;
    options.hoistLayoutConversion = true;
    pm.addPass(triton::gpu::createTritonGPUOptimizeDotOperands(options));
    // TODO: Tensor core options here
    // addd.....

    pm.addPass(createCanonicalizerPass());
    // pm.addPass(triton::createTritonAMDGPUInsertInstructionSchedHintsPass());
    // TODO: Does not work
    pm.addPass(triton::gpu::createTritonGPUOptimizeDotOperands(options));
    pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(triton::gpu::createTritonGPUReduceDataDuplication());
    if (numStages != 0) {
      pm.addPass(createTritonAMDGPUReorderInstructionsPass());
    }
    pm.addPass(createTritonAMDGPUCanonicalizePointersPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSymbolDCEPass());
    if (failed(pm.run(mod))) {
      return false;
    }

    return true;
  }

  bool add_llir_passes(ModuleOp &mod, std::string computeCapability) {
    PassManager pm(mod.getContext());

    pm.addPass(triton::AMD::createDecomposeUnsupportedConversionsPass(
        computeCapability));
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(triton::gpu::createAllocateSharedMemoryPass());
    pm.addPass(
        triton::createConvertTritonAMDGPUToLLVMPass(computeCapability, true));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSymbolDCEPass());
    pm.addPass(triton::createConvertBuiltinFuncToLLVMPass());

    if (failed(pm.run(mod))) {
      return false;
    }

    return true;
  }
}

class LowerTritonDeviceToHIP
    : public mlir::comet::LowerTritonDeviceToHIPBase<LowerTritonDeviceToHIP> {
public:

  LowerTritonDeviceToHIP() = default;

  LowerTritonDeviceToHIP(int numWarps, int threadsPerWarp, int numCTAs,
                         int numStages, std::string computeCapability,
                         mlir::tensorAlgebra::GPUCompilationFormat codeFormat) {

    this->numWarps = numWarps;
    this->threadsPerWarp = threadsPerWarp;
    this->numCTAs = numCTAs;
    this->numStages = numStages;
    this->computeCapability = computeCapability;
    this->codeFormat = codeFormat;
  }

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    OpBuilder builder(modOp);
    auto target = mlir::ROCDL::ROCDLTargetAttr::get(modOp->getContext(), 3,
                                        "amdgcn-amd-amdhsa", computeCapability);
    auto add_ttgir_passes = [this](ModuleOp& modOp) { return hip_device::add_ttgir_passes(modOp, numWarps, threadsPerWarp, numStages, numCTAs, computeCapability); };
    auto add_llir_passes = [this](ModuleOp& modOp) { return hip_device::add_llir_passes(modOp, computeCapability); };
    if(failed(specializeGpuKernel(builder, modOp, this->codeFormat, target, hip_device::add_ttir_passes, add_ttgir_passes, add_llir_passes)))
    {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createLowerTritonDeviceToHIPPass() {
  return std::make_unique<::LowerTritonDeviceToHIP>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createLowerTritonDeviceToHIPPass(
    int numWarps, int threadsPerWarp, int numCTAs, int numStages,
    std::string computeCapability,
    mlir::tensorAlgebra::GPUCompilationFormat format) {
  return std::make_unique<::LowerTritonDeviceToHIP>(
      numWarps, threadsPerWarp, numCTAs, numStages, computeCapability, format);
}

class LowerGpuHostToHIP
    : public mlir::comet::LowerHostToHIPBase<LowerGpuHostToHIP> {
public:
  LowerGpuHostToHIP() = default;

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    OpBuilder builder(modOp);
    declare_vendor_funcs(builder, modOp, "Hip");

    if(failed(specializeGpuHost(builder, modOp, std::string("Hip"))))
    {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createLowerGpuHostToHIPPass() {
  return std::make_unique<::LowerGpuHostToHIP>();
}
