//
// Copyright 2022 Battelle Memorial Institute
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#include <cstddef>
#include <list>
#include <memory>
#include "comet/Conversion/TritonToCuda/TritonToCudaPass.h"
#include "comet/Conversion/GpuUtils/GpuUtils.h"
#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

// #include "triton/Conversion/NVGPUToLLVM/TritonGPUToLLVMPass.h"
// #include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "third_party/nvidia/include/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include <set>
#include <string>

#define GEN_PASS_CLASSES
#include "comet/Conversion/TritonToCuda/Passes.h"

using namespace mlir;

namespace cuda_device {

  bool add_ttir_passes(ModuleOp &mod) 
  {
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

  bool add_ttgir_passes(ModuleOp &mod, int32_t numWarps, int32_t threadsPerWarp, int32_t numStages, int32_t numCTAs, int32_t computeCapability) 
  {
    PassManager pm(mod.getContext());
  
    pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
    "cuda:" + std::to_string(computeCapability), numWarps,
    threadsPerWarp, numCTAs));
    pm.addPass(triton::gpu::createTritonGPUCoalesce());
    if (computeCapability / 10 >= 8) {
      pm.addPass(triton::gpu::createTritonGPUF32DotTC());
    }
    pm.addPass(createTritonNvidiaGPUPlanCTAPass());
    pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(triton::gpu::createTritonGPUOptimizeThreadLocality());
    pm.addPass(triton::gpu::createTritonGPUAccelerateMatmul());
    pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
    mlir::triton::gpu::TritonGPUOptimizeDotOperandsOptions options;
    options.hoistLayoutConversion = computeCapability >= 80;
    pm.addPass(triton::gpu::createTritonGPUOptimizeDotOperands(options));
    pm.addPass(createCSEPass());
    if (computeCapability / 10 >= 8) {
      pm.addPass(triton::gpu::createTritonGPUOptimizeAccumulatorInit());
      pm.addPass(triton::gpu::createTritonGPUCombineTensorSelectAndIf());
      mlir::triton::gpu::TritonGPUPipelineOptions options;
      options.numStages = numStages;
      pm.addPass(triton::gpu::createTritonGPUPipeline(options));
    }
    pm.addPass(triton::gpu::createTritonGPUPrefetch());
    pm.addPass(triton::gpu::createTritonGPUOptimizeDotOperands(options));
    pm.addPass(triton::gpu::createTritonGPUReduceDataDuplication());
    pm.addPass(triton::gpu::createTritonGPUReorderInstructions());
    pm.addPass(createCSEPass());
    pm.addPass(createSymbolDCEPass());
    if (computeCapability / 10 >= 9) {
      pm.addPass(createTritonNvidiaGPUFenceInsertionPass());
      pm.addPass(createTritonNvidiaGPUTMALoweringPass());
    }
    pm.addPass(createCanonicalizerPass());
    
    if (failed(pm.run(mod))) {
      return false;
    }
    return true;
  }
  
  bool add_llir_passes(ModuleOp &mod, int32_t computeCapability) 
  {
    PassManager pm(mod.getContext());
    
    pm.addPass(triton::NVIDIA::createDecomposeUnsupportedConversionsPass());
    pm.addPass(triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(triton::gpu::createAllocateSharedMemoryPass());
    pm.addPass(mlir::triton::createRewriteTensorPointerPass());
    pm.addPass(
      triton::createConvertTritonGPUToLLVMPass(computeCapability));
      pm.addPass(triton::createConvertNVGPUToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      pm.addPass(createSymbolDCEPass());
      
      if (failed(pm.run(mod))) {
        return false;
      }
      return true;
  }  
}
class LowerTritonDeviceToCuda
    : public mlir::comet::LowerTritonDeviceToCudaBase<LowerTritonDeviceToCuda> {
public:
  LowerTritonDeviceToCuda() = default;

  LowerTritonDeviceToCuda(
      int numWarps, int threadsPerWarp, int numCTAs, int numStages,
      int computeCapability,
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
    std::string chip = "sm_" + std::to_string(computeCapability.getValue());
    auto target = mlir::NVVM::NVVMTargetAttr::get(modOp->getContext(), 3,
                                        "nvptx64-nvidia-cuda", chip);
    auto add_ttgir_passes = [this](ModuleOp& modOp) { return cuda_device::add_ttgir_passes(modOp, numWarps, threadsPerWarp, numStages, numCTAs, computeCapability); };
    auto add_llir_passes = [this](ModuleOp& modOp) { return cuda_device::add_llir_passes(modOp, computeCapability); };
    if(failed(specializeGpuKernel(builder, modOp, this->codeFormat, target, cuda_device::add_ttir_passes, add_ttgir_passes, add_llir_passes)))
    {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createLowerTritonDeviceToCudaPass() {
  return std::make_unique<::LowerTritonDeviceToCuda>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createLowerTritonDeviceToCudaPass(
    int numWarps, int threadsPerWarp, int numCTAs, int numStages,
    int computeCapability, mlir::tensorAlgebra::GPUCompilationFormat format) {
  return std::make_unique<::LowerTritonDeviceToCuda>(
      numWarps, threadsPerWarp, numCTAs, numStages, computeCapability, format);
}

class LowerGpuHostToCuda
    : public mlir::comet::LowerHostToCudaBase<LowerGpuHostToCuda> {
public:
  LowerGpuHostToCuda() = default;

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    OpBuilder builder(modOp);
    declare_vendor_funcs(builder, modOp, "cuda");

    if(failed(specializeGpuHost(builder, modOp, std::string("cuda"))))
    {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createLowerGpuHostToCudaPass() {
  return std::make_unique<::LowerGpuHostToCuda>();
}
