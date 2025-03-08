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
#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
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
      signalPassFailure();
      return false;
    }
    return true;
  }

  bool add_ttgir_passes(ModuleOp &mod) {
    PassManager pm(mod.getContext());

    pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
        "cuda:" + std::to_string(computeCapability), numWarps.getValue(),
        threadsPerWarp.getValue(), numCTAs.getValue()));
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
      signalPassFailure();
      return false;
    }
    return true;
  }

  bool add_llir_passes(ModuleOp &mod) {
    PassManager pm(mod.getContext());

    pm.addPass(triton::NVIDIA::createDecomposeUnsupportedConversionsPass());
    pm.addPass(triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(triton::gpu::createAllocateSharedMemoryPass());
    pm.addPass(mlir::triton::createRewriteTensorPointerPass());
    pm.addPass(
        triton::createConvertTritonGPUToLLVMPass(computeCapability.getValue()));
    pm.addPass(triton::createConvertNVGPUToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSymbolDCEPass());

    if (failed(pm.run(mod))) {
      signalPassFailure();
      return false;
    }
    return true;
  }

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    std::vector<mlir::triton::FuncOp> TTFuncs;
    modOp->walk([&TTFuncs](mlir::triton::FuncOp op) { TTFuncs.push_back(op); });

    // Triton expects one kernel per module so we create temporary modules
    std::vector<ModuleOp> tempMods;
    OpBuilder builder(modOp);
    std::string chip = "sm_" + std::to_string(computeCapability.getValue());
    // The pass that converts LLVM kernels to cubin expects all such kernels to
    // be within gpu.module, so we create such a module to insert the kernels
    // lowered by Triton
    auto tempMod = builder.create<mlir::gpu::GPUModuleOp>(
        modOp->getLoc(), "gpu_module",
        mlir::NVVM::NVVMTargetAttr::get(modOp->getContext(), 3,
                                        "nvptx64-nvidia-cuda", chip));

    if (TTFuncs.empty()) {
      return signalPassFailure();
    }

    for (auto ttFunc : TTFuncs) {
      auto tempMod = ModuleOp::create(modOp.getLoc());
      OpBuilder builder(tempMod.getBodyRegion());
      builder.clone(*ttFunc.getOperation());

      if (!add_ttir_passes(tempMod)) {
        return;
      }

      if (!add_ttgir_passes(tempMod)) {
        return;
      }

      if (!add_llir_passes(tempMod)) {
        return;
      }

      auto oldAttrs = ttFunc->getAttrs().vec();
      // tempMod->dump();

      oldAttrs.insert(oldAttrs.end(), tempMod->getAttrs().begin(),
                      tempMod->getAttrs().end());
      // ttFunc->setAttrs(tempMod->getAttrs());
      ttFunc->setAttrs(oldAttrs);
      tempMods.push_back(tempMod);
    }

    // We have lowered Triton kernels to LLVM, now we need to add them to the
    // gpu.module
    builder.setInsertionPointToStart(&tempMod.getBodyRegion().front());
    for (auto mod : tempMods) {
      for (auto &op : *mod.getBody()) {
        builder.clone(op);
      }
    }

    // GPU dialect verifier expects this
    modOp->setAttr("gpu.container_module", builder.getUnitAttr());

    // The function transformGpuModulesToBinaries expects a module that contains
    // gpu.module(s), so we create one with the kernels we want to convert to
    // cubin
    auto tempOuterMod = ModuleOp::create(modOp.getLoc());
    OpBuilder gbuilder(tempOuterMod.getBodyRegion());
    gbuilder.clone(*tempMod);
    gpu::TargetOptions opts;
    if (this->codeFormat ==
        mlir::tensorAlgebra::GPUCompilationFormat::Assembly) {
      opts = gpu::TargetOptions({}, {}, {}, gpu::CompilationTarget::Assembly);
    } else if (this->codeFormat ==
               mlir::tensorAlgebra::GPUCompilationFormat::Binary) {
      opts = gpu::TargetOptions({}, {}, {}, gpu::CompilationTarget::Binary);
    } else if (this->codeFormat ==
               mlir::tensorAlgebra::GPUCompilationFormat::Fatbin) {
      opts = gpu::TargetOptions({}, {}, {}, gpu::CompilationTarget::Fatbin);
    } else {
      assert(false && "Unexpected gpu compilation code format");
    }
    auto res =
        mlir::gpu::transformGpuModulesToBinaries(tempOuterMod, nullptr, opts);
    if (res.failed()) {
      signalPassFailure();
      return;
    }

    builder.setInsertionPointToStart(modOp.getBody());

    // GPU kernels are now converted to cubin, add them to the main module as
    // global strings
    for (auto &op : *tempOuterMod.getBody()) {
      if (auto binOp = dyn_cast<mlir::gpu::BinaryOp>(op)) {
        auto result = mlir::cast<gpu::ObjectAttr>(*binOp.getObjects().begin())
                          .getObject()
                          .str();
        auto type = LLVM::LLVMArrayType::get(
            IntegerType::get(builder.getContext(), 8), result.size());
        builder.setInsertionPointToStart(modOp.getBody());
        builder.create<LLVM::GlobalOp>(
            modOp->getLoc(), type, /*isConstant=*/false,
            LLVM::Linkage::Internal, "gpu_code", builder.getStringAttr(result),
            /*alignment=*/32);
      }
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



TypedValue<MemRefType> collapseMemref(TypedValue<MemRefType> val, mlir::OpBuilder& builder)
{
    
  auto memref = mlir::cast<mlir::MemRefType>(val.getType());
  if (memref.getRank() == 1)
  {
      return val;
  }

  llvm::SmallVector<llvm::SmallVector<int64_t,2>,1> indices;
  indices.push_back(llvm::SmallVector<int64_t,2>());
  for(int64_t i = 0; i < memref.getRank(); i++)
  {
      indices[0].push_back(i);
  }

    /// Collapse memref to 1D
  auto collapsedMemref = builder.create<mlir::memref::CollapseShapeOp>(val.getLoc(), val, mlir::ArrayRef(indices)).getResult();
  return collapsedMemref;
}

mlir::Value get_memref_num_elements(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc, mlir::Value memref) 
{
    mlir::Value rank = builder.create<mlir::memref::RankOp>(loc, memref);
    mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

    mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>(loc, zero, rank, one, mlir::ValueRange({one}));
    mlir::Block* body = forOp.getBody();
    mlir::Value inductionvar = forOp.getInductionVar();
    mlir::IRRewriter::InsertPoint ip  = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(body);
    mlir::Value dim = builder.create<mlir::memref::DimOp>(loc, memref, inductionvar);
    auto mul = builder.create<mlir::arith::MulIOp>(loc, forOp.getRegionIterArg(0), dim);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange({mul}));
    builder.restoreInsertionPoint(ip);
    
    return forOp.getResult(0);
}

class LowerGpuHostToCuda
    : public mlir::comet::LowerHostToCudaBase<LowerGpuHostToCuda> {
public:
  LowerGpuHostToCuda() = default;

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    OpBuilder builder(modOp);
    std::map<std::string, Value> funcs;
    std::map<std::string, std::string> gpu_to_triton_kernel;
    std::map<std::string, triton::FuncOp> triton_name_to_triton_func_op;

    modOp->walk([&gpu_to_triton_kernel, &triton_name_to_triton_func_op](mlir::triton::FuncOp TTFuncOp) {
      gpu::GPUModuleOp gpuModuleOp = TTFuncOp->getParentOfType<gpu::GPUModuleOp>();
      gpu_to_triton_kernel[gpuModuleOp.getName().str() +"::"+ TTFuncOp.getName().substr(3).str()] = TTFuncOp.getName().str();
      triton_name_to_triton_func_op[TTFuncOp.getName().str()] = TTFuncOp;
      // llvm::errs() << "Created " << gpuModuleOp.getName().str() +"::"+ TTFuncOp.getName().substr(3).str() << "\n";
      // TTFuncOp.erase();
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
      auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(launchOp->getLoc(), launchOp.getKernel(), launchOp.getGridSizeOperandValues(), launchOp.getBlockSizeOperandValues(), launchOp.getDynamicSharedMemorySize(), newValues);
      launchOp->erase();
    });

    auto memrefI32 =
        MemRefType::get({ShapedType::kDynamic}, builder.getIntegerType(32));
    auto memrefF32 =
        MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
    auto memrefI64 =
        MemRefType::get({ShapedType::kDynamic}, builder.getIntegerType(64));
    auto memrefIndex =
        MemRefType::get({ShapedType::kDynamic}, builder.getIndexType());
    auto memrefF64 =
        MemRefType::get({ShapedType::kDynamic}, builder.getF64Type());
    builder.clearInsertionPoint();
    auto mallocI32Type = builder.getFunctionType({builder.getIndexType()},
                                                 builder.getIndexType());
    auto mallocI32 = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaMallocI32", mallocI32Type);
    mallocI32.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(mallocI32);

    auto mallocI64Type = builder.getFunctionType({builder.getIndexType()},
                                                 builder.getIndexType());
    auto mallocI64 = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaMallocI64", mallocI64Type);
    mallocI64.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(mallocI64);

    auto mallocF32Type = builder.getFunctionType({builder.getIndexType()},
                                                 builder.getIndexType());
    auto mallocF32 = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaMallocF32", mallocF32Type);
    mallocF32.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(mallocF32);

    auto mallocF64Type = builder.getFunctionType({builder.getIndexType()},
                                                 builder.getIndexType());
    auto mallocF64 = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaMallocF64", mallocF64Type);
    mallocF64.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(mallocF64);

    auto cudaMemcpyI32Type = builder.getFunctionType(
        {builder.getIndexType(), memrefI32, builder.getIndexType()}, {});
    auto cudaMemcpyI32 = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaMemcpyI32", cudaMemcpyI32Type);
    cudaMemcpyI32.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(cudaMemcpyI32);

    auto cudaMemcpyI64Type = builder.getFunctionType(
        {builder.getIndexType(), memrefI64, builder.getIndexType()}, {});
    auto cudaMemcpyI64 = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaMemcpyI64", cudaMemcpyI64Type);
    cudaMemcpyI64.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(cudaMemcpyI64);

    auto cudaMemcpyIndexType = builder.getFunctionType(
        {builder.getIndexType(), memrefIndex, builder.getIndexType()}, {});
    auto cudaMemcpyIndex = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaMemcpyIndex", cudaMemcpyIndexType);
    cudaMemcpyIndex.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(cudaMemcpyIndex);

    auto cudaMemcpyF32Type = builder.getFunctionType(
        {builder.getIndexType(), memrefF32, builder.getIndexType()}, {});
    auto cudaMemcpyF32 = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaMemcpyF32", cudaMemcpyF32Type);
    cudaMemcpyF32.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(cudaMemcpyF32);

    auto cudaMemcpyF64Type = builder.getFunctionType(
        {builder.getIndexType(), memrefF64, builder.getIndexType()}, {});
    auto cudaMemcpyF64 = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaMemcpyF64", cudaMemcpyF64Type);
    cudaMemcpyF64.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(cudaMemcpyF64);

    auto cudaLaunchKernelT = builder.getFunctionType(
        {builder.getIndexType(), builder.getIndexType(), builder.getIndexType(),
         builder.getIndexType(), builder.getIndexType(), builder.getIndexType(),
         MemRefType::get({ShapedType::kDynamic}, builder.getIndexType()),
         LLVM::LLVMPointerType::get(builder.getContext()),
         builder.getIndexType(), builder.getIntegerType(32),
         builder.getIndexType(), builder.getIndexType()},
        {});
    auto cudaLaunchKernel = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaLaunchKernel", cudaLaunchKernelT);
    cudaLaunchKernel.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(cudaLaunchKernel);

    auto cudaSetModuleImageT = builder.getFunctionType(
        {LLVM::LLVMPointerType::get(builder.getContext())}, {});
    auto cudaSetModuleImage = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "cudaSetModuleImage", cudaSetModuleImageT);
    cudaSetModuleImage.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(cudaSetModuleImage);

    auto cudaFreeT = builder.getFunctionType({builder.getIndexType()}, {});
    auto cudaFree = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                       "cudaFree", cudaFreeT);
    cudaFree.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(cudaFree);
    
    auto cudaFinitT = builder.getFunctionType({}, {});
    auto cudaFinit = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "cudaFinit", cudaFinitT);
    cudaFinit.setVisibility(mlir::SymbolTable::Visibility::Private);
    modOp.push_back(cudaFinit);


    std::vector<mlir::gpu::LaunchFuncOp> launchOps;

    modOp->walk([&launchOps](mlir::gpu::LaunchFuncOp launchOp) {
      launchOps.push_back(launchOp);
    });

    std::set<mlir::func::FuncOp> initFuncs;
    std::vector<Value> toAlloc;
    // auto symbols = SymbolTable(modOp);

    for (auto launchOp : launchOps) {
      auto name =
          gpu_to_triton_kernel[(launchOp.getKernelModuleName().strref() +
                                "::" + launchOp.getKernelName().strref())
                                   .str()];

      if (initFuncs.find(launchOp->getParentOfType<mlir::func::FuncOp>()) ==
          initFuncs.end()) {
        builder.setInsertionPointToStart(
            &launchOp->getParentOfType<mlir::func::FuncOp>()
                 .getFunctionBody()
                 .front());
        Value gpu_code = builder.create<mlir::LLVM::AddressOfOp>(
            launchOp->getLoc(), LLVM::LLVMPointerType::get(&getContext()),
            "gpu_code");
        builder.create<mlir::func::CallOp>(launchOp.getLoc(),
                                           "cudaSetModuleImage", TypeRange(),
                                           ValueRange({gpu_code}));

        initFuncs.insert(launchOp->getParentOfType<mlir::func::FuncOp>());
      }
      builder.setInsertionPoint(launchOp);
      auto ttFunc = triton_name_to_triton_func_op[name];
      Value sharedMem = builder.create<arith::ConstantIntOp>(
          launchOp->getLoc(),
          ttFunc->getAttrOfType<IntegerAttr>("triton_gpu.shared").getInt(), 32);
      launchOp.getDynamicSharedMemorySizeMutable().assign(sharedMem);
      for (auto &operand : launchOp->getOpOperands()) {
        // if(!isa<IndexType>(operand.get().getType()) )
        // {
        // //   auto i32Operand =
        // builder.create<arith::IndexCastOp>(operand.get().getLoc(),
        // builder.getIntegerType(32), operand.get());
        // //   // operand.set(i32Operand);
        // //   cudaCallArgs.push_back(i32Operand);
        // }
        // else
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

    for (auto gpuAlloc : gpuAllocs) {
      builder.setInsertionPoint(gpuAlloc->getBlock()->getTerminator());
      builder.create<mlir::gpu::DeallocOp>(gpuAlloc->getLoc(), ValueRange(),
                                           gpuAlloc.getResult(0));

      builder.setInsertionPointAfter(gpuAlloc);
      Value allocSize;
      if (gpuAlloc.getMemref().getType().hasStaticShape()) {
        allocSize = builder
                        .create<mlir::arith::ConstantIndexOp>(
                            gpuAlloc->getLoc(),
                            gpuAlloc.getMemref().getType().getNumElements())
                        .getResult();
      } else {
        allocSize = get_memref_num_elements(builder.getContext(), builder, gpuAlloc.getLoc(), gpuAlloc.getMemref());
      }

      // builder.create<memref::DimOp>(gpuAlloc.getLoc(), gpuAlloc.getMemref(),
      // 0);
      if (FloatType floatType = mlir::dyn_cast<FloatType>(
              gpuAlloc.getMemref().getType().getElementType())) {
        int width = floatType.getWidth();
        auto cudaOp = builder.create<mlir::func::CallOp>(
            gpuAlloc->getLoc(), "cudaMallocF" + std::to_string(width),
            TypeRange(builder.getIndexType()), ValueRange(allocSize));
        gpuAlloc->replaceAllUsesWith(cudaOp);
      } else if (IntegerType intType = mlir::dyn_cast<IntegerType>(
                     gpuAlloc.getMemref().getType().getElementType())) {
        int width = intType.getWidth();
        auto cudaOp = builder.create<mlir::func::CallOp>(
            gpuAlloc->getLoc(), "cudaMallocI" + std::to_string(width),
            TypeRange(builder.getIndexType()), ValueRange(allocSize));
        gpuAlloc->replaceAllUsesWith(cudaOp);
      } else if (IndexType intType = mlir::dyn_cast<IndexType>(
                     gpuAlloc.getMemref().getType().getElementType())) {
        auto cudaOp = builder.create<mlir::func::CallOp>(
            gpuAlloc->getLoc(), "cudaMallocI" + std::to_string(64),
            TypeRange(builder.getIndexType()), ValueRange(allocSize));
        gpuAlloc->replaceAllUsesWith(cudaOp);
      }
      gpuAlloc->erase();
    }

    std::vector<mlir::gpu::MemcpyOp> gpuCopies;
    modOp->walk([&gpuCopies](mlir::gpu::MemcpyOp gpuCopy) {
      gpuCopies.push_back(gpuCopy);
    });

    llvm::SmallMapVector<TypedValue<BaseMemRefType>, TypedValue<BaseMemRefType>, 4> memref_to_collapsed_memref;
    for (auto cpy : gpuCopies) {
      builder.setInsertionPoint(cpy);
      auto hToD = builder.create<arith::ConstantIndexOp>(cpy->getLoc(), 0);
      auto dToH = builder.create<arith::ConstantIndexOp>(cpy->getLoc(), 1);

      if (cpy.getOperand(0).getDefiningOp() &&
          (isa<mlir::func::CallOp>(cpy.getOperand(0).getDefiningOp()) &&
           cast<mlir::func::CallOp>(cpy.getOperand(0).getDefiningOp())
               .getCallee()
               .starts_with("cudaMalloc"))) {
        if(memref_to_collapsed_memref.find(mlir::cast<TypedValue<BaseMemRefType>>(cpy.getSrc())) == memref_to_collapsed_memref.end())
        {
          TypedValue<MemRefType> collapsedMemref = collapseMemref(cpy.getSrc(), builder);
          auto cast = builder.create<mlir::memref::CastOp>(
              cpy->getLoc(),
              MemRefType::get({ShapedType::kDynamic},
                              collapsedMemref.getType().getElementType()),
              collapsedMemref);
          memref_to_collapsed_memref[mlir::cast<TypedValue<BaseMemRefType>>(cpy.getSrc())] = cast.getResult();
        }
        mlir::Value cast = memref_to_collapsed_memref[mlir::cast<TypedValue<BaseMemRefType>>(cpy.getSrc())];


        if (IntegerType intType = mlir::dyn_cast<IntegerType>(
                mlir::cast<MemRefType>(cpy.getOperand(1).getType())
                    .getElementType())) {
          int width = intType.getWidth();
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), "cudaMemcpyI" + std::to_string(width), TypeRange(),
              ValueRange({cpy.getOperand(0), cast, hToD}));
        } else if (FloatType floatType = mlir::dyn_cast<FloatType>(
                       mlir::cast<MemRefType>(cpy.getOperand(1).getType())
                           .getElementType())) {
          int width = floatType.getWidth();
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), "cudaMemcpyF" + std::to_string(width), TypeRange(),
              ValueRange({cpy.getOperand(0), cast, hToD}));
        } else if (IndexType indexType = mlir::dyn_cast<IndexType>(
                       mlir::cast<MemRefType>(cpy.getOperand(1).getType())
                           .getElementType())) {
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), "cudaMemcpyIndex", TypeRange(),
              ValueRange({cpy.getOperand(0), cast, hToD}));
        }
      } else if (cpy.getOperand(1).getDefiningOp() &&
                 (isa<mlir::func::CallOp>(cpy.getOperand(1).getDefiningOp()) &&
                  cast<mlir::func::CallOp>(cpy.getOperand(1).getDefiningOp())
                      .getCallee()
                      .starts_with("cudaMalloc"))) {
        if(memref_to_collapsed_memref.find(mlir::cast<TypedValue<BaseMemRefType>>(cpy.getDst())) == memref_to_collapsed_memref.end())
        {
          TypedValue<MemRefType> collapsedMemref = collapseMemref(cpy.getDst(), builder);
          auto cast = builder.create<mlir::memref::CastOp>(
              cpy->getLoc(),
              MemRefType::get({ShapedType::kDynamic},
                              collapsedMemref.getType().getElementType()),
              collapsedMemref);
          memref_to_collapsed_memref[mlir::cast<TypedValue<BaseMemRefType>>(cpy.getDst())] = cast.getResult();
        }
        mlir::Value cast = memref_to_collapsed_memref[mlir::cast<TypedValue<BaseMemRefType>>(cpy.getDst())];

        if (IntegerType intType = mlir::dyn_cast<IntegerType>(
                mlir::cast<MemRefType>(cpy.getOperand(0).getType())
                    .getElementType())) {
          int width = intType.getWidth();
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), "cudaMemcpyI" + std::to_string(width), TypeRange(),
              ValueRange({cpy.getOperand(1), cast, dToH}));
        } else if (FloatType floatType = mlir::dyn_cast<FloatType>(
                       mlir::cast<MemRefType>(cpy.getOperand(0).getType())
                           .getElementType())) {
          int width = floatType.getWidth();
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), "cudaMemcpyF" + std::to_string(width), TypeRange(),
              ValueRange({cpy.getOperand(1), cast, dToH}));
        } else if (IndexType indexType = mlir::dyn_cast<IndexType>(
                       mlir::cast<MemRefType>(cpy.getOperand(0).getType())
                           .getElementType())) {
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), "cudaMemcpyIndex", TypeRange(),
              ValueRange({cpy.getOperand(1), cast, dToH}));
        }
      }

      cpy->erase();
    }

    for (auto launchOp : launchOps) {
      builder.setInsertionPoint(launchOp);
      auto zeroIndex =
          builder.create<arith::ConstantIndexOp>(launchOp->getLoc(), 0);

      int64_t numOps = launchOp.getNumKernelOperands();
      std::vector<Value> ptr_ops;
      // memref::AllocaOp temp = builder.create<mlir::memref::AllocaOp>(
      //     launchOp->getLoc(),
      //     MemRefType::get({1}, launchOp.getGridSizeX().getType()));
      // builder.create<memref::StoreOp>(launchOp->getLoc(),
      //                                 launchOp.getGridSizeY(), temp,
      //                                 ValueRange({zeroIndex}));
      // ptr_ops.push_back(
      //     builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
      //         launchOp->getLoc(), temp));
      // temp = builder.create<mlir::memref::AllocaOp>(
      //     launchOp->getLoc(),
      //     MemRefType::get({1}, launchOp.getGridSizeY().getType()));
      // builder.create<memref::StoreOp>(launchOp->getLoc(),
      //                                 launchOp.getGridSizeX(), temp,
      //                                 ValueRange({zeroIndex}));
      // ptr_ops.push_back(
      //     builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
      //         launchOp->getLoc(), temp));

      for (auto op : launchOp.getKernelOperands()) {
        if (mlir::isa<MemRefType>(op.getType())) {
          ptr_ops.push_back(
              builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
                  launchOp->getLoc(), op));
        } else {
          auto temp = builder.create<mlir::memref::AllocaOp>(
              launchOp->getLoc(), MemRefType::get({1}, op.getType()));
          builder.create<memref::StoreOp>(launchOp->getLoc(), op, temp,
                                          ValueRange({zeroIndex}));
          ptr_ops.push_back(
              builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
                  launchOp->getLoc(), temp));
        }
      }

      auto args = builder.create<memref::AllocaOp>(
          launchOp->getLoc(),
          MemRefType::get({numOps}, builder.getIndexType()));
      auto args_dynamic = builder.create<memref::CastOp>(
          launchOp->getLoc(),
          MemRefType::get({ShapedType::kDynamic},
                          args.getType().getElementType()),
          args);
      for (size_t i = 0; i < ptr_ops.size(); i++) {
        Value op = ptr_ops[i];
        builder.create<memref::StoreOp>(
            launchOp->getLoc(), op, args,
            builder.create<arith::ConstantIndexOp>(launchOp->getLoc(), i)
                .getResult());
      }
      std::string &funcName =
          gpu_to_triton_kernel[(launchOp.getKernelModuleName().strref() +
                                "::" + launchOp.getKernelName().strref())
                                   .str()];
      if (funcs.find(funcName) == funcs.end()) {
        // We need the global string to include the \0 character so that it is
        // correctly read by the cuda lib Hence the StringRef(funcName.c_str(),
        // funcName.size()+1)
        funcs[funcName] = LLVM::createGlobalString(
            modOp->getLoc(), builder, funcName + "_str",
            StringRef(funcName.c_str(), funcName.size() + 1),
            LLVM::linkage::Linkage::Private);
      }

      auto name =
          gpu_to_triton_kernel[(launchOp.getKernelModuleName().strref() +
                                "::" + launchOp.getKernelName().strref())
                                   .str()];
      auto ttFunc = triton_name_to_triton_func_op[name];
      Value numWarps = builder.create<arith::ConstantIndexOp>(
          launchOp->getLoc(),
          ttFunc->getAttrOfType<IntegerAttr>("triton_gpu.num-warps").getInt());
      Value threadsPerWarp = builder.create<arith::ConstantIndexOp>(
          launchOp->getLoc(),
          ttFunc->getAttrOfType<IntegerAttr>("triton_gpu.threads-per-warp")
              .getInt());
      builder.create<mlir::func::CallOp>(
          launchOp->getLoc(), "cudaLaunchKernel", TypeRange(),
          ValueRange({launchOp.getGridSizeX(), launchOp.getGridSizeY(),
                      launchOp.getGridSizeZ(), launchOp.getBlockSizeX(),
                      launchOp.getBlockSizeY(), launchOp.getBlockSizeZ(),
                      args_dynamic, funcs[funcName],
                      builder.create<arith::ConstantIndexOp>(launchOp->getLoc(),
                                                             funcName.size()),
                      launchOp.getDynamicSharedMemorySize(), numWarps,
                      threadsPerWarp}));
      launchOp->erase();
    }

    std::vector<mlir::gpu::DeallocOp> gpuDeallocs;
    modOp->walk([&gpuDeallocs](mlir::gpu::DeallocOp gpuDeallocOp) {
      gpuDeallocs.push_back(gpuDeallocOp);
    });

    for (auto dealloc : gpuDeallocs) {
      builder.setInsertionPoint(dealloc);
      builder.create<mlir::func::CallOp>(dealloc->getLoc(), "cudaFree",
                                         TypeRange(),
                                         ValueRange(dealloc->getOperand(0)));
      dealloc.erase();
    }
    func::FuncOp funcOp = *initFuncs.begin();
    builder.setInsertionPoint(funcOp.getBody().front().getTerminator());

    builder.create<mlir::func::CallOp>(funcOp.getBody().front().getTerminator()->getLoc(), "cudaFinit", TypeRange(), ValueRange());
    modOp->walk([](mlir::triton::FuncOp TTFuncOp) {
      TTFuncOp.erase();
    });
    modOp->walk([](mlir::gpu::GPUModuleOp gpuMod) {gpuMod.erase();});
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::comet::createLowerGpuHostToCudaPass() {
  // std::cout << "Running createLowerGpuHostToCudaPass\n";

  return std::make_unique<::LowerGpuHostToCuda>();
}
