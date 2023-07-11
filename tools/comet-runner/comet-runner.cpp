#include "comet/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

static LogicalResult runMLIRPasses(Operation *op, JitRunnerOptions &options) {
    auto module = dyn_cast<ModuleOp>(op);
    if (!module)
        return op->emitOpError("expected a 'builtin.module' op");
    PassManager passManager(module.getContext());
    applyPassManagerCLOptions(passManager);
    passManager.addPass(createInsertMCLCallsPass());
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createConvertSCFToCFPass());
    passManager.addPass(memref::createExpandStridedMetadataPass());
    passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
    passManager.addNestedPass<func::FuncOp>(LLVM::createRequestCWrappersPass());
    ConvertFuncToLLVMPassOptions funcToLLVMOptions{};
    funcToLLVMOptions.indexBitwidth =
            DataLayout(module).getTypeSizeInBits(IndexType::get(module.getContext()));
    passManager.addPass(createConvertFuncToLLVMPass(funcToLLVMOptions));
    passManager.addPass(createReconcileUnrealizedCastsPass());
    return passManager.run(module);
}

int main(int argc, char **argv) {
    registerPassManagerCLOptions();
    llvm::InitLLVM y(argc, argv);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    JitRunnerOptions options;
    auto runPassesWithOptions = [&options](Operation *op, JitRunnerOptions &) {
        return runMLIRPasses(op, options);
    };
    JitRunnerConfig config;
    config.mlirTransformer = runPassesWithOptions;
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, LLVM::LLVMDialect, gpu::GPUDialect,
        memref::MemRefDialect, scf::SCFDialect, spirv::SPIRVDialect, func::FuncDialect>();
    registerLLVMDialectTranslation(registry);
    return mlir::JitRunnerMain(argc, argv, registry, config);
}
