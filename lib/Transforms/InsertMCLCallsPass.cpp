#include "comet/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

#include <fstream>

namespace mlir {
#define GEN_PASS_DEF_INSERTMCLCALLS
#include "comet/Transforms/CometTransforms.h.inc"
}

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;
using namespace LLVM;

#define GET_RW_LOC rewriter.getInsertionPoint()->getLoc()

#define DECLARE_MCL_OP(type, name) rewriter.create<LLVMFuncOp>(rewriter.getInsertionPoint()->getLoc(), (name), (type))

namespace cl = llvm::cl;

static cl::opt<enum Targets> mclTarget(
        "mcl-target", cl::init(ANY), cl::desc("Target device to run on"),
        cl::values(clEnumValN(CPU, "cpu", "Target to run on cpu")),
        cl::values(clEnumValN(GPU, "gpu", "Target to run on gpu")),
        cl::values(clEnumValN(FPGA, "fpga", "Target to run on fpga")),
        cl::values(clEnumValN(ANY, "any", "Target to run on any device"))
);

static cl::opt<std::string> clPrgName(
        "xclbin-file", cl::init("-"), cl::desc("Path to xclbin file synthesized for this program. Required when targeting fpga"),
        cl::value_desc("filename")
);

static cl::opt<int> mclWorkers(
        "mcl-workers", cl::init(1), cl::desc("Number of workers to run mcl with"),
        cl::value_desc("int")
);

static cl::opt<std::string> fpgaKernelName(
        "fpga-kernel-name", cl::init("-"), cl::desc("Name of the kernel to be run on fpga. Required to target fpga when host code contains more then one kernel"),
        cl::value_desc("string")
);

static constexpr const char *kMCLInitFname = "mcl_init";
static constexpr const char *kMCLPrgLoadFname = "mcl_prg_load";
static constexpr const char *kMCLCreateTaskFname = "mcl_task_create";
static constexpr const char *kMCLSetKernelFname = "mcl_task_set_kernel";
static constexpr const char *kMCLSetArgFname = "mcl_task_set_arg";
static constexpr const char *kMCLExecFname = "mcl_exec";
static constexpr const char *kMCLWaitFname = "mcl_wait";

GEPOp gepOpWrapper(LLVMPointerType returnType, Value base, Value realIndex, Location loc, PatternRewriter &rewriter) {
    if (realIndex.getType().isIndex())
        realIndex = rewriter.create<IndexCastOp>(loc, rewriter.getIntegerType(64), realIndex).getResult();
    auto constZeroOp = rewriter.create<ConstantIntOp>(loc, 0, 64);
    rewriter.setInsertionPointAfter(constZeroOp);
    auto elemPtrOp = rewriter.create<GEPOp>(GET_RW_LOC, returnType, base, ValueRange{constZeroOp, realIndex});
    return elemPtrOp;
}

llvm::DenseMap<const char*, LLVMFuncOp> declareMclFuncs(func::FuncOp func, PatternRewriter &rewriter) {
    IntegerType int32_type = rewriter.getIntegerType(32);
    IntegerType int64_type = rewriter.getIntegerType(64);
    LLVMPointerType int8_ptr_type = LLVMPointerType::get(rewriter.getIntegerType(8));
    LLVMPointerType xesPtrType = LLVMPointerType::get(LLVMArrayType::get(int64_type, MCL_DEV_DIMS));
    LLVMStructType mclHandleType = LLVMStructType::getLiteral(rewriter.getContext(), {int64_type, int32_type, int64_type, int64_type, int32_type});
    LLVMPointerType mclHandlePtrType = LLVMPointerType::get(mclHandleType);
    llvm::DenseMap<const char*, LLVMFuncOp> mclOps;
    rewriter.setInsertionPointAfter(func);
    LLVMFunctionType taskCreateFtype = LLVMFunctionType::get(mclHandlePtrType, {});
    LLVMFunctionType setKernelFtype = LLVMFunctionType::get(int32_type, {mclHandlePtrType, int8_ptr_type, int64_type});
    LLVMFunctionType setArgFtype = LLVMFunctionType::get(int32_type, {mclHandlePtrType, int64_type, int8_ptr_type, int64_type, int64_type});
    LLVMFunctionType execFtype = LLVMFunctionType::get(int32_type, {mclHandlePtrType, xesPtrType, xesPtrType, int64_type});
    LLVMFunctionType waitFtype = LLVMFunctionType::get(int32_type, {mclHandlePtrType});
    mclOps.insert(std::pair<const char*, LLVMFuncOp>(kMCLCreateTaskFname, DECLARE_MCL_OP(taskCreateFtype, kMCLCreateTaskFname)));
    mclOps.insert(std::pair<const char*, LLVMFuncOp>(kMCLSetKernelFname, DECLARE_MCL_OP(setKernelFtype, kMCLSetKernelFname)));
    mclOps.insert(std::pair<const char*, LLVMFuncOp>(kMCLSetArgFname, DECLARE_MCL_OP(setArgFtype, kMCLSetArgFname)));
    mclOps.insert(std::pair<const char*, LLVMFuncOp>(kMCLExecFname, DECLARE_MCL_OP(execFtype, kMCLExecFname)));
    mclOps.insert(std::pair<const char*, LLVMFuncOp>(kMCLWaitFname, DECLARE_MCL_OP(waitFtype, kMCLWaitFname)));
    return mclOps;
}

LogicalResult insertMCLInitCall(FuncOp func, uint64_t workers, PatternRewriter &rewriter) {
    IntegerType int64_type = rewriter.getIntegerType(64);
    rewriter.setInsertionPointAfter(func);
    LLVMFunctionType llvmFuncType = LLVMFunctionType::get(rewriter.getIntegerType(32), {int64_type, int64_type});
    auto llvmFuncOp = rewriter.create<LLVMFuncOp>(GET_RW_LOC, kMCLInitFname, llvmFuncType);
    rewriter.setInsertionPointToStart(&func.getBody().front());
    auto constNumWorkersOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, workers, 64);
    auto constNoCpuBindOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, 0, 64);
    rewriter.create<LLVM::CallOp>(GET_RW_LOC, llvmFuncOp, ValueRange{constNumWorkersOp.getResult(), constNoCpuBindOp.getResult()});
    return success();
}

LogicalResult insertMCLPrgLoadCall(FuncOp func, int64_t target, const std::string &prgName, PatternRewriter &rewriter) {
    IntegerType int8_type = rewriter.getIntegerType(8);
    llvm::SmallVector<spirv::ModuleOp> spirvModules;
    func->getParentOp()->walk([&spirvModules](spirv::ModuleOp op) { spirvModules.push_back(op); });
    llvm::SmallVector<std::pair<ConstantIntOp, std::string>> programs;
    rewriter.setInsertionPointAfter(func.getBody().front().front().getNextNode());
    if (target & MCL_TASK_FPGA) {
        programs.push_back({rewriter.create<ConstantIntOp>(GET_RW_LOC, MCL_PRG_BIN, 64), prgName});
    }
    if (target & MCL_TASK_CPU || target & MCL_TASK_GPU) {
        auto prgIrOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, MCL_PRG_IR, 64);
        if (!spirvModules.empty())
            for (spirv::ModuleOp mOp : spirvModules) {
                SmallVector<uint32_t, 0> binary;
                if (!failed(spirv::serialize(mOp, binary))) {
                    std::string pathname("spirv_comet_");
                    //try to drop __spv__ prefix
                    pathname += mOp.getName()->data();
                    pathname += ".bin";
                    //pathname.erase(pathname.find("_spv__"), 6);
                    std::ofstream textout(pathname.c_str(), std::ios::out | std::ios::binary);
                    textout.write(reinterpret_cast<char *>(binary.data()),
                              binary.size() * sizeof(uint32_t));
                    textout.close();
                    programs.push_back({prgIrOp, pathname});
                }
            }
    }
    if (!spirvModules.empty())
        for (spirv::ModuleOp mOp : spirvModules)
            mOp.erase(); //Not needed at this point and can cause compile to fail if left in
    rewriter.setInsertionPointAfter(func);
    LLVMArrayType emptyStrType = LLVMArrayType::get(int8_type, 1); // 1 for null
    auto constNoCopts = rewriter.create<GlobalOp>(GET_RW_LOC, emptyStrType, true, Linkage::Internal, "c_opts",
                                                      StringAttr::get(std::string() + '\0', int8_type));
    LLVMFunctionType llvmFuncType = LLVMFunctionType::get(rewriter.getIntegerType(32),
                                                          {LLVMPointerType::get(int8_type),
                                                           LLVMPointerType::get(int8_type),
                                                           rewriter.getIntegerType(64)});
    auto llvmFuncOp = rewriter.create<LLVMFuncOp>(GET_RW_LOC, kMCLPrgLoadFname, llvmFuncType);
    for (auto prg : programs) {
        LLVMArrayType prgNameArrType = LLVMArrayType::get(int8_type, prg.second.size() + 1); // +1 for null
        rewriter.setInsertionPointAfter(func);
        auto constPrgName = rewriter.create<GlobalOp>(GET_RW_LOC, prgNameArrType, true, Linkage::Internal,
                                                          prg.second, StringAttr::get(prg.second + '\0', int8_type));
        rewriter.setInsertionPointAfter(prg.first);
        auto prgNamePtrOp = rewriter.create<AddressOfOp>(GET_RW_LOC, constPrgName);
        auto cOptsPtrOp = rewriter.create<AddressOfOp>(GET_RW_LOC, constNoCopts);
        auto ptrToIntOp = rewriter.create<PtrToIntOp>(GET_RW_LOC, rewriter.getIntegerType(64), prgNamePtrOp.getResult());
        auto intToPtrOp = rewriter.create<IntToPtrOp>(GET_RW_LOC, LLVMPointerType::get(int8_type), ptrToIntOp->getResult(0));
        auto cOptsPtrToIntOp = rewriter.create<PtrToIntOp>(GET_RW_LOC, rewriter.getIntegerType(64), cOptsPtrOp.getResult());
        auto cOptsIntToPtrOp = rewriter.create<IntToPtrOp>(GET_RW_LOC, LLVMPointerType::get(int8_type), cOptsPtrToIntOp->getResult(0));

        rewriter.create<LLVM::CallOp>(GET_RW_LOC, llvmFuncOp,
                                      ValueRange{intToPtrOp.getResult(), cOptsIntToPtrOp.getResult(),
                                                 prg.first.getResult()});
    }
    return success();
}

LLVM::CallOp insertMCLTaskCreateCall(gpu::LaunchFuncOp rootKernelCallOp, LLVMFuncOp taskCreateOp, PatternRewriter &rewriter) {
    rewriter.setInsertionPointAfter(rootKernelCallOp->getPrevNode());
    auto mclTaskCreateCall = rewriter.create<LLVM::CallOp>(GET_RW_LOC, taskCreateOp, ValueRange());
    return mclTaskCreateCall;
}

LogicalResult insertMCLSetKernelCall(gpu::LaunchFuncOp rootKernelCallOp, Value mclHandle, LLVMFuncOp setKernelOp, PatternRewriter &rewriter) {
    IntegerType int8_type = rewriter.getIntegerType(8);
    rewriter.setInsertionPointAfter(rootKernelCallOp->getPrevNode());
    StringRef kName = rootKernelCallOp.getKernelName();
    size_t kernelNameLen = kName.size() + 1; //+1 for null
    LLVMArrayType nameArrayType = LLVMArrayType::get(int8_type, kernelNameLen);
    StringAttr kernelName = StringAttr::get(kName.str() + '\0', int8_type);
    rewriter.setInsertionPointAfter(setKernelOp);
    auto constNameOp = rewriter.create<GlobalOp>(GET_RW_LOC, nameArrayType, true, Linkage::Internal, "kernel_name", kernelName);
    rewriter.setInsertionPointAfter(rootKernelCallOp->getPrevNode());
    auto getNamePtrOp = rewriter.create<AddressOfOp>(GET_RW_LOC, constNameOp);
    auto ptrToIntOp = rewriter.create<PtrToIntOp>(GET_RW_LOC, rewriter.getIntegerType(64), getNamePtrOp.getResult());
    auto intToPtrOp = rewriter.create<IntToPtrOp>(GET_RW_LOC, LLVMPointerType::get(int8_type), ptrToIntOp->getResult(0));
    auto constNumArgsOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, rootKernelCallOp.getNumKernelOperands(), 64);
    rewriter.create<LLVM::CallOp>(GET_RW_LOC, setKernelOp, ValueRange({mclHandle, intToPtrOp.getResult(), constNumArgsOp.getResult()}));
    return success();
}

LogicalResult insertMCLSetArgCalls(gpu::LaunchFuncOp rootKernelCallOp, Value mclHandle, LLVMFuncOp setArgsOp, PatternRewriter &rewriter) {
    IntegerType int64_type = rewriter.getIntegerType(64);
    IntegerType int8_type = rewriter.getIntegerType(8);
    LLVMPointerType int8_ptr_type = LLVMPointerType::get(int8_type);
    rewriter.setInsertionPointAfter(rootKernelCallOp->getPrevNode());
    ConstantIntOp flagsOp = nullptr;
    for(unsigned int i = 0; i < rootKernelCallOp.getNumKernelOperands(); i++) {
        Value constantArgNum = rewriter.create<ConstantIntOp>(GET_RW_LOC, i, 64).getResult();
        Value kArg = rootKernelCallOp.getKernelOperand(i);
        //TODO: Will this work if a memref is being passed to this function instead of being allocated?
        Type kArgType;
        Operation *sizeToIntOp;
        if (memref::AllocOp alloc = llvm::dyn_cast<memref::AllocOp>(kArg.getDefiningOp())) {
            kArgType = alloc.getType().getElementType();
            sizeToIntOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, alloc.getType().getNumElements(), 64);
            //Cast to i8/void ptr
            auto ext = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(GET_RW_LOC, rewriter.getIndexType(), kArg);
            auto idxCast = rewriter.create<IndexCastOp>(GET_RW_LOC, int64_type, ext.getResult());
            auto intToPtrOp = rewriter.create<IntToPtrOp>(GET_RW_LOC, int8_ptr_type, idxCast.getResult());
            kArg = intToPtrOp.getResult();
	} else {
            kArgType = kArg.getType();
            sizeToIntOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, 1, 64);
        }
	//TODO: Improve this hueristic
	switch ((i == rootKernelCallOp.getNumKernelOperands() - 1) + (2 * (kArg.getType().getTypeID() == LLVMPointerType::getTypeID()))) {
	  case 0: //Not last arg and not a pointer
	    flagsOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, MCL_ARG_SCALAR | MCL_ARG_INPUT, 64);
	    break;
	  case 1: //Last arg and not a pointer
	    flagsOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, MCL_ARG_SCALAR | MCL_ARG_OUTPUT, 64);
	    break;
	  case 2: //Not last arg and is a pointer
	    flagsOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, MCL_ARG_BUFFER | MCL_ARG_INPUT, 64);
	    break;
	  case 3: //Last arg and is a pointer
	    flagsOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, MCL_ARG_BUFFER | MCL_ARG_OUTPUT, 64);
	    break;
	  default: //Should not be possible
	    llvm_unreachable("Error determining mcl arg flags. This error should not be reachable.");
	}
        ConstantIntOp typeNumBytes;
        if (kArgType.isIntOrFloat())
            typeNumBytes = rewriter.create<ConstantIntOp>(GET_RW_LOC, (int64_t) ceil((double) kArgType.getIntOrFloatBitWidth() / 8.), 64);
        else
	        return failure();
        auto numArgBytesOp = rewriter.create<MulIOp>(GET_RW_LOC, sizeToIntOp->getResult(0), typeNumBytes.getResult());
        rewriter.create<LLVM::CallOp>(GET_RW_LOC, setArgsOp, ValueRange({mclHandle, constantArgNum, kArg, numArgBytesOp.getResult(), flagsOp.getResult()}));
    }
    return success();
}

LogicalResult insertMCLExecCall(gpu::LaunchFuncOp rootKernelCallOp, int64_t target, Value mclHandle, LLVMFuncOp execOp, PatternRewriter &rewriter) {
    IntegerType int64_type = rewriter.getIntegerType(64);
    LLVMPointerType int64_ptr_type = LLVMPointerType::get(int64_type);
    rewriter.setInsertionPointAfter(rootKernelCallOp->getPrevNode());
    auto constMclDevDimsOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, MCL_DEV_DIMS, 64);
    LLVMPointerType xesPtrType = LLVMPointerType::get(LLVMArrayType::get(int64_type, MCL_DEV_DIMS));
    auto llvmPesAlloc = rewriter.create<AllocaOp>(GET_RW_LOC, xesPtrType, constMclDevDimsOp.getResult());
    auto llvmLesAlloc = rewriter.create<AllocaOp>(GET_RW_LOC, xesPtrType, constMclDevDimsOp.getResult());
    auto blockSizesEnum = rootKernelCallOp.getBlockSizeOperandValues();
    auto gridSizesEnum = rootKernelCallOp.getGridSizeOperandValues();
    Value blockSizes[MCL_DEV_DIMS] = {blockSizesEnum.x, blockSizesEnum.y, blockSizesEnum.z};
    Value gridSizes[MCL_DEV_DIMS] = {gridSizesEnum.x, gridSizesEnum.y, gridSizesEnum.z};
    for (unsigned int i = 0; i < MCL_DEV_DIMS; i++) {
        auto storeIndex = rewriter.create<ConstantIntOp>(GET_RW_LOC, i, 64);
        auto blockIndexCast = rewriter.create<IndexCastOp>(GET_RW_LOC, int64_type, blockSizes[i]);
        GEPOp blockElemPtrOp = gepOpWrapper(int64_ptr_type, llvmLesAlloc.getResult(),
                                            storeIndex.getResult(), GET_RW_LOC, rewriter);
        rewriter.create<StoreOp>(GET_RW_LOC, blockIndexCast.getResult(), blockElemPtrOp.getResult());
        auto threadIndexCast = rewriter.create<IndexCastOp>(GET_RW_LOC, int64_type, gridSizes[i]);
        GEPOp threadElemPtrOp = gepOpWrapper(int64_ptr_type, llvmPesAlloc.getResult(),
                                             storeIndex.getResult(), GET_RW_LOC, rewriter);
        rewriter.create<StoreOp>(GET_RW_LOC, threadIndexCast.getResult(), threadElemPtrOp.getResult());
    }
    auto execFlagsOp = rewriter.create<ConstantIntOp>(GET_RW_LOC, target, 64);
    rewriter.setInsertionPointAfter(rootKernelCallOp->getPrevNode());
    rewriter.create<LLVM::CallOp>(GET_RW_LOC, execOp, ValueRange({mclHandle, llvmPesAlloc.getResult(), llvmLesAlloc.getResult(), execFlagsOp.getResult()}));
    return success();
}

LogicalResult insertMCLWaitCall(gpu::LaunchFuncOp rootKernelCallOp, LLVMFuncOp waitOp, FuncOp func, Value mclHandle, PatternRewriter &rewriter) {
    rewriter.setInsertionPointAfter(rootKernelCallOp->getPrevNode());
    auto waitCallOp = rewriter.create<LLVM::CallOp>(GET_RW_LOC, waitOp, ValueRange{mclHandle});
    unsigned int numResults = func->getNumResults();
    if (numResults > 1)
        return failure(); //Needs to return a single f32 result
    if (numResults == 1 && func->getResult(0).getType().isF32())
        return success(); //Needs to return a single f32 result
    if (numResults == 0) {
        auto toFpOp = rewriter.create<LLVM::SIToFPOp>(GET_RW_LOC, rewriter.getF32Type(), waitCallOp.getResult());
        func.walk([&toFpOp](func::ReturnOp retOp) {retOp->insertOperands(0, toFpOp.getResult());});
        func.setType(FunctionType::get(rewriter.getContext(), func->getOperandTypes(), rewriter.getF32Type()));
        return success();
    }
    return failure();
}

LogicalResult insertMCLCalls(gpu::LaunchFuncOp rootKernelCall, PatternRewriter &rewriter) {
    auto mOp = rootKernelCall->getParentOfType<ModuleOp>();
    if (clPrgName == "-") {
        if (mclTarget == FPGA) {
            mOp.emitError("Targeting fpga requires specifying --xclbin-file");
            return failure();
        }
        if (mclTarget == ANY)
            mOp.emitWarning("Target any which includes fpga was selected, but fpga will not be used because --xclbin-file wasn't specified.");
    }
    SmallVector<gpu::LaunchFuncOp> kernelModules;
    mOp.walk([&kernelModules](gpu::LaunchFuncOp op) {
        kernelModules.push_back(op);
    });
    FuncOp main = nullptr;
    mOp.walk([&main](FuncOp op) {
        if (op.getSymName().str() == "main")
            main = op;
    });
    if (main == nullptr) {
        mOp.emitError("Could not find main function");
        return failure();
    }
    if (kernelModules.size() > 1 && fpgaKernelName == "-") {
        if (mclTarget == FPGA) {
            mOp.emitError(
                    "Targeting fpga requires specifying --fpga-kernel-name when the host code contains more than one kernel");
            return failure();
        }
        if (mclTarget == ANY)
            mOp.emitWarning("Target any which includes fpga was selected, but fpga will not be used because --fpga-kernel-name wasn't specified and the host code contains more than one kernel");
    }
    const auto mclOps = declareMclFuncs(main, rewriter);
    if (failed(insertMCLPrgLoadCall(main, int64_t(mclTarget), clPrgName, rewriter)))
        return failure();
    if (failed(insertMCLInitCall(main, mclWorkers, rewriter)))
        return failure();
    for (auto rootKernelCallOp : kernelModules) {
        if (kernelModules.size() > 1)
            if (mclTarget == FPGA && rootKernelCallOp.getKernelName().str() != fpgaKernelName)
                continue;
        auto func = rootKernelCallOp->getParentOfType<FuncOp>();
        Value mclHandle = insertMCLTaskCreateCall(rootKernelCallOp, mclOps.lookup(kMCLCreateTaskFname), rewriter).getResult();
        if (failed(insertMCLSetKernelCall(rootKernelCallOp, mclHandle, mclOps.lookup(kMCLSetKernelFname), rewriter)))
            return failure();
        if (failed(insertMCLSetArgCalls(rootKernelCallOp, mclHandle, mclOps.lookup(kMCLSetArgFname), rewriter)))
            return failure();
        if (failed(insertMCLExecCall(rootKernelCallOp, int64_t(mclTarget), mclHandle, mclOps.lookup(kMCLExecFname), rewriter)))
            return failure();
        if (failed(insertMCLWaitCall(rootKernelCallOp, mclOps.lookup(kMCLWaitFname), func, mclHandle, rewriter)))
            return failure();
        rewriter.replaceOp(rootKernelCallOp, ValueRange());
    }
    return success();
}

namespace {
    struct InsertMCLCallsPass : mlir::impl::InsertMCLCallsBase<InsertMCLCallsPass> {
        void runOnOperation() override {
            MLIRContext *context = &getContext();
            ModuleOp module = getOperation();
            RewritePatternSet patterns(context);
            LLVMConversionTarget llvmTarget(*context);
            llvmTarget.addLegalDialect<scf::SCFDialect, memref::MemRefDialect, func::FuncDialect, linalg::LinalgDialect, arith::ArithDialect>();
            patterns.insert(&insertMCLCalls);
            gpu::LaunchFuncOp kernel = nullptr;
            module.walk([&kernel](gpu::LaunchFuncOp op) {
                kernel = op;
            });
            if (kernel == nullptr)
                return signalPassFailure();
            if (failed(applyPartialConversion(kernel, llvmTarget, std::move(patterns))))
                return signalPassFailure();
            for (auto gpuModule :
                    llvm::make_early_inc_range(module.getOps<gpu::GPUModuleOp>()))
                gpuModule.erase();
        };
    };
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createInsertMCLCallsPass() {
    return std::make_unique<InsertMCLCallsPass>();
}
