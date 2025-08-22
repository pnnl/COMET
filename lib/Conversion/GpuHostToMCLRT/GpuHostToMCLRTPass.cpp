
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>

#include "comet/Conversion/GpuHostToMCLRT/GpuHostToMCLRTPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "minos.h"

#define GEN_PASS_CLASSES
#include "comet/Conversion/GpuHostToMCLRT/Passes.h.inc"


static constexpr const char *kMCLInitFname = "mcl_init";
static constexpr const char *kMCLFinitFname = "mcl_finit";
static constexpr const char *kMCLPrgLoadFname = "mcl_prg_load";
static constexpr const char *kMCLCreateTaskFname = "mcl_task_create";
static constexpr const char *kMCLSetKernelFname = "mcl_task_set_kernel";
static constexpr const char *kMCLSetArgFname = "mcl_task_set_arg";
static constexpr const char *kMCLExecFname = "mcl_exec";
static constexpr const char *kMCLWaitFname = "mcl_wait";
static constexpr const char *kMCLWaitAllFname = "mcl_wait_all";

mlir::func::FuncOp declare_function(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, const char* name, mlir::TypeRange inputs, mlir::TypeRange outputs) {

    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), name, builder.getFunctionType(inputs, outputs));
    func.setVisibility(mlir::SymbolTable::Visibility::Private);

    return func;
}


void declare_mcl_funcs(mlir::MLIRContext* ctx, mlir::OpBuilder& builder) {
    mlir::IntegerType u64 = mlir::IntegerType::get(ctx, 64);
    mlir::IntegerType i32 = mlir::IntegerType::get(ctx, 32);
    mlir::LLVM::LLVMPointerType ptr = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::IndexType index = mlir::IndexType::get(ctx);

    declare_function(ctx, builder, kMCLInitFname, {u64, u64}, {i32});
    declare_function(ctx, builder, kMCLFinitFname, {}, {i32});
    declare_function(ctx, builder, kMCLCreateTaskFname, {}, {ptr});
    declare_function(ctx, builder, kMCLPrgLoadFname, {ptr, ptr, u64}, {i32});
    declare_function(ctx, builder, kMCLSetKernelFname, {ptr, ptr, u64}, {i32});
    declare_function(ctx, builder, kMCLSetArgFname, {ptr, u64, ptr, index, u64}, {i32});
    declare_function(ctx, builder, kMCLExecFname, {ptr, ptr, ptr, u64}, {i32});
    declare_function(ctx, builder, kMCLWaitFname, {ptr}, {i32});
    declare_function(ctx, builder, kMCLWaitAllFname, {}, {i32});
}

void insert_mcl_init(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc, mlir::Value num_workers, mlir::Value flags)  
{
    mlir::IntegerType i32 = mlir::IntegerType::get(ctx, 32);
    builder.create<mlir::func::CallOp>(loc, kMCLInitFname, mlir::TypeRange({i32}), mlir::ValueRange({num_workers, flags}));
}

void insert_mcl_finit(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc)  
{
    mlir::IntegerType i32 = mlir::IntegerType::get(ctx, 32);
    builder.create<mlir::func::CallOp>(loc, kMCLFinitFname, mlir::TypeRange({i32}), mlir::ValueRange());
}

mlir::Value insert_task_create(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc)  
{
    mlir::LLVM::LLVMPointerType ptr = mlir::LLVM::LLVMPointerType::get(ctx);
    return builder.create<mlir::func::CallOp>(loc, kMCLCreateTaskFname, mlir::TypeRange({ptr}), mlir::ValueRange()).getResult(0);
}

mlir::Value insert_prg_load_call(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc, const std::string &prgName, mlir::Value flags) 
{
    static int prg_n = 0;

    mlir::IntegerType i32 = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i8 = mlir::IntegerType::get(ctx, 8);
    mlir::LLVM::LLVMArrayType prgNameArrType = mlir::LLVM::LLVMArrayType::get(i8, prgName.size() + 1);
    mlir::LLVM::LLVMArrayType copsType = mlir::LLVM::LLVMArrayType::get(i8, 1);
    
    auto prev = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(builder.getInsertionPoint()->getParentOfType<mlir::ModuleOp>().getBody());

    auto xclbinpath = builder.create<mlir::LLVM::GlobalOp>(loc, prgNameArrType, true, mlir::LLVM::Linkage::Internal, "prg_"+std::to_string(prg_n)+"_path", mlir::StringAttr::get(prgName + '\0', i8));
    // mlir::LLVM::createGlobalString(loc, builder, "prg_"+std::to_string(prg_n)+"_path", prgName, mlir::LLVM::linkage::Linkage::Private );
    // mlir::LLVM::createGlobalString(loc, builder, "cops", "", mlir::LLVM::linkage::Linkage::Private );
    auto cops_global = builder.create<mlir::LLVM::GlobalOp>(loc, copsType, true, mlir::LLVM::Linkage::Internal, "cops", mlir::StringAttr::get(std::string() + '\0', i8));
    builder.restoreInsertionPoint(prev);
    mlir::Value prg_path = builder.create<mlir::LLVM::AddressOfOp>(loc, xclbinpath);
    mlir::Value cops = builder.create<mlir::LLVM::AddressOfOp>(loc, cops_global);
    prg_n++;
    return builder.create<mlir::func::CallOp>(loc, kMCLPrgLoadFname, mlir::TypeRange({i32}), mlir::ValueRange({prg_path, cops, flags})).getResult(0);
}

mlir::Value insert_set_task_kernel_call(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc, mlir::gpu::LaunchFuncOp launchOp, mlir::Value mcl_handle)  
{
    static int kernel_n = 0;
    mlir::IntegerType i8 = mlir::IntegerType::get(ctx, 8);
    mlir::IntegerType i32 = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType u64 = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMArrayType kernelNameArrType = mlir::LLVM::LLVMArrayType::get(i8, launchOp.getKernelName().size() + 1);
    auto prev = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(builder.getInsertionPoint()->getParentOfType<mlir::ModuleOp>().getBody());
    auto kernel_global = builder.create<mlir::LLVM::GlobalOp>(loc, kernelNameArrType, true, mlir::LLVM::Linkage::Internal, "kernel_"+std::to_string(kernel_n)+"_path", mlir::StringAttr::get(launchOp.getKernelName().str() + '\0', i8));
    builder.restoreInsertionPoint(prev);
    // mlir::LLVM::createGlobalString(loc, builder, "kernel_"+std::to_string(kernel_n)+"_str", launchOp.getKernelName(), mlir::LLVM::linkage::Linkage::Private );
    mlir::Value kernel_str = builder.create<mlir::LLVM::AddressOfOp>(loc, kernel_global);
    mlir::Value num_args = builder.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(u64, launchOp.getNumKernelOperands()));
    kernel_n++;
    return builder.create<mlir::func::CallOp>(loc, kMCLSetKernelFname, mlir::TypeRange({i32}), mlir::ValueRange({mcl_handle, kernel_str,num_args})).getResult(0);
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

void insert_set_task_arg_calls(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc, mlir::Value mcl_handle, mlir::gpu::LaunchFuncOp launchOp, mlir::gpu::GPUFuncOp gpuFuncOp)  
{
    mlir::IntegerType i32 = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType u64 = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType ptr = mlir::LLVM::LLVMPointerType::get(ctx);
    for(size_t i = 0; i < launchOp.getNumKernelOperands(); i++)
    {
        mlir::Value arg = launchOp.getKernelOperand(i);
        mlir::Value arg_id = builder.create<mlir::arith::ConstantOp>(loc,  mlir::IntegerAttr::get(u64, i));
        mlir::Type element_type;
        mlir::Value size_in_bytes;
        mlir::Value element_type_size;
        mlir::Value num_elements;
        uint64_t flag_val = 0;
        mlir::Value flag;
        if(auto memref =  mlir::dyn_cast<mlir::MemRefType>(arg.getType()))
        {
            element_type = memref.getElementType();
            if(memref.hasStaticShape())
            {
                num_elements = builder.create<mlir::arith::ConstantIndexOp>(loc, memref.getNumElements());
            }
            else 
            {
                num_elements = get_memref_num_elements(ctx, builder, loc, arg);
            }
            arg = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, arg);
            arg = builder.create<mlir::arith::IndexCastOp>(loc, u64, arg);
            arg = builder.create<mlir::LLVM::IntToPtrOp>(loc,  ptr, arg);
            flag_val |= MCL_ARG_BUFFER;
        }
        else 
        {
            auto zeroIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto alloca = builder.create<mlir::memref::AllocaOp>(loc, mlir::MemRefType::get({1}, arg.getType()));
            builder.create<mlir::memref::StoreOp>(loc, arg, alloca, mlir::ValueRange({zeroIndex}));
            element_type = arg.getType();
            arg = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, alloca);
            arg = builder.create<mlir::arith::IndexCastOp>(loc, u64, arg);
            arg = builder.create<mlir::LLVM::IntToPtrOp>(loc,  ptr, arg);
            num_elements = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

            flag_val |= MCL_ARG_SCALAR | MCL_ARG_INPUT;
        }
        if(element_type.isIntOrFloat()) 
        {
            element_type_size = builder.create<mlir::arith::ConstantIndexOp>(loc, element_type.getIntOrFloatBitWidth()/8);
        }
        else if(element_type.isIndex())
        {
            auto bits_in_byte = builder.create<mlir::arith::ConstantIndexOp>(loc, 8);
            element_type_size = builder.create<mlir::arith::DivUIOp>(loc, builder.create<mlir::index::SizeOfOp>(loc), bits_in_byte);
        }
        size_in_bytes = builder.create<mlir::arith::MulIOp>(loc, element_type_size, num_elements);
        for(auto& use: gpuFuncOp.getArgument(i).getUses())
        {
            if(auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(use.getOwner()); loadOp && loadOp.getMemRef() == use.get())
            {
                flag_val |= MCL_ARG_INPUT;
            }
            else if(auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(use.getOwner()); storeOp && storeOp.getMemRef() == use.get())
            {
                flag_val |= MCL_ARG_OUTPUT;
            }
        }
        flag = builder.create<mlir::arith::ConstantOp>(loc,  mlir::IntegerAttr::get(u64, flag_val));
        builder.create<mlir::func::CallOp>(loc, kMCLSetArgFname, mlir::TypeRange({i32}), mlir::ValueRange({mcl_handle, arg_id, arg, size_in_bytes, flag}));
    } 
}

void insert_task_exec_calls(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc, mlir::Value mcl_handle, mlir::Value flags, mlir::gpu::LaunchFuncOp launchOp)
{
    mlir::IntegerType u64 = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType ptr = mlir::LLVM::LLVMPointerType::get(ctx);

    auto grid = launchOp.getGridSizeOperandValues();
    mlir::Value grid_sizes[3] = {   builder.create<mlir::arith::IndexCastOp>(loc, u64, grid.x), 
                                    builder.create<mlir::arith::IndexCastOp>(loc, u64, grid.y),   
                                    builder.create<mlir::arith::IndexCastOp>(loc, u64, grid.z)
    };
    
    mlir::Value global_wg_mem = builder.create<mlir::memref::AllocaOp>(loc, mlir::MemRefType::get({3}, u64));
    
    auto block = launchOp.getBlockSizeOperandValues();
    mlir::Value block_sizes[3] = {
                                builder.create<mlir::arith::IndexCastOp>(loc, u64, block.x), 
                                builder.create<mlir::arith::IndexCastOp>(loc, u64, block.y), 
                                builder.create<mlir::arith::IndexCastOp>(loc, u64, block.z)
    };

    mlir::Value block_mem = builder.create<mlir::memref::AllocaOp>(loc, mlir::MemRefType::get({3}, u64));
    mlir::Value indices[3] = {builder.create<mlir::arith::ConstantIndexOp>(loc, 0), builder.create<mlir::arith::ConstantIndexOp>(loc, 1), builder.create<mlir::arith::ConstantIndexOp>(loc, 2)};

    for(size_t i = 0; i < 3; i++)
    {
        mlir::Value global_index = builder.create<mlir::arith::MulIOp>(loc, grid_sizes[i], block_sizes[i]);
        builder.create<mlir::memref::StoreOp>(loc, global_index, global_wg_mem, indices[i]);
        builder.create<mlir::memref::StoreOp>(loc, block_sizes[i], block_mem, indices[i]);
    }

    mlir::IntegerType i32 = mlir::IntegerType::get(ctx, 32);
    mlir::Value global_wg_mem_ptr = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, global_wg_mem);
    global_wg_mem_ptr = builder.create<mlir::arith::IndexCastOp>(loc, u64, global_wg_mem_ptr);
    global_wg_mem_ptr = builder.create<mlir::LLVM::IntToPtrOp>(loc,  ptr, global_wg_mem_ptr);
    mlir::Value block_mem_ptr = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, block_mem);
    block_mem_ptr = builder.create<mlir::arith::IndexCastOp>(loc, u64, block_mem_ptr);
    block_mem_ptr = builder.create<mlir::LLVM::IntToPtrOp>(loc,  ptr, block_mem_ptr);
    builder.create<mlir::func::CallOp>(loc, kMCLExecFname, mlir::TypeRange({i32}), mlir::ValueRange({mcl_handle, global_wg_mem_ptr, block_mem_ptr, flags}));
}  

void insert_task_wait_calls(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc, mlir::Value mcl_handle) 
{
    mlir::IntegerType i32 = mlir::IntegerType::get(ctx, 32);
    // builder.create<mlir::func::CallOp>(loc, kMCLWaitFname, mlir::TypeRange({i32}), mlir::ValueRange({mcl_handle}));
    builder.create<mlir::func::CallOp>(loc, kMCLWaitAllFname, mlir::TypeRange({i32}), mlir::ValueRange());
}



class ConvertGpuHostToMCLRT
    : public ConvertGpuHostToMCLRTPassBase<ConvertGpuHostToMCLRT> {
public:
    ConvertGpuHostToMCLRT() = default;
    ConvertGpuHostToMCLRT(const char* xclbin_path)
    {
        this->xclbin_path = xclbin_path; 
    }

    void runOnOperation() override {
        mlir::MLIRContext* ctx = &getContext();
        mlir::ModuleOp module = getOperation();
        llvm::SmallVector<mlir::gpu::LaunchFuncOp, 3> launch_ops;
        llvm::SmallMapVector<mlir::StringAttr, mlir::gpu::GPUFuncOp, 4> gpu_func_ops;
        mlir::IntegerType u64 = mlir::IntegerType::get(ctx, 64);

        module->walk([&launch_ops](mlir::gpu::LaunchFuncOp launch_op) {
            launch_ops.push_back(launch_op);
        });
        auto gpu_modules = module.getOps<mlir::gpu::GPUModuleOp>();
        
        for(auto gpu_module: gpu_modules)
        {
            gpu_module.walk([&gpu_func_ops](mlir::gpu::GPUFuncOp gpu_func) {
                gpu_func_ops.insert(std::make_pair(gpu_func.getNameAttr(), gpu_func));
            });

        }


        mlir::OpBuilder builder(module);
        builder.setInsertionPointToStart(module.getBody());
        declare_mcl_funcs(ctx, builder);
        mlir::func::FuncOp parent_func_op = nullptr;


        for(auto launch_op: launch_ops)
        {
            if(parent_func_op == nullptr)
            {
                parent_func_op = launch_op->getParentOfType<mlir::func::FuncOp>();
            }
            else 
            {
                assert(parent_func_op == launch_op->getParentOfType<mlir::func::FuncOp>());
            }
            mlir::Location loc = launch_op->getLoc();
            builder.setInsertionPoint(launch_op);
            mlir::Value target = builder.create<mlir::arith::ConstantOp>(launch_op->getLoc(), mlir::IntegerAttr::get(u64, MCL_TASK_FPGA));
            insert_prg_load_call(ctx, builder, loc, std::string(this->xclbin_path), target);
            mlir::Value mcl_handle = insert_task_create(ctx, builder, loc);
            insert_set_task_kernel_call(ctx, builder, loc, launch_op, mcl_handle);
            insert_set_task_arg_calls(ctx, builder, loc, mcl_handle, launch_op, gpu_func_ops[launch_op.getKernelName()]);
            insert_task_exec_calls(ctx, builder, loc, mcl_handle, target, launch_op);
            insert_task_wait_calls(ctx, builder, loc, mcl_handle);
            launch_op->erase();
        }
        if(parent_func_op)
        {
            builder.setInsertionPointToStart(&parent_func_op.getBody().front());
            mlir::Value num_workers = builder.create<mlir::arith::ConstantOp>(parent_func_op->getLoc(), mlir::IntegerAttr::get(u64,1));
            mlir::Value flags = builder.create<mlir::arith::ConstantOp>(parent_func_op->getLoc(), mlir::IntegerAttr::get(u64,0));
            insert_mcl_init(ctx, builder, parent_func_op->getLoc(), num_workers, flags);
            auto return_op = *parent_func_op.getOps<mlir::func::ReturnOp>().begin();
            builder.setInsertionPoint(return_op);
            insert_mcl_finit(ctx, builder, parent_func_op->getLoc());
        }

        // for(auto func_op: gpu_func_ops)
        // {
        //     func_op.second->erase();
        // }

        // for(auto gpu_module: llvm::make_early_inc_range(gpu_modules))
        // {
        //     gpu_module->erase();
        // }
    }
};


std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::comet::createConvertGpuHostToMCLRTPass() 
{
    return std::make_unique<::ConvertGpuHostToMCLRT>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::comet::createConvertGpuHostToMCLRTPass(const char* xclbin_path) 
{
    return std::make_unique<::ConvertGpuHostToMCLRT>(xclbin_path);
}