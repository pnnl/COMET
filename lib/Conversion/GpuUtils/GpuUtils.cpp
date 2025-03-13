#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <functional>
#include <map>
using namespace mlir;

mlir::LogicalResult specializeGpuHost(mlir::OpBuilder& builder, mlir::ModuleOp modOp, std::string vendor_prefix)
{
    std::map<std::string, Value> funcs;
    std::vector<mlir::gpu::LaunchFuncOp> launchOps;
    modOp->walk([&launchOps](mlir::gpu::LaunchFuncOp launchOp) {
      launchOps.push_back(launchOp);
    });

    std::set<mlir::func::FuncOp> initFuncs;
    std::vector<Value> toAlloc;

    for (auto launchOp : launchOps) {

      if (initFuncs.find(launchOp->getParentOfType<mlir::func::FuncOp>()) ==
          initFuncs.end()) {
        builder.setInsertionPointToStart(
            &launchOp->getParentOfType<mlir::func::FuncOp>()
                 .getFunctionBody()
                 .front());
        Value gpu_code = builder.create<mlir::LLVM::AddressOfOp>(
            launchOp->getLoc(), LLVM::LLVMPointerType::get(builder.getContext()),
            "gpu_code");
        builder.create<mlir::func::CallOp>(launchOp.getLoc(),
                                           vendor_prefix+"SetModuleImage", TypeRange(),
                                           ValueRange({gpu_code}));

        initFuncs.insert(launchOp->getParentOfType<mlir::func::FuncOp>());
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
        allocSize = mlir::tensorAlgebra::get_memref_num_elements(builder.getContext(), builder, gpuAlloc.getLoc(), gpuAlloc.getMemref());
      }

      // builder.create<memref::DimOp>(gpuAlloc.getLoc(), gpuAlloc.getMemref(),
      // 0);
      if (FloatType floatType = mlir::dyn_cast<FloatType>(
              gpuAlloc.getMemref().getType().getElementType())) {
        int width = floatType.getWidth();
        auto cudaOp = builder.create<mlir::func::CallOp>(
            gpuAlloc->getLoc(), vendor_prefix+"MallocF" + std::to_string(width),
            TypeRange(builder.getIndexType()), ValueRange(allocSize));
        gpuAlloc->replaceAllUsesWith(cudaOp);
      } else if (IntegerType intType = mlir::dyn_cast<IntegerType>(
                     gpuAlloc.getMemref().getType().getElementType())) {
        int width = intType.getWidth();
        auto cudaOp = builder.create<mlir::func::CallOp>(
            gpuAlloc->getLoc(), vendor_prefix+"MallocI" + std::to_string(width),
            TypeRange(builder.getIndexType()), ValueRange(allocSize));
        gpuAlloc->replaceAllUsesWith(cudaOp);
      } else if (IndexType intType = mlir::dyn_cast<IndexType>(
                     gpuAlloc.getMemref().getType().getElementType())) {
        auto cudaOp = builder.create<mlir::func::CallOp>(
            gpuAlloc->getLoc(), vendor_prefix+"MallocI" + std::to_string(64),
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
               .starts_with(vendor_prefix+"Malloc"))) {
        if(memref_to_collapsed_memref.find(mlir::cast<TypedValue<BaseMemRefType>>(cpy.getSrc())) == memref_to_collapsed_memref.end())
        {
          TypedValue<MemRefType> collapsedMemref = mlir::tensorAlgebra::collapseMemref(cpy.getSrc(), builder);
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
              cpy->getLoc(), vendor_prefix+"MemcpyI" + std::to_string(width), TypeRange(),
              ValueRange({cpy.getOperand(0), cast, hToD}));
        } else if (FloatType floatType = mlir::dyn_cast<FloatType>(
                       mlir::cast<MemRefType>(cpy.getOperand(1).getType())
                           .getElementType())) {
          int width = floatType.getWidth();
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), vendor_prefix+"MemcpyF" + std::to_string(width), TypeRange(),
              ValueRange({cpy.getOperand(0), cast, hToD}));
        } else if (IndexType indexType = mlir::dyn_cast<IndexType>(
                       mlir::cast<MemRefType>(cpy.getOperand(1).getType())
                           .getElementType())) {
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), vendor_prefix+"MemcpyIndex", TypeRange(),
              ValueRange({cpy.getOperand(0), cast, hToD}));
        }
      } else if (cpy.getOperand(1).getDefiningOp() &&
                 (isa<mlir::func::CallOp>(cpy.getOperand(1).getDefiningOp()) &&
                  cast<mlir::func::CallOp>(cpy.getOperand(1).getDefiningOp())
                      .getCallee()
                      .starts_with(vendor_prefix+"Malloc"))) {
        if(memref_to_collapsed_memref.find(mlir::cast<TypedValue<BaseMemRefType>>(cpy.getDst())) == memref_to_collapsed_memref.end())
        {
          TypedValue<MemRefType> collapsedMemref = mlir::tensorAlgebra::collapseMemref(cpy.getDst(), builder);
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
              cpy->getLoc(), vendor_prefix+"MemcpyI" + std::to_string(width), TypeRange(),
              ValueRange({cpy.getOperand(1), cast, dToH}));
        } else if (FloatType floatType = mlir::dyn_cast<FloatType>(
                       mlir::cast<MemRefType>(cpy.getOperand(0).getType())
                           .getElementType())) {
          int width = floatType.getWidth();
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), vendor_prefix+"MemcpyF" + std::to_string(width), TypeRange(),
              ValueRange({cpy.getOperand(1), cast, dToH}));
        } else if (IndexType indexType = mlir::dyn_cast<IndexType>(
                       mlir::cast<MemRefType>(cpy.getOperand(0).getType())
                           .getElementType())) {
          builder.create<mlir::func::CallOp>(
              cpy->getLoc(), vendor_prefix+"MemcpyIndex", TypeRange(),
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
      
      std::string funcName = "tt_"+ launchOp.getKernelModuleName().str() + launchOp.getKernelName().strref().str();
      if (funcs.find(funcName) == funcs.end()) {
        // We need the global string to include the \0 character so that it is
        // correctly read by the cuda lib Hence the StringRef(funcName.c_str(),
        // funcName.size()+1)
        funcs[funcName] = LLVM::createGlobalString(
            modOp->getLoc(), builder, funcName + "_str",
            StringRef(funcName.c_str(), funcName.size() + 1),
            LLVM::linkage::Linkage::Private);
      }
      builder.create<mlir::func::CallOp>(
          launchOp->getLoc(), vendor_prefix+"LaunchKernel", TypeRange(),
          ValueRange({launchOp.getGridSizeX(), launchOp.getGridSizeY(),
                      launchOp.getGridSizeZ(), launchOp.getBlockSizeX(),
                      launchOp.getBlockSizeY(), launchOp.getBlockSizeZ(),
                      args_dynamic, funcs[funcName],
                      builder.create<arith::ConstantIndexOp>(launchOp->getLoc(),
                                                             funcName.size()),
                      launchOp.getDynamicSharedMemorySize()}));
      launchOp->erase();
    }

    std::vector<mlir::gpu::DeallocOp> gpuDeallocs;
    modOp->walk([&gpuDeallocs](mlir::gpu::DeallocOp gpuDeallocOp) {
      gpuDeallocs.push_back(gpuDeallocOp);
    });

    for (auto dealloc : gpuDeallocs) {
      builder.setInsertionPoint(dealloc);
      builder.create<mlir::func::CallOp>(dealloc->getLoc(), vendor_prefix+"Free",
                                         TypeRange(),
                                         ValueRange(dealloc->getOperand(0)));
      dealloc.erase();
    }
    
    func::FuncOp funcOp = *initFuncs.begin();
    builder.setInsertionPoint(funcOp.getBody().front().getTerminator());

    builder.create<mlir::func::CallOp>(funcOp.getBody().front().getTerminator()->getLoc(), vendor_prefix+"Finit", TypeRange(), ValueRange());

    modOp->walk([](mlir::triton::FuncOp TTFuncOp) { TTFuncOp.erase(); });
    modOp->walk([](mlir::gpu::GPUModuleOp gpuMod) { gpuMod.erase(); });

    return success();
}


void declare_vendor_funcs(OpBuilder& builder, ModuleOp modOp, std::string vendor_prefix)
{
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
    builder.getUnknownLoc(), vendor_prefix+"MallocI32", mallocI32Type);
  mallocI32.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(mallocI32);

  auto mallocI64Type = builder.getFunctionType({builder.getIndexType()},
                                            builder.getIndexType());
  auto mallocI64 = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"MallocI64", mallocI64Type);
  mallocI64.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(mallocI64);

  auto mallocF32Type = builder.getFunctionType({builder.getIndexType()},
                                            builder.getIndexType());
  auto mallocF32 = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"MallocF32", mallocF32Type);
  mallocF32.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(mallocF32);

  auto mallocF64Type = builder.getFunctionType({builder.getIndexType()},
                                            builder.getIndexType());
  auto mallocF64 = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"MallocF64", mallocF64Type);
  mallocF64.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(mallocF64);

  auto cudaMemcpyI32Type = builder.getFunctionType(
    {builder.getIndexType(), memrefI32, builder.getIndexType()}, {});
  auto cudaMemcpyI32 = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"MemcpyI32", cudaMemcpyI32Type);
  cudaMemcpyI32.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(cudaMemcpyI32);

  auto cudaMemcpyI64Type = builder.getFunctionType(
    {builder.getIndexType(), memrefI64, builder.getIndexType()}, {});
  auto cudaMemcpyI64 = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"MemcpyI64", cudaMemcpyI64Type);
  cudaMemcpyI64.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(cudaMemcpyI64);

  auto cudaMemcpyIndexType = builder.getFunctionType(
    {builder.getIndexType(), memrefIndex, builder.getIndexType()}, {});
  auto cudaMemcpyIndex = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"MemcpyIndex", cudaMemcpyIndexType);
  cudaMemcpyIndex.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(cudaMemcpyIndex);

  auto cudaMemcpyF32Type = builder.getFunctionType(
    {builder.getIndexType(), memrefF32, builder.getIndexType()}, {});
  auto cudaMemcpyF32 = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"MemcpyF32", cudaMemcpyF32Type);
  cudaMemcpyF32.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(cudaMemcpyF32);

  auto cudaMemcpyF64Type = builder.getFunctionType(
    {builder.getIndexType(), memrefF64, builder.getIndexType()}, {});
  auto cudaMemcpyF64 = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"MemcpyF64", cudaMemcpyF64Type);
  cudaMemcpyF64.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(cudaMemcpyF64);

  auto cudaLaunchKernelT = builder.getFunctionType(
    {builder.getIndexType(), builder.getIndexType(), builder.getIndexType(),
    builder.getIndexType(), builder.getIndexType(), builder.getIndexType(),
    MemRefType::get({ShapedType::kDynamic}, builder.getIndexType()),
    LLVM::LLVMPointerType::get(builder.getContext()),
    builder.getIndexType(), builder.getIntegerType(32)},
    {});
  auto cudaLaunchKernel = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"LaunchKernel", cudaLaunchKernelT);
  cudaLaunchKernel.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(cudaLaunchKernel);

  auto cudaSetModuleImageT = builder.getFunctionType(
    {LLVM::LLVMPointerType::get(builder.getContext())}, {});
  auto cudaSetModuleImage = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), vendor_prefix+"SetModuleImage", cudaSetModuleImageT);
  cudaSetModuleImage.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(cudaSetModuleImage);

  auto cudaFreeT = builder.getFunctionType({builder.getIndexType()}, {});
  auto cudaFree = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                  vendor_prefix+"Free", cudaFreeT);
  cudaFree.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(cudaFree);
  auto cudaFinitT = builder.getFunctionType({}, {});
  auto cudaFinit = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), vendor_prefix+"Finit", cudaFinitT);
  cudaFinit.setVisibility(mlir::SymbolTable::Visibility::Private);
  modOp.push_back(cudaFinit);
}

mlir::LogicalResult specializeGpuKernel(mlir::OpBuilder& builder, mlir::ModuleOp modOp, mlir::tensorAlgebra::GPUCompilationFormat codeFormat, Attribute target, std::function<bool (ModuleOp& mod)> add_ttir_passes, std::function<bool (ModuleOp& mod)> add_ttgir_passes, std::function<bool (ModuleOp& mod)> add_llir_passes)
{
  std::vector<mlir::triton::FuncOp> TTFuncs;
  modOp->walk([&TTFuncs](mlir::triton::FuncOp op) { TTFuncs.push_back(op); });

  auto tempMod = builder.create<mlir::gpu::GPUModuleOp>(modOp->getLoc(), "gpu_module", target);
  std::vector<ModuleOp> tempMods;
  if (TTFuncs.empty()) {
    return failure();
  }
  for (auto ttFunc : TTFuncs) {
    auto tempMod = ModuleOp::create(modOp.getLoc());
    OpBuilder builder(tempMod.getBodyRegion());
    builder.clone(*ttFunc.getOperation());

    if (!add_ttir_passes(tempMod)) {
      return failure();
    }

    if (!add_ttgir_passes(tempMod)) {
      return failure();
    }

    if (!add_llir_passes(tempMod)) {
      return failure();
    }

    auto oldAttrs = ttFunc->getAttrs().vec();

    oldAttrs.insert(oldAttrs.end(), tempMod->getAttrs().begin(),
                    tempMod->getAttrs().end());
    // ttFunc->setAttrs(tempMod->getAttrs());
    ttFunc->setAttrs(oldAttrs);
    tempMods.push_back(tempMod);
  }

  llvm::SmallMapVector<StringRef, bool, 4> glob_symbols;
  // We have lowered Triton kernels to LLVM, now we need to add them to the
  // gpu.module
  builder.setInsertionPointToStart(&tempMod.getBodyRegion().front());
  for (auto mod : tempMods) {
    for (auto &op : *mod.getBody()) {
      // Triton will insert one "global_smem" symbol per module so we have to only keep the first
      if(LLVM::GlobalOp glob = mlir::dyn_cast<LLVM::GlobalOp>(op))
      {
        if(glob_symbols.find(glob.getSymName()) == glob_symbols.end() )
        {
          glob_symbols[glob.getSymName()] = true;
          builder.clone(op);
        }
        else
        {
          assert(glob.getSymName() == "global_smem");
        }
      }
      else
      {
        builder.clone(op);
      }
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
  if (codeFormat ==
      mlir::tensorAlgebra::GPUCompilationFormat::Assembly) {
    opts = gpu::TargetOptions({}, {}, {}, gpu::CompilationTarget::Assembly);
  } else if (codeFormat ==
             mlir::tensorAlgebra::GPUCompilationFormat::Binary) {
    opts = gpu::TargetOptions({}, {}, {}, gpu::CompilationTarget::Binary);
  } else if (codeFormat ==
             mlir::tensorAlgebra::GPUCompilationFormat::Fatbin) {
    opts = gpu::TargetOptions({}, {}, {}, gpu::CompilationTarget::Fatbin);
  } else {
    assert(false && "Unexpected gpu compilation code format");
  }
  auto res =
      mlir::gpu::transformGpuModulesToBinaries(tempOuterMod, nullptr, opts);
  if (res.failed()) 
  {
    return failure();
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

  return success();
}
