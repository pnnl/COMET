#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include "comet/Dialect/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

mlir::LogicalResult specializeGpuHost(mlir::OpBuilder& builder, mlir::ModuleOp modOp, std::string vendor_prefix);
void declare_vendor_funcs(mlir::OpBuilder& builder, mlir::ModuleOp modOp, std::string vendor_prefix);
mlir::LogicalResult specializeGpuKernel(mlir::OpBuilder& builder, mlir::ModuleOp modOp, mlir::tensorAlgebra::GPUCompilationFormat codeFormat, mlir::Attribute target, std::function<bool (mlir::ModuleOp& mod)> add_ttir_passes, std::function<bool (mlir::ModuleOp& mod)> add_ttgir_passes, std::function<bool (mlir::ModuleOp& mod)> add_llir_passes);

#endif