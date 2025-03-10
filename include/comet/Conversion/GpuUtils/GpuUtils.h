#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

mlir::LogicalResult specializeGpuHost(mlir::OpBuilder& builder, mlir::ModuleOp modOp, std::string vendor_prefix);
void declare_vendor_funcs(mlir::OpBuilder& builder, mlir::ModuleOp modOp, std::string vendor_prefix);

#endif