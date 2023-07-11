#ifndef COMET_TRANSFORMS_PASSES_H
#define COMET_TRANSFORMS_PASSES_H

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "minos.h"

namespace {
    enum Targets : int64_t
    {
        CPU  = MCL_TASK_CPU,
        GPU  = MCL_TASK_GPU,
        FPGA = MCL_TASK_FPGA,
        ANY  = MCL_TASK_ANY
    };
}

namespace mlir {
    class ModuleOp;
    class PatternRewriter;
    class DominanceInfo;

#define GEN_PASS_DECL
#include "comet/Transforms/CometTransforms.h.inc"

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createInsertMCLCallsPass();
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createGPUToOCLSPIRVPass();
    std::unique_ptr<Pass> createRaiseSCFWhileToForPass();

    // from https://github.com/llvm/Polygeist/blob/main/include/polygeist/Passes/Passes.h
    std::unique_ptr<Pass> createRaiseSCFToAffinePass();
    std::unique_ptr<Pass> replaceAffineCFGPass();

#define GEN_PASS_REGISTRATION
#include "comet/Transforms/CometTransforms.h.inc"

} // namespace mlir

void fully2ComposeAffineMapAndOperands(
        mlir::PatternRewriter &rewriter, mlir::AffineMap *map,
        llvm::SmallVectorImpl<mlir::Value> *operands, mlir::DominanceInfo &DI);
bool isValidIndex(mlir::Value val);

// from https://github.com/llvm/Polygeist/blob/main/include/polygeist/Ops.h
struct ValueOrInt {
    bool isValue;
    mlir::Value v_val;
    int64_t i_val;
    ValueOrInt(mlir::Value v) { initValue(v); }
    void initValue(mlir::Value v) {
        using namespace mlir;
        if (v) {
            IntegerAttr iattr;
            if (matchPattern(v, m_Constant(&iattr))) {
                i_val = iattr.getValue().getSExtValue();
                v_val = nullptr;
                isValue = false;
                return;
            }
        }
        isValue = true;
        v_val = v;
    }
    ValueOrInt(size_t i) : isValue(false), v_val(), i_val(i) {}

    bool operator>=(int64_t v) {
        if (isValue)
            return false;
        return i_val >= v;
    }
    bool operator>(int64_t v) {
        if (isValue)
            return false;
        return i_val > v;
    }
    bool operator==(int64_t v) {
        if (isValue)
            return false;
        return i_val == v;
    }
    bool operator<(int64_t v) {
        if (isValue)
            return false;
        return i_val < v;
    }
    bool operator<=(int64_t v) {
        if (isValue)
            return false;
        return i_val <= v;
    }
    bool operator>=(llvm::APInt v) {
        if (isValue)
            return false;
        return i_val >= v.getSExtValue();
    }
    bool operator>(llvm::APInt v) {
        if (isValue)
            return false;
        return i_val > v.getSExtValue();
    }
    bool operator==(llvm::APInt v) {
        if (isValue)
            return false;
        return i_val == v.getSExtValue();
    }
    bool operator<(llvm::APInt v) {
        if (isValue)
            return false;
        return i_val < v.getSExtValue();
    }
    bool operator<=(llvm::APInt v) {
        if (isValue)
            return false;
        return i_val <= v.getSExtValue();
    }
};

enum class Cmp { EQ, LT, LE, GT, GE };

bool valueCmp(Cmp cmp, mlir::AffineExpr expr, size_t numDim,
              mlir::ValueRange operands, ValueOrInt val);

bool valueCmp(Cmp cmp, mlir::Value bval, ValueOrInt val);

bool isReadOnly(mlir::Operation *op);

#endif // COMET_TRANSFORMS_PASSES_H
