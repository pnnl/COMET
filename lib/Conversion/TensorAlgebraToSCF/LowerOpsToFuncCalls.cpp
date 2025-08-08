    
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace mlir;

class TensorSortOpLowering : public ConversionPattern
{
  public:
    explicit TensorSortOpLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::TensorSortOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto destMemref = op->getOperand(0); 
    auto inputType = op->getOperand(0).getType();
    ShapedType shaped_type = cast<ShapedType>(inputType);

    /// Declare comet_sort_index()
    Type elemType = shaped_type.getElementType();
    Type indexType = rewriter.getIndexType();

    auto sort_func = FunctionType::get(ctx, {UnrankedMemRefType::get(elemType, 0), indexType, indexType}, {});
    std::string func_name = "comet_sort";
    if(auto integer = dyn_cast<IntegerType>(elemType))
    {
      func_name +=  std::to_string(integer.getIntOrFloatBitWidth());
    }

    if (!tensorAlgebra::hasFuncDeclaration(module, func_name))
    {
      func::FuncOp func_declare = func::FuncOp::create(loc,
                                                      func_name,
                                                      sort_func,
                                                      ArrayRef<NamedAttribute>{});
      func_declare.setPrivate();
      module.push_back(func_declare);
    }
    
    
    auto unrankedMemrefType = mlir::UnrankedMemRefType::get(shaped_type.getElementType(), 0);
    auto unrankedMemref = rewriter.create<memref::CastOp>(loc, unrankedMemrefType, destMemref);
    std::vector<mlir::Value> new_operands;
    new_operands.push_back(unrankedMemref);
    auto all_but_first = operands.drop_front();
    new_operands.insert(new_operands.end(), all_but_first.begin(), all_but_first.end());
    rewriter.replaceOpWithNewOp<func::CallOp>(op, func_name, SmallVector<Type, 2>{}, new_operands);

      return success();
    }
};



namespace
{
  struct LowerOpsToFuncCallsPass
      : public PassWrapper<LowerOpsToFuncCallsPass, OperationPass<mlir::ModuleOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOpsToFuncCallsPass)
    void runOnOperation() override
    {
        ConversionTarget target(getContext());
        target.addLegalDialect<mlir::func::FuncDialect, 
            scf::SCFDialect,
            arith::ArithDialect,
            memref::MemRefDialect,
            bufferization::BufferizationDialect>();
        
        target.addIllegalOp<tensorAlgebra::TensorSortOp>();
        RewritePatternSet patterns(&getContext());
        patterns.insert<TensorSortOpLowering>(&getContext());

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        {
            signalPassFailure();
        }
    }
  };
}


std::unique_ptr<mlir::Pass> mlir::comet::createLowerOpsToFuncCallsPass()
{
  return std::make_unique<LowerOpsToFuncCallsPass>();
}
