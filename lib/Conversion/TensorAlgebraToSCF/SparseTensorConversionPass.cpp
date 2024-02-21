#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h"
#include "mlir/Pass/Pass.h"

/** Derived from mlir upstream SparseTensorConversion pass */
/** Significant credit goes to all those authors. */

using namespace mlir;
using namespace mlir::tensorAlgebra;

/** Helper structures to turn sparse tensor into pointers */
struct {
  Value pos_size;
  Value pos;
  Value crd_size;
  Value crd;

  bool has_tile;
  Value tile_pos_size;
  Value tile_pos;
  Value tile_crd_size;
  Value tile_crd;  
} Dimension;

struct {
  Dimension* dims;
  Value val_size;
  Value vals;
} SparseTensor;

static void parse_sparse_tensor_args(SparseTensorType type, ArrayRef<Value> args, SpraseTensor& result)
{
  /** Helper function to turn arguments from an unrealized cast to sparse tensor */
  return;
}

namespace {
struct ConvertSpTensorInsertOp
    : public OpRewritePattern<TensorInsertOp> {
  using OpRewritePattern<TensorInsertOp>::OpRewritePattern;
  ConvertTensorInsertOp(MLIRContext *context)
      : OpRewritePattern(context) {}
  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return success();
  }
};

struct ConvertWorkspaceTensorInsertOp
    : public OpRewritePattern<TensorInsertOp> {
  using OpRewritePattern<TensorInsertOp>::OpRewritePattern;
  ConvertTensorInsertOp(MLIRContext *context)
      : OpRewritePattern(context) {}
  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return success();
  }
};

}


struct SparseTensorConversionPass
    : public impl::SparseTensorConversionPassBase<SparseTensorConversionPass> {
  SparseTensorConversionPass() = default;
  SparseTensorConversionPass(const SparseTensorConversionPass &pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    TypeConverter typeConverter;
    ConversionTarget target(*ctx);
    // Everything in the sparse dialect must go!
    target.addIllegalDialect<tensorAlgebra::TADialect>();
    
    // The following operations and dialects may be introduced by the
    // rewriting rules, and are therefore marked as legal.
    target.addLegalOp<tensor::ExtractOp, tensor::InsertOp>();
    target.addLegalDialect<
        arith::ArithDialect, bufferization::BufferizationDialect,
        LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect>();

    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
      [](tensorAlgebra::SparseTensorType type, SmallVectorImpl<Type> &types) {
        ArrayRef<int> dim_sizes = type.getDims();
        ArrayRef<unsigned> format = type.getFormat();

        auto context = type.getContext();
        Type index_type = IndexType::get(context);
        bool is_known_size = true;
        int known_size = 1;
        for(unsigned i = 0; i < dims_sizes.size()) {
          switch((TensorFormatEnum)format[i])
          {
            case TensorFormatEnum::D:
            {
              if(dim_sizes[i] != ShapedType::kDynamic) {
                known_size *= dim_sizes[i]; 
              } else {
                is_known_size = false;
              }
              auto pos_type = mlir::RankedTensorType::get({1,}, builder.getI32Type());
              types.push_back(pos_type); //Pos tensor
              types.push_back(index_type); //Pos size
              types.push_back(index_type); //Dimension size
            }
            case TensorFormatEnum::CU:
            case TensorFormatEnum::CN:
            {
              Type pos_type;
              if(is_known_size) {
                pos_type = mlir::RankedTensorType::get({known_size,}, builder.getI32Type());
              } else {
                pos_type = mlir::RankedTensorType::get({ShapedType::kDynamic,}, 
                                                        builder.getI32Type());
              }
              crd_type = mlir::RankedTensorType::get({ShapedType::kDynamic,}, 
                                                      builder.getI32Type());
              is_known_size = false;

              types.push_back(pos_type); //Pos tensor
              types.push_back(crd_type); //Crd tensor
              types.push_back(index_type); //Pos size
              types.push_back(index_type); //Crd size
              types.push_back(index_type); //Dimension size
            }
            case TensorFormatEnum::S:
            {
              crd_type = mlir::RankedTensorType::get({ShapedType::kDynamic,}, builder.getI32Type());
              types.push_back(crd_type); //Crd tensor
              types.push_back(index_type); //Crd size
              types.push_back(index_type); //Dimension size
            }
          }
        }
        Type element_type = type.getElementType();
        Type value_type = mlir::RankedTensorType::get({ShapedType::kDynamic,}, element_type);
        types.push_back(value_type); //Value tensor
        types.push_back(index_type); //Value size
        return success();
      });

    typeConverter.addArgumentMaterialization(
      [](OpBuilder &builder, SpraseTensorType resultType, ValueRange inputs,
          Location loc) -> Optional<Value> {
        Value value = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs);
        return value;
      });

    // Populate with rules and apply rewriting rules.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    populateSparseTensorConversionPatterns(converter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
