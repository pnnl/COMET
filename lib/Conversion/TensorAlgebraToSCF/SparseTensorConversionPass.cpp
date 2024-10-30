#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"
#include "comet/Conversion/TensorAlgebraToSCF/TensorAlgebraToSCF.h"
#include "comet/Dialect/IndexTree/Patterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include <cstddef>

using namespace mlir;
using namespace mlir::tensorAlgebra;

#define DEBUG_TYPE "sparse_tensor"

namespace mlir {
  namespace comet{
    #define GEN_PASS_DEF_SPARSETENSORCONVERSIONPASS
    #include "comet/Conversion/Passes.h.inc"
  }
}

/** Helper structures to turn sparse tensor into pointers */
struct Dimension {
  Value dim_size;
  Value insert_pos;
  TensorFormatEnum format;

  Value pos;
  Value pos_size;
  Value crd;
  Value crd_size;

  bool has_block;
  Value block_pos;
  Value block_pos_size;
  Value block_crd; 
  Value block_crd_size;
   
};

struct SparseTensor {
  Value dim_sizes;
  SmallVector<Dimension, 3> dims;
  Value vals;
  Value val_size;
};

struct Workspace {
    Value workspace;
    Value mark_value;
    Value mark_array;
    Value num_crds;
    Value crds;
};

static bool unpack_sparse_tensor(Value sparse_tensor, SparseTensor& result)
{
  /** Helper function to turn arguments from an unrealized cast to sparse tensor */
  if (auto cast =
          sparse_tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
    SparseTensorType type = llvm::dyn_cast<SparseTensorType>(sparse_tensor.getType());
    if(!type)
      return false;

    auto format = type.getFormat();
    auto dim_sizes = type.getDims();
    auto cur_arg = cast.getInputs().begin();
    result.dim_sizes = *cur_arg;
    ++cur_arg;

    for(unsigned i = 0; i < dim_sizes.size(); i++){
      Dimension d;
      d.insert_pos = *cur_arg;
      cur_arg++;
      d.format = (TensorFormatEnum)format[2 * i];
      switch(d.format){
        case TensorFormatEnum::D: {
          d.pos = *cur_arg;
          cur_arg++;
          break;
        }
        case TensorFormatEnum::CU:
        case TensorFormatEnum::CN: {
          d.pos = *cur_arg;
          cur_arg++;

          d.crd = *cur_arg;
          cur_arg++;
          break;
        }
        case TensorFormatEnum::S: {
          d.crd = *cur_arg;
          cur_arg++;
          break;
        }
        default: {
          assert(false && "Could not unpack unknown format to sparse tensor.");
        }
      }
      result.dims.push_back(d);
    }
    result.vals = *cur_arg;
    return true;
  }
  else if(auto cast =
          sparse_tensor.getDefiningOp<SparseTensorConstructOp>())
  {
    SparseTensorType type = llvm::dyn_cast<SparseTensorType>(sparse_tensor.getType());
    if(!type)
      return false;
    OpBuilder builder(cast);

    auto dims = cast.getDims();
    auto crd = cast.getCrdIndices();
    auto pos = cast.getPosIndices();
    auto vals = cast.getVals();
    unsigned rank = type.getDims().size();
    result.dim_sizes = dims;
    for(unsigned i = 0; i < rank; i++)
    {
      Dimension d;
      d.format = (TensorFormatEnum) type.getFormat()[2 * i];
      d.insert_pos = builder.create<index::ConstantOp>(cast.getLoc(), builder.getIndexType(), builder.getIndexAttr(0));
      d.pos = pos[i];
      d.crd = crd[i];
      result.dims.push_back(d);
    }

    result.vals = vals;
    return true;
  }

  return false;
}

static void pack_sparse_tensor(SparseTensorType type, SparseTensor& sparse_tensor, SmallVectorImpl<Value>& result)
{
  result.push_back(sparse_tensor.dim_sizes);
  for(Dimension d : sparse_tensor.dims)
  {
    result.push_back(d.insert_pos);
    switch(d.format){
      case TensorFormatEnum::D: {
        result.push_back(d.pos);
        break;
      }
      case TensorFormatEnum::CU:
      case TensorFormatEnum::CN: {
        result.push_back(d.pos);
        result.push_back(d.crd);
        break;
      }
      case TensorFormatEnum::S: {
        result.push_back(d.crd);
        break;
      }
      default: {
          assert(false && "Could not unpack unknown format to sparse tensor.");
        }
    }
  }
  result.push_back(sparse_tensor.vals);
  return;
}

static bool unpack_workspace(Value workspace_val, Workspace& result)
{
  if (auto cast =
          workspace_val.getDefiningOp<UnrealizedConversionCastOp>()) {
    auto cur_arg = cast.getInputs().begin();
    result.workspace = *cur_arg;
    cur_arg++;
    result.mark_value = *cur_arg;
    cur_arg++;
    result.mark_array = *cur_arg;
    cur_arg++;
    result.num_crds = *cur_arg;
    cur_arg++;
    result.crds = *cur_arg;
    return true;
  }
  return false;

}
static void pack_workspace(WorkspaceType type, Workspace& workspace, SmallVectorImpl<Value>& result)
{
  result.push_back(workspace.workspace);
  result.push_back(workspace.mark_value);
  result.push_back(workspace.mark_array);
  result.push_back(workspace.num_crds);
  result.push_back(workspace.crds);
}

namespace {
class ConvertSpTensorConstructOp
    : public OpConversionPattern<SparseTensorConstructOp> {
  using OpConversionPattern<SparseTensorConstructOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SparseTensorConstructOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SparseTensor sp_tensor;
    SparseTensorType sp_tensor_type = llvm::cast<SparseTensorType>(op->getResult(0).getType());
    auto dims = op.getDims();
    auto crd = op.getCrdIndices();
    auto pos = op.getPosIndices();
    auto vals = op.getVals();
    unsigned rank = sp_tensor_type.getDims().size();
    sp_tensor.dim_sizes = dims;
    for(unsigned i = 0; i < rank; i++)
    {
      Dimension d;
      d.format = (TensorFormatEnum) sp_tensor_type.getFormat()[2 * i];
      d.insert_pos = rewriter.create<index::ConstantOp>(op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
      d.pos = pos[i];
      d.crd = crd[i];
      sp_tensor.dims.push_back(d);
    }

    sp_tensor.vals = vals;

    SmallVector<Value, 12> cast_args;
    pack_sparse_tensor(sp_tensor_type, sp_tensor, cast_args);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, sp_tensor_type, cast_args);
    return success();
  }
};

class ConvertSpTensorInsertOp
    : public OpConversionPattern<TensorInsertOp> {
  using OpConversionPattern<TensorInsertOp>::OpConversionPattern;
  ConvertSpTensorInsertOp(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(tensorAlgebra::TensorInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if(!llvm::isa<SparseTensorType>(op.getTensor().getType())){
      return failure();
    }

    SparseTensor sp_tensor;
    TensorInsertOpAdaptor insertAdpator = llvm::cast<TensorInsertOpAdaptor>(adaptor);
    if(!unpack_sparse_tensor(insertAdpator.getTensor(), sp_tensor)) {
      return failure();
    }

    // Match successful!
    auto loc = op.getLoc();
    Type index_type = rewriter.getIndexType();
    unsigned i = 0;
    for(Dimension& dim : sp_tensor.dims) {
      if(dim.format != TensorFormatEnum::D) {
        Value crd_idx = insertAdpator.getPos()[i];
        Value crd = insertAdpator.getCrds()[i];
        Value crd_tensor = dim.crd;
        Value crd_size = dim.crd_size;
        crd_tensor = rewriter.create<tensor::InsertOp>(
          loc,
          crd_tensor.getType(),
          crd,
          crd_tensor,
          crd_idx);
        dim.crd = crd_tensor;

        // TODO: This is wrong if we insert the same crd multiple times but format is CU?
        Value inc = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
        dim.insert_pos = rewriter.create<index::AddOp>(loc, index_type, dim.insert_pos, inc);

        /** TODO: Implement tensor resize */
        /** TODO: Insert into CSR only has to be done once per idx? */
      }
      
      i++;
    }
    Value vals = sp_tensor.vals;
    Value val_idx = insertAdpator.getPos()[insertAdpator.getPos().size() - 1];
    vals = rewriter.create<tensor::InsertOp>(loc, 
                                             vals.getType(),
                                             insertAdpator.getValue(),
                                             vals,
                                             val_idx);
    sp_tensor.vals = vals;
    SparseTensorType sp_tensor_type = llvm::cast<SparseTensorType>(op.getTensor().getType());

    SmallVector<Value, 12> cast_args;
    pack_sparse_tensor(sp_tensor_type, sp_tensor, cast_args);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, sp_tensor_type, cast_args);
    return success();
  }
};

class ConvertSpTensorExtractOp
    : public OpConversionPattern<TensorExtractOp> {
  using OpConversionPattern<TensorExtractOp>::OpConversionPattern;
  ConvertSpTensorExtractOp(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(tensorAlgebra::TensorExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorExtractOpAdaptor extractAdaptor = llvm::cast<TensorExtractOpAdaptor>(adaptor);
    if(!llvm::isa<SparseTensorType>(extractAdaptor.getTensor().getType())){
      return failure();
    }

    SparseTensor sp_tensor;
    if(!unpack_sparse_tensor(extractAdaptor.getTensor(), sp_tensor)) {
      return failure();
    }

    llvm::ScopedPrinter logger{llvm::dbgs()};
    LLVM_DEBUG({
      logger.startLine() << "Unpacked sparse tensor: " << extractAdaptor.getTensor().getDefiningOp<UnrealizedConversionCastOp>() <<  "\n";
    });
    // Match successful!
    auto loc = op.getLoc();
    Type float_type = llvm::cast<TensorType>(sp_tensor.vals.getType()).getElementType();
    Value result = rewriter.create<tensor::ExtractOp>(loc, float_type, sp_tensor.vals, extractAdaptor.getPos());
    rewriter.replaceOp(op, {result});
    return success();
  }
};

class ConvertSpTensorInsertCrd
    : public OpConversionPattern<SpTensorInsertCrd> {
  using OpConversionPattern<SpTensorInsertCrd>::OpConversionPattern;
  ConvertSpTensorInsertCrd(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(SpTensorInsertCrd op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SparseTensor sp_tensor;
    auto opAdaptor = llvm::cast<SpTensorInsertCrdAdaptor>(adaptor);
    if(!unpack_sparse_tensor(opAdaptor.getTensor(), sp_tensor)) {
      return failure();
    }

    // Match successful!
    auto loc = op.getLoc();
    Type index_type = rewriter.getIndexType();
    Dimension& dim = sp_tensor.dims[opAdaptor.getDim()];
    if(dim.format != TensorFormatEnum::D) {
      Value crd_idx = opAdaptor.getIdx();
      Value crd = opAdaptor.getCrd();
      Value crd_tensor = dim.crd;
      Value crd_size = dim.crd_size;
      crd_tensor = rewriter.create<tensor::InsertOp>(loc,
                                                     crd_tensor.getType(),
                                                     crd,
                                                     crd_tensor,
                                                     crd_idx);
      dim.crd = crd_tensor;

      // Update tensor insert state
      Value inc = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
      dim.insert_pos = rewriter.create<index::AddOp>(loc, index_type, dim.insert_pos, inc);
    }

    SparseTensorType sp_tensor_type = llvm::cast<SparseTensorType>(opAdaptor.getTensor().getType());
    SmallVector<Value, 12> cast_args;
    pack_sparse_tensor(sp_tensor_type, sp_tensor, cast_args);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, sp_tensor_type, cast_args);
    return success();
  }
};

class ConvertSpTensorGetDimSize
    : public OpConversionPattern<SpTensorGetDimSize> {
  using OpConversionPattern<SpTensorGetDimSize>::OpConversionPattern;
  ConvertSpTensorGetDimSize(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(SpTensorGetDimSize op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SpTensorGetDimSizeAdaptor tensorAdaptor = llvm::cast<SpTensorGetDimSizeAdaptor>(adaptor);
    SparseTensor sp_tensor;
    if(!unpack_sparse_tensor(tensorAdaptor.getTensor(), sp_tensor)) {
      return failure();
    }
    Value index = rewriter.create<index::ConstantOp>(op->getLoc(), tensorAdaptor.getDim());
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, sp_tensor.dim_sizes, index);
    return success();
  }
};


class ConvertSpTensorGetDimCrd
    : public OpConversionPattern<SpTensorGetDimCrd> {
  using OpConversionPattern<SpTensorGetDimCrd>::OpConversionPattern;
  ConvertSpTensorGetDimCrd(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(SpTensorGetDimCrd op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SpTensorGetDimCrdAdaptor tensorAdaptor = llvm::cast<SpTensorGetDimCrdAdaptor>(adaptor);
    SparseTensor sp_tensor;
    if(!unpack_sparse_tensor(tensorAdaptor.getTensor(), sp_tensor)) {
      return failure();
    }

    if(sp_tensor.dims[tensorAdaptor.getDim()].crd == nullptr)
    {
      auto zero = rewriter.create<index::ConstantOp>(op->getLoc(), 0);
      rewriter.replaceOp(op, rewriter.create<tensor::EmptyOp>(op->getLoc(), RankedTensorType::get({ShapedType::kDynamic,}, rewriter.getIndexType()), ValueRange(zero)));
    }
    else
    {
      rewriter.replaceOp(op, {sp_tensor.dims[tensorAdaptor.getDim()].crd});
    }
    return success();
  }
};


/// TODO: Implement this conversion to actually handle blocks/tiles
class ConvertSpTensorGetDimBlockPos
    : public OpConversionPattern<SpTensorGetDimBlockPos> {
  using OpConversionPattern<SpTensorGetDimBlockPos>::OpConversionPattern;
  ConvertSpTensorGetDimBlockPos(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(SpTensorGetDimBlockPos op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SpTensorGetDimBlockPosAdaptor tensorAdaptor = llvm::cast<SpTensorGetDimBlockPosAdaptor>(adaptor);
    
    
    // SparseTensor sp_tensor;
    // if(!unpack_sparse_tensor(tensorAdaptor.getTensor(), sp_tensor)) {
    //   return failure();
    // }

    auto zero = rewriter.create<index::ConstantOp>(op->getLoc(), 0);
    rewriter.replaceOp(op, rewriter.create<tensor::EmptyOp>(op->getLoc(), RankedTensorType::get({ShapedType::kDynamic,}, rewriter.getIndexType()), ValueRange(zero)));

    return success();
  }
};

/// TODO: Implement this conversion to actually handle blocks/tiles
class ConvertSpTensorGetDimBlockCrd
    : public OpConversionPattern<SpTensorGetDimBlockCrd> {
  using OpConversionPattern<SpTensorGetDimBlockCrd>::OpConversionPattern;
  ConvertSpTensorGetDimBlockCrd(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(SpTensorGetDimBlockCrd op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SpTensorGetDimBlockCrdAdaptor tensorAdaptor = llvm::cast<SpTensorGetDimBlockCrdAdaptor>(adaptor);
    
    
    // SparseTensor sp_tensor;
    // if(!unpack_sparse_tensor(tensorAdaptor.getTensor(), sp_tensor)) {
    //   return failure();
    // }

    auto zero = rewriter.create<index::ConstantOp>(op->getLoc(), 0);
    rewriter.replaceOp(op, rewriter.create<tensor::EmptyOp>(op->getLoc(), RankedTensorType::get({ShapedType::kDynamic,}, rewriter.getIndexType()), ValueRange(zero)));

    return success();
  }
};

class ConvertSpTensorGetVals
    : public OpConversionPattern<SpTensorGetVals> {
  using OpConversionPattern<SpTensorGetVals>::OpConversionPattern;
  ConvertSpTensorGetVals(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(SpTensorGetVals op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SpTensorGetValsAdaptor tensorAdaptor = llvm::cast<SpTensorGetValsAdaptor>(adaptor);
    SparseTensor sp_tensor;
    if(!unpack_sparse_tensor(tensorAdaptor.getTensor(), sp_tensor)) {
      return failure();
    }
    rewriter.replaceOp(op, {sp_tensor.vals});
    return success();
  }
};


class ConvertSpTensorGetDimPos
    : public OpConversionPattern<SpTensorGetDimPos> {
  using OpConversionPattern<SpTensorGetDimPos>::OpConversionPattern;
  ConvertSpTensorGetDimPos(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(SpTensorGetDimPos op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SpTensorGetDimPosAdaptor tensorAdaptor = llvm::cast<SpTensorGetDimPosAdaptor>(adaptor);
    SparseTensor sp_tensor;
    if(!unpack_sparse_tensor(tensorAdaptor.getTensor(), sp_tensor)) {
      return failure();
    }
    if(sp_tensor.dims[tensorAdaptor.getDim()].pos == nullptr)
    {
      auto zero = rewriter.create<index::ConstantOp>(op->getLoc(), 0);
      rewriter.replaceOp(op, rewriter.create<tensor::EmptyOp>(op->getLoc(), RankedTensorType::get({ShapedType::kDynamic,}, rewriter.getIndexType()), ValueRange(zero)));
    }
    else
    {
      rewriter.replaceOp(op, {sp_tensor.dims[tensorAdaptor.getDim()].pos});
    }

    return success();
  }
};

class ConvertSpTensorFindPos
    : public OpConversionPattern<TensorFindPos> {
  using OpConversionPattern<TensorFindPos>::OpConversionPattern;
  ConvertSpTensorFindPos(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(TensorFindPos op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorFindPosAdaptor tensorAdaptor = llvm::cast<TensorFindPosAdaptor>(adaptor);
    SparseTensor sp_tensor;
    if(!unpack_sparse_tensor(tensorAdaptor.getTensor(), sp_tensor)) {
      return failure();
    }

    if(tensorAdaptor.getIsLinear())
    {
      rewriter.replaceOp(op, {sp_tensor.dims[tensorAdaptor.getDim()].insert_pos});
    } else {
      assert(false && "Lowering non-unique inserts is not yet supported, please use workspace transform");
    }
    
    return success();
  }
};

class ConvertAllocWorkspaceOp
    : public OpConversionPattern<AllocWorkspaceOp> {
  using OpConversionPattern<AllocWorkspaceOp>::OpConversionPattern;
  ConvertAllocWorkspaceOp(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(AllocWorkspaceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto alloc_adaptor = llvm::cast<AllocWorkspaceOpAdaptor>(adaptor);
    auto loc = op.getLoc();
    Type index_type = rewriter.getIndexType();

    Value sp_tensor = op.getTensor();
    auto sp_tensor_type = llvm::cast<SparseTensorType>(sp_tensor.getType());
    auto dims = alloc_adaptor.getDims();
    SmallVector<int64_t> dim_attrs(dims.size(), ShapedType::kDynamic);
    SmallVector<Value> sizes;
    for(auto dim : dims)
    {
      Value dim_size = rewriter.create<SpTensorGetDimSize>(loc, index_type, sp_tensor, llvm::cast<IntegerAttr>(dim));
      sizes.push_back(dim_size);
    }

    Workspace workspace;
    auto workspace_tensor_type = RankedTensorType::get(dim_attrs, sp_tensor_type.getElementType());
    workspace.workspace = rewriter.create<bufferization::AllocTensorOp>(loc, workspace_tensor_type, sizes);
    workspace.mark_value = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    auto workspace_mark_type = RankedTensorType::get(dim_attrs, rewriter.getI32Type());
    workspace.mark_array = rewriter.create<bufferization::AllocTensorOp>(loc, workspace_mark_type, sizes);
    workspace.num_crds = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
    auto crds_type = RankedTensorType::get({ShapedType::kDynamic,}, index_type);
    workspace.crds = rewriter.create<bufferization::AllocTensorOp>(loc, crds_type, sizes);

    auto workspace_type = llvm::cast<WorkspaceType>(op->getResult(0).getType());
    /** TODO: Support higher dimensional workspaces! */
    assert(workspace_type.getDims().size() == 1 && "Workspace dimensions > 1 are currently unsupported.");

    SmallVector<Value, 6> cast_args;
    pack_workspace(workspace_type, workspace, cast_args);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, workspace_type, cast_args);
    return success();    
  }
};

class ConvertWorkspaceGetNNZ
    : public OpConversionPattern<SpTensorGetNNZ> {
  using OpConversionPattern<SpTensorGetNNZ>::OpConversionPattern;
  ConvertWorkspaceGetNNZ(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(SpTensorGetNNZ op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto opAdaptor = llvm::cast<SpTensorGetNNZAdaptor>(adaptor);
    if(!llvm::isa<WorkspaceType>(opAdaptor.getTensor().getType())){
      return failure();
    }
    Workspace workspace;
    if(!unpack_workspace(opAdaptor.getTensor(), workspace)){
      return failure();
    }

    rewriter.replaceOp(op, {workspace.num_crds,});
    return success();
  }
};

class ConvertFunCallOp
    : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;
  ConvertFunCallOp(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto opAdaptor = llvm::cast<func::CallOpAdaptor>(adaptor);

    SmallVector<Value, 8> args;
    SmallVector<Type, 8> args_types;
    for(auto operand: opAdaptor.getOperands())
    {
      if(mlir::isa<SparseTensorType>(operand.getType()))
      {
        SparseTensor sp_tensor;
        if(!unpack_sparse_tensor(operand, sp_tensor))
        {
          return failure();
        }

        args.push_back(sp_tensor.dim_sizes);
        args_types.push_back(sp_tensor.dim_sizes.getType());
        for(unsigned i = 0; i < sp_tensor.dims.size(); i++) {
          Dimension dim = sp_tensor.dims[i];
          args.push_back(dim.insert_pos);
          args_types.push_back(dim.insert_pos.getType());

          switch(dim.format) {
            case TensorFormatEnum::D:
            {
              args.push_back(dim.pos);
              args_types.push_back(dim.pos.getType());
              break;
            }
            case TensorFormatEnum::CU:
            case TensorFormatEnum::CN:
            {
              args.push_back(dim.pos);
              args_types.push_back(dim.pos.getType()); //Pos tensor
              args.push_back(dim.crd);
              args_types.push_back(dim.crd.getType()); //Crd tensor
              break;
            }
            case TensorFormatEnum::S:
            {
              args.push_back(dim.crd); //Crd tensor
              args_types.push_back(dim.crd.getType()); //Crd tensor
              break;
            }
            default: {
            }
          }
        }
        args.push_back(sp_tensor.vals);
        args_types.push_back(sp_tensor.vals.getType());
      }
      else
      {
        args.push_back(operand);
        args_types.push_back(operand.getType());
      }
    }
    
    rewriter.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), TypeRange(), args);

    return success();
  }
};

class ConvertWorkspaceGetCrds
    : public OpConversionPattern<SpTensorGetCrd> {
  using OpConversionPattern<SpTensorGetCrd>::OpConversionPattern;
  ConvertWorkspaceGetCrds(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(SpTensorGetCrd op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto opAdaptor = llvm::cast<SpTensorGetCrdAdaptor>(adaptor);
    if(!llvm::isa<WorkspaceType>(opAdaptor.getTensor().getType())){
      return failure();
    }
    Workspace workspace;
    if(!unpack_workspace(opAdaptor.getTensor(), workspace)){
      return failure();
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, op->getResultTypes(), workspace.crds, opAdaptor.getIdx());
    return success();
  }
};

class ConvertWorkspaceGetDimSize
    : public OpConversionPattern<SpTensorGetDimSize> {
  using OpConversionPattern<SpTensorGetDimSize>::OpConversionPattern;
  ConvertWorkspaceGetDimSize(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(SpTensorGetDimSize op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opAdaptor = llvm::cast<SpTensorGetDimSizeAdaptor>(adaptor);
    if(!llvm::isa<WorkspaceType>(opAdaptor.getTensor().getType())){
      return failure();
    }

    Workspace workspace;
    if(!unpack_workspace(opAdaptor.getTensor(), workspace)) {
      return failure();
    }
    Value dim = rewriter.create<index::ConstantOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(opAdaptor.getDim()));
    rewriter.replaceOpWithNewOp<tensor::DimOp>(op, op->getResultTypes(), workspace.workspace, dim);
    return success();
  }
};

class ConvertWorkspaceTensorInsertOp
    : public OpConversionPattern<TensorInsertOp> {
  using OpConversionPattern<TensorInsertOp>::OpConversionPattern;
  ConvertWorkspaceTensorInsertOp(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(TensorInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto opAdaptor = llvm::cast<TensorInsertOpAdaptor>(adaptor);
    if(!llvm::isa<WorkspaceType>(opAdaptor.getTensor().getType())){
      return failure();
    }
    WorkspaceType workspace_type = llvm::cast<WorkspaceType>(opAdaptor.getTensor().getType());
    assert(workspace_type.getDims().size() == 1 && "Workspace dimensions > 1 are currently unsupported.");

    Workspace workspace;
    if(!unpack_workspace(opAdaptor.getTensor(), workspace)) {
      return failure();
    }

    auto loc = op.getLoc();
    auto context = op.getContext();
    ValueRange crds = opAdaptor.getCrds();
    Value crd = crds[opAdaptor.getCrds().size() - 1];
    Value mark_at_crd = rewriter.create<tensor::ExtractOp>(
      loc,
      rewriter.getI32Type(),
      workspace.mark_array,
      crd      
    );
    Value not_seen = rewriter.create<arith::CmpIOp>(
      loc, 
      rewriter.getI1Type(),
      arith::CmpIPredicateAttr::get(context, arith::CmpIPredicate::ne),
      mark_at_crd,
      workspace.mark_value
    );
    
    Operation* if_op = rewriter.create<scf::IfOp>(
      loc,
      not_seen,
      [workspace, crd] (OpBuilder& builder, Location loc) {
        Type index_type = builder.getIndexType();
        Value new_mark = builder.create<tensor::InsertOp>(loc, workspace.mark_array.getType(), workspace.mark_value, workspace.mark_array, crd);
        Value new_crds = builder.create<tensor::InsertOp>(loc, workspace.crds.getType(), crd, workspace.crds, workspace.num_crds);
        Value inc = builder.create<index::ConstantOp>(loc, index_type, builder.getIndexAttr(1));
        Value new_crd_size = builder.create<index::AddOp>(loc, index_type, workspace.num_crds, inc);
        builder.create<scf::YieldOp>(loc, ArrayRef<Value>({new_mark, new_crd_size, new_crds}));
      },
      [workspace] (OpBuilder& builder, Location loc) {
        builder.create<scf::YieldOp>(loc, ArrayRef<Value>({workspace.mark_array, workspace.num_crds, workspace.crds}));
      }
    );
    workspace.mark_array = if_op->getResult(0);
    workspace.num_crds = if_op->getResult(1);
    workspace.crds = if_op->getResult(2);
    workspace.workspace = rewriter.create<tensor::InsertOp>(
      loc,
      workspace.workspace.getType(),
      opAdaptor.getValue(),
      workspace.workspace,
      crd
    );

    SmallVector<Value, 6> cast_args;
    pack_workspace(workspace_type, workspace, cast_args);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, workspace_type, cast_args);
    return success();
  }
};

class ConvertWorkspaceTensorExtractOp
    : public OpConversionPattern<TensorExtractOp> {
  using OpConversionPattern<TensorExtractOp>::OpConversionPattern;
  ConvertWorkspaceTensorExtractOp(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(TensorExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto opAdaptor = llvm::cast<TensorExtractOpAdaptor>(adaptor);
    if(!llvm::isa<WorkspaceType>(opAdaptor.getTensor().getType())){
      return failure();
    }

    Workspace workspace;
    if(!unpack_workspace(opAdaptor.getTensor(), workspace)) {
      return failure();
    }

    auto loc = op.getLoc();
    auto context = op.getContext();
    Value pos = opAdaptor.getPos();
    Value mark_at_pos = rewriter.create<tensor::ExtractOp>(
      loc,
      rewriter.getI32Type(),
      workspace.mark_array,
      pos      
    );
    Value seen = rewriter.create<arith::CmpIOp>(
      loc, 
      rewriter.getI1Type(),
      arith::CmpIPredicateAttr::get(context, arith::CmpIPredicate::eq),
      mark_at_pos,
      workspace.mark_value
    );


    Operation* if_op = rewriter.create<scf::IfOp>(
      loc,
      seen,
      [op, workspace, pos] (OpBuilder& builder, Location loc) {
        Value extracted = builder.create<tensor::ExtractOp>(loc, op->getResultTypes(), workspace.workspace, pos);
        builder.create<scf::YieldOp>(loc, ArrayRef<Value>({extracted}));
      },
      [&] (OpBuilder& builder, Location loc) {
        // TODO: Does the zero value depend on the semi-ring?
        Type result_type  = op->getResult(0).getType();
        Value zero = builder.create<arith::ConstantOp>(loc, result_type, op.getZeroAttr());
        builder.create<scf::YieldOp>(loc, ArrayRef<Value>({zero}));
      }
    );

    rewriter.replaceOp(op, if_op->getResults());
    return success();
  }
};

class ConvertWorkspaceClearOp
    : public OpConversionPattern<WorkspaceClearOp> {
  using OpConversionPattern<WorkspaceClearOp>::OpConversionPattern;
  ConvertWorkspaceClearOp(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(WorkspaceClearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto opAdaptor = llvm::cast<WorkspaceClearOpAdaptor>(adaptor);
    if(!llvm::isa<WorkspaceType>(opAdaptor.getTensor().getType())){
      return failure();
    }
    WorkspaceType workspace_type = llvm::cast<WorkspaceType>(opAdaptor.getTensor().getType());
    Workspace workspace;
    if(!unpack_workspace(opAdaptor.getTensor(), workspace)) {
      return failure();
    }
    auto loc = op.getLoc();
    Value inc = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    workspace.mark_value = rewriter.create<arith::AddIOp>(loc, rewriter.getI32Type(), workspace.mark_value, inc);
    workspace.num_crds = rewriter.create<index::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));

    SmallVector<Value, 6> cast_args;
    pack_workspace(workspace_type, workspace, cast_args);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, workspace_type, cast_args);
    return success();
  }
};

class PrintOpLowering : public OpConversionPattern<PrintOp> {
  using OpConversionPattern<PrintOp>::OpConversionPattern;
  PrintOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(PrintOp op,  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    Location loc = op->getLoc();
    auto inputType = adaptor.getInput().getType();
    Type index_type = rewriter.getIndexType();
    SmallVector<int64_t> empty_size(1, 1);
    auto empty_type = RankedTensorType::get(empty_size, index_type);
    Value empty_tensor = rewriter.create<bufferization::AllocTensorOp>(loc, empty_type, ValueRange(), (Value)nullptr);
    Value neg = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(-1));
    Value zero = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
    empty_tensor = rewriter.create<tensor::InsertOp>(loc, empty_type, neg, empty_tensor, zero);

    if (inputType.isa<SparseTensorType>())
    {
      SparseTensor sp_tensor;
      if(!unpack_sparse_tensor(adaptor.getInput(), sp_tensor)) {
        return failure();
      }
      for (Dimension& dim : sp_tensor.dims)
      {
        switch(dim.format){
          case TensorFormatEnum::D: {
            rewriter.create<PrintOp>(loc, dim.pos);
            rewriter.create<PrintOp>(loc, empty_tensor);
            break;
          }
          case TensorFormatEnum::CU:
          case TensorFormatEnum::CN: {
            rewriter.create<PrintOp>(loc, dim.pos);
            rewriter.create<PrintOp>(loc, dim.crd);
            break;
          }
          case TensorFormatEnum::S: {
            rewriter.create<PrintOp>(loc, empty_tensor);
            rewriter.create<PrintOp>(loc, dim.crd);
            break;
          }
          default: {
              assert(false && "Could not print unknown format to sparse tensor.");
            }
        }
      }
      rewriter.create<PrintOp>(loc, sp_tensor.vals);
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

class GetTimeLowering : public OpConversionPattern<GetTimeOp> {
  using OpConversionPattern<GetTimeOp>::OpConversionPattern;
  GetTimeLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(GetTimeOp op,  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto f64Type = rewriter.getF64Type();
    std::string getTimeStr = "getTime";

    if (!hasFuncDeclaration(module, getTimeStr))
    {
      auto getTimeFunc = FunctionType::get(ctx, {}, {FloatType::getF64(ctx)});
      /// func @getTime() -> f64
      func::FuncOp func1 = func::FuncOp::create(op->getLoc(), getTimeStr,
                                                getTimeFunc, ArrayRef<NamedAttribute>{});
      func1.setPrivate();
      module.push_back(func1);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, getTimeStr, SmallVector<Type, 2>{f64Type});

    return success();
  }
};

class PrintElapsedTimeLowering : public OpConversionPattern<PrintElapsedTimeOp> {
  using OpConversionPattern<PrintElapsedTimeOp>::OpConversionPattern;
  PrintElapsedTimeLowering(MLIRContext *context) : OpConversionPattern(context) {}
    
  LogicalResult
  matchAndRewrite(PrintElapsedTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();

    auto start = adaptor.getStart();
    auto end = adaptor.getEnd();
    std::string printElapsedTimeStr = "printElapsedTime";
    auto f64Type = rewriter.getF64Type();

    if (!hasFuncDeclaration(module, printElapsedTimeStr))
    {
      auto printElapsedTimeFunc = FunctionType::get(ctx, {f64Type, f64Type}, {});
      /// func @printElapsedTime(f64, f64) -> ()
      func::FuncOp func1 = func::FuncOp::create(op->getLoc(), printElapsedTimeStr,
                                                printElapsedTimeFunc, ArrayRef<NamedAttribute>{});
      func1.setPrivate();
      module.push_back(func1);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, printElapsedTimeStr, SmallVector<Type, 2>{}, ValueRange{start, end});

    return success();
  }
};

}

void mlir::comet::populateSparseTensorConversionPatterns(MLIRContext *context, RewritePatternSet &patterns, TypeConverter &typeConverter) {
  typeConverter.addConversion(
    [](tensorAlgebra::SparseTensorType type, SmallVectorImpl<Type> &types) {
      ArrayRef<int64_t> dim_sizes = type.getDims();
      ArrayRef<int32_t> format = type.getFormat();

      auto context = type.getContext();
      Type index_type = IndexType::get(context);
      bool is_known_size = true;
      int known_size = 1;
      types.push_back(RankedTensorType::get({static_cast<long long>(dim_sizes.size())}, IndexType::get(context))); //Dimension sizes
      for(unsigned i = 0; i < dim_sizes.size(); i++) {
        types.push_back(index_type); //Insert pos
        switch((TensorFormatEnum)format[2 * i])
        {
          case TensorFormatEnum::D:
          {
            if(dim_sizes[i] != ShapedType::kDynamic) {
              known_size *= dim_sizes[i]; 
            } else {
              is_known_size = false;
            }
            auto pos_type = mlir::RankedTensorType::get({ShapedType::kDynamic,}, index_type);
            types.push_back(pos_type); //Pos tensor
            break;
          }
          case TensorFormatEnum::CU:
          case TensorFormatEnum::CN:
          {
            Type pos_type = mlir::RankedTensorType::get({ShapedType::kDynamic,}, 
                                                      index_type);
            Type crd_type = mlir::RankedTensorType::get({ShapedType::kDynamic,}, 
                                                    index_type);
            is_known_size = false;

            types.push_back(pos_type); //Pos tensor
            types.push_back(crd_type); //Crd tensor
            break;
          }
          case TensorFormatEnum::S:
          {
            Type crd_type = mlir::RankedTensorType::get({ShapedType::kDynamic,}, index_type);
            types.push_back(crd_type); //Crd tensor
            break;
          }
          default: {
            assert(false && "Could not unpack unknown format to sparse tensor.");
          }
        }
      }
      Type element_type = type.getElementType();
      Type value_type = mlir::RankedTensorType::get({ShapedType::kDynamic,}, element_type);
      types.push_back(value_type); //Value tensor
      return success();
    });

  typeConverter.addConversion(
    [](WorkspaceType type, SmallVectorImpl<Type> &types) {
      Type element_type = type.getElementType();
      ArrayRef<int64_t> dim_sizes = type.getDims();
      auto context = type.getContext();
      types.push_back(RankedTensorType::get(dim_sizes, element_type)); // Workspace
      types.push_back(IntegerType::get(context, 32)); // Mark Value
      types.push_back(RankedTensorType::get(dim_sizes, IntegerType::get(context, 32))); // Mark array
      types.push_back(IndexType::get(context)); // Crd Size
      types.push_back(RankedTensorType::get({ShapedType::kDynamic,}, IndexType::get(context)));// Crd tensors
      return success();
    });

  typeConverter.addArgumentMaterialization(
    [](OpBuilder &builder, SparseTensorType resultType, ValueRange inputs,
        Location loc) -> std::optional<Value> {
      auto op = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs);
      return op.getResult(0);
    });

  typeConverter.addSourceMaterialization(
    [](OpBuilder &builder, SparseTensorType resultType, ValueRange inputs,
        Location loc) -> std::optional<Value> {
      auto op = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs);
      return op.getResult(0);
    });
  
  typeConverter.addArgumentMaterialization(
    [](OpBuilder &builder, WorkspaceType resultType, ValueRange inputs,
        Location loc) -> std::optional<Value> {
      auto op = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs);
      return op.getResult(0);
    });

  typeConverter.addSourceMaterialization(
    [](OpBuilder &builder, WorkspaceType resultType, ValueRange inputs,
        Location loc) -> std::optional<Value> {
      auto op = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs);
      return op.getResult(0);
    });

  patterns.add<PrintOpLowering, GetTimeLowering, PrintElapsedTimeLowering>(typeConverter, context);
  patterns.add<ConvertSpTensorConstructOp, ConvertSpTensorInsertOp, ConvertSpTensorExtractOp, ConvertSpTensorInsertCrd, ConvertSpTensorGetDimSize, ConvertSpTensorGetDimCrd, ConvertSpTensorGetDimPos, ConvertSpTensorGetDimBlockCrd, ConvertSpTensorGetDimBlockPos, ConvertSpTensorGetVals, ConvertSpTensorFindPos, ConvertFunCallOp>(typeConverter, context);
  patterns.add<ConvertAllocWorkspaceOp, ConvertWorkspaceGetNNZ, ConvertWorkspaceGetCrds, ConvertWorkspaceTensorInsertOp, ConvertWorkspaceTensorExtractOp, ConvertWorkspaceGetDimSize, ConvertWorkspaceClearOp>(typeConverter, context);
}

struct SparseTensorConversionPass : comet::impl::SparseTensorConversionPassBase<SparseTensorConversionPass> {
  using SparseTensorConversionPassBase::SparseTensorConversionPassBase;
  
  SparseTensorConversionPass() = default;
  SparseTensorConversionPass(const SparseTensorConversionPass &pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    TypeConverter typeConverter;
    ConversionTarget target(*ctx);

    // Everything in the TADialect must go
    target.addIllegalDialect<tensorAlgebra::TADialect>();
    
    // The following operations and dialects may be introduced by the
    // rewriting rules, and are therefore marked as legal.
    // target.addLegalOp<TensorInsertOp, TensorExtractOp, SpTensorGetDimCrd, SpTensorInsertCrd, SpTensorGetDimPos, SpTensorGetDimSize, SpTensorGetVals, TensorFindPos, PrintOp, PrintElapsedTimeOp, GetTimeOp, tensor::ExtractOp, tensor::InsertOp>();
    target.addLegalOp<tensor::ExtractOp, tensor::InsertOp>();
    target.addLegalDialect<
        arith::ArithDialect, bufferization::BufferizationDialect,
        tensor::TensorDialect, memref::MemRefDialect, scf::SCFDialect,
        func::FuncDialect, index::IndexDialect, BuiltinDialect
    >();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op){
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op){
      return typeConverter.isLegal(op);
    });

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<tensorAlgebra::PrintOp>([&](tensorAlgebra::PrintOp op) {
      return typeConverter.isLegal(op->getOperandTypes());
    });

    typeConverter.addConversion([](Type type) { return type; });
   

    // Populate with rules and apply rewriting rules.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                         target);
    mlir::indexTree::populateIndexTreeTypeConversionPatterns(ctx, patterns, typeConverter, target);
    mlir::comet::populateSparseTensorConversionPatterns(ctx, patterns, typeConverter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
    {
      return signalPassFailure(); 
    }
  }
};

std::unique_ptr<Pass> mlir::comet::createSparseTensorConversionPass()
{
  return std::make_unique<SparseTensorConversionPass>();
}
