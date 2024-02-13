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

using namespace mlir;
using llvm::SmallVector;

namespace mlir {
    namespace comet{
      #define GEN_PASS_DEF_CONVERTSYMBOLICDOMAINS
      #include "comet/Conversion/Passes.h.inc"
    }
}

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "SymbolicDomainPatterns.inc"
} // namespace

namespace {

template <typename SourceOp>
class DecomposeSymbolicDomainOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  DecomposeSymbolicDomainOpConversionPattern(TypeConverter &typeConverter,
                                             MLIRContext *context,
                                             ValueDecomposer &decomposer,
                                             PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit),
        decomposer(decomposer) {}

protected:
  ValueDecomposer &decomposer;
};
} // namespace

namespace {
/// Expand return operands according to the provided TypeConverter and
/// ValueDecomposer.
struct ConvertDomainYieldOp
    : public DecomposeSymbolicDomainOpConversionPattern<scf::YieldOp> {
  using DecomposeSymbolicDomainOpConversionPattern::
      DecomposeSymbolicDomainOpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 2> newOperands;
    for (Value operand : adaptor.getOperands())
      decomposer.decomposeValue(rewriter, op.getLoc(), operand.getType(),
                                operand, newOperands);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, newOperands);
    return success();
  }
};

struct ConvertDomainConditionOp
    : public DecomposeSymbolicDomainOpConversionPattern<scf::ConditionOp> {
  using DecomposeSymbolicDomainOpConversionPattern::
      DecomposeSymbolicDomainOpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 8> newOperands;
    for (Value operand : adaptor.getOperands())
      decomposer.decomposeValue(rewriter, op.getLoc(), operand.getType(),
                                operand, newOperands);
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, TypeRange(), newOperands);
    return success();
  }
};

struct ConvertDomainForOp
    : public DecomposeSymbolicDomainOpConversionPattern<scf::ForOp> {
  using DecomposeSymbolicDomainOpConversionPattern::
      DecomposeSymbolicDomainOpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    SmallVector<Value, 4> newOperands;
    for (Value operand : adaptor.getOperands())
      decomposer.decomposeValue(rewriter, op.getLoc(), operand.getType(),
                                operand, newOperands);

    // Create the new result types for the new `CallOp` and track the indices in
    // the new call op's results that correspond to the old call op's results.
    //
    // expandedResultIndices[i] = "list of new result indices that old result i
    // expanded to".
    SmallVector<Type, 3> newResultTypes;
    SmallVector<SmallVector<unsigned, 3>, 2> expandedResultIndices;
    for (Type resultType : op.getResultTypes()) {
      unsigned oldSize = newResultTypes.size();
      if (failed(typeConverter->convertType(resultType, newResultTypes)))
        return failure();
      auto &resultMapping = expandedResultIndices.emplace_back();
      for (unsigned i = oldSize, e = newResultTypes.size(); i < e; i++)
        resultMapping.push_back(i);
    }
    auto iter_args = SmallVector<Value>(newOperands.begin() + 3, newOperands.end());
    scf::ForOp newForOp = rewriter.create<scf::ForOp>(op.getLoc(), newOperands[0], newOperands[1], newOperands[2], iter_args);
    newForOp.getRegion().takeBody(op.getRegion());

    // Build a replacement value for each result to replace its uses. If a
    // result has multiple mapping values, it needs to be materialized as a
    // single value.
    SmallVector<Value, 2> replacedValues;
    replacedValues.reserve(op.getNumResults());
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      auto decomposedValues = llvm::to_vector<6>(
          llvm::map_range(expandedResultIndices[i],
                          [&](unsigned i) { return newForOp.getResult(i); }));
      if (decomposedValues.empty()) {
        // No replacement is required.
        replacedValues.push_back(nullptr);
      } else if (decomposedValues.size() == 1) {
        replacedValues.push_back(decomposedValues.front());
      } else {
        // Materialize a single Value to replace the original Value.
        Value materialized = getTypeConverter()->materializeArgumentConversion(
            rewriter, op.getLoc(), op.getType(i), decomposedValues);
        replacedValues.push_back(materialized);
      }
    }
    rewriter.replaceOp(op, replacedValues);

    // Convert region arguments using the type converter
    TypeConverter::SignatureConversion conversion(newForOp.getBody()->getNumArguments());
    for (const auto &argType : llvm::enumerate(newForOp.getBody()->getArgumentTypes())) {
      SmallVector<Type, 3> decomposedTypes;
      if (failed(typeConverter->convertType(argType.value(), decomposedTypes)))
        return failure();
      if (!decomposedTypes.empty())
        conversion.addInputs(argType.index(), decomposedTypes);
    }
    // If the SignatureConversion doesn't apply, bail out.
    if (failed(rewriter.convertRegionTypes(&newForOp.getRegion(), *getTypeConverter(),
                                           &conversion)))
      return failure();

    return success();
  }
};

struct ConvertDomainWhileOp
    : public DecomposeSymbolicDomainOpConversionPattern<scf::WhileOp> {
  using DecomposeSymbolicDomainOpConversionPattern::
      DecomposeSymbolicDomainOpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 4> newOperands;
    for (Value operand : adaptor.getOperands())
      decomposer.decomposeValue(rewriter, op.getLoc(), operand.getType(),
                                operand, newOperands);

    // Create the new result types for the new `CallOp` and track the indices in
    // the new call op's results that correspond to the old call op's results.
    //
    // expandedResultIndices[i] = "list of new result indices that old result i
    // expanded to".
    SmallVector<Type, 3> newResultTypes;
    SmallVector<SmallVector<unsigned, 3>, 2> expandedResultIndices;
    for (Type resultType : op.getResultTypes()) {
      unsigned oldSize = newResultTypes.size();
      if (failed(typeConverter->convertType(resultType, newResultTypes)))
        return failure();
      auto &resultMapping = expandedResultIndices.emplace_back();
      for (unsigned i = oldSize, e = newResultTypes.size(); i < e; i++)
        resultMapping.push_back(i);
    }
    scf::WhileOp newWhileOp = rewriter.create<scf::WhileOp>(op.getLoc(), newResultTypes, newOperands);
    newWhileOp.getBefore().takeBody(op.getBefore());
    newWhileOp.getAfter().takeBody(op.getAfter());

    // Build a replacement value for each result to replace its uses. If a
    // result has multiple mapping values, it needs to be materialized as a
    // single value.
    SmallVector<Value, 2> replacedValues;
    replacedValues.reserve(op.getNumResults());
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      auto decomposedValues = llvm::to_vector<6>(
          llvm::map_range(expandedResultIndices[i],
                          [&](unsigned i) { return newWhileOp.getResult(i); }));
      if (decomposedValues.empty()) {
        // No replacement is required.
        replacedValues.push_back(nullptr);
      } else if (decomposedValues.size() == 1) {
        replacedValues.push_back(decomposedValues.front());
      } else {
        // Materialize a single Value to replace the original Value.
        Value materialized = getTypeConverter()->materializeArgumentConversion(
            rewriter, op.getLoc(), op.getType(i), decomposedValues);
        replacedValues.push_back(materialized);
      }
    }
    rewriter.replaceOp(op, replacedValues);

    // Convert region arguments using the type converter
    TypeConverter::SignatureConversion conversion(newWhileOp->getNumOperands());
    for (const auto &argType : llvm::enumerate(newWhileOp->getOperandTypes())) {
      SmallVector<Type, 3> decomposedTypes;
      if (failed(typeConverter->convertType(argType.value(), decomposedTypes)))
        return failure();
      if (!decomposedTypes.empty())
        conversion.addInputs(argType.index(), decomposedTypes);
    }
    // If the SignatureConversion doesn't apply, bail out.
    if (failed(rewriter.convertRegionTypes(&newWhileOp.getBefore(), *getTypeConverter(),
                                           &conversion)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&newWhileOp.getAfter(), *getTypeConverter(),
                                           &conversion)))
      return failure();
      
    return success();
  }
};

struct ConvertDomainInsertOp
    : public DecomposeSymbolicDomainOpConversionPattern<indexTree::SymbolicDomainInsertOp> {
  using DecomposeSymbolicDomainOpConversionPattern::
      DecomposeSymbolicDomainOpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::SymbolicDomainInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto context = op.getContext();
    Type index_type = rewriter.getIndexType();
    Type memref_type = MemRefType::get({ShapedType::kDynamic,}, index_type);
    Value pos_size = rewriter.create<indexTree::SymbolicDomainGetPosSize>(loc, index_type, op.getDomain());
    Value pos_alloc_size = rewriter.create<indexTree::SymbolicDomainGetPosAllocSize>(loc, index_type, op.getDomain());
    Value crd_size = rewriter.create<indexTree::SymbolicDomainGetCrdSize>(loc, index_type, op.getDomain());
    Value dim_size = rewriter.create<indexTree::SymbolicDomainGetDimSize>(loc, index_type, op.getDomain());
    Value pos = rewriter.create<indexTree::SymbolicDomainGetPos>(loc, memref_type, op.getDomain());
    Value mark_array = rewriter.create<indexTree::SymbolicDomainGetMarkArray>(loc, memref_type, op.getDomain());

    Value one = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1)); 
    if(op.getIsUnique())
    {
      // If we know the crd is unique, we can just increment the crd_size value
      crd_size = rewriter.create<index::AddOp>(loc, index_type, crd_size, one);
    } else 
    {
      
      Value mark = rewriter.create<index::AddOp>(loc, index_type, pos_size, one);
      Value mark_val = rewriter.create<memref::LoadOp>(loc, index_type, mark_array, op.getCrd());
      Value is_marked = rewriter.create<index::CmpOp>(loc, 
                                    rewriter.getI1Type(),
                                    index::IndexCmpPredicateAttr::get(context, index::IndexCmpPredicate::EQ), 
                                    mark,
                                    mark_val);
      scf::IfOp if_op = rewriter.create<scf::IfOp>(loc, index_type, is_marked, true);
      // We have seen this crd before
      rewriter.setInsertionPointToStart(if_op.thenBlock());
      rewriter.create<scf::YieldOp>(loc, crd_size);

      // We haven't seen this crd before
      rewriter.setInsertionPointToStart(if_op.elseBlock());
      rewriter.create<memref::StoreOp>(loc, TypeRange(), mark, mark_array, op.getCrd());
      Value new_crd_size = rewriter.create<index::AddOp>(loc, index_type, crd_size, one);
      rewriter.create<scf::YieldOp>(loc, new_crd_size);

      crd_size = if_op.getResult(0);
    }

    Value materialized = getTypeConverter()->materializeArgumentConversion(
              rewriter, 
              op.getLoc(), 
              op.getDomain().getType(), 
              {pos_size, pos_alloc_size, crd_size, dim_size, pos, mark_array});
    rewriter.replaceOp(op, {materialized});
    return success();
  }
};

struct ConvertDomainEndRowOp
    : public DecomposeSymbolicDomainOpConversionPattern<indexTree::SymbolicDomainEndRowOp> {
  using DecomposeSymbolicDomainOpConversionPattern::
      DecomposeSymbolicDomainOpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::SymbolicDomainEndRowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto context = op.getContext();
    Type index_type = rewriter.getIndexType();
    Type memref_type = MemRefType::get({ShapedType::kDynamic,}, index_type);
    Value pos_size = rewriter.create<indexTree::SymbolicDomainGetPosSize>(loc, index_type, op.getDomain());
    Value pos_alloc_size = rewriter.create<indexTree::SymbolicDomainGetPosAllocSize>(loc, index_type, op.getDomain());
    Value crd_size = rewriter.create<indexTree::SymbolicDomainGetCrdSize>(loc, index_type, op.getDomain());
    Value dim_size = rewriter.create<indexTree::SymbolicDomainGetDimSize>(loc, index_type, op.getDomain());
    Value pos = rewriter.create<indexTree::SymbolicDomainGetPos>(loc, memref_type, op.getDomain());
    Value mark_array = rewriter.create<indexTree::SymbolicDomainGetMarkArray>(loc, memref_type, op.getDomain());

    rewriter.create<memref::StoreOp>(loc, TypeRange(), crd_size, pos, pos_size);
    Value inc = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
    Value new_pos_size = rewriter.create<index::AddOp>(loc, index_type, pos_size, inc);
    // TODO: Dynamically resize array?
    
    Value materialized = getTypeConverter()->materializeArgumentConversion(
            rewriter,
            op.getLoc(),
            op.getDomain().getType(),
            {new_pos_size, pos_alloc_size, crd_size, dim_size, pos, mark_array});

    rewriter.replaceOp(op, {materialized});        
    return success();
  }
};
}

struct IndexTreeOpInlining : public mlir::ConversionPattern {
  IndexTreeOpInlining(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(indexTree::IndexTreeOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    Block& block = op->getRegion(0).front();
    Operation *terminator = block.getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.mergeBlockBefore(&block, op);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);
    return success();
  }
};

namespace{
struct ConvertSparseTensorOp
    : public DecomposeSymbolicDomainOpConversionPattern<indexTree::IndexTreeSparseTensorOp> {
  using DecomposeSymbolicDomainOpConversionPattern::
      DecomposeSymbolicDomainOpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::IndexTreeSparseTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Value, 12> arrays;
    llvm::SmallVector<Value, 12> array_sizes;
    llvm::SmallVector<Value, 3> dim_sizes;

    auto ctx = op.getContext();
    auto format_unk = tensorAlgebra::TensorFormatEnumAttr::get(ctx, tensorAlgebra::TensorFormatEnum::UNK);
    auto format_dense = tensorAlgebra::TensorFormatEnumAttr::get(ctx, tensorAlgebra::TensorFormatEnum::D);
    auto format_compressed = tensorAlgebra::TensorFormatEnumAttr::get(ctx, tensorAlgebra::TensorFormatEnum::CU);
    SmallVector<Attribute, 3> dim_format;
    uint32_t rank = 0;

    auto loc = op.getLoc();
    Type index_type = rewriter.getIndexType();
    Type memref_type = MemRefType::get({ShapedType::kDynamic,}, index_type);
    Value zero = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
    Value one = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
    Value nnz = one;

    for(Value domain : op.getDomains())
    {
      rank += 1;
      if(llvm::isa<indexTree::DomainType>(domain.getType()))
      {
        Operation* domain_op = domain.getDefiningOp();
        if(llvm::isa<indexTree::IndexTreeDenseDomainOp>(domain_op))
        {
          auto dense_domain_op = llvm::cast<indexTree::IndexTreeDenseDomainOp>(domain_op);
          Value dim_size = dense_domain_op.getDimSize();

          Value pos = rewriter.create<memref::AllocOp>(loc, MemRefType::get({1,}, index_type));
          rewriter.create<memref::StoreOp>(loc, TypeRange(), one, pos, zero);
          Value crd = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, index_type));
          Value pos_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, index_type));
          Value crd_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, index_type));

          arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
          arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
          arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
          arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));

          array_sizes.push_back(one);
          array_sizes.push_back(zero);
          array_sizes.push_back(zero);
          array_sizes.push_back(zero);

          dim_sizes.push_back(dim_size);
          nnz = rewriter.create<index::MulOp>(loc, index_type, nnz, dim_size);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_unk);
        } else if(llvm::isa<indexTree::IndexTreeSparseDomainOp>(domain_op))
        {
          auto sparse_domain_op = llvm::cast<indexTree::IndexTreeSparseDomainOp>(domain_op);
          Value dim_size = sparse_domain_op.getDimSize();
          Value pos_size = sparse_domain_op.getPosSize();
          Value crd_size = sparse_domain_op.getCrdSize();

          Value pos = sparse_domain_op.getPos();
          Value crd = sparse_domain_op.getCrd();
          Value pos_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, index_type));
          Value crd_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, index_type));

          arrays.push_back(pos);
          arrays.push_back(crd);
          arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
          arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));

          array_sizes.push_back(pos_size);
          array_sizes.push_back(crd_size);
          array_sizes.push_back(zero);
          array_sizes.push_back(zero);

          dim_sizes.push_back(dim_size);
          nnz = crd_size;
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
        } else
          return failure();
      } else 
      {
        
        Value pos_size = rewriter.create<indexTree::SymbolicDomainGetPosSize>(loc, index_type, domain);
        Value crd_size = rewriter.create<indexTree::SymbolicDomainGetCrdSize>(loc, index_type, domain);
        Value dim_size = rewriter.create<indexTree::SymbolicDomainGetDimSize>(loc, index_type, domain);
        
        Value pos = rewriter.create<indexTree::SymbolicDomainGetPos>(loc, memref_type, domain);
        Value crd = rewriter.create<memref::AllocOp>(loc, memref_type, crd_size);
        Value pos_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, index_type));
        Value crd_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, index_type));

        arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
        arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
        arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
        arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));

        array_sizes.push_back(pos_size);
        array_sizes.push_back(crd_size);
        array_sizes.push_back(zero);
        array_sizes.push_back(zero);

        dim_sizes.push_back(dim_size);
        nnz = crd_size;
      }
    }

    //Allocate values array and initialize
    Value val_array = rewriter.create<memref::AllocOp>(loc, memref_type, nnz);
    auto for_loop = rewriter.create<scf::ForOp>(loc, zero, nnz, one);
    rewriter.setInsertionPointToStart(for_loop.getBody());
    auto induction_var = for_loop.getInductionVar();
    rewriter.create<memref::StoreOp>(loc, TypeRange(), zero, val_array, induction_var);
    rewriter.setInsertionPointAfter(for_loop);
    rewriter.create<bufferization::ToTensorOp>(loc, val_array, rewriter.getUnitAttr(), rewriter.getUnitAttr());

    std::vector<Value> args;
    args.insert(args.end(), arrays.begin(), arrays.end());
    args.push_back(val_array);
    args.insert(args.end(), array_sizes.begin(), array_sizes.end());
    args.push_back(nnz);
    args.insert(args.end(), dim_sizes.begin(), dim_sizes.end());

    rewriter.replaceOpWithNewOp<tensorAlgebra::SparseTensorConstructOp>(op, op.getResult().getType(), args, rank, rewriter.getArrayAttr(dim_format));
    return success();
  }
};
}



// struct SetOpRemoval : public mlir::ConversionPattern {
//   SetOpRemoval(mlir::MLIRContext *ctx)
//       : mlir::ConversionPattern(tensorAlgebra::TensorSetOp::getOperationName(), 1, ctx) {}

//   mlir::LogicalResult
//   match(Operation* op) const override{
//     return success();
//   }


//   void rewrite(Operation* op, ArrayRef<mlir::Value> operands,
//                   mlir::ConversionPatternRewriter &rewriter) const {

//     Value lhs = op->getOperand(0);
//     Value rhs = op->getOperand(1);
//     IRMapping map;
//     map.map(rhs, lhs);

//     for(Operation* rhs_user : rhs.getUsers())
//     {
//       if(rhs_user != op && op->getBlock() == rhs_user->getBlock() && op->isBeforeInBlock(rhs_user))
//       {
//         // TODO: Fix me!!!!!!
//         assert(llvm::isa<tensorAlgebra::PrintOp>(rhs_user));
//         PrintOp print = llvm::cast<tensorAlgebra::PrintOp>(rhs_user);
//         print.getInputMutable().assign(lhs);
//       }
//     }
//     rewriter.eraseOp(op);
//     return;
//   }
// };

struct ConvertSymbolicDomainsPass
    : public PassWrapper<ConvertSymbolicDomainsPass, OperationPass<func::FuncOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSymbolicDomainsPass)

  void runOnOperation()
  {
    // Convert the rest of the index tree dialect to SCF
    TypeConverter typeConverter;
    ValueDecomposer decomposer;
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<index::IndexDialect, arith::ArithDialect, scf::SCFDialect>();
    target.addLegalOp<tensorAlgebra::SparseTensorConstructOp>();
    target.addIllegalOp<indexTree::DeclDomainOp, indexTree::SymbolicDomainEndRowOp, indexTree::SymbolicDomainInsertOp>();
    target.addIllegalOp<indexTree::IndexTreeSparseTensorOp>();
    target.addIllegalOp<indexTree::IndexTreeOp, tensorAlgebra::TensorSetOp>();

    decomposer.addDecomposeValueConversion([](OpBuilder &builder, Location loc,
                                              indexTree::SymbolicDomainType resultType, Value value,
                                              SmallVectorImpl<Value> &values) {
      Type index_type = builder.getIndexType();
      Type memref_type = MemRefType::get({ShapedType::kDynamic,}, index_type);
      Value pos_size = builder.create<indexTree::SymbolicDomainGetPosSize>(loc, index_type, value);
      values.push_back(pos_size);
      Value pos_alloc_size = builder.create<indexTree::SymbolicDomainGetPosAllocSize>(loc, index_type, value);
      values.push_back(pos_alloc_size);
      Value crd_size = builder.create<indexTree::SymbolicDomainGetCrdSize>(loc, index_type, value);
      values.push_back(crd_size);
      Value dim_size = builder.create<indexTree::SymbolicDomainGetDimSize>(loc, index_type, value);
      values.push_back(dim_size);
      Value pos = builder.create<indexTree::SymbolicDomainGetPos>(loc, memref_type, value);
      values.push_back(pos);
      Value mark_array = builder.create<indexTree::SymbolicDomainGetMarkArray>(loc, memref_type, value);
      values.push_back(mark_array);
      return success();
    });

    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
      [](indexTree::SymbolicDomainType domainType, SmallVectorImpl<Type> &types) {
        auto context = domainType.getContext();
        Type index_type = IndexType::get(context);
        Type memref_type = MemRefType::get({ShapedType::kDynamic,}, index_type);
        types.push_back(index_type);
        types.push_back(index_type);
        types.push_back(index_type);
        types.push_back(index_type);
        types.push_back(memref_type);
        types.push_back(memref_type);
        types.push_back(IntegerType::get(context, 1));
        return success();
      });

    typeConverter.addArgumentMaterialization(
      [](OpBuilder &builder, indexTree::SymbolicDomainType resultType, ValueRange inputs,
          Location loc) -> Optional<Value> {
        Value value = builder.create<indexTree::SymbolicDomainConstructOp>(loc, resultType, inputs);
        return value;
      });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertDomainYieldOp, ConvertDomainConditionOp, 
                 ConvertDomainForOp, ConvertDomainWhileOp, 
                 ConvertDomainInsertOp, ConvertDomainEndRowOp>(
                    typeConverter, &getContext(), decomposer);
    populateWithGenerated(patterns);

    // patterns.add<IndexTreeOpInlining, SetOpRemoval>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

/// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createConvertSymbolicDomainsPass()
{
  return std::make_unique<ConvertSymbolicDomainsPass>();
}