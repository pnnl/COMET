#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"
#include "comet/Dialect/IndexTree/Patterns.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using llvm::SmallVector;

namespace mlir {
    namespace comet{
      #define GEN_PASS_DEF_CONVERTSYMBOLICDOMAINS
      #include "comet/Conversion/Passes.h.inc"
    }
}

struct SymbolicDomain {
  Value pos_size;
  Value pos_alloc_size;
  Value crd_size;
  Value dim_size;
  Value pos;
  Value mark_array;
};

static bool unpack_symbolic_domain(Value symbolic_domain, SymbolicDomain& result)
{
  if (auto cast = symbolic_domain.getDefiningOp<UnrealizedConversionCastOp>()) {
    result.pos_size = cast->getOperand(0);
    result.pos_alloc_size = cast->getOperand(1);
    result.crd_size = cast->getOperand(2);
    result.dim_size = cast->getOperand(3);
    result.pos = cast->getOperand(4);
    result.mark_array = cast->getOperand(5);
    return true;
  }
  return false;
}

namespace {
struct ConvertDomainInsertOp
    : public OpConversionPattern<indexTree::SymbolicDomainInsertOp> {
  using OpConversionPattern<indexTree::SymbolicDomainInsertOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::SymbolicDomainInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto context = op.getContext();
    Type index_type = rewriter.getIndexType();
    SymbolicDomain domain;
    if(!unpack_symbolic_domain(llvm::cast<indexTree::SymbolicDomainInsertOpAdaptor>(adaptor).getDomain(), domain)){
      return failure();
    }

    Value one = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1)); 
    if(op.getIsUnique())
    {
      // If we know the crd is unique, we can just increment the crd_size value
      domain.crd_size = rewriter.create<index::AddOp>(loc, index_type, domain.crd_size, one);
    } else 
    {
      
      Value mark = rewriter.create<index::AddOp>(loc, index_type, domain.pos_size, one);
      Value mark_val = rewriter.create<memref::LoadOp>(loc, index_type, domain.mark_array, op.getCrd());
      Value is_marked = rewriter.create<index::CmpOp>(loc, 
                                    rewriter.getI1Type(),
                                    index::IndexCmpPredicateAttr::get(context, index::IndexCmpPredicate::EQ), 
                                    mark,
                                    mark_val);
      scf::IfOp if_op = rewriter.create<scf::IfOp>(loc, index_type, is_marked, true);
      // We have seen this crd before
      rewriter.setInsertionPointToStart(if_op.thenBlock());
      rewriter.create<scf::YieldOp>(loc, domain.crd_size);

      // We haven't seen this crd before
      rewriter.setInsertionPointToStart(if_op.elseBlock());
      rewriter.create<memref::StoreOp>(loc, TypeRange(), mark, domain.mark_array, op.getCrd());
      Value new_crd_size = rewriter.create<index::AddOp>(loc, index_type, domain.crd_size, one);
      rewriter.create<scf::YieldOp>(loc, new_crd_size);
      rewriter.setInsertionPointAfter(if_op);
      domain.crd_size = if_op.getResult(0);
    }

    Value materialized = getTypeConverter()->materializeArgumentConversion(
              rewriter, 
              op.getLoc(), 
              op.getDomain().getType(), 
              {
                domain.pos_size,
                domain.pos_alloc_size,
                domain.crd_size,
                domain.dim_size,
                domain.pos, 
                domain.mark_array
              }
    );
    rewriter.replaceOp(op, {materialized});
    return success();
  }
};

struct ConvertDomainEndRowOp
    : public OpConversionPattern<indexTree::SymbolicDomainEndRowOp> {
  using OpConversionPattern<indexTree::SymbolicDomainEndRowOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::SymbolicDomainEndRowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Type index_type = rewriter.getIndexType();
    SymbolicDomain domain;
    if(!unpack_symbolic_domain(llvm::cast<indexTree::SymbolicDomainEndRowOpAdaptor>(adaptor).getDomain(), domain)){
      return failure();
    }

    rewriter.create<memref::StoreOp>(loc, TypeRange(), domain.crd_size, domain.pos, domain.pos_size);
    Value inc = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
    Value new_pos_size = rewriter.create<index::AddOp>(loc, index_type, domain.pos_size, inc);
    // TODO: Dynamically resize array?
    
    Value materialized = getTypeConverter()->materializeArgumentConversion(
            rewriter,
            op.getLoc(),
            op.getDomain().getType(),
            {
              new_pos_size,
              domain.pos_alloc_size,
              domain.crd_size, 
              domain.dim_size,
              domain.pos, 
              domain.mark_array
            }
    );
    rewriter.replaceOp(op, {materialized});        
    return success();
  }
};

struct ConvertDomainDeclarationOp
    : public OpConversionPattern<indexTree::DeclDomainOp> {
  using OpConversionPattern<indexTree::DeclDomainOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::DeclDomainOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Type index_type = rewriter.getIndexType();
    Type memref_type = MemRefType::get({ShapedType::kDynamic,}, index_type);

    Value zero = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
    Value inc = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1)); 
    Value pos_alloc_size = rewriter.create<index::AddOp>(loc, index_type, op.getNumRows(), inc);
    Value pos = rewriter.create<memref::AllocOp>(loc, memref_type, ValueRange{pos_alloc_size}, ValueRange(), nullptr);
    Value mark_array = rewriter.create<memref::AllocOp>(loc, memref_type, ValueRange{op.getDimSize()}, ValueRange(), nullptr);
    auto new_op = rewriter.create<UnrealizedConversionCastOp>(
      loc, 
      op->getResultTypes(),
      ValueRange({zero, pos_alloc_size, zero, op.getDimSize(), pos, mark_array})
    );
    rewriter.replaceOp(op, new_op->getResults());
    return success();
  }
};

struct ConvertSparseTensorOp
    : public OpConversionPattern<indexTree::IndexTreeSparseTensorOp> {
  using OpConversionPattern<indexTree::IndexTreeSparseTensorOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::IndexTreeSparseTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    

    for(Value domain : llvm::cast<indexTree::IndexTreeSparseTensorOpAdaptor>(adaptor).getDomains())
    {
      if(llvm::isa<indexTree::SymbolicDomainType>(domain.getType())){
        if(!domain.getDefiningOp<UnrealizedConversionCastOp>())
          return failure();
      }
    }

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

    for(Value domain : llvm::cast<indexTree::IndexTreeSparseTensorOpAdaptor>(adaptor).getDomains())
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
      } else if(llvm::isa<indexTree::SymbolicDomainType>(domain.getType()))
      {
        SymbolicDomain domain_struct;
        assert(unpack_symbolic_domain(domain, domain_struct));
        Value crd = rewriter.create<memref::AllocOp>(loc, memref_type, ValueRange{domain_struct.crd_size}, ValueRange(), nullptr);
        Value pos_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, index_type));
        Value crd_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, index_type));

        arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, domain_struct.pos, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
        arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
        arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
        arrays.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));

        array_sizes.push_back(domain_struct.pos_size);
        array_sizes.push_back(domain_struct.crd_size);
        array_sizes.push_back(zero);
        array_sizes.push_back(zero);

        dim_sizes.push_back(domain_struct.dim_size);
        nnz = domain_struct.crd_size;
        dim_format.push_back(format_compressed);
        dim_format.push_back(format_unk);
      }
    }

    //Allocate values array and initialize
    Type float_type = llvm::cast<tensorAlgebra::SparseTensorType>(op.getResult().getType()).getElementType();
    Value val_array = rewriter.create<memref::AllocOp>(loc, MemRefType::get({ShapedType::kDynamic,}, float_type), ValueRange{nnz}, ValueRange(), nullptr);
    Value float_zero = rewriter.create<arith::ConstantOp>(loc, float_type, rewriter.getFloatAttr(float_type, 0.0));
    auto for_loop = rewriter.create<scf::ForOp>(loc, zero, nnz, one);
    rewriter.setInsertionPointToStart(for_loop.getBody());
    auto induction_var = for_loop.getInductionVar();
    rewriter.create<memref::StoreOp>(loc, TypeRange(), float_zero, val_array, induction_var);
    rewriter.setInsertionPointAfter(for_loop);
    val_array = rewriter.create<bufferization::ToTensorOp>(loc, val_array, rewriter.getUnitAttr(), rewriter.getUnitAttr());

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

class EraseDenseDomainOp : public OpConversionPattern<indexTree::IndexTreeDenseDomainOp>{
  public:
    using OpConversionPattern<indexTree::IndexTreeDenseDomainOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(indexTree::IndexTreeDenseDomainOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class EraseSparseDomainOp : public OpConversionPattern<indexTree::IndexTreeSparseDomainOp>{
  public:
    using OpConversionPattern<indexTree::IndexTreeSparseDomainOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(indexTree::IndexTreeSparseDomainOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} //namespace

struct ConvertSymbolicDomainsPass
    : public PassWrapper<ConvertSymbolicDomainsPass, OperationPass<func::FuncOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSymbolicDomainsPass)

  void runOnOperation() override
  {
    // Convert the rest of the index tree dialect to SCF
    TypeConverter typeConverter;
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
        return success();
      });

    typeConverter.addSourceMaterialization(
      [](OpBuilder &builder, indexTree::SymbolicDomainType resultType, ValueRange inputs,
          Location loc) -> Optional<Value> {
        assert(inputs.size() == 6);
        Value value = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)->getResult(0);
        return value;
      });

    typeConverter.addArgumentMaterialization(
      [](OpBuilder &builder, indexTree::SymbolicDomainType resultType, ValueRange inputs,
          Location loc) -> Optional<Value> {
        assert(inputs.size() == 6);
        Value value = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)->getResult(0);
        return value;
      });

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<index::IndexDialect, arith::ArithDialect, scf::SCFDialect, memref::MemRefDialect, bufferization::BufferizationDialect, func::FuncDialect, tensor::TensorDialect>();
    target.addLegalDialect<tensorAlgebra::TADialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<indexTree::DeclDomainOp, indexTree::SymbolicDomainEndRowOp, indexTree::SymbolicDomainInsertOp>();
    target.addIllegalOp<indexTree::IndexTreeSparseTensorOp>();

    

    mlir::RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
    indexTree::populateIndexTreeTypeConversionPatterns(&getContext(), patterns, typeConverter, target);
    patterns.add<ConvertDomainDeclarationOp, ConvertDomainInsertOp, ConvertDomainEndRowOp, ConvertSparseTensorOp>(typeConverter, &getContext());
    patterns.add<EraseDenseDomainOp, EraseSparseDomainOp>(typeConverter, &getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

/// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createConvertSymbolicDomainsPass()
{
  return std::make_unique<ConvertSymbolicDomainsPass>();
}