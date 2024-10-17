//===- LowerIndexTreeIRToSCF.cpp  ------===//
//
// Copyright 2022 Battelle Memorial Institute
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// This file implements a lowering of index tree dialect to SCF dialect
//===----------------------------------------------------------------------===//

#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Conversion/IndexTreeToSCF/AbstractLoopOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/SliceAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/SetVector.h"
#include <iostream>
#include <algorithm>
#include <ostream>
#include <vector>

#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <string>
#include <utility>
#include <queue>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

using llvm::SmallVector;
using llvm::StringRef;
using llvm::SmallDenseMap;

#define DEBUG_TYPE "lowering-it-to-scf"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
// *********** For debug purpose *********//

namespace comet
{
#define GEN_PASS_DEF_CONVERTINDEXTREETOSCF
#include "comet/Conversion/Passes.h.inc"
} /// namespace comet

namespace
{

  /// Valid semiring operators.
  static const llvm::StringSet<> Semiring_ops{
      "atan2", "div", "eq", "first", "ge", "gt", "hypot",
      "land", "le", "lor", "lt", "max", "minxy", "minus",
      "ne", "pairxy", "plusxy", "pow", "rdiv", "second", "times"};

  /// List of valid semiring operands for reduce op
  static const llvm::StringSet<> Semiring_reduceOps{"any", "land", "lor", "max",
                                                    "minxy", "plusxy", "times",
                                                    "noop"}; /// noop is for monoid op support

  /// Operands' indices in the sptensor_construct function argument list.
  /// ta.sptensor_construct is called to declare a sparse tensor.
  /// The information of the output matrix C
  /// %55 = ta.sptensor_construct(%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %8, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  /// %91 = ta.sptensor_construct(%73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %11, %12) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
  /// The 20-argument one is the current version
  /**
      sptensor_construct(
          0) A1pos,  /// number of rows
          1) A1crd,  /// discard
          2) A2pos,  /// rowptr array
          3) A2crd,  /// col_id array
          4) Aval, /// data array
          5) A1pos_size,
          6) A1crd_size,
          7) A2pos_size,
          8) A2crd_size,
          9) Aval_size,
          10) dim1_size,
          11) dim2_size,
          ------------------------------
          0) A1pos,  /// number of rows
          1) A1crd,  /// discard
          2) A1tile_pos,
          3) A1tile_crd,
          4) A2pos,  /// rowptr array
          5) A2crd,  /// col_id array
          6) A2tile_pos,
          7) A2tile_crd,
          8) Aval, /// data array
          9) A1pos_size,
          10) A1crd_size,
          11) A1tile_pos_size,
          12) A1tile_crd_size,
          13) A2pos_size,
          14) A2crd_size,
          15) A2tile_pos_size,
          16) A2tile_crd_size,
          17) Aval_size,
          18) dim1_size,
          19) dim2_size,
      )
  */
  enum CSR_sptensor_construct_arguments_indices
  {
    CSR_A1POS = 0, /// number of rows
    CSR_A1CRD = 1, /// discard for CSR
    CSR_A1TILE_POS = 2,
    CSR_A1TILE_CRD = 3,
    CSR_A2POS = 4, /// rowptr array
    CSR_A2CRD = 5, /// col_id array
    CSR_A2TILE_POS = 6,
    CSR_A2TILE_CRD = 7,
    CSR_AVAL = 8, /// data array
    CSR_A1POS_SIZE = 9,
    CSR_A1CRD_SIZE = 10,
    CSR_A1TILE_POS_SIZE = 11,
    CSR_A1TILE_CRD_SIZE = 12,
    CSR_A2POS_SIZE = 13,
    CSR_A2CRD_SIZE = 14,
    CSR_A2TILE_POS_SIZE = 15,
    CSR_A2TILE_CRD_SIZE = 16,
    CSR_AVAL_SIZE = 17,
    CSR_DIM1_SIZE = 18,
    CSR_DIM2_SIZE = 19
  };

  /// MASKING_TYPE to indicate what type of masking is used.
  enum MASKING_TYPE
  {
    NO_MASKING = 0,
    PUSH_BASED_MASKING = 1,
    PULL_BASED_MASKING = 2
  };

  /// class MaksingInfo, passed as a parameter to the formSemiringLoopBody() to indicate if using masking or not.
  struct MaskingInfo
  {
  public:
    MASKING_TYPE mask_type;

    mlir::Value mask_tensor;
    mlir::Value mask_rowptr;
    mlir::Value mask_col;
    mlir::Value mask_val;

    /// TODO(zhen.peng): Pull-based mask info and auxiliary variables.

  public:
    MaskingInfo() : mask_type(NO_MASKING) {}

    ///  MaskingInfo(MASKING_TYPE type_, mlir::Value states_) : maskType(type_), states(states_) { }

    void dump()
    {
      switch (mask_type)
      {
      case NO_MASKING:
        std::cout << "maskType: NO_MASKING\n";
        break;
      case PUSH_BASED_MASKING:
        std::cout << "maskType: PUSH_BASED_MASKING "
                  << "mask_tensor: ";
        mask_tensor.dump();
        ///        std::cout << "maskType: PUSH_BASED_MASKING " << "states: ";
        ///        states.dump();
        break;
      case PULL_BASED_MASKING:
        std::cout << "maskType: PULL_BASED_MASKING ... Not supported";
        break;
      }
    }
  };


  /// ----------------- ///
  /// Add declaration of the function comet_index_func;
  /// ----------------- ///
  void declareSortFunc(ModuleOp &module,
                       MLIRContext *ctx,
                       Location loc)
  {
    IndexType indexType = IndexType::get(ctx);

    /// Declare comet_sort_index()
    auto sort_index_func = FunctionType::get(ctx,
                                             {UnrankedMemRefType::get(indexType, 0), indexType, indexType} /* inputs */, {} /* return */);
    std::string func_name = "comet_sort_index";
    if (!hasFuncDeclaration(module, func_name /* func name */))
    {
      func::FuncOp func_declare = func::FuncOp::create(loc,
                                                       func_name,
                                                       sort_index_func,
                                                       ArrayRef<NamedAttribute>{});
      func_declare.setPrivate();
      module.push_back(func_declare);
    }
  }

  Value getSemiringSecondVal(OpBuilder &builder, Location &loc,
                             llvm::StringRef &semiringSecond, Value &Input0, Value &Input1)
  {

    Value elementWiseResult;
    if (semiringSecond == "times")
    {
      elementWiseResult = builder.create<MulFOp>(loc, Input0, Input1);
    }
    else if (semiringSecond == "first")
    {
      elementWiseResult = Input0;
    }
    else if (semiringSecond == "second")
    {
      elementWiseResult = Input1;
    }
    else if (semiringSecond == "atan2")
    {
      elementWiseResult = builder.create<math::Atan2Op>(loc, Input0, Input1);
    }
    else if (semiringSecond == "div")
    {
      elementWiseResult = builder.create<DivFOp>(loc, Input0, Input1);
    }
    else if (semiringSecond == "eq")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OEQ, Input0, Input1);
    }
    else if (semiringSecond == "ge")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OGE, Input0, Input1);
    }
    else if (semiringSecond == "gt")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
    }
    else if (semiringSecond == "le")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OLE, Input0, Input1);
    }
    else if (semiringSecond == "lt")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
    }
    else if (semiringSecond == "land")
    {
      /// land requires integer type input
      llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                   << "land"
                   << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
      exit(1);
    }
    else if (semiringSecond == "lor")
    {
      /// lor requires integer type input
      llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                   << "lor"
                   << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
      exit(1);
    }
    else if (semiringSecond == "lxor")
    {
      /// lxor requires integer type input
      llvm::errs() << "Not supported semiring operator: "
                   << "lxor"
                   << "\n";
    }
    else if (semiringSecond == "minxy")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
      elementWiseResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringSecond == "max")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
      elementWiseResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringSecond == "ne")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, Input0, Input1);
    }
    else if (semiringSecond == "minus")
    {
      elementWiseResult = builder.create<SubFOp>(loc, Input0, Input1);
    }
    else if (semiringSecond == "plusxy")
    {
      elementWiseResult = builder.create<AddFOp>(loc, Input0, Input1);
    }
    else if (semiringSecond == "pairxy")
    {
      elementWiseResult = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(1));
    }
    else if (semiringSecond == "pow")
    {
      elementWiseResult = builder.create<math::PowFOp>(loc, Input0, Input1);
    }
    else
    {
      llvm::errs() << "Not supported semiring operator: " << semiringSecond << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
    }

    return elementWiseResult;
  }

  Value getSemiringFirstVal(OpBuilder &builder, Location &loc,
                            llvm::StringRef &semiringFirst, Value &Input0, Value &Input1)
  {

    Value reduceResult;
    if (semiringFirst == "times")
    {
      reduceResult = builder.create<MulFOp>(loc, Input0, Input1);
    }
    else if (semiringFirst == "plusxy")
    {
      reduceResult = builder.create<AddFOp>(loc, Input0, Input1);
    }
    else if (semiringFirst == "minxy")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
      reduceResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringFirst == "max")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
      reduceResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringFirst == "land")
    {
      /// land requires integer type input
      llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                   << "land"
                   << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
    }
    else if (semiringFirst == "lor")
    {
      /// lor requires integer type input
      llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                   << "lor"
                   << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
    }
    else if (semiringFirst == "any")
    {
      reduceResult = Input1;
    }
    else if (semiringFirst == "noop")
    {
      reduceResult = Input1;
    }
    else
    {
      llvm::errs() << "Not supported semiring operator: " << semiringFirst << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
    }

    return reduceResult;
  }

  struct LowerIndexTreeToSCFPass
      : public PassWrapper<LowerIndexTreeToSCFPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerIndexTreeToSCFPass)
    void runOnOperation() override;

    SetVector<Operation*> collectChildren(IndexTreeIndicesOp root);
    void fillSubtree(Location loc, IRRewriter &rewriter, 
                     const SetVector<Operation*>& subtree,
                     const SmallVector<Value>& old_outputs,
                     ValueRange new_outputs,
                     IRMapping& map);
    void deleteDomain(Operation* op, IRRewriter &rewriter);
    Value convertOperand(IndexTreeLHSOperandOp op, IRRewriter &rewriter);
    Value convertOperand(IndexTreeOperandOp op, IRRewriter &rewriter);
    mlir::LogicalResult convertCompute(Operation* op, IRRewriter &rewriter);
    mlir::LogicalResult convertIndexNode(Operation* op, IRRewriter &rewriter);
  };
}

Value
LowerIndexTreeToSCFPass::convertOperand(IndexTreeOperandOp op, IRRewriter &rewriter)
{
  Location loc = op->getLoc();
  Value tensor = op.getTensor();
  auto crds = op.getCrds();
  auto positions = op.getPos();

  TensorType tensor_type;
  if((tensor_type = llvm::dyn_cast<mlir::TensorType>(tensor.getType()))){
    return rewriter.create<tensor::ExtractOp>(loc, tensor_type.getElementType(), tensor, crds);
  } else {
    Type element_type;
    if(llvm::isa<SparseTensorType>(tensor.getType()))
    {
      element_type = llvm::cast<SparseTensorType>(tensor.getType()).getElementType();
    } else if(llvm::isa<WorkspaceType>(tensor.getType()))
    {
      element_type = llvm::cast<WorkspaceType>(tensor.getType()).getElementType();
    }
    Value pos = positions[positions.size() - 1];
    return rewriter.create<TensorExtractOp>(loc, element_type, tensor, pos);
  }
}

Value
LowerIndexTreeToSCFPass::convertOperand(IndexTreeLHSOperandOp op, IRRewriter &rewriter)
{
  Location loc = op->getLoc();
  Value tensor = op.getTensor();
  auto crds = op.getCrds();
  auto positions = op.getPos();

  TensorType tensor_type;
  if((tensor_type = llvm::dyn_cast<mlir::TensorType>(tensor.getType()))){
    return rewriter.create<tensor::ExtractOp>(loc, tensor_type.getElementType(), tensor, crds);
  } else {
    // LHS may not be constant (i.e. if we are inserting into a tensor that we need to resize), 
    // so cannot directly lower like we can the RHS
    Value pos = positions[positions.size() - 1];
    return rewriter.create<tensorAlgebra::TensorExtractOp>(loc, rewriter.getF64Type(), tensor, pos);
  }
}

mlir::LogicalResult
LowerIndexTreeToSCFPass::convertCompute(Operation *op, 
                                        IRRewriter &rewriter)
{
  Location loc = op->getLoc();
  IndexTreeComputeOp compute_op = llvm::cast<IndexTreeComputeOp>(op);
  auto semiringParts = compute_op.getSemiring().split('_');

  Value elementwise_result;
  for(auto rhs = compute_op.getRhs().begin(); rhs != compute_op.getRhs().end(); rhs++)
  {
    Value rhs_value = convertOperand(llvm::cast<IndexTreeOperandOp>((*rhs).getDefiningOp()), rewriter);
    if(rhs == compute_op.getRhs().begin()){
      elementwise_result = rhs_value;
    } else {
      elementwise_result = getSemiringSecondVal(rewriter, loc, semiringParts.second, 
                                                elementwise_result, rhs_value);
    }
  }

  IndexTreeLHSOperandOp lhs = llvm::cast<IndexTreeLHSOperandOp>(compute_op.getLhs().getDefiningOp());
  Value reduce_result = convertOperand(lhs, rewriter);
  reduce_result = getSemiringFirstVal(rewriter, loc, semiringParts.first, 
                                      reduce_result, elementwise_result);

  Value old_tensor = lhs.getTensor();
  Value output_tensor;

  // LLVM_DEBUG({
  //   lhs->emitOpError() << "Creating Tensor Insert Op" <<  "\n";
  // });
  
  if(llvm::isa<mlir::TensorType>(old_tensor.getType()))
  {
    output_tensor = rewriter.create<tensor::InsertOp>(loc, old_tensor.getType(), reduce_result, old_tensor, lhs.getCrds());
  } else {
    output_tensor = rewriter.create<tensorAlgebra::TensorInsertOp>(loc, old_tensor.getType(), old_tensor, lhs.getPos(), lhs.getCrds(), reduce_result);
  }
  rewriter.replaceAllUsesWith(op->getResult(0), output_tensor);

  std::vector<Value> rhs; 
  for(auto v: compute_op.getRhs())
  {
    rhs.push_back(v);
  }
  rewriter.eraseOp(compute_op);
  for(Value operand : rhs){
    rewriter.eraseOp(operand.getDefiningOp());
  }
  rewriter.eraseOp(lhs);

  return success();
}

SetVector<Operation*>
LowerIndexTreeToSCFPass::collectChildren(IndexTreeIndicesOp root)
{
  SetVector<Operation*> result;
  for(Operation* user : root->getUsers())
  {
    if(llvm::isa<IndexTreeIndicesOp>(user))
    {
      Value domain = llvm::cast<IndexTreeIndicesOp>(user).getDomain();
      Operation* domain_op = domain.getDefiningOp();
      if(llvm::isa<IndexTreeDomainIntersectionOp>(domain_op))
      {
        auto intersection_domain_op = llvm::cast<IndexTreeDomainIntersectionOp>(domain_op);
        for(Value subdomain : intersection_domain_op.getDomains()){
          result.insert(subdomain.getDefiningOp());
        }
      } else if(llvm::isa<IndexTreeDomainUnionOp>(domain_op))
      {
        auto union_domain_op = llvm::cast<IndexTreeDomainUnionOp>(domain_op);
        for(Value subdomain : union_domain_op.getDomains()){
          result.insert(subdomain.getDefiningOp());
        }
      }
      result.insert(domain_op);
    }
    if(llvm::isa<IndexTreeComputeOp>(user))
    {
      IndexTreeComputeOp compute_op = llvm::cast<IndexTreeComputeOp>(user);
      Value lhs = compute_op.getLhs();
      result.insert(lhs.getDefiningOp());
      for(Value rhs_operand : compute_op.getRhs())
      {
        result.insert(rhs_operand.getDefiningOp());
      }
    }
    result.insert(user);
    if(llvm::isa<IndexTreeIndicesOp>(user))
    {
      auto slice = collectChildren(llvm::cast<IndexTreeIndicesOp>(user));
      result.insert(slice.begin(), slice.end());
    }
  }

  return result;
}

void
LowerIndexTreeToSCFPass::fillSubtree(Location loc,
                                     IRRewriter &rewriter, 
                                     const SetVector<Operation*>& subtree,
                                     const SmallVector<Value>& old_outputs,
                                     ValueRange new_outputs,
                                     IRMapping& map)
{
  for(Operation* child : subtree)
  {
    rewriter.clone(*child, map);
  }

  // Create yield
  SmallVector<Value> yield_args;
  for(Value result : old_outputs)
  {
    Value new_result = map.lookup(result);
    yield_args.push_back(new_result);
  }
  rewriter.create<scf::YieldOp>(loc, yield_args);

  auto replacement = new_outputs.begin();
  for(auto old = old_outputs.begin(); old != old_outputs.end(); old++, replacement++)
  {
    rewriter.replaceAllUsesWith(*old, *replacement);
  }

  for(auto r_iter = subtree.rbegin(); r_iter != subtree.rend(); r_iter++)
  {
    Operation* remove = *r_iter;
    // LLVM_DEBUG({
    //   remove->emitOpError() << "Trying to erase op" <<  "\n";
    //   for(auto user : remove->getUsers()) {
    //     user->emitOpError() << "Still used by op" << "\n";
    //   }
    // });
    rewriter.eraseOp(remove);
  }
}

void
LowerIndexTreeToSCFPass::deleteDomain(Operation* op, IRRewriter &rewriter) {
  
  ValueRange subdomains;
  if(llvm::isa<IndexTreeDomainIntersectionOp>(op)){
    subdomains = llvm::cast<IndexTreeDomainIntersectionOp>(op).getDomains();
  } else if(llvm::isa<IndexTreeDomainUnionOp>(op)){
    subdomains = llvm::cast<IndexTreeDomainUnionOp>(op).getDomains();
  }
  std::vector<Value> subdomains_vec;
  for(Value subdomain : subdomains) {
    subdomains_vec.push_back(subdomain);
  }
  rewriter.eraseOp(op);
  
  for(Value subdomain : subdomains_vec) {
    deleteDomain(subdomain.getDefiningOp(), rewriter);
  }
}

mlir::LogicalResult
LowerIndexTreeToSCFPass::convertIndexNode(Operation *op, 
                                          IRRewriter &rewriter) {
  // Generate loop
  auto loc = op->getLoc();
  auto context = rewriter.getContext();
  IndexTreeIndicesOp index_node_op = llvm::cast<IndexTreeIndicesOp>(op);
  Operation* domain_op = index_node_op.getDomain().getDefiningOp();
  auto index_type = rewriter.getIndexType();

  Value crd;
  Value induction_var;
  OpBuilder::InsertPoint before;
  OpBuilder::InsertPoint loop_end;

  SetVector<Operation*> subtree = collectChildren(index_node_op);
  subtree = topologicalSort(subtree);
  SmallVector<Value> loop_outputs;
  SmallVector<Value> loop_init_args;
  SmallDenseMap<std::pair<Value, unsigned>, std::pair<Value, Value>> tensor_access_map;
  IRMapping map;

  
  IndexTreeComputeOp compute_op; 
  ComputeSymbolicDomainOp symbolic_op = nullptr;
  ComputeSymbolicDomainRowOp end_row_op = nullptr;
  for(Operation* child : subtree){
    
    if((compute_op = llvm::dyn_cast<IndexTreeComputeOp>(child)))
    {
      Value loop_output = compute_op->getResult(0);
      Value lhs_tensor = compute_op.getLhs().getDefiningOp()->getOperand(0);
      loop_outputs.push_back(loop_output);

      if(auto clean_op = lhs_tensor.getDefiningOp<IndexTreeCleanWorkspaceOp>()){
        loop_init_args.push_back(clean_op.getWorkspace());
      } else {
        loop_init_args.push_back(lhs_tensor);
      }
      compute_op = nullptr;
    } else if((symbolic_op = llvm::dyn_cast<ComputeSymbolicDomainOp>(child)))
    {
      loop_init_args.push_back(symbolic_op.getDomain());      
    } else if((end_row_op = llvm::dyn_cast<ComputeSymbolicDomainRowOp>(child)))
    {
      loop_outputs.push_back(end_row_op->getResult(0));
    } 
  }
  // TODO: Determining loop outputs won't work if computing two domains at once!!!
  // If we are computing a symbolic domain, but do not see the end of the row, 
  // yield symbolic domain
  if(symbolic_op && !end_row_op) {
    loop_outputs.push_back(symbolic_op->getResult(0));
  }


  before = rewriter.saveInsertionPoint();

  if(llvm::isa<IndexTreeDenseDomainOp>(domain_op))
  {
    // Dense domain
    Value lb = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), 
                                                  rewriter.getIndexAttr(0));
    Value ub = domain_op->getOperand(0);
    Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), 
                                                  rewriter.getIndexAttr(1));
    scf::ForOp for_loop = rewriter.create<scf::ForOp>(loc, lb, ub, step, loop_init_args);
    crd = for_loop.getInductionVar();
    induction_var = crd;    

    unsigned init_arg_idx = 0;
    for(Value init_arg : loop_init_args){
      map.map(init_arg, for_loop.getRegionIterArg(init_arg_idx));
      init_arg_idx += 1;
    }
    rewriter.setInsertionPointToStart(for_loop.getBody());
    fillSubtree(loc, rewriter, subtree, loop_outputs, for_loop.getResults(), map);
    rewriter.setInsertionPoint(for_loop.getBody()->getTerminator());
    loop_end = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(for_loop);


  } else if(llvm::isa<IndexTreeSparseDomainOp>(domain_op))
  {
    // Sparse domain
    IndexTreeSparseDomainOp sparse_domain = llvm::cast<IndexTreeSparseDomainOp>(domain_op);
    TensorFormatEnum format = (TensorFormatEnum)sparse_domain.getFormat();
    switch(format)
    {
      case TensorFormatEnum::CU:
      case TensorFormatEnum::CN: //TODO: Figure out difference
      {
        Value start_idx = sparse_domain.getParent();
        if(!start_idx){
          start_idx = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
        }
        Value inc = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
        Value end_idx = rewriter.create<arith::AddIOp>(loc, index_type, start_idx, inc);
        Value lb = rewriter.create<tensor::ExtractOp>(loc, index_type, sparse_domain.getPos(), start_idx);
        Value ub = rewriter.create<tensor::ExtractOp>(loc, index_type, sparse_domain.getPos(), end_idx);
        Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), 
                                                        rewriter.getIndexAttr(1));
        scf::ForOp for_loop = rewriter.create<scf::ForOp>(loc, lb, ub, step, loop_init_args);
        Block* loop_body = for_loop.getBody();

        rewriter.setInsertionPointToStart(loop_body);
        Value crd_idx = for_loop.getInductionVar();
        induction_var = crd_idx;
        crd = rewriter.create<tensor::ExtractOp>(loc, index_type, sparse_domain.getCrd(), crd_idx);

        
        unsigned init_arg_idx = 0;
        for(Value init_arg : loop_init_args){
          map.map(init_arg, for_loop.getRegionIterArg(init_arg_idx));
          init_arg_idx += 1;
        }
        
        fillSubtree(loc, rewriter, subtree, loop_outputs, for_loop.getResults(), map);
        rewriter.setInsertionPoint(for_loop.getBody()->getTerminator());
        loop_end = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointAfter(for_loop);

        tensor_access_map.insert(std::make_pair(
          std::make_pair(sparse_domain.getTensor(), sparse_domain.getDim()), 
          std::make_pair(crd_idx, crd)
        ));
        break;
      }
      case TensorFormatEnum::S:
      {
        Value crd_idx = sparse_domain.getParent();
        induction_var = crd_idx;
        crd = rewriter.create<tensor::ExtractOp>(loc, index_type, sparse_domain.getCrd(), crd_idx);
        loop_end = rewriter.saveInsertionPoint();
        tensor_access_map.insert(std::make_pair(
          std::make_pair(sparse_domain.getTensor(), sparse_domain.getDim()), 
          std::make_pair(crd_idx, crd)
        ));
        break;
      }
    }
  } else if(llvm::isa<IndexTreeWorkspaceDomainOp>(domain_op)) {
    auto workspace_domain_op = llvm::cast<IndexTreeWorkspaceDomainOp>(domain_op);
    Value start_idx = workspace_domain_op.getParent();
    if(!start_idx){
      start_idx = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
    }
    Value inc = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
    Value end_idx = rewriter.create<arith::AddIOp>(loc, index_type, start_idx, inc);

    /** TODO: Sort crd array? **/
    Value lb = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
    Value ub = rewriter.create<tensorAlgebra::SpTensorGetNNZ>(loc, index_type, workspace_domain_op.getTensor());
    Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), 
                                                    rewriter.getIndexAttr(1));
    scf::ForOp for_loop = rewriter.create<scf::ForOp>(loc, lb, ub, step, loop_init_args);
    Block* loop_body = for_loop.getBody();

    rewriter.setInsertionPointToStart(loop_body);
    Value crd_idx = for_loop.getInductionVar();
    induction_var = crd_idx;
    crd = rewriter.create<tensorAlgebra::SpTensorGetCrd>(loc, index_type, workspace_domain_op.getTensor(), crd_idx, nullptr);

    unsigned init_arg_idx = 0;
    for(Value init_arg : loop_init_args){
      map.map(init_arg, for_loop.getRegionIterArg(init_arg_idx));
      init_arg_idx += 1;
    }
    
    fillSubtree(loc, rewriter, subtree, loop_outputs, for_loop.getResults(), map);
    rewriter.setInsertionPoint(for_loop.getBody()->getTerminator());
    loop_end = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(for_loop);

    tensor_access_map.insert(std::make_pair(
      std::make_pair(workspace_domain_op.getTensor(), 0), 
      std::make_pair(crd_idx, crd)
    ));
  } else if(llvm::isa<IndexTreeDomainIntersectionOp>(domain_op)) {
    // Intersection between sparse domains
    auto domains = llvm::cast<IndexTreeDomainIntersectionOp>(domain_op).getDomains();
    Value inc = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
    SmallVector<Value> loop_conditions;
    SmallVector<Value> array_crds;

    Block* cond_block = new Block();
    Block* body_block = new Block();
    
    // Create loop carried arguments for output tensors and iteration counter
    for(Value init_arg : loop_init_args){
      cond_block->addArgument(init_arg.getType(), loc);
      BlockArgument body_arg = body_block->addArgument(init_arg.getType(), loc);
      map.map(init_arg, body_arg);
    }
    
    Value loop_ctr_init = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
    loop_init_args.push_back(loop_ctr_init);
    cond_block->addArgument(index_type, loc);
    body_block->addArgument(index_type, loc);
    unsigned loop_carry_args = loop_init_args.size();

    // Create control iterators for each of the tensors
    for(Value domain : domains)
    {
      IndexTreeSparseDomainOp sparse_domain = llvm::cast<IndexTreeSparseDomainOp>(domain.getDefiningOp());
      TensorFormatEnum format = (TensorFormatEnum)sparse_domain.getFormat();
      switch(format)
      {
        case TensorFormatEnum::CN:
        case TensorFormatEnum::S:
        {
          // Not yet supported!!!
          break;
        }
        case TensorFormatEnum::CU:
        {
          rewriter.restoreInsertionPoint(before);
          Value start_idx = sparse_domain.getParent();
          if(!start_idx){
            start_idx = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
          }
          Value end_idx = rewriter.create<arith::AddIOp>(loc, index_type, start_idx, inc);
          Value start = rewriter.create<tensor::ExtractOp>(loc, index_type, sparse_domain.getPos(), start_idx);
          Value end = rewriter.create<tensor::ExtractOp>(loc, index_type, sparse_domain.getPos(), end_idx);
          loop_init_args.push_back(start);
          before = rewriter.saveInsertionPoint();

          Value crd_idx = cond_block->addArgument(start.getType(), loc);
          rewriter.setInsertionPointToStart(cond_block);
          Value cnd = rewriter.create<arith::CmpIOp>(
            loc, rewriter.getI1Type(),
            arith::CmpIPredicateAttr::get(context, arith::CmpIPredicate::ult), 
            crd_idx, end
          );
          loop_conditions.push_back(cnd);

          crd_idx = body_block->addArgument(start.getType(), loc);
          rewriter.setInsertionPointToStart(body_block);
          Value array_crd = rewriter.create<tensor::ExtractOp>(loc, index_type, sparse_domain.getCrd(), crd_idx);
          array_crds.push_back(array_crd);

          tensor_access_map.insert(std::make_pair(
            std::make_pair(sparse_domain.getTensor(), sparse_domain.getDim()), 
            std::make_pair(crd_idx, array_crd)
          ));
        }
      }
    }
    rewriter.restoreInsertionPoint(before);

    // Create while loop
    scf::WhileOp while_loop = rewriter.create<scf::WhileOp>(loc, cond_block->getArgumentTypes(), loop_init_args);
    while_loop.getBefore().push_front(cond_block);

    rewriter.setInsertionPointToEnd(cond_block);
    Value loop_condition = nullptr;
    for(Value cnd : loop_conditions)
    {
      if(loop_condition == nullptr)
      {
        loop_condition = cnd;
      } else {
        loop_condition = rewriter.create<arith::AndIOp>(loc, rewriter.getI1Type(), loop_condition, cnd);
      }
    }
    rewriter.create<scf::ConditionOp>(loc, loop_condition, cond_block->getArguments());

    while_loop.getAfter().push_front(body_block);
    // Create intersection
    rewriter.setInsertionPointToEnd(body_block);
    crd = nullptr;
    for(Value array_crd : array_crds){
      if(crd == nullptr)
      {
        crd = array_crd;
      } else {
        crd = rewriter.create<arith::MinUIOp>(loc, index_type, crd, array_crd);
      }
    }

    Value intersection_cnd = nullptr;
    SmallVector<Value> intersections;
    for(Value array_crd : array_crds)
    {
      Value is_intersect = rewriter.create<arith::CmpIOp>(
        loc, rewriter.getI1Type(),
        arith::CmpIPredicateAttr::get(context, arith::CmpIPredicate::eq),
        crd, array_crd
      );
      if(intersection_cnd == nullptr)
      {
        intersection_cnd = is_intersect;
      } else {
        intersection_cnd = rewriter.create<arith::AndIOp>(loc, rewriter.getI1Type(), intersection_cnd, is_intersect);
      }
      intersections.push_back(is_intersect);
    }

    SmallVector<Type> if_types;
    for(unsigned i = 0; i < loop_carry_args; i++)
    {
      if_types.push_back(loop_init_args[i].getType());
    }

    scf::IfOp if_op = rewriter.create<scf::IfOp>(loc, if_types, intersection_cnd, true);
    rewriter.setInsertionPointToStart(if_op.elseBlock());
    rewriter.create<scf::YieldOp>(
      loc, 
      std::vector<Value>(
        body_block->args_begin(),
        body_block->args_begin() + if_op->getNumResults())
    );
    rewriter.setInsertionPointToStart(if_op.thenBlock());
    fillSubtree(loc, rewriter, subtree, loop_outputs, while_loop.getResults(), map);
    Operation* yield_op = if_op.thenBlock()->getTerminator();
    rewriter.setInsertionPoint(yield_op);
    loop_end = rewriter.saveInsertionPoint();

    // Increment the induction variable
    induction_var = body_block->getArgument(loop_outputs.size());
    Value step = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
    Value loop_ctr = rewriter.create<index::AddOp>(loc, index_type, induction_var, step);
    yield_op->insertOperands(yield_op->getNumOperands(), loop_ctr);

    // Increment each argument
    rewriter.setInsertionPointAfter(if_op);
    SmallVector<Value> yield_args;
    for(auto result : if_op.getResults()) {
      yield_args.push_back(result);
    }
    auto cntrl_arg = body_block->args_begin() + loop_outputs.size() + 1;
    for(Value cnd : intersections)
    {
      Value inc = rewriter.create<index::CastUOp>(loc, index_type, cnd);
      yield_args.push_back(rewriter.create<index::AddOp>(loc, index_type, *cntrl_arg, inc));
      cntrl_arg += 1;
    }

    // Create YieldOp
    rewriter.create<scf::YieldOp>(loc, yield_args);
    rewriter.setInsertionPointAfter(while_loop);
  }
  OpBuilder::InsertPoint after = rewriter.saveInsertionPoint();
  
  for(Operation* user : llvm::make_early_inc_range(op->getUsers()))
  {
    if(llvm::isa<IndexTreeIndexToTensorOp>(user))
    {
      IndexTreeIndexToTensorOp access_op = llvm::cast<IndexTreeIndexToTensorOp>(user);
      Value tensor = access_op.getTensor();
      
      if(llvm::isa<tensorAlgebra::SparseTensorType>(tensor.getType()))
      {
        // Tensor is Sparse
        auto tensor_type = llvm::cast<tensorAlgebra::SparseTensorType>(tensor.getType());
        auto dim = access_op.getDim();
        TensorFormatEnum format = (TensorFormatEnum) tensor_type.getFormat()[2 * dim];
        Value access_pos;
        Value access_crd;
        switch(format)
        {
          case TensorFormatEnum::CU:
          case TensorFormatEnum::CN: //TODO: Figure out difference
          case TensorFormatEnum::S:
          {
            auto key = std::make_pair(tensor, dim);
            if(tensor_access_map.find(key) != tensor_access_map.end()){
              auto access_pair = tensor_access_map[key];
              access_pos = access_pair.first;
              access_crd = access_pair.second;
            } else {
              // TODO: Determine weather of not the sequence of accesses is linear or not
              // Right now we blindly say that it will be linear. But this will cause problems
              // A better temporary solution would be to determine that it is not linear,
              // then raise a not implemented error when trying to lower the resulting op
              rewriter.restoreInsertionPoint(loop_end);
              access_crd = crd;
              access_pos = rewriter.create<tensorAlgebra::TensorFindPos>(loc, index_type, tensor, crd, rewriter.getI32IntegerAttr(dim), rewriter.getBoolAttr(true));
            }
            break;
          }
          case TensorFormatEnum::D:
          {
            // TODO: This is incorrect, deal with reordering!!!!
            if(!access_op.getPrevDim()){
              access_pos = crd;
            } else {
              rewriter.restoreInsertionPoint(before);
              Value dim_idx = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
              Value dim_size = rewriter.create<tensorAlgebra::SpTensorGetDimSize>(loc, index_type, tensor, rewriter.getI32IntegerAttr(dim));
              Value pos_start = rewriter.create<arith::MulIOp>(loc, index_type, dim_size, access_op.getPrevDim()); 
              rewriter.restoreInsertionPoint(loop_end);
              access_pos = rewriter.create<arith::AddIOp>(loc, index_type, pos_start,crd);
            }
            access_crd = crd;
          }
        }
        rewriter.replaceAllUsesWith(access_op.getPos(), access_pos);
        rewriter.replaceAllUsesWith(access_op.getCrd(), access_crd);
      } else {
        rewriter.replaceAllUsesWith(access_op.getPos(), crd);
        rewriter.replaceAllUsesWith(access_op.getCrd(), crd);
      }
      rewriter.eraseOp(access_op);
    }
  }

  rewriter.restoreInsertionPoint(loop_end);
  auto users = topologicalSort(llvm::SetVector<Operation*>(op->user_begin(), op->user_end()));
  llvm::SetVector<Operation*> toRemove;
  for(Operation* user : llvm::make_early_inc_range(users))
  {
    LLVM_DEBUG({
      user->emitOpError() << "Converting node from " << op <<  "\n";
    });
    if (llvm::isa<IndexTreeCleanWorkspaceOp>(user))
    {
      auto clean_workspace_op = llvm::cast<IndexTreeCleanWorkspaceOp>(user);
      Value workspace = rewriter.create<WorkspaceClearOp>(loc, user->getResultTypes(), clean_workspace_op.getWorkspace());
      rewriter.replaceOp(user, {workspace});
      toRemove.insert(user);
    }
  }
  for(Operation * user : toRemove) {
    users.remove(user);
  }
  
  for(Operation* user : llvm::make_early_inc_range(users))
  {
    LLVM_DEBUG({
      user->emitOpError() << "Converting node from " << op <<  "\n";
    });

    if(llvm::isa<IndexTreeIndicesOp>(user))
    {
      // Recurse down tree
      if(mlir::failed(convertIndexNode(user, rewriter)))
        return failure();
      // loop_end = rewriter.saveInsertionPoint();

    } else if (llvm::isa<IndexTreeComputeOp>(user))
    {
      // Generate available compute expressions
      if(mlir::failed(convertCompute(user, rewriter)))
        return failure();
    } else if(llvm::isa<ComputeSymbolicDomainOp>(user))
    {
      ComputeSymbolicDomainOp symbolic_domain_op = llvm::cast<ComputeSymbolicDomainOp>(user);
      Value symbolic_domain = symbolic_domain_op.getDomain();
      Value new_domain = rewriter.create<SymbolicDomainInsertOp>(loc, 
                                                                 symbolic_domain.getType(),
                                                                 symbolic_domain,
                                                                 crd,
                                                                 symbolic_domain_op.getIsUniqueAttr());
      rewriter.replaceOp(user, {new_domain});
    }
    else if(llvm::isa<ComputeSymbolicDomainRowOp>(user))
    {
      ComputeSymbolicDomainRowOp symbolic_domain_op = llvm::cast<ComputeSymbolicDomainRowOp>(user);
      Value symbolic_domain = symbolic_domain_op.getDomain();
      Value new_domain = rewriter.create<SymbolicDomainEndRowOp>(loc, 
                                                                 symbolic_domain.getType(),
                                                                 symbolic_domain,
                                                                 symbolic_domain_op.getNeedsMarkAttr());
      rewriter.replaceOp(user, {new_domain});
    }
    else{
      op->getParentOfType<ModuleOp>()->dump();
      assert(false && "Unhandled case");
    }
  }
  rewriter.eraseOp(op);
  deleteDomain(domain_op, rewriter);
  rewriter.restoreInsertionPoint(after);
  return success();
}

void LowerIndexTreeToSCFPass::runOnOperation()
{

  // Convert all the index trees to loops,
  // Happens outside conversion pattern rewriter for convenience
  // TODO: Should this also be part of the conversion pattern rewriter?
  std::vector<IndexTreeRootOp> iTreeRoots;
  func::FuncOp funcOp = getOperation();
  auto *context = &getContext();
  funcOp.walk([&](IndexTreeRootOp op){ iTreeRoots.push_back(op); });
  
  for(auto op : iTreeRoots)
  {
    OpBuilder builder(op);
    IRRewriter rewriter(builder);
    for(Operation* user : op->getUsers())
    {
      if(llvm::isa<IndexTreeIndicesOp>(user))
      {
        if(convertIndexNode(user, rewriter).failed())
        {
          signalPassFailure();
        }
      }
      else if (llvm::isa<IndexTreeComputeOp>(user))
      {
        if(convertCompute(user, rewriter).failed())
        {
          signalPassFailure();
        }
      }
    }
    rewriter.eraseOp(op);
  }
}

/// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createLowerIndexTreeToSCFPass()
{
  return std::make_unique<LowerIndexTreeToSCFPass>();
}
