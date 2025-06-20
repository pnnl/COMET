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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

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
#define COMET_DEBUG_MODE
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

  struct SymbolicDomainInfo {
    Value pos_size;        /// pos array's current size, [the row that is working on], mark value base.
    Value pos_alloc_size;  /// [constant for now, could be dynamic for future] pos array's capacity
    Value crd_size;        /// [private to thread, and set to 0 for each row] crd size of each row
    Value dim_size;        /// [constant] mark array's capcity
    Value pos;             /// [shared by rows] pos array.
    Value mark_array;      /// [private to thread]
  };

  struct NumericSparseTensorInfo {
    Value pos;
    Value crds;
    Value vals;
    Value sparseTensor;
  };

  /// ----------------- ///
  /// Add declaration of the function comet_index_func;
  /// ----------------- ///
  [[maybe_unused]] void declareSortFunc(ModuleOp &module,
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
      elementWiseResult = builder.create<arith::SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringSecond == "max")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
      elementWiseResult = builder.create<arith::SelectOp>(loc, cmp, Input0, Input1);
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
      reduceResult = builder.create<arith::SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringFirst == "max")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
      reduceResult = builder.create<arith::SelectOp>(loc, cmp, Input0, Input1);
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

  struct TensorSubsetInfo {
    int64_t dim;
    int64_t tiles;
    int64_t tile_size;

    void dump() const
    {
      llvm::errs() << "{dim: " << dim << ", tiles: " << tiles << ", tile_size: " << tile_size << "}\n";
    }
  };


  class IndexTreeInferOutputSets {
    SmallDenseMap<IndexTreeIndicesOp, SmallDenseMap<Value, TensorSubsetInfo>> output_sets;

    public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexTreeInferOutputSets)

    IndexTreeInferOutputSets(Operation* op)
    {
      // TODO: Will not work with workspace op or with sparse tensors!
      // Also will not work with indices that don't align to tensor dims!

      IndexTreeOp tree = llvm::cast<IndexTreeOp>(op);
      for(Value input : tree.getBody()->getArguments()) {
        for(auto user : input.getUsers()) {
          if(auto lhs = llvm::dyn_cast<IndexTreeLHSOperandOp>(user)) {
            int64_t dim = 0;
            for(Value crd : lhs.getCrds()) {
              if(auto access = crd.getDefiningOp<IndexTreeIndexToTensorOp>()) {
                auto node = access.getIndex().getDefiningOp<IndexTreeIndicesOp>();
                TensorSubsetInfo slice = {dim, -1, -1};
                output_sets[node].insert(std::make_pair(input, slice));
              }
              dim += 1; 
            }
          }
        }
      }
    }

    SmallDenseMap<Value, TensorSubsetInfo> getOutputSets(IndexTreeIndicesOp op)
    {
      if(output_sets.contains(op)){
        return output_sets[op];
      }

      // This also will cause errors!!!
      return SmallDenseMap<Value, TensorSubsetInfo>();
    }

  };

  class IndexVar {
    public:
      virtual Value getCrd(IRRewriter& rewriter) = 0;
      virtual Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) = 0;
      virtual ~IndexVar(){};
  };

  class LoopInfo : public IndexVar {
    protected:
      SmallVector<Value> currentInputs;
      ValueRange results;

    public:
      Operation* loopBody;
      IRMapping map; // Maps values from outer scope (i.e. TreeRegion) to loop

      LoopInfo(ValueRange inputs, ValueRange outputs, Operation* body, IRMapping& ir_map):
        currentInputs(inputs), results(outputs), loopBody(body), map(ir_map) {}

      virtual Value getCrd(IRRewriter& rewriter) override = 0;
      virtual Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override = 0;
      virtual void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) = 0;
      virtual ~LoopInfo(){};

      ValueRange getInputs(){return currentInputs;}
      Value getInput(uint32_t idx) {return currentInputs[idx];}
      virtual Value getInput(IRRewriter& rewriter, uint32_t idx, SmallVector<int> dims) {
        return getInput(idx);
      }
      virtual ValueRange getResults() {return results;}
  };

  class SentinelLoopInfo : LoopInfo {
    private:
      indexTree::YieldOp terminator;
    public:
      SentinelLoopInfo(ValueRange inputs, ValueRange outputs, Operation* body, IRMapping& ir_map, indexTree::YieldOp yield_op):
        LoopInfo(inputs, outputs, body, ir_map), terminator(yield_op) {}

      static LoopInfo* build(ValueRange inputs, ValueRange outputs, Operation* body, IRMapping& ir_map, indexTree::YieldOp yield_op) {
        return new SentinelLoopInfo(inputs, outputs, body, ir_map, yield_op);
      }

      virtual Value getCrd(IRRewriter& rewriter) override {return nullptr;}
      virtual Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {return nullptr;}
      virtual void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        rewriter.modifyOpInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  class DenseLoopInfo : LoopInfo {
    private:
      Value inductionVar;
      scf::YieldOp terminator;

    public:
      DenseLoopInfo(ValueRange inputs, 
                    ResultRange outputs,
                    Operation* body,
                    IRMapping ir_map,
                    Value i, 
                    scf::YieldOp yield): 
        LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), terminator(yield){}

      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs)
      {
        auto loc = domain_op->getLoc();
        Value lb = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
        Value ub = domain_op->getOperand(0);
        Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(1));
        scf::ForOp for_loop = rewriter.create<scf::ForOp>(loc, lb, ub, step, inputs);
        
        IRMapping map;
        rewriter.setInsertionPointToStart(for_loop.getBody());
        auto yield_op = rewriter.create<scf::YieldOp>(loc, for_loop.getRegionIterArgs());
        rewriter.setInsertionPointAfter(for_loop);
        return new DenseLoopInfo(for_loop.getRegionIterArgs(), for_loop->getResults(), yield_op, map, for_loop.getInductionVar(), yield_op);
      }

      Value getCrd(IRRewriter& rewriter) override {return inductionVar;}

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        if(dyn_cast<TensorType>(tensor.getType())){
          return inductionVar;
        } else if(SparseTensorType tt = dyn_cast<SparseTensorType>(tensor.getType())){
          if((TensorFormatEnum)(tt.getFormat()[2 * dim]) == TensorFormatEnum::D) {
            return inductionVar;
          }
          assert(false && "Invalid type passed to DenseLoopInfo getPos");
          return nullptr;
         }
        assert(false && "Invalid type passed to DenseLoopInfo getPos");
        return nullptr;
      }

      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
       rewriter.modifyOpInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  class DenseParallelLoopInfo : public LoopInfo {
    private:
      Value inductionVar;
      SmallVector<Operation*> terminator_ops;
      SmallDenseMap<Value, TensorSubsetInfo> output_sets;

    public:
      DenseParallelLoopInfo(ValueRange inputs, 
                            ResultRange outputs,
                            Operation* body,
                            IRMapping ir_map,
                            Value i,
                            SmallVector<Operation*>& terminator_ops,
                            SmallDenseMap<Value, TensorSubsetInfo> output_sets): 
        LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), terminator_ops(terminator_ops), output_sets(output_sets) {}

      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs, SmallDenseMap<Value, TensorSubsetInfo> output_sets)
      {
        auto loc = domain_op->getLoc();
        Value ub = domain_op->getOperand(0);
        scf::ForallOp for_loop = rewriter.create<scf::ForallOp>(
          loc,
          ArrayRef<OpFoldResult>(ub),
          inputs,
          std::nullopt,
          nullptr);
        Value inductionVar = for_loop.getInductionVar(0);
        rewriter.setInsertionPointToStart(for_loop.getBody());
        {/// test
          comet_vdump(domain_op->getParentOfType<ModuleOp>());
          comet_pdump(domain_op);
          comet_vdump(ub);
          comet_vdump(for_loop);
          for ([[maybe_unused]] auto &input : for_loop.getOutputsMutable())
          {
            comet_vdump(input.get());
            comet_vdump(output_sets.at(input.get()));
          }
        }

        SmallVector<Value> input_slices;
        SmallVector<Value> input_for_loop_args;
        for(auto& input : for_loop.getOutputsMutable())
        {
          auto& os = output_sets.at(input.get());
          comet_vdump(input.get());
          comet_vdump(os);
          RankedTensorType tt;
          if (llvm::isa<RankedTensorType>(input.get().getType()))
          {
            tt = llvm::cast<RankedTensorType>(input.get().getType());
          }
          else if (llvm::isa<indexTree::SymbolicDomainType>(input.get().getType()) /* for symbolic phase's symbolic_domain */
              || llvm::isa<tensorAlgebra::SparseTensorType>(input.get().getType()) /* for numeric phase's sparse_tensor */
              || llvm::isa<tensorAlgebra::WorkspaceType>(input.get().getType()) /* for numeric phase's workspace */)
          { /// A dummy RankedTensorType. Later, the pos array is supposed to be the same type.
            tt = mlir::RankedTensorType::get(/*shape=*/{ShapedType::kDynamic}, /*elementType=*/rewriter.getI64Type());
          }
//          else if (llvm::isa<tensorAlgebra::WorkspaceType>(input.get().getType()))
//          {
//            continue;
//          }
          else
          {
            comet_vdump(input.get());
            assert(false && "Non supported type of input.");
          }
//          RankedTensorType tt = llvm::cast<RankedTensorType>(input.get().getType());
          int64_t nDims = tt.getRank();
          SmallVector<Value> sizes;
          for(int i = 0; i < nDims; i++)
          {
            if(i != os.dim && tt.getDimSize(i) == ShapedType::kDynamic) {
              Value idx = rewriter.create<index::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(i));
              sizes.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.getIndexType(), input.get(), idx));
            }
          }

          SmallVector<int64_t> static_offsets(nDims, 0);
          SmallVector<int64_t> static_sizes(tt.getShape());
          static_offsets[os.dim] = ShapedType::kDynamic;
          static_sizes[os.dim] = 1;
          {
            for ([[maybe_unused]] auto size : sizes) {
              comet_debug() << "size: " << size << "\n";
            }
            for ([[maybe_unused]] auto size : static_offsets) {
              comet_debug() << "static_offset: " << size << "\n";
            }
            for ([[maybe_unused]] auto size : static_sizes) {
              comet_debug() << "static_size: " << size << "\n";
            }
          }

          auto slice = rewriter.create<tensor::ExtractSliceOp>(
            loc,
            RankedTensorType::get(static_sizes, tt.getElementType()),
            for_loop.getTiedBlockArgument(&input),
            ValueRange(inductionVar),
            sizes,
            ValueRange(),
            rewriter.getDenseI64ArrayAttr(static_offsets),
            rewriter.getDenseI64ArrayAttr(static_sizes),
            rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(nDims, 1))
          );
          input_slices.push_back(slice->getResult(0));
          output_sets.insert(std::make_pair(slice->getResult(0), os));
          input_for_loop_args.push_back(for_loop.getTiedBlockArgument(&input));
          {
            comet_vdump(slice->getResult(0));
          }
        }

        SmallVector<Operation*> terminator_ops;
        auto par_op = for_loop.getTerminator();
        auto slice = input_slices.begin();
        {
          comet_vdump(par_op);
          comet_pdump(slice);
        }
        for(auto& input : for_loop.getOutputsMutable()) {
//          if (llvm::isa<tensorAlgebra::WorkspaceType>(input.get().getType()))
//          {
//            continue;
//          }
          rewriter.setInsertionPoint(par_op);
          auto& os = output_sets.at(input.get());
          auto tt = cast<RankedTensorType>(slice->getType());
          int64_t nDims = tt.getRank();
          SmallVector<Value> sizes;
          for(int i = 0; i < nDims; i++)
          {
            if(tt.getDimSize(i) == ShapedType::kDynamic) {
              Value idx = rewriter.create<index::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(i));
              sizes.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.getIndexType(), input.get(), idx));
            }
          }

          rewriter.setInsertionPointToEnd(par_op.getBody());
          SmallVector<int64_t> static_offsets(nDims, 0);
          static_offsets[os.dim] = ShapedType::kDynamic;
          {
            comet_vdump(os);
            comet_vdump(tt);
            for ([[maybe_unused]] auto size : sizes) {
              comet_debug() << "size: " << size << "\n";
            }
            for ([[maybe_unused]] auto size : static_offsets) {
              comet_debug() << "static_offset: " << size << "\n";
            }
            for ([[maybe_unused]] auto size : tt.getShape()) {
              comet_debug() << "static_size: " << size << "\n";
            }
          }
          Operation* insert = rewriter.create<tensor::ParallelInsertSliceOp>(
            loc, 
            *slice,
            for_loop.getTiedBlockArgument(&input),
            inductionVar,
            sizes,
            ValueRange(),
            rewriter.getDenseI64ArrayAttr(static_offsets),
            rewriter.getDenseI64ArrayAttr(tt.getShape()),
            rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(nDims, 1))
          );
          comet_pdump(insert);
          terminator_ops.push_back(insert);
          slice++;
        }
        rewriter.setInsertionPointAfter(for_loop);

        for ([[maybe_unused]] auto arg : input_for_loop_args)
        {
          comet_vdump(arg);
        }

        IRMapping map;
        if (inputs.size() == 1 && llvm::isa<indexTree::SymbolicDomainType>(inputs.front().getType()))
        { /// Use the symbolic_domain argument as the inputs.
          return new DenseParallelLoopInfo(input_for_loop_args, for_loop.getResults(), par_op, map, inductionVar,
                                           terminator_ops, output_sets);
        }
        else
        {
          return new DenseParallelLoopInfo(input_slices, for_loop.getResults(), par_op, map, inductionVar,
                                           terminator_ops, output_sets);
        }
      }

      Value getCrd(IRRewriter& rewriter) override {return inductionVar;}

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        if(dyn_cast<TensorType>(tensor.getType())){
          // TODO: Fix me. Right now, getCrd on the output tensor is just zero becuase 
          // we have already extracted the slice that we want. This will only work 
          // in very limited scenarios
          if(std::find(currentInputs.begin(), currentInputs.end(), tensor) != currentInputs.end()){
            Value zero = rewriter.create<index::ConstantOp>(tensor.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
            return zero;
          } else {
            return inductionVar;
          }
          
        } else if(SparseTensorType tt = dyn_cast<SparseTensorType>(tensor.getType())){
          if((TensorFormatEnum)(tt.getFormat()[2 * dim]) == TensorFormatEnum::D) {
            return inductionVar;
          }
          assert(false && "Invalid type passed to DenseLoopInfo getPos");
          return nullptr;
        }
        assert(false && "Invalid type passed to DenseLoopInfo getPos");
        return nullptr;
      }

      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        Operation* terminator = terminator_ops[idx];
        rewriter.modifyOpInPlace(terminator, [&](){terminator->setOperand(0, newOutput);});
      }
  };

  class SparseLoopInfo : LoopInfo {
    private:
      Value inductionVar;
      scf::YieldOp terminator;
      Value controlTensor;
      uint32_t dim;
      Value crd;

    public:
      SparseLoopInfo(ValueRange inputs, ResultRange outputs, Operation* body, IRMapping ir_map, Value i, scf::YieldOp yield, Value tensor, uint32_t d): 
          LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), terminator(yield), controlTensor(tensor), dim(d), crd(nullptr) {}
        
      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs)
      {
        auto loc = domain_op->getLoc();
        auto sparse_domain = cast<IndexTreeSparseDomainOp>(domain_op);
        auto index_type = rewriter.getIndexType();

        Value start_idx = sparse_domain.getParent();
        if(!start_idx){
          start_idx = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
        }

        Value inc = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
        Value end_idx = rewriter.create<arith::AddIOp>(loc, index_type, start_idx, inc);
        Value lb = rewriter.create<tensor::ExtractOp>(loc, sparse_domain.getPos(), start_idx);
        lb = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), lb);
        Value ub = rewriter.create<tensor::ExtractOp>(loc, sparse_domain.getPos(), end_idx);
        ub = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), ub);
        Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), 
                                                        rewriter.getIndexAttr(1));
        scf::ForOp for_loop = rewriter.create<scf::ForOp>(loc, lb, ub, step, inputs);

        IRMapping map;
        unsigned init_arg_idx = 0;
        for(Value init_arg : inputs){
          map.map(init_arg, for_loop.getRegionIterArg(init_arg_idx));
          init_arg_idx += 1;
        }
        
        rewriter.setInsertionPointToStart(for_loop.getBody());
        auto yield_op = rewriter.create<scf::YieldOp>(loc, for_loop.getRegionIterArgs());
        rewriter.setInsertionPointAfter(for_loop);

        return new SparseLoopInfo(ValueRange(for_loop.getRegionIterArgs()), for_loop.getResults(), yield_op, map, for_loop.getInductionVar(), yield_op, sparse_domain.getTensor(), sparse_domain.getDim());
      }

      Value getCrd(IRRewriter& rewriter) override {
        // if(crd != nullptr) return crd;
        auto loc = controlTensor.getLoc();
        // SparseTensorType tt = cast<SparseTensorType>(controlTensor.getType());
        crd = rewriter.create<SpTensorGetCrd>(loc, controlTensor, inductionVar, rewriter.getI32IntegerAttr(dim));
        crd = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), crd);
        return crd;
      }

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        if(tensor == controlTensor && dim == this->dim){
          return inductionVar;
        }

        if(dyn_cast<TensorType>(tensor.getType()) || dyn_cast<WorkspaceType>(tensor.getType())){
          return crd;
        } else if(SparseTensorType tt = dyn_cast<SparseTensorType>(tensor.getType())){
          if((TensorFormatEnum)(tt.getFormat()[2 * dim]) == TensorFormatEnum::D) {
            return crd;
          }
        }
        auto loc = controlTensor.getLoc();
        Value crd = rewriter.create<SpTensorGetCrd>(loc, controlTensor, inductionVar, rewriter.getI32IntegerAttr(dim));
        crd = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), crd);
        Value pos = rewriter.create<tensorAlgebra::TensorFindPos>(loc, rewriter.getIndexType(), tensor, crd, rewriter.getI32IntegerAttr(dim), rewriter.getBoolAttr(true));
        return pos;
      }

      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        rewriter.modifyOpInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  class SingletonLoopInfo : LoopInfo {
    private:
      LoopInfo* parent;
      Value inductionVar;
      Value tensor;
      int32_t dim;

    public:
      SingletonLoopInfo(ValueRange inputs, Operation* body, IRMapping ir_map, LoopInfo* parent_info, Value i, Value sparse_tensor, int32_t dim) : 
        LoopInfo(inputs, ValueRange(inputs), body, ir_map), parent(parent_info), inductionVar(i), tensor(sparse_tensor), dim(dim)
        {}
      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs, LoopInfo* parent_info)
      {
        auto sparse_domain = cast<IndexTreeSparseDomainOp>(domain_op);
        Value inductionVar = sparse_domain.getParent();
        IRMapping map;
        return new SingletonLoopInfo(inputs, &(*(rewriter.saveInsertionPoint().getPoint())), map, parent_info, inductionVar, sparse_domain.getTensor(), sparse_domain.getDim());
      }

      Value getCrd(IRRewriter& rewriter) override {
        auto loc = rewriter.getUnknownLoc(); // This is not correct
        // auto tt = RankedTensorType::get({ShapedType::kDynamic}, rewriter.getIndexType());
        Value crd_tensor = rewriter.create<SpTensorGetDimCrd>(loc, tensor, rewriter.getI32IntegerAttr(dim));
        Value crd = rewriter.create<tensor::ExtractOp>(loc, crd_tensor, inductionVar);
        crd = rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), crd);
        return crd;
      }

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        // This is needed 
        if(llvm::isa<TensorType>(tensor.getType())) {
          // For dense tensors, positions and crd should be the same. 
          return getCrd(rewriter);
        }
        return inductionVar;
      }


      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        parent->updateOutput(rewriter, idx, newOutput);
      }

      ValueRange getResults() override {
        return ValueRange(currentInputs);
      }
      
  };

  class WorkspaceLoopInfo : LoopInfo {
    private:
      Value inductionVar;
      scf::YieldOp terminator;
      Value workspaceTensor;
      Value crd;

    public:
      WorkspaceLoopInfo(ValueRange inputs, ResultRange outputs, Operation* body, IRMapping ir_map, Value i, scf::YieldOp yield, Value workspace) : 
        LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), terminator(yield), workspaceTensor(workspace)  {}

      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs)
      {
        auto loc = domain_op->getLoc();
        auto workspace_domain_op = llvm::cast<IndexTreeWorkspaceDomainOp>(domain_op);
        auto index_type = rewriter.getIndexType();
        
        Value workspace = workspace_domain_op.getTensor();
        Type workspace_type = workspace.getType();
        Value sorted_workspace = rewriter.create<tensorAlgebra::SortCrdOp>(loc, workspace_type, workspace);
        Value lb = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
        Value ub = rewriter.create<tensorAlgebra::SpTensorGetNNZ>(loc, index_type, sorted_workspace);
        Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), 
                                                        rewriter.getIndexAttr(1));
        int32_t workspace_idx = -1;
        SmallVector<Value, 4> mutable_inputs = SmallVector<Value>(inputs.begin(), inputs.end());
        for(unsigned i = 0; i < mutable_inputs.size(); i++) {
          if(mutable_inputs[i] == workspace){
            mutable_inputs[i] = sorted_workspace;
            workspace_idx = i;
            break;
          }
        }
        scf::ForOp for_loop = rewriter.create<scf::ForOp>(loc, lb, ub, step, ValueRange(mutable_inputs));
        Block* loop_body = for_loop.getBody();
        Value loop_workspace = for_loop.getRegionIterArg(workspace_idx);

        rewriter.setInsertionPointToStart(loop_body);
        IRMapping map;
        auto yield_op = rewriter.create<scf::YieldOp>(loc, for_loop.getRegionIterArgs());
        rewriter.setInsertionPointAfter(for_loop);
        return new WorkspaceLoopInfo(ValueRange(for_loop.getRegionIterArgs()), for_loop.getResults(), yield_op, map, for_loop.getInductionVar(), yield_op, loop_workspace);
      }

      Value getCrd(IRRewriter& rewriter) override {
        if(crd != nullptr) return crd;
        auto loc = workspaceTensor.getLoc();
        crd = rewriter.create<SpTensorGetCrd>(loc, workspaceTensor, inductionVar, rewriter.getI32IntegerAttr(0));
        crd = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), crd);

        return crd;
      }

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        if(llvm::isa<SparseTensorType>(tensor.getType()))
        {
          auto loc = rewriter.getUnknownLoc();
          return rewriter.create<TensorFindPos>(loc, rewriter.getIndexType(), tensor, nullptr, (int32_t)dim, true);
        }
        return inductionVar;
      }


      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        rewriter.modifyOpInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  class IntersectionLoopInfo : LoopInfo {
    private:
      Value inductionVar;
      Value crd;
      scf::YieldOp terminator;
      llvm::SmallDenseMap<std::pair<Value, uint32_t>, std::pair<Value, Value>> controlVars;
    public:
      IntersectionLoopInfo(ValueRange inputs, ResultRange outputs, Operation* body, IRMapping ir_map, Value i, Value c, scf::YieldOp yield, SmallDenseMap<std::pair<Value, uint32_t>, std::pair<Value, Value>> controls) : 
          LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), crd(c), terminator(yield), controlVars(controls){}

      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs)
      {
        auto loc = domain_op->getLoc();
        auto intersection_domain_op = llvm::cast<IndexTreeDomainIntersectionOp>(domain_op);
        auto index_type = rewriter.getIndexType();
        auto context = rewriter.getContext();

        SmallVector<Value> loop_args = SmallVector<Value>(inputs);

        SmallDenseMap<std::pair<Value, unsigned>, std::pair<Value, Value>> tensor_access_map;

        // Intersection between sparse domains
        auto domains = intersection_domain_op.getDomains();
        Value inc = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
        SmallVector<Value> loop_conditions;
        SmallVector<Value> array_crds;

        Block* cond_block = new Block();
        Block* body_block = new Block();
        
        // Create loop carried arguments for output tensors and iteration counter
        IRMapping map;
        for(Value init_arg : loop_args){
          cond_block->addArgument(init_arg.getType(), loc);
          BlockArgument body_arg = body_block->addArgument(init_arg.getType(), loc);
          map.map(init_arg, body_arg);
        }
        
        Value loop_ctr_init = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
        loop_args.push_back(loop_ctr_init);
        cond_block->addArgument(index_type, loc);
        body_block->addArgument(index_type, loc);
        unsigned loop_carry_args = loop_args.size();

        // Create control iterators for each of the tensors
        OpBuilder::InsertPoint before;
        for(Value domain : domains)
        {
          IndexTreeSparseDomainOp sparse_domain = llvm::cast<IndexTreeSparseDomainOp>(domain.getDefiningOp());
          TensorFormatEnum format = (TensorFormatEnum)sparse_domain.getFormat();
          switch(format)
          {
            case TensorFormatEnum::D:
            case TensorFormatEnum::UNK:
            {
              assert(false && "Invalid format for IndexTreeSparseDomainOp");
              break;
            }
            case TensorFormatEnum::CN:
            case TensorFormatEnum::S:
            {
              // Not yet supported!!!
              break;
            }
            case TensorFormatEnum::CU:
            {
              Value start_idx = sparse_domain.getParent();
              if(!start_idx){
                start_idx = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
              }
              Value end_idx = rewriter.create<arith::AddIOp>(loc, index_type, start_idx, inc);
              Value start = rewriter.create<tensor::ExtractOp>(loc, sparse_domain.getPos(), start_idx);
              start = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), start);
              Value end = rewriter.create<tensor::ExtractOp>(loc, sparse_domain.getPos(), end_idx);
              end = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), end);
              loop_args.push_back(start);
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
              auto dim = rewriter.getI32IntegerAttr(sparse_domain.getDim());
              Value array_crd = rewriter.create<SpTensorGetCrd>(loc, sparse_domain.getTensor(), crd_idx, dim);
              array_crd = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), array_crd);
              array_crds.push_back(array_crd);

              tensor_access_map.insert(std::make_pair(
                std::make_pair(sparse_domain.getTensor(), sparse_domain.getDim()), 
                std::make_pair(crd_idx, array_crd)
              ));
            }
          }
          rewriter.restoreInsertionPoint(before);
        }

        // Create while loop
        scf::WhileOp while_loop = rewriter.create<scf::WhileOp>(loc, cond_block->getArgumentTypes(), loop_args);
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
        Value crd = nullptr;
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
          if_types.push_back(loop_args[i].getType());
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
        // OpBuilder::InsertPoint loop_end = rewriter.saveInsertionPoint();
        Value induction_var = body_block->getArgument(loop_carry_args - 1);
        auto step = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
        auto loop_ctr = rewriter.create<index::AddOp>(loc, index_type, induction_var, step);
        auto yield_op = rewriter.create<scf::YieldOp>(
          loc, 
          std::vector<Value>(
            body_block->args_begin(),
            body_block->args_begin() + if_op->getNumResults())
        );
        yield_op->setOperand(loop_carry_args - 1, loop_ctr.getResult());

        // Increment each argument
        rewriter.setInsertionPointAfter(if_op);
        SmallVector<Value> yield_args;
        for(auto result : if_op.getResults()) {
          yield_args.push_back(result);
        }
        auto cntrl_arg = body_block->args_begin() + loop_carry_args;
        for(Value cnd : intersections)
        {
          Value inc = rewriter.create<index::CastUOp>(loc, index_type, cnd);
          yield_args.push_back(rewriter.create<index::AddOp>(loc, index_type, *cntrl_arg, inc));
          cntrl_arg += 1;
        }

        // Create YieldOp
        rewriter.create<scf::YieldOp>(loc, yield_args);
        rewriter.setInsertionPointAfter(while_loop);

        ValueRange inner_inputs = body_block->getArguments().drop_back(loop_args.size() - loop_carry_args - 1);
        ResultRange outputs = ResultRange(while_loop->result_begin(), while_loop->result_begin() + (loop_carry_args - 1));
        return new IntersectionLoopInfo(inner_inputs, outputs, loop_ctr, map, induction_var, crd, yield_op, tensor_access_map);
      }

      Value getCrd(IRRewriter& rewriter) override {
        return crd;
      }

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        auto control = controlVars.find(std::make_pair(tensor, dim));
        if(control != controlVars.end()) 
          return control->getSecond().first;
        auto loc = tensor.getLoc();
        Value pos = rewriter.create<tensorAlgebra::TensorFindPos>(loc, rewriter.getIndexType(), tensor, crd, rewriter.getI32IntegerAttr(dim), rewriter.getBoolAttr(true));
        return pos;
      }


      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        rewriter.modifyOpInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  class MaskedLoopInfo : LoopInfo {
    private:
      scf::YieldOp terminator;
      LoopInfo* internal_loop;
    public:
      MaskedLoopInfo(ValueRange inputs, ValueRange outputs, Operation* body, IRMapping ir_map, scf::YieldOp yield, LoopInfo* internal) : 
          LoopInfo(inputs, outputs, body, ir_map), terminator(yield), internal_loop(internal) {}

      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs)
      {
        auto loc = domain_op->getLoc();
        auto masked_domain = llvm::cast<IndexTreeMaskedDomainOp>(domain_op);
        [[maybe_unused]] auto index_type = rewriter.getIndexType();
        [[maybe_unused]] auto context = rewriter.getContext();

        // Create internal loop
        Operation* child = masked_domain.getBase().getDefiningOp();
        LoopInfo* loop_info = llvm::TypeSwitch<Operation*, LoopInfo*>(child) 
        .Case<IndexTreeDenseDomainOp>([&](IndexTreeDenseDomainOp op) {
          return DenseLoopInfo::build(op, rewriter, inputs);
        })
        .Case<IndexTreeSparseDomainOp>([&](IndexTreeSparseDomainOp op) {
          switch((TensorFormatEnum)op.getFormat()){
            case TensorFormatEnum::D:
            case TensorFormatEnum::UNK:
            {
              assert(false && "Invalid format for IndexTreeSparseDomainOp");
              return (LoopInfo*)nullptr;

              break;
            }
            case TensorFormatEnum::CN:
            case TensorFormatEnum::CU:
              return SparseLoopInfo::build(op, rewriter, inputs);
            case TensorFormatEnum::S:
              assert(false && "Singleton loop inside a mask is not supported.");
              // return SingletonLoopInfo::build(op, rewriter, inputs, parent_info);
          }
        })
        .Case<IndexTreeWorkspaceDomainOp>([&](IndexTreeWorkspaceDomainOp op) {
          return WorkspaceLoopInfo::build(op, rewriter, inputs);
        })
        .Case<IndexTreeDomainIntersectionOp>([&](IndexTreeDomainIntersectionOp op) {
          return IntersectionLoopInfo::build(op, rewriter, inputs);
        })
        .Default([](Operation *op) {
          assert(false && "IndexNode not given a valid domain");
          return nullptr;
        });
        auto after_internal_loop = rewriter.saveInsertionPoint();

        // Create if op
        rewriter.setInsertionPoint(loop_info->loopBody);
        SmallVector<Type> if_types;
        for(Value input : loop_info->getInputs())
        {
          if_types.push_back(input.getType());
        }
        Value cond = rewriter.create<tensor::ExtractOp>(loc, masked_domain.getMask(), loop_info->getCrd(rewriter));
        auto if_op = rewriter.create<scf::IfOp>(loc, if_types, cond, true);
        rewriter.setInsertionPointToStart(if_op.elseBlock());
        rewriter.create<scf::YieldOp>(loc, loop_info->getInputs());
        rewriter.setInsertionPointToStart(if_op.thenBlock());
        SmallVector<Value> new_inputs = SmallVector<Value>(loop_info->getInputs().begin(), loop_info->getInputs().end());
        scf::YieldOp terminator = rewriter.create<scf::YieldOp>(loc, new_inputs);
        uint32_t i = 0;
        for(Value output : if_op.getResults()){
          loop_info->updateOutput(rewriter, i, output);
          i += 1;
        }
        rewriter.restoreInsertionPoint(after_internal_loop);

        return new MaskedLoopInfo(new_inputs, loop_info->getResults(), terminator, loop_info->map, terminator, loop_info);
      }

      Value getCrd(IRRewriter& rewriter) override {
        return internal_loop->getCrd(rewriter);
      }

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        return internal_loop->getPos(rewriter, tensor, dim);
      }

      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        rewriter.modifyOpInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

//  struct TransformSymbolicForallOp
//      : public OpConversionPattern<scf::ForallOp> {
//    using OpConversionPattern<scf::ForallOp>::OpConversionPattern;
//    LogicalResult matchAndRewrite(scf::ForallOp op,
//                                  OpAdaptor adaptor,
//                                  ConversionPatternRewriter &rewriter) const override
//    {
//      comet_vdump(op);
//      return success();
//    }
//  };
//  struct TransformSymbolicForallOp
//      : public OpRewritePattern<scf::ForallOp> {
//    TransformSymbolicForallOp(mlir::MLIRContext *context)
//      : OpRewritePattern<scf::ForallOp>(context) {}
//
//    LogicalResult matchAndRewrite(scf::ForallOp op,
//                                  PatternRewriter &rewriter) const override
//    {
//      comet_vdump(op);
//      return success();
//    }
//  };
//  struct TransformSymbolicForallOp
//      : public ConversionPattern {
//    TransformSymbolicForallOp(mlir::MLIRContext *context)
//      : ConversionPattern(scf::ForallOp::getOperationName(), 1, context) {}
//
//    LogicalResult matchAndRewrite(mlir::Operation *op,
//                                  ArrayRef<Value> operands,
//                                  mlir::ConversionPatternRewriter &rewriter) const override
//    {
//      comet_pdump(op);
//      return success();
//    }
//  };

  scf::ForallOp CreateNewForallOp(scf::ForallOp old_forall_loop,
                                  mlir::IRRewriter &rewriter,
                                  mlir::Location &loc,
                                  SymbolicDomainInfo &symbolic_domain_info /*out*/)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(old_forall_loop);

    /// Create the `builtin.unrealized_conversion_cast`.
    Value old_symbolic_domain = old_forall_loop.getOutputsMutable()[0].get();
    IndexType indexType = rewriter.getIndexType();
    RankedTensorType rankedTensorType = mlir::RankedTensorType::get(/*shape=*/{ShapedType::kDynamic}, /*elementType=*/rewriter.getI64Type());
    llvm::SmallVector<Type> result_types = {
        indexType,         /// pos_size
        indexType,         /// pos_alloc_size
        indexType,         /// crd_size
        indexType,         /// dim_size
        rankedTensorType,  /// pos
        rankedTensorType   /// mark_array
    };
    /// Get the pos, which will be the new `scf.forall`'s operand.
    mlir::UnrealizedConversionCastOp unrealized_op =
        rewriter.create<mlir::UnrealizedConversionCastOp>(loc,
                                                          result_types,
                                                /*input=*/old_symbolic_domain);
    symbolic_domain_info.pos_alloc_size = unrealized_op.getResult(1);
    symbolic_domain_info.dim_size = unrealized_op.getResult(3);
    Value pos = unrealized_op->getResult(4);
    symbolic_domain_info.mark_array = unrealized_op.getResult(5);
    comet_vdump(pos);

    /// This does not work, because it only changes the for-loop's operand, but doesn't change the uses of this operand in side the loop.
//    rewriter.replaceAllUsesWith(old_forall_loop.getOperand(1), pos);
//    comet_vdump(old_forall_loop);

    /// Create the new `scf.forall`
    scf::ForallOp new_forall_loop = rewriter.create<scf::ForallOp>(
        loc,
        /*upperBound=*/old_forall_loop.getMixedUpperBound(),
        /*inputs=*/ValueRange{pos},
        std::nullopt,
        nullptr);

//    comet_vdump(new_forall_loop);
//    comet_debug() << "\n";

    /// Move the old_forall_loop's body to the new_forall_loop
//    llvm::SmallVector<Type> argTypes = {indexType};
//    argTypes.push_back(indexTree::SymbolicDomainType::get(rewriter.getContext(), 64));
//    llvm::SmallVector<Location> locs;
//    for (auto arg : new_forall_loop.getOperands()) {
//      locs.push_back(arg.getLoc());
//      comet_vdump(arg);
//    }
//    Block *block = rewriter.createBlock(/*body=*/&new_forall_loop.getRegion(),
//                                                 {},
//                                                 TypeRange(argTypes),
//                                                 locs);
//    for ([[maybe_unused]] auto arg : block->getArguments()) {
//      comet_vdump(arg);
//    }
//    rewriter.inlineBlockBefore(&old_forall_loop.getRegion().front(),
//                               new_forall_loop.getTerminator(),
////                              new_forall_loop->getOperands());
//                               block->getArguments());
    llvm::SmallVector<Value> argValues;
    for (auto var : new_forall_loop.getInductionVars()) {
      argValues.push_back(var);
    }
    for (auto arg : new_forall_loop.getRegionIterArgs()) {
      argValues.push_back(arg);
    }
//    argValues.insert(argValues.end(),
//                     new_forall_loop.getInductionVars().begin(),
//                     new_forall_loop.getInductionVars().end());
//    argValues.insert(argValues.end(),
//                     new_forall_loop.getRegionIterArgs().begin(),
//                     new_forall_loop.getRegionIterArgs().end());
//    rewriter.inlineBlockBefore(old_forall_loop.getBody(),
//                               new_forall_loop.getTerminator(),
//                               argValues);
    rewriter.inlineBlockBefore(old_forall_loop.getBody(),
                               new_forall_loop.getBody(),
                               new_forall_loop.getBody()->begin(),
                               argValues);
    rewriter.eraseOp(new_forall_loop.getBody()->getTerminator());  /// For some reason, there is an empty scf.forall.in_parallel op that can be deleted, otherwise we cannot get the real terminator (scf.forall.in_parallel op).
    comet_vdump(old_forall_loop);
    comet_vdump(new_forall_loop);

//    rewriter.eraseOp(old_forall_loop);
    rewriter.replaceOp(old_forall_loop, new_forall_loop);
    return new_forall_loop;
  }

  mlir::UnrealizedConversionCastOp CreateInnerSymbolicDomain(scf::ForallOp forall_loop,
                                 mlir::IRRewriter &rewriter,
                                 mlir::Location &loc,
                                 SymbolicDomainInfo symbolic_domain_info)
  {
    /// Generate the new symbolic_domain.
    /// %pos_new_2 = scf.for_all ... %i, %arg1=%pos:
    ///    %symbolic_domain_inner = mlir.unrealized_cast(/*pos_size=*/ %i,
    ///                                                  /*constant for now*/ %pos_alloc_size,
    ///                                                  /*crd_size=*/ %zero,
    ///                                                  /*constant*/ %dim_size,
    ///                                                  /*pos=*/ %arg_1,
    ///                                                  /*private*/ %mark)
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forall_loop.getBody());
    auto result_type = indexTree::SymbolicDomainType::get(rewriter.getContext(), 64);
    llvm::SmallVector<Value> inputs;
    for (auto var : forall_loop.getInductionVars()) {
      inputs.push_back(var);
    }  /// SymbolicDomain.pos_size
    inputs.push_back(symbolic_domain_info.pos_alloc_size);  /// SymbolicDomain.pos_alloc_size
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);  /// SymbolicDomain.crd_size
    inputs.push_back(zero);
    inputs.push_back(symbolic_domain_info.dim_size);  /// SymbolicDomain.dim_size
    for (auto arg : forall_loop.getRegionIterArgs()) {
      inputs.push_back(arg);
    }  /// SymbolicDomain.pos
    inputs.push_back(symbolic_domain_info.mark_array);
    mlir::UnrealizedConversionCastOp symbolic_domain_inner =
        rewriter.create<mlir::UnrealizedConversionCastOp>(loc,
                                                          result_type,
                                                          inputs);

    return symbolic_domain_inner;
  }

  scf::ForOp GetInnerForOp(scf::ForallOp forall_loop)
  {
    scf::ForOp inner_for_loop = nullptr;
    for (auto pos_i : forall_loop.getRegionIterArgs()) {
      for (auto user : pos_i.getUsers()) {
        if ((inner_for_loop = llvm::dyn_cast<scf::ForOp>(user))) {
          break;
        }
      }
      if (inner_for_loop) {
        break;
      }
    }
    assert(inner_for_loop && "Expected at least one scf.for inside the scf.forall.");

    return inner_for_loop;
  }

  scf::ForOp GetInnerForOp(scf::ForOp for_loop)
  {
    scf::ForOp inner_for_loop = nullptr;
    for (auto pos_i : for_loop.getRegionIterArgs()) {
      for (auto user : pos_i.getUsers()) {
        if ((inner_for_loop = llvm::dyn_cast<scf::ForOp>(user))) {
          break;
        }
      }
      if (inner_for_loop) {
        break;
      }
    }
    assert(inner_for_loop && "Expected at least one scf.for inside the scf.forall.");

    return inner_for_loop;
  }

  scf::ForOp GetConsumerForOp(scf::ForOp for_loop)
  {
    scf::ForOp consumer_for_loop = nullptr;
    for (auto output : for_loop.getResults()) {
      for (auto user : output.getUsers()) {
        if ((consumer_for_loop = llvm::dyn_cast<scf::ForOp>(user))) {
          break;
        }
      }
      if (consumer_for_loop) {
        break;
      }
    }
    assert(consumer_for_loop && "Expected at least one scf.for inside the scf.forall.");

    return consumer_for_loop;
  }

  scf::ForOp ReplaceInnerForOp(scf::ForOp old_inner_for_loop,
//                               Value inner_symbolic_domain,
                               llvm::SmallVector<Value> inputs,
                               mlir::IRRewriter &rewriter,
                               mlir::Location &loc)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(old_inner_for_loop);

    /// Create a new inner for-loop
    auto lower_bound = old_inner_for_loop.getLowerBound();
    auto upper_bound = old_inner_for_loop.getUpperBound();
    auto step = old_inner_for_loop.getStep();
//    llvm::SmallVector<Value> inputs = {inner_symbolic_domain};
    scf::ForOp new_inner_for_loop = rewriter.create<scf::ForOp>(loc,
                                                                lower_bound,
                                                                upper_bound,
                                                                step,
                                                                inputs);
    /// Move the body from the old for-loop to the new one.
    llvm::SmallVector<Value> argValues = {new_inner_for_loop.getInductionVar()};
    for (auto arg : new_inner_for_loop.getRegionIterArgs()) {
      argValues.push_back(arg);
    }
    comet_debug() << old_inner_for_loop.getBody()->getNumArguments() << "\n";
    rewriter.inlineBlockBefore(old_inner_for_loop.getBody(),
                               new_inner_for_loop.getBody(),
                               new_inner_for_loop.getBody()->begin(),
                               argValues);

    comet_vdump(old_inner_for_loop);
    comet_vdump(new_inner_for_loop);

    rewriter.replaceOp(old_inner_for_loop, new_inner_for_loop);

    return new_inner_for_loop;
  }

  /*
    %68 = "it.SymbolicDomainEndRowOp"(%67) <{needs_mark = true}> : (!it.symbolic_domain<64>) -> !it.symbolic_domain<64>
    %69:6 = builtin.unrealized_conversion_cast %68 : !it.symbolic_domain<64> to index, index, index, index, tensor<?xi64>, tensor<?xi64>
    %extracted_slice_52 = tensor.extract_slice %69#4[%69#4] [1] [1] : tensor<?xi64> to tensor<1xi64>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %extracted_slice_52 into %arg2[%69#4] [1] [1] : tensor<1xi64> into tensor<?xi64>
    }
   */
  void InsertExtractSliceForPos(scf::ForallOp forall_loop,
                                mlir::IRRewriter &rewriter,
                                mlir::Location &loc)
  {
    /// Get the SymbolicDomainEndRowOp
    indexTree::SymbolicDomainEndRowOp symbolic_domain_end_row_op = nullptr;
    forall_loop.walk([&](indexTree::SymbolicDomainEndRowOp op) {
      symbolic_domain_end_row_op = op;
    });
    assert(symbolic_domain_end_row_op && "Expected at least one indexTree::SymbolicDomainEndRowOp");
    comet_vdump(symbolic_domain_end_row_op);
    Value symbolic_domain_inner = symbolic_domain_end_row_op.getResult();

    /// Unpack the symbolic_domain
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(symbolic_domain_end_row_op);
    IndexType indexType = rewriter.getIndexType();
    RankedTensorType rankedTensorType = mlir::RankedTensorType::get(/*shape=*/{ShapedType::kDynamic}, /*elementType=*/rewriter.getI64Type());
    llvm::SmallVector<Type> result_types = {
        indexType,         /// pos_size
        indexType,         /// pos_alloc_size
        indexType,         /// crd_size
        indexType,         /// dim_size
        rankedTensorType,  /// pos
        rankedTensorType   /// mark_array
    };
    mlir::UnrealizedConversionCastOp unpack_symbolic_domain =
        rewriter.create<mlir::UnrealizedConversionCastOp>(loc,
                                                          result_types,
                                                          /*input=*/symbolic_domain_inner);
    SymbolicDomainInfo inner_symbolic_domain_info;
    inner_symbolic_domain_info.pos_size = unpack_symbolic_domain.getResult(0);
    inner_symbolic_domain_info.pos_alloc_size = unpack_symbolic_domain.getResult(1);
    inner_symbolic_domain_info.crd_size = unpack_symbolic_domain.getResult(2);
    inner_symbolic_domain_info.dim_size = unpack_symbolic_domain.getResult(3);
    inner_symbolic_domain_info.pos = unpack_symbolic_domain.getResult(4);
    inner_symbolic_domain_info.mark_array = unpack_symbolic_domain.getResult(5);

    /// Insert the extract_slice
    Value input = inner_symbolic_domain_info.pos;
    llvm::SmallVector<Value> dynamic_offsets = {inner_symbolic_domain_info.pos_size};
    int64_t os_dim = 0;
    int64_t nDims = rankedTensorType.getRank();
    llvm::SmallVector<Value> sizes;
    for (int64_t i = 0; i < nDims; ++i) {
      if (i != os_dim && rankedTensorType.getDimSize(i) == ShapedType::kDynamic) {
        Value idx = rewriter.create<index::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(i));
        sizes.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.getIndexType(), input, idx));
      }
    }
    SmallVector<int64_t> static_offsets(nDims, 0);
    static_offsets[os_dim] = ShapedType::kDynamic;
    llvm::SmallVector<int64_t> static_sizes(rankedTensorType.getShape());
    static_sizes[os_dim] = 1;
    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        loc,
        /*result_type=*/RankedTensorType::get(static_sizes, rankedTensorType.getElementType()),
        /*tensor_source=*/input,
        /*dynamic_offsets=*/dynamic_offsets,
        /*dynamic_sizes=*/sizes,
        /*dynamic_strides=*/ValueRange(),
        /*static_offsets=*/rewriter.getDenseI64ArrayAttr(static_offsets),
        /*static_sizes=*/rewriter.getDenseI64ArrayAttr(static_sizes),
        /*static_strides=*/rewriter.getDenseI64ArrayAttr(llvm::SmallVector<int64_t>(nDims, 1))
        );

    /// Replace the element in the scf.forall.in_parallel
    auto par_op = forall_loop.getTerminator();
    tensor::ParallelInsertSliceOp old_insert_slice = nullptr;
    par_op.walk([&](tensor::ParallelInsertSliceOp op) {
      old_insert_slice = op;
    });
    assert(old_insert_slice && "Expected at least one tensor::ParallelInsertSliceOp op.");
    comet_vdump(old_insert_slice);
    Value arg_tensor = old_insert_slice.getDest();
    rewriter.setInsertionPoint(old_insert_slice);
    auto new_insert_slice = rewriter.create<tensor::ParallelInsertSliceOp>(
        loc,
        /*src_tensor=*/slice,
        /*dst_tensor=*/arg_tensor,
        /*dynamic_offsets=*/dynamic_offsets,
        /*dynamic_sizes=*/sizes,
        /*dynamic_strides=*/ValueRange(),
        /*static_offsets=*/rewriter.getDenseI64ArrayAttr(static_offsets),
        /*static_sizes=*/rewriter.getDenseI64ArrayAttr(static_sizes),
        /*static_strides=*/rewriter.getDenseI64ArrayAttr(llvm::SmallVector<int64_t>(nDims, 1))
        );
    rewriter.replaceOp(old_insert_slice, new_insert_slice);

    comet_vdump(slice.getResult());
    comet_vdump(new_insert_slice);
    comet_vdump(forall_loop);
    comet_debug() << "\n";
  }

  /*
    %accumulator = arith.constant 0 : i64
    %68:2 = scf.for %i = %c1 to %N_plus_one step %c1 iter_args(%arg6 = %accumulator, %arg7 = %pos) -> (i64, tensor<?xi64>) {
        %curr = tensor.extract %arg7[%i] : tensor<?xi64>
        %added = arith.addi %arg6, %curr : i64
        %inserted = tensor.insert %added into %arg7[%i] : tensor<?xi64>
        scf.yield %added, %inserted : i64, tensor<?xi64>
    }
   */
  ValueRange InsertAccumulatingLoopForPos(scf::ForallOp forall_loop,
                              mlir::IRRewriter &rewriter,
                              mlir::Location &loc)
  {
    /// Find the `it.yield` op
    auto itree_op = forall_loop->getParentOfType<indexTree::IndexTreeOp>();
    auto yield_op = itree_op.getRegion().front().getTerminator();
    comet_pdump(yield_op);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(yield_op);

    Value acc_0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    Value pos_input = forall_loop.getResults().front();
    comet_vdump(pos_input);
    llvm::SmallVector<Value> inputs = {acc_0, pos_input};

    Value cst_index_one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value forall_loop_bound = forall_loop.getUpperBound(rewriter).front();
    Value lower_bound = cst_index_one;
    auto upper_bound = rewriter.create<arith::AddIOp>(loc, forall_loop_bound, cst_index_one);
    Value step = cst_index_one;
    auto acc_for_loop = rewriter.create<scf::ForOp>(loc,
                                lower_bound,
                                upper_bound,
                                step,
                                inputs);
    rewriter.setInsertionPointToStart(acc_for_loop.getBody());

    /* %curr = tensor.extract %arg7[%i] : tensor<?xi64> */
    Value i_var = acc_for_loop.getInductionVar();
    Value acc_size = acc_for_loop.getRegionIterArg(0);
    Value acc_pos = acc_for_loop.getRegionIterArg(1);
    Value curr = rewriter.create<tensor::ExtractOp>(
        loc,
        /*tensor=*/acc_pos,
        /*index=*/i_var);

    /* %added = arith.addi %arg6, %curr : i64 */
    Value added = rewriter.create<arith::AddIOp>(loc,
                                                 rewriter.getI64Type(),
                                                 curr,
                                                 acc_size);

    /* %inserted = tensor.insert %added into %arg7[%i] : tensor<?xi64> */
    Value inserted = rewriter.create<tensor::InsertOp>(
        loc,
        /*return_type=*/pos_input.getType(),
        /*element=*/added,
        /*tensor=*/acc_pos,
        /*index=*/i_var);
    /* scf.yield %added, %inserted : i64, tensor<?xi64> */
    rewriter.create<scf::YieldOp>(loc,
                                  ValueRange{added, inserted});
    comet_vdump(acc_for_loop);

    return acc_for_loop.getResults();
  }

  /*
    %final_domain = unrealized_cast(%num_iters_of_loop,
                                    %alloc_size,
                                    %68#0, // %accumulator
                                    %dim_size,
                                    %68#1, // %pos_final
                                    %mark_array);
    it.yield %final_domain;
   */
  void FinalizeSymbolicDomain(scf::ForallOp forall_loop,
                              const SymbolicDomainInfo &old_symbolic_domain_info,
                              ValueRange &acc_for_loop_results,
                              mlir::IRRewriter &rewriter,
                              mlir::Location &loc)
  {
    /// Find the `it.yield` op
    auto itree_op = forall_loop->getParentOfType<indexTree::IndexTreeOp>();
    auto yield_op = itree_op.getRegion().front().getTerminator();
    comet_pdump(yield_op);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(yield_op);
    Value acc_size = acc_for_loop_results[0];
    Value acc_pos = acc_for_loop_results[1];
    Value num_rows = forall_loop.getUpperBound(rewriter).front();

    /*
      %final_domain = unrealized_cast(%num_iters_of_loop,
                                      %alloc_size,
                                      %68#0, // %accumulator
                                      %dim_size,
                                      %68#1, // %pos_final
                                      %mark_array);
     */
    auto result_type = indexTree::SymbolicDomainType::get(rewriter.getContext(), 64);
    llvm::SmallVector<Value> inputs;
    inputs.push_back(num_rows);  /// pos_size
    inputs.push_back(old_symbolic_domain_info.pos_alloc_size);              /// pos_alloc_size
    inputs.push_back(rewriter.createOrFold<arith::IndexCastOp>(loc,
                                                               rewriter.getIndexType(),
                                                               acc_size));  /// crd_size
    inputs.push_back(old_symbolic_domain_info.dim_size);                    /// dim_size
    inputs.push_back(acc_pos);                                              /// pos
    inputs.push_back(old_symbolic_domain_info.mark_array);                  /// mark_array
    mlir::UnrealizedConversionCastOp final_symbolic_domain =
        rewriter.create<mlir::UnrealizedConversionCastOp>(loc,
                                                          result_type,
                                                          inputs);

    /* it.yield %final_domain; */
    yield_op->setOperand(0, final_symbolic_domain.getResult(0));
    comet_vdump(forall_loop->getParentOfType<indexTree::IndexTreeOp>());
  }


  void LegalizeSymbolicForallOp(func::FuncOp func)
  {
    scf::ForallOp old_forall_loop = nullptr;
    func.walk([&](scf::ForallOp op) {
      int count = 0;
      for (auto &arg : op.getOutputsMutable()) {
        ++count;
        if (count > 1 || !llvm::isa<indexTree::SymbolicDomainType>(arg.get().getType())) {
          return;
        } else {
          old_forall_loop = op;
        }
      }
    });

    if (!old_forall_loop) {
      return;
    }
    comet_vdump(old_forall_loop);
    comet_debug() << "\n";
    SymbolicDomainInfo symbolic_domain_info;
    mlir::OpBuilder builder(old_forall_loop);
    mlir::IRRewriter rewriter(builder);
    mlir::Location loc = old_forall_loop->getLoc();

    /// Create a new scf.forall taking the vector `pos` as its operand
    scf::ForallOp new_forall_loop = CreateNewForallOp(old_forall_loop,
                                                      rewriter,
                                                      loc,
                                                      symbolic_domain_info/*out*/);

    /// Generate the new symbolic_domain.
    mlir::UnrealizedConversionCastOp symbolic_domain_inner =
        CreateInnerSymbolicDomain(new_forall_loop,
                                  rewriter,
                                  loc,
                                  symbolic_domain_info);
    comet_vdump(symbolic_domain_inner);
    comet_vdump(new_forall_loop);

    /// Find the inner for-loop
    scf::ForOp inner_for_loop = GetInnerForOp(new_forall_loop);
    comet_vdump(inner_for_loop);

    /// For the inner for-loop, replace its operand with the new symbolic_domain.
    ReplaceInnerForOp(inner_for_loop,
                      /*inputs=*/llvm::SmallVector<Value>{symbolic_domain_inner.getResult(0)},
                      rewriter,
                      loc);

    /// Use unrealized_cast to unpack the finished symbolic_domain, so that we can get the vector `pos`,
    /// create `extract_slice` from this new `pos`, then do `parallel_insert_slice` from the slice to
    /// the input scf.forall's argumet `pos`
    InsertExtractSliceForPos(new_forall_loop,
                             rewriter,
                             loc);

    /// Insert the accumulating for-loop for vector `pos` after the new-forall-loop, and also
    /// get the final `crd_size`
    mlir::ValueRange acc_for_loop_results = InsertAccumulatingLoopForPos(new_forall_loop,
                          rewriter,
                          loc);

    /// Generate unrealized_cast from the `crd_size` and vector `final-pos` to a `final-symbolic-domain`.
    /// The `itree` will yield this `final-symbolic-domain`.
    FinalizeSymbolicDomain(new_forall_loop,
                           symbolic_domain_info,
                           acc_for_loop_results,
                           rewriter,
                           loc);
    comet_vdump(new_forall_loop->getParentOfType<ModuleOp>());
    comet_debug() << "\n";
  }


  Value GetNumericWorkspace(scf::ForallOp old_forall_loop)
  {
    /*
     * For example,
     * %output:2 = scf.forall (%arg0) in (%bound) shared_outs(%arg1 = %sparse_tensor, %arg2 = %workspace)
     * forall_loop.getRegionIterArgs(): %arg1, %arg2
     * forall_loop.getOperands(): %bound, %sparse_tensor, %workspace
     * forall_loop.getOutputs(): %output#0, %output#1.
        * forall_loop.getTiedBlockArgument(%output#0) == %arg1.
        * forall_loop.getTiedBlockArgument(%output#1) == %arg2.
     * forall_loop.getOutputsMutable(): arg.get(): %output#0, %output#1
     */
//    for (auto arg : old_forall_loop.getRegionIterArgs()) {
//      comet_vdump(arg);
//    }
//    for (auto arg : old_forall_loop.getOperands()) {
//      comet_vdump(arg);
//    }
//    for (auto &arg : old_forall_loop.getOutputsMutable()) {
//      comet_vdump(arg.get());
//    }
    uint32_t count = 0;
    Value workspace = nullptr;
    for (auto arg : old_forall_loop.getOutputs()) {
      comet_vdump(arg);
      if (llvm::isa<tensorAlgebra::WorkspaceType>(arg.getType())) {
        workspace = arg;
        ++count;
      }
    }
    assert(count == 1 && "Error: expected only one workspace.");
    return workspace;
  }


  scf::ForallOp CreateNumericNewForallOp(scf::ForallOp old_forall_loop,
                                         Value workspace,
                                         mlir::IRRewriter &rewriter,
                                         mlir::Location &loc,
                                         NumericSparseTensorInfo &sparseTensorInfo /*out*/)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(old_forall_loop);

    /// Create %pos, %crds, %vals = builtin.unrealized_conversion_cast(%old_sparse_tensor);
    Value old_sparse_tensor = nullptr;
    uint32_t count = 0;
    for (auto arg : old_forall_loop.getOutputs()) {
      if (llvm::isa<tensorAlgebra::SparseTensorType>(arg.getType())) {
        old_sparse_tensor = arg;
        ++count;
      }
    }
    assert(count == 1 && "Error: expect one sparse_tensor");
    RankedTensorType posTensorType = mlir::RankedTensorType::get(/*shape=*/{ShapedType::kDynamic}, /*elementType=*/rewriter.getI64Type());
    RankedTensorType crdsTensorType = posTensorType;
    RankedTensorType valsTensorType = mlir::RankedTensorType::get(/*shape=*/{ShapedType::kDynamic}, /*elementType=*/rewriter.getF64Type());
    llvm::SmallVector<Type> result_types = {
        posTensorType,
        crdsTensorType,
        valsTensorType
    };
    mlir::UnrealizedConversionCastOp unrealized_op =
        rewriter.create<mlir::UnrealizedConversionCastOp>(loc,
                                                          result_types,
                                                          /*input=*/old_sparse_tensor);
    comet_pdump(unrealized_op->getParentOp());
    Value pos = unrealized_op->getResult(0);
    Value crds = unrealized_op->getResult(1);
    Value vals = unrealized_op->getResult(2);
    sparseTensorInfo = {pos, crds, vals, old_sparse_tensor};
    SmallVector<Value> inputs = {crds, vals};
    scf::ForallOp new_forall_loop = rewriter.create<scf::ForallOp>(
        loc,
        /*upperBound=*/old_forall_loop.getMixedUpperBound(),
        /*inputs=*/inputs,
        std::nullopt,
        nullptr);

    /// Move the old_forall_loop's body to the new_forall_loop
    llvm::SmallVector<Value> argValues;
    for (auto var : new_forall_loop.getInductionVars()) {
      argValues.push_back(var);
    }
    for (auto arg : new_forall_loop.getRegionIterArgs()) {
      argValues.push_back(arg);
    }
    rewriter.inlineBlockBefore(old_forall_loop.getBody(),
                               new_forall_loop.getBody(),
                               new_forall_loop.getBody()->begin(),
                               argValues);
    rewriter.eraseOp(new_forall_loop.getBody()->getTerminator());  /// For some reason, there is an empty scf.forall.in_parallel op that can be deleted, otherwise we cannot get the real terminator (scf.forall.in_parallel op).
    comet_vdump(old_forall_loop);
    comet_vdump(new_forall_loop);
    rewriter.replaceOp(old_forall_loop, new_forall_loop);
    return new_forall_loop;
//    /// Replace the inner workspace with the shared workspace from outside.
//    Value inner_worksapce = nullptr;
//    count = 0;
//    for (auto &arg : old_forall_loop.getOutputsMutable()) {
//      if (llvm::isa<tensorAlgebra::WorkspaceType>(arg.get().getType())) {
//        inner_worksapce = arg.get();
//        ++count;
//      }
//    }
//    assert(count == 1 && "Error: expect only one workspace.");
//    rewriter.replaceAllUsesWith(inner_worksapce, workspace);
  }

  void UpdateExtractSliceOps(scf::ForallOp new_forall_loop,
                             const NumericSparseTensorInfo &sparseTensorInfo,
                             mlir::IRRewriter &rewriter,
                             mlir::Location &loc,
                             NumericSparseTensorInfo &innerSparseTensorInfo/*out*/)
  {
    Value crds_arg = new_forall_loop.getRegionIterArgs()[0];
    Value vals_arg = new_forall_loop.getRegionIterArgs()[1];

    auto FindTensorExtractSliceOp = [&](Value op) {
      Operation *result;
      for (auto user : op.getUsers()) {
        if (llvm::isa<tensor::ExtractSliceOp>(user)) {
          result = user;
          break;
        }
      }
      return result;
    };
    Operation *crds_old_extract_slice = FindTensorExtractSliceOp(crds_arg);
    Operation *vals_old_extract_slice = FindTensorExtractSliceOp(vals_arg);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(crds_old_extract_slice);

    /// Create new tensor.extract_slice, using pos to get length
    /*
     * %offset = %pos[%i]
     * %size = %pos[%i + 1] - %pos[%i]
     * %stride = 1
     * %crds_extract_slice = tensor.extract_slice %crds[%offset] [%size] [%stride]
     * %vals_extract_slice = tensor.extract_slice %vals[%offset] [%size] [%stride]
     */
    Value index = new_forall_loop.getInductionVar(0);
    Value pos = sparseTensorInfo.pos;
    Type pos_element_type = mlir::getElementTypeOrSelf(pos);

    Value offset = rewriter.create<tensor::ExtractOp>(loc,
                                                      /*return_type=*/pos_element_type,
                                                      /*tensor=*/pos,
                                                      /*index=*/index);
    Value offset_index = rewriter.createOrFold<arith::IndexCastOp>(loc, rewriter.getIndexType(), offset);
//    auto oneAttr = rewriter.getIntegerAttr(pos_element_type, 1);
    Value const_one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(1));
    Value index_plus_one = rewriter.create<arith::AddIOp>(loc, index, const_one);
    Value offset_next = rewriter.create<tensor::ExtractOp>(loc,
                                                           pos_element_type,
                                                           pos,
                                                           index_plus_one);
    Value offset_next_index = rewriter.createOrFold<arith::IndexCastOp>(loc, rewriter.getIndexType(), offset_next);
    Value size = rewriter.create<arith::SubIOp>(loc, offset_next_index, offset_index);
    Type crds_element_type = mlir::getElementTypeOrSelf(crds_arg);
    Type vals_element_type = mlir::getElementTypeOrSelf(vals_arg);
    RankedTensorType crds_tensor_type = mlir::RankedTensorType::get(/*shape=*/{ShapedType::kDynamic}, /*elementType=*/crds_element_type);
    RankedTensorType vals_tensor_type = mlir::RankedTensorType::get(/*shape=*/{ShapedType::kDynamic}, /*elementType=*/vals_element_type);
    llvm::SmallVector<Value> dynamic_offsets = {offset_index};
    llvm::SmallVector<Value> dynamic_sizes = {size};
    llvm::SmallVector<Value> dynamic_strides;
    llvm::SmallVector<int64_t> static_offsets = {ShapedType::kDynamic};
    llvm::SmallVector<int64_t> static_sizes = {ShapedType::kDynamic};
    llvm::SmallVector<int64_t> static_strides = {1};
    auto crds_new_extract_slice = rewriter.create<tensor::ExtractSliceOp>(
        loc,
        /*result_type=*/crds_tensor_type,
        /*tensor_source=*/crds_arg,
        /*dynamic_offsets=*/dynamic_offsets,
        /*dynamic_sizes=*/dynamic_sizes,
        /*dynamic_strides=*/dynamic_strides,
        /*static_offsets=*/static_offsets,
        /*static_sizes=*/static_sizes,
        /*static_strides=*/static_strides);
    auto vals_new_extract_slice = rewriter.create<tensor::ExtractSliceOp>(
        loc,
        /*result_type=*/vals_tensor_type,
        /*tensor_source=*/vals_arg,
        /*dynamic_offsets=*/dynamic_offsets,
        /*dynamic_sizes=*/dynamic_sizes,
        /*dynamic_strides=*/dynamic_strides,
        /*static_offsets=*/static_offsets,
        /*static_sizes=*/static_sizes,
        /*static_strides=*/static_strides);

    /// Pack the sparse_tensor using extracted slices
    Type sparse_tensor_type = sparseTensorInfo.sparseTensor.getType();
    comet_vdump(sparse_tensor_type);
    llvm::SmallVector<Value> inputs = {pos, crds_new_extract_slice, vals_new_extract_slice};
    mlir::UnrealizedConversionCastOp unrealized_op =
        rewriter.create<mlir::UnrealizedConversionCastOp>(loc,
                                                          sparse_tensor_type,
                                                          inputs);

    rewriter.replaceOp(crds_old_extract_slice, crds_new_extract_slice);
    rewriter.replaceOp(vals_old_extract_slice, vals_new_extract_slice);
    /// Output
    innerSparseTensorInfo.pos = pos;
    innerSparseTensorInfo.crds = crds_new_extract_slice;
    innerSparseTensorInfo.vals = vals_new_extract_slice;
    innerSparseTensorInfo.sparseTensor = unrealized_op.getResult(0);

    comet_vdump(crds_arg);
    comet_vdump(vals_arg);
//    comet_pdump(crds_old_extract_slice);
//    comet_pdump(vals_old_extract_slice);
    comet_vdump(const_one);
    comet_vdump(index_plus_one);
    comet_vdump(crds_new_extract_slice);
    comet_vdump(vals_new_extract_slice);
    comet_vdump(innerSparseTensorInfo.sparseTensor);
    comet_vdump(new_forall_loop);
  }

  Value UpdateWorkspaceClearOp(scf::ForallOp new_forall_loop,
                             Value workspace,
                             mlir::IRRewriter &rewriter,
                             mlir::Location &loc)
  {
    Value old_workspace_clear_op;
    uint32_t count = 0;
    new_forall_loop.walk([&](tensorAlgebra::WorkspaceClearOp op) {
      old_workspace_clear_op = op;
      ++count;
    });
    assert(count == 1 && "Error: expect only one ta.WorkspaceClear op.");

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(old_workspace_clear_op.getDefiningOp());
    Value new_workspace_clear_op = rewriter.create<tensorAlgebra::WorkspaceClearOp>(loc, workspace.getType(), workspace);
    rewriter.replaceOp(old_workspace_clear_op.getDefiningOp(), new_workspace_clear_op.getDefiningOp());
    comet_vdump(new_forall_loop);

    return new_workspace_clear_op;
  }

  scf::ForOp GetNumericInnerForOp(Value workspaceClearOp)
  {
    scf::ForOp inner_for_loop = nullptr;
    for (auto user : workspaceClearOp.getUsers()) {
      if ((inner_for_loop = llvm::dyn_cast<scf::ForOp>(user))) {
        break;
      }
    }
    assert(inner_for_loop && "Expected at least one scf.for inside the scf.forall.");

    return inner_for_loop;
  }

  scf::ForOp UpdateNumericInnerForOp(scf::ForOp inner_for_loop,
                               Value inner_sparse_tensor,
                               Value clear_workspace,
                               mlir::IRRewriter &rewriter,
                               mlir::Location &loc)
  {
    scf::ForOp old_inner_most_for_loop = GetInnerForOp(inner_for_loop);
    scf::ForOp new_inner_for_loop = ReplaceInnerForOp(inner_for_loop,
                      /*inputs=*/llvm::SmallVector<Value>{inner_sparse_tensor, clear_workspace},
                      rewriter,
                      loc);
    ReplaceInnerForOp(old_inner_most_for_loop,
                      /*inputs=*/llvm::SmallVector<Value>{new_inner_for_loop.getRegionIterArgs()},
                      rewriter,
                      loc);
//    comet_vdump(new_forall_loop->getParentOfType<ModuleOp>());
//    comet_debug() << "\n";
    comet_vdump(new_inner_for_loop);
    return new_inner_for_loop;
  }

  void UpdateWorkspaceForLoop(scf::ForOp new_inner_for_loop,
                            mlir::IRRewriter &rewriter,
                            mlir::Location &loc)
  {
    scf::ForOp workspace_for_loop = GetConsumerForOp(new_inner_for_loop);

    comet_vdump(workspace_for_loop);
  }

  void LegalizeNumericForallOp(func::FuncOp func)
  {
    /// Found the numeric forall
    scf::ForallOp old_forall_loop = nullptr;
    func.walk([&](scf::ForallOp op) {
      int count = 0;
      for (auto &arg : op.getOutputsMutable()) {
        if (llvm::isa<tensorAlgebra::SparseTensorType>(arg.get().getType()) ||
            llvm::isa<tensorAlgebra::WorkspaceType>(arg.get().getType())) {
          ++count;
        }
      }
      if (2 == count) {
        old_forall_loop = op;
      }
    });

    if (!old_forall_loop) {
      return;
    }
    comet_vdump(old_forall_loop);
    comet_debug() << "\n";
    mlir::OpBuilder builder(old_forall_loop);
    mlir::IRRewriter rewriter(builder);
    mlir::Location loc = old_forall_loop->getLoc();

    Value workspace = GetNumericWorkspace(old_forall_loop);
    NumericSparseTensorInfo sparseTensorInfo;
    scf::ForallOp new_forall_loop = CreateNumericNewForallOp(old_forall_loop,
                                                             workspace,
                                                             rewriter,
                                                             loc,
                                                             sparseTensorInfo/*out*/);
    NumericSparseTensorInfo innerSparseTensorInfo;
    UpdateExtractSliceOps(new_forall_loop,
                          sparseTensorInfo,
                          rewriter,
                          loc,
                          innerSparseTensorInfo/*out*/);

    /// Update the ta.WorkspaceClear op to use the correct workspace
    Value clear_workspace = UpdateWorkspaceClearOp(new_forall_loop,
                                                   workspace,
                                                   rewriter,
                                                   loc);

    /// Find the inner for-loop
    scf::ForOp old_inner_for_loop = GetNumericInnerForOp(/*workspaceClearOp=*/clear_workspace);
    comet_vdump(old_inner_for_loop);

    /// Update the inner for loop to take the new packed sparse tensor.
    scf::ForOp new_inner_for_loop = UpdateNumericInnerForOp(
        old_inner_for_loop,
        /*inner_sparse_tensor=*/innerSparseTensorInfo.sparseTensor,
        clear_workspace,
        rewriter,
        loc);

    /// TODO: update the workspace for-loop
    UpdateWorkspaceForLoop(new_inner_for_loop,
                           rewriter,
                           loc);
    comet_vdump(new_forall_loop);
    comet_debug() << "\n";
  }


  struct LowerIndexTreeToSCFPass
      : public PassWrapper<LowerIndexTreeToSCFPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerIndexTreeToSCFPass)

    SmallDenseMap<Value, LoopInfo*> nodeMap;
    SmallDenseMap<Value, uint32_t> leafMap;
    llvm::ScopedPrinter logger{llvm::dbgs()};

    Value mapInputIntoLoop(Value input, LoopInfo* loop_info)
    {
      auto input_idx_iter = leafMap.find(input);
      if(input_idx_iter != leafMap.end()){
        uint32_t input_idx = input_idx_iter->getSecond();
        return loop_info->getInput(input_idx);
      }
      return input;
    }

    void updateOutput(Value input, Value old_output, Value new_output, LoopInfo* loop_info, IRRewriter& rewriter)
    {
      uint32_t input_idx = leafMap.find(input)->getSecond();
      leafMap.insert(std::pair<Value, uint32_t>(old_output, input_idx));
      loop_info->updateOutput(rewriter, input_idx, new_output);
    }

    Value convertOperand(LoopInfo* loop_info, IndexTreeLHSOperandOp op, StringRef semiring, IRRewriter &rewriter)
    {
      Location loc = op->getLoc();
      Value tensor = mapInputIntoLoop(op.getTensor(), loop_info);
      auto crds = op.getCrds();
      auto positions = op.getPos();

      TensorType tensor_type;
      if (llvm::isa<mlir::FloatType>(tensor.getType()) || llvm::isa<mlir::IntegerType>(tensor.getType())) {
        /// LHS operand is a constant value.
        return tensor;
      } else if((tensor_type = llvm::dyn_cast<mlir::TensorType>(tensor.getType()))){
        return rewriter.create<tensor::ExtractOp>(loc, tensor_type.getElementType(), tensor, positions);
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
        double zero = 0;
        if(semiring == "minxy"){
          zero = INFINITY;
        }
        return rewriter.create<TensorExtractOp>(loc, element_type, tensor, pos, crds, rewriter.getF64FloatAttr(zero));
      }
    }

    Value convertOperand(LoopInfo* loop_info, IndexTreeOperandOp op, StringRef semiring, IRRewriter &rewriter)
    {
      Location loc = op->getLoc();
      Value tensor = mapInputIntoLoop(op.getTensor(), loop_info);
      auto crds = op.getCrds();
      auto positions = op.getPos();

      TensorType tensor_type;
      if (llvm::isa<mlir::FloatType>(tensor.getType()) || llvm::isa<mlir::IntegerType>(tensor.getType())) {
        /// RHS operand is a constant value.
        return tensor;
      } else if((tensor_type = llvm::dyn_cast<mlir::TensorType>(tensor.getType()))){
        return rewriter.create<tensor::ExtractOp>(loc, tensor_type.getElementType(), tensor, crds);
      } else {
        // LHS may not be constant (i.e. if we are inserting into a tensor that we need to resize), 
        // so cannot directly lower like we can the RHS
        Value pos = positions[positions.size() - 1];
        double zero = 0;
        if(semiring == "minxy"){
          zero = INFINITY;
        }

        return rewriter.create<TensorExtractOp>(loc, rewriter.getF64Type(), tensor, pos, crds, rewriter.getF64FloatAttr(zero));
      }
    }

    mlir::LogicalResult convertCompute(IndexTreeComputeOp compute_op, IRRewriter &rewriter)
    {
      auto loc = compute_op->getLoc();
      LoopInfo* parent_info = nodeMap.find(compute_op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      
      auto semiringParts = compute_op.getSemiring().split('_');
      Value elementwise_result;
      for(auto rhs = compute_op.getRhs().begin(); rhs != compute_op.getRhs().end(); rhs++)
      {
        Value rhs_value = convertOperand(parent_info, cast<IndexTreeOperandOp>((*rhs).getDefiningOp()), semiringParts.first, rewriter);
        if(rhs == compute_op.getRhs().begin()){
          elementwise_result = rhs_value;
        } else {
          elementwise_result = getSemiringSecondVal(rewriter, loc, semiringParts.second, 
                                                    elementwise_result, rhs_value);
        }
      }

      IndexTreeLHSOperandOp lhs = llvm::cast<IndexTreeLHSOperandOp>(compute_op.getLhs().getDefiningOp());
      Value reduce_result = convertOperand(parent_info, lhs, semiringParts.first, rewriter);
      reduce_result = getSemiringFirstVal(rewriter, loc, semiringParts.first, 
                                          reduce_result, elementwise_result);

      Value old_tensor = mapInputIntoLoop(lhs.getTensor(), parent_info);
      Value output_tensor;
      if (llvm::isa<mlir::FloatType>(old_tensor.getType())) {
        LLVM_DEBUG({logger.startLine() << __FILE__ << ":" << __LINE__ << " " << old_tensor << "\n";});
        LLVM_DEBUG({logger.startLine() << __FILE__ << ":" << __LINE__ << " " << reduce_result << "\n";});
        output_tensor = reduce_result;
      } else if(llvm::isa<mlir::TensorType>(old_tensor.getType())) {
        output_tensor = rewriter.create<tensor::InsertOp>(loc, old_tensor.getType(), reduce_result, old_tensor, lhs.getPos());
      } else {
        output_tensor = rewriter.create<tensorAlgebra::TensorInsertOp>(loc, old_tensor.getType(), old_tensor, lhs.getPos(), lhs.getCrds(), reduce_result);
      }
      updateOutput(lhs.getTensor(), compute_op.getResult(), output_tensor, parent_info, rewriter);
      return success();
    }

    mlir::LogicalResult convertClearWorkspaceOp(IndexTreeCleanWorkspaceOp op, IRRewriter &rewriter)
    {
      Location loc = op->getLoc();
      LoopInfo* parent_info = nodeMap.find(op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      Value input = mapInputIntoLoop(op.getWorkspace(), parent_info);
      Value workspace = rewriter.create<WorkspaceClearOp>(loc, op->getResultTypes(), input);
      updateOutput(op.getWorkspace(), op.getResult(), workspace, parent_info, rewriter);
      return success();
    }

    mlir::LogicalResult convertSymbolicDomainOp(ComputeSymbolicDomainOp op, IRRewriter &rewriter)
    {
      
      Location loc = op->getLoc();
      LoopInfo* parent_info = nodeMap.find(op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      Value symbolic_domain = mapInputIntoLoop(op.getDomain(), parent_info);
      {
        comet_vdump(symbolic_domain);
      }
      Value new_domain = rewriter.create<SymbolicDomainInsertOp>(loc, 
                                                                 symbolic_domain.getType(),
                                                                 symbolic_domain,
                                                                 parent_info->getCrd(rewriter),
                                                                 op.getIsUniqueAttr());
      updateOutput(op.getDomain(), op.getResult(), new_domain, parent_info, rewriter);
      return success();
    }

    mlir::LogicalResult convertSymbolicDomainEndRowOp(ComputeSymbolicDomainRowOp op, IRRewriter &rewriter)
    {
      Location loc = op->getLoc();
      LoopInfo* parent_info = nodeMap.find(op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      Value symbolic_domain = mapInputIntoLoop(op.getDomain(), parent_info);
      Value new_domain = rewriter.create<SymbolicDomainEndRowOp>(loc, 
                                                                 symbolic_domain.getType(),
                                                                 symbolic_domain,
                                                                 op.getNeedsMarkAttr());
      updateOutput(op.getDomain(), op.getResult(), new_domain, parent_info, rewriter);
      return success();
    }

    mlir::LogicalResult convertFillMaskOp(IndexTreeFillMaskOp fill_mask_op, IRRewriter &rewriter) {
      auto loc = fill_mask_op.getLoc();
      LoopInfo* parent_info = nodeMap.find(fill_mask_op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      Value mask_tensor = mapInputIntoLoop(fill_mask_op.getInit(), parent_info);

      // Create loop to fill mask
      Operation* mask = fill_mask_op.getDomain().getDefiningOp();
      ValueRange fill_loop_inputs(mask_tensor);
      LoopInfo* fill_loop = llvm::TypeSwitch<Operation*, LoopInfo*>(mask) 
      .Case<IndexTreeDenseDomainOp>([&](IndexTreeDenseDomainOp op) {
        return DenseLoopInfo::build(op, rewriter, fill_loop_inputs);
      })
      .Case<IndexTreeSparseDomainOp>([&](IndexTreeSparseDomainOp op) {
        switch((TensorFormatEnum)op.getFormat()){
          case TensorFormatEnum::UNK:
          case TensorFormatEnum::D:
            assert(false && "Invalid format for IndexTreeSparseDomainOp");
            return (LoopInfo*)nullptr;
            break;
          case TensorFormatEnum::CN:
          case TensorFormatEnum::CU:
            return SparseLoopInfo::build(op, rewriter, fill_loop_inputs);
          case TensorFormatEnum::S:
            assert(false && "Singleton loop inside a mask is not supported.");
            return (LoopInfo*)nullptr;
            // return SingletonLoopInfo::build(op, rewriter, inputs, parent_info);
            break;
          }
      })
      .Case<IndexTreeWorkspaceDomainOp>([&](IndexTreeWorkspaceDomainOp op) {
        return WorkspaceLoopInfo::build(op, rewriter, fill_loop_inputs);
      })
      .Case<IndexTreeDomainIntersectionOp>([&](IndexTreeDomainIntersectionOp op) {
        return IntersectionLoopInfo::build(op, rewriter, fill_loop_inputs);
      })
      .Default([](Operation *op) {
        assert(false && "IndexNode not given a valid domain");
        return nullptr;
      });

      // Fill in bit tensor
      auto after_fill_loop = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(fill_loop->loopBody);
      Value crd = fill_loop->getCrd(rewriter);
      Value t = rewriter.create<arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(true));
      Value updated_tensor = rewriter.create<tensor::InsertOp>(loc, fill_loop->getInput(0).getType(), t, fill_loop->getInput(0), crd);
      fill_loop->updateOutput(rewriter, 0, updated_tensor);
      rewriter.restoreInsertionPoint(after_fill_loop);

      updateOutput(
        fill_mask_op.getInit(), 
        fill_mask_op.getResult(), 
        fill_loop->getResults()[0], 
        parent_info, 
        rewriter
      );
      
      delete fill_loop;
      return success();
    }

    mlir::LogicalResult convertZeroMaskOp(IndexTreeZeroMaskOp zero_mask_op, IRRewriter &rewriter) {
      auto loc = zero_mask_op.getLoc();
      LoopInfo* parent_info = nodeMap.find(zero_mask_op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      Value mask_tensor = mapInputIntoLoop(zero_mask_op.getInit(), parent_info);

      // Create loop to fill mask
      Operation* mask = zero_mask_op.getDomain().getDefiningOp();
      ValueRange zero_loop_inputs(mask_tensor);
      LoopInfo* zero_loop = llvm::TypeSwitch<Operation*, LoopInfo*>(mask) 
      .Case<IndexTreeDenseDomainOp>([&](IndexTreeDenseDomainOp op) {
        return DenseLoopInfo::build(op, rewriter, zero_loop_inputs);
      })
      .Case<IndexTreeSparseDomainOp>([&](IndexTreeSparseDomainOp op) {
        switch((TensorFormatEnum)op.getFormat()){
          case TensorFormatEnum::UNK:
          case TensorFormatEnum::D:
            assert(false && "Invalid format for IndexTreeSparseDomainOp");
            return (LoopInfo*)nullptr;

            break;
          case TensorFormatEnum::CN:
          case TensorFormatEnum::CU:
            return SparseLoopInfo::build(op, rewriter, zero_loop_inputs);
          case TensorFormatEnum::S:
            assert(false && "Singleton loop inside a mask is not supported.");
            return (LoopInfo*)nullptr;
            // return SingletonLoopInfo::build(op, rewriter, inputs, parent_info);
        }
      })
      .Case<IndexTreeWorkspaceDomainOp>([&](IndexTreeWorkspaceDomainOp op) {
        return WorkspaceLoopInfo::build(op, rewriter, zero_loop_inputs);
      })
      .Case<IndexTreeDomainIntersectionOp>([&](IndexTreeDomainIntersectionOp op) {
        return IntersectionLoopInfo::build(op, rewriter, zero_loop_inputs);
      })
      .Default([](Operation *op) {
        assert(false && "IndexNode not given a valid domain");
        return nullptr;
      });

      // Fill in bit tensor
      auto after_zero_loop = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(zero_loop->loopBody);
      Value crd = zero_loop->getCrd(rewriter);
      Value f = rewriter.create<arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
      Value updated_tensor = rewriter.create<tensor::InsertOp>(loc, zero_loop->getInput(0).getType(), f, zero_loop->getInput(0), crd);
      zero_loop->updateOutput(rewriter, 0, updated_tensor);
      rewriter.restoreInsertionPoint(after_zero_loop);
      updateOutput(
        zero_mask_op.getInit(), 
        zero_mask_op.getResult(), 
        zero_loop->getResults()[0], 
        parent_info, 
        rewriter
      );
      
      delete zero_loop;
      return success();
    }

    mlir::LogicalResult convertTensorAccessOp(IndexTreeIndexToTensorOp access_op, IRRewriter &rewriter)
    {
      // Find the position to insert these operations based off the nearest use.
      // TODO: figure out order of users?
      for(auto user : access_op->getUsers()){
        if(auto node = llvm::dyn_cast<IndexTreeNode>(user)) {
          Value parent = node.getParentNode();
          auto parent_iterator = nodeMap.find(parent);
          if(parent_iterator != nodeMap.end()){
            // Jump out of inner loop if necessary
            rewriter.setInsertionPoint(parent_iterator->getSecond()->loopBody);
          }
          break;
        }
      }
      LoopInfo* parent_info = nodeMap.find(access_op.getIndex())->getSecond();
      Value tensor = mapInputIntoLoop(access_op.getTensor(), parent_info);
      auto dim = access_op.getDim();

      Value access_crd = parent_info->getCrd(rewriter);
      Value access_pos = parent_info->getPos(rewriter, tensor, dim);
      if(auto tensor_type = dyn_cast<tensorAlgebra::SparseTensorType>(tensor.getType()))
      {
        TensorFormatEnum format = (TensorFormatEnum) tensor_type.getFormat()[2 * dim];
        if(format == TensorFormatEnum::D)
        {
          // TODO: This is incorrect, deal with reordering!!!!
          if(access_op.getPrevDim()) {
            auto loc = access_op.getLoc();
            auto index_type = rewriter.getIndexType();
            Value dim_size = rewriter.create<tensorAlgebra::SpTensorGetDimSize>(loc, index_type, tensor, rewriter.getI32IntegerAttr(dim));
            Value pos_start = rewriter.create<arith::MulIOp>(loc, index_type, dim_size, access_op.getPrevDim());
            access_pos = rewriter.create<arith::AddIOp>(loc, index_type, pos_start, access_pos);
          }
        }
      } 
      rewriter.replaceAllUsesWith(access_op.getPos(), access_pos);
      rewriter.replaceAllUsesWith(access_op.getCrd(), access_crd);
      return success();
    }

    mlir::LogicalResult convertRoot(IndexTreeRootOp root_op, IRRewriter &rewriter) {
      IndexTreeOp tree = root_op->getParentOfType<IndexTreeOp>();
      indexTree::YieldOp yield = cast<indexTree::YieldOp>(tree.getBody()->getTerminator());
      IRMapping map;
      LoopInfo* sentinel_info = SentinelLoopInfo::build(tree.getBody()->getArguments(), tree.getResults(), root_op, map, yield);
      nodeMap.insert(std::make_pair(root_op.getResult(), sentinel_info));
      
      uint32_t i = 0;
      for(Value v: tree.getBody()->getArguments()){
        leafMap.insert(std::make_pair(v, i));
        i++;
      }
      return success();
    }

    mlir::LogicalResult convertIndexNode(IndexTreeIndicesOp index_node_op, IRRewriter &rewriter)
    {
      IndexTreeOp tree = index_node_op->getParentOfType<IndexTreeOp>();
      Operation* domain_op = index_node_op.getDomain().getDefiningOp();
      Value index_node = index_node_op->getResult(0);
      LoopInfo* parent_info = nodeMap.find(index_node_op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);

      LoopInfo* loop_info = llvm::TypeSwitch<Operation*, LoopInfo*>(domain_op) 
        .Case<IndexTreeDenseDomainOp>([&](IndexTreeDenseDomainOp op) {
          if(index_node_op.getIsParallel()){
            ValueRange inputs = parent_info->getInputs();
            auto output_analysis = Pass::getChildAnalysis<IndexTreeInferOutputSets>(index_node_op->getParentOp());
            auto output_sets = output_analysis.getOutputSets(index_node_op);
            uint32_t i = 0;
            for(Value v: inputs){
              output_sets.insert(std::make_pair(v, output_sets[tree.getBody()->getArgument(i)]));
              i++;
            }
            return DenseParallelLoopInfo::build(op, rewriter, parent_info->getInputs(), output_sets);
          }
          return DenseLoopInfo::build(op, rewriter, parent_info->getInputs());
        })
        .Case<IndexTreeSparseDomainOp>([&](IndexTreeSparseDomainOp op) {
          switch((TensorFormatEnum)op.getFormat()){
            case TensorFormatEnum::D:
            case TensorFormatEnum::UNK:
            {
              assert(false && "Invalid format for IndexTreeSparseDomainOp");
              return (LoopInfo*)nullptr;
              break;
            }
            case TensorFormatEnum::CN:
            case TensorFormatEnum::CU:
              return SparseLoopInfo::build(op, rewriter, parent_info->getInputs());
            case TensorFormatEnum::S:
              return SingletonLoopInfo::build(op, rewriter, parent_info->getInputs(), parent_info);
          }
        })
        .Case<IndexTreeWorkspaceDomainOp>([&](IndexTreeWorkspaceDomainOp op) {
          op.getTensorMutable().assign(mapInputIntoLoop(op.getTensor(), parent_info));
          return WorkspaceLoopInfo::build(op, rewriter, parent_info->getInputs());
        })
        .Case<IndexTreeDomainIntersectionOp>([&](IndexTreeDomainIntersectionOp op) {
          return IntersectionLoopInfo::build(op, rewriter, parent_info->getInputs());
        })
        .Case<IndexTreeMaskedDomainOp>([&](IndexTreeMaskedDomainOp op) {
          op.getMaskMutable().assign(mapInputIntoLoop(op.getMask(), parent_info));
          return MaskedLoopInfo::build(op, rewriter, parent_info->getInputs());
        })
        .Default([](Operation *op) {
          assert(false && "IndexNode not given a valid domain");
          return nullptr;
        });

      ValueRange loop_outputs = loop_info->getResults();
      uint32_t i = 0;
      for(Value v : loop_outputs)
      {
        parent_info->updateOutput(rewriter, i, v);
        i++;
      }
      nodeMap.insert(std::make_pair(index_node, loop_info));
      rewriter.setInsertionPoint(loop_info->loopBody);
      return success();
    }

    mlir::LogicalResult convertTree(IndexTreeOp treeOp, IRRewriter &rewriter)
    {
      // Walk the Block of the IndexTreeOp and Convert Operands in order to maintain
      // order in generated code
      SmallVector<Operation*> toDelete;
      LLVM_DEBUG({logger.startLine() << "Current Tree: \n" << treeOp << "\n";});
      for (Operation &op : treeOp.getRegion().front()){
        // A node in the tree should be able to use the loopMaps map to find where it 
        // should be rewritten to
        LogicalResult result = llvm::TypeSwitch<Operation*, LogicalResult>(&op) 
          .Case<IndexTreeRootOp>([&](IndexTreeRootOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertRoot(op, rewriter);
          })
          .Case<IndexTreeComputeOp>([&](IndexTreeComputeOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertCompute(op, rewriter);
          })
          .Case<IndexTreeCleanWorkspaceOp>([&](IndexTreeCleanWorkspaceOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertClearWorkspaceOp(op, rewriter);
          })
          .Case<ComputeSymbolicDomainOp>([&](ComputeSymbolicDomainOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertSymbolicDomainOp(op, rewriter);
          })
          .Case<ComputeSymbolicDomainRowOp>([&](ComputeSymbolicDomainRowOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertSymbolicDomainEndRowOp(op, rewriter);
          })
          .Case<IndexTreeFillMaskOp>([&](IndexTreeFillMaskOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertFillMaskOp(op, rewriter);
          })
          .Case<IndexTreeZeroMaskOp>([&](IndexTreeZeroMaskOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertZeroMaskOp(op, rewriter);
          })
          .Case<IndexTreeIndexToTensorOp>([&](IndexTreeIndexToTensorOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertTensorAccessOp(op, rewriter);
          })
          .Case<IndexTreeIndicesOp>([&](IndexTreeIndicesOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            LogicalResult result = convertIndexNode(op, rewriter);
            // LLVM_DEBUG({logger.startLine() << "Current Tree: \n" << treeOp << "\n";});
            return result;

          })
          .Default([](Operation *op) {
            return success();
          });

          if(result.failed()){
            LLVM_DEBUG({
              logger.startLine() << "Conversion failed!" <<  "\n";
            });
            return failure();
          }

          if(isa<indexTree::YieldOp>(op)){
            break;
          }
          toDelete.push_back(&op);
      }

      LLVM_DEBUG({logger.startLine() << "Current Tree: \n" << treeOp << "\n";});
      for (auto op = toDelete.rbegin(); op != toDelete.rend(); ++op){
        // Erase all the old ops in the region
        LLVM_DEBUG({
          logger.startLine() <<  "Removing: " << (*op)->getName() << "\n";
          if(!(*op)->use_empty()) {
            logger.indent();
            for(auto user : (*op)->getUsers()) {
              logger.startLine() << "Op still used by: " << user->getName() << "\n";
            }
          }
        });

        logger.resetIndent();
        rewriter.eraseOp(*op);
      }

      return success();
    }

    void runOnOperation() override {

      std::vector<IndexTreeOp> iTrees;
      func::FuncOp funcOp = getOperation();
      funcOp.walk([&](IndexTreeOp op){ iTrees.push_back(op); });

      comet_vdump(funcOp->getParentOfType<ModuleOp>());
      comet_debug() << "\n";

      for(auto op : iTrees)
      {
        OpBuilder builder(op);
        IRRewriter rewriter(builder);
        if(failed(convertTree(op, rewriter))){
          return signalPassFailure();
        }      
      }

      TypeConverter typeConverter;
      mlir::ConversionTarget target(getContext());
      target.addLegalDialect<scf::SCFDialect, tensor::TensorDialect>();

      mlir::RewritePatternSet patterns(&getContext());
//      patterns.add<TransformSymbolicForallOp>(&getContext());
      if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();

      LegalizeSymbolicForallOp(getOperation());

      LegalizeNumericForallOp(getOperation());

      comet_vdump(funcOp->getParentOfType<ModuleOp>());
      comet_debug() << "\n";
    }


  };
}  /// anonymous namespace

/// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createLowerIndexTreeToSCFPass()
{
  return std::make_unique<LowerIndexTreeToSCFPass>();
}
