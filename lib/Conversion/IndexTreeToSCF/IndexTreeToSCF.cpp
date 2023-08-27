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
// #include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "llvm/Support/Debug.h"
#include "llvm/ADT/StringSet.h"
#include <iostream>
#include <algorithm>
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

#define DEBUG_TYPE "lowering-it-to-scf"

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_LowerIndexTreeToSCFPass
// #define DEBUG_MODE_LowerIndexTreeToSCFPass
// #endif
// #ifndef DEBUG_MODE_LowerIndexTreeToSCFPass
// #define DEBUG_MODE_LowerIndexTreeToSCFPass
// #endif

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
#define comet_debug() llvm::errs() << __FILE__ << ":" << __LINE__ << " "
#define comet_pdump(n)                                \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " "; \
  n->dump()
#define comet_vdump(n)                                \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " "; \
  n.dump()
#else
#define comet_debug() if(true){}else llvm::errs()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

namespace comet {
#define GEN_PASS_DEF_CONVERTINDEXTREETOSCF
#include "comet/Conversion/Passes.h.inc"
} // namespace comet

#define TENSOR_NUMS 3
#define INPUT_TENSOR_NUMS 2

namespace {

// Valid semiring operators.
static const llvm::StringSet<> Semiring_ops{
  "atan2", "div", "eq", "first", "ge", "gt", "hypot",
  "land", "le", "lor", "lt", "max", "minxy", "minus",
  "ne", "pairxy", "plusxy", "pow", "rdiv", "second", "times"};

// List of valid semiring operands for reduce op
static const llvm::StringSet<> Semiring_reduceOps{"any", "land", "lor", "max",
                                                  "minxy", "plusxy", "times",
                                                  "noop"}; // noop is for monoid op support

/// MASKING_TYPE to indicate what type of masking is used.
enum MASKING_TYPE {
  NO_MASKING = 0,
  PUSH_BASED_MASKING = 1,
  PULL_BASED_MASKING = 2
};

/// class MaksingInfo, passed as a parameter to the formSemiringLoopBody() to indicate if using masking or not.
struct MaskingInfo {
public:
  MASKING_TYPE mask_type;

  /// %44 = ta.sptensor_construct(%39, %40, %41, %42, %43, %32, %33, %34, %35, %36, %37, %38) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  /// mask_tensor = %44
  /// mask_rowptr = mask_tensor.getOperand(2);
  /// mask_col = mask_tensor.getOperand(3);
  /// mask_val = mask_tensor.getOperand(4);
  mlir::Value mask_tensor;
  mlir::Value mask_rowptr;
  mlir::Value mask_col;
  mlir::Value mask_val;

//  /// Push-based mask auxiliary dense vector
//  mlir::Value states;  /// %states = memref.alloc(%25) {alignment = 32 : i64} : memref<?xi1>

  /// TODO(zhen.peng): Pull-based mask info and auxiliary variables.

public:
  MaskingInfo() : mask_type(NO_MASKING) { }

//  MaskingInfo(MASKING_TYPE type_, mlir::Value states_) : maskType(type_), states(states_) { }

  void dump() {
    switch (mask_type) {
      case NO_MASKING:
        std::cout << "maskType: NO_MASKING\n";
        break;
      case PUSH_BASED_MASKING:
        std::cout << "maskType: PUSH_BASED_MASKING " << "mask_tensor: ";
        mask_tensor.dump();
//        std::cout << "maskType: PUSH_BASED_MASKING " << "states: ";
//        states.dump();
        break;
      case PULL_BASED_MASKING:
        std::cout << "maskType: PULL_BASED_MASKING ... Not supported";
        break;
    }
  }
};

class OpsTree {
  // private:
public:
  std::vector<scf::ForOp> forOps; // The (nested) for loops
  std::vector<Value> accessIdx;   // The coordinate of accessing that dimension
  std::vector<scf::ForOp> symbolicForOps;  // For-loops in symbolic phase (if necessary)
  std::vector<Value> symbolicAccessIdx;  // The accessing index for that for-loop in symbolic phase (if necessary)
//  std::vector<Value> cmptOps;     // The computation ops (no used?)
  std::vector<OpsTree *> children;
  OpsTree *parent;
  int id;

public:
  OpsTree() {}

  OpsTree(std::vector<scf::ForOp> &forOps, std::vector<Value> &accessIdx,
          OpsTree *parent, int id) :
          forOps(forOps), accessIdx(accessIdx), parent(parent), id(id) {
//    this->forOps = forOps;
//    this->accessIdx = accessIdx;
//    this->cmptOps = cmptOps;
//    this->parent = parent;
//    this->id = id;
  }

  OpsTree(std::vector<scf::ForOp> &forOps, std::vector<Value> &accessIdx,
          OpsTree *parent) :
          forOps(forOps), accessIdx(accessIdx), parent(parent){
//    this->forOps = forOps;
//    this->accessIdx = accessIdx;
//    this->cmptOps = cmptOps;
//    this->parent = parent;
  }

  ~OpsTree() {}

  void addChild(OpsTree *tree) { // const T& node
    this->children.push_back(tree);
  }

  std::vector<scf::ForOp> &getForOps() {
    return this->forOps;
  }

  OpsTree *getParent() {
    return this->parent;
  }

  void setForOps(std::vector<scf::ForOp> &forOps) {
    this->forOps = forOps;
  }

  std::vector<OpsTree *> &getChildren() {
    return this->children;
  }
};

/// ----------------- ///
/// struct to pass symbolic phase information to the numeric phase
/// ----------------- ///
struct SymbolicInfo {
  bool is_SpGEMM = false;   /// Flag if the compute node is SpGEMM.
                            /// All other members are only used when is_SpGEMM is true;
  Value mtxC_num_rows;
  Value mtxC_num_cols;

  Value mtxC_rowptr;   /// Output C's rowptr array when do C = A * B and they are all sparse
                       /// %alloc_100 = memref.alloc(%43) : memref<?xindex>
  Value mtxC_col;   /// Output C's col array when do C = A * B and they are all sparse
                    /// %alloc_104 = memref.alloc(%44) : memref<?xindex>
  Value mtxC_val;   /// Output C's val array when do C = A * B and they are all sparse
                    /// %alloc_108 = memref.alloc(%44) : memref<?xf64>
  Value row_offset;  /// In Numeric Phase, row_offset is the insertion location in the C_col and C_val.
  Value mtxC;   /// The sparse tensor
                /// It is %55 below.

  /// The information of the output matrix C
  /// %55 = ta.sptensor_construct(%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %8, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  /**
    sptensor_construct(
        A1pos,  /// number of rows
        A1crd,  /// discard
        A2pos,  /// rowptr array
        A2crd,  /// col_id array
        Aval, /// data array
        A1pos_size,
        A1crd_size,
        A2pos_size,
        A2crd_size,
        Aval_size,
        dim1_size,
        dim2_size,
    )
  */

//  Value ws_bitmap;   /// workspace's bitmap to tell if a column ID is visited.
//  Value mask_array;  /// the intermediate dense vector for a row of the mask.
};

/// ----------------- ///
/// Auxiliary structures for the numeric phase
/// ----------------- ///
struct NumericInfo {
  Value ws_bitmap;   /// workspace's bitmap to tell if a column ID is visited.
  Value mask_array;  /// the intermediate dense vector for a row of the mask.
};


/// ----------------- ///
/// Remove an operantion's user who is a memref.store
/// This is very ad-hoc, just to avoid segmentation fault for old very large C.val array and C.col array.
/// ----------------- ///
void removeMemrefStoreUser(Value &opd) {
  {
    comet_vdump(opd);
  }
  std::vector<Operation *> users;
  for (Operation *user : opd.getUsers()) {
    if (isa<memref::StoreOp>(user)) {
      users.push_back(user);
      {
        comet_pdump(user);
      }
    }
  }
  for (Operation *user : users) {
    user->erase();
  }
}

/// ----------------- ///
/// Find all users of the old_Value, and replace those users' corresponding operand to new_Value. For example,
/// "ta.print"(%old_Value)  =>  "ta.print"(%new_Value)
/// ----------------- ///
void replaceOldValueToNewValue(Value &old_Value,
                               Value &new_Value) {
  {
    comet_vdump(old_Value);
    comet_vdump(new_Value);
  }

  /// Traverse each user of new_Value
  std::vector<Operation *> users;
  for (Operation *user : old_Value.getUsers()) {
    users.push_back(user);
  }
  DominanceInfo domInfo(new_Value.getDefiningOp());  // To check dominance
  for (Operation *user : users) {
    {
      comet_debug() << "before replace operand.\n";
      comet_pdump(user);
    }
    if (!domInfo.dominates(new_Value, user)) {
      continue;
    }
    uint64_t op_i = 0;
    for (Value op : user->getOperands()) {
      /// Find the mtxC in the user's operands
      if (op.getDefiningOp() == old_Value.getDefiningOp()) {
        /// Replace the old sparse tensor to the new one
        user->setOperand(op_i, new_Value);
        {
          comet_debug() << "after replace operand.\n";
          comet_pdump(user);
        }
      }
      ++op_i;
    }
  }
}


/// ----------------- ///
/// Add declaration of the function comet_index_func;
/// ----------------- ///
/// func.func private @comet_sort_index(memref<*xindex>, index, index)
void declareSortFunc(ModuleOp &module,
                     MLIRContext *ctx,
                     Location loc) {
  IndexType indexType = IndexType::get(ctx);

  /// Declare comet_sort_index()
  /// func.func private @comet_sort_index(memref<*xindex>, index, index)
  auto sort_index_func = FunctionType::get(ctx,
                                           {UnrankedMemRefType::get(indexType, 0), indexType, indexType} /* inputs */, {} /* return */);
  std::string func_name = "comet_sort_index";
  if (!hasFuncDeclaration(module, func_name /* func name */)) {
    func::FuncOp func_declare = func::FuncOp::create(loc,
                                                     func_name,
                                                     sort_index_func,
                                                     ArrayRef<NamedAttribute>{});
    func_declare.setPrivate();
    module.push_back(func_declare);
  }
}


/// Get mask_rowptr, mask_col, and mask_val arrays.
/// ----------------- ///
/// mask_tensor = %44
/// mask_rowptr = %alloc_99
/// mask_col = %alloc_104
/// mask_val = %alloc_109
/// ----------------- ///
/// %41 = bufferization.to_tensor %alloc_99 : memref<?xindex>
/// %42 = bufferization.to_tensor %alloc_104 : memref<?xindex>
/// %43 = bufferization.to_tensor %alloc_109 : memref<?xf64>
/// %44 = ta.sptensor_construct(%39, %40, %41, %42, %43, %32, %33, %34, %35, %36, %37, %38) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
/// ----------------- ///
void getMaskSparseTensorInfo(MaskingInfo &maskingInfo /* contents updated after call*/) {
  Value &mask_tensor = maskingInfo.mask_tensor;

  // A2pos
  Value mask_rowtpr_buff = mask_tensor.getDefiningOp()->getOperand(4);    // 2
  // A2pos
  Value mask_rowtpr_buff = mask_tensor.getDefiningOp()->getOperand(4);    // 2
  maskingInfo.mask_rowptr = mask_rowtpr_buff.getDefiningOp()->getOperand(0);

  // A2crd
  Value mask_col_buff = mask_tensor.getDefiningOp()->getOperand(5);   // 3
  // A2crd
  Value mask_col_buff = mask_tensor.getDefiningOp()->getOperand(5);   // 3
  maskingInfo.mask_col = mask_col_buff.getDefiningOp()->getOperand(0);

  // Aval
  Value mask_val_buff = mask_tensor.getDefiningOp()->getOperand(8);   // 4
  // Aval
  Value mask_val_buff = mask_tensor.getDefiningOp()->getOperand(8);   // 4
  maskingInfo.mask_val = mask_val_buff.getDefiningOp()->getOperand(0);

  {
    comet_vdump(mask_tensor);
    comet_vdump(maskingInfo.mask_rowptr);
    comet_vdump(maskingInfo.mask_col);
    comet_vdump(maskingInfo.mask_val);
  }
}


/// ----------------- ///
/// Generate the symbolic for-loop that initialize the mark_array by using the mask.
/// ----------------- ///
///      %j_loc_start = memref.load %mask_rowptr[%i_idx] : memref<?xindex>
///      %j_loc_bound = memref.load %mask_rowptr[%i_idx_plus_one] : memref<?xindex>
///      scf.for %j_loc = %j_loc_start to %j_loc_bound step %c1 {
///        %val = memref.load %mask_val[%j_loc] : memref<?xf64>
///        %70 = arith.cmpf une, %val, %cst : f64
///        scf.if %70 {
///          %j_idx = memref.load %mask_col[%arg1] : memref<?xindex>
///          memref.store %mark, %mark_array[%j_idx] : memref<?xi1>
///        }
///      }
void genSymbolicInitMarkArrayByMask(std::vector<OpsTree *> &three_index_ancestors,
                            Value &new_mark_reg,
                            Value &alloc_mark_array,
                            MaskingInfo &maskingInfo,
                            OpBuilder &builder,
                            Location &loc) {
  /// Store the insertion point
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Set the Insertion Point to the place before the 2nd-level symbolic for-loop
  builder.setInsertionPoint(three_index_ancestors[1]->symbolicForOps[0]);

  /// Generate the for-loop entry
  Value &mask_rowptr = maskingInfo.mask_rowptr;
  Value &mask_col = maskingInfo.mask_col;
  Value &mask_val = maskingInfo.mask_val;
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  Value i_idx = three_index_ancestors[2]->symbolicAccessIdx[0];
  Value i_idx_plus_one = builder.create<AddIOp>(loc, i_idx, const_index_1);
  Value j_loc_start = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx});
  Value j_loc_bound = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx_plus_one});
  auto for_loop = builder.create<scf::ForOp>(loc,
                                             j_loc_start /* lower_bound */,
                                             j_loc_bound /* upper_bound*/,
                                             const_index_1 /* step */);
  builder.setInsertionPointToStart(for_loop.getBody());

  /// Generate the for-loop body
  Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0));
  Value j_loc = for_loop.getInductionVar();
  Value val = builder.create<memref::LoadOp>(loc,mask_val, ValueRange{j_loc});
  Value not_zero = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, val, const_f64_0);
  auto if_not_zero = builder.create<scf::IfOp>(loc, not_zero, false /*NoElseRegion*/);
  builder.setInsertionPointToStart(&if_not_zero.getThenRegion().front());
  Value j_idx = builder.create<memref::LoadOp>(loc, mask_col, ValueRange{j_loc});
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_mark = builder.create<memref::StoreOp>(loc,
                                                    new_mark_reg,
                                                    alloc_mark_array,
                                                    ValueRange{j_idx});
#else
  builder.create<memref::StoreOp>(loc,
                                  new_mark_reg,
                                  alloc_mark_array,
                                  ValueRange{j_idx});
#endif
  {
    comet_vdump(store_mark);
    comet_vdump(for_loop);
  }
  /// Restore the insertion point
  builder.restoreInsertionPoint(last_insertion_point);
}

/// ----------------- ///
/// Generate the numeric for-loop that initialize the mark_array by using the mask
/// ----------------- ///
///      %j_loc_start = memref.load %mask_rowptr[%i_idx] : memref<?xindex>
///      %j_loc_bound = memref.load %mask_rowptr[%i_idx_plus_one] : memref<?xindex>
///      scf.for %j_loc = %j_loc_start to %j_loc_bound step %c1 {
///        %val = memref.load %mask_val[%j_loc] : memref<?xf64>
///        %70 = arith.cmpf une, %val, %cst : f64
///        scf.if %70 {
///          %j_idx = memref.load %mask_col[%arg1] : memref<?xindex>
///          memref.store %mark, %mark_array[%j_idx] : memref<?xindex>
///        }
///      }
void genNumericInitMarkArrayByMask(std::vector<scf::ForOp> &forLoops /* numeric for-loops, from innermost to outermost*/,
                                   Value &new_mark_reg,
                                   Value &alloc_mark_array,
                                   MaskingInfo &maskingInfo,
                                   OpBuilder &builder,
                                   Location &loc) {
  /// Store the insertion point
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Set the Insertion Point to the place before the 2nd-level symbolic for-loop
  builder.setInsertionPoint(forLoops[1]);

  /// Generate the for-loop entry
  Value &mask_rowptr = maskingInfo.mask_rowptr;
  Value &mask_col = maskingInfo.mask_col;
  Value &mask_val = maskingInfo.mask_val;
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  Value i_idx = forLoops[2].getInductionVar();  /// i_idx is the induction variable of the outermost for-loop.
  Value i_idx_plus_one = builder.create<AddIOp>(loc, i_idx, const_index_1);
  Value j_loc_start = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx});
  Value j_loc_bound = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx_plus_one});
  auto for_loop = builder.create<scf::ForOp>(loc,
                                             j_loc_start /* lower_bound */,
                                             j_loc_bound /* upper_bound*/,
                                             const_index_1 /* step */);
  builder.setInsertionPointToStart(for_loop.getBody());

  /// Generate the for-loop body
  Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0));
  Value j_loc = for_loop.getInductionVar();
  Value val = builder.create<memref::LoadOp>(loc,mask_val, ValueRange{j_loc});
  Value not_zero = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, val, const_f64_0);
  auto if_not_zero = builder.create<scf::IfOp>(loc, not_zero, false /*NoElseRegion*/);
  builder.setInsertionPointToStart(&if_not_zero.getThenRegion().front());
  Value j_idx = builder.create<memref::LoadOp>(loc, mask_col, ValueRange{j_loc});
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_mark = builder.create<memref::StoreOp>(loc,
                                                    new_mark_reg,
                                                    alloc_mark_array,
                                                    ValueRange{j_idx});
#else
  builder.create<memref::StoreOp>(loc,
                                  new_mark_reg,
                                  alloc_mark_array,
                                  ValueRange{j_idx});
#endif

  {
    comet_vdump(store_mark);
    comet_vdump(for_loop);
  }
  /// Restore the insertion point
  builder.restoreInsertionPoint(last_insertion_point);
}

/// ----------------- ///
/// Generate the 1st level symbolic for-loop
/// ----------------- ///
void genSymbolicForLoopsLevel1(Value &mtxA,
                               OpBuilder &builder,
                               Location &loc,
                               std::vector<OpsTree *> &three_index_ancestors /* output: three_index_ancestors[2] */) {
  /// ----------------- ///
  /// Generate the 1st level (outermost) for-loops.
  /// C[i, j] = A[i, k] * B[k, j]
  /// for (every A[i,:] in A[])
  /// ----------------- ///
  /// scf.for %i_idx = %c0 to %num_rows step %c1
  /// ----------------- ///
  /// mtxA.A1pos is num_rows and the 0-th operand of mtxA.
  /// %10 = bufferization.to_tensor %alloc_12 : memref<?xindex>
  /// %15 = ta.sptensor_construct(%10, %11, %12, %13, %14, %3, %4, %5, %6, %7, %8, %9) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  Value num_rows_buffer = mtxA.getDefiningOp()->getOperand(0);
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  Value num_rows_alloc = num_rows_buffer.getDefiningOp()->getOperand(0);
  Value num_rows = builder.create<memref::LoadOp>(loc, num_rows_alloc, ValueRange{const_index_0});
  {
    comet_vdump(num_rows_buffer);
    comet_vdump(num_rows_alloc);
    comet_vdump(num_rows);
  }
  auto for_loop_0 = builder.create<scf::ForOp>(loc,
                                               const_index_0,
                                               num_rows,
                                               const_index_1);
  {
    comet_vdump(for_loop_0);
  }
  Value i_idx = for_loop_0.getInductionVar();
  three_index_ancestors[2]->symbolicForOps.push_back(for_loop_0);
  three_index_ancestors[2]->symbolicAccessIdx.push_back(i_idx);
  builder.setInsertionPointToStart(for_loop_0.getBody());
}

/// ----------------- ///
/// Generate the 2nd level of symbolic for-loop.
/// ----------------- ///
void genSymbolicForLoopsLevel2(Value &mtxA,
                               OpBuilder &builder,
                               Location &loc,
                               std::vector<OpsTree *> &three_index_ancestors /* output: three_index_ancestors[1] */) {
  /// ----------------- ///
  /// Generate the 2nd level of for-loop.
  /// for (every A[i, k] in A[i, :])
  /// ----------------- ///
  /// %i_idx_plus_1 = arith.addi %i_idx, %c1 : index
  /// %k_loc_start = memref.load %A_rowptr[%i_idx] : memref<?xindex>
  /// %k_loc_bound = memref.load %A_rowptr[%i_idx_plus_1] : memref<?xindex>
  /// scf.for %k_loc = %k_loc_start to %k_loc_bound step %c1 {
  ///     %k_idx = memref.load %A_col[%k_loc] : memref<?xindex>
  /// ----------------- ///
  /// mtxA.A2pos is A_rowptr, the 2nd operand of mtxA. mtxA.A2crd is A_col, the 3nd operand of mtxA.
  /// %12 = bufferization.to_tensor %alloc_22 : memref<?xindex>
  /// %15 = ta.sptensor_construct(%10, %11, %12, %13, %14, %3, %4, %5, %6, %7, %8, %9) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  Value A_rowptr_buffer = mtxA.getDefiningOp()->getOperand(4);  //2
  Value A_rowptr_alloc = A_rowptr_buffer.getDefiningOp()->getOperand(0);
  Value &i_idx = three_index_ancestors[2]->symbolicAccessIdx[0];
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  auto i_idx_plus_1 = builder.create<AddIOp>(loc, i_idx, const_index_1);
  {
    comet_vdump(A_rowptr_buffer);
    comet_vdump(A_rowptr_alloc);
    comet_vdump(i_idx_plus_1);
  }
  auto k_loc_start = builder.create<memref::LoadOp>(loc, A_rowptr_alloc, ValueRange{i_idx});
  auto k_loc_bound = builder.create<memref::LoadOp>(loc, A_rowptr_alloc, ValueRange{i_idx_plus_1});
  auto for_loop_1 = builder.create<scf::ForOp>(loc,
                                               k_loc_start /* lowerBound */,
                                               k_loc_bound /* upperBound */,
                                               const_index_1 /* step */);
  builder.setInsertionPointToStart(for_loop_1.getBody());
  {
    comet_vdump(k_loc_start);
    comet_vdump(k_loc_bound);
    comet_vdump(for_loop_1);
  }
  Value k_loc = for_loop_1.getInductionVar();
  Value A_col_buffer = mtxA.getDefiningOp()->getOperand(5); //3
  Value A_col_buffer = mtxA.getDefiningOp()->getOperand(5); //3
  Value A_col_alloc = A_col_buffer.getDefiningOp()->getOperand(0);
  Value k_idx = builder.create<memref::LoadOp>(loc, A_col_alloc, ValueRange{k_loc});
  {
    comet_vdump(A_col_buffer);
    comet_vdump(A_col_alloc);
    comet_vdump(k_idx);
  }
  three_index_ancestors[1]->symbolicForOps.push_back(for_loop_1);
  three_index_ancestors[1]->symbolicAccessIdx.push_back(k_idx);
  {
    comet_vdump(for_loop_1);
  }
}

/// ----------------- ///
/// Generate the 3rd level of symbolic for-loop.
/// ----------------- ///
void genSymbolicForLoopsLevel3(Value &mtxB,
                               OpBuilder &builder,
                               Location &loc,
                               std::vector<OpsTree *> &three_index_ancestors /* output: three_index_ancestors[0] */) {
  /// ----------------- ///
  /// Generate the 3rd level of for-loop.
  /// for (every B[k, j] in B[k, :])
  /// ----------------- ///
  /// %k_idx_plus_1 = arith.addi %k_idx, %c1 : index
  /// %j_loc_start = memref.load %B_rowptr[%k_idx] : memref<?xindex>
  /// %j_loc_bound = memref.load %B_rowptr[%k_idx_plus_1] : memref<?xindex>
  /// scf.for %j_loc = %j_loc_start to %j_loc_bound step %c1 {
  ///     %j_idx = memref.load %B_col[%j_loc] : memref<?xindex>
  /// mtxB.A2pos is B_rowptr, the 2nd operand of mtxB. mtxB.A2crd is B_col, the 3nd operand of mtxB.
  Value B_rowptr_buffer = mtxB.getDefiningOp()->getOperand(4);    // 2
  Value B_rowptr_buffer = mtxB.getDefiningOp()->getOperand(4);    // 2
  Value B_rowptr_alloc = B_rowptr_buffer.getDefiningOp()->getOperand(0);
  Value k_idx = three_index_ancestors[1]->symbolicAccessIdx[0];
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  auto k_idx_plus_1 = builder.create<AddIOp>(loc, k_idx, const_index_1);
  {
    comet_vdump(B_rowptr_buffer);
    comet_vdump(B_rowptr_alloc);
    comet_vdump(k_idx_plus_1);
  }
  auto j_loc_start = builder.create<memref::LoadOp>(loc, B_rowptr_alloc, ValueRange{k_idx});
  auto j_loc_bound = builder.create<memref::LoadOp>(loc, B_rowptr_alloc, ValueRange{k_idx_plus_1});
  auto for_loop_2 = builder.create<scf::ForOp>(loc,
                                               j_loc_start /* lowerBound */,
                                               j_loc_bound /* upperBound */,
                                               const_index_1 /* step */);
  builder.setInsertionPointToStart(for_loop_2.getBody());
  {
    comet_vdump(j_loc_start);
    comet_vdump(j_loc_bound);
    comet_vdump(for_loop_2);
  }
  Value j_loc = for_loop_2.getInductionVar();
  Value B_col_buffer = mtxB.getDefiningOp()->getOperand(5);   //3
  Value B_col_buffer = mtxB.getDefiningOp()->getOperand(5);   //3
  Value B_col_alloc = B_col_buffer.getDefiningOp()->getOperand(0);
  Value j_idx = builder.create<memref::LoadOp>(loc, B_col_alloc, ValueRange{j_loc});
  {
    comet_vdump(B_col_buffer);
    comet_vdump(B_col_alloc);
    comet_vdump(j_idx);
  }
  three_index_ancestors[0]->symbolicForOps.push_back(for_loop_2);
  three_index_ancestors[0]->symbolicAccessIdx.push_back(j_idx);
  {
    comet_vdump(for_loop_2);
    comet_vdump(three_index_ancestors[2]->symbolicForOps[0]);
  }
}

/// ----------------- ///
/// Generate symbolic for-loops
/// ----------------- ///
void genSymbolicForLoops(indexTree::IndexTreeComputeOp &cur_op,
                         OpsTree *opstree,
                         OpBuilder &builder,
                         Location &loc,
                         std::vector<OpsTree *> &three_index_ancestors /* output */,
                         Value &num_rows) {
//                         Value &num_rows_alloc /* output */) {
//                         std::vector<scf::ForOp> &symbolicForLoops) {

  /// Store the insertion point
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Find the 3rd ancestor and set the insertion point.
  /// three_index_ancestors[0] is the nearest ancestor, and
  /// three_index_ancestors[2] is the farthest one.
  OpsTree *third_ancestor = opstree;
  for (int an_i = 0;  third_ancestor && an_i < 3; ++an_i) {
    third_ancestor = third_ancestor->getParent();
    three_index_ancestors.push_back(third_ancestor);
  }
  assert(third_ancestor && "This compute node cur_op should have at least 3 ancestors (that all are index nodes).\n");
  scf::ForOp &ancestor_for_loop = third_ancestor->getForOps()[0];
  {
    comet_vdump(ancestor_for_loop);
  }

  /// Set the Insertion Point to before the 3rd ancestor's for-loop (the outermost for-loop).
  builder.setInsertionPoint(ancestor_for_loop);

  /// ---------------- ///
  /// Get for-loop arguments.
  /// ---------------- ///
  /// cur_op
  /// %41 = "it.Compute"(%39, %40) {comp_worksp_opt = true, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
  /// cur_RHS
  /// %39 = "it.ComputeRHS"(%15, %30) {allFormats = [["D", "CU"], ["D", "CU"]], allPerms = [[0, 1], [1, 2]]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>) -> tensor<*xf64>
  /// matrix_A
  /// %15 = ta.sptensor_construct(%10, %11, %12, %13, %14, %3, %4, %5, %6, %7, %8, %9) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  /// matrix_B
  /// %30 = ta.sptensor_construct(%25, %26, %27, %28, %29, %18, %19, %20, %21, %22, %23, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  /*
  sptensor_construct(
      A1pos,  /// number of rows
      A1crd,  /// discard
      A2pos,  /// rowptr array
      A2crd,  /// col_id array
      Aval, /// data array
      A1pos_size,
      A1crd_size,
      A2pos_size,
      A2crd_size,
      Aval_size,
      dim1_size,
      dim2_size,
  )
  */
  Value cur_RHS = cur_op.getRhs()[0];
  Value mtxA = cur_RHS.getDefiningOp()->getOperand(0);
  Value mtxB = cur_RHS.getDefiningOp()->getOperand(1);
  {
    comet_vdump(cur_RHS);
    comet_vdump(mtxA);
    comet_vdump(mtxB);
  }

  /* /// The algorithm we wanted to generate.
  for (i = 0 to mtxA.A1pos) {
      i_plus_1 = i + 1;

      k_loc_start = mtxA.A2pos[i];
      k_loc_bound = mtxA.A2pos[i_plus_1];
      for (k_loc = k_loc_start to k_loc_bound) {
          k = mtxA.A2crd[k_loc];
          k_plus_1 = k + 1;

          j_loc_start = mtxB.A2pos[k];
          j_loc_bound = mtxB.A2pos[k_plus_1];
          for (j_loc = j_loc_start to j_loc_bound) {
              j = mtxB.A2crd[j_loc];

              /// Kernel is here
          }
      }

      /// Set mtxC.rowptr here
  }
  /// Do reduce mtxC.rowptr here
  */

  /// Get the number of rows of mtxA
  num_rows = mtxA.getDefiningOp()->getOperand(18);    //10
  num_rows = mtxA.getDefiningOp()->getOperand(18);    //10
  {
    comet_vdump(num_rows);
  }

  /// ----------------- ///
  /// Generate the 1st level (outermost) for-loops.
  /// C[i, j] = A[i, k] * B[k, j]
  /// for (every A[i,:] in A[])
  /// ----------------- ///
  genSymbolicForLoopsLevel1(mtxA,
                            builder,
                            loc,
                            three_index_ancestors /* output: three_index_ancestors[2] */);

  /// ----------------- ///
  /// Generate the 2nd level of for-loop.
  /// for (every A[i, k] in A[i, :])
  /// ----------------- ///
  genSymbolicForLoopsLevel2(mtxA,
                            builder,
                            loc,
                            three_index_ancestors /* output: three_index_ancestors[1] */);

  /// ----------------- ///
  /// Generate the 3rd level of for-loop.
  /// for (every B[k, j] in B[k, :])
  /// ----------------- ///
  genSymbolicForLoopsLevel3(mtxB,
                            builder,
                            loc,
                            three_index_ancestors /* output: three_index_ancestors[0] */);

  /// Restore the insertion point
  builder.restoreInsertionPoint(last_insertion_point);
}

/// ----------------- ///
/// Get the number of columns of B in C = A * B.
/// cur_op is the compute node. cur_op.RHS is A * B.
/// B is a sparse matrix,
///     sptensor_construct(
///         A1pos,  /// number of rows
///         A1crd,  /// discard
///         A2pos,  /// rowptr array
///         A2crd,  /// col_id array
///         Aval, /// data array
///         A1pos_size,
///         A1crd_size,
///         A2pos_size,
///         A2crd_size,
///         Aval_size,
///         dim1_size,
///         dim2_size,
///     )
/// Therefore, num_cols = B.getOperand(11)
/// ----------------- ///
void getNumOfCols(indexTree::IndexTreeComputeOp &cur_op,
                  Value &num_cols /* output */) {
  Value cur_RHS = cur_op.getRhs()[0];
  Value mtxB = cur_RHS.getDefiningOp()->getOperand(1);
  num_cols = mtxB.getDefiningOp()->getOperand(19);    // 11
  num_cols = mtxB.getDefiningOp()->getOperand(19);    // 11
  {
    comet_vdump(cur_op);
    comet_vdump(cur_RHS);
    comet_vdump(mtxB);
    comet_vdump(num_cols);
  }
}

/// ----------------- ///
/// Generate the variable mark and initialize it to 0.
/// ----------------- ///
void genMarkInit(OpBuilder &builder,
                 Location &loc,
                 Value &alloc_mark /* output */) {
  /// Create the variable mark
  /// %mark = memref.alloc() : memref<1xindex>
  /// memref.store %c0, %mark[%c0] : memref<1xindex>
  MemRefType memTy_alloc_mark = MemRefType::get({1}, builder.getIndexType());
  alloc_mark = builder.create<memref::AllocOp>(loc, memTy_alloc_mark);
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_0_to_mark = builder.create<memref::StoreOp>(loc,
                                                         const_index_0,
                                                         alloc_mark,
                                                         ValueRange{const_index_0});
#else
  builder.create<memref::StoreOp>(loc,
                                 const_index_0,
                                 alloc_mark,
                                 ValueRange{const_index_0});
#endif
  {
    comet_vdump(alloc_mark);
    comet_vdump(store_0_to_mark);
  }
}

/// ----------------- ///
/// Generate the variable mark and initialize it to 0.
/// ----------------- ///
void genMarkArrayInit(indexTree::IndexTreeComputeOp &cur_op,
                      OpBuilder &builder,
                      Location &loc,
                      Value &alloc_mark_array /* output */,
                      Value &num_cols /* output */) {
  /// Store the insertion point
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Get the number of columns of matrix B in C = A * B
  getNumOfCols(cur_op, num_cols);
  {
    comet_vdump(num_cols);
  }

  /// Create the mark_array
  /// %num_cols = memref.load %num_cols_alloc[%c0] : memref<?xindex>
  /// %mark_array = memref.alloc(%num_cols) {alignment = 32 : i64} : memref<?xindex>
  /// scf.for %arg = %c0 to %num_cols step %c1 {
  ///     memref.store %c0, %mark_array[%arg] : memref<?xindex>
  /// }
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  MemRefType memTy_alloc_mark_array = MemRefType::get({ShapedType::kDynamic}, builder.getIndexType());
  alloc_mark_array = builder.create<memref::AllocOp>(loc,
                                                     memTy_alloc_mark_array,
                                                     ValueRange{num_cols},
                                                     builder.getI64IntegerAttr(8) /* alignment bytes */);
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  auto mark_array_init_loop = builder.create<scf::ForOp>(loc,
                                                         const_index_0 /* lowerBound */,
                                                         num_cols /* upperBound */,
                                                         const_index_1 /* step */);
  builder.setInsertionPointToStart(mark_array_init_loop.getBody());
  Value i_idx = mark_array_init_loop.getInductionVar();
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_0_to_mark_array = builder.create<memref::StoreOp>(loc,
                                                               const_index_0,
                                                               alloc_mark_array,
                                                               ValueRange{i_idx});
#else
  builder.create<memref::StoreOp>(loc,
                                  const_index_0,
                                  alloc_mark_array,
                                  ValueRange{i_idx});
#endif
  {
    comet_vdump(alloc_mark_array);
    comet_vdump(mark_array_init_loop);
    comet_vdump(store_0_to_mark_array);
  }

  /// Restore insertion point
  builder.restoreInsertionPoint(last_insertion_point);
}

/// ----------------- ///
/// Update mark += 2 for every row A[i,:] in A
/// ----------------- ///
void genMarkUpdate(Value &alloc_mark,
                   std::vector<OpsTree *> &three_index_ancestors,
                   OpBuilder &builder,
                   Location &loc,
                   Value &new_mark_reg /* output */) {
  /* ----------------- *
  %c2 = arith.constant 2 : index
  %old_val = memref.load %mark[%c0] : memref<1xindex>
  %new_mark = arith.addi %old_val, %c2 : index
  memref.store %new_mark, %mark[%c0] : memref<1xindex>
 * ----------------- */

  /// Store the current insertion point.
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Set the insertion point to the body of the outermost for-loop (three_index_ancestors[2])
  builder.setInsertionPointToStart(three_index_ancestors[2]->symbolicForOps[0].getBody());

  /// Generate mark += 2
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  Value const_index_2 = builder.create<ConstantIndexOp>(loc, 2);

  Value old_mark = builder.create<memref::LoadOp>(loc, alloc_mark,  ValueRange{const_index_0});
  new_mark_reg = builder.create<AddIOp>(loc, old_mark, const_index_2);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_new_mark = builder.create<memref::StoreOp>(loc,
                                                        new_mark_reg,
                                                        alloc_mark,
                                                        ValueRange{const_index_0});
#else
  builder.create<memref::StoreOp>(loc,
                                  new_mark_reg,
                                  alloc_mark,
                                  ValueRange{const_index_0});
#endif
  {
    comet_vdump(old_mark);
    comet_vdump(new_mark_reg);
    comet_vdump(store_new_mark);
  }

  /// Restore the last insertion point.
  builder.restoreInsertionPoint(last_insertion_point);

}

/// ----------------- ///
/// Deallocate mark_array after the outermost for-loop
/// ----------------- ///
/// memref.dealloc %mark_array : memref<?xindex>
void genMarkArrayDealloc(std::vector<OpsTree *> &three_index_ancestors,
                         Value &alloc_mark_array,
                         OpBuilder &builder,
                         Location &loc) {
  /// Set Insertion Point to after the outermost for-loop
  builder.setInsertionPointAfter(three_index_ancestors[2]->symbolicForOps[0]);

  /// Generate deallocating mark_array
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto deallocate_mark_array = builder.create<memref::DeallocOp>(loc, alloc_mark_array);
  {
    comet_vdump(deallocate_mark_array);
  }
#else
  builder.create<memref::DeallocOp>(loc, alloc_mark_array);
#endif
}

/// ----------------- ///
/// Initialize the variable Mark in front of the outermost for-loop.
/// The mark and mark_array are used to replace the WS_bitmap.
/// ----------------- ///
///     %mark = memref.alloc() : memref<1xindex>
///     memref.store %c0, %mark[%c0] : memref<1xindex>
/// ----------------- ///
/// Update the variable Mark for every row A[i,:] in A.
///     %old_mark = memref.load %mark[%c0] : memref<1xindex>
///     %new_mark = arith.addi %old_mark, %c2 : index
///     memref.store %new_mark, %mark[%c0] : memref<1xindex>
/// ----------------- ///
void genMarkAndMarkArray(std::vector<OpsTree *> &three_index_ancestors,
                         indexTree::IndexTreeComputeOp &cur_op,
//                         Value &num_rows_alloc,
                         OpBuilder &builder,
                         Location &loc,
                         Value &alloc_mark /* output */,
                         Value &alloc_mark_array /* output */,
                         Value &new_mark_reg /* output */,
                         Value &num_cols /* output */) {
  /// Store the insertion point
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Set Insertion point to before the outermost for-loop.
  /// three_index_ancestor[0] is the closet ancestor, and three_index_ancestor[2] is the farthest one.
  builder.setInsertionPoint(three_index_ancestors[2]->symbolicForOps[0]);

  /// Create the variable mark
  /// %mark = memref.alloc() : memref<1xindex>
  /// memref.store %c0, %mark[%c0] : memref<1xindex>
  genMarkInit(builder,
              loc,
              alloc_mark /* output */);

  /// Create the mark_array
  /// %num_cols = memref.load %num_cols_alloc[%c0] : memref<?xindex>
  /// %mark_array = memref.alloc(%num_cols) {alignment = 32 : i64} : memref<?xindex>
  /// scf.for %arg = %c0 to %num_cols step %c1 {
  ///     memref.store %c0, %mark_array[%arg] : memref<?xindex>
  /// }
  genMarkArrayInit(cur_op,
                   builder,
                   loc,
                   alloc_mark_array /* output */,
                   num_cols /* output */);

  /// Generate updating mark in every iteration
  ///   %old_val = memref.load %mark[%c0] : memref<1xindex>
  ///   %new_mark = arith.addi %old_val, %c2 : index
  ///   memref.store %new_mark, %mark[%c0] : memref<1xindex>
  genMarkUpdate(alloc_mark,
                three_index_ancestors,
                builder,
                loc,
                new_mark_reg /* output */);

  /// Generate deallocating mark_array after the outermost for-loop
  genMarkArrayDealloc(three_index_ancestors,
                      alloc_mark_array,
                      builder,
                      loc);
  /// Restore the insertion point
  builder.restoreInsertionPoint(last_insertion_point);
  {
    comet_vdump(three_index_ancestors[2]->symbolicForOps[0]);
    comet_pdump(cur_op->getParentOp());
  }
}

/// ----------------- ///
/// Find the output matrix (C in C = A * B) rowptr array, col array, and val array.
/// ----------------- ///
void findOutputMatrixRowptrAndColAndVal(indexTree::IndexTreeComputeOp &cur_op,
                            std::vector<Value> &wp_ops,
                            SymbolicInfo &symbolicInfo /* output */) {
//                            Value &mtxC_rowptr /* output */,
//                            Value &mtxC_col /* output */,
//                            Value &mtxC_val /* output */) {
  /// ----------------- ///
  /// cur_op
  /// %41 = "it.Compute"(%39, %40) {comp_worksp_opt = true, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
  /// cur_op.LHS
  /// %40 = "it.ComputeLHS"(%32, %33, %34, %35) {allFormats = [["D"]], allPerms = [[2]]} : (tensor<?xf64>, tensor<?xindex>, tensor<?xindex>, tensor<1xindex>) -> tensor<*xf64>
  /// We want to find cmp_op,
  /// The compute node that uses %40 as its RHS
  /// %57 = "it.Compute"(%42, %56) {comp_worksp_opt = true, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
  /// %42 = "it.ComputeRHS"(%32, %33, %34, %35) {allFormats = [["D"]], allPerms = [[2]]} : (tensor<?xf64>, tensor<?xindex>, tensor<?xindex>, tensor<1xindex>) -> tensor<*xf64>
  /// Condition:
  /// 1. %40 is current compute node (cur_op)'s LHS
  /// 2. %42 is another compute node (cmp_op)'s RHS
  /// 3. %40 and %42 have the same operands.
  /// ----------------- ///

  /// Get cur_op.LHS's all operands.
  Value cur_LHS = cur_op.getLhs();
  std::vector<Value> cur_LHS_operands;
  for (Value operand : cur_LHS.getDefiningOp()->getOperands()) {
    cur_LHS_operands.push_back(operand);
  }

  bool found_cmp_op = false;

  /// Goal: from cur_op, find cmp_op
  indexTree::IndexTreeComputeOp cmp_op;
  for (Value op : wp_ops) {
    if ((cmp_op = dyn_cast<indexTree::IndexTreeComputeOp>(op.getDefiningOp()))) {
      if (cmp_op.getRhs().size() != 1) {
        continue;
      }
      Value cmp_RHS = cmp_op.getRhs()[0];
      if (cmp_RHS.getDefiningOp()->getNumOperands() != cur_LHS_operands.size()) {
        continue;
      }
      bool all_operands_same = true;
      for (Value operand : cmp_RHS.getDefiningOp()->getOperands()) {
        if (std::find(cur_LHS_operands.begin(), cur_LHS_operands.end(), operand) == cur_LHS_operands.end()) {
          /// operand is not in cur_LHS_operands
          all_operands_same = false;
          break;
        }
      }
      /// If all operands in cmp_RHS are the same with those in cur_LHS, then the cmp_RHS is the same with cur_LHS.
      if (all_operands_same) {
        found_cmp_op = true;
        break;
      }
    }
  }
  assert(found_cmp_op && "Should find the other compute node whose RHS is the same as cur_op's LHS.\n");
  {
    comet_vdump(cur_op);
    comet_vdump(cur_LHS);
    comet_vdump(cmp_op);
    comet_vdump(cmp_op.getRhs()[0]);
  }

  /// cmp_op is %57,
  /// cmp_LHS is %56
  /// %57 = "it.Compute"(%42, %56) {comp_worksp_opt = true, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
  /// %56 = "it.ComputeLHS"(%55) {allFormats = [["D", "CU"]], allPerms = [[0, 2]]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>) -> tensor<*xf64>
  /// %55 = ta.sptensor_construct(%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %8, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  /**
  sptensor_construct(
      A1pos,  /// number of rows
      A1crd,  /// discard
      A2pos,  /// rowptr array
      A2crd,  /// col_id array
      Aval, /// data array
      A1pos_size,
      A1crd_size,
      A2pos_size,
      A2crd_size,
      Aval_size,
      dim1_size,
      dim2_size,
  )
  */
  /// Therefore, mtxC is %55, and mtxC.rowptr is %47, mtxC.col is %48, mtxC.val is %49.
  /// %56 = "it.ComputeLHS"(%55) {allFormats = [["D", "CU"]], allPerms = [[0, 2]]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>) -> tensor<*xf64>
  /// %55 = ta.sptensor_construct(%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %8, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  /// %47 = bufferization.to_tensor %alloc_100 : memref<?xindex>
  Value cmp_LHS = cmp_op.getLhs();
  Value mtxC = cmp_LHS.getDefiningOp()->getOperand(0);
  Value rowptr_buffer = mtxC.getDefiningOp()->getOperand(4);  //2
  Value rowptr_buffer = mtxC.getDefiningOp()->getOperand(4);  //2
  symbolicInfo.mtxC_rowptr = rowptr_buffer.getDefiningOp()->getOperand(0);
  Value col_buffer = mtxC.getDefiningOp()->getOperand(5);   //3
  Value col_buffer = mtxC.getDefiningOp()->getOperand(5);   //3
  symbolicInfo.mtxC_col = col_buffer.getDefiningOp()->getOperand(0);
  Value val_buffer = mtxC.getDefiningOp()->getOperand(8);   //4
  Value val_buffer = mtxC.getDefiningOp()->getOperand(8);   //4
  symbolicInfo.mtxC_val = val_buffer.getDefiningOp()->getOperand(0);
  symbolicInfo.mtxC = mtxC;
  {
    comet_vdump(cmp_LHS);
    comet_vdump(mtxC);
    comet_vdump(rowptr_buffer);
    comet_vdump(symbolicInfo.mtxC_rowptr);
    comet_vdump(col_buffer);
    comet_vdump(symbolicInfo.mtxC_col);
    comet_vdump(val_buffer);
    comet_vdump(symbolicInfo.mtxC_val);
  }
}


/// ----------------- ///
/// Generate ws_col_list_size before the 1st-level for-loop.
/// ----------------- ///
///     %ws_col_list_size = memref.alloc() : memref<1xindex>
//      memref.store %c0, %ws_col_list_size[%c0] : memref<1xindex>
void genSymbolicKernelWSColListSize(std::vector<OpsTree *> &three_index_ancestors,
                                    OpBuilder &builder,
                                    Location &loc,
                                    Value &ws_col_list_size /* output */) {
  /// Set insertion point to before the outermost for-loop
  builder.setInsertionPoint(three_index_ancestors[2]->symbolicForOps[0]);
  MemRefType memTy_alloc_1_index = MemRefType::get({1}, builder.getIndexType());
  ws_col_list_size = builder.create<memref::AllocOp>(loc, memTy_alloc_1_index);
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_0_to_size = builder.create<memref::StoreOp>(loc,
                                                         const_index_0,
                                                         ws_col_list_size,
                                                         ValueRange{const_index_0});
#else
  builder.create<memref::StoreOp>(loc,
                                  const_index_0,
                                  ws_col_list_size,
                                  ValueRange{const_index_0});
#endif
  {
    comet_vdump(ws_col_list_size);
    comet_vdump(store_0_to_size);
  }
}


/// ----------------- ///
/// Generate Symbolic Phase Kernel's if-condition to get the ws_col_list_size
/// ----------------- ///
/*
    %76 = memref.load %mark_array[%j_idx] : memref<?xindex>
    %77 = arith.cmpi ne, %76, %new_mark : index
    scf.if %77 {  // if (mark_array[j_idx] != mark)
      memref.store %new_mark, %mark_array[%j_idx] : memref<?xindex>  // mark_array[B_col_id] = new_mark
      %81 = memref.load %ws_col_list_size[%c0] : memref<1xindex>  // %81 = ws_col_list_size
      %82 = arith.addi %81, %c1 : index  // %82 = ++%81
      memref.store %82, %ws_col_list_size[%c0] : memref<1xindex>  // ws_col_list_size = %82
    }
 */
void genSymbolicKernelIfCondition(std::vector<OpsTree *> &three_index_ancestors,
//                                  Value &alloc_mark,
                                  Value &alloc_mark_array,
                                  Value &new_mark_reg,
                                  MaskingInfo &maskingInfo,
                                  Value &ws_col_list_size /* contents updated after call */,
                                  OpBuilder &builder,
                                  Location &loc) {
  {
    comet_vdump(three_index_ancestors[2]->symbolicForOps[0]);
    comet_vdump(alloc_mark_array);
  }
  /// Set the insertion point to the end of innermost for-loop's body
//  builder.setInsertionPointToEnd(three_index_ancestors[0]->symbolicForOps[0].getBody());  /// This doesn't work because it inserts even after the scf.yield, which is wrong.
  builder.setInsertionPointAfter(three_index_ancestors[0]->symbolicAccessIdx[0].getDefiningOp());

  /// This is for no-masking
  ///    %76 = memref.load %mark_array[%j_idx] : memref<?xindex>
  ///    %77 = arith.cmpi ne, %76, %new_mark : index
  ///    scf.if %77 {  // if (mark_array[j_idx] != mark)
  /// For pushed-based masking
  ///    %m_v = memref.load %mark_array[%j_idx] : memref<?xindex>
  ///    %equal_mark = arith.cmpi eq, %m_v, %new_mark_reg : index
  ///    scf.if %equal_mark {  // if (mark_array[j_idx] == mark
  Value &j_idx = three_index_ancestors[0]->symbolicAccessIdx[0];
  Value mark_value = builder.create<memref::LoadOp>(loc, alloc_mark_array, ValueRange{j_idx});
  {
    comet_vdump(j_idx);
    comet_vdump(mark_value);
    comet_vdump(three_index_ancestors[2]->symbolicForOps[0]);
  }

  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  scf::IfOp ifNotSeen;
  ///
  if (NO_MASKING == maskingInfo.mask_type) {
    Value if_not_equal = builder.create<arith::CmpIOp>(loc, CmpIPredicate::ne, mark_value, new_mark_reg);
    ifNotSeen = builder.create<scf::IfOp>(loc, if_not_equal, false /*NoElseRegion*/);
    {
      comet_vdump(if_not_equal);
    }

    /// Set Insertion Point to the if-condition's then-region.
    builder.setInsertionPointToStart(&ifNotSeen.getThenRegion().front());
    ///      memref.store %new_mark, %mark_array[%j_idx] : memref<?xindex>  // mark_array[B_col_id] = new_mark
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    auto store_new_mark = builder.create<memref::StoreOp>(loc,
                                                          new_mark_reg,
                                                          alloc_mark_array,
                                                          ValueRange{j_idx});
#else
    builder.create<memref::StoreOp>(loc,
                                    new_mark_reg,
                                    alloc_mark_array,
                                    ValueRange{j_idx});
#endif
    {
      comet_vdump(store_new_mark);
//      comet_vdump(ifNotSeen);
    }
  } else if (PUSH_BASED_MASKING == maskingInfo.mask_type) {
    Value equal_mark = builder.create<arith::CmpIOp>(loc,
                                                     CmpIPredicate::eq,
                                                     mark_value,
                                                     new_mark_reg);
    ifNotSeen = builder.create<scf::IfOp>(loc, equal_mark, false /*NoElseRegion*/);
    {
      comet_vdump(equal_mark);
    }

    /// Set Insertion Point to the if-condition's then-region.
    builder.setInsertionPointToStart(&ifNotSeen.getThenRegion().front());
    ///     %m_v_plus_one = arith.addi %m_v, %c1 : index
    ///     memref.store %m_v_plus_one, %mark_array[%j_idx] : memref<?xindex>  // mark_array[B_col_id] = new_mark
    Value mark_value_plus_one = builder.create<AddIOp>(loc, mark_value, const_index_1);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    auto store_mark_value_plus_one = builder.create<memref::StoreOp>(loc,
                                                                     mark_value_plus_one,
                                                                     alloc_mark_array,
                                                                     ValueRange{j_idx});
#else
    builder.create<memref::StoreOp>(loc,
                                    mark_value_plus_one,
                                    alloc_mark_array,
                                    ValueRange{j_idx});
#endif
    {
      comet_vdump(mark_value_plus_one);
      comet_vdump(store_mark_value_plus_one);
//      comet_vdump(ifNotSeen);
    }
  } else {
    llvm::errs() << "Error: masking type " << maskingInfo.mask_type << " is not supported, yet.\n";
  }
  ///

  ///      %81 = memref.load %ws_col_list_size[%c0] : memref<1xindex>  // %81 = ws_col_list_size
  ///      %82 = arith.addi %81, %c1 : index  // %82 = ++%81
  ///      memref.store %82, %ws_col_list_size[%c0] : memref<1xindex>  // ws_col_list_size = %82
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  Value old_size = builder.create<memref::LoadOp>(loc,
                                                  ws_col_list_size,
                                                  ValueRange{const_index_0});
  Value new_size = builder.create<AddIOp>(loc, old_size, const_index_1);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_new_size = builder.create<memref::StoreOp>(loc,
                                                         new_size,
                                                         ws_col_list_size,
                                                         ValueRange{const_index_0});
#else
  builder.create<memref::StoreOp>(loc,
                                  new_size,
                                  ws_col_list_size,
                                  ValueRange{const_index_0});
#endif
  {
    comet_vdump(store_new_size);
    comet_vdump(ifNotSeen);
  }
}


/// ----------------- ///
/// After the 2nd-level for-loop, generate assigning C_rowptr[i_idx] = ws_col_list_size .
/// Generate resetting ws_col_list_size = 0 .
/// ----------------- ///
///      %ws_size = memref.load %ws_col_list_size[%c0] : memref<1xindex> /// C.rowptr[A_row_id] = ws_sizes[tid];
///      memref.store %ws_size, %C_rowptr[%arg0] : memref<?xindex>
///      memref.store %c0, %ws_col_list_size[%c0] : memref<1xindex>
void genUpdateMtxCRowptr(std::vector<OpsTree *> &three_index_ancestors,
                         Value &ws_col_list_size,
                         Value &mtxC_rowptr /* contents updated after call */,
                         OpBuilder &builder,
                         Location &loc) {
  /// Set Insertion Point to after the 2nd-level for-loop
  builder.setInsertionPointAfter(three_index_ancestors[1]->symbolicForOps[0]);

  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  Value ws_size = builder.create<memref::LoadOp>(loc, ws_col_list_size, ValueRange{const_index_0});
  Value &i_idx = three_index_ancestors[2]->symbolicAccessIdx[0];
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto update_rowptr = builder.create<memref::StoreOp>(loc,
                                                        ws_size,
                                                        mtxC_rowptr,
                                                        ValueRange{i_idx});
  auto reset_ws_size = builder.create<memref::StoreOp>(loc,
                                                       const_index_0,
                                                       ws_col_list_size,
                                                       ValueRange{const_index_0});
#else
  builder.create<memref::StoreOp>(loc,
                                  ws_size,
                                  mtxC_rowptr,
                                  ValueRange{i_idx});
  builder.create<memref::StoreOp>(loc,
                                  const_index_0,
                                  ws_col_list_size,
                                  ValueRange{const_index_0});
#endif
  {
    comet_vdump(ws_size);
    comet_vdump(update_rowptr);
    comet_vdump(reset_ws_size);
    comet_vdump(three_index_ancestors[2]->symbolicForOps[0]);
  }
}

/// ----------------- ///
/// After the outermost for-loop, do reduce over C_rowptr to change sizes to offsets.
/*
    memref.store %c0, %C_rowptr[%num_cols] : memref<?xindex>
    %row_i_bound = arith.addi %num_cols, %c1 : index
    %C_size = memref.alloc() : memref<1xindex>
    memref.store %c0, %C_size[%c0] : memref<1xindex>
    scf.for %arg0 = %c0 to %row_i_bound step %c1 {
      %curr = memref.load %C_rowptr[%arg0] : memref<?xindex>
      %size = memref.load %C_size[%c0] : memref<1xindex>
      memref.store %size, %C_rowptr[%arg0] : memref<?xindex>
      %add_up = arith.addi %size, %curr : index
      memref.store %add_up, %C_size[%c0] : memref<1xindex>
    }
 */
void genReduceMtxCRowptr(std::vector<OpsTree *> &three_index_ancestors,
                         Value &num_rows,
                         OpBuilder &builder,
                         Location &loc,
                         Value &mtxC_rowptr /* contents updated after call */,
                         scf::ForOp &reduce_for_loop /* output */,
                         Value &mtxC_val_size /* output */) {
  /// Set Insertion Point after the outermost symbolic for-loop
  builder.setInsertionPointAfter(three_index_ancestors[2]->symbolicForOps[0]);

  ///     memref.store %c0, %C_rowptr[%num_cols] : memref<?xindex>
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass

  auto set_last_zero = builder.create<memref::StoreOp>(loc,
                                                       const_index_0,
                                                       mtxC_rowptr,
                                                       ValueRange{num_rows});
#else
  builder.create<memref::StoreOp>(loc,
                                  const_index_0,
                                  mtxC_rowptr,
                                  ValueRange{num_rows});
#endif
  {
    comet_vdump(set_last_zero);
  }

  ///    %row_i_bound = arith.addi %num_cols, %c1 : index
  ///    %C_size = memref.alloc() : memref<1xindex>
  ///    memref.store %c0, %C_size[%c0] : memref<1xindex>
  Value idx_bound = builder.create<AddIOp>(loc, num_rows, const_index_1);
  MemRefType memTy_alloc_1_index = MemRefType::get({1}, builder.getIndexType());
  Value C_size = builder.create<memref::AllocOp>(loc, memTy_alloc_1_index);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto set_C_size_zero = builder.create<memref::StoreOp>(loc,
                                                         const_index_0,
                                                         C_size,
                                                         ValueRange{const_index_0});
#else
  builder.create<memref::StoreOp>(loc,
                                  const_index_0,
                                  C_size,
                                  ValueRange{const_index_0});
#endif
  {
    comet_vdump(idx_bound);
    comet_vdump(C_size);
    comet_vdump(set_C_size_zero);
  }

  ///    scf.for %arg0 = %c0 to %row_i_bound step %c1 {
  ///      %curr = memref.load %C_rowptr[%arg0] : memref<?xindex>
  ///      %size = memref.load %C_size[%c0] : memref<1xindex>
  ///      memref.store %size, %C_rowptr[%arg0] : memref<?xindex>
  ///      %add_up = arith.addi %size, %curr : index
  ///      memref.store %add_up, %C_size[%c0] : memref<1xindex>
  ///    }
  reduce_for_loop = builder.create<scf::ForOp>(loc,
                                                    const_index_0 /* lowerBound */,
                                                    idx_bound /* upperBound */,
                                                    const_index_1 /* step */);
  builder.setInsertionPointToStart(reduce_for_loop.getBody());
  Value idx = reduce_for_loop.getInductionVar();
  Value curr = builder.create<memref::LoadOp>(loc, mtxC_rowptr, ValueRange{idx});
  Value size = builder.create<memref::LoadOp>(loc, C_size, ValueRange{const_index_0});
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_size_to_rowptr = builder.create<memref::StoreOp>(loc,
                                                              size,
                                                              mtxC_rowptr,
                                                              ValueRange{idx});
  Value add_up = builder.create<AddIOp>(loc, size, curr);
  auto store_sum_to_size = builder.create<memref::StoreOp>(loc,
                                                           add_up,
                                                           C_size,
                                                           ValueRange{const_index_0});
#else
  builder.create<memref::StoreOp>(loc,
                                  size,
                                  mtxC_rowptr,
                                  ValueRange{idx});
  Value add_up = builder.create<AddIOp>(loc, size, curr);
  builder.create<memref::StoreOp>(loc,
                                  add_up,
                                  C_size,
                                  ValueRange{const_index_0});
#endif

  /// %mtxC_val_size = memref.load %C_size[%c0] : memref<1xindex>
  builder.setInsertionPointAfter(reduce_for_loop);
  mtxC_val_size = builder.create<memref::LoadOp>(loc, C_size, ValueRange{const_index_0});
  {
    comet_vdump(curr);
    comet_vdump(size);
    comet_vdump(store_size_to_rowptr);
    comet_vdump(add_up);
    comet_vdump(store_sum_to_size);
    comet_vdump(reduce_for_loop);
    comet_vdump(mtxC_val_size);
  }
}


/// ----------------- ///
/// Deallocate the old mtxC_col array and mtxC_val array, and reallocate them with the correct size mtxC_val_size.
/// ----------------- ///
/// memref.dealloc %mtxC_col : memref<?xindex>
/// memref.dealloc %mtxC_val : memref<?xindex>
/// %mtxC_col_new = memref.alloc(%mtxC_val_size) {alignment = 8 : i64} : memref<?xindex>
/// %mtxC_val_new = memref.alloc(%mtxC_val_size) {alignment = 8 : i64} : memref<?xf64>
void reallocMtxCColAndVal(Value &mtxC_val_size,
                    OpBuilder &builder,
                    Location &loc,
                    SymbolicInfo &symbolicInfo) {
//                    Value &mtxC_col /* updated after the call */,
//                    Value &mtxC_val /* updated after the call */,
//                    Value &mtxC /* updated after the call */) {
  /// Set Insertion Point to after the mtxC_val_size, so the mtxC_val_size is ready at that point.
  builder.setInsertionPointAfter(mtxC_val_size.getDefiningOp());

  Value &mtxC_col = symbolicInfo.mtxC_col;
  Value &mtxC_val = symbolicInfo.mtxC_val;
//  Value &mtxC = symbolicInfo.mtxC;

  /// memref.dealloc %C_col : memref<?xindex>
  /// memref.dealloc %mtxC_val : memref<?xindex>
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto dealloc_mtxC_col = builder.create<memref::DeallocOp>(loc, mtxC_col);
  auto dealloc_mtxC_val = builder.create<memref::DeallocOp>(loc, mtxC_val);
  {
    comet_vdump(dealloc_mtxC_col);
    comet_vdump(dealloc_mtxC_val);
  }
#else
  builder.create<memref::DeallocOp>(loc, mtxC_col);
  builder.create<memref::DeallocOp>(loc, mtxC_val);
#endif

  /// -------------- ///
  /// Remove mtxC_col's user who is a memref.store operation
  /// This is very ad-hoc, just to avoid segmentation fault for old very large C.val array and C.col array.
  /// -------------- ///
  removeMemrefStoreUser(mtxC_col);
  removeMemrefStoreUser(mtxC_val);


  /// %C_col_new = memref.alloc(%mtxC_val_size) {alignment = 8 : i64} : memref<?xindex>
  /// %mtxC_val_new = memref.alloc(%mtxC_val_size) {alignment = 8 : i64} : memref<?xf64>
  MemRefType memTy_alloc_dynamic_index = MemRefType::get({ShapedType::kDynamic}, builder.getIndexType());
  MemRefType memTy_alloc_dynamic_f64 = MemRefType::get({ShapedType::kDynamic}, builder.getF64Type());
  Value new_mtxC_col = builder.create<memref::AllocOp>(loc,
                                             memTy_alloc_dynamic_index,
                                             ValueRange{mtxC_val_size},
                                             builder.getI64IntegerAttr(8) /* alignment bytes */);
  Value new_mtxC_val = builder.create<memref::AllocOp>(loc,
                                             memTy_alloc_dynamic_f64,
                                             ValueRange{mtxC_val_size},
                                             builder.getI64IntegerAttr(8) /* alignment bytes */);
  {
    comet_vdump(new_mtxC_col);
    comet_vdump(new_mtxC_val);
  }

  /// Set the symbolicInfo
  mtxC_col = new_mtxC_col;
  mtxC_val = new_mtxC_val;

  /// DEPRECATED below: replacing the old operands with new ones will not work, because
  ///             the sparse tensor %55 is declared before those new operands.
  /// 1. generate corresponding bufferization.to_tensor
  /// 2. Set them as the new operands for the output sparse tensor.
  /// The sparse tensor is
  /// %55 = ta.sptensor_construct(%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %8, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  /// Replace %47 and %48 with new operands.
  /**
  sptensor_construct(
      A1pos,  /// number of rows
      A1crd,  /// discard
      A2pos,  /// rowptr array
      A2crd,  /// col_id array
      Aval, /// data array
      A1pos_size,
      A1crd_size,
      A2pos_size,
      A2crd_size,
      Aval_size,
      dim1_size,
      dim2_size,
  )
  */
  /// Generate the new bufferization.to_tensor
//  {
//    comet_vdump(mtxC);
//  }
//  Value mtxC_col_buffer = builder.create<bufferization::ToTensorOp>(loc, mtxC_col);
//  Value mtxC_val_buffer = builder.create<bufferization::ToTensorOp>(loc, mtxC_val);
//  /// Record the old operands for erasing them later.
//  Value old_C_col = mtxC.getDefiningOp()->getOperand(2);
//  Value old_C_val = mtxC.getDefiningOp()->getOperand(3);
//  /// Set the new operands.
//  mtxC.getDefiningOp()->setOperand(3, mtxC_col_buffer);
//  mtxC.getDefiningOp()->setOperand(4, mtxC_val_buffer);
//  /// Erase the old operands.
//  old_C_col.getDefiningOp()->erase();
//  old_C_val.getDefiningOp()->erase();
//  {
//    comet_vdump(mtxC_col_buffer);
//    comet_vdump(mtxC_val_buffer);
//    comet_vdump(mtxC);
//  }

}


/// ----------------- ///
/// Change the old value in C_col_size (A2crd_size) and C_val_size (Aval_size) to new mtxC_val_size.
/// ----------------- ///
/// %68 = ta.sptensor_construct(%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %8, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
/// ----------------- ///
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
void genChangeOld_CColSize_And_CValSize(Value &mtxC_val_size,
                                        OpBuilder &builder,
                                        Location &loc,
                                        SymbolicInfo &symbolicInfo) {
  Value &mtxC = symbolicInfo.mtxC;
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);

  /// Find the alloc of C_col_size (Arcrd_size)
  ///     %66 = memref.load %alloc_153[%c0_128] : memref<1xindex>
  Value C_col_size_alloc = mtxC.getDefiningOp()->getOperand(14).getDefiningOp()->getOperand(0);    //8
  Value C_col_size_alloc = mtxC.getDefiningOp()->getOperand(14).getDefiningOp()->getOperand(0);    //8
  /// Store the new mtxC_val_size to C_col_size
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_C_col_size_alloc = builder.create<memref::StoreOp>(loc,
                                                                mtxC_val_size,
                                                                C_col_size_alloc,
                                                                ValueRange{const_index_0});
  comet_vdump(C_col_size_alloc);
  comet_vdump(store_C_col_size_alloc);
#else
  builder.create<memref::StoreOp>(loc,
                                  mtxC_val_size,
                                  C_col_size_alloc,
                                  ValueRange{const_index_0});
#endif


  /// Find the alloc of C_val_size (Aval_size)
  ///     %67 = memref.load %alloc_154[%c0_128] : memref<1xindex>
  Value C_val_size_alloc = mtxC.getDefiningOp()->getOperand(17).getDefiningOp()->getOperand(0);  //9
  Value C_val_size_alloc = mtxC.getDefiningOp()->getOperand(17).getDefiningOp()->getOperand(0);  //9
  /// Store the new mtxC_val_size to C_val_size
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_C_val_size_alloc = builder.create<memref::StoreOp>(loc,
                                                                mtxC_val_size,
                                                                C_val_size_alloc,
                                                                ValueRange{const_index_0});
  comet_vdump(C_val_size_alloc);
  comet_vdump(store_C_val_size_alloc);
#else
  builder.create<memref::StoreOp>(loc,
                                  mtxC_val_size,
                                  C_val_size_alloc,
                                  ValueRange{const_index_0});
#endif


}


/// ----------------- ///
/// Replace the old C_col array (A2crd) and C_val array (Aval) to new mtxC_col and mtxC_val, respectively.
/// ----------------- ///
/// %68 = ta.sptensor_construct(%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %8, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
/// ----------------- ///
/**
sptensor_construct(
    A1pos,  /// number of rows
    A1crd,  /// discard
    A2pos,  /// rowptr array
    A2crd,  /// col_id array
    Aval, /// data array
    A1pos_size,
    A1crd_size,
    A2pos_size,
    A2crd_size,
    Aval_size,
    dim1_size,
    dim2_size,
)
*/
void replaceOld_CCol_And_CVal(SymbolicInfo &symbolicInfo) {
  Value &mtxC = symbolicInfo.mtxC;

  /// Find the alloc of old_C_col (A2crd)
  ///     %61 = bufferization.to_tensor %alloc_142 : memref<?xindex>
  Value old_C_col = mtxC.getDefiningOp()->getOperand(5).getDefiningOp()->getOperand(0);   // 3
  Value old_C_col = mtxC.getDefiningOp()->getOperand(5).getDefiningOp()->getOperand(0);   // 3
  /// Replace old_C_col with the new mtxC_col
  replaceOldValueToNewValue(old_C_col, symbolicInfo.mtxC_col);

  /// Find the allod of old_C_val (Aval)
  ///     %62 = bufferization.to_tensor %alloc_146 : memref<?xf64>
  Value old_C_val = mtxC.getDefiningOp()->getOperand(8).getDefiningOp()->getOperand(0);   // 4
  Value old_C_val = mtxC.getDefiningOp()->getOperand(8).getDefiningOp()->getOperand(0);   // 4
  /// Replace old_C_val with the new mtxC_val
  replaceOldValueToNewValue(old_C_val, symbolicInfo.mtxC_val);

}



/// ----------------- ///
/// Generate Symbolic Phase kernel inside the for-loops
/// ----------------- ///
void genSymbolicKernel(std::vector<OpsTree *> &three_index_ancestors,
                       Value &alloc_mark,
                       Value &alloc_mark_array,
                       Value &new_mark_reg,
                       Value &num_rows,
                       MaskingInfo &maskingInfo,
                       OpBuilder &builder,
                       Location &loc,
                       SymbolicInfo &symbolicInfo /* contents updated after call */) {
//                       Value &mtxC_rowptr /* contents updated after call */,
//                       Value &mtxC_col /* contents updated after call */,
//                       Value &mtxC_val /* contents updated after call */) {
  /// Store the insertion point
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Generate ws_col_list_size before the 1st-level for-loop.
  Value ws_col_list_size;
  genSymbolicKernelWSColListSize(three_index_ancestors,
                                 builder,
                                 loc,
                                 ws_col_list_size /* output */);

  /// Generate the if-condidtion kernel at the end of 3rd-level for-loop.
  genSymbolicKernelIfCondition(three_index_ancestors,
                               alloc_mark_array,
                               new_mark_reg,
                               maskingInfo,
                               ws_col_list_size /* contents updated after call */,
                               builder,
                               loc);

  /// Generate updating C_rowptr[i_idx] = ws_col_list_size,
  /// and generate resetting ws_col_list_size = 0 .
  genUpdateMtxCRowptr(three_index_ancestors,
                      ws_col_list_size,
                      symbolicInfo.mtxC_rowptr /* contents updated after call */,
                      builder,
                      loc);

  /// Generate reducing C_rowptr to get offsets from sizes.
  scf::ForOp reduce_for_loop;
  Value mtxC_val_size;
  genReduceMtxCRowptr(three_index_ancestors,
                      num_rows,
                      builder,
                      loc,
                      symbolicInfo.mtxC_rowptr /* contents updated after call */,
                      reduce_for_loop /* output */,
                      mtxC_val_size /* output */);

  /// Generate deallocating old C_col and C_val because they are too large.
  /// Generate allocating the new C_col and C_val using the correct size.
  /// Update the operands of tensor C to the new C_col and C_val.
  reallocMtxCColAndVal(mtxC_val_size,
                       builder,
                       loc,
                       symbolicInfo /* updated after the call */);
//                       symbolicInfo.mtxC_col /* updated after the call */,
//                       symbolicInfo.mtxC_val /* updated after the call */,
//                       symbolicInfo.mtxC /* updated after the call */);


  /// Change the value in old Aval_size and A2crd_size to mtxC_val_size
  genChangeOld_CColSize_And_CValSize(mtxC_val_size,
                                     builder,
                                     loc,
                                     symbolicInfo);

  /// Replace the old C_val (Aval) to new mtxC_val, and old C_col (A2crd) to new mtxC_col
  /// Currently, C_col don't have places needs to replacement. This is for safety in future.
  replaceOld_CCol_And_CVal(symbolicInfo);

  /// Restore insertion point
  builder.restoreInsertionPoint(last_insertion_point);
}


/// ----------------- ///
/// Generate the symbolic phase for SpGEMM
/// ----------------- ///
/// cur_op is the compute node
/// %41 = "it.Compute"(%39, %40) {comp_worksp_opt = true, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
void genSymbolicPhase(indexTree::IndexTreeComputeOp &cur_op,
                      OpsTree *opstree,
                      std::vector<Value> &wp_ops,
                      MaskingInfo &maskingInfo,
                      OpBuilder &builder,
                      Location &loc,
                      SymbolicInfo &symbolicInfo /* updated after call */) {
  assert(symbolicInfo.is_SpGEMM && "genSymbolicPhase is only for SpGEMM so far.");

  /// Generate the for-loops structure.
//  std::vector<scf::ForOp> symbolicForLoops;
  std::vector<OpsTree *> three_index_ancestors; /* 3 index node ancestors of the current comput node
                                                   three_index_ancestors[0] is the nearest ancestor, and
                                                   three_index_ancestors[2] is the farthest one. */
//  Value num_rows_alloc; /* the memory alloc of num_rows, needed by mark_array */
  Value num_rows;   /// number of rows of output C = A * B, which is the number of rows of A.
  genSymbolicForLoops(cur_op,
                      opstree,
                      builder,
                      loc,
                      three_index_ancestors /* output */,
                      num_rows /* output */);
//                      num_rows_alloc /* output */);

  /// Generate the variable mark and the mark_array.
  /// Generate updating mark in every iteration.
  Value alloc_mark;
  Value alloc_mark_array;
  Value new_mark_reg;
  Value num_cols;   /// number of columns of output C = A * B, which is the number of columns of B.

  genMarkAndMarkArray(three_index_ancestors,
                      cur_op,
                      builder,
                      loc,
                      alloc_mark /* output */,
                      alloc_mark_array /* output */,
                      new_mark_reg /* output */,
                      num_cols /* output */);

  symbolicInfo.mtxC_num_rows = num_rows;
  symbolicInfo.mtxC_num_cols = num_cols;

  /// Generate the for-loop that initializes mark_array by using the mask.
  if (PUSH_BASED_MASKING == maskingInfo.mask_type) {
    genSymbolicInitMarkArrayByMask(three_index_ancestors,
                           new_mark_reg,
                           alloc_mark_array,
                           maskingInfo,
                           builder,
                           loc);
  }


  /// Find the output matrix C's rowptr (mtxC_rowptr). It will be updated in the symbolic phase.
//  Value mtxC_rowptr;
//  Value mtxC_col;
//  Value mtxC_val;
  findOutputMatrixRowptrAndColAndVal(cur_op,
                                     wp_ops,
                                     symbolicInfo /* output */);
//                                     mtxC_rowptr /* output */,
//                                     mtxC_col /* output */,
//                                     mtxC_val /* output */);

  /// Generate Symbolic Phase Kernel to update mtxC_rowptr
  genSymbolicKernel(three_index_ancestors,
                    alloc_mark,
                    alloc_mark_array,
                    new_mark_reg,
                    num_rows,
                    maskingInfo,
                    builder,
                    loc,
                    symbolicInfo /* contents updated after call */);
//                    mtxC_rowptr /* contents updated after call */,
//                    mtxC_col /* contents updated after call */,
//                    mtxC_val /* contents updated after call */);

//  /// Record the output of Symbolic Phase
//  symbolicInfo.mtxC_rowptr = mtxC_rowptr;
//  symbolicInfo.mtxC_col = mtxC_col;
//  symbolicInfo.mtxC_val = mtxC_val;
}

/// ----------------- ///
/// Insert a row_offset before the outermost numeric for-loop.
/// insert row_offset = C_rowptr[i_idx] at the beginning of the body of the outermost numeric for-loop.
/// Store row_offset in SymbolicInfo
/// ----------------- ///
void insertRowOffsetFromMatrixCRowptr(std::vector<scf::ForOp> &nested_forops,
                                      SymbolicInfo &symbolicInfo,
                                      OpBuilder &builder,
                                      Location &loc) {
  /// Store the insertion point
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Set Insertion Point to be before the outermost numeric for-loop
  /// nested_forops[0] is the innermost for-loop, and nested_forops[2] is the outermost one.
  ///     %row_offset = memref.alloc() : memref<1xindex>
  builder.setInsertionPoint(nested_forops[2]);
  MemRefType memTy_alloc_1_index = MemRefType::get({1}, builder.getIndexType());
  Value row_offset = builder.create<memref::AllocOp>(loc, memTy_alloc_1_index);

  /// Set Insertion Point at the beginning of the body of the outermost numeric for-loop.
  builder.setInsertionPointToStart(nested_forops[2].getBody());
  ///      %rowptr_start = memref.load %C_rowptr[%i_idx] : memref<?xindex>
  ///      memref.store %rowptr_start, %rowptr[%c0] : memref<1xindex>
  Value i_idx = nested_forops[2].getInductionVar();
  Value &mtxC_rowptr = symbolicInfo.mtxC_rowptr;
  Value rowptr = builder.create<memref::LoadOp>(loc, mtxC_rowptr, ValueRange{i_idx});
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto init_row_offset = builder.create<memref::StoreOp>(loc,
                                                       rowptr,
                                                       row_offset,
                                                       ValueRange{const_index_0});
#else
  builder.create<memref::StoreOp>(loc,
                                  rowptr,
                                  row_offset,
                                  ValueRange{const_index_0});
#endif
  {
    comet_vdump(row_offset);
    comet_vdump(rowptr);
    comet_vdump(init_row_offset);
    comet_vdump(nested_forops[2]);
  }
  symbolicInfo.row_offset = row_offset;

  /// Restore the insertion point
  builder.restoreInsertionPoint(last_insertion_point);
}


unsigned int findIndexInVector_OpsTree(std::vector<OpsTree *> vec, OpsTree *e) {
  // Check if element e exists in vector
  auto it = std::find(vec.begin(), vec.end(), e);

  // It accepts a range and an element to search in the given range. If element is found then it returns an iterator to the first element in the given range that’s equal to given element, else it returns an end of the list.
  unsigned int ret = 0;
  if (it != vec.end()) {
    // Get index of element from iterator
    ret = std::distance(vec.begin(), it);
  } else {
    ret = vec.size();
  }
  return ret;
}

/// ----------------- ///
/// Deprecated: never used
/// Removed by Zhen Peng on 8/12/2023
/// ----------------- ///
//struct indexInTensor {
//  Value tensor;
//  unsigned int id;
//  std::string format;
//};
/// ----------------- ///
/// End Deprecated
/// ----------------- ///


Value findCorrespondingAlloc(Value &iOp) {
  comet_debug() << "findCorrespondingAlloc for loop upper bound\n";
  comet_vdump(iOp);
  auto init_alloc = iOp.getDefiningOp()->getOperand(0);
  comet_vdump(init_alloc);

  while (true) {
    if (isa<memref::AllocOp>(init_alloc.getDefiningOp())) {
      if (init_alloc.getType().dyn_cast<MemRefType>().getDimSize(0) != ShapedType::kDynamic) {
        return init_alloc;
      }
    }
    if (init_alloc.getDefiningOp()->getNumOperands() > 0) {
      init_alloc = init_alloc.getDefiningOp()->getOperand(0);
    } else {
      // Alloc related to another sparse tensor construct such as coming from sparse transpose
      comet_debug() << "Return alloc op - comes from sptensor_construct\n";
      comet_vdump(init_alloc);
      return init_alloc;
    }
  }
}

/// Get allocs for a tensor (sparse or dense)
std::vector<Value> getAllocs(Value &tensor) {
  std::vector<Value> allocs;
  if (tensor.getType().isa<mlir::TensorType>()) { // Dense tensor
    comet_debug() << " getAllocs() -  it is dense\n";
    if (isa<ToTensorOp>(tensor.getDefiningOp())) {
      Operation *tensorload = cast<ToTensorOp>(tensor.getDefiningOp());
      auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
      comet_vdump(alloc_op);
      allocs.push_back(alloc_op);
    } else {
      for (unsigned int i = 0; i < tensor.getDefiningOp()->getNumOperands(); i++) {
        if (isa<ToTensorOp>(tensor.getDefiningOp()->getOperand(i).getDefiningOp())) {
          Operation *tensorload = cast<ToTensorOp>(tensor.getDefiningOp()->getOperand(i).getDefiningOp());
          auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
          comet_vdump(alloc_op);
          allocs.push_back(alloc_op);
        }
      }
    }
  } else if (tensor.getType().isa<tensorAlgebra::SparseTensorType>()) { // nSparse tensor
    comet_debug() << " getAllocs() -  it is sparse\n";
    auto defop = tensor.getDefiningOp<tensorAlgebra::SparseTensorConstructOp>();

    for (unsigned int n = 0; n < defop.getTotalDimArrayCount(); n++) {
      comet_vdump(defop.getIndices()[n]);
      Operation *tensorload = defop.getIndices()[n].getDefiningOp<ToTensorOp>();
      auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
      allocs.push_back(alloc_op);
      comet_vdump(alloc_op);
    }
  } else if (dyn_cast<ConstantOp>(tensor.getDefiningOp())) { // ConstantOp
    allocs.push_back(tensor);
  }
  return allocs;
}

std::vector<std::vector<Value>> getAllAllocs(std::vector<Value> &tensors) {
  std::vector<std::vector<Value>> allAllocs(tensors.size());
  for (unsigned int i = 0; i < tensors.size(); i++) {
    allAllocs[i] = getAllocs(tensors[i]);
  }
  return allAllocs;
}

/// while until parent == null
void getAncestorsOps(OpsTree *opstree, std::vector<OpsTree *> &ret) {

  while (opstree->parent != nullptr) {
    ret.push_back(opstree->parent);
    opstree = opstree->parent;
  }
}

/// Generate scf.for op for indices
/// The index is the "idx"th index of "tensor"
void genForOps(std::vector<Value> &tensors,
               std::vector<unsigned int> &ids,
               std::vector<std::string> &formats,
               indexTree::IndexTreeOp rootOp,
//               PatternRewriter &rewriter,
               OpBuilder &builder,
               OpsTree *opstree) {
  comet_debug() << " genForOps indexTreeOp\n";
  comet_vdump(rootOp);
  Location loc = rootOp.getLoc();
  /// The insertion location should be "the end of the body of parent loop"
  std::vector<OpsTree *> ancestorsOps;
  getAncestorsOps(opstree, ancestorsOps);
  comet_debug() << " genForOps ancestorsOps.size(): " << ancestorsOps.size() << "\n";
  for (unsigned int i = 0; i < ancestorsOps.size(); i++) {
    comet_debug() << " ancestorsOps[" << i << "]->forOps.size(): " << ancestorsOps[i]->forOps.size()
                  << ", ancestorsOps->id: "
                  << ancestorsOps[i]->id << "\n";
  }

  comet_debug() << "\n";
  /// If parent is for loop, insert into the body, How to get end of body?
  if (ancestorsOps.size() > 0) {
    // ancestorsOps[0] stores the closest parent
    scf::ForOp parent_forop = nullptr;
    comet_debug() << "\n";
    std::vector<scf::ForOp> parent_forops = ancestorsOps[0]->forOps;
    comet_debug() << " parent_forops.size(): " << parent_forops.size() << " \n";

    parent_forop = parent_forops[parent_forops.size() - 1];

    comet_debug() << " reset the insertion point\n";
    comet_vdump(parent_forop);

    unsigned int order = findIndexInVector_OpsTree(ancestorsOps[0]->getChildren(), opstree);
    comet_debug() << " order: " << order << "\n";
    if (order == ancestorsOps[0]->getChildren().size()) {
      llvm::errs() << __LINE__ << "Not belong to parent's children\n";
    } else {
      // Get the children of the parent_forop
      comet_debug() << " number of children: " << parent_forops.size() << "\n";
      if (order == 0) {
        // builder.setInsertionPointToStart(parent_forop.getBody());
        comet_debug() << "Insertion point order == 0\n";
        builder.setInsertionPoint(parent_forop.getBody()->getTerminator());
      } else {
        comet_debug() << "\n";
        std::vector<scf::ForOp> brother_forops = ancestorsOps[0]->getChildren()[order - 1]->forOps;
        if (brother_forops.size() > 0) {
          comet_debug() << " brother_forops.size(): " << brother_forops.size() << "\n";
          if (opstree->forOps.size() == 0) {
            comet_debug() << "\n";
            comet_vdump(brother_forops[0]);
            comet_debug() << "Insertion point (brother_forops.size() > 0 &&  opstree->forOps.size() == 0)\n";
            builder.setInsertionPointAfter(brother_forops[0]);
          } else { // current opstree contains loops, insert in the body of the loops
            comet_debug() << " -------- current opstree contain loops --- impossible\n";
            comet_debug() << "Insertion point (brother_forops.size() > 0 &&  opstree->forOps.size() != 0)\n";
            builder.setInsertionPoint(opstree->forOps[opstree->forOps.size() - 1].getBody()->getTerminator());
          }
        }
      }
    }
    comet_debug() << " reset the insertion point\n";
  }
  comet_debug() << "\n";

  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);

  comet_debug() << "Tensor size: " << tensors.size() << "\n";
  std::vector<std::vector<Value>> allAllocs = getAllAllocs(tensors);

  comet_debug() << "Tensors:\n";
  for (unsigned int i = 0; i < tensors.size(); i++) {
    comet_vdump(tensors[i]);
  }
  // Dense, get dimension size --> loop upperbound
  // If the index is in rhs1, get it from rhs1; Otherwise, get it from rhs2

  Value upperBound, lowerBound;
  for (unsigned int i = 0; i < tensors.size(); i++) {
    if (i > 0) {
      // insertion point: the body of the previous i's loop body
      comet_debug() << " -------- current opstree contain loops\n";
      builder.setInsertionPoint(opstree->forOps[opstree->forOps.size() - 1].getBody()->getTerminator());
    }

    Value tensor = tensors[i];
    std::string format = formats[i];
    unsigned int id = ids[i];

    comet_debug() << " current index format: " << format << "\n";
    if (format.compare(0, 1, "D") == 0) {
      // Check which tensor is sparse, which is dense;
      // Since this function only handles mixed sparse/dense, then "D" only occurs in one tensor
      // Both the dense and sparse tensor contain the dim size; But they are different. Use one.
      int64_t maxSize = 0;
      comet_debug() << " ";
      comet_vdump(tensor);
      if (tensor.getType().isa<mlir::RankedTensorType>()) { // Dense tensor

        auto tensorTy = tensor.getType().cast<mlir::TensorType>();
        maxSize = tensorTy.getDimSize(id);

        // Check if dynamic size
        // Check upperBoundsize
        if (maxSize == ShapedType::kDynamic) {
          // Find defOp allocOp, check the parameter
          comet_debug() << " Dynamic size ";
          comet_pdump(tensor.getDefiningOp());                // tensor_load
          comet_vdump(tensor.getDefiningOp()->getOperand(0)); // alloc <?x32xf64>
          // Check the order of the current dynamic size
          auto rhs1_alloc = tensor.getDefiningOp()->getOperand(0);
          std::vector<unsigned int> dyn_dims_vec;
          for (unsigned i = 0; i < tensorTy.getRank(); i++) {
            if (tensorTy.isDynamicDim(i)) {
              dyn_dims_vec.push_back(i);
            }
          } // ? x ? x 20 x ?
          auto rhs1_loc_dyn = findIndexInVector<unsigned int>(dyn_dims_vec, id);
          comet_vdump(rhs1_alloc.getDefiningOp()->getOperand(rhs1_loc_dyn));

          upperBound = rhs1_alloc.getDefiningOp()->getOperand(rhs1_loc_dyn);
        } else {
          upperBound = builder.create<ConstantIndexOp>(loc, maxSize);
        }

        lowerBound = builder.create<ConstantIndexOp>(loc, 0);
        auto step = builder.create<ConstantIndexOp>(loc, 1);
        auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        comet_debug() << " D Loop\n";
        comet_vdump(loop);

        opstree->forOps.push_back(loop);
        opstree->accessIdx.push_back(loop.getInductionVar());
      } else if (tensor.getType().isa<mlir::UnrankedTensorType>()) {
        comet_debug() << " \n";
        comet_pdump(tensor.getDefiningOp());
        if (indexTree::IndexTreeComputeRHSOp rhsop = dyn_cast<indexTree::IndexTreeComputeRHSOp>(
          tensor.getDefiningOp())) {
          comet_debug() << " \n";
        }
      } else if (tensor.getType().cast<tensorAlgebra::SparseTensorType>()) {
        comet_debug() << "cur_idx is in tensor " << i << "\n";

        lowerBound = builder.create<ConstantIndexOp>(loc, 0);
        auto index_0 = builder.create<ConstantIndexOp>(loc, 0);
        std::vector<Value> upper_indices = {index_0};
        upperBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices);

        auto step = builder.create<ConstantIndexOp>(loc, 1);
        auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        comet_debug() << " D Loop\n";
        comet_vdump(loop);

        opstree->forOps.push_back(loop);
        opstree->accessIdx.push_back(loop.getInductionVar());
      }
      // }
    }
      // mix sparse dense tensor contraction, only one sparse tensor
    else if (format.compare(0, 2, "CU") == 0) {
      // Generate for(int m = pos[0]; m < pos[1]; m++){int i = crd[m];}
      // if i = 0, index is [0,1]
      // if parent loop and child loop is accessing the same sparse tensor (CSF), index is [m, m+1], m is the nearest loop induction variable
      // Otherwise, the m comes from load operation of the input sparse tensor such as
      // j = crd[i];
      // for (int m = pos[j]; m < pos[j+1]; m++)

      comet_debug() << " format is CU id: " << id << "\n";
      comet_debug() << " Tensor: \n";
      comet_vdump(tensor);
      Value index_lower;
      Value index_upper;
      if (tensor.getType().cast<tensorAlgebra::SparseTensorType>()) {
        comet_debug() << " Tensor type is sparse\n";
        // cur_idx is in ith input tensor, which is sparse
        if (id == 0) { // The first index in the tensor
          index_lower = builder.create<ConstantIndexOp>(loc, 0);
          comet_debug() << " index_lower is set here, id == 0 " << opstree->forOps.size() << "\n";
          comet_vdump(index_lower);
        } else {
          if (opstree->parent != nullptr) {
            comet_debug() << " opstree->parent is not NULL\n";
            // Parent loop is
            scf::ForOp parent_forop = opstree->parent->forOps[opstree->parent->forOps.size() - 1];
            comet_debug() << " parent forop\n";
            comet_vdump(parent_forop);
            auto parent_UpperBound = parent_forop.getUpperBound();
            comet_debug() << " parent upperBound:\n";
            comet_vdump(parent_UpperBound);

            //  check if parent's and child's upper bounds come from the same sparse tensor
            auto alloc_parent_bounds = findCorrespondingAlloc(parent_UpperBound);
            comet_debug() << " parent upperBound alloc\n";
            comet_vdump(alloc_parent_bounds);

            comet_debug() << " child upperBound:\n";
            comet_vdump(allAllocs[i][4 * id]);
            auto alloc_child_bounds = findCorrespondingAlloc(allAllocs[i][4 * id]);
            comet_debug() << " child upperBound alloc\n";
            comet_vdump(alloc_child_bounds);

            if (alloc_child_bounds == alloc_parent_bounds) // m is the nearest loop induction variable
            {
              comet_debug() << " THESAME: Parent and Child has the same alloc\n";
              index_lower = parent_forop.getInductionVar();
            } else { // m comes from the load
              comet_debug() << " DIFFERENT:Parent and Child has the different alloc\n";
              comet_vdump(alloc_parent_bounds);
              comet_vdump(alloc_child_bounds);
              index_lower = opstree->parent->accessIdx[opstree->parent->forOps.size() - 1];
            }
          } else
            assert(false && "Unexpected condition\n");
        }

        comet_debug() << " index_lower:";
        comet_vdump(index_lower);
        comet_vdump(const_index_1);
        index_upper = builder.create<AddIOp>(loc, index_lower, const_index_1);
        comet_debug() << " AddIOps (index_upper):";
        comet_vdump(index_upper);

        std::vector<Value> lower_indices = {index_lower};
        lowerBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], lower_indices);
        lowerBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], lower_indices);

        std::vector<Value> upper_indices = {index_upper};
        upperBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices);
        upperBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices);
        auto step = builder.create<ConstantIndexOp>(loc, 1);
        auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        comet_debug() << " CU Loop\n";
        comet_vdump(loop);

        builder.setInsertionPoint(loop.getBody()->getTerminator());

        std::vector<Value> crd_indices = {loop.getInductionVar()};
        auto get_index = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);

        comet_debug() << "CU loop generated\n";
        comet_vdump(loop);
        opstree->forOps.push_back(loop);
        opstree->accessIdx.push_back(get_index);
      }
    } else if (format.compare(0, 2, "CN") == 0) {
      // Generate for(int m = pos[0]; m < pos[1]; m++){int i = crd[m];}
      if (tensor.getType().cast<tensorAlgebra::SparseTensorType>()) {
        auto index_0 = builder.create<ConstantIndexOp>(loc, 0);
        std::vector<Value> lower_indices = {index_0};
        lowerBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], lower_indices);
        lowerBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], lower_indices);

        auto index_1 = builder.create<ConstantIndexOp>(loc, 1);
        std::vector<Value> upper_indices = {index_1};
        upperBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices);
        auto step = builder.create<ConstantIndexOp>(loc, 1);
        auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        comet_debug() << " CN Loop\n";
        comet_vdump(loop);

        builder.setInsertionPoint(loop.getBody()->getTerminator());

        std::vector<Value> crd_indices = {loop.getInductionVar()};
        auto get_index = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);
        auto get_index = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);

        opstree->forOps.push_back(loop);
        opstree->accessIdx.push_back(get_index);
      }
      // DVSpM_CSR .. here .. why there is tensorIsSparse[i] == true?
    } else if (format.compare(0, 1, "S") == 0) {
      // Currently supported formats, Singleton is not the format of first dimension
      // and it doesn't produce a loop
      // Generate: int j = A2crd[m];

      if (tensor.getType().cast<tensorAlgebra::SparseTensorType>()) {
        comet_debug() << "cur_idx is in tensor " << i << "\n";
        // Accesing the last level loop info
        scf::ForOp last_forop;
        if (opstree->forOps.size() > 0) { // current node contain at least 1 level loop
          last_forop = opstree->forOps[opstree->forOps.size() - 1];
        } else {
          if (opstree->parent != nullptr)
            last_forop = opstree->parent->forOps[opstree->parent->forOps.size() - 1];
        }

        std::vector<Value> crd_indices = {last_forop.getInductionVar()};
        auto get_index = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);
        auto get_index = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);

        /// Adding one iteration loop to provide consistency with the corresponding index tree.
        /// Index tree includes an index node for the dimension but "S" format for this dimension
        /// doesn't produce a loop.
        lowerBound = builder.create<ConstantIndexOp>(loc, 0);
        upperBound = builder.create<ConstantIndexOp>(loc, 1);
        auto step = builder.create<ConstantIndexOp>(loc, 1);
        auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);
        comet_debug() << " S Loop\n";
        comet_vdump(loop);
        opstree->forOps.push_back(loop);

        opstree->accessIdx.push_back(get_index);
      } else {
        llvm::errs() << "Not supported tensor type\n";
      }
    } else {
      llvm::errs() << "Not supported format: " << format << "\n";
    }

    comet_debug() << " UpperBound: (i) " << i << " ";
    comet_vdump(upperBound);

    comet_debug() << "finish generate loops for current index format: " << format << "\n";
  }
}

//Value getSemiringSecondVal(PatternRewriter &rewriter, Location loc,
Value getSemiringSecondVal(OpBuilder &builder, Location &loc,
                           llvm::StringRef &semiringSecond, Value &Input0, Value &Input1,
                           bool compressedWorkspace) {

  Value elementWiseResult;
  if (semiringSecond == "times") {
    elementWiseResult = builder.create<MulFOp>(loc, Input0, Input1);
  } else if (semiringSecond == "first") {
    elementWiseResult = Input0;
  } else if (semiringSecond == "second") {
    elementWiseResult = Input1;
  } else if (semiringSecond == "atan2") {
    elementWiseResult = builder.create<math::Atan2Op>(loc, Input0, Input1);
  } else if (semiringSecond == "div") {
    elementWiseResult = builder.create<DivFOp>(loc, Input0, Input1);
  } else if (semiringSecond == "eq") {
    elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OEQ, Input0, Input1);
  } else if (semiringSecond == "ge") {
    elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OGE, Input0, Input1);
  } else if (semiringSecond == "gt") {
    elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
  } else if (semiringSecond == "le") {
    elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OLE, Input0, Input1);
  } else if (semiringSecond == "lt") {
    elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
  } else if (semiringSecond == "land") {
    // land requires integer type input
    llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                 << "land"
                 << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    exit(1);
  } else if (semiringSecond == "lor") {
    // lor requires integer type input
    llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                 << "lor"
                 << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    exit(1);
  } else if (semiringSecond == "lxor") {
    // lxor requires integer type input
    llvm::errs() << "Not supported semiring operator: "
                 << "lxor"
                 << "\n";
  } else if (semiringSecond == "minxy") {
    Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
    elementWiseResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
  } else if (semiringSecond == "max") {
    Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
    elementWiseResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
  } else if (semiringSecond == "ne") {
    elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, Input0, Input1);
  } else if (semiringSecond == "minus") {
    elementWiseResult = builder.create<SubFOp>(loc, Input0, Input1);
  } else if (semiringSecond == "plusxy") {
    elementWiseResult = builder.create<AddFOp>(loc, Input0, Input1);
  } else if (semiringSecond == "pairxy") {
    elementWiseResult = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(1));
  } else if (semiringSecond == "pow") {
    elementWiseResult = builder.create<math::PowFOp>(loc, Input0, Input1);
  } else {
    llvm::errs() << "Not supported semiring operator: " << semiringSecond << "\n";
    assert(false);
    // we should not proceed forward from this point to avoid faulty behavior.
  }

  return elementWiseResult;
}

//Value getSemiringFirstVal(PatternRewriter &rewriter, Location loc,
Value getSemiringFirstVal(OpBuilder &builder, Location &loc,
                          llvm::StringRef &semiringFirst, Value &Input0, Value &Input1,
                          bool compressedWorkspace) {

  Value reduceResult;
  if (semiringFirst == "times") {
    reduceResult = builder.create<MulFOp>(loc, Input0, Input1);
  } else if (semiringFirst == "plusxy") {
    reduceResult = builder.create<AddFOp>(loc, Input0, Input1);
  } else if (semiringFirst == "minxy") {
    if (!compressedWorkspace) {
      llvm::errs() << "Not supported semiring operator "
                      "(please use compressed workspace optimization or opt-comp-workspace "
                      "where this operation is known to work): "
                   << "min"
                   << "\n";
      // we should not proceed forward from this point to avoid in-correct results from generated code.
      assert(false);
    }
    Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
    reduceResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
  } else if (semiringFirst == "max") {
    if (!compressedWorkspace) {
      llvm::errs() << "Not supported semiring operator "
                      "(please use compressed workspace optimization or opt-comp-workspace "
                      "where this operation is known to work): "
                   << "max"
                   << "\n";
      // we should not proceed forward from this point to avoid in-correct results from generated code.
      assert(false);
    }
    Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
    reduceResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
  } else if (semiringFirst == "land") {
    // land requires integer type input
    llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                 << "land"
                 << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    assert(false);
  } else if (semiringFirst == "lor") {
    // lor requires integer type input
    llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                 << "lor"
                 << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    assert(false);
  } else if (semiringFirst == "any") {
    reduceResult = Input1;
  } else if (semiringFirst == "noop") {
    reduceResult = Input1;
  } else {
    llvm::errs() << "Not supported semiring operator: " << semiringFirst << "\n";
    // we should not proceed forward from this point to avoid faulty behavior.
    assert(false);
  }

  return reduceResult;
}

/// ----------------- ///
/// Allocate the ws_bitmap and mask_array before the outermost for-loop for the numeric phase.
/// ----------------- ///
///    %ws_bitmap = memref.alloc(%18) {alignment = 8 : i64} : memref<?xi1>
///    scf.for %arg0 = %c0 to %18 step %c1 {
///      memref.store %false, %ws_bitmap[%arg0] : memref<?xi1>
///    }
///    %array_mask = memref.alloc(%18) {alignment = 8 : i64} : memref<?xi1>
///    scf.for %arg0 = %c0 to %18 step %c1 {
///      memref.store %false, %array_mask[%arg0] : memref<?xi1>
///    }
void genAllocBitmapAndMaskArray(OpBuilder &builder,
                                Location &loc,
                                std::vector<scf::ForOp> &for_loops /*for-loop statements, from innermost to outermost*/,
                                SymbolicInfo &symbolicInfo,
                                MaskingInfo &maskingInfo,
                                NumericInfo &numericInfo /* contents updated after call */) {
//                                NumericAuxiliary &numericAuxiliary /* output */) {
  /// Save the old Insertion Point
  auto previous_loc = builder.saveInsertionPoint();

  /// Jump Insertion Point to the front of the outermost for-loop
  builder.setInsertionPoint(for_loops.back());

  /// Allocate the ws_bitmap
  Value &num_cols = symbolicInfo.mtxC_num_cols;
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  MemRefType memTy_alloc_bitarray = MemRefType::get({ShapedType::kDynamic}, builder.getI1Type());
  numericInfo.ws_bitmap = builder.create<memref::AllocOp>(loc,
                                                     memTy_alloc_bitarray,
                                                     ValueRange{num_cols},
                                                     builder.getI64IntegerAttr(8) /* alignment bytes */);
  /// Initialize ws_bitmap to zeros.
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  auto bitmap_init_loop = builder.create<scf::ForOp>(loc,
                                                         const_index_0 /* lowerBound */,
                                                         num_cols /* upperBound */,
                                                         const_index_1 /* step */);
  auto before_for_loop_body_loc = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(bitmap_init_loop.getBody());
  Value i_idx = bitmap_init_loop.getInductionVar();
  auto const_i1_false = builder.create<ConstantOp>(loc,
                                                        builder.getI1Type(),
                                                              builder.getBoolAttr(false));
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_0_to_bitmap = builder.create<memref::StoreOp>(loc,
                                                           const_i1_false,
                                                           numericInfo.ws_bitmap,
                                                               ValueRange{i_idx});
#else
  builder.create<memref::StoreOp>(loc,
                                  const_i1_false,
                                  numericInfo.ws_bitmap,
                                  ValueRange{i_idx});
#endif
  {
    comet_vdump(numericInfo.ws_bitmap);
    comet_vdump(store_0_to_bitmap);
    comet_vdump(bitmap_init_loop);
  }

  /// Allocate the mask_array
  if (PUSH_BASED_MASKING == maskingInfo.mask_type) {
    builder.restoreInsertionPoint(before_for_loop_body_loc);
    numericInfo.mask_array = builder.create<memref::AllocOp>(loc,
                                                             memTy_alloc_bitarray,
                                                             ValueRange{num_cols},
                                                             builder.getI64IntegerAttr(8) /* alignment bytes */);
    /// Initialize mask_array to zeros
    auto mask_array_init_loop = builder.create<scf::ForOp>(loc,
                                                           const_index_0 /* lowerBound */,
                                                           num_cols /* upperBound */,
                                                           const_index_1 /* step */);
    builder.setInsertionPointToStart(mask_array_init_loop.getBody());
    i_idx = mask_array_init_loop.getInductionVar();
    const_i1_false = builder.create<ConstantOp>(loc,
                                                builder.getI1Type(),
                                                builder.getBoolAttr(false));
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    auto store_0_to_mask_array = builder.create<memref::StoreOp>(loc,
                                                                 const_i1_false,
                                                                 numericInfo.mask_array,
                                                                 ValueRange{i_idx});
#else
    builder.create<memref::StoreOp>(loc,
                                    const_i1_false,
                                    numericInfo.mask_array,
                                    ValueRange{i_idx});
#endif

    {
      comet_vdump(numericInfo.mask_array);
      comet_vdump(store_0_to_mask_array);
      comet_vdump(mask_array_init_loop);
    }
  }

  /// Restore the previous Insertion Point
  builder.restoreInsertionPoint(previous_loc);
}

/// ----------------- ///
/// Generate the numeric for-loop that initialize the mask_array using the mask
/// ----------------- ///
///      %j_loc_start = memref.load %mask_rowptr[%i_idx] : memref<?xindex>  /// alloc_16 = mask.rowptr
///      %j_loc_bound = memref.load %mask_rowptr[%i_idx_plus_one] : memref<?xindex>
///      scf.for %arg1 = %j_loc_start to %j_loc_bound step %c1 {
///        %val = memref.load %mask_val[%arg1] : memref<?xf64>
///        %54 = arith.cmpf une, %val, %cst : f64
///        scf.if %54 {
///          %j_idx = memref.load %mask_col[%arg1] : memref<?xindex>
///          memref.store %true, %mask_array[%j_idx] : memref<?xi1>
///        }
///      }
/// ----------------- ///
/// Reset mask_array after the 2nd numeric for-loop
/// ----------------- ///
///      scf.for %arg1 = %j_loc_start to %j_loc_bound step %c1 {
///        %j_idx = memref.load %mask_col[%arg1] : memref<?xindex>
///        memref.store %false, %array_mask[%j_idx] : memref<?xi1>
///      }
void genNumericInitAndResetMaskArrayByMask(std::vector<scf::ForOp> &forLoops /* numeric for-loops, from innermost to outermost*/,
                                   NumericInfo &numericInfo,
                                   MaskingInfo &maskingInfo,
                                   OpBuilder &builder,
                                   Location &loc) {
  /// ----------------- ///
  /// Generate the initialization for-loop
  /// ----------------- ///
  /// Store the insertion point
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Set the Insertion Point to the place before the 2nd-level numeric for-loop
  builder.setInsertionPoint(forLoops[1]);

  /// Generate the for-loop entry
  Value &mask_array = numericInfo.mask_array;
  Value &mask_rowptr = maskingInfo.mask_rowptr;
  Value &mask_col = maskingInfo.mask_col;
  Value &mask_val = maskingInfo.mask_val;
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  Value i_idx = forLoops[2].getInductionVar();  /// i_idx is the induction variable of the outermost for-loop.
  Value i_idx_plus_one = builder.create<AddIOp>(loc, i_idx, const_index_1);
  Value j_loc_start = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx});
  Value j_loc_bound = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx_plus_one});
  auto init_for_loop = builder.create<scf::ForOp>(loc,
                                             j_loc_start /* lower_bound */,
                                             j_loc_bound /* upper_bound*/,
                                             const_index_1 /* step */);
  builder.setInsertionPointToStart(init_for_loop.getBody());

  /// Generate the for-loop body
  Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0));
  Value j_loc = init_for_loop.getInductionVar();
  Value val = builder.create<memref::LoadOp>(loc, mask_val, ValueRange{j_loc});
  Value not_zero = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, val, const_f64_0);
  auto if_not_zero = builder.create<scf::IfOp>(loc, not_zero, false /*NoElseRegion*/);
  builder.setInsertionPointToStart(&if_not_zero.getThenRegion().front());
  Value j_idx = builder.create<memref::LoadOp>(loc, mask_col, ValueRange{j_loc});
  Value const_i1_1 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_true = builder.create<memref::StoreOp>(loc,
                                                    const_i1_1,
                                                    mask_array,
                                                    ValueRange{j_idx});
#else
  builder.create<memref::StoreOp>(loc,
                                  const_i1_1,
                                  mask_array,
                                  ValueRange{j_idx});
#endif
  {
    comet_vdump(store_true);
    comet_vdump(init_for_loop);
  }

  /// ----------------- ///
  /// Generate the resetting for-loop after the 2nd-level numeric for-loop
  /// ----------------- ///
  builder.setInsertionPointAfter(forLoops[1]);
  auto reset_for_loop = builder.create<scf::ForOp>(loc,
                                                   j_loc_start /* lower_bound */,
                                                   j_loc_bound /* upper_bound*/,
                                                   const_index_1 /* step */);
  builder.setInsertionPointToStart(reset_for_loop.getBody());

  /// Generate the for-loop body
  j_loc = reset_for_loop.getInductionVar();
  j_idx = builder.create<memref::LoadOp>(loc, mask_col, ValueRange{j_loc});
  Value const_i1_0 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
  builder.create<memref::StoreOp>(loc,
                                  const_i1_0,
                                  mask_array,
                                  ValueRange{j_idx});
  {
    comet_vdump(reset_for_loop);
  }

  /// Restore the insertion point
  builder.restoreInsertionPoint(last_insertion_point);
}



/// ----------------- ///
/// Declare the variable Mark = 0.
/// Mark is used along with the mark_array to replace the WS_bitmap.
///     %mark = memref.alloc() : memref<1xindex>
///     memref.store %c0, %mark[%c0] : memref<1xindex>
/// ----------------- ///
Value initVariableMarkNumeric(
//    PatternRewriter &rewriter,
        OpBuilder &builder,
        Location &loc,
        std::vector<scf::ForOp> &for_loops /*for-loop statements, from innermost to outermost*/) {
  /* ----------------- *
    %c0 = arith.constant 0 : index
    %mark = memref.alloc() : memref<1xindex>
    memref.store %c0, %mark[%c0] : memref<1xindex>
   * ----------------- */

  /// Save the old Insertion Point
  auto previous_loc = builder.saveInsertionPoint();

  /// Jump Insertion Point to the front of the outermost for-loop
  builder.setInsertionPoint(for_loops.back());
  {
    comet_debug() << "for_loops.back()\n";
    comet_vdump(for_loops.back());
  }

  /// Create the variable Mark
  /// %mark = memref.alloc() : memref<1xindex>
  MemRefType memTy_alloc_mark = MemRefType::get({1}, builder.getIndexType());
  Value alloc_mark = builder.create<memref::AllocOp>(loc, memTy_alloc_mark);
  {
    comet_debug() << "Allocate the variable Mark\n";
    comet_vdump(alloc_mark);
  }

  /// Initialize Mark = 0
  /// memref.store %c0, %mark[%c0] : memref<1xindex>
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  std::vector<Value> store_index = {const_index_0};
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_0_to_mark = builder.create<memref::StoreOp>(loc, const_index_0, alloc_mark, store_index);
#else
  builder.create<memref::StoreOp>(loc, const_index_0, alloc_mark, store_index);
#endif
  {
    comet_debug() << "Initialize Mark to zero\n";
    comet_vdump(store_0_to_mark);
  }

  /// Restore the previous Insertion Point
  builder.restoreInsertionPoint(previous_loc);

  return alloc_mark;
} /// End function initVariableMarkNumeric()


/// ----------------- ///
/// Update mark += 2 for every row A[i,:] in A
/// ----------------- ///
///      %old_mark = memref.load %mark[%c0] : memref<1xindex>
///      %new_mark = arith.addi %old_mark, %c2 : index
///      memref.store %new_mark, %mark[%c0] : memref<1xindex>
/// ----------------- ///
Value updateVariableMarkNumeric(
        Value &alloc_mark,
//    PatternRewriter &rewriter,
        OpBuilder &builder,
        Location &loc,
        std::vector<scf::ForOp> &for_loops /*for-loop statements, from innermost to outermost*/) {
  /* ----------------- *
    %c2 = arith.constant 2 : index
    %old_val = memref.load %mark[%c0] : memref<1xindex>
    %new_mark = arith.addi %old_val, %c2 : index
    memref.store %new_mark, %mark[%c0] : memref<1xindex>
   * ----------------- */


  /// Save the old Insertion Point
  auto previous_loc = builder.saveInsertionPoint();

  /// Jump Insertion Point to the front of the 2nd outermost for-loop
  builder.setInsertionPoint(for_loops[for_loops.size() - 2]);

  /// %c2 = arith.constant 2 : index
  Value const_index_2 = builder.create<ConstantIndexOp>(loc, 2);
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);

  /// %old_val = memref.load %mark[%c0] : memref<1xindex>
  Value old_mark = builder.create<memref::LoadOp>(loc, alloc_mark, ValueRange{const_index_0});
  {
    comet_debug() << "Load old mark\n";
    comet_vdump(old_mark);
  }

  /// %new_mark = arith.addi %old_val, %c2 : index
  Value new_mark = builder.create<AddIOp>(loc, old_mark, const_index_2);
  {
    comet_debug() << "new_mark = old_mark + 2\n";
    comet_vdump(new_mark);
  }

  /// memref.store %new_mark, %mark[%c0] : memref<1xindex>
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_new_mark = builder.create<memref::StoreOp>(loc, new_mark, alloc_mark, ValueRange{const_index_0});
  comet_debug() << "Store the new mark\n";
  comet_vdump(store_new_mark);
#else
  builder.create<memref::StoreOp>(loc, new_mark, alloc_mark, ValueRange{const_index_0});
#endif

  /// Restore the previous Insertion Point
  builder.restoreInsertionPoint(previous_loc);

  return new_mark;
}

/// ----------------- ///
/// Generate if statement for the Compute node
///     %58 = memref.load %array_mask[%57] : memref<?xi1>
///     %59 = arith.cmpi eq, %58, %true : i1
///     scf.if %59 {
///       %b_t = memref.load %ws_bitmap[%57] : memref<?xi1>
///       %not_visited = arith.cmpi eq, %b_t, %false : i1
///       scf.if %not_visited {
/// ----------------- ///
void genIfStatementConditionNumeric(
        OpBuilder &builder,
        Location &loc,
//        int lhs_loc,
//        const std::vector<std::vector<Value>> &tensors_lhs_Allocs,
//        const std::vector<std::vector<Value>> &allValueAccessIdx,
//        Value &new_mark,
        Value &accessIdx,
        NumericInfo &numericInfo,
        MaskingInfo &maskingInfo,
        scf::IfOp &if_notAlreadySet /* output */) {

  Value const_i1_0 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
  Value const_i1_1 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));

  if (NO_MASKING == maskingInfo.mask_type) {
    ///     %b_t = memref.load %ws_bitmap[%57] : memref<?xi1>
    ///     %not_visited = arith.cmpi eq, %b_t, %false : i1
    ///     scf.if %not_visited {

    Value ele_bitmap = builder.create<memref::LoadOp>(loc, numericInfo.ws_bitmap, ValueRange{accessIdx});
    Value not_visited = builder.create<CmpIOp>(loc, CmpIPredicate::eq, ele_bitmap, const_i1_0);
    if_notAlreadySet = builder.create<scf::IfOp>(loc, not_visited, /*WithElseRigion*/ true);
    {
      comet_vdump(ele_bitmap);
      comet_vdump(not_visited);
      comet_vdump(if_notAlreadySet);
    }

  } else if (PUSH_BASED_MASKING == maskingInfo.mask_type) {
    ///     %58 = memref.load %array_mask[%57] : memref<?xi1>
    ///     %59 = arith.cmpi eq, %58, %true : i1
    ///     scf.if %59 {
    ///       %b_t = memref.load %ws_bitmap[%57] : memref<?xi1>
    ///       %not_visited = arith.cmpi eq, %b_t, %false : i1
    ///       scf.if %not_visited {

    Value ele_mask_array = builder.create<memref::LoadOp>(loc, numericInfo.mask_array, ValueRange{accessIdx});
    Value is_set = builder.create<CmpIOp>(loc, CmpIPredicate::eq, ele_mask_array, const_i1_1);
    auto if_mask_set = builder.create<scf::IfOp>(loc, is_set, /*NoElseRegion*/ false);
    builder.setInsertionPointToStart(&if_mask_set.getThenRegion().front());
    Value ele_bitmap = builder.create<memref::LoadOp>(loc, numericInfo.ws_bitmap, ValueRange{accessIdx});
    Value not_visited = builder.create<CmpIOp>(loc, CmpIPredicate::eq, ele_bitmap, const_i1_0);
    if_notAlreadySet = builder.create<scf::IfOp>(loc, not_visited, /*WithElseRigion*/ true);
    {
      comet_vdump(ele_mask_array);
      comet_vdump(is_set);
      comet_vdump(if_mask_set);
    }

  } else {
    llvm::errs() << "Error: genIfStatementConditionNumeric(): mask_type " << maskingInfo.mask_type << " is not supported.\n";
  }


//  //////////////////////////////////////////
//  // Workspace tensors are on the lhs
//  comet_debug() << " lhs_loc: " << lhs_loc << "\n";
//  Value checkAlreadySet = builder.create<memref::LoadOp>(loc, tensors_lhs_Allocs[1][0], allValueAccessIdx[lhs_loc]);
////  comet_debug() << " ";
//  comet_vdump(checkAlreadySet);
////  comet_debug() << " ";
//  comet_vdump(checkAlreadySet.getType());
////  comet_debug() << " ";
////  comet_vdump(const_i1_0.getType());
//
//  scf::IfOp if_notAlreadySet;
//  if (NO_MASKING == maskingInfo.mask_type) {
//    Value notAlreadySet = builder.create<CmpIOp>(loc, CmpIPredicate::ne, checkAlreadySet, new_mark);
////  Value notAlreadySet = builder.create<CmpIOp>(loc, CmpIPredicate::eq, checkAlreadySet, const_i1_0);
//    comet_vdump(notAlreadySet);
//    if_notAlreadySet = builder.create<scf::IfOp>(loc, notAlreadySet, /*WithElseRegion*/ true);
//  } else if (PUSH_BASED_MASKING == maskingInfo.mask_type) {
//    Value notAlreadySet = builder.create<CmpIOp>(loc, CmpIPredicate::eq, checkAlreadySet, new_mark);
//    comet_vdump(notAlreadySet);
//    if_notAlreadySet = builder.create<scf::IfOp>(loc, notAlreadySet, /*WithElseRegion*/ true);
//  } else {
//    llvm::errs() << "Error: genIfStatementConditionNumeric(): mask_type " << maskingInfo.mask_type << " is not supported.\n";
//  }
//
//  {
//    comet_debug() << " If branch:\n";
//    comet_vdump(if_notAlreadySet);
//  }
//
//  return if_notAlreadySet;
}

/// ----------------- ///
/// Generate if statement's then region in the numeric kernel
/// ----------------- ///
/// For no masking
/// if (mark_array[j_idx] != mark) {
///     mark_array[j_idx] = mark;
///     C_col[rowptr] = j_idx;
///     rowptr += 1;
///     WS_data[j_idx] = a * b;
/// }
/// For push-based masking
/// if (mark_array[j_idx] == mark) {
///     mark_array[j_idx] = mark + 1;
///     C_col[rowptr] = j_idx;
///     rowptr += 1;
///     WS_data[j_idx] = a * b;
/// }
///
///
///     scf.if %not_visited {
///       memref.store %true, %ws_bitmap[%57] : memref<?xi1>
///       %61 = memref.load %alloc_20[%arg2] : memref<?xf64> // alloc_20 = B.val
///       %62 = arith.mulf %60, %61 : f64
///       memref.store %62, %alloc_22[%57] : memref<?xf64>  // workspace = A * B
///       %64 = memref.load %alloc_42[%c0] : memref<1xindex>  // rowptr
///       memref.store %57, %alloc_40[%64] : memref<?xindex>  // C.col[rowptr] = col
///       %65 = arith.addi %64, %c1 : index   // rowptr += 1
///       memref.store %65, %alloc_42[%c0] : memref<1xindex>
///     }
void genIfStatementThenRegionNumeric(
        scf::IfOp &if_notAlreadySet,
//    PatternRewriter &rewriter,
        OpBuilder &builder,
        Location &loc,
        int main_tensor_nums,
        int lhs_loc,
        llvm::StringRef &semiringSecond,
        bool compressedWorkspace,
//    Value &const_i1_1,
//        Value &new_mark,
//        Value &const_index_0,
        const std::vector<std::vector<Value>> &main_tensors_all_Allocs,
        const std::vector<std::vector<Value>> &allValueAccessIdx,
//        const std::vector<std::vector<Value>> &tensors_lhs_Allocs,
        SymbolicInfo &symbolicInfo,
        NumericInfo &numericInfo) {
//        MaskingInfo &maskingInfo) {

  builder.setInsertionPointToStart(&if_notAlreadySet.getThenRegion().front());

  /// ----------------- ///
  /// Do computation Wj = Aik * Bkj
  /// ----------------- ///

  std::vector<Value> allLoadsIf(main_tensor_nums);
  for (int m = 0; m < main_tensor_nums; m++) {
    Value s = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1],
                                             allValueAccessIdx[m]);
    allLoadsIf[m] = s;
    comet_vdump(s);
  }
  comet_debug() << " allLoadsIf.size(): " << allLoadsIf.size() << "\n";

  comet_debug() << "calculate elementWise operation only\n";
  Value elementWiseResult = getSemiringSecondVal(builder, loc, semiringSecond, allLoadsIf[0], allLoadsIf[1],
                                                 compressedWorkspace);
  builder.create<memref::StoreOp>(loc,
                                  elementWiseResult,
                                  main_tensors_all_Allocs[lhs_loc].back(),
                                  allValueAccessIdx[lhs_loc]);


  /// ----------------- ///
  /// Backup of W = A * B
  /// ----------------- ///
  // Wj = Aik * Bkj          // computation wj, outer has k, so +=/= need if/else
  // W_already_set[j] = 1
  // W_index_list[W_index_list_size] = j
  // W_index_list_size++
//
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//  auto store_sum = builder.create<memref::StoreOp>(loc, elementWiseResult,
//                                                   main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1],
//                                                   allValueAccessIdx[2]);
//  comet_vdump(elementWiseResult);
//  comet_vdump(store_sum);
//#else
//  builder.create<memref::StoreOp>(loc, elementWiseResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], allValueAccessIdx[2]);
//#endif
  /// ----------------- ///
  /// End Backup of W = A * B
  /// ----------------- ///

  /// ----------------- ////
  /// Look-up Table
  /// ----------------- ////
  /// tensors_lhs_Allocs[1][0]                : mark_array (DEPRECATED)
  /// allValueAccessIdx[lhs_loc][0]           : j_idx (no-masking's lhs_loc is 2, push-masking's lhs_loc is 3)
  /// main_tensors_all_Allocs[lhs_loc].back() : W_data
  /// main_tensors_all_Allocs[2].back()       : W_data (NO_MASKING)
  /// main_tensors_all_Allocs[3].back()       : W_data (PUSH_BASED_MASKING)
  /// tensors_lhs_Allocs[2][0]                : W_index_list (DEPRECATED)
  /// tensors_lhs_Allocs[3][0]                : W_index_list_size (DEPRECATED)
  /// ----------------- ///

  /// ----------------- ///
  /// Update bitmap ws_bitmap[j_idx] = true;
  /// ----------------- ///
  Value const_i1_1 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
  builder.create<memref::StoreOp>(loc,
                                  const_i1_1,
                                  numericInfo.ws_bitmap,
                                  allValueAccessIdx[lhs_loc]);

//  if (NO_MASKING == maskingInfo.mask_type) {
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//    auto assign_new_mark = builder.create<memref::StoreOp>(loc, new_mark, tensors_lhs_Allocs[1][0],
//                                                           allValueAccessIdx[lhs_loc]);
//    comet_vdump(assign_new_mark);
//#else
//    builder.create<memref::StoreOp>(loc, new_mark, tensors_lhs_Allocs[1][0], allValueAccessIdx[lhs_loc]);
//#endif
//
//  } else if (PUSH_BASED_MASKING == maskingInfo.mask_type) {
//    ///     %m_v_plus_one = arith.addi %m_v, %c1 : index
//    ///     memref.store %m_v_plus_one, %mark_array[%j_idx] : memref<?xindex>  // mark_array[B_col_id] = new_mark
//    Value mark_value_plus_one = builder.create<AddIOp>(loc, new_mark, const_index_1);
//
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//    auto store_mark_value_plus_one = builder.create<memref::StoreOp>(loc,
//                                                                     mark_value_plus_one,
//                                                                     tensors_lhs_Allocs[1][0],
//                                                                     allValueAccessIdx[lhs_loc]);
//    {
//      comet_vdump(mark_value_plus_one);
//      comet_vdump(store_mark_value_plus_one);
//    }
//#else
//    builder.create<memref::StoreOp>(loc,
//                                    mark_value_plus_one,
//                                    tensors_lhs_Allocs[1][0],
//                                    allValueAccessIdx[lhs_loc]);
//#endif
//  } else {
//    llvm::errs() << "Error: genIfStatementThenRegionNumeric(): masking type " << maskingInfo.mask_type << " is not supported, yet.\n";
//  }

  /// ----------------- ///
  /// Store column ID: C_col[rowpt] = j_idx
  /// Update rowptr: rowptr += 1
  /// ----------------- ///
  /// ----------------- ///
  ///     %row = memref.load %rowptr[%c0] : memref<1xindex>  // row = rowptr
  ///     memref.store %j_idx, %C_col[%row] : memref<?xindex>  // C_col[row] = B_col_id
  ///     %new_rowptr = arith.addi %row, %c1 : index // new_rowptr = ++row
  ///     memref.store %new_rowptr, %rowptr[%c0] : memref<1xindex> //  rowptr = new_rowptr
//  ///     memref.store %mul_val, %ws_data[%j_idx] : memref<?xf64>  // ws_data[B_col_id] = A_val * B_val
  /// ----------------- ///
  /// allValueAccessIdx[lhs_loc][0] : %j_idx
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  Value &row_offset = symbolicInfo.row_offset;
  Value &mtxC_col = symbolicInfo.mtxC_col;
  Value old_row_offset = builder.create<memref::LoadOp>(loc, row_offset, const_index_0);
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//  auto store_col_id = builder.create<memref::StoreOp>(loc,
//                                                      allValueAccessIdx[lhs_loc][0],
//                                                      mtxC_col,
//                                                      ValueRange{old_row_offset});
//#else
  builder.create<memref::StoreOp>(loc,
                                  allValueAccessIdx[lhs_loc][0],
                                  mtxC_col,
                                  ValueRange{old_row_offset});
//#endif

  Value new_row_offset = builder.create<AddIOp>(loc, old_row_offset, const_index_1);


//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//  auto store_new_row_offset = builder.create<memref::StoreOp>(loc,
//                                                               new_row_offset,
//                                                               row_offset,
//                                                               ValueRange{const_index_0});
////  auto store_sum = builder.create<memref::StoreOp>(loc,
////                                                     elementWiseResult,
////                                                     main_tensors_all_Allocs[lhs_loc].back(),
////                                                     allValueAccessIdx[lhs_loc]);
//#else
  builder.create<memref::StoreOp>(loc,
                                  new_row_offset,
                                  row_offset,
                                  ValueRange{const_index_0});
//  builder.create<memref::StoreOp>(loc,
//                                  elementWiseResult,
//                                  main_tensors_all_Allocs[lhs_loc].back(),
//                                  allValueAccessIdx[lhs_loc]);
//#endif


//  if (NO_MASKING == maskingInfo.mask_type) {
//    /// main_tensors_all_Allocs[2].back() is W_data
//
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//    auto store_sum = builder.create<memref::StoreOp>(loc,
//                                                     elementWiseResult,
//                                                     main_tensors_all_Allocs[2].back(),
//                                                     allValueAccessIdx[lhs_loc]);
//    comet_vdump(store_sum);
//#else
//    builder.create<memref::StoreOp>(loc,
//                                    elementWiseResult,
//                                    main_tensors_all_Allocs[2].back(),
//                                    allValueAccessIdx[lhs_loc]);
//#endif
//
//  } else if (PUSH_BASED_MASKING == maskingInfo.mask_type) {
//    /// main_tensors_all_Allocs[3].back() is W_data
//
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//    auto store_sum = builder.create<memref::StoreOp>(loc,
//                                                     elementWiseResult,
//                                                     main_tensors_all_Allocs[3].back(),
//                                                     allValueAccessIdx[lhs_loc]);
//    comet_vdump(store_sum);
//#else
//    builder.create<memref::StoreOp>(loc,
//                                    elementWiseResult,
//                                    main_tensors_all_Allocs[3].back(),
//                                    allValueAccessIdx[lhs_loc]);
//#endif
//
//  } else {
//    llvm::errs() << "Error: genIfStatementThenRegionNumeric(): masking type " << maskingInfo.mask_type << " is not supported, yet.\n";
//  }
  {
//    comet_vdump(old_row_offset);
//    comet_vdump(store_col_id);
//    comet_vdump(new_row_offset);
//    comet_vdump(store_new_row_offset);
//    comet_vdump(store_sum);
    comet_vdump(if_notAlreadySet);
  }


  /// ----------------- ///
  /// Backup of W_index_list_size and W_index_list (tensors_lhs_Allocs[2][0]).
  /// tensors_lhs_Allocs[2][0] : W_index_list
  /// tensors_lhs_Allocs[3][0] : W_index_list_size
  /// ----------------- ///
//  Value W_index_list_size_old = builder.create<memref::LoadOp>(loc, tensors_lhs_Allocs[3][0],
//                                                               ValueRange{const_index_0});
//
//  assert(allValueAccessIdx[lhs_loc].size() == 1 && " more than one access id for auxiliary array\n");
//  builder.create<memref::StoreOp>(loc, allValueAccessIdx[lhs_loc][0], tensors_lhs_Allocs[2][0],
//                                  ValueRange{W_index_list_size_old});
//
//  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
//  Value W_index_list_size_new = builder.create<AddIOp>(loc, W_index_list_size_old, const_index_1);
//  comet_debug() << " AddIOps (W_index_list_size_new) \n";
//  comet_vdump(W_index_list_size_new);
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//  auto store_new_value = builder.create<memref::StoreOp>(loc, W_index_list_size_new, tensors_lhs_Allocs[3][0],
//                                                         ValueRange{const_index_0});
//  comet_vdump(store_new_value);
//#else
//  builder.create<memref::StoreOp>(loc, W_index_list_size_new, tensors_lhs_Allocs[3][0], ValueRange{const_index_0});
//#endif
  /// ----------------- ///
  /// End Backup
  /// ----------------- ///
}


/// ----------------- ///
/// Generate if statement's else region in the numeric kernel
/// ----------------- ///
void genIfStatementElseRegionNumeric(
        scf::IfOp &if_notAlreadySet,
//    PatternRewriter &rewriter,
        OpBuilder &builder,
        Location &loc,
        int main_tensor_nums,
        int lhs_loc,
        llvm::StringRef &semiringFirst,
        bool compressedWorkspace,
        llvm::StringRef &semiringSecond,
        const std::vector<std::vector<Value>> &main_tensors_all_Allocs,
        const std::vector<std::vector<Value>> &allValueAccessIdx,
        MaskingInfo &maskingInfo) {
  builder.setInsertionPointToStart(&if_notAlreadySet.getElseRegion().front());

  std::vector<Value> allLoadsElse(main_tensor_nums);
  for (auto m = 0; m < main_tensor_nums; m++) {
    Value s = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[m].back(),
                                             allValueAccessIdx[m]);
    allLoadsElse[m] = s;
    comet_debug() << " ";
    comet_vdump(s);
  }
  comet_debug() << " allLoadsElse.size(): " << allLoadsElse.size() << "\n";

  comet_debug() << "calculate elementWise operation and reduction\n";
  Value elementWiseResult = getSemiringSecondVal(builder, loc, semiringSecond, allLoadsElse[0], allLoadsElse[1],
                                                 compressedWorkspace);
//  Value reduceResult = getSemiringFirstVal(builder, loc, semiringFirst, allLoadsElse[2], elementWiseResult,
//                                           compressedWorkspace);
//  builder.create<memref::StoreOp>(loc, reduceResult,
//                                  main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1],
//                                  allValueAccessIdx[lhs_loc]);


  /// main_tensors_all_Allocs[lhs_loc].back() is W_data
  /// ATTENTION: this is tested only for NO_MASKING and PUSH_BASED_MASKING.

  Value reduceResult = getSemiringFirstVal(builder, loc, semiringFirst, allLoadsElse[lhs_loc], elementWiseResult,
                                           compressedWorkspace);

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto store_sum = builder.create<memref::StoreOp>(loc,
                                                   reduceResult,
                                                   main_tensors_all_Allocs[lhs_loc].back(),
                                                   allValueAccessIdx[lhs_loc]);
  comet_vdump(store_sum);
#else
  builder.create<memref::StoreOp>(loc,
                                  reduceResult,
                                  main_tensors_all_Allocs[lhs_loc].back(),
                                  allValueAccessIdx[lhs_loc]);
#endif
  {
    comet_vdump(if_notAlreadySet);
  }

//  if (NO_MASKING == maskingInfo.mask_type) {
//    /// main_tensors_all_Allocs[2].back() is W_data
//
//    Value reduceResult = getSemiringFirstVal(builder, loc, semiringFirst, allLoadsElse[2], elementWiseResult,
//                                             compressedWorkspace);
//
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//    auto store_sum = builder.create<memref::StoreOp>(loc,
//                                                     reduceResult,
//                                                     main_tensors_all_Allocs[2].back(),
//                                                     allValueAccessIdx[lhs_loc]);
//#else
//    builder.create<memref::StoreOp>(loc,
//                                    reduceResult,
//                                    main_tensors_all_Allocs[2].back(),
//                                    allValueAccessIdx[lhs_loc]);
//#endif
//
//  } else if (PUSH_BASED_MASKING == maskingInfo.mask_type) {
//    /// main_tensors_all_Allocs[3].back() is W_data
//
//    Value reduceResult = getSemiringFirstVal(builder, loc, semiringFirst, allLoadsElse[3], elementWiseResult,
//                                             compressedWorkspace);
//
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//    auto store_sum = builder.create<memref::StoreOp>(loc,
//                                                     reduceResult,
//                                                     main_tensors_all_Allocs[3].back(),
//                                                     allValueAccessIdx[lhs_loc]);
//#else
//    builder.create<memref::StoreOp>(loc,
//                                    reduceResult,
//                                    main_tensors_all_Allocs[3].back(),
//                                    allValueAccessIdx[lhs_loc]);
//#endif
//
//  } else {
//    llvm::errs() << "Error: genIfStatementElseRegionNumeric(): masking type " << maskingInfo.mask_type << " is not supported, yet.\n";
//  }
}



void formSemiringLoopBody(bool comp_worksp_opt, llvm::StringRef &semiringFirst,
                          llvm::StringRef &semiringSecond,
//                          PatternRewriter &rewriter, Location loc, int lhs_loc,
                          OpBuilder &builder, Location &loc, int lhs_loc,
                          std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                          std::vector<std::vector<Value>> &tensors_lhs_Allocs,
                          std::vector<std::vector<Value>> &tensors_rhs_Allocs,
                          std::vector<std::vector<Value>> &allValueAccessIdx,
                          std::vector<std::vector<Value>> &allAccessIdx,
                          std::vector<scf::ForOp> &forLoops /*for-loop statements, from innermost to outermost*/,
                          std::vector<std::vector<int>> &rhsPerms,
                          std::vector<std::vector<std::string>> &rhsFormats,
                          std::vector<std::vector<std::string>> &lhsFormats,
                          SymbolicInfo &symbolicInfo,
                          NumericInfo &numericInfo,
                          MaskingInfo &maskingInfo) {
  bool isMixedMode = checkIsMixedMode(rhsFormats);
  bool isElementwise = checkIsElementwise(rhsPerms);
  comet_debug() << " isElementwise:" << isElementwise << " isMixedMode: " << isMixedMode << "\n";
  auto ctx = builder.getContext();
  IndexType indexType = IndexType::get(ctx);

  if ((semiringFirst.size() == 0) || (semiringSecond.size() == 0))
    llvm::errs() << "Error during semiring parsing!"
                 << "\n";

  if (main_tensors_all_Allocs.size() != allValueAccessIdx.size())
    llvm::errs() << "DEBUG ONLY: issue with main_tensor_nums size"
                 << "\n";

//  Value const_i1_0 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(0));
//  Value const_i1_1 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(1));
  Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
  auto f64Type = builder.getF64Type();
  auto const_f64_0 = builder.create<ConstantOp>(loc, f64Type, builder.getF64FloatAttr(0));

  int main_tensor_nums = main_tensors_all_Allocs.size();
  bool compressedWorkspace = false;

  if (comp_worksp_opt) // always lhs is dense after workspace transformations
  {
    assert(symbolicInfo.is_SpGEMM && "comp_worksp_opt should work on SpGEMM.\n");
    compressedWorkspace = true;

    /// ----------------- ///
    /// Allocate the ws_bitmap and mask_array before the outermost for-loop for the numeric phase.
    /// ----------------- ///
    genAllocBitmapAndMaskArray(builder,
                               loc,
                               forLoops /*for-loop statements, from innermost to outermost*/,
                               symbolicInfo,
                               maskingInfo,
                               numericInfo /* contents updated after call */);
//                               numericAuxiliary /* output */);

//    /// ----------------- ///
//    /// Initialize the variable Mark in front of the outermost for-loop
//    ///     %mark = memref.alloc() : memref<1xindex>
//    ///     memref.store %c0, %mark[%c0] : memref<1xindex>
//    /// ----------------- ///
//    Value alloc_mark = initVariableMarkNumeric(builder,
//                                        loc,
//                                        forLoops /*for-loop statements, from innermost to outermost*/);
//
//    /// ----------------- ///
//    /// Update the variable Mark for every row A[i,:] in A.
//    ///      %old_mark = memref.load %mark[%c0] : memref<1xindex>
//    ///      %new_mark = arith.addi %old_mark, %c2 : index
//    ///      memref.store %new_mark, %mark[%c0] : memref<1xindex>
//    /// ----------------- ///
//    Value new_mark = updateVariableMarkNumeric(alloc_mark,
//                                        builder,
//                                        loc,
//                                        forLoops);


    if (PUSH_BASED_MASKING == maskingInfo.mask_type) {
//      /// ----------------- ///
//      /// Generate the numeric for-loop that initialize the mark_array by using the mask
//      /// new_mark: the update value of mark
//      /// tensors_lhs_Allocs[1][0] : the mask tensor
//      /// ----------------- ///
//      {
//        comet_vdump(new_mark);
//        comet_vdump(tensors_lhs_Allocs[1][0]);
//      }
//      genNumericInitMarkArrayByMask(forLoops /* numeric for-loops, from innermost to outermost*/,
//                                    new_mark,
//                                    tensors_lhs_Allocs[1][0],
//                                    maskingInfo,
//                                    builder,
//                                    loc);
      /// ----------------- ///
      /// Generate the numeric for-loop that initialize the mask_array using the mask,
      /// and reset the mask_array after the 2nd numeric for-loop.
      /// ----------------- ///
      genNumericInitAndResetMaskArrayByMask(forLoops /* numeric for-loops, from innermost to outermost*/,
                                    numericInfo,
                                    maskingInfo,
                                    builder,
                                    loc);
    }
    /// ----------------- ///
    /// Generate if statement's condition
    ///     %58 = memref.load %array_mask[%57] : memref<?xi1>
    ///     %59 = arith.cmpi eq, %58, %true : i1
    ///     scf.if %59 {
    ///       %b_t = memref.load %ws_bitmap[%57] : memref<?xi1>
    ///       %not_visited = arith.cmpi eq, %b_t, %false : i1
    ///       scf.if %not_visited {
    /// ----------------- ///
    scf::IfOp if_notAlreadySet;
    genIfStatementConditionNumeric(builder,
                                   loc,
                                   allValueAccessIdx[lhs_loc][0],
                                   numericInfo,
                                   maskingInfo,
                                   if_notAlreadySet);

    if (!if_notAlreadySet.getThenRegion().empty()) {
      /// ----------------- ///
      /// Generate if statement's then region
      /// ----------------- ///
      genIfStatementThenRegionNumeric(if_notAlreadySet,
                                      builder,
                                      loc,
                                      main_tensor_nums,
                                      lhs_loc,
                                      semiringSecond,
                                      compressedWorkspace,
                                      main_tensors_all_Allocs,
                                      allValueAccessIdx,
                                      symbolicInfo,
                                      numericInfo);
//                                      maskingInfo);
    }

    if (!if_notAlreadySet.getElseRegion().empty()) {
      /// ----------------- ///
      /// Generate if statement's else region
      /// ----------------- ///
      genIfStatementElseRegionNumeric(if_notAlreadySet,
                                      builder,
                                      loc,
                                      main_tensor_nums,
                                      lhs_loc,
                                      semiringFirst,
                                      compressedWorkspace,
                                      semiringSecond,
                                      main_tensors_all_Allocs,
                                      allValueAccessIdx,
                                      maskingInfo);
    }
//    /// ----------------- ///
//    /// Generate if statement's condition
//    ///   %76 = memref.load %mark_array[%73] : memref<?xindex>
//    ///   %77 = arith.cmpi ne, %76, %new_mark : index
//    ///   scf.if %77 {
//    /// ----------------- ///
//    auto if_notAlreadySet = genIfStatementConditionNumeric(builder,
//                                                         loc,
//                                                         lhs_loc,
//                                                         tensors_lhs_Allocs,
//                                                         allValueAccessIdx,
////                                                         const_i1_0,
//                                                         new_mark,
//                                                         maskingInfo);
//
//    // if-then region corresponding to if_notAlreadySet instruction.
//    // if (&if_notAlreadySet. getThenRegion())
//    if (!if_notAlreadySet.getThenRegion().empty()) {
//      /// ----------------- ///
//      /// Generate if statement's then region
//      /// ----------------- ///
//      genIfStatementThenRegionNumeric(if_notAlreadySet,
//                                    builder,
//                                    loc,
//                                    main_tensor_nums,
//                                    lhs_loc,
//                                    semiringSecond,
//                                    compressedWorkspace,
////                                    const_i1_1,
//                                    new_mark,
//                                    const_index_0,
//                                    main_tensors_all_Allocs,
//                                    allValueAccessIdx,
//                                    tensors_lhs_Allocs,
//                                    symbolicInfo,
//                                    maskingInfo);
//    }
//
//    // if-else region corresponding to if_notAlreadySet instruction.
//    // if (&if_notAlreadySet.getElseRegion())
//    if (!if_notAlreadySet.getElseRegion().empty()) {
//      /// ----------------- ///
//      /// Generate if statement's else region
//      /// ----------------- ///
//      genIfStatementElseRegionNumeric(if_notAlreadySet,
//                                    builder,
//                                    loc,
//                                    main_tensor_nums,
//                                    lhs_loc,
//                                    semiringFirst,
//                                    compressedWorkspace,
//                                    semiringSecond,
//                                    main_tensors_all_Allocs,
//                                    allValueAccessIdx,
//                                    maskingInfo);
//    }

  } else { // general dense or mixed mode computation, no need workspace transformations
    std::vector<Value> allLoads(main_tensor_nums);
    for (auto m = 0; m < main_tensor_nums; m++) {
      Value load_op = builder.create<memref::LoadOp>(loc,
                                                     main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1], allValueAccessIdx[m]);
      allLoads[m] = load_op;
      comet_debug() << " ";
      comet_vdump(load_op);
    }
    comet_debug() << " allLoads.size(): " << allLoads.size() << "\n";

    // if computeOp is elementwise mixed mode operation, the output is sparse
    if (isMixedMode && isElementwise && !checkIsDense(lhsFormats[0])) {

      int dense_inputtensor_id = 0;
      for (unsigned int i = 0; i < rhsFormats.size(); i++) {
        if (checkIsDense(rhsFormats[i])) {
          dense_inputtensor_id = i;
          break;
        }
      }

      int sparse_inputtensor_id = dense_inputtensor_id ? 0 : 1;
      std::string sparse_format = getTensorFormat(rhsFormats, sparse_inputtensor_id);

      auto last_insertionPoint = builder.saveInsertionPoint();

      /// Need to initialize some memory accesses outside the nested loop
      /// Reset the insertion point: the body of the innermost loop
      comet_debug() << "LoopSize: " << forLoops.size() << " Loop:\n";
      comet_vdump(forLoops[forLoops.size() - 1]);
      builder.setInsertionPoint(forLoops[forLoops.size() - 1]);

      Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
      MemRefType memTy_alloc_Cnnz = MemRefType::get({1}, indexType);
      Value alloc_Cnnz = builder.create<memref::AllocOp>(loc, memTy_alloc_Cnnz);
      comet_debug() << " AllocOp for Cnnz: ";
      comet_vdump(alloc_Cnnz);

      std::vector<Value> alloc_Cnnz_insert_loc = {const_index_0};
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      auto store_Cnnz = builder.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz, alloc_Cnnz_insert_loc);
      comet_debug() << " StoreOp: ";
      comet_vdump(store_Cnnz);
#else
      builder.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz, alloc_Cnnz_insert_loc);
#endif

      // The following code block is needed to update Update C2pos in the case of output tensor is in DCSR
      Value Cnnz_index_old;
      Value alloc_Cnnz_row;
      if (sparse_format.compare("DCSR") == 0) {
        alloc_Cnnz_row = builder.create<memref::AllocOp>(loc, memTy_alloc_Cnnz);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
        auto store_Cnnz_row = builder.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz_row,
                                                              alloc_Cnnz_insert_loc);
        comet_debug() << " StoreOp DCSR: ";
        comet_vdump(store_Cnnz_row);
#else
        builder.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
#endif
        // Get Cnnz_old
        Cnnz_index_old = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
      }

      builder.restoreInsertionPoint(last_insertionPoint);

      comet_debug() << " dense_inputtensor_id: " << dense_inputtensor_id << "\n";
      comet_debug() << " sparse_inputtensor_id: " << sparse_inputtensor_id << "\n";
      Value denseInput_is_nonzero = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, allLoads[dense_inputtensor_id],
                                                           const_f64_0);
      auto if_nonzero = builder.create<scf::IfOp>(loc, denseInput_is_nonzero, /*WithElseRegion*/ false);
      comet_debug() << " If branch:\n";
      comet_vdump(if_nonzero);

      if (!if_nonzero.getThenRegion().empty()) {

        builder.setInsertionPointToStart(&if_nonzero.getThenRegion().front());

        comet_debug() << "calculate product and sum in \n";
        Value elementWiseResult = getSemiringSecondVal(builder, loc, semiringSecond, allLoads[0], allLoads[1],
                                                       compressedWorkspace);

        // Get Cnnz
        Value Cnnz_index = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);

// Store product to Cval
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
        comet_debug() << "Store product to Cval\n";
        auto store_Cval = builder.create<memref::StoreOp>(loc, elementWiseResult, main_tensors_all_Allocs[2][
          main_tensors_all_Allocs[2].size() - 1], Cnnz_index);
        comet_debug() << " StoreOp: ";
        comet_vdump(store_Cval);

        // Update C1crd, C2crd
        comet_debug() << "Getting A1crd\n";
        comet_debug() << "allValueAccessIdx[" << sparse_inputtensor_id << "].size(): "
                      << allAccessIdx[sparse_inputtensor_id].size() << "\n";
        comet_vdump(allAccessIdx[sparse_inputtensor_id][0]);

        for (unsigned int i = 0; i < allAccessIdx.size(); i++) {
          comet_debug() << "allAccessIdx[" << i << "].size(): " << allAccessIdx[i].size() << "\n";
          for (auto n: allAccessIdx[i]) {
            comet_vdump(n);
          }
        }
#else
        builder.create<memref::StoreOp>(loc, elementWiseResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], Cnnz_index);
#endif

        comet_debug() << "Store C1crd\n";
        /// Branch out COO... CSR... DCSR...
        if (sparse_format.compare("COO") == 0) {
          comet_debug() << "COO format for Elementwise MulOp, update all coordinates\n";
          for (unsigned d = 0; d < rhsPerms[sparse_inputtensor_id].size(); d++) {
            Value crd = allAccessIdx[sparse_inputtensor_id][d];
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
            auto store_coo_crd = builder.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1],
                                                                 Cnnz_index);
            comet_debug() << " COO StoreOp: ";
            comet_vdump(store_coo_crd);
#else
            builder.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1], Cnnz_index);
#endif
          }
        } else if (sparse_format.compare("CSR") == 0 || sparse_format.compare("DCSR") == 0) {
          for (unsigned int d = forLoops.size() - 1; d < rhsPerms[sparse_inputtensor_id].size(); d++) {
            Value crd = allAccessIdx[sparse_inputtensor_id][d];
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
            auto store_csr_crd = builder.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1],
                                                                 Cnnz_index);
            comet_debug() << " CSR or DCSR StoreOp: ";
            comet_vdump(store_csr_crd);
#else
            builder.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1], Cnnz_index);
#endif
          }
        }

        // Update Cnnz
        comet_debug() << "Update Cnnz\n";
        Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
        Value new_Cnnz_index = builder.create<AddIOp>(loc, Cnnz_index, const_index_1);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
        comet_debug() << "AddIOps (new_Cnnz_index): ";
        comet_vdump(new_Cnnz_index);
        auto store_updated_cnnz = builder.create<memref::StoreOp>(loc, new_Cnnz_index, alloc_Cnnz,
                                                                  alloc_Cnnz_insert_loc);
        comet_debug() << " Update Cnnz (store new value) StoreOp: ";
        comet_vdump(store_updated_cnnz);
#else
        builder.create<memref::StoreOp>(loc, new_Cnnz_index, alloc_Cnnz, alloc_Cnnz_insert_loc);
#endif
      }

      // Need to identify dense tensor upperbound to be able to update Cpos and Csize arrays
      std::vector<Value> denseAllocs = tensors_rhs_Allocs[dense_inputtensor_id];
      assert(denseAllocs.size() == 1);

      comet_debug() << " DenseAllocs: ";
      auto inputType = denseAllocs[0].getType();
      std::vector<Value> denseDimsSize;
      for (unsigned rank = 0; rank < inputType.cast<mlir::MemRefType>().getRank(); rank++) {
        auto dimSize = inputType.cast<mlir::MemRefType>().getDimSize(rank);
        Value upperBound;
        if (dimSize == ShapedType::kDynamic) {
          comet_debug() << " This dimension is a dynamic size:\n";
          unsigned dynamicDimPos = inputType.dyn_cast<MemRefType>().getDynamicDimIndex(rank);
          comet_debug() << " DynamicDimPos: " << dynamicDimPos << "\n";
          upperBound = denseAllocs[0].getDefiningOp()->getOperand(dynamicDimPos);
          comet_vdump(upperBound);
        } else {
          comet_debug() << " This dimension is a static size\n";
          upperBound = builder.create<ConstantIndexOp>(loc, dimSize);
          comet_vdump(upperBound);
        }
        denseDimsSize.push_back(upperBound);
      }

      // To update Cpos
      if (sparse_format.compare("CSR") == 0) {
        builder.setInsertionPointAfter(forLoops[0]);
        Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
        Value arg0_next = builder.create<AddIOp>(loc, forLoops[1].getInductionVar(), const_index_1);
        comet_debug() << "AddIOp (arg0_next): ";
        comet_vdump(arg0_next);

        Value Cnnz_index_final = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
        builder.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][4], arg0_next); //2

        builder.setInsertionPointAfter(forLoops[1]);
        // Update C2pos[0]
        comet_debug() << "Update C2pos[0]\n";
        std::vector<Value> insert_loc_0 = {const_index_0};
        builder.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][4], insert_loc_0); //2

        // Update C1pos[0]
        comet_debug() << "Update C1pos[0]\n";
        Value dim0_index = denseDimsSize[0];
        builder.create<memref::StoreOp>(loc, dim0_index, main_tensors_all_Allocs[2][0], insert_loc_0);
      } else {
        if (sparse_format.compare("DCSR") == 0) {
          // Update C2pos
          comet_debug() << "Update DCSR C2pos\n";
          builder.setInsertionPointAfter(forLoops[0]);
          auto Cnnz_index_new = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
          auto has_nnz_row = builder.create<CmpIOp>(loc, CmpIPredicate::ne, Cnnz_index_new, Cnnz_index_old);
          auto has_nnz_row_ifOp = builder.create<scf::IfOp>(loc, has_nnz_row, /*WithElseRegion*/ false);
          comet_debug() << " If branch:\n";
          comet_vdump(has_nnz_row_ifOp);

          if (!has_nnz_row_ifOp.getThenRegion().empty()) {
            builder.setInsertionPointToStart(&has_nnz_row_ifOp.getThenRegion().front());

            Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
            Value arg0_next = builder.create<AddIOp>(loc, forLoops[1].getInductionVar(), const_index_1);
            comet_debug() << "AddIOp (arg0_next): ";
            comet_vdump(arg0_next);

            Value Cnnz_index_final = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
            builder.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][4], arg0_next); // C2pos //2
            builder.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][4], arg0_next); // C2pos //2
            Value Cnnz_row_index = builder.create<memref::LoadOp>(loc, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
            Value idx_i = allAccessIdx[sparse_inputtensor_id][0];
            builder.create<memref::StoreOp>(loc, /*i*/ idx_i, main_tensors_all_Allocs[2][1], Cnnz_row_index); // C1crd
            Value Cnnz_row_index_new = builder.create<AddIOp>(loc, Cnnz_row_index, const_index_1);
            comet_debug() << "AddIOp (Cnnz_row_index_new): ";
            comet_vdump(Cnnz_row_index_new);
            builder.create<memref::StoreOp>(loc, Cnnz_row_index_new, alloc_Cnnz_row,
                                            alloc_Cnnz_insert_loc); // Update Cnnz_row
          }

          builder.setInsertionPointAfter(forLoops[1]);
          Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
          std::vector<Value> insert_loc_1 = {const_index_1};

          // Update C2pos[0]
          std::vector<Value> insert_loc_0 = {const_index_0};
          builder.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][4], insert_loc_0); //2
          builder.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][4], insert_loc_0); //2

          // Update C1pos[0], C1pos[1]
          Value Cnnz_row_index = builder.create<memref::LoadOp>(loc, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
          builder.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][0], insert_loc_0);
          builder.create<memref::StoreOp>(loc, Cnnz_row_index, main_tensors_all_Allocs[2][0], insert_loc_1);
        } else {
          if (sparse_format.compare("COO") == 0) {
            // Finally, Update C1pos
            comet_debug() << "Update C1pos\n";
            builder.setInsertionPointAfter(forLoops[0]);
            Value Cnnz_index_final = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
            Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
            builder.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][0], const_index_0);
            builder.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][0], const_index_1);
          } else
            llvm::errs() << "// Coordinate values are not updated for output sparse tensor in " << sparse_format
                         << " format\n";
        }
      }

    } // end if (isMixedMode && isElementwise)
    else {
      // calculate elementWise operation and reduction for general dense or mix mode computation (which has dense output)
      comet_debug()
        << "calculate elementWise operation and reduction for general dense or mix mode computation (which has dense output)\n";
      Value elementWiseResult = getSemiringSecondVal(builder, loc, semiringSecond, allLoads[0], allLoads[1],
                                                     compressedWorkspace);
      Value reduceResult = getSemiringFirstVal(builder, loc, semiringFirst, allLoads[2], elementWiseResult,
                                               compressedWorkspace);
      builder.create<memref::StoreOp>(loc, reduceResult,
                                      main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1],
                                      allValueAccessIdx[2]);
    }
  }
}


/// ----------------- ///
/// Generate the new ta.sptensor_construct() using the newly allocated C_col and C_val
/// Find the ta.print().
/// Change the ta.print()'s operand to the new sparse tensor.
/// ----------------- ///
/// %55 = ta.sptensor_construct(%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %8, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
/// ----------------- ///
/**
sptensor_construct(
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
void genNewSparseTensorToPrint(OpBuilder &builder,
                               Location &loc,
                               SymbolicInfo &symbolicInfo) {

  Value &mtxC = symbolicInfo.mtxC;
  Value &mtxC_col = symbolicInfo.mtxC_col;
  Value &mtxC_val = symbolicInfo.mtxC_val;

  /// Generate the new mtxC_col and new mtxC_val bufferization.to_tensor
  Value mtxC_col_buffer = builder.create<bufferization::ToTensorOp>(loc, mtxC_col);
  Value mtxC_val_buffer = builder.create<bufferization::ToTensorOp>(loc, mtxC_val);
  
  auto sp_op = cast<tensorAlgebra::SparseTensorConstructOp>(mtxC.getDefiningOp());
  int tensorRanks = sp_op.getTensorRank();

  /// Get the operands and their types for the sparse tensor ta.sptensor_construct() (which is mtxC).
  SmallVector<Value, 20> operands;
  operands.insert(operands.end(),
                  mtxC.getDefiningOp()->getOperands().begin(),
                  mtxC.getDefiningOp()->getOperands().end());
  operands[5] = mtxC_col_buffer;  // 3 (A2crd)
  operands[8] = mtxC_val_buffer;  // 4 (AVal)
  SmallVector<Type, 20> elementTypes;
  for (Value &opd : operands) {
    elementTypes.push_back(opd.getType());
  }
  auto ty = tensorAlgebra::SparseTensorType::get(elementTypes);
  Value sptensor = builder.create<tensorAlgebra::SparseTensorConstructOp>(loc,
                                                                          ty,
                                                                          operands,
                                                                          tensorRanks);
  {
    comet_vdump(mtxC_col_buffer);
    comet_vdump(mtxC_val_buffer);
    comet_vdump(sptensor);

    {
//    comet_pdump(rootOp.getOperation()->getParentOfType<ModuleOp>());
      comet_pdump(sptensor.getDefiningOp()->getParentOfType<ModuleOp>());
    }
  }

  /// ----------------- ///
  /// Find all users of the old sparse tensor mtxC, and replace those users' corresponding operands
  /// to the new sparse tensor (sptensor). For example,
  /// "ta.print"(%mtxC)  =>  "ta.print"(%sptensor)
  /// ----------------- ///
  replaceOldValueToNewValue(mtxC, sptensor);
//  std::vector<Operation *> users;
//  for (Operation *user : mtxC.getUsers()) {
//    users.push_back(user);
//  }
//  DominanceInfo domInfo(sptensor.getDefiningOp());  // To check dominance
//  for (Operation *user : users) {
//    {
//      comet_debug() << "before replace operand.\n";
//      comet_pdump(user);
//    }
//    if (!domInfo.dominates(sptensor, user)) {
//      continue;
//    }
//    uint64_t op_i = 0;
//    for (Value op : user->getOperands()) {
//      /// Find the mtxC in the user's operands
//      if (op.getDefiningOp() == mtxC.getDefiningOp()) {
//        /// Replace the old sparse tensor to the new one
//        user->setOperand(op_i, sptensor);
//        {
//          comet_debug() << "after replace operand.\n";
//          comet_pdump(user);
//        }
//      }
//      ++op_i;
//    }
//  }
  {
    comet_vdump(sptensor.getDefiningOp()->getParentOfType<ModuleOp>());
  }

  /// ----------------- ///
  /// Surprisingly, this code snippet below does not work.
  /// When there are 2 users of mtxC, the below code can only change 1 user.
  /// My guess is that, user->setOperand(op_i, sptensor) will damage mtxC.getUsers(),
  /// making the outermost for-loop end. However, I don't know why.
  /// The solution is the code snippet above, using a vector to store all users and then iterates the vector.
  /// ----------------- ///
//  for (Operation *user : mtxC.getUsers()) {
//    uint64_t op_i = 0;
//    /// Traverse each operand of the user
//    for (Value op : user->getOperands()) {
//      /// Find the mtxC in the user's operands
//      if (op.getDefiningOp() == mtxC.getDefiningOp()) {
//        /// Replace the old sparse tensor to the new one
//        {
//          comet_debug() << "before replace operand.\n";
//          comet_pdump(user);
//        }
//        user->setOperand(op_i, sptensor);
//        {
//          comet_debug() << "after replace operand.\n";
//          comet_pdump(user);
//        }
//        break;
//      }
//      ++op_i;
//    }
//  }


  /// ----------------- ///
  /// DEPRECATED: this is too ad-hoc, only works for the ta.print operation.
  /// should be more generic, but also avoid illegal replacement.
  /// ----------------- ///
  /// this only find the specific "ta.print" operation, not all kinds of operations.
  /// This code snippet is replaced by the above one which is more generic.
  /// ----------------- ///
  /// Find the ta.print that has the old sparse tensor as its operand.
//  Operation *ta_print = nullptr;
//  for (Operation *user : mtxC.getUsers()) {
//    if (isa<tensorAlgebra::PrintOp>(user)) {
//      ta_print = user;
//      break;
//    }
//  }
////  assert(ta_print && "Need to find the ta.print that is one of the users of the old sparse tensor mtxC.\n");
//  if (ta_print) {
//    /// Set the ta.print's operand as the new sparse tensor.
//    ta_print->setOperand(0, sptensor);
//    {
//      comet_pdump(ta_print);
//    }
//  }
  /// ----------------- ///
  /// End Deprecated
  /// ----------------- ///

}



/// ----------------- ///
/// Generate Cij = Wj to gather results from the workspace to the output matrix
/// ----------------- ///
///      %rowptr_bound = memref.load %rowptr[%c0] : memref<1xindex>
///      /// sort C_col at [rowptr_start,  rowptr)
///      %C_col_ptr = memref.cast %C_col : memref<?xindex> to memref<*xindex>
///      func.call @comet_sort_index(%C_col_ptr, %rowptr_start, %rowptr_bound) : (memref<*xindex>, index, index) -> ()
///
///      /// Store results from workspace to C
///      // %rowptr_bound = memref.load %rowptr[%c0] : memref<1xindex>
///      scf.for %ptr = %rowptr_start to %rowptr_bound step %c1 {
///        %c_col_id = memref.load %C_col[%ptr] : memref<?xindex>       // c_col_id = C_col[ptr]
///        %data = memref.load %ws_data[%c_col_id] : memref<?xf64>      // data = ws_data[c_col_id]
///        memref.store %data, %C_val[%ptr] : memref<?xf64>             // C_val[ptr] = data
///        memref.store %false, %ws_bitmap[%c_col_id] : memref<?xi1>    // ws_bitmap[c_col_id] = false
///      }
/// ----------------- ///
/// tensors_rhs_Allocs[0][0] : workspace (ws_data)
/// tensors_rhs_Allocs[1][0] : mark_array
void genNumericGatherLoop(indexTree::IndexTreeComputeOp &cur_op,
                          OpsTree *opstree,
                          std::vector<std::vector<Value>> &tensors_rhs_Allocs,
                          OpBuilder &builder,
                          Location &loc,
                          SymbolicInfo &symbolicInfo,
                          NumericInfo &numericInfo) {

  /// tensors_rhs_Allocs[0][0] : workspace (ws_data)
  /// tensors_rhs_Allocs[1][0] : mark_array
  Value &ws_data = tensors_rhs_Allocs[0][0];
  Value &mark_array = tensors_rhs_Allocs[1][0];

  /// Store the insertion point
  auto last_insertion_point = builder.saveInsertionPoint();

  /// Get current for-loop
  /// Set Insertion Point before current for-loop
  scf::ForOp curr_for_loop = opstree->getParent()->getForOps()[0];
  builder.setInsertionPoint(curr_for_loop);

  /// Get parent for-loop, to get its Induction Variable which is i_idx.
  scf::ForOp parent_for_loop;
  if (!(parent_for_loop = dyn_cast<scf::ForOp>(curr_for_loop->getParentOp()))) {
    llvm::errs() << "Error: current for-loop should be inside its parent for-loop.\n";
  }
  {
    comet_vdump(curr_for_loop);
    comet_vdump(parent_for_loop);
  }

  /// Get and set the boundary of current for-loop
  /// %rowptr_start = memref::LoadOp %C_rowptr[%i_idx] : memref<?xindex>
  /// %id_idx_plus_one = arith.addi %rowptr_start, %c1 : index
  /// %rowptr_bound = memref::LoadOp %C_rowptr[%i_idx_plus_one] : memref<?xindex>
  Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
  Value i_idx = parent_for_loop.getInductionVar();
  Value i_idx_plus_one = builder.create<AddIOp>(loc, i_idx, const_index_1);
  Value &mtxC_rowptr = symbolicInfo.mtxC_rowptr;
  Value rowptr_start = builder.create<memref::LoadOp>(loc, mtxC_rowptr, ValueRange{i_idx});
  Value rowptr_bound = builder.create<memref::LoadOp>(loc, mtxC_rowptr, ValueRange{i_idx_plus_one});
  {
    comet_vdump(rowptr_start);
    comet_vdump(rowptr_bound);
  }

  /// Generate calling comet_sort_index
  /// %C_col_ptr = memref.cast %C_col : memref<?xindex> to memref<*xindex>
  /// func.call @comet_sort_index(%C_col_ptr, %rowptr_start, %rowptr_bound) : (memref<*xindex>, index, index) -> ()
  std::string func_name = "comet_sort_index";
  Value &mtxC_col = symbolicInfo.mtxC_col;
  IndexType indexType = IndexType::get(cur_op.getContext());
  Value C_col_cast = builder.create<memref::CastOp>(loc,
                                                    UnrankedMemRefType::get(indexType, 0),
                                                    mtxC_col);
  builder.create<func::CallOp>(loc,
                               func_name,
                               SmallVector<Type, 3>{},
                               ValueRange{C_col_cast, rowptr_start, rowptr_bound});


  /// Change current for-loop boundaries
  curr_for_loop.setLowerBound(rowptr_start);
  curr_for_loop.setUpperBound(rowptr_bound);

  /// Generate current for-loop body

  Value &mtxC_val = symbolicInfo.mtxC_val;
  Value rowptr = curr_for_loop.getInductionVar();
  builder.setInsertionPointToStart(curr_for_loop.getBody());
  Value c_col_id = builder.create<memref::LoadOp>(loc, mtxC_col, ValueRange{rowptr});
  Value data = builder.create<memref::LoadOp>(loc, ws_data, ValueRange{c_col_id});


#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass

  auto store_results = builder.create<memref::StoreOp>(loc,
                                                       data,
                                                       mtxC_val,
                                                       ValueRange{rowptr});
#else
  builder.create<memref::StoreOp>(loc,
                                  data,
                                  mtxC_val,
                                  ValueRange{rowptr});
#endif
  Value const_i1_0 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
  builder.create<memref::StoreOp>(loc,
                                  const_i1_0,
                                  numericInfo.ws_bitmap,
                                  ValueRange{c_col_id});
  {
    comet_vdump(c_col_id);
    comet_vdump(data);
    comet_vdump(store_results);
    comet_vdump(curr_for_loop);
  }

  /// Free up workspace (ws_data) and mark_array.
  builder.setInsertionPointAfter(parent_for_loop);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  auto deallocate_ws_data = builder.create<memref::DeallocOp>(loc, ws_data);
  auto deallocate_mark_array = builder.create<memref::DeallocOp>(loc, mark_array);
  {
    comet_vdump(deallocate_ws_data);
    comet_vdump(deallocate_mark_array);
  }
#else
  builder.create<memref::DeallocOp>(loc, ws_data);
  builder.create<memref::DeallocOp>(loc, mark_array);
#endif
  builder.create<memref::DeallocOp>(loc, numericInfo.ws_bitmap);

  /// ----------------- ///
  /// Generate the new ta.sptensor_construct() using the newly allocated C_col and C_val
  /// Find the ta.print().
  /// Change the ta.print()'s operand to the new sparse tensor.
  /// ----------------- ///
  genNewSparseTensorToPrint(builder,
                            loc,
                            symbolicInfo);

  /// Restore the insertion point
  builder.restoreInsertionPoint(last_insertion_point);
}


/// 1. Get the nested loops
/// ---1.1 the nested loops corresponding indices can be infered from ancestors_wp
/// 2. get lhs and rhs. if only 1 rhs, then it's a fill op; otherwise, binary op
/// Note: 1. The auxiliary arrays does not contain the perms/formats information
///       2. We only apply the compressed workspace on the output of the tensor, then in this case, the workspace tensors will not be in the same side with the main tensors.
///         (main tensors: such as A, B, C, w;  auxiliary tensors: such as w_index_list ...)
void genCmptOps(indexTree::IndexTreeComputeOp &cur_op,
                indexTree::IndexTreeOp &rootOp,
//                PatternRewriter &rewriter,
                OpBuilder &builder,
                OpsTree *opstree,
                std::vector<Value> &ancestorsWps,
                std::vector<Value> &wp_ops,
                SymbolicInfo &symbolicInfo,
                NumericInfo &numericInfo) {
  comet_debug() << " calling genCmptOps\n";
  Location loc = rootOp.getLoc();
  comet_debug() << " \n";

  comet_debug() << " Current IndexTreeComputeOp:";
  comet_vdump(cur_op);

  const bool comp_worksp_opt(cur_op.getCompWorkspOpt());
  comet_debug() << " comp_worksp_opt (bool: true is compressed): " << comp_worksp_opt << "\n";

  // Two cases:
  // 1. for the initial workspace, only 1 auxiliary vector w
  // 2. for the compressed workspace, there are 4 auxiliaty vectors, w, w_already_set, w_index_list, w_index_list_size

  /// The insertion location should be "the end of the body of parent loop"
  std::vector<OpsTree *> ancestorsOps;
  getAncestorsOps(opstree, ancestorsOps);
  comet_debug() << " ancestorsOps.size(): " << ancestorsOps.size() << "\n";
  for (unsigned int i = 0; i < ancestorsOps.size(); i++) {
    comet_debug() << " ancestorsOps[i]->id:" << ancestorsOps[i]->id << "\n";
  }

  /// 1. get the nested loops, from innermost to outermost order
  std::vector<scf::ForOp> nested_forops;
  std::vector<Value> nested_AccessIdx;

  for (unsigned int i = 0; i < ancestorsOps.size(); i++) {
    comet_debug() << " ancestorsOps[" << i << "]->forOps.size(): " << ancestorsOps[i]->forOps.size()
                  << ", ancestorsOps->id: "
                  << ancestorsOps[i]->id << "\n";
    if (ancestorsOps[i]->forOps.size() > 0) { // for loops OpsTree node
      for (int j = ancestorsOps[i]->forOps.size() - 1; j >= 0; j--) {
        comet_debug() << " j: " << j << "\n";
        nested_forops.push_back(ancestorsOps[i]->forOps[j]);
        comet_debug() << "AccessIdx: " << ancestorsOps[i]->accessIdx[j] << "\n";
        nested_AccessIdx.push_back(ancestorsOps[i]->accessIdx[j]);
      }
    }
  }
  comet_debug() << " nested_forops.size(): " << nested_forops.size() << "\n";
  std::vector<int64_t> nested_forops_indices;
  for (unsigned int i = 0; i < ancestorsWps.size(); i++) {
    comet_debug() << " ";
    comet_vdump(ancestorsWps[i]);

    if (indexTree::IndexTreeIndicesOp cur_op = dyn_cast<mlir::indexTree::IndexTreeIndicesOp>(
      ancestorsWps[i].getDefiningOp())) {
      // Get indices
      ArrayAttr op_indices = cur_op.getIndices();

      if (op_indices.size() > 0) { // for loops OpsTree node
        for (int j = op_indices.size() - 1; j >= 0; j--) {
          // Get the indices;
          int64_t idx = op_indices[j].cast<IntegerAttr>().getInt();
          nested_forops_indices.push_back(idx);
        }
      }
    }
  }
  comet_debug() << " nested_forops_indices.size(): " << nested_forops_indices.size() << "\n";

  assert(
    nested_forops.size() == nested_forops_indices.size() && "nested_forops.size() != nested_forops_indices.size()");

  /// Reset the insertion point: the body of the innermost loop
  assert(nested_forops.size() > 0 && "No loops\n");
  comet_debug() << " ";
  comet_pdump(nested_forops[0].getBody());
  comet_debug() << " ";
  comet_pdump(nested_forops[0].getBody()->getTerminator());
  builder.setInsertionPoint(nested_forops[0].getBody()->getTerminator());

  auto f64Type = builder.getF64Type();
//  auto indexType = IndexType::get(rootOp.getContext());

  Value const_f64_0 = builder.create<ConstantOp>(loc, f64Type, builder.getF64FloatAttr(0));
//  Value const_i1_0 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(0));
//  Type unrankedMemrefType_index = UnrankedMemRefType::get(indexType, 0);

  /// Analyze the leafop, Get the tensors, rhs, lhs, and operator_type
  /// --- only one rhs, it will be a fill op; if two, check op_type (+, +=, *=)
  /// Check the indices contained in each tensor
  /// Generate loadOp, compute ops, StoreOp.
  comet_debug() << " \n";
  // New version
  Value lhs = cur_op.getLhs().getDefiningOp()->getOperand(0);
  comet_debug() << " ";
  comet_vdump(lhs);
// lhs is TensorLoadOp
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  Value lhs_alloc = (lhs.getDefiningOp())->getOperand(0);
  comet_debug() << " ";
  comet_vdump(lhs_alloc);
#endif

  comet_debug() << " ";
  comet_vdump(cur_op);
  std::vector<Value> tensors_rhs;
  for (auto n: cur_op.getRhs()) {
    comet_debug() << " ";
    comet_vdump(n);
    for (unsigned i = 0; i < n.getDefiningOp()->getNumOperands(); i++) {
      comet_debug() << " ";
      comet_vdump(n.getDefiningOp()->getOperand(i));
      tensors_rhs.push_back(n.getDefiningOp()->getOperand(i));
    }
  }

  std::vector<Value> tensors_lhs;
  for (unsigned i = 0; i < cur_op.getLhs().getDefiningOp()->getNumOperands(); i++) {
    comet_debug() << " ";
    comet_vdump(cur_op.getLhs().getDefiningOp()->getOperand(i));
    tensors_lhs.push_back(cur_op.getLhs().getDefiningOp()->getOperand(i));
  }

  // Currently, only one case, the rhs is constant. Wj = 0.0;
  std::vector<Value> tensors_all = tensors_rhs; // Including ConstantOp
  tensors_all.insert(tensors_all.end(), tensors_lhs.begin(), tensors_lhs.end());
  comet_debug() << " tensors_all.size(): " << tensors_all.size() << "\n";

  std::vector<std::vector<Value>> tensors_lhs_Allocs = getAllAllocs(tensors_lhs);
  comet_debug() << " tensors_lhs_Allocs.size(): " << tensors_lhs_Allocs.size() << "\n";
  std::vector<std::vector<Value>> tensors_rhs_Allocs = getAllAllocs(tensors_rhs);
  comet_debug() << " tensors_rhs_Allocs.size(): " << tensors_rhs_Allocs.size() << "\n";

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  comet_debug() << " tensors_rhs_Allocs: \n";
  for (auto m: tensors_rhs_Allocs) {
    comet_debug() << " ";
    for (auto n: m) {
      comet_vdump(n);
    }
    comet_debug() << "\n";
  }
#endif

//  OpBuilder builder(cur_op);
  std::vector<std::vector<int>> allPerms;
  getPermsOfComputeOp(cur_op.getOperation()->getResult(0), allPerms);

  comet_debug() << " allPerms: \n";
  for (auto m: allPerms) {
    comet_debug() << " "; // print_vector(m);
    for (auto n: m) {
      comet_debug() << n << " ";
    }
    comet_debug() << "\n";
  }

  std::vector<std::vector<std::string>> allFormats;
  getFormatsOfComputeOp(cur_op.getOperation()->getResult(0), allFormats);
  comet_debug() << " allFormats: \n";
  for (auto m: allFormats) {
    comet_debug() << " ";
    for (auto n: m) {
      comet_debug() << n << " ";
    }
    comet_debug() << "\n";
  }

  comet_debug() << " ";
  comet_vdump(cur_op);

  std::vector<std::vector<std::string>> rhsFormats;
  getRHSFormatsOfComputeOp(cur_op.getOperation()->getResult(0), rhsFormats);

  std::vector<std::vector<std::string>> lhsFormats;
  getLHSFormatsOfComputeOp(cur_op.getOperation()->getResult(0), lhsFormats);

  assert(allPerms.size() == allFormats.size() && "allPerms.size() != allFormats.size()\n");
  for (unsigned int m = 0; m < allPerms.size(); m++) {
    assert(allPerms[m].size() == allFormats[m].size() && "allPerms[m].size() != allFormats[m].size()\n");
  }
  comet_debug() << " allPerms.size(): " << allPerms.size() << "\n";
  // tensor_nums means the actual tensors except the auxiliary tensors
  // Suppose for LHSOp, there are "n" real tensors, then allPerms[m].size()

  std::vector<std::vector<int>> allPerms_rhs;
  getRHSPermsOfComputeOp(cur_op.getOperation()->getResult(0), allPerms_rhs);
  comet_debug() << " allPerms_rhs.size(): " << allPerms_rhs.size() << "\n";
  std::vector<std::vector<int>> allPerms_lhs;
  getLHSPermsOfComputeOp(cur_op.getOperation()->getResult(0), allPerms_lhs);

  comet_debug() << " allPerms_lhs.size(): " << allPerms_lhs.size() << "\n";
  std::vector<Value> main_tensors_all;
  std::vector<Value> main_tensors_rhs;
  std::vector<Value> main_tensors_lhs;
  if (tensors_rhs.size() == allPerms_rhs.size()) { // all are "main" tensors
    main_tensors_rhs.insert(main_tensors_rhs.end(), tensors_rhs.begin(), tensors_rhs.end());
  } else {                                                                                                                                   // the rhs contains the auxiliary tensors
    assert(allPerms_rhs.size() == 1 &&
           " rhs contains auxiliary tensors and main tensors at the same time, not support currently\n"); // only 1 main tensor on rhs
    main_tensors_rhs.push_back(tensors_rhs[0]);
  }
  comet_debug() << " main_tensors_rhs.size(): " << main_tensors_rhs.size() << "\n";

  if (tensors_lhs.size() == allPerms_lhs.size()) { // all are "main" tensors
    main_tensors_lhs.insert(main_tensors_lhs.end(), tensors_lhs.begin(), tensors_lhs.end());
  } else {                                                                                                                                   // the lhs contains the auxiliary tensors
    assert(allPerms_lhs.size() == 1 &&
           " lhs contains auxiliary tensors and main tensors at the same time, not support currently\n"); // only 1 main tensor on lhs
    main_tensors_lhs.push_back(tensors_lhs[0]);
  }
  comet_debug() << " main_tensors_lhs.size(): " << main_tensors_lhs.size() << "\n";

  main_tensors_all = main_tensors_rhs;
  main_tensors_all.insert(main_tensors_all.end(), main_tensors_lhs.begin(), main_tensors_lhs.end());
  comet_debug() << " main_tensors_all.size(): " << main_tensors_all.size() << "\n";

  int main_tensor_nums = main_tensors_all.size();
  comet_debug() << " main_tensor_nums: " << main_tensor_nums << "\n";

  // Check the loop arg in each tensor
  std::vector<std::vector<Value>> main_tensors_all_Allocs = getAllAllocs(main_tensors_all);
  comet_debug() << " main_tensors_all_Allocs.size(): " << main_tensors_all_Allocs.size() << "\n";

  std::vector<std::vector<Value>> allLoopsArg(main_tensor_nums);
  std::vector<std::vector<Value>> allAccessIdx(main_tensor_nums);
  for (unsigned int i = 0; i < main_tensors_all.size(); i++) {
    for (unsigned int j = 0; j < allPerms[i].size(); j++) {
      unsigned int index_loc = findIndexInVector<int64_t>(nested_forops_indices, allPerms[i][j]);
      comet_debug() << " index_loc " << index_loc << "\n";
      comet_debug() << " Perm: " << allPerms[i][j] << "\n";
      comet_debug() << " Format: " << allFormats[i][j] << "\n";
      assert(index_loc < nested_forops.size() &&
             "index_loc < nested_forops.size(), i.e. the index not exist in nested for loop\n");
      allLoopsArg[i].push_back(nested_forops[index_loc].getInductionVar());
      allAccessIdx[i].push_back(nested_AccessIdx[index_loc]);
    }
    // Consider for the case w_index_list_size
    // if allPerms[i].size() == 0
  }

  std::vector<std::vector<Value>> allValueAccessIdx(main_tensor_nums);
  for (int i = 0; i < main_tensor_nums; i++) { // If constantOp, do not consider it
    comet_debug() << " ";
    comet_vdump(main_tensors_all[i]);
    if (main_tensors_all[i].getType().isa<tensorAlgebra::SparseTensorType>()) { // sparse tensor

      // Find the last sparse index m, then loop_arg * all dense loop args
      unsigned lastSparseIndexLoc = allPerms[i].size();
      for (int d = (int) allPerms[i].size() - 1; d >= 0; d--) {
        if (allFormats[i][d].compare(0, 1, "D") != 0 &&
            allFormats[i][d].compare(0, 1, "S") != 0) { // sparse dimension and has a loop, i.e. "CU" or "CN"
          lastSparseIndexLoc = d;
          break;
        }
      }
      // Calculate for ModeGeneric style format: [CN, S, D (, ... ) ]
      auto valueAccessIdx_part = allLoopsArg[i][lastSparseIndexLoc];
      if (lastSparseIndexLoc < allPerms[i].size() - 1) { // There is dense index after the sparse index
        unsigned int last_d = lastSparseIndexLoc + 1;
        for (unsigned int d = lastSparseIndexLoc + 1; d < allPerms[i].size(); d++) { // i=0
          if (allFormats[i][d].compare(0, 1, "D") == 0) {
            // Get dense dim size
            auto index_0 = builder.create<ConstantIndexOp>(loc, 0);
            std::vector<Value> upper_indices = {index_0};
            auto upperBound = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[i][4 * d], upper_indices);
            comet_vdump(upperBound);
            valueAccessIdx_part = builder.create<MulIOp>(loc, upperBound, valueAccessIdx_part);
            last_d = d;
          }
        }
        if (allFormats[i][last_d].compare(0, 1, "D") == 0) {
          comet_debug() << " ";
          comet_vdump(allLoopsArg[i][allLoopsArg[i].size() - 1]);
          comet_vdump(valueAccessIdx_part);
          valueAccessIdx_part = builder.create<AddIOp>(loc, allLoopsArg[i][allLoopsArg[i].size() - 1],
                                                       valueAccessIdx_part);
          comet_debug() << " AddIOps (valueAccessIdx_part): ";
          comet_vdump(valueAccessIdx_part);
        }
      }

      allValueAccessIdx[i].push_back(valueAccessIdx_part);
    } else if (main_tensors_all[i].getType().isa<TensorType>()) { // dense tensor
      allValueAccessIdx[i] = allAccessIdx[i];
    }
  }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  for (unsigned int i = 0; i < allValueAccessIdx.size(); i++) {
    comet_debug() << "allValueAccessIdx[" << i << "].size(): " << allValueAccessIdx[i].size()
                  << ", main_tensors_all_Allocs[" << i << "].size()-1: " << main_tensors_all_Allocs[i].size() - 1
                  << "\n";
  }
#endif

  int rhs_loc = 0;
  int lhs_loc = main_tensors_rhs.size();
  if (main_tensors_rhs.size() == 1) { // Generate "a = b"
    if (ConstantOp cstop = dyn_cast<ConstantOp>(main_tensors_rhs[0].getDefiningOp())) { // "a = 1.0"
      comet_debug() << " ";
      comet_vdump(cstop);
      if (comp_worksp_opt) // true attr means compressed workspace
      {

        comet_debug() << " compressed_workspace ComputeOp\n";
        std::vector<MemRefType> tensors_lhs_Allocs_type;
        for (unsigned i = 0; i < tensors_lhs_Allocs.size(); i++) {
          // only 1 alloc op for dense type tensor or constant
          assert(tensors_lhs_Allocs[i].size() == 1 && " more than 1 alloc op for dense type tensor or constant\n");
          MemRefType resultType = tensors_lhs_Allocs[i][0].getDefiningOp()->getResult(0).getType().cast<MemRefType>();
          comet_debug() << " ";
          comet_vdump(resultType);
          tensors_lhs_Allocs_type.push_back(resultType);
        }

        // Generate Store 1.0, A[...]  this op
        // this case: allPerms[0] is empty, allFormats[0] is empty
        comet_debug() << " cstop.getValue(): " << cstop.getValue() << "\n";
        comet_debug() << " ";
        comet_vdump(main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1]);
        comet_debug() << " tensors_lhs_Allocs.size(): " << tensors_lhs_Allocs.size() << "\n";
        comet_debug() << " ";

        insertInitialize(loc, cstop, main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1],
                         builder);
        comet_debug() << " ";
      } else { // initial workspace
        // Generate Store 1.0, A[...]  this op
        // this case: allPerms[0] is empty, allFormats[0] is empty
        comet_debug() << " cstop.getValue(): " << cstop.getValue() << "\n";
        comet_debug() << " ";
        comet_vdump(main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1]);

        if (allValueAccessIdx[lhs_loc].size() > 0) {
          builder.create<memref::StoreOp>(loc, cstop,
                                          main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() -
                                                                           1], allValueAccessIdx[lhs_loc]);
        } else {
          insertInitialize(loc, cstop, main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1],
                           builder);
        }
      }
    } else if (main_tensors_rhs[0].getType().isa<mlir::TensorType>()) { // Cij = Wj
      // When Cij is dense type
      if (lhs.getType().isa<mlir::TensorType>()) {
        // %1 = load b[...]
        // store %1, a[...]
        comet_debug() << " main_tensors_all_Allocs[" << rhs_loc << "].size(): "
                      << main_tensors_all_Allocs[rhs_loc].size() << ", allValueAccessIdx[" << rhs_loc << "].size(): "
                      << allValueAccessIdx[rhs_loc].size() << "\n";

        Value rhs_value = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[rhs_loc][
          main_tensors_all_Allocs[rhs_loc].size() - 1], allValueAccessIdx[rhs_loc]);
        comet_debug() << " ";
        comet_vdump(rhs_value);

        comet_debug() << " ";
        comet_vdump(main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1]);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
        auto s1 = builder.create<memref::StoreOp>(loc, rhs_value, main_tensors_all_Allocs[lhs_loc][
          main_tensors_all_Allocs[lhs_loc].size() - 1], allValueAccessIdx[lhs_loc]);
        comet_debug() << " ";
        comet_vdump(s1);
#else
        builder.create<memref::StoreOp>(loc, rhs_value, main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1], allValueAccessIdx[lhs_loc]);
#endif
      }
        // Cij = Wj
      else if (lhs.getType().isa<tensorAlgebra::SparseTensorType>()) {
        
        // Get tensor ranks
        auto sp_op = cast<tensorAlgebra::SparseTensorConstructOp>(lhs.getDefiningOp());
        int lhs_ranks = sp_op.getTensorRank();
        
        // Get tensor ranks
        auto sp_op = cast<tensorAlgebra::SparseTensorConstructOp>(lhs.getDefiningOp());
        int lhs_ranks = sp_op.getTensorRank();

        //[0...2d,2d+1...4d+1,4d+2...5d+1]
        unsigned int lhs_val_size_loc = 8 * lhs_ranks + 1;    // 17 (2d)  // 15
        unsigned int lhs_2crd_size_loc = 7 * lhs_ranks;       // 14 (2d)  // 12
        unsigned int lhs_2pos_size_loc = 7 * lhs_ranks - 1;   // 13 (2d)  // 11
        unsigned int lhs_val_size_loc = 8 * lhs_ranks + 1;    // 17 (2d)  // 15
        unsigned int lhs_2crd_size_loc = 7 * lhs_ranks;       // 14 (2d)  // 12
        unsigned int lhs_2pos_size_loc = 7 * lhs_ranks - 1;   // 13 (2d)  // 11

        // [0...2d, 2d+1...4d+1, 4d+2...5d+1]
        comet_debug() << " ";
        comet_pdump(lhs.getDefiningOp());
        comet_pdump(lhs.getDefiningOp()->getParentOp());
        comet_debug() << " ";
        comet_vdump(lhs.getDefiningOp()->getOperand(lhs_val_size_loc));

        Value lhs_nnz_operand = lhs.getDefiningOp()->getOperand(lhs_val_size_loc);
        Value lhs_nnz_op;
        comet_debug() << " ";
        comet_vdump(lhs_nnz_operand);
        if (isa<IndexCastOp>(lhs_nnz_operand.getDefiningOp())) {
          lhs_nnz_op = lhs_nnz_operand.getDefiningOp()->getOperand(0);
        } else {
          lhs_nnz_op = lhs_nnz_operand;
        }
        comet_debug() << " ";
        comet_vdump(lhs_nnz_op);
        auto lhs_nnz_load = cast<memref::LoadOp>(lhs_nnz_op.getDefiningOp());                  // index
        Value lhs_nnz_alloc = cast<memref::AllocOp>(lhs_nnz_load.getMemRef().getDefiningOp()); // index

        Value cst_0_index = builder.create<ConstantIndexOp>(loc, 0);
        Value lhs_nnz = builder.create<memref::LoadOp>(loc, lhs_nnz_alloc, ValueRange{cst_0_index});

        std::vector<Value> lhs_accessIndex = {lhs_nnz};

        Value lhs_val = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1];
        comet_debug() << " ";
        comet_vdump(lhs_val);

        if (comp_worksp_opt) // true attr means compressed workspace
        {
          /// Generate Cij = Wj to gather results from the workspace to the output matrix
          genNumericGatherLoop(cur_op,
                               opstree,
                               tensors_rhs_Allocs,
                               builder,
                               loc,
                               symbolicInfo,
                               numericInfo);

          /// ----------------- ///
          /// Backup: previous Cij = Wj
          /// Modified by Zhen Peng on 6/2/2023
          /// ----------------- ///

          /// ----------------- ///
          /// quick_sort;
          /// Cij = Wj;
          /// ----------------- ///
//
//          // Get the parent for op, change the upperbound as w_index_list_size
//          auto last_insertionPoint = builder.saveInsertionPoint();
//
//          Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
//          scf::ForOp theForop = dyn_cast<scf::ForOp>(const_index_0.getDefiningOp()->getParentOp());
//          comet_debug() << " ";
//          comet_vdump(theForop);
//
//          builder.setInsertionPoint(theForop);
//
//          Value const_index_00 = builder.create<ConstantIndexOp>(loc, 0);
//          Value w_index_list_size = builder.create<memref::LoadOp>(loc, tensors_rhs_Allocs[3][0], const_index_00);
//
//          std::string quick_sort_Str = "quick_sort";
//          Value w_index_list_cast = builder.create<memref::CastOp>(loc, unrankedMemrefType_index,
//                                                                   tensors_rhs_Allocs[2][0]);
//          builder.create<func::CallOp>(loc, quick_sort_Str, SmallVector<Type, 2>{},
//                                       ValueRange{w_index_list_cast, w_index_list_size});
//
//          theForop.setUpperBound(w_index_list_size);
//          comet_debug() << " ";
//          comet_vdump(theForop);
//
//          builder.restoreInsertionPoint(last_insertionPoint);
//          Value crd_index = builder.create<memref::LoadOp>(loc, tensors_rhs_Allocs[2][0], theForop.getInductionVar());
//          Value c_value = builder.create<memref::LoadOp>(loc, tensors_rhs_Allocs[0][0], crd_index);
//          // Fill CVal
//          builder.create<memref::StoreOp>(loc, c_value, lhs_val, ValueRange{lhs_nnz});
//
//          /// Removed by Zhen Peng on 6/6/2023
//          /// mark_array does not need to reset now.
////          // w_already_set[crd_j] = 0
////          rewriter.create<memref::StoreOp>(loc, const_i1_0, tensors_rhs_Allocs[1][0], ValueRange{crd_index});
//
//          comet_debug() << " lhs_loc: " << lhs_loc << "\n";
//          comet_debug() << " format: " << allFormats[lhs_loc][allFormats[lhs_loc].size() - 1] << "\n";
//          if (allFormats[lhs_loc][allFormats[lhs_loc].size() - 1].compare(0, 2, "CU") == 0) {
//            Value lhs_2crd = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 2];
//            comet_debug() << " ";
//            comet_vdump(lhs_2crd);
//
//            builder.create<memref::StoreOp>(loc, crd_index, lhs_2crd, ValueRange{lhs_nnz});
//          }
//
//          comet_debug() << "\n";
//          Value cst_1_index = builder.create<ConstantIndexOp>(loc, 1);
//          comet_debug() << " ";
//          comet_vdump(lhs_nnz);
//          Value lhs_nnz_new = builder.create<AddIOp>(loc, lhs_nnz, cst_1_index);
//          comet_debug() << " AddIOps (lhs_nnz_new): ";
//          comet_vdump(lhs_nnz_new);
//          comet_debug() << " ";
//          comet_vdump(lhs_nnz_alloc);
//
//          builder.create<memref::StoreOp>(loc, lhs_nnz_new, lhs_nnz_alloc, ValueRange{cst_0_index});
//
//          Value lhs_2crd = lhs.getDefiningOp()->getOperand(lhs_2crd_size_loc);
//          Value lhs_2crd_op;
//          comet_vdump(lhs_2crd);
//          if (isa<IndexCastOp>(lhs_2crd.getDefiningOp())) {
//            lhs_2crd_op = lhs_2crd.getDefiningOp()->getOperand(0);
//          } else {
//            lhs_2crd_op = lhs_2crd;
//          }
//          comet_debug() << " ";
//          comet_vdump(lhs_2crd_op);
//          auto c2crd_size_load = cast<memref::LoadOp>(lhs_2crd_op.getDefiningOp());                    // index
//          Value c2crd_size_alloc = cast<memref::AllocOp>(c2crd_size_load.getMemRef().getDefiningOp()); // index
//          comet_debug() << " ";
//          comet_vdump(c2crd_size_alloc);
//
//          builder.create<memref::StoreOp>(loc, lhs_nnz_new, c2crd_size_alloc, ValueRange{cst_0_index});
//
//          // Fill C2pos
//          comet_debug() << " \n";
//          auto prev_forop = nested_forops[nested_forops.size() - 1 - 1];
//          builder.setInsertionPointAfter(prev_forop);
//
//          Value lhs_2pos_0 = lhs.getDefiningOp()->getOperand(lhs_2pos_size_loc);
//          Value lhs_2pos_op;
//          comet_debug() << " ";
//          comet_vdump(lhs_2pos_0);
//          if (isa<IndexCastOp>(lhs_2pos_0.getDefiningOp())) {
//            lhs_2pos_op = lhs_2pos_0.getDefiningOp()->getOperand(0);
//          } else {
//            lhs_2pos_op = lhs_2pos_0;
//          }
//          comet_debug() << " ";
//          comet_vdump(lhs_2pos_op);
//          auto c2pos_size_load = cast<memref::LoadOp>(lhs_2pos_op.getDefiningOp());                    // index
//          Value c2pos_size_alloc = cast<memref::AllocOp>(c2pos_size_load.getMemRef().getDefiningOp()); // index
//          Value cst_index_000 = builder.create<ConstantIndexOp>(loc, 0);
//          Value c2pos_size_value = builder.create<memref::LoadOp>(loc, c2pos_size_alloc, ValueRange{cst_index_000});
//
//          Value c2crd_size_nnz = builder.create<memref::LoadOp>(loc, c2crd_size_alloc, ValueRange{cst_index_000});
//
//          // store crd_size into pos
//          Value lhs_2pos = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 3];
//          comet_debug() << " ";
//          comet_vdump(lhs_2pos);
//          builder.create<memref::StoreOp>(loc, c2crd_size_nnz, lhs_2pos, ValueRange{c2pos_size_value});
//
//          Value cst_index_1 = builder.create<ConstantIndexOp>(loc, 1);
//          comet_debug() << " ";
//          comet_vdump(c2pos_size_value);
//          Value c2pos_size_value_new = builder.create<AddIOp>(loc, c2pos_size_value, cst_index_1);
//          comet_debug() << " AddIOps (c2pos_size_value_new): ";
//          comet_vdump(c2pos_size_value_new);
//
//          builder.create<memref::StoreOp>(loc, c2pos_size_value_new, c2pos_size_alloc, ValueRange{cst_index_000});

          /// ----------------- ///
          /// End Backup: previous Cij = Wj
          /// ----------------- ///
        } else {

          // %1 = load b[...]
          // if(%1 != 0) {
          //    Cnnz = load Cop.operand(4d+1)
          //    store %1, cval[Cnnz]
          //    store Cnnz+1, Cop.operand(4d+1)
          // }
          comet_debug() << " main_tensors_all_Allocs[" << rhs_loc << "].size(): "
                        << main_tensors_all_Allocs[rhs_loc].size() << ", allValueAccessIdx[" << rhs_loc
                        << "].size(): " << allValueAccessIdx[rhs_loc].size() << "\n";
          Value rhs_value = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[rhs_loc][
            main_tensors_all_Allocs[rhs_loc].size() - 1], allValueAccessIdx[rhs_loc]);
          comet_debug() << " ";
          comet_vdump(rhs_value);
          Value isNonzero = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, rhs_value, const_f64_0);
          comet_debug() << " ";
          comet_vdump(isNonzero);
          auto if_nonzero = builder.create<scf::IfOp>(loc, isNonzero, /*WithElseRegion*/ false);
          comet_debug() << " If branch:\n";
          comet_vdump(if_nonzero);

          if (!if_nonzero.getThenRegion().empty()) {
            auto last_insertionPoint = builder.saveInsertionPoint();
            builder.setInsertionPointToStart(&if_nonzero.getThenRegion().front());

            builder.create<memref::StoreOp>(loc, rhs_value, lhs_val, lhs_accessIndex);

            /// update pos/crd arrays
            // Fill C2crd in CSR format, parent loop's accessIdx
            /// Check format j in the output
            if (allFormats[lhs_loc][allFormats[lhs_loc].size() - 1].compare(0, 2, "CU") == 0) {
              Value crd_index = allAccessIdx[allAccessIdx.size() - 1][allAccessIdx[allAccessIdx.size() - 1].size() -
                                                                      1];
              comet_debug() << " ";
              comet_vdump(crd_index);
              Value lhs_2crd = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 4];   //-2
              Value lhs_2crd = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 4];   //-2
              comet_debug() << " ";
              comet_vdump(lhs_2crd);

              builder.create<memref::StoreOp>(loc, crd_index, lhs_2crd, lhs_accessIndex);
            }

            comet_debug() << "\n";
            Value cst_1_index = builder.create<ConstantIndexOp>(loc, 1);
            comet_debug() << " ";
            comet_vdump(lhs_nnz);
            Value lhs_nnz_new = builder.create<AddIOp>(loc, lhs_nnz, cst_1_index);
            comet_debug() << " AddIOps: (lhs_nnz_new)";
            comet_vdump(lhs_nnz_new);
            comet_debug() << " ";
            comet_vdump(lhs_nnz_alloc);

            builder.create<memref::StoreOp>(loc, lhs_nnz_new, lhs_nnz_alloc, ValueRange{cst_0_index});

            comet_debug() << "\n";
            Value lhs_2crd = lhs.getDefiningOp()->getOperand(lhs_2crd_size_loc);
            Value lhs_2crd_op;
            comet_vdump(lhs_2crd);
            if (isa<IndexCastOp>(lhs_2crd.getDefiningOp())) {
              lhs_2crd_op = lhs_2crd.getDefiningOp()->getOperand(0);
            } else {
              lhs_2crd_op = lhs_2crd;
            }
            comet_debug() << " ";
            comet_vdump(lhs_2crd_op);
            // unsigned int lhs_2crd_size_loc = 4*lhs_ranks;
            auto c2crd_size_load = cast<memref::LoadOp>(lhs_2crd_op.getDefiningOp());                    // index
            Value c2crd_size_alloc = cast<memref::AllocOp>(c2crd_size_load.getMemRef().getDefiningOp()); // index
            comet_debug() << " ";
            comet_vdump(c2crd_size_alloc);

            builder.create<memref::StoreOp>(loc, lhs_nnz_new, c2crd_size_alloc, ValueRange{cst_0_index});

            comet_debug() << " \n";
            builder.restoreInsertionPoint(last_insertionPoint);
          }

          comet_debug() << " \n";
          auto prev_forop = nested_forops[nested_forops.size() - 1 - 1];
          builder.setInsertionPointAfter(prev_forop);

          comet_debug() << " ";
          comet_vdump(lhs.getDefiningOp()->getOperand(lhs_2pos_size_loc));
          Value lhs_2pos_0 = lhs.getDefiningOp()->getOperand(lhs_2pos_size_loc);
          Value lhs_2pos_op;
          comet_vdump(lhs_2pos_0);
          if (isa<IndexCastOp>(lhs_2pos_0.getDefiningOp())) {
            lhs_2pos_op = lhs_2pos_0.getDefiningOp()->getOperand(0);
          } else {
            lhs_2pos_op = lhs_2pos_0;
          }
          comet_debug() << " ";
          comet_vdump(lhs_2pos_op);
          auto c2pos_size_load = cast<memref::LoadOp>(lhs_2pos_op.getDefiningOp());                    // index
          Value c2pos_size_alloc = cast<memref::AllocOp>(c2pos_size_load.getMemRef().getDefiningOp()); // index
          Value cst_0_index = builder.create<ConstantIndexOp>(loc, 0);
          Value c2pos_size_value = builder.create<memref::LoadOp>(loc, c2pos_size_alloc, ValueRange{cst_0_index});

          Value lhs_2crd = lhs.getDefiningOp()->getOperand(lhs_2crd_size_loc);
          Value lhs_2crd_op;
          comet_vdump(lhs_2crd);
          if (isa<IndexCastOp>(lhs_2crd.getDefiningOp())) {
            lhs_2crd_op = lhs_2crd.getDefiningOp()->getOperand(0);
          } else {
            lhs_2crd_op = lhs_2crd;
          }
          comet_debug() << " ";
          comet_vdump(lhs_2crd_op);
          auto c2crd_size_load = cast<memref::LoadOp>(lhs_2crd_op.getDefiningOp());                    // index
          Value c2crd_size_alloc = cast<memref::AllocOp>(c2crd_size_load.getMemRef().getDefiningOp()); // index
          Value c2crd_size_nnz = builder.create<memref::LoadOp>(loc, c2crd_size_alloc, ValueRange{cst_0_index});

          // store crd_size into pos
          Value lhs_2pos = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 5]; // -3
          comet_debug() << " ";
          comet_vdump(lhs_2pos);

          builder.create<memref::StoreOp>(loc, c2crd_size_nnz, lhs_2pos, ValueRange{c2pos_size_value});

          Value cst_1_index = builder.create<ConstantIndexOp>(loc, 1);
          comet_debug() << " ";
          comet_vdump(c2pos_size_value);
          Value c2pos_size_value_new = builder.create<AddIOp>(loc, c2pos_size_value, cst_1_index);
          comet_debug() << " AddIOps (c2pos_size_value_new): ";
          comet_vdump(c2pos_size_value_new);

          builder.create<memref::StoreOp>(loc, c2pos_size_value_new, c2pos_size_alloc, ValueRange{cst_0_index});
        }
      }
    }
      // Vj = Bij
    else if (main_tensors_rhs[0].getType().isa<tensorAlgebra::SparseTensorType>()) {
      // %Bvalue = load %Bval[..]
      // store %Bvalue, %v[%j]
      std::vector<Value> allLoads(main_tensor_nums);
      for (auto m = 0; m < main_tensor_nums; m++) {
        Value s = builder.create<memref::LoadOp>(loc,
                                                 main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1],
                                                 allValueAccessIdx[m]);
        allLoads[m] = s;
        comet_debug() << " ";
        comet_vdump(s);
      }
      comet_debug() << " allLoads.size(): " << allLoads.size() << "\n";

      builder.create<memref::StoreOp>(loc, allLoads[0],
                                      main_tensors_all_Allocs[1][main_tensors_all_Allocs[1].size() - 1],
                                      allValueAccessIdx[1]);
    }
  } else if (main_tensors_rhs.size() == 2) { // Generate " a = b * c" binary op

    comet_debug() << "No masking codegen...\n";

    auto semiringParts = cur_op.getSemiring().split('_');
    // check validity of semiring provided by user.
    if (!Semiring_reduceOps.contains(semiringParts.first) || !Semiring_ops.contains(semiringParts.second)) {
      llvm::errs() << "Not supported semiring operator: "
                   << semiringParts.first << " or " << semiringParts.second << " \n";
      llvm::errs() << "Please report this error to the developers!\n";
      // we should not proceed forward from this point to avoid faults.
      assert(false && "Not supported semiring operator");
    }

    MaskingInfo masking_info;
    masking_info.mask_type = NO_MASKING;
    if (symbolicInfo.is_SpGEMM && comp_worksp_opt) {

      /// cur_op is the compute node
      /// %41 = "it.Compute"(%39, %40) {comp_worksp_opt = true, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
      genSymbolicPhase(cur_op,
                       opstree,
                       wp_ops,
                       masking_info,
                       builder,
                       loc,
                       symbolicInfo /* updated after call */);

      /// Insert a row_offset before the outermost numeric for-loop.
      /// Insert row_offset = C_rowptr[i_idx] at the begninning of the body of the outmost numeric for-loop.
      /// Store row_offset in SymbolicInfo
      insertRowOffsetFromMatrixCRowptr(nested_forops,
                                       symbolicInfo,
                                       builder,
                                       loc);
    }



    formSemiringLoopBody(comp_worksp_opt,
                         semiringParts.first, semiringParts.second,
                         builder, loc, lhs_loc,
                         main_tensors_all_Allocs,
                         tensors_lhs_Allocs,
                         tensors_rhs_Allocs,
                         allValueAccessIdx,
                         allAccessIdx,
                         nested_forops,
                         allPerms_rhs,
                         rhsFormats,
                         lhsFormats,
                         symbolicInfo,
                         numericInfo,
                         masking_info);
  } else if (main_tensors_rhs.size() == 3) { // Generate " a<m> = b * c" binary op with masking

    {
//    comet_pdump(rootOp.getOperation()->getParentOfType<ModuleOp>());
      comet_pdump(rootOp->getParentOfType<ModuleOp>());
    }
    auto semiringParts = cur_op.getSemiring().split('_');
    // check validity of semiring provided by user.
    if (!Semiring_reduceOps.contains(semiringParts.first) || !Semiring_ops.contains(semiringParts.second)) {
      llvm::errs() << "Not supported semiring operator: "
                   << semiringParts.first << " or " << semiringParts.second << " \n";
      llvm::errs() << "Please report this error to the developers!\n";
      // we should not proceed forward from this point to avoid faults.
      assert(false && "Not supported semiring operator");
    }



    auto maskingAttr = cur_op.getMaskType();
    std::string maskingAttrStr (maskingAttr.data());
    comet_debug() << "mask attr: " << maskingAttrStr << "\n";

    MASKING_TYPE mask_type;
    if (maskingAttrStr == "push")
      mask_type = MASKING_TYPE::PUSH_BASED_MASKING;
    else if (maskingAttrStr == "pull")
      mask_type = MASKING_TYPE::PULL_BASED_MASKING;
    else if (maskingAttrStr == "auto")
      mask_type = MASKING_TYPE::PUSH_BASED_MASKING;
    else // none
      mask_type = MASKING_TYPE::NO_MASKING;

    switch (mask_type) {
      case NO_MASKING: {  /// Use no masking; we should not hit this case because it is handled
                          /// by the previous if-else branch when main_tensors_rhs.size() == 2
        break;
      }
      case PUSH_BASED_MASKING: {  /// Use push-based masking
//        mlir::Value states; /// The temporary dense vector for push-based masking
        /// mask_tensor should be the 3rd operand of ComputeRHS (tensors_rhs[2]).
        mlir::Value mask_tensor = tensors_rhs[2];
        {
          comet_debug() << "mask_tensor\n";
          comet_vdump(mask_tensor);
        }

        MaskingInfo masking_info;
        masking_info.mask_type = PUSH_BASED_MASKING;
        masking_info.mask_tensor = mask_tensor;

        /// Get mask_rowptr, mask_col, and mask_val arrays
        getMaskSparseTensorInfo(masking_info /* contents updated after call*/);

//        masking_info.states = states;
        if (symbolicInfo.is_SpGEMM && comp_worksp_opt) {

          /// cur_op is the compute node
          /// %41 = "it.Compute"(%39, %40) {comp_worksp_opt = true, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
          genSymbolicPhase(cur_op,
                           opstree,
                           wp_ops,
                           masking_info,
                           builder,
                           loc,
                           symbolicInfo /* updated after call */);

          /// Insert a row_offset before the outermost numeric for-loop.
          /// Insert row_offset = C_rowptr[i_idx] at the begninning of the body of the outmost numeric for-loop.
          /// Store row_offset in SymbolicInfo
          insertRowOffsetFromMatrixCRowptr(nested_forops,
                                           symbolicInfo,
                                           builder,
                                           loc);
        }
        formSemiringLoopBody(comp_worksp_opt,
                             semiringParts.first, semiringParts.second,
                             builder, loc, lhs_loc,
                             main_tensors_all_Allocs,
                             tensors_lhs_Allocs,
                             tensors_rhs_Allocs,
                             allValueAccessIdx,
                             allAccessIdx,
                             nested_forops,
                             allPerms_rhs,
                             rhsFormats,
                             lhsFormats,
                             symbolicInfo,
                             numericInfo,
                             masking_info);
        break;
      }
      case PULL_BASED_MASKING:  /// Use pull-based masking
        llvm::errs() << "Error: mask type PULL_BASED_MASKING is not supported, yet.\n";
        assert(false && "Not supported mask type.");
    }
  } else {
    llvm::errs() << "No support for operation with greater than two operands in workspace transforms!"
                 << "\n";
  }
}

/// ----------------- ///
/// Get the itree roots
/// ----------------- ///
void getIndexTreeOps(func::FuncOp &function,
                     std::vector<indexTree::IndexTreeOp> &iTreeRoots /* output */) {
  function.walk([&](indexTree::IndexTreeOp op) {
    iTreeRoots.push_back(op);
  });
}


/// ----------------- ///
/// Check if the compute node is doing SpGEMM
/// ----------------- ///
bool checkIfSpGEMM(indexTree::IndexTreeComputeOp &cur_op) {
  std::vector< std::vector<std::string> > opFormats;
  getRHSFormatsOfComputeOp(cur_op, opFormats);

  /// Condition: SpGEMM has 2 or 3 operands in RHS (i.e., two input matrices, or plus one mask)
  if (!(opFormats.size() == 2 /* no mask */ || opFormats.size() == 3 /* mask */)) {
    return false;
  }

  /// Condition: SpGEMM's two input matrices are both 2D matrices
  if (!(opFormats[0].size() == 2 && opFormats[1].size() == 2)) {
    return false;
  }

  /// Condition: SpGEMM's two input matrices are both in CSR format.
  if(!(opFormats[0][0] == "D" && opFormats[0][1] == "CU" && \
       opFormats[1][0] == "D" && opFormats[1][1] == "CU")) {
    return false;
  }

  return true;
}

/// ----------------- ///
/// Delete every objects in opstree_vec, preventing memory leak.
/// ----------------- ///
void cleanOpstreeVec(std::vector<OpsTree *> &opstree_vec) {
  for (auto &t : opstree_vec) {
    delete t;
  }
}


//===----------------------------------------------------------------------===//
// LowerIndexTreeIRToSCF PASS
//===----------------------------------------------------------------------===//

/// Lower the ta.tc (tensor contraction operation in TA dialect) into scf dialect.
  struct LowerIndexTreeToSCFPass
    : public PassWrapper<LowerIndexTreeToSCFPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerIndexTreeToSCFPass)
    void runOnOperation() override;

    void doLoweringIndexTreeToSCF(indexTree::IndexTreeOp &rootOp,
                                  OpBuilder &builder);
  };


} // end anonymous namespace.


/**
 * @brief :
 * Goal: IndexTreeOp(i.e. a tree structure), convert into OpsTree(also tree structure)
 * Steps: 1.Iterate over IndexTreeOptree
 *        2.pass info to opsGen(), including tensors, current workspacetreeop, parent OpsTree node
 *          -- the parent of "current workspacetreeop" can get from getUser(). Only one user(tree structure)
 *          -- DFS traverse the workspacetreeop. How?
 * */
void LowerIndexTreeToSCFPass::doLoweringIndexTreeToSCF(indexTree::IndexTreeOp &rootOp,
                                                       OpBuilder &builder) {
  assert(isa<indexTree::IndexTreeOp>(rootOp));
  comet_debug() << "\ndoLoweringIndexTreeToSCF in LowerIndexTreeIRToSCF\n";
    // auto module = rootOp->getParentOfType<ModuleOp>();
  {
//    comet_pdump(rootOp.getOperation()->getParentOfType<ModuleOp>());
    comet_pdump(rootOp->getParentOfType<ModuleOp>());
  }


//  comet_pdump(rootOp.getOperation()->getParentOp());
  // Here, should check the operands, at least one operand should be sparse;
  // Otherwise, if all dense operands, just return.
  // rootOp only contains one workspace child, no indices

  std::vector<mlir::Value> wp_ops;
  dfsRootOpTree(rootOp.getChildren(), wp_ops);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  comet_debug() << " wp_ops.size(): " << wp_ops.size() << "\n";
      for (auto n : wp_ops)
      {
        comet_debug() << " ";
        comet_vdump(n);
        // Declare opsTree
      }
#endif

  // In ops vector, for each op, the parent of each op can get from getUsers()
  // Since it's a tree structure, only one user ==> which is the parent
  // We can initialize the OpsTree structure with this relationship.
  // Search the location of the parent of current op, if rootOp, return ops.size;
  // Otherwise, return the location index.
  std::vector<unsigned int> parent_idx;
  for (unsigned int i = 0; i < wp_ops.size(); i++)
  {
    mlir::Value wp_op = wp_ops[i];
    mlir::Value wp_parent;

    for (auto n : wp_op.getDefiningOp()->getUsers())
    {
      comet_debug() << " " << i << " ";
      comet_pdump(n);
      wp_parent = n->getResult(0);

      comet_debug() << " parent: " << findIndexInVector_Value(wp_ops, wp_parent) << "\n";
      bool isInTree = false;
      if (findIndexInVector_Value(wp_ops, wp_parent) < wp_ops.size())
      {
        isInTree = true;
      }

      if (isInTree || isRealRoot(wp_op.getDefiningOp()))
        parent_idx.push_back(findIndexInVector_Value(wp_ops, wp_parent));
    }
  }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  comet_debug() << " parent_idx: " << parent_idx.size() << "\n";
      for (auto n : parent_idx)
      {
        comet_debug() << " " << n << " \n";
        // Declare opsTree
      }
#endif

  std::vector<OpsTree *> opstree_vec;
  for (unsigned int i = 0; i < wp_ops.size(); i++)
  {
    std::vector<scf::ForOp> forOps;
    std::vector<Value> accessIdx;
//    std::vector<Value> cmptOps;

    OpsTree *parent = nullptr;
    if (i >= 1)
    { // Not rootop
      parent = opstree_vec[parent_idx[i]];
    }
    comet_debug() << " \n";
    OpsTree *ops = new OpsTree(forOps, accessIdx, parent, i);
    if (parent != nullptr)
    { // add child to the parent
      parent->addChild(ops);
    }

    opstree_vec.push_back(ops);
  }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  {
    int opstree_i = 0;
    for (auto n: opstree_vec) {
      comet_debug() << " " << n->id << "\n";
      comet_debug() << "opstree_vec[" << opstree_i << "] " \
              << "forOps.size():" << n->forOps.size() << " " \
              << "accessIdx.size():" << n->accessIdx.size() << "\n";
//              << "cmptOps.size():" << n->cmptOps.size() << "\n";
      if (n->parent != nullptr) {
        comet_debug() << "parent: " << n->parent->id << "\n";
      } else {
        comet_debug() << "parent: null \n";
      }
      ++opstree_i;
    }
  }
#endif

  /// ----------------- ///
  /// Added by Zhen Peng on 07/19/2023
  /// ----------------- ///
//  bool is_SpGEMM = false;
  SymbolicInfo symbolicInfo;
  NumericInfo numericInfo;
  {//test
    comet_debug() << "Before generating nodes\n";
    comet_pdump(rootOp.getOperation()->getParentOp());
  }
  for (unsigned int i = 0; i < wp_ops.size(); i++)
  {
    comet_debug() << " i: " << i << "\n";
    comet_vdump(wp_ops[i]);
    if (indexTree::IndexTreeIndicesOp cur_op = dyn_cast<mlir::indexTree::IndexTreeIndicesOp>(wp_ops[i].getDefiningOp()))
    {
// Get indices
      ArrayAttr op_indices = cur_op.getIndices();
      comet_debug() << "curOp is IndexTreeIndicesOp\n";
      comet_vdump(cur_op);

      std::vector<int> indices;
      for (unsigned int j = 0; j < op_indices.size(); j++)
      {
// Get the indices;
        int idx = op_indices[j].cast<IntegerAttr>().getInt();
        indices.push_back(idx);
      }
      comet_debug() << " indices.size(): " << indices.size() << "\n";

      std::vector<Value> leafs;
      findLeafs(cur_op, indices, wp_ops, leafs);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      comet_debug() << " leafs.size(): " << leafs.size() << "\n";
          for (auto n : leafs)
          {
            comet_debug() << " ";
            comet_vdump(n);
          }
#endif

      std::vector<Value> tensors;
      std::vector<unsigned int> ids;
      std::vector<std::string> formats;

      comet_debug() << " ";
      comet_vdump(cur_op);

      getFormatsInfo(cur_op, indices, leafs, tensors, ids, formats);

      comet_debug() << " indices.size(): " << indices.size() << " tensors.size(): " << tensors.size() << "\n";
      for (unsigned int m = 0; m < tensors.size(); m++)
      {
        comet_debug() << " Formats:" << formats[m] << " " << ids[m] << " ";
        comet_vdump(tensors[m]);
      }

      /// ----------------- ////
      /// Removed by Zhen Peng on 07/19/2023
      /// Dead code. ancestors_wp is never used.
      /// ----------------- ////
///// Generate loops
//      std::vector<Value> ancestors_wp; // workspace tree ancestor
//      getAncestorsWp(cur_op, ancestors_wp, wp_ops);
//#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
//      comet_debug() << " Current Op (IndexTreeIndicesOp): ";
//          comet_vdump(cur_op);
//          for (auto n : ancestors_wp)
//          {
//            comet_debug() << " ancestors_wp:";
//            comet_vdump(n);
//          }
//#endif

      comet_debug() << " call genForOps, i = " << i << "\n";
//      genForOps(tensors, ids, formats, rootOp, rewriter, opstree_vec[i], ancestors_wp);
      genForOps(tensors, ids, formats, rootOp, builder, opstree_vec[i]);
      comet_debug() << " finished call genForOps, i = " << i << "\n";
    }
    else if (indexTree::IndexTreeComputeOp cur_op = dyn_cast<mlir::indexTree::IndexTreeComputeOp>(wp_ops[i].getDefiningOp()))
    {
// Generate computation ops.
      std::vector<Value> ancestors_wp; // workspace tree ancestor
      getAncestorsWp(cur_op, ancestors_wp, wp_ops);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      comet_debug() << " Current Op (IndexTreeComputeOp):";
      comet_vdump(cur_op);
      for (auto n : ancestors_wp)
      {
        comet_debug() << " ";
        comet_vdump(n);
      }
#endif

      comet_debug() << " call genCmptOps, i = " << i << "\n";
      /// ----------------- ///
      /// Added by Zhen Peng on 07/19/2023
      /// ----------------- ///
      if (!symbolicInfo.is_SpGEMM) {
        symbolicInfo.is_SpGEMM = checkIfSpGEMM(cur_op);
        comet_debug() << "is_SpGEMM: " << symbolicInfo.is_SpGEMM << "\n";
      }
      {//
        comet_pdump(cur_op.getOperation()->getParentOp());
      }
      // ancestors_wp can give all the indices of the nested loops
//      genCmptOps(cur_op, rootOp, rewriter, opstree_vec[i], ancestors_wp);
      genCmptOps(cur_op, rootOp, builder, opstree_vec[i], ancestors_wp,
                 wp_ops, symbolicInfo, numericInfo);
      comet_debug() << " finished call genCmptOps, i = " << i << "\n";
    }
  }

//  {
//    comet_pdump(rootOp->getParentOfType<ModuleOp>());
//    comet_pdump(rootOp.getOperation()->getParentOp());
//  }

  {
    comet_debug() << "End of doLoweringIndexTreeToSCF()\n";
//    comet_pdump(rootOp.getOperation()->getParentOfType<ModuleOp>());
    comet_pdump(rootOp->getParentOfType<ModuleOp>());
  }

  comet_debug() << "Cleaning up IndexTree Operations\n";
  comet_vdump(rootOp);
  std::vector<Operation *> operations_dumpster;
//  rewriter.eraseOp(rootOp);
  rootOp.erase();
  for (auto itOp : wp_ops)
  {
    if (indexTree::IndexTreeComputeOp cur_op = dyn_cast<mlir::indexTree::IndexTreeComputeOp>(itOp.getDefiningOp()))
    {
      comet_pdump(itOp.getDefiningOp()->getOperand(0).getDefiningOp());
      comet_pdump(itOp.getDefiningOp()->getOperand(1).getDefiningOp());
//      rewriter.eraseOp(itOp.getDefiningOp()->getOperand(0).getDefiningOp()); //RHS
//      rewriter.eraseOp(itOp.getDefiningOp()->getOperand(1).getDefiningOp()); //LHS
//      cur_op.getOperand(0).getDefiningOp()->erase();
//      cur_op->getOperand(1).getDefiningOp()->erase();
      operations_dumpster.push_back(cur_op.getOperand(0).getDefiningOp());
      operations_dumpster.push_back(cur_op.getOperand(1).getDefiningOp());
    }
    comet_pdump(itOp.getDefiningOp());
//    rewriter.eraseOp(itOp.getDefiningOp());
    itOp.getDefiningOp()->erase();
  }
  for (auto op : operations_dumpster) {
    op->erase();
  }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  {
    int opstree_i = 0;
    for (auto n: opstree_vec) {
      comet_debug() << " " << n->id << "\n";
      comet_debug() << "opstree_vec[" << opstree_i << "] " \
              << "forOps.size():" << n->forOps.size() << " " \
              << "accessIdx.size():" << n->accessIdx.size() << "\n";
//              << "cmptOps.size():" << n->cmptOps.size() << "\n";
      if (n->parent != nullptr) {
        comet_debug() << "parent: " << n->parent->id << "\n";
      } else {
        comet_debug() << "parent: null \n";
      }
      ++opstree_i;
    }
  }
#endif
  /// ----------------- ///
  /// Free the memory occupied by each element in opstree_vec.
  /// ----------------- ///
  cleanOpstreeVec(opstree_vec);

  {
    comet_debug() << "After doLoweringIndexTreeToSCF.\n";
//    comet_pdump(rootOp.getOperation()->getParentOfType<ModuleOp>());
//    comet_pdump(rootOp->getParentOfType<ModuleOp>());
//    comet_pdump(rootOp.getOperation()->getParentOp());
//    auto func = rootOp->getParentOfType<func::FuncOp>();
//    for (Block &B : func.getBody()) {
//      for (Operation &op: B) {
//        comet_vdump(op);
//      }
//    }
  }
//  {
//    comet_debug() << "End of doLoweringIndexTreeToSCF()\n";
////    comet_pdump(rootOp.getOperation()->getParentOfType<ModuleOp>());
//    comet_pdump(rootOp->getParentOfType<ModuleOp>());
//  }
}  /// End doLoweringIndexTreeToSCF()

void LowerIndexTreeToSCFPass::runOnOperation()
{
  comet_debug() << "LowerIndexTreeToSCFPass\n";
  func::FuncOp function = getOperation();
  auto module = function.getOperation()->getParentOfType<ModuleOp>();
  auto *ctx = &getContext();

  /// ----------------- ////
  /// Backup: add function declaration "quick_sort".
  /// ----------------- ////
//  IndexType indexType = IndexType::get(ctx);
//  auto quickSortFunc = FunctionType::get(ctx, {mlir::UnrankedMemRefType::get(indexType, 0), indexType}, {});
//
//  if (!hasFuncDeclaration(module, "quick_sort"))
//  {
//    mlir::func::FuncOp func1 = mlir::func::FuncOp::create(function.getLoc(), "quick_sort",
//                                                          quickSortFunc, ArrayRef<NamedAttribute>{});
//    func1.setPrivate();
//    module.push_back(func1);
//  }

  /// Declare comet_sort_index()
  /// func.func private @comet_sort_index(memref<*xindex>, index, index)
  declareSortFunc(module,
                  ctx,
                  function.getLoc());

//  indexTree::IndexTreeOp iTreeRoot;
  std::vector<indexTree::IndexTreeOp> iTreeRoots;
  getIndexTreeOps(function, iTreeRoots /* output */);
  for (auto root : iTreeRoots) {
    comet_vdump(root);
    OpBuilder builder(root);
//    Location loc = root->getLoc();
    doLoweringIndexTreeToSCF(root, builder);
  }

  {
//    for (Block &B : function.getBody()) {
//      for (Operation &op : B) {
//        comet_vdump(op);
//      }
//    }
    comet_vdump(module);
  }

}

// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createLowerIndexTreeToSCFPass()
{
  return std::make_unique<LowerIndexTreeToSCFPass>();
}
