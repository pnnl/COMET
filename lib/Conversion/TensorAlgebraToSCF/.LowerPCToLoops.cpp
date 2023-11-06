//===- LowerPCToLoops.cpp ------===//
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
// This file implements a lowering of some programming constructs such as for-loops, etc.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <queue>
#include <vector>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::tensorAlgebra;
using namespace mlir::indexTree;

#define DEBUG_TYPE "PC-lowering"

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

namespace
{
  struct PCToLoopsLoweringPass
      : public PassWrapper<PCToLoopsLoweringPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PCToLoopsLoweringPass)
    void runOnOperation() override;

    // lowers ForLoopBeginOp and ForLoopEndOp to scf.for, one loop at a time.
    // pre-condition: output from ProcessLoopOps();
    void PCToLoopsLowering(tensorAlgebra::ForLoopBeginOp op, tensorAlgebra::ForLoopEndOp op_end, std::vector<Operation *> &listOps);

    // find all ops to be placed inside the loop body.
    std::vector<Operation *> ProcessLoopOps(tensorAlgebra::ForLoopBeginOp op_start, tensorAlgebra::ForLoopEndOp op_end);

    // replicates Ops for loop-body
    // pre-condition: some ops require previously generated ops.
    void replicateOpsForLoopBody(Location loc, OpBuilder &builder, Operation *op, Value &rhs, Value &lhs, Value &compute, Value &indices, Value &transpose);
  };
} // end anonymous namespace.

// find all ops to be placed inside the loop body.
std::vector<Operation *> PCToLoopsLoweringPass::ProcessLoopOps(tensorAlgebra::ForLoopBeginOp op_start, tensorAlgebra::ForLoopEndOp op_end)
{
  comet_debug() << "START: Pre-processing to detect ops to be placed inside loop body.\n";
  comet_pdump(op_start);

  std::vector<Operation *> loop_blk;

  auto *B = op_start.getOperation()->getBlock();
  bool match = false;
  // collect all ops to for `one` loop body
  for (Operation &op : *B)
  {
    if (isa<tensorAlgebra::ForLoopBeginOp>(&op))
    {
      // check for match
      if (cast<tensorAlgebra::ForLoopBeginOp>(op).getIterator() == op_start.getIterator())
      {
        match = true;
        continue; // skip this iteration or op;
      }
    }

    if (isa<tensorAlgebra::ForLoopEndOp>(&op))
    {
      if (cast<tensorAlgebra::ForLoopEndOp>(op) == op_end)
      {
        match = false;
        break; // we are done with match of `one` loop-body
      }
    }

    if (match)
    { // add to list of ops to be replicated
      loop_blk.push_back(&op);
    }
  }

  comet_debug() << "END: Pre-processing to detect ops to be placed inside loop body.\n";
  return loop_blk;
}

// replicates Ops for loop-body
// pre-condition: some ops require previously generated ops.
void PCToLoopsLoweringPass::replicateOpsForLoopBody(Location loc, OpBuilder &builder, Operation *op,
                                                    Value &rhs, Value &lhs, Value &compute, Value &indices, Value &transpose)
{
  IntegerType i64Type = IntegerType::get(builder.getContext(), 64);

  if (isa<tensorAlgebra::TransposeOp>(op))
  {
    comet_debug() << "creating TransposeOp\n";
    tensorAlgebra::TransposeOp ta_transpose_op = llvm::dyn_cast<tensorAlgebra::TransposeOp>(op);
    std::vector<Value> lhs_lbls_value = {ta_transpose_op.getOperand(1), ta_transpose_op.getOperand(2)};
    transpose = builder.create<tensorAlgebra::TransposeOp>(loc, ta_transpose_op.getOperand(0).getType(), ta_transpose_op.getOperand(0),
                                                           lhs_lbls_value, ta_transpose_op.getIndexingMaps(), ta_transpose_op.getFormats());
  }

  // TensorSetOp goes with TransposeOp at this stage of the lowering.
  if (isa<tensorAlgebra::TensorSetOp>(op) && transpose != NULL)
  {
    comet_debug() << "creating SetOp after TransposeOp\n";
    tensorAlgebra::TensorSetOp ta_set_op = llvm::dyn_cast<tensorAlgebra::TensorSetOp>(op);
    builder.create<TensorSetOp>(loc, transpose, ta_set_op.getOperand(1));
  }

  // create IndexTreeComputeRHSOp, no dependency to earlier replications
  if (isa<indexTree::IndexTreeComputeRHSOp>(op))
  {
    indexTree::IndexTreeComputeRHSOp it_compute_rhs_op = llvm::dyn_cast<indexTree::IndexTreeComputeRHSOp>(op);
    ArrayAttr op_formats_ArrayAttr = it_compute_rhs_op.getAllFormats();
    ArrayAttr op_perms_ArrayAttr = it_compute_rhs_op.getAllPerms();

    rhs = builder.create<indexTree::IndexTreeComputeRHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()),
                                                           it_compute_rhs_op->getOperands(), // tensors
                                                           op_perms_ArrayAttr, op_formats_ArrayAttr);
  }

  // create IndexTreeComputeLHSOp, no dependency to earlier replications
  if (isa<indexTree::IndexTreeComputeLHSOp>(op))
  {
    indexTree::IndexTreeComputeLHSOp it_compute_lhs_op = llvm::dyn_cast<indexTree::IndexTreeComputeLHSOp>(op);
    ArrayAttr op_formats_ArrayAttr = it_compute_lhs_op.getAllFormats();
    ArrayAttr op_perms_ArrayAttr = it_compute_lhs_op.getAllPerms();

    lhs = builder.create<indexTree::IndexTreeComputeLHSOp>(loc, mlir::UnrankedTensorType::get(builder.getF64Type()),
                                                           it_compute_lhs_op->getOperands(), /// tensors
                                                           op_perms_ArrayAttr, op_formats_ArrayAttr);
  }

  /// create IndexTreeComputeOp only if rhs and lhs are ready
  if (isa<indexTree::IndexTreeComputeOp>(op) && rhs != NULL && lhs != NULL)
  {
    indexTree::IndexTreeComputeOp it_compute_op = llvm::dyn_cast<indexTree::IndexTreeComputeOp>(op);
    compute = builder.create<indexTree::IndexTreeComputeOp>(loc, i64Type, rhs, lhs,
                                                            it_compute_op.getCompWorkspOpt(), it_compute_op.getSemiring(), it_compute_op.getMaskType());
  }

  /// create IndexTreeIndicesOp from existing IndexTreeIndicesOp (checks condition: indices != NULL)
  if (isa<indexTree::IndexTreeIndicesOp>(op) && indices != NULL)
  {
    indexTree::IndexTreeIndicesOp it_indices_op = llvm::dyn_cast<indexTree::IndexTreeIndicesOp>(op);
    Value indices_op_new = builder.create<indexTree::IndexTreeIndicesOp>(loc, i64Type, indices, it_indices_op.getIndices());

    indices = indices_op_new; /// for subsequent IndexTreeIndicesOp creation
  }

  /// create the first instance of IndexTreeIndicesOp from IndexTreeComputeOp
  if (isa<indexTree::IndexTreeIndicesOp>(op) && compute != NULL && indices == NULL)
  {
    indexTree::IndexTreeIndicesOp it_indices_op = llvm::dyn_cast<indexTree::IndexTreeIndicesOp>(op);
    indices = builder.create<indexTree::IndexTreeIndicesOp>(loc, i64Type, compute, it_indices_op.getIndices());
  }

  if (isa<indexTree::IndexTreeOp>(op) && indices != NULL)
  {
    builder.create<indexTree::IndexTreeOp>(loc, i64Type, indices);
  }
}

/// lowers ForLoopBeginOp and ForLoopEndOp to scf.for, one loop at a time.
void PCToLoopsLoweringPass::PCToLoopsLowering(tensorAlgebra::ForLoopBeginOp op, tensorAlgebra::ForLoopEndOp op_end, std::vector<Operation *> &listOps)
{
  comet_debug() << "PCToLoopsLowering start\n";
  OpBuilder builder(op);

  comet_pdump(op);
  comet_pdump(op_end);

  auto loc = op->getLoc();
  auto ForLoopStart = cast<tensorAlgebra::ForLoopBeginOp>(op);

  /// get info of loop
  auto upperBound = ForLoopStart.getMax();
  auto lowerBound = ForLoopStart.getMin();
  auto step = ForLoopStart.getStep();

  auto loop = builder.create<scf::ForOp>(op_end->getLoc(), lowerBound, upperBound, step);
  comet_vdump(loop);

  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(loop.getBody());

  /// loop-body: listOps contains all ops obtained thru ProcessLoopOps()
  ///            to be placed inside 'one' loop-body.
  /// TODO(gkestor): verify all ops are covered here!
  Value new_rhs_op, new_lhs_op, new_compute_op, indices_op;
  Value transpose_op;
  std::vector<Value> loop_bounds; /// carries loop bounds
  for (unsigned int i = 0; i < listOps.size(); i++)
  {
    comet_pdump(listOps[i]);

    replicateOpsForLoopBody(loc, builder, listOps[i], new_rhs_op, new_lhs_op, new_compute_op, indices_op, transpose_op);

    /// nested loop
    if (isa<ConstantIndexOp>(listOps[i]))
    {
      ConstantIndexOp constant_index = llvm::dyn_cast<ConstantIndexOp>(listOps[i]);
      loop_bounds.push_back(builder.create<ConstantIndexOp>(loc, constant_index.value()));
    }

    /// scf::for op
    /// the inner loop bodies have already been created. preserve them.
    /// the loop_bounds array should have lowerBound, upperBound and step Values.
    if (isa<scf::ForOp>(listOps[i]) && loop_bounds.size() == 3)
    {
      scf::ForOp scf_for_op = llvm::dyn_cast<scf::ForOp>(listOps[i]);
      scf::ForOp nested_scf_for_op = builder.create<scf::ForOp>(loc, loop_bounds[0], loop_bounds[1], loop_bounds[2]);

      auto insertPt_nested = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(nested_scf_for_op.getBody());

      Block *B = scf_for_op.getBody();
      Value nested_rhs_op, nested_lhs_op, nested_compute_op, nested_indices_op;
      Value nested_transpose_op;
      comet_debug() << "going to replicate ops inside nested loop...\n";
      for (Operation &op_for : *B)
      {
        replicateOpsForLoopBody(loc, builder, &op_for, nested_rhs_op, nested_lhs_op, nested_compute_op, nested_indices_op, nested_transpose_op);
      }
      /// need to restore the insertion point to the previous point
      builder.restoreInsertionPoint(insertPt_nested);
      builder.setInsertionPoint(op_end); /// TODO(gkestor): need to re-visit this for nested loops.
    }
  }

  /// remove old ops, since now we are done with the clone inside loop-body.
  /// this is done in reverse order.
  comet_debug() << "Removing ops that have been cloned\n";
  for (unsigned int i = 0; i < listOps.size(); i++)
  {
    comet_pdump(listOps[listOps.size() - i - 1]);
    listOps[listOps.size() - i - 1]->erase();
  }

  /// need to restore the insertion point to the previous point
  builder.restoreInsertionPoint(insertPt);
  builder.setInsertionPoint(op_end);

  /// remove ForLoopBeginOp and ForLoopEndOp
  op->erase();
  op_end->erase();

  comet_debug() << "PCToLoopsLowering end\n";
}

void PCToLoopsLoweringPass::runOnOperation()
{
  comet_debug() << "start PCToLoopsLoweringPass\n";

  func::FuncOp function = getOperation();

  std::vector<tensorAlgebra::ForLoopBeginOp> startOps;
  std::vector<tensorAlgebra::ForLoopEndOp> endOps;

  /// collect all the loops (begin and ends) here in vector data-structure. e.g.,
  ///  for-start1 ():
  ///    for-start2 ():
  ///       do_work2();
  ///    end2
  ///    do_work1();
  ///  end1

  /// vector: for-start1, for-start2
  /// vector: end2, end1

  for (Block &B : function.getBody())
  {
    for (Operation &op : B)
    {
      if (isa<tensorAlgebra::ForLoopBeginOp>(&op))
      {
        startOps.push_back(cast<tensorAlgebra::ForLoopBeginOp>(op));
      }
      if (isa<tensorAlgebra::ForLoopEndOp>(&op))
      {
        endOps.push_back(cast<tensorAlgebra::ForLoopEndOp>(op));
      }
    }
  }

  /// if there are no for-loops, quit.
  if (startOps.empty())
    return;

  /// the size of the two datastructure should be same
  assert(startOps.size() == endOps.size() && "the for-begins must match the ends");
  std::vector<Operation *> opList;
  for (unsigned int i = 0; i < startOps.size(); i++)
  {
    /// start with inner most loop and move outwards.
    opList = ProcessLoopOps(startOps[startOps.size() - i - 1], endOps[i]); /// for-1, end-1
    PCToLoopsLowering(startOps[startOps.size() - i - 1], endOps[i], opList);
    opList.clear(); /// clear for next round.
  }
  startOps.clear();
  endOps.clear();

  /// debug
  /// auto module = function.getOperation()->getParentOfType<ModuleOp>();

  comet_debug() << "end PCToLoopsLoweringPass\n";
}

/// Create a pass for lowering programming constructs
std::unique_ptr<Pass> mlir::comet::createPCToLoopsLoweringPass()
{
  return std::make_unique<PCToLoopsLoweringPass>();
}