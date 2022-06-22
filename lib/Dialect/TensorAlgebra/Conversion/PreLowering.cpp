//===- PreLowering.cpp - PreLowering Pass -- initlization the result of tmp tensor result------------------===//
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
// This file implements a initlization the result of tmp tensor result.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"

#include <limits>
#include <map>
#include <set>
#include <unordered_map>

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using namespace mlir::tensorAlgebra;

#define DEBUG_TYPE "pre-lowering"

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_PreLoweringPass
// #define DEBUG_MODE_PreLoweringPass
// #endif

#ifdef DEBUG_MODE_PreLoweringPass
#define comet_errs() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n) n->dump()
#define comet_vdump(n) n.dump()
#else
#define comet_errs() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

using namespace mlir;

namespace
{
  // Add tensor decl ops for tmp result
  struct PreLoweringPass
      : public PassWrapper<PreLoweringPass, FunctionPass>
  {
    void runOnFunction() final;
  };
} // namespace

template <typename t>
bool isNeedTensorDecl(t op)
{
  bool isUsedInSetSource = true;
  std::string op_str = dump2str(op);
  mlir::Value result = op.getOperation()->getResult(0);
  comet_errs() << " ";
  comet_vdump(result);
  for (auto u1 : result.getUsers())
  {
    comet_errs() << " ";
    comet_pdump(u1);
    // If not used as source tensor of set_op, it is tmp result
    if (isa<tensorAlgebra::TensorSetOp>(u1))
    {
      comet_errs() << " used in ta.set_new op\n";
      auto p = cast<tensorAlgebra::TensorSetOp>(u1).getOperation();
      for (auto i = 0; i < p->getNumOperands(); i++)
      {
        comet_errs() << " the " << i << "th operand\n";
        std::string n_str = dump2str(p->getOperand(i));
        if (n_str.compare(0, op_str.size(), op_str) == 0)
        {
          comet_errs() << " FIND IT: " << i << "\n";
          if (i == 0)
          { // DONE(ruiqin): used as source tensor
            isUsedInSetSource = false;
          }
        }
      }
    }
  }
  return isUsedInSetSource;
}

template <typename t>
void addTensorDecl(t op)
{
  std::vector<mlir::Value> lbls_value;
  mlir::Value ret_value;
  std::string ret_format;
  if (isa<tensorAlgebra::TransposeOp>(op))
  {
    for (auto i = 1; i < op.getOperation()->getNumOperands(); i++)
    {
      comet_errs() << " ";
      comet_vdump(op.getOperation()->getOperand(i));
      lbls_value.push_back(op.getOperation()->getOperand(i));
    }
    ret_value = op.getOperation()->getResult(0);

    mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TransposeOp>(op.getOperation()).formats();
    unsigned int i = opFormatsArrayAttr.size() - 1;
    std::string ret_format_local(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
    ret_format = ret_format_local;
  }
  else if (isa<tensorAlgebra::TensorMultOp>(op))
  {
    for (auto i = 2; i < op.getOperation()->getNumOperands(); i++)
    {
      comet_errs() << " ";
      comet_vdump(op.getOperation()->getOperand(i));
      lbls_value.push_back(op.getOperation()->getOperand(i));
    }
    ret_value = op.getOperation()->getResult(0);

    mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorMultOp>(op.getOperation()).formats();
    unsigned int i = opFormatsArrayAttr.size() - 1;
    std::string ret_format_local(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
    ret_format = ret_format_local;
  }
  else if (isa<tensorAlgebra::TensorElewsMultOp>(op))
  {
    for (auto i = 2; i < op.getOperation()->getNumOperands(); i++)
    {
      comet_errs() << " ";
      comet_vdump(op.getOperation()->getOperand(i));
      lbls_value.push_back(op.getOperation()->getOperand(i));
    }
    ret_value = op.getOperation()->getResult(0);

    mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorElewsMultOp>(op.getOperation()).formats();
    unsigned int i = opFormatsArrayAttr.size() - 1;
    std::string ret_format_local(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
    ret_format = ret_format_local;
  }

  OpBuilder builder(op);
  auto location = op.getLoc();

  mlir::Value itensor;
  if (ret_format.compare("Dense") == 0)
    itensor = builder.create<DenseTensorDeclOp>(location, ret_value.getType(), lbls_value, ret_format);
  else
    itensor = builder.create<SparseTensorDeclOp>(location, ret_value.getType(), lbls_value, ret_format);
  comet_errs() << " ";
  comet_vdump(itensor);
  op.replaceAllUsesWith(itensor);

  builder.setInsertionPointAfter(op);
  auto setop = builder.create<TensorSetOp>(location, ret_value, itensor);
  comet_errs() << " ";
  comet_vdump(setop);

  comet_errs() << " ";
  comet_vdump(ret_value);
}

void PreLoweringPass::runOnFunction()
{
  FuncOp f = getFunction();

  // The walker proceeds in post-order, but we need to process outer loops first
  // to control the number of outer parallel loops, so push candidate loops to
  // the front of a deque.
  f.walk([&](mlir::Operation *cur_op)
         {
           comet_errs() <<  " find a transpose op\n";
           // if the output is not used as a source tensor of a set op
           // Need to store use a sparse/dense tensor decl op to store the result
           if (isa<tensorAlgebra::TransposeOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TransposeOp>(cur_op);
             if (isNeedTensorDecl<TransposeOp>(op))
             {
               addTensorDecl<TransposeOp>(op);
             }
           }
           else if (isa<tensorAlgebra::TensorMultOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TensorMultOp>(cur_op);
             if (isNeedTensorDecl<TensorMultOp>(op))
             {
               addTensorDecl<TensorMultOp>(op);
             }
           }
           else if (isa<tensorAlgebra::TensorElewsMultOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TensorElewsMultOp>(cur_op);
             if (isNeedTensorDecl<TensorElewsMultOp>(op))
             {
               addTensorDecl<TensorElewsMultOp>(op);
             }
           } });
}

std::unique_ptr<Pass> mlir::tensorAlgebra::createPreLoweringPass()
{
  return std::make_unique<PreLoweringPass>();
}
