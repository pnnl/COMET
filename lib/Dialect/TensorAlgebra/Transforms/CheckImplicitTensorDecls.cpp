//===- CheckImplicitTensorDecl.cpp - check if it is needed to add tensor declarations introduced by compound expressions------------------===//
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
/// This file implements a pass that adds temporary tensor declaration introduced by compound expressions in the COMET DSL
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


#include <limits>
#include <map>
#include <set>
#include <unordered_map>

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tensorAlgebra;

#define DEBUG_TYPE "check-implicit-tensor-decl"

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

using namespace mlir;

namespace
{
  /// Add tensor decl ops for tmp result
  struct TensorAlgebraCheckImplicitTensorDeclPass
      : public PassWrapper<TensorAlgebraCheckImplicitTensorDeclPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorAlgebraCheckImplicitTensorDeclPass)
    void runOnOperation() override;
  };
} /// namespace

template <typename t>
bool isNeedTensorDecl(t op)
{
  bool isUsedInSetSource = true;
  std::string op_str = dump2str(op);
  mlir::Value result = op.getOperation()->getResult(0);
  comet_debug() << " ";
  comet_vdump(result);
  for (auto u1 : result.getUsers())
  {
    comet_debug() << " ";
    comet_pdump(u1);
    /// If not used as source tensor of set_op, it is tmp result
    if (isa<tensorAlgebra::TensorSetOp>(u1))
    {
      comet_debug() << " used in ta.set_new op\n";
      auto p = cast<tensorAlgebra::TensorSetOp>(u1).getOperation();
      for (unsigned int i = 0; i < p->getNumOperands(); i++)
      {
        comet_debug() << " the " << i << "th operand\n";
        std::string n_str = dump2str(p->getOperand(i));
        if (n_str.compare(0, op_str.size(), op_str) == 0)
        {
          comet_debug() << " FIND IT: " << i << "\n";
          if (i == 0)
          { /// used as source tensor
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
    for (unsigned int i = 1; i < op.getOperation()->getNumOperands(); i++)
    {
      comet_debug() << " ";
      comet_vdump(op.getOperation()->getOperand(i));
      lbls_value.push_back(op.getOperation()->getOperand(i));
    }
    ret_value = op.getOperation()->getResult(0);

    mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<tensorAlgebra::TransposeOp>(op.getOperation()).getFormats();
    unsigned int i = opFormatsArrayAttr.size() - 1;
    std::string ret_format_local(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
    ret_format = ret_format_local;
  }
  else if (isa<TensorMultOp>(op) ||
           isa<TensorElewsMultOp>(op) ||
           isa<TensorAddOp>(op) ||
           isa<TensorSubtractOp>(op))
  {
    for (unsigned int i = 2; i < op.getOperation()->getNumOperands(); i++)
    {
      comet_debug() << " ";
      comet_vdump(op.getOperation()->getOperand(i));
      lbls_value.push_back(op.getOperation()->getOperand(i));
    }
    ret_value = op.getOperation()->getResult(0);

    mlir::ArrayAttr opFormatsArrayAttr;
    if (isa<tensorAlgebra::TensorMultOp>(op))
      opFormatsArrayAttr = dyn_cast<TensorMultOp>(op.getOperation()).getFormats();

    if (isa<tensorAlgebra::TensorElewsMultOp>(op))
      opFormatsArrayAttr = dyn_cast<TensorElewsMultOp>(op.getOperation()).getFormats();

    if (isa<tensorAlgebra::TensorAddOp>(op))
      opFormatsArrayAttr = dyn_cast<TensorAddOp>(op.getOperation()).getFormats();

    if (isa<tensorAlgebra::TensorSubtractOp>(op))
      opFormatsArrayAttr = dyn_cast<TensorSubtractOp>(op.getOperation()).getFormats();

    unsigned int i = opFormatsArrayAttr.size() - 1;
    std::string ret_format_local(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
    ret_format = ret_format_local;
  }
  else
  {
    llvm::errs() << __FILE__ << " " << __LINE__ <<  "ERROR: Unexpected operation\n";
  }

  OpBuilder builder(op);
  auto location = op.getLoc();

  mlir::Value itensor;
  if (ret_format.compare("Dense") == 0)
  {
    itensor = builder.create<DenseTensorDeclOp>(location, ret_value.getType(), lbls_value, ret_format);
    builder.create<TensorFillOp>(location, itensor, builder.getF64FloatAttr(0));
  }
  else
  {
    /// It is a temporal tensor declaration generated by compound expressions, BoolAttr is true
    /// to identify SparseTensorDeclOp is for temporaries
    // auto is_temporal_tensor = builder.getBoolAttr(true);
    itensor = builder.create<SparseTensorDeclOp>(location, ret_value.getType(), lbls_value, ret_format, true);
  }
  comet_debug() << "PreLowering SparseTensorDeclaration creation\n";
  comet_vdump(itensor);
  op.replaceAllUsesWith(itensor);

  builder.setInsertionPointAfter(op);
#ifdef DEBUG_MODE_ADDTEMPTENSORDECLARATION
  auto setop = builder.create<TensorSetOp>(location, ret_value, itensor);
  comet_debug() << " ";
  comet_vdump(setop);

  comet_debug() << " ";
  comet_vdump(ret_value);
#else
  builder.create<TensorSetOp>(location, ret_value, itensor);
#endif
}

void TensorAlgebraCheckImplicitTensorDeclPass::runOnOperation()
{
  func::FuncOp func = getOperation();
  comet_debug() << "Before TensorAlgebraCheckImplicitTensorDeclPass\n";

  /// The walker proceeds in post-order, but we need to process outer loops first
  /// to control the number of outer parallel loops, so push candidate loops to
  /// the front of a deque.
  func.walk([&](mlir::Operation *cur_op)
            {
           comet_debug() <<  " find a transpose op\n";
           /// if the output is not used as a source tensor of a set op
           /// Need to store use a sparse/dense tensor decl op to store the result
           if (isa<tensorAlgebra::TransposeOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TransposeOp>(cur_op);
             if (isNeedTensorDecl<tensorAlgebra::TransposeOp>(op))
             {
               addTensorDecl<tensorAlgebra::TransposeOp>(op);
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
           }
           else if (isa<tensorAlgebra::TensorAddOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TensorAddOp>(cur_op);
             if (isNeedTensorDecl<TensorAddOp>(op))
             {
               addTensorDecl<TensorAddOp>(op);
             }
           }
           else if (isa<tensorAlgebra::TensorSubtractOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TensorSubtractOp>(cur_op);
             if (isNeedTensorDecl<TensorSubtractOp>(op))
             {
               addTensorDecl<TensorSubtractOp>(op);
             }
           } });
  comet_debug() << "After TensorAlgebraCheckImplicitTensorDeclPass\n";
}

std::unique_ptr<Pass> mlir::comet::createTensorAlgebraCheckImplicitTensorDeclPass()
{
  return std::make_unique<TensorAlgebraCheckImplicitTensorDeclPass>();
}
