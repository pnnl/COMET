//===- MLIRGen.cpp - MLIR Generation from a COMET language AST
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
// =============================================================================
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for COMET DSL.
//
//===----------------------------------------------------------------------===//

#include "Lexer.h"

#include "MLIRGen.h"
#include "AST.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <numeric>
#include <cstdlib> // for random num generation
#include <random>  // for seed of random num generation

#include "mlir/IR/Types.h"

using namespace mlir::tensorAlgebra;
using namespace tensorAlgebra;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

using StringSet = std::set<std::string>;

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_MLIRGEN
// #define DEBUG_MODE_MLIRGEN
// #endif

#ifdef DEBUG_MODE_MLIRGEN
#define comet_debug() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n)                                \
  llvm::errs() << __FILE__ << " " << __LINE__ << " "; \
  n->dump()
#define comet_vdump(n)                                \
  llvm::errs() << __FILE__ << " " << __LINE__ << " "; \
  n.dump()
#else
#define comet_debug() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

namespace
{

  enum class ElType
  {
    inv = 0b1000,
    i32 = 0b0000,
    i64 = 0b0001,
    fp32 = 0b0010,
    fp64 = 0b0011,
    cfp32 = 0b0110,
    cfp64 = 0b0111
  };

  std::string getSemiringOpName(int num)
  {
    std::string opName;
    switch (num)
    {
    case 100:
      opName = "lor";
      break;
    case 101:
      opName = "land";
      break;
    case 102:
      opName = "minxy";
      break;
    case 103:
      opName = "first";
      break;
    case 104:
      opName = "plusxy";
      break;
    case 105:
      opName = "times";
      break;
    case 106:
      opName = "any";
      break;
    case 107:
      opName = "pairxy";
      break;
    case 108:
      opName = "second";
      break;
    case 109:
      opName = "noop";
      break;
    default:
      llvm::errs() << "[ERR] Semiring operator number not defined \n";
    }
    comet_debug() << "Semiring op name: " << opName << "\n";
    return opName;
  };

  constexpr ElType lub(ElType first, ElType second)
  {
    return ElType(static_cast<int>(first) | static_cast<int>(second));
  }

  /// Implementation of a simple MLIR emission from the Tensor Algebra AST.
  ///
  /// This will emit operations that are specific to the Tensor Algebra language,
  /// preserving the semantics of the language and (hopefully) allow to perform
  /// accurate analysis and transformation based on these high level semantics.
  class MLIRGenImpl
  {
  public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    /// Public API: convert the AST for a Tensor Algebra module (source file) to
    /// an MLIR Module operation.
    mlir::ModuleOp mlirGen(ModuleAST &moduleAST)
    {
      // We create an empty MLIR module and codegen functions one at a time and
      // add them to the module.
      theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

      for (FunctionAST &F : moduleAST)
      {
        auto func = mlirGen(F);
        if (!func)
          return nullptr;
        theModule.push_back(func);
      }

      // Verify the module after we have finished constructing it, this will check
      // the structural properties of the IR and invoke any specific verifiers we
      // have on the Tensor Algebra operations.
      if (failed(mlir::verify(theModule)))
      {
        theModule.emitError("module verification error");
        return nullptr;
      }

      return theModule;
    }

  private:
    /// A "module" matches a Tensor Algebra source file: containing a list of
    /// functions.
    mlir::ModuleOp theModule;

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    /// The symbol table maps a variable name to a value in the current scope.
    /// Entering a function creates a new scope, and the function arguments are
    /// added to the mapping. When the processing of a function is terminated, the
    /// scope is destroyed and the mappings created in this scope are dropped.
    llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

    /// Helper conversion for a Tensor Algebra AST location to an MLIR location.
    mlir::Location loc(Location loc)
    {
      // return builder.getFileLineColLoc(builder.getIdentifier(*loc.file), loc.line,
      //                                  loc.col);
      return mlir::FileLineColLoc::get(builder.getIdentifier(*loc.file), loc.line,
                                       loc.col);
    }

    // For scalar-arithmetic-op:
    // var a = 0; and c = a + b; are all specified in VarDeclExprAST
    // in which all var are added to the symboltable; in this case,
    // a is added twice. So, we cannot return failure for this case if found
    // an existing variable from the symbol table.
    // Change to:
    // If not exist, add the variable to symbol table; return success anyway
    mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value)
    {
      if (!symbolTable.count(var))
      {
        symbolTable.insert(var, value);
      }
      return mlir::success();
    }

    /// Create the prototype for an MLIR function with as many arguments as the
    /// provided Tensor Algebra AST prototype.
    mlir::FuncOp mlirGen(PrototypeAST &proto)
    {
      auto location = loc(proto.loc());

      // This is a generic function, the return type will be inferred later.
      // Arguments type are uniformly unranked tensors.
      llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(),
                                                 getType(VarType{}));
      auto func_type = builder.getFunctionType(arg_types, llvm::None);
      return mlir::FuncOp::create(location, proto.getName(), func_type);
    }

    /// Emit a new function and add it to the MLIR module.
    mlir::FuncOp mlirGen(FunctionAST &funcAST)
    {
      // Create a scope in the symbol table to hold variable declarations.
      ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);

      // Create an MLIR function for the given prototype.
      mlir::FuncOp function(mlirGen(*funcAST.getProto()));
      if (!function)
        return nullptr;

      // Let's start the body of the function now!
      // In MLIR the entry block of the function is special: it must have the same
      // argument list as the function itself.
      auto &entryBlock = *function.addEntryBlock();
      auto &protoArgs = funcAST.getProto()->getArgs();

      // Declare all the function arguments in the symbol table.
      for (const auto name_value :
           llvm::zip(protoArgs, entryBlock.getArguments()))
      {
        if (failed(declare(std::get<0>(name_value)->getName(),
                           std::get<1>(name_value))))
          return nullptr;
      }

      // Set the insertion point in the builder to the beginning of the function
      // body, it will be used throughout the codegen to create operations in this
      // function.
      builder.setInsertionPointToStart(&entryBlock);

      // Emit the body of the function.
      if (mlir::failed(mlirGen(*funcAST.getBody())))
      {
        function.erase();
        return nullptr;
      }

      // Implicitly return void if no return statement was emitted.
      // FIXME: we may fix the parser instead to always return the last expression
      // (this would possibly help the REPL case later)
      TAReturnOp returnOp;
      if (!entryBlock.empty())
        returnOp = dyn_cast<TAReturnOp>(entryBlock.back());
      if (!returnOp)
      {
        builder.create<TAReturnOp>(loc(funcAST.getProto()->loc()));
      }
      else if (returnOp.hasOperand())
      {
        // Otherwise, if this return operation has an operand then add a result to
        // the function.
        function.setType(builder.getFunctionType(function.getType().getInputs(),
                                                 getType(VarType{})));
      }

      return function;
    }

    /// Emit a binary operation
    mlir::Value mlirGen(BinaryExprAST &binop,
                        const std::set<std::string> &out_lbls = {})
    {
      comet_debug() << " mlirGen for  BinaryExprAST \n";
      // First emit the operations for each side of the operation before emitting
      // the operation itself. For example if the expression is `a + foo(a)`
      // 1) First it will visiting the LHS, which will return a reference to the
      //    value holding `a`. This value should have been emitted at declaration
      //    time and registered in the symbol table, so nothing would be
      //    codegen'd. If the value is not in the symbol table, an error has been
      //    emitted and nullptr is returned.
      // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
      //    and the result value is returned. If an error occurs we get a nullptr
      //    and propagate.
      //
      std::set<std::string> rhs_lbls{out_lbls};
      std::set<std::string> lhs_lbls{out_lbls};
      auto *lhsAST = binop.getLHS();
      auto *rhsAST = binop.getRHS();

      if (lhsAST->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor)
      {
        comet_debug() << " lhsAST is Expr_LabeledTensor \n";
        auto lblsVec = cast<LabeledTensorExprAST>(lhsAST)->getLabelNames();
        for (const auto &lbl : lblsVec)
        {
          lhs_lbls.insert(lbl);
        }
      }
      else if (lhsAST->getKind() == ExprAST::ExprASTKind::Expr_BinOp)
      {
        comet_debug() << "\n"
                      << __LINE__ << " lhsAST is Expr_BinOp \n";
        auto lblsSet = cast<BinaryExprAST>(lhsAST)->getLabels();
        comet_debug() << "\n"
                      << __LINE__ << " lhsAST.getOp(): " << cast<BinaryExprAST>(lhsAST)->getOp() << "\n";

        for (const auto &lbl : lblsSet)
        {
          lhs_lbls.insert(lbl);
        }
      }
      else if (lhsAST->getKind() == ExprAST::ExprASTKind::Expr_Call)
      {
        comet_debug() << "\n"
                      << __LINE__ << " lhsAST is Expr_Call  \n";
      }
      else
      {
        comet_debug() << "\n"
                      << __LINE__ << " lhsAST is not Expr_BinOp/Expr_LabeledTensor/CallExprAST  \n";
      }

      if (rhsAST->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor)
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST is Expr_LabeledTensor \n";
        auto lblsVec = cast<LabeledTensorExprAST>(rhsAST)->getLabelNames();
        for (const auto &lbl : lblsVec)
        {
          rhs_lbls.insert(lbl);
        }
      }
      else if (rhsAST->getKind() == ExprAST::ExprASTKind::Expr_BinOp)
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST is Expr_BinOp \n";
        auto lblsSet = cast<BinaryExprAST>(rhsAST)->getLabels();
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST.getOp(): " << cast<BinaryExprAST>(rhsAST)->getOp() << "\n";

        for (const auto &lbl : lblsSet)
        {
          rhs_lbls.insert(lbl);
        }
      }
      else if (rhsAST->getKind() == ExprAST::ExprASTKind::Expr_Transpose)
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST is Expr_Transpose  \n";
      }
      else
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST is not Expr_BinOp/Expr_LabeledTensor/CallExprAST/TransposeExprAST  \n";
      }

      mlir::Value lhs = mlirGen(*binop.getLHS(), rhs_lbls);
      if (!lhs)
        return nullptr;
      mlir::Value rhs = mlirGen(*binop.getRHS(), lhs_lbls);
      if (!rhs)
        return nullptr;
      auto location = loc(binop.loc());

      comet_vdump(lhs);
      comet_vdump(rhs);
      comet_debug() << " " << lhsAST->getKind() << " " << rhsAST->getKind();

      int op = 0;
      if (lhsAST->getKind() == ExprAST::ExprASTKind::Expr_Var &&
          rhsAST->getKind() == ExprAST::ExprASTKind::Expr_Var)
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST and lhsAST are all Expr_Var\n";

        switch (binop.getOp())
        {
        case '+':
          op = '+';
          break;
        case '-':
          op = '-';
          break;
        case '*':
          op = '*';
          break;
        case '/':
          op = '/';
          break;
        default:
          comet_debug() << "ERROR: unsupported operator type: ASCII Code(" << binop.getOp() << ")\n";
        }
        mlir::IntegerAttr opAttr = builder.getI32IntegerAttr(op);
        return builder.create<ScalarOp>(location, builder.getF64Type(), rhs, lhs, opAttr);
      }
      else if (isa<DenseConstantOp>(lhs.getDefiningOp()))
      {
        comet_debug() << "\n"
                      << __LINE__ << " lhs is DenseConstantOp\n";
        switch (binop.getOp())
        {
        case '+':
          return builder.create<AddOp>(location, lhs, rhs);
        case '-':
          return builder.create<SubstractOp>(location, lhs, rhs);
        case '*':
          return builder.create<MulOp>(location, lhs, rhs);
        case '/':
          return builder.create<DivOp>(location, lhs, rhs);
        }
      }
      else if (isa<DenseConstantOp>(rhs.getDefiningOp()))
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhs is DenseConstantOp\n";
        switch (binop.getOp())
        {
        case '+':
          return builder.create<AddOp>(location, lhs, rhs);
        case '-':
          return builder.create<SubstractOp>(location, lhs, rhs);
        case '*':
          return builder.create<MulOp>(location, lhs, rhs);
        case '/':
          return builder.create<DivOp>(location, lhs, rhs);
        }
      }
      else
      {
        comet_debug() << " lhs or rhs are binaryop\n";
        // lhs && rhs are binaryop
        std::set<mlir::Operation *> summed_labels;

        if (lhsAST->getKind() == ExprAST::ExprASTKind::Expr_BinOp)
        {
          comet_debug() << " lhs is binaryop\n";
          auto lhsOp = lhs.getDefiningOp();

          comet_pdump(lhsOp);
          if (isa<MulOp>(lhsOp))
          {
            auto lhsMulOp = cast<MulOp>(lhsOp);
            for (auto val : lhsMulOp.sum_labels())
            {
              summed_labels.insert(val.getDefiningOp());
            }
          }
          else if (isa<AddOp>(lhsOp))
          {
            // TODO(gkestor) check for AddOp
          }
        }
        else
        {
          comet_debug() << " lhs is not binaryop\n";
        }

        if (rhsAST->getKind() == ExprAST::ExprASTKind::Expr_BinOp)
        {
          comet_debug() << " rhs is binaryop\n";
          auto rhsOp = rhs.getDefiningOp();

          comet_pdump(rhsOp);
          if (isa<MulOp>(rhsOp))
          {
            auto rhsMulOp = cast<MulOp>(rhsOp);
            for (auto val : rhsMulOp.sum_labels())
            {
              summed_labels.insert(val.getDefiningOp());
            }
          }
          else if (isa<AddOp>(rhsOp))
          {
            // TODO(gkestor) check for AddOp
          }
          else if (isa<TransposeOp>(rhsOp))
          { // * transpose(A[i,j])
            comet_debug() << " "
                          << "\n";
          }
        }
        else
        {
          comet_debug() << " rhs is not binaryop\n";
        }

        auto in_labels = binop.getLabels();
        comet_debug() << __LINE__ << " in_labels: ";
        for (auto n : in_labels)
        {
          comet_debug() << n << " ";
        }
        comet_debug() << "\n";

        std::vector<std::string> sum_labels;
        std::set_difference(in_labels.begin(), in_labels.end(), out_lbls.begin(),
                            out_lbls.end(), std::back_inserter(sum_labels));
        comet_debug() << __LINE__ << " sum_labels: ";
        for (auto n : sum_labels)
        {
          comet_debug() << n << " ";
        }
        comet_debug() << "\n";

        std::vector<mlir::Value> lbls;
        for (const auto &lbl_str : sum_labels)
        {
          if (auto var = symbolTable.lookup(lbl_str))
          {
            if (isa<IndexLabelStaticOp>(var.getDefiningOp()))
            {
              if (summed_labels.find(var.getDefiningOp()) ==
                  summed_labels.end())
              {
                lbls.push_back(var);
              }
            }
            else if (isa<IndexLabelDynamicOp>(var.getDefiningOp()))
            {
              if (summed_labels.find(var.getDefiningOp()) ==
                  summed_labels.end())
              {
                lbls.push_back(var);
              }
            }
            else
            {
              emitError(loc(binop.loc()), "Index label variable required '")
                  << lbl_str << "'";
            }
          }
          else
          {
            emitError(loc(binop.loc()), " Unknown variable BinaryExprAST' ")
                << lbl_str << "'";
          }
        }

        std::vector<std::string> result_lbls;
        std::set<std::string> sum_lbls_set(sum_labels.begin(), sum_labels.end());
        std::set_difference(in_labels.begin(), in_labels.end(),
                            sum_lbls_set.begin(), sum_lbls_set.end(),
                            std::back_inserter(result_lbls));
        comet_debug() << __LINE__ << " result_lbls: ";
        for (auto n : result_lbls)
        {
          comet_debug() << n << " ";
        }
        comet_debug() << "\n";

        auto lhs_tensor = lhs.getDefiningOp()->getOpResult(0).getType();
        assert(lhs_tensor.isa<mlir::TensorType>());

        comet_pdump(lhs.getDefiningOp());
        auto lhs_labeledtensor = lhs.getDefiningOp()->getOpResult(0);

        comet_vdump(lhs_labeledtensor); // ta.labeled_tensor
        auto lhs_el_type = lhs_tensor.cast<mlir::TensorType>().getElementType();

        auto rhs_tensor = rhs.getDefiningOp()->getOpResult(0).getType();

        comet_pdump(rhs.getDefiningOp());
        assert(rhs_tensor.isa<mlir::TensorType>());

        auto rhs_labeledtensor = rhs.getDefiningOp()->getOpResult(0);

        comet_vdump(rhs_labeledtensor);
        auto rhs_el_type = rhs_tensor.cast<mlir::TensorType>().getElementType();
        auto result_type = getBinOpResultType(lhs_el_type, rhs_el_type);
        comet_debug() << __LINE__ << " ";
        comet_vdump(result_type);

        comet_debug() << __LINE__ << " binop.getOp(): " << binop.getOp() << "\n";

        std::map<int, mlir::Value> lblsValue_map;
        std::vector<mlir::Value> lhs_lbls_value;
        if (isa<SparseTensorDeclOp, DenseTensorDeclOp>(lhs_labeledtensor.getDefiningOp()))
        {
          for (unsigned int i = 0; i < lhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
          {

            comet_vdump(lhs_labeledtensor.getDefiningOp()->getOperand(i));
            lhs_lbls_value.push_back(lhs_labeledtensor.getDefiningOp()->getOperand(i));
          }
        }
        else if (isa<LabeledTensorOp>(lhs_labeledtensor.getDefiningOp()))
        {
          for (unsigned int i = 1; i < lhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
          {

            comet_vdump(lhs_labeledtensor.getDefiningOp()->getOperand(i));
            lhs_lbls_value.push_back(lhs_labeledtensor.getDefiningOp()->getOperand(i));
          }
        }
        else if (isa<TensorMultOp, TensorElewsMultOp>(lhs_labeledtensor.getDefiningOp()))
        {
          // check the output indices of ta.tc(), if it is
          for (unsigned int i = 2; i < lhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
          {

            comet_vdump(lhs_labeledtensor.getDefiningOp()->getOperand(i));
            lhs_lbls_value.push_back(lhs_labeledtensor.getDefiningOp()->getOperand(i));
          }
        }
        else if (isa<TransposeOp>(lhs_labeledtensor.getDefiningOp()))
        {
          // check the output indices of ta.tc(), if it is
          for (unsigned int i = 1; i < lhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
          {

            comet_vdump(lhs_labeledtensor.getDefiningOp()->getOperand(i));
            lhs_lbls_value.push_back(lhs_labeledtensor.getDefiningOp()->getOperand(i));
          }
        }
        else
        {
          llvm::errs() << __FILE__ << __LINE__ << " unknown lhs \n";
        }

        std::vector<mlir::Value> rhs_lbls_value;

        if (isa<SparseTensorDeclOp, DenseTensorDeclOp>(rhs_labeledtensor.getDefiningOp()))
        {
          for (unsigned int i = 0; i < rhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
          {

            comet_vdump(rhs_labeledtensor.getDefiningOp()->getOperand(i));
            rhs_lbls_value.push_back(rhs_labeledtensor.getDefiningOp()->getOperand(i));
          }
        }
        else if (isa<LabeledTensorOp>(rhs_labeledtensor.getDefiningOp()))
        {
          for (unsigned int i = 1; i < rhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
          {

            comet_vdump(rhs_labeledtensor.getDefiningOp()->getOperand(i));
            rhs_lbls_value.push_back(rhs_labeledtensor.getDefiningOp()->getOperand(i));
          }
        }
        else if (isa<TensorMultOp, TensorElewsMultOp>(rhs_labeledtensor.getDefiningOp()))
        {
          // check the output indices of ta.tc(), if it is
          for (unsigned int i = 2; i < rhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
          {

            comet_vdump(rhs_labeledtensor.getDefiningOp()->getOperand(i));
            rhs_lbls_value.push_back(rhs_labeledtensor.getDefiningOp()->getOperand(i));
          }
        }
        else if (isa<TransposeOp>(rhs_labeledtensor.getDefiningOp()))
        {
          // check the output indices of ta.tc(), if it is
          for (unsigned int i = 1; i < rhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
          {

            comet_vdump(rhs_labeledtensor.getDefiningOp()->getOperand(i));
            rhs_lbls_value.push_back(rhs_labeledtensor.getDefiningOp()->getOperand(i));
          }
        }
        else
        {
          llvm::errs() << __FILE__ << __LINE__ << " unknown rhs \n";
        }

        std::vector<mlir::Value> all_lbls_value;

        for (auto n : lhs_lbls_value)
        {
          all_lbls_value.push_back(n);
          comet_vdump(n);
        }
        for (auto m : rhs_lbls_value)
        {
          comet_vdump(m);
          auto result1 = std::find(all_lbls_value.begin(), all_lbls_value.end(), m);
          if (result1 == all_lbls_value.end())
          {
            all_lbls_value.push_back(m);
          }
        }
        comet_debug() << " " << all_lbls_value.size() << "\n";

        std::vector<int> lhs_lbls;
        std::vector<int> rhs_lbls;
        for (unsigned int i = 0; i < all_lbls_value.size(); i++)
        {
          auto result1 = std::find(lhs_lbls_value.begin(), lhs_lbls_value.end(), all_lbls_value[i]);
          if (result1 != lhs_lbls_value.end())
          {
            lhs_lbls.push_back(i);
          }

          auto result2 = std::find(rhs_lbls_value.begin(), rhs_lbls_value.end(), all_lbls_value[i]);
          if (result2 != rhs_lbls_value.end())
          {
            rhs_lbls.push_back(i);
          }
        }

        comet_debug() << " print lhs_lbls\n";
        for (auto n : lhs_lbls)
        {
          comet_debug() << n << " \n";
        }
        comet_debug() << "\n";
        comet_debug() << " print rhs_lbls\n";
        for (auto n : rhs_lbls)
        {
          comet_debug() << n << " \n";
        }
        comet_debug() << "\n";

        std::vector<int> sum_lbls;
        std::set_intersection(lhs_lbls.begin(), lhs_lbls.end(), rhs_lbls.begin(), rhs_lbls.end(), std::back_inserter(sum_lbls));
        std::vector<int> all_lbls;
        std::set_union(lhs_lbls.begin(), lhs_lbls.end(), rhs_lbls.begin(), rhs_lbls.end(), std::back_inserter(all_lbls));
        std::vector<int> ret_lbls;
        if (binop.getOp() == '*')
        {
          std::set_difference(all_lbls.begin(), all_lbls.end(), sum_lbls.begin(), sum_lbls.end(), std::back_inserter(ret_lbls));
        }
        else if (binop.getOp() == tok_elews)
        {
          std::copy(lhs_lbls.begin(), lhs_lbls.end(), std::back_inserter(ret_lbls));
        }
        comet_debug() << " print ret_lbls\n";
        for (auto n : ret_lbls)
        {
          comet_debug() << n << " \n";
        }
        comet_debug() << "\n";

        std::map<int, mlir::AffineExpr> expr_map;
        unsigned dim = 0;
        for (const auto &lbl : all_lbls)
        {
          expr_map[lbl] = getAffineDimExpr(dim++, builder.getContext());
        }

        std::vector<mlir::AffineExpr> lhs_exprs;
        std::vector<mlir::AffineExpr> rhs_exprs;
        std::vector<mlir::AffineExpr> ret_exprs;

        for (const auto &lbl : lhs_lbls)
        {
          lhs_exprs.push_back(expr_map[lbl]);
        }

        for (const auto &lbl : rhs_lbls)
        {
          rhs_exprs.push_back(expr_map[lbl]);
        }

        for (const auto &lbl : ret_lbls)
        {
          ret_exprs.push_back(expr_map[lbl]);
        }

        auto context = builder.getContext();
        SmallVector<mlir::AffineMap, 8> affine_maps{
            mlir::AffineMap::get(dim, 0, lhs_exprs, context),
            mlir::AffineMap::get(dim, 0, rhs_exprs, context),
            mlir::AffineMap::get(dim, 0, ret_exprs, context)};

        std::vector<mlir::Value> ret_lbls_value;
        for (auto n : ret_lbls)
        {
          ret_lbls_value.push_back(all_lbls_value[n]);
        }

        std::vector<int64_t> result_dims = getDimSizes(ret_lbls_value);
        auto ret_tensor_type = mlir::RankedTensorType::get(result_dims, result_type);

        auto affineMapArrayAttr = builder.getAffineMapArrayAttr(affine_maps);

        SmallVector<mlir::StringRef, 8> formats;
        std::vector<mlir::Value> exprs{lhs_labeledtensor, rhs_labeledtensor};
        std::vector<mlir::Value> tensors;
        for (auto e : exprs)
        {
          if (isa<DenseTensorDeclOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";
            // infer the format
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(e.getDefiningOp()).format();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<DenseTensorDeclOp>(e.getDefiningOp()));
          }
          else if (isa<SparseTensorDeclOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";
            // infer the format
            auto lhs_format = dyn_cast<SparseTensorDeclOp>(e.getDefiningOp()).format();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<SparseTensorDeclOp>(e.getDefiningOp()));
          }
          else if (isa<LabeledTensorOp>(e.getDefiningOp()))
          {
            comet_debug() << " is LabeledTensorOp\n";
            mlir::Value tensordecl = e.getDefiningOp()->getOperand(0);
            if (isa<DenseTensorDeclOp>(tensordecl.getDefiningOp()))
            {
              comet_debug() << " is TensorDeclOp\n";
              // infer the format
              auto lhs_format = dyn_cast<DenseTensorDeclOp>(tensordecl.getDefiningOp()).format();
              comet_debug() << " lhs_format: " << lhs_format << "\n";
              formats.push_back(lhs_format);

              tensors.push_back(dyn_cast<DenseTensorDeclOp>(tensordecl.getDefiningOp()));
            }
            else if (isa<SparseTensorDeclOp>(tensordecl.getDefiningOp()))
            {
              comet_debug() << " is TensorDeclOp\n";
              // infer the format
              auto lhs_format = dyn_cast<SparseTensorDeclOp>(tensordecl.getDefiningOp()).format();
              comet_debug() << " lhs_format: " << lhs_format << "\n";
              formats.push_back(lhs_format);

              tensors.push_back(dyn_cast<SparseTensorDeclOp>(tensordecl.getDefiningOp()));
            }
            else if (isa<TensorMultOp>(tensordecl.getDefiningOp()))
            {
              comet_debug() << " is TensorMultOp\n";
              // infer the format
              mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorMultOp>(tensordecl.getDefiningOp()).formats();
              unsigned int i = opFormatsArrayAttr.size() - 1;
              std::string lhs_format(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
              comet_debug() << __LINE__ << " lhs_format: " << lhs_format << "\n";

              comet_debug() << " lhs_format: " << lhs_format << "\n";
              formats.push_back(lhs_format);

              tensors.push_back(dyn_cast<TensorMultOp>(tensordecl.getDefiningOp()).getOperation()->getResult(0));
            }
            else if (isa<TensorElewsMultOp>(e.getDefiningOp()))
            {
              comet_debug() << " is TensorMultOp\n";

              // infer the format
              mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorElewsMultOp>(e.getDefiningOp()).formats();
              unsigned int i = opFormatsArrayAttr.size() - 1;
              std::string lhs_format(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
              comet_debug() << __LINE__ << " lhs_format: " << lhs_format << "\n";

              comet_debug() << " lhs_format: " << lhs_format << "\n";
              formats.push_back(lhs_format);
              tensors.push_back(dyn_cast<TensorElewsMultOp>(e.getDefiningOp()).getOperation()->getResult(0));
            }
            else
            {
              comet_debug() << " not DenseTensorDecl op, not SparseTensorDecl op, not TensorMultOp, not TensorElewsMultOp\n";
            }
          }
          else if (isa<TensorMultOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorMultOp\n";
            // infer the format
            mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorMultOp>(e.getDefiningOp()).formats();
            unsigned int i = opFormatsArrayAttr.size() - 1;
            std::string lhs_format(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
            comet_debug() << __LINE__ << " lhs_format: " << lhs_format << "\n";
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<TensorMultOp>(e.getDefiningOp()).getOperation()->getResult(0));
          }
          else if (isa<TensorElewsMultOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorMultOp\n";
            // infer the format
            mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorElewsMultOp>(e.getDefiningOp()).formats();
            unsigned int i = opFormatsArrayAttr.size() - 1;
            std::string lhs_format(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
            comet_debug() << __LINE__ << " lhs_format: " << lhs_format << "\n";

            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<TensorElewsMultOp>(e.getDefiningOp()).getOperation()->getResult(0));
          }
          else if (isa<TransposeOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorMultOp\n";

            // infer the format
            mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TransposeOp>(e.getDefiningOp()).formats();
            unsigned int i = opFormatsArrayAttr.size() - 1;
            std::string lhs_format(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
            comet_debug() << __LINE__ << " lhs_format: " << lhs_format << "\n";

            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<TransposeOp>(e.getDefiningOp()).getOperation()->getResult(0));
          }
          else
          {
            comet_debug() << " not DenseTensorDecl op, not SparseTensorDecl op, not LabeledTensorOp, not TensorMultOp, not TensorElewsMultOp\n";
          }
        }
        comet_debug() << __LINE__ << " formats.size(): " << formats.size() << "\n";
        assert(formats.size() == 2 && " less than 2 input tensors\n");
        if (formats[0].compare("CSR") == 0 && formats[1].compare("CSR") == 0)
        {
          formats.push_back("CSR");
        }
        else
        {
          formats.push_back("Dense");
        }
        comet_debug() << " formats.size(): " << formats.size() << "\n";
        auto strAttr = builder.getStrArrayAttr(formats);

        assert(tensors.size() == 2 && " less than 2 input tensors for ta.tc or ta.elews_mul\n");

        // Derive the operation name from the binary operator. At the moment we
        // only support '+' and '*'.
        switch (binop.getOp())
        {
        case '+':
          return builder.create<AddOp>(location, lhs, rhs);
        case '-':
          return builder.create<SubstractOp>(location, lhs, rhs);
        case '*':
        {
          std::vector<mlir::Value> labels;
          for (auto i : ret_lbls)
          {
            labels.push_back(all_lbls_value[i]);
          }

          comet_vdump(lhs_tensor);
          comet_debug() << "\n";

          comet_vdump(rhs_tensor);
          comet_debug() << "\n";
          auto SemiringAttr = builder.getStringAttr("plusxy_times"); // this is for standard matrix multiplication
          mlir::Value tcop = builder.create<TensorMultOp>(location, ret_tensor_type, tensors[0], tensors[1],
                                                          labels, affineMapArrayAttr, strAttr, SemiringAttr);
          tcop.getDefiningOp()->setAttr("__alpha__", builder.getF64FloatAttr(1.0));
          tcop.getDefiningOp()->setAttr("__beta__", builder.getF64FloatAttr(0.0));

          comet_vdump(tcop);
          return tcop;
        }
        case '/':
          return builder.create<DivOp>(location, ret_tensor_type, lhs, rhs, lbls);

        case tok_elews:
          std::vector<mlir::Value> labels;
          for (auto i : ret_lbls)
          {
            labels.push_back(all_lbls_value[i]);
          }
          comet_vdump(lhs_tensor);
          comet_debug() << "\n";

          comet_vdump(rhs_tensor);
          comet_debug() << "\n";
          auto SemiringAttr = builder.getStringAttr("noop_times"); // this is for standard element-wise multiplication
          mlir::Value tcop = builder.create<TensorElewsMultOp>(location, ret_tensor_type, tensors[0], tensors[1], labels,
                                                               affineMapArrayAttr, strAttr, SemiringAttr);

          comet_vdump(tcop);
          return tcop;
        }
      }

      emitError(location, "invalid binary operator '") << binop.getOp() << "'";
      return nullptr;
    }

    /// This is a reference to a variable in an expression. The variable is
    /// expected to have been declared and so should have a value in the symbol
    /// table, otherwise emit an error and return nullptr.
    mlir::Value mlirGen(VariableExprAST &expr)
    {
      if (auto variable = symbolTable.lookup(expr.getName()))
        return variable;

      emitError(loc(expr.loc()), "Unknown variable VariableExprAST '")
          << expr.getName() << "'";
      return nullptr;
    }

    /// Emit a return operation. This will return failure if any generation fails.
    mlir::LogicalResult mlirGen(ReturnExprAST &ret)
    {
      auto location = loc(ret.loc());

      // 'return' takes an optional expression, handle that case here.
      mlir::Value expr = nullptr;
      if (ret.getExpr().hasValue())
      {
        if (!(expr = mlirGen(*ret.getExpr().getValue())))
          return mlir::failure();
      }

      // Otherwise, this return operation has zero operands.
      builder.create<TAReturnOp>(location, expr ? makeArrayRef(expr)
                                                : ArrayRef<mlir::Value>());
      return mlir::success();
    }

    mlir::Value mlirGen(LiteralExprAST &lit)
    {
      comet_debug() << __FILE__ << " " << __LINE__ << " mlirGen for LiteralExprAST.\n";

      auto type = getType(lit.getDims());

      // The attribute is a vector with a floating point value per element
      // (number) in the array, see `collectData()` below for more details.
      std::vector<double> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                   std::multiplies<int>()));
      collectData(lit, data);

      // The type of this attribute is tensor of 64-bit floating-point with the
      // shape of the literal.
      mlir::Type elementType = builder.getF64Type();
      auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

      // This is the actual attribute that holds the list of values for this
      // tensor literal.
      auto dataAttribute =
          mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

      // Build the MLIR op `ta.constant`. This invokes the `DenseConstantOp::build`
      // method.
      return builder.create<DenseConstantOp>(loc(lit.loc()), type, dataAttribute);
    }

    /// Recursive helper function to accumulate the data that compose an array
    /// literal. It flattens the nested structure in the supplied vector. For
    /// example with this array:
    ///  [[1, 2], [3, 4]]
    /// we will generate:
    ///  [ 1, 2, 3, 4 ]
    /// Individual numbers are represented as doubles.
    /// Attributes are the way MLIR attaches constant to operations.
    void collectData(ExprAST &expr, std::vector<double> &data)
    {
      if (auto *lit = dyn_cast<LiteralExprAST>(&expr))
      {
        for (auto &value : lit->getValues())
          collectData(*value, data);
        return;
      }

      assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
      data.push_back(cast<NumberExprAST>(expr).getValue());
    }

    /// Emit a call expression. It emits specific operations for the `sum`
    /// builtin. Other identifiers are assumed to be user-defined functions.
    mlir::Value mlirGen(CallExprAST &call)
    {
      llvm::StringRef callee = call.getCallee();
      auto location = loc(call.loc());

      mlir::Value sumVal;
      if (callee == "SUM")
      {
        auto *expr = call.getArgs();
        // Check if it SUM(A[i,j]) or SUM(A[i,j] * B[j,k])
        // Case 1: SUM(A[i,j])
        if (llvm::isa<LabeledTensorExprAST>(expr))
        {
          auto *rhsLT = llvm::cast<LabeledTensorExprAST>(expr);
          auto name = rhsLT->getTensorName();
          mlir::Value tensorValue = symbolTable.lookup(name);
          comet_debug() << " generate ta.sum op\n";
          sumVal = builder.create<SUMOp>(location, builder.getF64Type(), tensorValue);
        }

        // Case 2: SUM(A[i,j]*B[j,k])
        if (llvm::isa<BinaryExprAST>(expr))
        {
          comet_debug() << " SUM parameter is a BinaryExprAST, Generate ta.SUM() \n";
          // Generate ta.SUM
          // parse binary
          std::set<std::string> out_lbls = {};
          mlir::Value tensorValue = mlirGen(*expr, out_lbls);
          comet_debug() << " generate ta.sum op\n";
          sumVal = builder.create<SUMOp>(location, builder.getF64Type(), tensorValue);
        }
      }

      // Otherwise this is a call to a user-defined function. Calls to ser-defined
      // functions are mapped to a custom call that takes the callee name as an
      // attribute.
      return sumVal;
    }

    /// Emit a print expression. It emits specific operations for two builtins:
    /// transpose(x) and print(x).
    mlir::LogicalResult mlirGen(PrintExprAST &call)
    {
      auto arg = mlirGen(*call.getArg());
      if (!arg)
        return mlir::failure();

      builder.create<PrintOp>(loc(call.loc()), arg);
      return mlir::success();
    }

    /// Emit a constant for a single number (FIXME: semantic? broadcast?)
    mlir::Value mlirGen(NumberExprAST &num)
    {
      mlir::FloatAttr valueAttr = builder.getF64FloatAttr(num.getValue());
      return builder.create<ScalarAssignOp>(loc(num.loc()), builder.getF64Type(), valueAttr);
    }

    /// Emit a tensor expression
    /// D[a, d] = ....
    mlir::Value mlirGen(TensorOpExprAST &tensor_op)
    {
      comet_debug() << " mlirGen TensorOpExprAST\n";
      auto lhs = mlirGen(*tensor_op.getLHS());
      if (!lhs)
        return nullptr;
      comet_debug() << " get lhs\n";

      if (!isa<LabeledTensorOp>(lhs.getDefiningOp()))
      {
        emitError(loc(tensor_op.loc()),
                  "error: labeled tensor expression required '");
      }
      comet_debug() << " lhs is LabeledTensorOp \n";

      auto out_lbls_vec =
          cast<tensorAlgebra::LabeledTensorExprAST>(*tensor_op.getLHS())
              .getLabelNames();
      std::set<std::string> out_labels(out_lbls_vec.begin(), out_lbls_vec.end());

      comet_debug() << " out_labels: ";
      for (auto n : out_labels)
      {
        comet_debug() << n << " ";
      }
      comet_debug() << "\n";

      auto rhs = mlirGen(*tensor_op.getRHS(), out_labels);
      if (!rhs)
        return nullptr;
      comet_debug() << " get rhs\n";

      auto tens_beta = tensor_op.getBeta();

      auto ret_op = builder.create<TensorSetOp>(loc(tensor_op.loc()), rhs, lhs);
      ret_op.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));

      return rhs;
    }

    /// Emit a tensor expression
    mlir::Value mlirGen(LabeledTensorExprAST &lbl_tensor)
    {
      comet_debug() << " mlirGen LabeledTensorExprAST \n";
      auto tensor_name = lbl_tensor.getTensorName();
      auto label_names = lbl_tensor.getLabelNames();

      mlir::Value tensor_op;
      std::vector<mlir::Value> labels;

      if ((tensor_op = symbolTable.lookup(tensor_name)) != NULL)
      {
        if (!isa<DenseTensorDeclOp>(tensor_op.getDefiningOp()) && !isa<SparseTensorDeclOp>(tensor_op.getDefiningOp()))
        {
          emitError(loc(lbl_tensor.loc()), "Tensor declaration required '")
              << tensor_name << "'";
        }
      }
      else
      {
        emitError(loc(lbl_tensor.loc()),
                  "Unknown variable LabeledTensorExprAST'");
      }

      for (const auto &lbl_str : label_names)
      {
        if (auto var = symbolTable.lookup(lbl_str))
        {
          if (isa<IndexLabelStaticOp>(var.getDefiningOp()))
          {
            labels.push_back(var);
          }
          else if (isa<IndexLabelDynamicOp>(var.getDefiningOp()))
          {
            labels.push_back(var);
          }
          else
          {
            emitError(loc(lbl_tensor.loc()), "Index label variable required '")
                << lbl_str << "'";
          }
        }
        else
        {
          emitError(loc(lbl_tensor.loc()),
                    " Unknown variable LabeledTensorExprAST' ")
              << lbl_str << "'";
        }
      }

      mlir::Value value;
      if (isa<DenseTensorDeclOp>(tensor_op.getDefiningOp()))
      {
        auto return_type =
            cast<DenseTensorDeclOp>(tensor_op.getDefiningOp()).getResult().getType();
        value = builder.create<LabeledTensorOp>(
            loc(lbl_tensor.loc()), return_type, tensor_op, labels);
      }
      else if (isa<SparseTensorDeclOp>(tensor_op.getDefiningOp()))
      {
        auto return_type =
            cast<SparseTensorDeclOp>(tensor_op.getDefiningOp()).getResult().getType();
        value = builder.create<LabeledTensorOp>(
            loc(lbl_tensor.loc()), return_type, tensor_op, labels);
      }

      return value;
    }

    /// Dispatch codegen for the right expression subclass using RTTI.
    mlir::Value mlirGen(ExprAST &expr,
                        const std::set<std::string> out_lbls = {})
    {
      comet_debug() << " mlirGen ExprAST " << expr.getKind() << " \n";
      switch (expr.getKind())
      {
      case tensorAlgebra::ExprAST::Expr_BinOp:
        return mlirGen(cast<BinaryExprAST>(expr), out_lbls);
      case tensorAlgebra::ExprAST::Expr_Var:
        return mlirGen(cast<VariableExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Literal:
        return mlirGen(cast<LiteralExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Call:
        return mlirGen(cast<CallExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Transpose:
        return mlirGen(cast<TransposeExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Num:
        return mlirGen(cast<NumberExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_LabeledTensor:
        comet_debug() << " Is Expr_LabeledTensor\n";
        return mlirGen(cast<LabeledTensorExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Tensor:
        comet_debug() << " Is Expr_Tensor\n";
        return mlirGen(cast<TensorOpExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_GetTime:
        return mlirGen(cast<GetTimeExprAST>(expr));
      default:
        emitError(loc(expr.loc()))
            << "MLIR codegen encountered an unhandled expr kind '"
            << Twine(expr.getKind()) << "'";
        return nullptr;
      }
    }

    mlir::Value mlirGen(GetTimeExprAST &gettime)
    {
      return builder.create<GetTimeOp>(loc(gettime.loc()), builder.getF64Type());
    }

    /// Emit a print expression. It emits specific operations for two builtins:
    /// transpose(x) and print(x).
    mlir::LogicalResult mlirGen(PrintElapsedTimeExprAST &call)
    {
      auto start = mlirGen(*call.getStart());
      if (!start)
        return mlir::failure();

      auto end = mlirGen(*call.getEnd());
      if (!end)
        return mlir::failure();

      builder.create<PrintElapsedTimeOp>(loc(call.loc()), start, end);
      return mlir::success();
    }

    /// Handle a variable declaration, we'll codegen the expression that forms the
    /// initializer and record the value in the symbol table before returning it.
    /// Future expressions will be able to reference this variable through symbol
    /// table lookup.
    mlir::Value mlirGen(VarDeclExprAST &vardecl)
    {
      auto init = vardecl.getInitVal();
      if (!init)
      {
        emitError(loc(vardecl.loc()),
                  "missing initializer in variable declaration");
        return nullptr;
      }

      mlir::Value value = mlirGen(*init);
      if (!value)
        return nullptr;

      // We have the initializer value, but in case the variable was declared
      // with specific shape, we emit a "reshape" operation. It will get
      // optimized out later as needed.
      if (!vardecl.getType().shape.empty())
      {
        value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                          getType(vardecl.getType()), value);
      }

      // Register the value in the symbol table.
      if (failed(declare(vardecl.getName(), value)))
        return nullptr;
      return value;
    }

    /// Handle index label declaration
    mlir::Value mlirGen(IndexLabelDeclExprAST &labeldecl)
    {

      mlir::Value lo = builder.create<mlir::ConstantIndexOp>(
          loc(labeldecl.loc()), labeldecl.getBegin());
      mlir::Value hi = builder.create<mlir::ConstantIndexOp>(loc(labeldecl.loc()),
                                                             labeldecl.getEnd());
      mlir::Value step = builder.create<mlir::ConstantIndexOp>(
          loc(labeldecl.loc()), labeldecl.getIncrement());

      mlir::Value value;
      if (labeldecl.getEnd() == mlir::ShapedType::kDynamicSize)
      {
        value =
            builder.create<IndexLabelDynamicOp>(loc(labeldecl.loc()), lo, step);
      }
      else
      {
        value =
            builder.create<IndexLabelStaticOp>(loc(labeldecl.loc()), lo, hi, step);
      }

      if (failed(declare(labeldecl.getName(), value)))
        return nullptr;
      return value;
    }

    /// Handle index label declaration
    mlir::Value mlirGen(IndexLabelDeclDynamicExprAST &labeldecl)
    {

      mlir::Value lo = builder.create<mlir::ConstantIndexOp>(
          loc(labeldecl.loc()), labeldecl.getBegin());
      mlir::Value step = builder.create<mlir::ConstantIndexOp>(
          loc(labeldecl.loc()), labeldecl.getIncrement());

      mlir::Value value =
          builder.create<IndexLabelDynamicOp>(loc(labeldecl.loc()), lo, step);
      if (failed(declare(labeldecl.getName(), value)))
        return nullptr;
      return value;
    }

    /// Handle tensor declaration
    mlir::Value mlirGen(TensorDeclExprAST &tensordecl)
    {

      auto dim_lbls = tensordecl.getDims();
      std::vector<int64_t> dims_sizes;
      std::vector<mlir::Value> labels;
      auto tensor_format = tensordecl.getFormat();
      for (const auto &lbl_str : dim_lbls)
      {
        if (auto var = symbolTable.lookup(lbl_str))
        {
          if (isa<IndexLabelStaticOp>(var.getDefiningOp()))
          {
            labels.push_back(var);
            auto range = cast<IndexLabelStaticOp>(var.getDefiningOp());
            auto min_idx =
                cast<mlir::ConstantIndexOp>(range.min().getDefiningOp());
            auto max_idx =
                cast<mlir::ConstantIndexOp>(range.max().getDefiningOp());
            auto step_idx =
                cast<mlir::ConstantIndexOp>(range.step().getDefiningOp());

            auto min = min_idx.getValue();
            auto max = max_idx.getValue();
            auto step = step_idx.getValue();
            if (max == mlir::ShapedType::kDynamicSize)
            {
              dims_sizes.push_back(mlir::ShapedType::kDynamicSize);
            }
            else
            {
              dims_sizes.push_back((max - min) / step);
            }
          }
          else if (isa<IndexLabelDynamicOp>(var.getDefiningOp()))
          {
            labels.push_back(var);
            dims_sizes.push_back(mlir::ShapedType::kDynamicSize);
          }
          else
          {
            emitError(loc(tensordecl.loc()), "Index label variable required '")
                << lbl_str << "'";
          }
        }
        else
        {
          emitError(loc(tensordecl.loc()), "Unknown variable TensorDeclExprAST '")
              << lbl_str << "'";
        }
      }

      auto vartype = tensordecl.getElementType();
      vartype.shape = dims_sizes;
      auto tensor_type = getTensorType(vartype);
      auto name = tensordecl.getName();

      mlir::Value value;
      std::string formats_str(tensor_format.data());
      if (isDense(formats_str, ", ") == false)
      {
        value = builder.create<SparseTensorDeclOp>(loc(tensordecl.loc()),
                                                   tensor_type, labels, tensor_format);
      }
      else
      {
        value = builder.create<DenseTensorDeclOp>(loc(tensordecl.loc()),
                                                  tensor_type, labels, tensor_format);
      }

      if (failed(declare(name, value)))
        return nullptr;
      return value;
    }

    /// Handle tensor declaration
    mlir::Value mlirGen(OutputTensorDeclExprAST &tensordecl)
    {
      comet_debug() << "OutputTensorDeclExprAST \n";

      auto dim_lbls = tensordecl.getDims();
      std::vector<int64_t> dims_sizes;
      std::vector<mlir::Value> labels;
      auto tensor_format = tensordecl.getFormat();
      for (const auto &lbl_str : dim_lbls)
      {
        if (auto var = symbolTable.lookup(lbl_str))
        {
          if (isa<IndexLabelStaticOp>(var.getDefiningOp()))
          {
            labels.push_back(var);
            auto range = cast<IndexLabelStaticOp>(var.getDefiningOp());
            auto min_idx =
                cast<mlir::ConstantIndexOp>(range.min().getDefiningOp());
            auto max_idx =
                cast<mlir::ConstantIndexOp>(range.max().getDefiningOp());
            auto step_idx =
                cast<mlir::ConstantIndexOp>(range.step().getDefiningOp());

            auto min = min_idx.getValue();
            auto max = max_idx.getValue();
            auto step = step_idx.getValue();

            if (max == mlir::ShapedType::kDynamicSize)
            {
              dims_sizes.push_back(mlir::ShapedType::kDynamicSize);
            }
            else
            {
              dims_sizes.push_back((max - min) / step);
            }
          }
          else if (isa<IndexLabelDynamicOp>(var.getDefiningOp()))
          {

            labels.push_back(var);
            dims_sizes.push_back(mlir::ShapedType::kDynamicSize);
          }
          else
          {
            emitError(loc(tensordecl.loc()), "Index label variable required '")
                << lbl_str << "'";
          }
        }
        else
        {
          emitError(loc(tensordecl.loc()), "Unknown variable TensorDeclExprAST '")
              << lbl_str << "'";
        }
      }

      auto vartype = tensordecl.getElementType();
      vartype.shape = dims_sizes;
      auto tensor_type = getTensorType(vartype);
      auto name = tensordecl.getName();

      mlir::Value value = builder.create<SparseOutputTensorDeclOp>(loc(tensordecl.loc()),
                                                                   tensor_type, labels, tensor_format);
      comet_vdump(value);

      if (failed(declare(name, value)))
        return nullptr;
      return value;
    }

    // Handle B[j, i] = tranpose(A[i,j], {j, i}) in DSL
    mlir::Value mlirGen(TransposeExprAST &transpose, LabeledTensorExprAST &lhsLT)
    {
      comet_debug() << "TransposeExprAST \n";

      mlir::Value rhs_tensor = symbolTable.lookup(transpose.getName());

      comet_vdump(rhs_tensor);

      auto rhs_lbls = transpose.getSrcDims();
      auto lhs_lbls = transpose.getDstDims();

      StringSet all_lbls;
      all_lbls.insert(rhs_lbls.begin(), rhs_lbls.end());
      all_lbls.insert(lhs_lbls.begin(), lhs_lbls.end());

      std::map<std::string, mlir::AffineExpr> expr_map;
      unsigned dim = 0;
      for (const auto &lbl : all_lbls)
      {
        expr_map[lbl] = getAffineDimExpr(dim++, builder.getContext());
      }
      std::vector<mlir::AffineExpr> rhs_exprs;
      std::vector<mlir::AffineExpr> lhs_exprs;

      for (const auto &lbl : rhs_lbls)
      {
        rhs_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : lhs_lbls)
      {
        lhs_exprs.push_back(expr_map[lbl]);
      }

      auto context = builder.getContext();

      SmallVector<mlir::AffineMap, 8> affine_maps{
          mlir::AffineMap::get(dim, 0, rhs_exprs, context),
          mlir::AffineMap::get(dim, 0, lhs_exprs, context)};

      auto affineMapArrayAttr = builder.getAffineMapArrayAttr(affine_maps);

      SmallVector<mlir::StringRef, 8> formats;

      // Firstly, Look at the rhs
      if (isa<DenseTensorDeclOp>(rhs_tensor.getDefiningOp()))
      {
        comet_debug() << " is TensorDeclOp\n";
        // infer the format
        auto rhs_format = dyn_cast<DenseTensorDeclOp>(rhs_tensor.getDefiningOp()).format();
        comet_debug() << " rhs_format: " << rhs_format << "\n";
        formats.push_back(rhs_format);
      }
      else if (isa<SparseTensorDeclOp>(rhs_tensor.getDefiningOp()))
      {
        comet_debug() << " is TensorDeclOp\n";
        // infer the format
        auto rhs_format = dyn_cast<SparseTensorDeclOp>(rhs_tensor.getDefiningOp()).format();
        comet_debug() << " rhs_format: " << rhs_format << "\n";
        formats.push_back(rhs_format);
      }
      else
      {
        comet_debug() << " not TensorDeclOp\n";
      }

      // Secondly, Look at the lhs
      std::vector<LabeledTensorExprAST *> exprs{&lhsLT};
      std::vector<mlir::Value> lhs_lbls_value;
      for (auto e : exprs)
      {
        auto lhsLT_tensor_name = e->getTensorName();
        mlir::Value lhsLT_op;
        if ((lhsLT_op = symbolTable.lookup(lhsLT_tensor_name)) != NULL)
        {
          // comet_pdump(lhsLT_op.getDefiningOp());
          if (isa<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";
            // infer the format
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()).format();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            for (unsigned int i = 0; i < lhsLT_op.getDefiningOp()->getNumOperands(); i++)
            {
              lhs_lbls_value.push_back(lhsLT_op.getDefiningOp()->getOperand(i));
            }
          }
          else if (isa<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";
            // infer the format
            auto lhs_format = dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()).format();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);
            for (unsigned int i = 0; i < lhsLT_op.getDefiningOp()->getNumOperands(); i++)
            {
              lhs_lbls_value.push_back(lhsLT_op.getDefiningOp()->getOperand(i));
            }
          }
          else
          {
            comet_debug() << " not TensorDeclOp\n";
          }
        }
      }
      comet_debug() << " formats.size(): " << formats.size() << "\n";
      auto strAttr = builder.getStrArrayAttr(formats);

      // auto rhs_tensor = symbolTable.lookup(rhsLT->getTensorName());
      auto lhs_tensor = symbolTable.lookup(lhsLT.getTensorName());

      comet_debug() << " create TransposeOp\n";
      mlir::Value t = builder.create<TransposeOp>(loc(transpose.loc()), lhs_tensor.getType(),
                                                  rhs_tensor, lhs_lbls_value, affineMapArrayAttr, strAttr);
      builder.create<TensorSetOp>(loc(transpose.loc()), t.getDefiningOp()->getResult(0), lhs_tensor);
      comet_vdump(t);

      return t;
    }

    /// Codegen for-loop 
    mlir::LogicalResult mlirGen(ForLoopExprAST &forLoop)
    {
      comet_debug() << "codegen: ForLoopExprAST \n";

      mlir::Value lo = builder.create<mlir::ConstantIndexOp>(
          loc(forLoop.loc()), forLoop.getBegin());
      mlir::Value hi = builder.create<mlir::ConstantIndexOp>(loc(forLoop.loc()),
                                                             forLoop.getEnd());
      mlir::Value step = builder.create<mlir::ConstantIndexOp>(
          loc(forLoop.loc()), forLoop.getIncrement());

      mlir::Value value = builder.create<IndexLabelStaticOp>(loc(forLoop.loc()), lo, hi, step);
    
      builder.create<ForLoopStartOp>(loc(forLoop.loc()), value, forLoop.getName());

      if (failed(declare(forLoop.getName(), value)))
        return mlir::failure();

      comet_debug() << "codegen: ForLoopExprAST done \n";
      return mlir::success();
    }

    /// Codegen for-loop 
    mlir::LogicalResult mlirGen(ForLoopEndExprAST &forLoopEnd)
    {
      comet_debug() << "codegen: ForLoopEndExprAST \n";

      builder.create<ForLoopEndOp>(loc(forLoopEnd.loc()));

      comet_debug() << "codegen: ForLoopEndExprAST done \n";
      return mlir::success();
    }

    /// Codegen a list of expression, return failure if one of them hit an error.
    mlir::LogicalResult mlirGen(ExprASTList &blockAST)
    {
      ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);
      for (auto &expr : blockAST)
      {
        // Specific handling for variable declarations, return statement, and
        // print. These can only appear in block list and not in nested
        // expressions.

        if (auto *labeldecl = dyn_cast<IndexLabelDeclExprAST>(expr.get()))
        {
          if (!mlirGen(*labeldecl))
            return mlir::failure();
          continue;
        }
        if (auto *labeldecl = dyn_cast<IndexLabelDeclDynamicExprAST>(expr.get()))
        {
          if (!mlirGen(*labeldecl))
            return mlir::failure();
          continue;
        }

        if (auto *tensordecl = dyn_cast<TensorDeclExprAST>(expr.get()))
        {
          if (!mlirGen(*tensordecl))
            return mlir::failure();
          continue;
        }

        if (auto *tensordecl = dyn_cast<OutputTensorDeclExprAST>(expr.get()))
        {
          if (!mlirGen(*tensordecl))
            return mlir::failure();
          continue;
        }

        if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get()))
        {
          if (!mlirGen(*vardecl))
            return mlir::failure();
          continue;
        }
        if (auto *transpose = dyn_cast<TransposeExprAST>(expr.get()))
        {
          if (!mlirGen(*transpose))
            return mlir::failure();
          continue;
        }
        if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
          return mlirGen(*ret);
        if (auto *print = dyn_cast<PrintExprAST>(expr.get()))
        {
          if (mlir::failed(mlirGen(*print)))
            return mlir::success();
          continue;
        }

        if (auto *printElapsedTime = dyn_cast<PrintElapsedTimeExprAST>(expr.get()))
        {
          if (mlir::failed(mlirGen(*printElapsedTime)))
            return mlir::success();
          continue;
        }

        if (auto *forLoopStart = dyn_cast<ForLoopExprAST>(expr.get()))
        {
          if (mlir::failed(mlirGen(*forLoopStart)))
            return mlir::success();
          continue;
        }

        if (auto *forLoopEnd = dyn_cast<ForLoopEndExprAST>(expr.get()))
        {
          if (mlir::failed(mlirGen(*forLoopEnd)))
            return mlir::success();
          continue;
        }

        if (auto *tensor_op = dyn_cast<TensorOpExprAST>(expr.get()))
        {
          comet_debug() << " generate ops for TensorOpExprAST\n";
          comet_debug() << " Right hand side " << tensor_op->getRHS()->getKind() << "\n";
          comet_debug() << " Left hand side " << tensor_op->getLHS()->getKind() << "\n";
          /// A[i,j] = ...
          if (tensor_op->getLHS()->getKind() ==
              ExprAST::ExprASTKind::Expr_LabeledTensor)
          {
            comet_debug() << " in TensorOpExprAST, lhs is labeledTensor\n";

            /// A[i,j] = B[i,k] * C[k,j]
            if (tensor_op->getRHS()->getKind() ==
                    ExprAST::ExprASTKind::Expr_BinOp &&
                llvm::cast<BinaryExprAST>(tensor_op->getRHS())
                        ->getLHS()
                        ->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor &&
                llvm::cast<BinaryExprAST>(tensor_op->getRHS())
                        ->getRHS()
                        ->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is BinaryExprAST and its lhs and rhs are Expr_LabeledTensor\n";

              if (mlir::failed(mlirGenTensorContraction(*tensor_op)))
                return mlir::success();
              continue;
            }
            /// A[i,j] = 1.0
            else if (tensor_op->getRHS()->getKind() ==
                     ExprAST::ExprASTKind::Expr_Num)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is Expr_Num\n";

              auto tensor_name =
                  llvm::cast<LabeledTensorExprAST>(tensor_op->getLHS())
                      ->getTensorName();
              auto value =
                  llvm::cast<NumberExprAST>(tensor_op->getRHS())->getValue();

              if (mlir::failed(mlirGenTensorFill(loc(tensor_op->loc()), tensor_name, value)))
                return mlir::success();
              continue;
            }

            /// A[i,j] = read_from_file()
            else if (tensor_op->getRHS()->getKind() == ExprAST::ExprASTKind::Expr_Call)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is Expr_Call\n";
              auto tensor_name =
                  llvm::cast<LabeledTensorExprAST>(tensor_op->getLHS())
                      ->getTensorName();
              auto call = llvm::cast<CallExprAST>(tensor_op->getRHS());
              llvm::StringRef callee = call->getCallee();

              // Builting calls have their custom operation, meaning this is a
              // straightforward emission.
              if (callee == "read_from_file")
              {
                comet_debug() << " call read_from_file \n";

                ExprAST *filenames = call->getArgs();
                comet_debug() << "\n";
                std::string filenamestring;
                llvm::StringRef filenamestr;
                if (filenames == nullptr)
                {
                  comet_debug() << __LINE__ << " Empty filename\n";
                  filenamestring = "SPARSE_FILE_NAME";
                  filenamestr = filenamestring;
                }
                else
                { // Not empty filename
                  comet_debug() << __LINE__ << " An argument was provided in read_from_file().\n";

                  // User will provide num arg in read_from_file()
                  // that will be used to read file based on unique env vars.
                  // e.g., read_from_file(0) --> SPARSE_FILE_NAME0
                  if (filenames->getKind() == NumberExprAST::Expr_Num)
                  {
                    comet_debug() << __LINE__ << "\n";
                    auto *filenameast = llvm::cast<NumberExprAST>(filenames);
                    comet_debug() << __LINE__ << "\n";

                    // get arg val
                    int val = (int)cast<NumberExprAST>(filenameast)->getValue();
                    filenamestring = "SPARSE_FILE_NAME" + std::to_string(val);
                    filenamestr = filenamestring;

                    comet_debug() << " " << filenamestr << "\n";
                  }

                  if (filenames->getKind() == ExprAST::ExprASTKind::Expr_Var)
                  {
                    comet_debug() << __LINE__ << "\n";
                    auto *filenameast = llvm::cast<VariableExprAST>(filenames);
                    comet_debug() << __LINE__ << "\n";

                    filenamestr = filenameast->getName();
                    comet_debug() << " " << filenamestr << "\n";
                  }
                }

                if (mlir::failed(mlirGenTensorFillFromFile(loc(tensor_op->loc()), tensor_name, filenamestr)))
                  return mlir::success();
              }
              if (callee == "random")
              {
                comet_debug() << " call random \n";

                std::random_device os_seed;
                srand(static_cast<unsigned>(os_seed())); // seed the random-num generator
                                                         // something like time(0) does not work!

                if (mlir::failed(mlirGenTensorFillRandom(loc(tensor_op->loc()), tensor_name)))
                  return mlir::success();
              }
              continue;
            }
            /// A[i,j] = transpose(B[j,i], {i,j})
            else if (tensor_op->getRHS()->getKind() == ExprAST::ExprASTKind::Expr_Transpose)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is Expr_Transpose\n";
              // create transpose op
              LabeledTensorExprAST *lhsLabeledTensorExprAST = llvm::cast<LabeledTensorExprAST>(tensor_op->getLHS());

              TransposeExprAST *transpose = llvm::cast<TransposeExprAST>(tensor_op->getRHS());
              mlirGen(*transpose, *lhsLabeledTensorExprAST);
              continue;
            }
            else if ((tensor_op->getRHS()->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor &&
                      tensor_op->getRHS()->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor))
            {

              if (mlir::failed(mlirGenTensorarithexprs(*tensor_op)))
                return mlir::success();
              continue;
            }
            // there is a fall through case at the end, so don't put an else here.
          }
          else
          {
            comet_debug() << " in TensorOpExprAST, lhs is NOT labeledTensor\n";
          }
        }

        // Generic expression dispatch codegen.
        comet_debug() << " expr->getKind(): " << expr->getKind() << "\n";
        if (!mlirGen(*expr))
          return mlir::failure();
      }
      return mlir::success();
    }

    /// Build a tensor type from a list of shape dimensions.
    mlir::Type getType(ArrayRef<int64_t> shape)
    {
      // If the shape is empty, then this type is unranked.
      if (shape.empty())
        return mlir::UnrankedTensorType::get(builder.getF64Type());

      // Otherwise, we use the given shape.
      return mlir::RankedTensorType::get(shape, builder.getF64Type());
    }

    /// Build an MLIR type from a Tensor Algebra AST variable type (forward to the
    /// generic getType above).
    mlir::Type getType(const VarType &type) { return getType(type.shape); }

    /// Build an MLIR type from a Tensor Algebra AST variable type with respect to
    /// the element type
    mlir::Type getTensorType(const VarType &type)
    {
      if (type.shape.empty())
        return mlir::UnrankedTensorType::get(builder.getF64Type());

      mlir::Type retType;
      switch (type.elt_ty)
      {
      case VarType::TY_FLOAT:
        retType = mlir::RankedTensorType::get(type.shape, builder.getF32Type());
        break;
      case VarType::TY_DOUBLE:
        retType = mlir::RankedTensorType::get(type.shape, builder.getF64Type());
        break;
      case VarType::TY_INT:
        retType = mlir::RankedTensorType::get(type.shape,
                                              builder.getIntegerType(64));
        break;
      }

      return retType;
    }

    ElType convertType(mlir::Type type)
    {

      if (type.isF32())
        return ElType::fp32;
      else if (type.isF64())
        return ElType::fp64;
      else if (type.isInteger(64))
        return ElType::i64;
      else
        return ElType::inv;
    }

    mlir::Type getBinOpResultType(mlir::Type lhs, mlir::Type rhs)
    {
      ElType type = lub(convertType(lhs), convertType(rhs));

      switch (type)
      {
      case ElType::i64:
        return builder.getIntegerType(64);
      case ElType::fp32:
        return builder.getF32Type();
      case ElType::fp64:
        return builder.getF64Type();
      case ElType::inv:
        return builder.getF64Type();
      default:
        return builder.getF64Type();
        break;
      }
    }

    std::vector<int64_t> getDimSizes(const std::vector<std::string> &labels)
    {
      std::vector<int64_t> dims_sizes;
      for (const auto &lbl_str : labels)
      {
        auto lbl = symbolTable.lookup(lbl_str);
        if (isa<IndexLabelStaticOp>(lbl.getDefiningOp()))
        {
          comet_debug() << "\n";
          auto range = cast<IndexLabelStaticOp>(lbl.getDefiningOp());
          auto min_idx = cast<mlir::ConstantIndexOp>(range.min().getDefiningOp());
          auto max_idx = cast<mlir::ConstantIndexOp>(range.max().getDefiningOp());
          auto step_idx = cast<mlir::ConstantIndexOp>(range.step().getDefiningOp());

          auto min = min_idx.getValue();
          auto max = max_idx.getValue();
          auto step = step_idx.getValue();

          if (max == mlir::ShapedType::kDynamicSize)
          {
            dims_sizes.push_back(mlir::ShapedType::kDynamicSize);
          }
          else
          {
            dims_sizes.push_back((max - min) / step);
          }
          comet_debug() << "\n";
        }
        else if (isa<IndexLabelDynamicOp>(lbl.getDefiningOp()))
        {
          dims_sizes.push_back(mlir::ShapedType::kDynamicSize);
        }
        else
        {
          llvm::errs() << "Neither IndexLabelStaticOp nor IndexLabelDynamicOp\n";
        }
      }

      return dims_sizes;
    }

    std::vector<int64_t> getDimSizes(std::vector<mlir::Value> &labels)
    {
      std::vector<int64_t> dims_sizes;
      for (auto &lbl : labels)
      {
        if (isa<IndexLabelStaticOp>(lbl.getDefiningOp()))
        {
          auto range = cast<IndexLabelStaticOp>(lbl.getDefiningOp());
          auto min_idx = cast<mlir::ConstantIndexOp>(range.min().getDefiningOp());
          auto max_idx = cast<mlir::ConstantIndexOp>(range.max().getDefiningOp());
          auto step_idx = cast<mlir::ConstantIndexOp>(range.step().getDefiningOp());

          auto min = min_idx.getValue();
          auto max = max_idx.getValue();
          auto step = step_idx.getValue();

          if (max == mlir::ShapedType::kDynamicSize)
          {
            dims_sizes.push_back(mlir::ShapedType::kDynamicSize);
          }
          else
          {
            dims_sizes.push_back((max - min) / step);
          }
        }
        else if (isa<IndexLabelDynamicOp>(lbl.getDefiningOp()))
        {
          dims_sizes.push_back(mlir::ShapedType::kDynamicSize);
        }
        else
        {
          llvm::errs() << "Neither IndexLabelStaticOp nor IndexLabelDynamicOp\n";
        }
      }

      return dims_sizes;
    }

    mlir::LogicalResult mlirGenTensorarithexprs(TensorOpExprAST &tensor_op)
    {

      LabeledTensorExprAST *lhsLT =
          llvm::cast<LabeledTensorExprAST>(tensor_op.getLHS());
      LabeledTensorExprAST *rhsLT =
          llvm::cast<LabeledTensorExprAST>(tensor_op.getRHS());

      std::vector<mlir::Value> tensors;
      auto binop = tensor_op.getOp();
      SmallVector<mlir::StringRef, 8> formats;
      std::vector<LabeledTensorExprAST *> exprs{rhsLT, lhsLT};
      for (auto e : exprs)
      {
        auto lhsLT_tensor_name = e->getTensorName();
        mlir::Value lhsLT_op;
        if ((lhsLT_op = symbolTable.lookup(lhsLT_tensor_name)) != NULL)
        {
          if (isa<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            // infer the format
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()).format();
            formats.push_back(lhs_format);
            tensors.push_back(dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()));
          }
          else if (isa<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            // infer the format
            auto lhs_format = dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()).format();
            formats.push_back(lhs_format);
            tensors.push_back(dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()));
          }
          else
          {
            comet_debug() << " not TensorDeclOp\n";
          }
        }
      }

      std::vector<mlir::Operation *> lhsLabelOps, rhsLabelOps;
      if (binop == TensorOpKind::Tensor_Red_Add)
      {

        auto op = builder.create<AddOp>(loc(tensor_op.loc()), tensors[1], tensors[0]);
        builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[1]);
      }

      else if (binop == TensorOpKind::Tensor_Red_Sub)
      {

        auto op = builder.create<SubstractOp>(loc(tensor_op.loc()), tensors[1], tensors[0]);
        builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[1]);
      }

      return mlir::success();
    }

    /// handle case: Expr_LabeledTensor = Expr_LabeledTensor * Expr_LabeledTensor
    mlir::LogicalResult mlirGenTensorContraction(TensorOpExprAST &tensor_op)
    {
      assert(tensor_op.getLHS()->getKind() ==
             ExprAST::ExprASTKind::Expr_LabeledTensor);
      assert(tensor_op.getRHS()->getKind() == ExprAST::ExprASTKind::Expr_BinOp);

      LabeledTensorExprAST *lhsLT =
          llvm::cast<LabeledTensorExprAST>(tensor_op.getLHS());
      BinaryExprAST *rhsBinOp = llvm::cast<BinaryExprAST>(tensor_op.getRHS());
      auto tens_beta = tensor_op.getBeta();
      if (rhsBinOp->getLHS()->getKind() !=
              ExprAST::ExprASTKind::Expr_LabeledTensor ||
          rhsBinOp->getRHS()->getKind() !=
              ExprAST::ExprASTKind::Expr_LabeledTensor)
      {
        emitError(loc(tensor_op.loc()), "error: the tensor operation allows only "
                                        "two labeled tensor operands '");
      }

      LabeledTensorExprAST *rhs1LT =
          llvm::cast<LabeledTensorExprAST>(rhsBinOp->getLHS());
      LabeledTensorExprAST *rhs2LT =
          llvm::cast<LabeledTensorExprAST>(rhsBinOp->getRHS());
      auto binop = rhsBinOp->getOp();
      comet_debug() << __FILE__ << " " << __LINE__ << " binop: " << binop << "\n";

      int SemiringOp1st = int(rhsBinOp->getSemiringOp1());
      int SemiringOp2nd = int(rhsBinOp->getSemiringOp2());
      comet_debug() << __FILE__ << " " << __LINE__ << " semirop: 1) " << SemiringOp1st << " 2) " << SemiringOp2nd << "\n";
      std::string SemiringOp1str = getSemiringOpName(SemiringOp1st);
      comet_debug() << __FILE__ << " " << __LINE__ << " SemiringOp1str " << SemiringOp1str << "\n";
      std::string SemiringOp2str = getSemiringOpName(SemiringOp2nd);
      comet_debug() << __FILE__ << " " << __LINE__ << " SemiringOp2str " << SemiringOp2str << "\n";
      std::string SemiringOperators = SemiringOp1str + "_" + SemiringOp2str;
      auto SemiringAttr = builder.getStringAttr(SemiringOperators);

      auto lhs_lbls = lhsLT->getLabelNames();
      auto rhs1_lbls = rhs1LT->getLabelNames();
      auto rhs2_lbls = rhs2LT->getLabelNames();
      auto all_lbls = rhsBinOp->getLabels();

      std::vector<mlir::Value> lhs_lbls_value;
      for (auto n : lhs_lbls)
      {
        lhs_lbls_value.push_back(symbolTable.lookup(n));
      }

      std::map<std::string, mlir::AffineExpr> expr_map;
      unsigned dim = 0;
      for (const auto &lbl : all_lbls)
      {
        expr_map[lbl] = getAffineDimExpr(dim++, builder.getContext());
      }
      std::vector<mlir::AffineExpr> rhs1_exprs;
      std::vector<mlir::AffineExpr> rhs2_exprs;
      std::vector<mlir::AffineExpr> lhs_exprs;

      for (const auto &lbl : rhs1_lbls)
      {
        rhs1_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : rhs2_lbls)
      {
        rhs2_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : lhs_lbls)
      {
        lhs_exprs.push_back(expr_map[lbl]);
      }

      auto context = builder.getContext();
      SmallVector<mlir::AffineMap, 8> affine_maps{
          mlir::AffineMap::get(dim, 0, rhs1_exprs, context),
          mlir::AffineMap::get(dim, 0, rhs2_exprs, context),
          mlir::AffineMap::get(dim, 0, lhs_exprs, context)};

      auto affineMapArrayAttr = builder.getAffineMapArrayAttr(affine_maps);

      SmallVector<mlir::StringRef, 8> formats;
      std::vector<LabeledTensorExprAST *> exprs{rhs1LT, rhs2LT, lhsLT};
      std::vector<mlir::Value> tensors;
      for (auto e : exprs)
      {
        auto lhsLT_tensor_name = e->getTensorName();
        mlir::Value lhsLT_op;
        if ((lhsLT_op = symbolTable.lookup(lhsLT_tensor_name)) != NULL)
        {
          if (isa<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";

            // infer the format
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()).format();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()));
          }
          else if (isa<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";

            // infer the format
            auto lhs_format = dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()).format();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()));
          }
          else
          {
            comet_debug() << " not TensorDeclOp\n";
          }
        }
      }
      comet_debug() << " formats.size(): " << formats.size() << "\n";
      auto strAttr = builder.getStrArrayAttr(formats);

      assert(tensors.size() == 3 && "Not 3 tensors for ta.tc or ta.elews_mul\n");

      if (binop != '*' && binop != tok_elews &&
          binop != tok_semiring && binop != tok_monoid)
      {
        emitError(loc(tensor_op.loc()),
                  "error: only tensor multiplication and elementwise multiplication operations are supported "
                  "for tensor ops'");
      }

      if (binop == '*' || binop == tok_semiring)
      {
        // auto SemiringAttr = builder.getStringAttr("plus_times");  // this is for standard matrix multiplication
        auto op = builder.create<TensorMultOp>(loc(tensor_op.loc()),
                                               tensors[2].getType(),
                                               tensors[0], tensors[1],
                                               lhs_lbls_value,
                                               affineMapArrayAttr,
                                               strAttr, SemiringAttr);
        op.getOperation()->setAttr("__alpha__", builder.getF64FloatAttr(1.0));
        op.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));

        // source is 1st parameter, dest is the second
        auto setop = builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[2]);
        setop.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));
      }
      else if (binop == tok_elews || binop == tok_monoid)
      {
        auto op = builder.create<TensorElewsMultOp>(loc(tensor_op.loc()), tensors[2].getType(), tensors[0], tensors[1], lhs_lbls_value, affineMapArrayAttr, strAttr, SemiringAttr);
        op.getOperation()->setAttr("__alpha__", builder.getF64FloatAttr(1.0));
        op.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));

        auto setop = builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[2]);
        setop.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));
      }
      else
      {
        emitError(loc(tensor_op.loc()), "error: the tensor operation allows only "
                                        "specific kind of binary operators. '");
      }

      return mlir::success();
    }

    mlir::LogicalResult mlirGenTensorFill(mlir::Location loc,
                                          StringRef tensor_name, double value)
    {
      mlir::Value tensorValue = symbolTable.lookup(tensor_name);
      auto tensorType = tensorValue.getDefiningOp()->getOpResult(0).getType();
      auto tensorElType = tensorType.cast<mlir::TensorType>().getElementType();

      mlir::FloatAttr valueAttr;
      if (tensorElType.isF64())
      {
        valueAttr = builder.getF64FloatAttr(value);
      }
      else if (tensorElType.isF32())
      {
        valueAttr = builder.getF32FloatAttr(value);
      }
      else
      {
        emitError(loc,
                  "error: only float and double tensors are supported "
                  "for tensor ops'");
      }

      builder.create<TensorFillOp>(loc, tensorValue, valueAttr);

      return mlir::success();
    }

    mlir::LogicalResult mlirGenTensorFillRandom(mlir::Location loc,
                                                StringRef tensor_name)
    {
      comet_debug() << __FILE__ << " " << __LINE__ << " in mlirGenTensorFillRandom\n";

      mlir::Value tensorValue = symbolTable.lookup(tensor_name);
      auto lhs_labeledtensor = tensorValue.getDefiningOp()->getOpResult(0);
      comet_debug() << "\n";
      comet_vdump(lhs_labeledtensor);

      std::vector<mlir::Value> lhs_lbls_value;
      if (isa<DenseTensorDeclOp>(lhs_labeledtensor.getDefiningOp()))
      {
        for (unsigned int i = 0; i < lhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
        {
          comet_debug() << " a densor tensor decl.\n";
          comet_vdump(lhs_labeledtensor.getDefiningOp()->getOperand(i));
          lhs_lbls_value.push_back(lhs_labeledtensor.getDefiningOp()->getOperand(i));
        }
      }
      else if (isa<LabeledTensorOp>(lhs_labeledtensor.getDefiningOp()))
      {
        for (unsigned i = 1; i < lhs_labeledtensor.getDefiningOp()->getNumOperands(); i++)
        {
          comet_debug() << " a labeled tensor op: it's lowering is currently not supported.\n";
          comet_vdump(lhs_labeledtensor.getDefiningOp()->getOperand(i));
          lhs_lbls_value.push_back(lhs_labeledtensor.getDefiningOp()->getOperand(i));
        }
      }
      else
      {
        if (isa<SparseTensorDeclOp>(lhs_labeledtensor.getDefiningOp()))
          assert(false && "random initialization is currently not supported for sparse tensors.\n");

        assert(false && "Not supported format encountered during random initialization of tensor.\n");
      }

      std::vector<int64_t> result_dims = getDimSizes(lhs_lbls_value);
      comet_debug() << __FILE__ << " " << __LINE__ << " dims size: " << result_dims.size() << "\n";

      auto type = getType(result_dims);

      // The attribute is a vector with a floating point value per element
      // (number) in the array
      std::vector<double> data;
      int dataArraySize = std::accumulate(result_dims.begin(), result_dims.end(), 1,
                                          std::multiplies<int>());

      // fill data array with random numbers
      double upperLimit = 10.0;
      for (int i = 0; i < dataArraySize; i++)
      {
        double randNum = static_cast<double>(rand()) / (static_cast<float>(RAND_MAX / upperLimit));
        data.push_back(randNum);
      }

      // The type of this attribute is tensor of 64-bit floating-point with the
      // shape of the literal.
      mlir::Type elementType = builder.getF64Type();
      auto dataType = mlir::RankedTensorType::get(result_dims, elementType);

      // This is the actual attribute that holds the list of values for this
      // tensor literal.
      auto dataAttribute =
          mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

      // Build the MLIR op `ta.constant`. This invokes the `DenseConstantOp::build`
      // method.
      auto denseConst = builder.create<DenseConstantOp>(loc, type, dataAttribute);
      builder.create<TensorSetOp>(loc, denseConst, lhs_labeledtensor);

      return mlir::success();
    }

    mlir::LogicalResult mlirGenTensorFillFromFile(mlir::Location loc,
                                                  StringRef tensor_name, StringRef filename)
    {
      mlir::Value tensorValue = symbolTable.lookup(tensor_name);
      mlir::StringAttr filenameAttr = builder.getStringAttr(filename);
      builder.create<TensorFillFromFileOp>(loc, tensorValue, filenameAttr);

      return mlir::success();
    }
  };

} // namespace

namespace tensorAlgebra
{

  // The public API for codegen.
  mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                                ModuleAST &moduleAST)
  {
    return MLIRGenImpl(context).mlirGen(moduleAST);
  }

} // namespace tensorAlgebra