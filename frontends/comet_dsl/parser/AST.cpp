//===- AST.cpp - Helper for printing out the COMET AST
//
/// Copyright 2022 Battelle Memorial Institute
///
/// Redistribution and use in source and binary forms, with or without modification,
/// are permitted provided that the following conditions are met:
///
/// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
/// and the following disclaimer.
///
/// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
/// and the following disclaimer in the documentation and/or other materials provided with the distribution.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
/// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
/// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
/// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
/// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
/// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///
/// =============================================================================
///
/// This file implements the AST dump for COMET DSL.
///
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace tensorAlgebra;

namespace
{

  /// RAII helper to manage increasing/decreasing the indentation as we traverse
  /// the AST
  struct Indent
  {
    Indent(int &level) : level(level) { ++level; }
    ~Indent() { --level; }
    int &level;
  };

  /// Helper class that implement the AST tree traversal and print the nodes along
  /// the way. The only data member is the current indentation level.
  class ASTDumper
  {
  public:
    void dump(ModuleAST *Node);

  private:
    void dump(const VarType &type);
    void dump(VarDeclExprAST *varDecl);
    void dump(ExprAST *expr);
    void dump(ExprASTList *exprList);
    void dump(NumberExprAST *num);
    void dump(LiteralExprAST *Node);
    void dump(VariableExprAST *Node);
    void dump(ReturnExprAST *Node);
    void dump(BinaryExprAST *Node);
    void dump(CallExprAST *Node);
    void dump(PrintExprAST *Node);
    void dump(PrototypeAST *Node);
    void dump(FunctionAST *Node);
    void dump(IndexLabelDeclExprAST *node);
    void dump(TensorDeclExprAST *node);
    void dump(LabeledTensorExprAST *node);
    void dump(ForLoopExprAST *node);
    void dump(ForLoopEndExprAST *node);
    void dump(TensorOpExprAST *node);
    void dump(TransposeExprAST *node);
    void dump(PrintElapsedTimeExprAST *node);
    void dump(GetTimeExprAST *node);

    /// Actually print spaces matching the current indentation level
    void indent()
    {
      for (int i = 0; i < curIndent; i++)
        llvm::errs() << "  ";
    }
    int curIndent = 0;
  };

} /// namespace

/// Return a formatted string for the location of any node
template <typename T>
static std::string loc(T *Node)
{
  const auto &loc = Node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col))
      .str();
}

/// Helper Macro to bump the indentation level and print the leading spaces for
/// the current indentations
#define INDENT()            \
  Indent level_(curIndent); \
  indent();

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(ExprAST *expr)
{
#define dispatch(CLASS)                          \
  if (CLASS *node = llvm::dyn_cast<CLASS>(expr)) \
    return dump(node);
  dispatch(VarDeclExprAST);
  dispatch(LiteralExprAST);
  dispatch(NumberExprAST);
  dispatch(VariableExprAST);
  dispatch(ReturnExprAST);
  dispatch(BinaryExprAST);
  dispatch(CallExprAST);
  dispatch(PrintExprAST);
  dispatch(IndexLabelDeclExprAST);
  dispatch(TensorDeclExprAST);
  dispatch(LabeledTensorExprAST);
  dispatch(ForLoopExprAST);
  dispatch(ForLoopEndExprAST);
  dispatch(TensorOpExprAST);
  dispatch(TransposeExprAST);
  dispatch(PrintElapsedTimeExprAST);
  dispatch(GetTimeExprAST);
  /// No match, fallback to a generic message
  INDENT();
  llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(VarDeclExprAST *varDecl)
{
  INDENT();
  llvm::errs() << "VarDecl " << varDecl->getName();
  dump(varDecl->getType());
  llvm::errs() << " " << loc(varDecl) << "\n";
  if (auto *initVal = varDecl->getInitVal())
    dump(initVal);
}

/// A "block", or a list of expression
void ASTDumper::dump(ExprASTList *exprList)
{
  INDENT();
  llvm::errs() << "Block {\n";
  for (auto &expr : *exprList)
    dump(expr.get());
  indent();
  llvm::errs() << "} /// Block\n";
}

/// A literal number, just print the value.
void ASTDumper::dump(NumberExprAST *num)
{
  INDENT();
  llvm::errs() << num->getValue() << " " << loc(num) << "\n";
}

/// Helper to print recursively a literal. This handles nested array like:
///    [ [ 1, 2 ], [ 3, 4 ] ]
/// We print out such array with the dimensions spelled out at every level:
///    <2,2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]
void printLitHelper(ExprAST *lit_or_num)
{
  /// Inside a literal expression we can have either a number or another literal
  if (auto num = llvm::dyn_cast<NumberExprAST>(lit_or_num))
  {
    llvm::errs() << num->getValue();
    return;
  }
  auto *literal = llvm::cast<LiteralExprAST>(lit_or_num);

  /// Print the dimension for this literal first
  llvm::errs() << "<";
  {
    const char *sep = "";
    for (auto dim : literal->getDims())
    {
      llvm::errs() << sep << dim;
      sep = ", ";
    }
  }
  llvm::errs() << ">";

  /// Now print the content, recursing on every element of the list
  llvm::errs() << "[ ";
  const char *sep = "";
  for (auto &elt : literal->getValues())
  {
    llvm::errs() << sep;
    printLitHelper(elt.get());
    sep = ", ";
  }
  llvm::errs() << "]";
}

/// Print a literal, see the recursive helper above for the implementation.
void ASTDumper::dump(LiteralExprAST *Node)
{
  INDENT();
  llvm::errs() << "Literal: ";
  printLitHelper(Node);
  llvm::errs() << " " << loc(Node) << "\n";
}

/// Print a variable reference (just a name).
void ASTDumper::dump(VariableExprAST *Node)
{
  INDENT();
  llvm::errs() << "var: " << Node->getName() << " " << loc(Node) << "\n";
}

/// Return statement print the return and its (optional) argument.
void ASTDumper::dump(ReturnExprAST *Node)
{
  INDENT();
  llvm::errs() << "Return\n";
  if (Node->getExpr().has_value())
    return dump(*Node->getExpr());
  {
    INDENT();
    llvm::errs() << "(void)\n";
  }
}

/// Print a binary operation, first the operator, then recurse into LHS and RHS.
void ASTDumper::dump(BinaryExprAST *Node)
{
  INDENT();
  llvm::errs() << "BinOp: " << Node->getOp() << " " << loc(Node) << " Available labels: [";
  for (auto &str : Node->getLabels())
  {
    llvm::errs() << str << ", ";
  }

  llvm::errs() << "\b\b"
               << "]\n";
  dump(Node->getLHS());
  dump(Node->getRHS());
}

/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument.
void ASTDumper::dump(CallExprAST *Node)
{
  INDENT();
  llvm::errs() << "Call '" << Node->getCallee() << "' [ " << loc(Node) << "\n";
  /// dump(Node->getArgs());
  indent();
  llvm::errs() << "]\n";
}

/// Print a builtin print call, first the builtin name and then the argument.
void ASTDumper::dump(PrintExprAST *Node)
{
  INDENT();
  llvm::errs() << "Print [ " << loc(Node) << "\n";
  dump(Node->getArg());
  indent();
  llvm::errs() << "]\n";
}

/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(const VarType &type)
{
  llvm::errs() << "<";
  if (type.elt_ty == VarType::TY_DOUBLE)
    llvm::errs() << "double";
  else if (type.elt_ty == VarType::TY_INT)
    llvm::errs() << "int";
  else if (type.elt_ty == VarType::TY_FLOAT)
    llvm::errs() << "float";

  const char *sep = "";
  for (auto shape : type.shape)
  {
    llvm::errs() << sep << shape;
    sep = ", ";
  }
  llvm::errs() << ">";
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST *Node)
{
  INDENT();
  llvm::errs() << "Proto '" << Node->getName() << "' " << loc(Node) << "'\n";
  indent();
  llvm::errs() << "Params: [";
  const char *sep = "";
  for (auto &arg : Node->getArgs())
  {
    llvm::errs() << sep << arg->getName();
    sep = ", ";
  }
  llvm::errs() << "]\n";
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(FunctionAST *Node)
{
  INDENT();
  llvm::errs() << "Function \n";
  dump(Node->getProto());
  dump(Node->getBody());
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *Node)
{
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &F : *Node)
    dump(&F);
}

void ASTDumper::dump(IndexLabelDeclExprAST *node)
{
  INDENT();
  llvm::errs() << "IndexLabelDeclExprAST " << node->getName() << "["
               << node->getBegin() << ":" << node->getEnd() << ":"
               << node->getIncrement() << "] " << loc(node) << "\n";
}

void ASTDumper::dump(TensorDeclExprAST *node)
{
  INDENT();
  llvm::errs() << "Tensor Declaration: ";

  dump(node->getElementType());
  llvm::errs() << ' ' << node->getName() << '{';
  for (auto &lbl : node->getDims())
    llvm::errs() << lbl << ", ";
  llvm::errs() << node->getFormat() << " \n";
  llvm::errs() << "\b\b} " << loc(node) << "\n";
}

void ASTDumper::dump(LabeledTensorExprAST *node)
{
  INDENT();
  llvm::errs() << "Labeled Tensor " << node->getTensorName() << "[";

  for (auto &lbl : node->getLabelNames())
    llvm::errs() << lbl << ", ";

  llvm::errs() << "\b\b"
               << "] " << loc(node) << "\n";
}

void ASTDumper::dump(ForLoopExprAST *node)
{
  INDENT();
  llvm::errs() << "For Loop start with it-var: " << node->getName() << " (";

  llvm::errs() << node->getBegin() << "," << node->getEnd() << "," << node->getIncrement()
               << ") " << loc(node) << "\n";
}

void ASTDumper::dump(ForLoopEndExprAST *node)
{
  INDENT();
  llvm::errs() << "For Loop End " << loc(node) << "\n";
}

void ASTDumper::dump(TensorOpExprAST *node)
{
  INDENT();
  llvm::errs() << "TensorOp: " << node->getOpStr() << " " << loc(node)
               << "\n";
  dump(node->getLHS());
  dump(node->getRHS());
}

void ASTDumper::dump(TransposeExprAST *node)
{
  INDENT();
  llvm::errs() << "Transpose: " << node->getName() << " " << loc(node);
  //<< "\n";
  llvm::errs() << "  Src Dims: [";
  for (auto elem : node->getSrcDims())
    llvm::errs() << elem << ", ";
  llvm::errs() << "], ";

  llvm::errs() << "Dst Dims: [";
  for (auto elem : node->getDstDims())
    llvm::errs() << elem << ", ";
  llvm::errs() << "]"
               << "\n";
}

void ASTDumper::dump(PrintElapsedTimeExprAST *node)
{
  INDENT();
  llvm::errs() << "PrintElapsedTime[ " << loc(node)
               << "\n";
  dump(node->getStart());
  dump(node->getEnd());
  indent();
  llvm::errs() << "]\n";
}
void ASTDumper::dump(GetTimeExprAST *node)
{
  INDENT();
  llvm::errs() << "getTime " << loc(node)
               << "\n";
}

namespace tensorAlgebra
{

  /// Public API
  void dump(ModuleAST &module) { ASTDumper().dump(&module); }

} /// namespace tensorAlgebra
