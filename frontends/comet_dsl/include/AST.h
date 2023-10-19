//===- AST.h - Node definition for he COMET Domain Specific Language AST -===//
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
// This file implements the Abstract Syntax Tree for COMET Domain specific Language.
//
//===----------------------------------------------------------------------===//

#ifndef COMET_DSL_AST_H_
#define COMET_DSL_AST_H_

#include "Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>
#include <set>

using llvm::cast;

namespace tensorAlgebra
{
  /// A variable type with either name or shape information.

  /// A variable type with shape information.
  struct VarType
  {
    enum
    {
      TY_FLOAT,
      TY_INT,
      TY_DOUBLE
    } elt_ty;
    std::vector<int64_t> shape;

    VarType() : elt_ty{TY_DOUBLE}, shape{} {}
  };

  enum TensorOpKind
  {
    Tensor_Set,
    Tensor_Acc,
    Tensor_Red_Add,
    Tensor_Red_Sub,
  };

  /// Base class for all expression nodes.
  class ExprAST
  {
  public:
    enum ExprASTKind
    {
      Expr_VarDecl,
      Expr_Return,
      Expr_Num,
      Expr_Literal,
      Expr_Var,
      Expr_BinOp,
      Expr_Call,
      Expr_FileRead,
      Expr_Print,
      Expr_IndexLabelDecl,
      Expr_IndexLabelDeclDynamic,
      Expr_TensorDecl,
      Expr_Transpose,
      Expr_LabeledTensor,
      Expr_Tensor,
      Expr_PrintElapsed,
      Expr_GetTime,
      Expr_ForLoop,
      Expr_ForEnd,
      Expr_Mask,
      Expr_FuncArg,
    };

    ExprAST(ExprASTKind kind, Location location)
        : kind(kind), location(location) {}

    virtual ~ExprAST() = default;

    ExprASTKind getKind() const { return kind; }

    const Location &loc() { return location; }

  private:
    const ExprASTKind kind;
    Location location;
  };

  /// A block-list of expressions.
  using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

  /// Expression class for numeric literals like "1.0".
  class NumberExprAST : public ExprAST
  {
    double Val;

  public:
    NumberExprAST(Location loc, double Val) : ExprAST(Expr_Num, loc), Val(Val) {}

    double getValue() { return Val; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_Num; }
  };

  /// Expression class for a literal value.
  class LiteralExprAST : public ExprAST
  {
    std::vector<std::unique_ptr<ExprAST>> values;
    std::vector<int64_t> dims;

  public:
    LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                   std::vector<int64_t> dims)
        : ExprAST(Expr_Literal, loc), values(std::move(values)),
          dims(std::move(dims)) {}

    std::vector<std::unique_ptr<ExprAST>> &getValues() { return values; }
    std::vector<int64_t> &getDims() { return dims; }
    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_Literal; }
  };

  /// Expression class for referencing a variable, like "a".
  class VariableExprAST : public ExprAST
  {
    std::string name;

  public:
    VariableExprAST(Location loc, const std::string &name)
        : ExprAST(Expr_Var, loc), name(name) {}

    llvm::StringRef getName() { return name; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_Var; }
  };

  class FuncArgAST : public ExprAST
  {
    Location loc;
    std::string name;
    VarType type;

  public:
    FuncArgAST(Location loc, llvm::StringRef name, VarType type)
        : ExprAST(Expr_FuncArg, loc), loc(std::move(loc)), name(name), type(std::move(type)) {}

    llvm::StringRef getName() { return name; }
    const VarType &getType() { return type; }

    static bool classof(const ExprAST *C) { return C->getKind() == Expr_FuncArg; }
  };

  /// Expression class for defining a variable.
  class VarDeclExprAST : public ExprAST
  {
    std::string name;
    VarType type;
    std::unique_ptr<ExprAST> initVal;

  public:
    VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                   std::unique_ptr<ExprAST> initVal = nullptr)
        : ExprAST(Expr_VarDecl, std::move(loc)), name(name),
          type(std::move(type)), initVal(std::move(initVal)) {}

    llvm::StringRef getName() { return name; }
    ExprAST *getInitVal() { return initVal.get(); }
    const VarType &getType() { return type; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
  };

  /// Expression class for an index label declaration, i.e. IndexLabel i = [0:10];
  class IndexLabelDeclExprAST : public ExprAST
  {
    std::string name;
    int64_t begin;
    int64_t end;
    int64_t increment;

  public:
    IndexLabelDeclExprAST(Location loc, const std::string &name, int64_t begin,
                          int64_t end, int64_t increment = 1)
        : ExprAST(Expr_IndexLabelDecl, loc), name(name), begin(begin), end(end),
          increment(increment) {}

    llvm::StringRef getName() { return name; }
    int64_t getBegin() { return begin; }
    int64_t getEnd() { return end; }
    int64_t getIncrement() { return increment; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C)
    {
      return C->getKind() == Expr_IndexLabelDecl;
    }
  };

  /// Expression class for an index label declaration, i.e. IndexLabel i = [0:10];
  class IndexLabelDeclDynamicExprAST : public ExprAST
  {
    std::string name;
    int64_t begin;
    int64_t increment;

  public:
    IndexLabelDeclDynamicExprAST(Location loc, const std::string &name, int64_t begin,
                                 int64_t increment = 1)
        : ExprAST(Expr_IndexLabelDeclDynamic, loc), name(name), begin(begin), increment(increment) {}

    llvm::StringRef getName() { return name; }
    int64_t getBegin() { return begin; }
    int64_t getIncrement() { return increment; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C)
    {
      return C->getKind() == Expr_IndexLabelDeclDynamic;
    }
  };

  /// Add format attributes
  /// Expression class for an index label declaration, i.e. Tensor<double> T{i,j};
  class TensorDeclExprAST : public ExprAST
  {
    std::string name;
    VarType element_type;
    std::vector<std::string> dims;
    std::string format;
    ExprASTList values;

  public:
    TensorDeclExprAST(Location loc, const std::string &name, VarType element_type,
                      const std::vector<std::string> &dims, const std::string &format)
        : ExprAST(Expr_TensorDecl, loc), name(name), element_type(element_type),
          dims(dims), format(format) {}

    llvm::StringRef getName() { return name; }
    VarType &getElementType() { return element_type; }
    std::vector<std::string> &getDims() { return dims; }
    std::string &getFormat() { return format; }
    ExprASTList &getValues() { return values; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C)
    {
      return C->getKind() == Expr_TensorDecl;
    }
  };

  /// Add format attributes
  /// Expression class for an index label declaration, i.e. transpose(A[i,j], {j, i})
  class TransposeExprAST : public ExprAST
  {
    std::string name; /// tensor name
    std::vector<std::string> src_dims;
    std::vector<std::string> dst_dims;

  public:
    TransposeExprAST(Location loc, const std::string &name,
                     const std::vector<std::string> &src_dims,
                     const std::vector<std::string> &dst_dims)
        : ExprAST(Expr_Transpose, loc), name(name),
          src_dims(src_dims), dst_dims(dst_dims) {}

    llvm::StringRef getName() { return name; }
    std::vector<std::string> &getSrcDims() { return src_dims; }
    std::vector<std::string> &getDstDims() { return dst_dims; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C)
    {
      return C->getKind() == Expr_Transpose;
    }
  };

  class LabeledTensorExprAST : public ExprAST
  {
    std::string tensor_name;
    std::vector<std::string> label_names;

  public:
    LabeledTensorExprAST(Location loc, const std::string tensor_name,
                         const std::vector<std::string> &label_names)
        : ExprAST(Expr_LabeledTensor, loc), tensor_name(tensor_name),
          label_names(label_names) {}

    llvm::StringRef getTensorName() { return tensor_name; }
    std::vector<std::string> &getLabelNames() { return label_names; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C)
    {
      return C->getKind() == Expr_LabeledTensor;
    }
  };

  /// Expression class to support masking based operations.
  class MaskExprAST : public ExprAST
  {
    std::string tensor_name;
    std::string maskType;

  public:
    MaskExprAST(Location loc, const std::string &tensor_name,
                const std::string &maskType)
        : ExprAST(Expr_Mask, loc), tensor_name(tensor_name),
          maskType(maskType) {}

    const llvm::StringRef getTensorName() { return tensor_name; }
    const llvm::StringRef getMaskType() { return maskType; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C)
    {
      return C->getKind() == Expr_Mask;
    }
  };

  class TensorOpExprAST : public ExprAST
  {
    TensorOpKind Op;
    std::unique_ptr<ExprAST> LHS;
    std::unique_ptr<ExprAST> RHS;
    std::unique_ptr<ExprAST> Mask;
    int beta;

  public:
    TensorOpExprAST(Location loc, TensorOpKind Op, std::unique_ptr<ExprAST> LHS,
                    std::unique_ptr<ExprAST> RHS, std::unique_ptr<ExprAST> Mask, int in_beta = 1)
        : ExprAST(Expr_Tensor, loc), Op(Op), LHS(std::move(LHS)),
          RHS(std::move(RHS)), Mask(std::move(Mask)), beta(in_beta) {}

    TensorOpKind getOp() { return Op; }
    std::string getOpStr()
    {
      std::string ret = "";
      switch (Op)
      {
      case TensorOpKind::Tensor_Set:
        ret = "SetOp";
        break;
      case TensorOpKind::Tensor_Acc:
        ret = "AccOp";
        break;
      default:
        ret = "";
        break;
      }
      return ret;
    }
    ExprAST *getLHS() { return LHS.get(); }
    ExprAST *getRHS() { return RHS.get(); }
    ExprAST *getMask() { return Mask.get(); }
    int getBeta() { return beta; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_Tensor; }
  };

  /// Expression class for a return operator.
  class ReturnExprAST : public ExprAST
  {
    std::optional<std::unique_ptr<ExprAST>> expr;

  public:
    ReturnExprAST(Location loc, std::optional<std::unique_ptr<ExprAST>> expr)
        : ExprAST(Expr_Return, loc), expr(std::move(expr)) {}

    std::optional<ExprAST *> getExpr()
    {
      if (expr.has_value())
        return expr->get();
      return std::nullopt;
    }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_Return; }
  };

  /// Expression class for a binary operator.
  class BinaryExprAST : public ExprAST
  {
    using StringSet = std::set<std::string>;

    int Op;
    int SemiringOp1st;
    int SemiringOp2nd;
    std::unique_ptr<ExprAST> LHS, RHS;
    StringSet available_labels;

  public:
    int getOp() { return Op; }
    int getSemiringOp1() { return SemiringOp1st; }
    int getSemiringOp2() { return SemiringOp2nd; }
    ExprAST *getLHS() { return LHS.get(); }
    ExprAST *getRHS() { return RHS.get(); }
    StringSet &getLabels() { return available_labels; }

    BinaryExprAST(Location loc, int Op,
                  int SemiringOp1st,
                  int SemiringOp2nd,
                  std::unique_ptr<ExprAST> LHS,
                  std::unique_ptr<ExprAST> RHS)
        : ExprAST(Expr_BinOp, loc), Op(Op),
          SemiringOp1st(SemiringOp1st),
          SemiringOp2nd(SemiringOp2nd),
          LHS(std::move(LHS)),
          RHS(std::move(RHS)), available_labels(computeAvailableLabels()) {}

    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_BinOp; }

  private:
    StringSet computeAvailableLabels()
    {

      StringSet result;
      if (LHS->getKind() == ExprAST::Expr_LabeledTensor)
      {
        auto lbls = cast<LabeledTensorExprAST>(*LHS).getLabelNames();
        result.insert(lbls.begin(), lbls.end());
      }
      else if (LHS->getKind() == ExprAST::Expr_BinOp)
      {
        auto lbls = cast<BinaryExprAST>(*LHS).getLabels();
        result.insert(lbls.begin(), lbls.end());
      }

      if (RHS->getKind() == ExprAST::Expr_LabeledTensor)
      {
        auto lbls = cast<LabeledTensorExprAST>(*RHS).getLabelNames();
        result.insert(lbls.begin(), lbls.end());
      }
      else if (RHS->getKind() == ExprAST::Expr_BinOp)
      {
        auto lbls = cast<BinaryExprAST>(*RHS).getLabels();
        result.insert(lbls.begin(), lbls.end());
      }

      return result;
    }
  };

  /// Expression class for function calls.
  class CallExprAST : public ExprAST
  {
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;

  public:
    CallExprAST(Location loc, const std::string &Callee,
                std::vector<std::unique_ptr<ExprAST>> Args)
        : ExprAST(Expr_Call, loc), Callee(Callee), Args(std::move(Args)) {}

    llvm::StringRef getCallee() { return Callee; }
    ExprAST *getArg(int index) { return Args[index].get(); }
    size_t getNumArgs() { return Args.size(); }
    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_Call; }
  };

  /// Expression class for file-read calls.
  class FileReadExprAST : public ExprAST
  {
    std::string Callee;
    std::unique_ptr<ExprAST> fileID;
    std::unique_ptr<ExprAST> ReadMode;

  public:
    FileReadExprAST(Location loc, const std::string &Callee, std::unique_ptr<ExprAST> id, std::unique_ptr<ExprAST> mode)
        : ExprAST(Expr_FileRead, loc), Callee(Callee), fileID(std::move(id)), ReadMode(std::move(mode)) {}

    ExprAST *getFileID() { return fileID.get(); }
    ExprAST *getReadMode() { return ReadMode.get(); }
    llvm::StringRef getCallee() { return Callee; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_FileRead; }
  };

  /// Expression class for loops, i.e. for index in range(start, end, increment);
  class ForLoopExprAST : public ExprAST
  {
    std::string name;
    int64_t begin;
    int64_t end;
    int64_t increment;

  public:
    ForLoopExprAST(Location loc, const std::string &name, int64_t begin,
                   int64_t end, int64_t increment = 1)
        : ExprAST(Expr_ForLoop, loc), name(name), begin(begin), end(end),
          increment(increment) {}

    llvm::StringRef getName() { return name; }
    int64_t getBegin() { return begin; }
    int64_t getEnd() { return end; }
    int64_t getIncrement() { return increment; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C)
    {
      return C->getKind() == Expr_ForLoop;
    }
  };

  /// Expression class for end of loops, i.e. end
  class ForLoopEndExprAST : public ExprAST
  {
  public:
    ForLoopEndExprAST(Location loc) : ExprAST(Expr_ForEnd, loc) {}

    static bool classof(const ExprAST *C)
    {
      return C->getKind() == Expr_ForEnd;
    }
  };

  /// Expression class for builtin print calls.
  class PrintExprAST : public ExprAST
  {
    std::unique_ptr<ExprAST> Arg;

  public:
    PrintExprAST(Location loc, std::unique_ptr<ExprAST> Arg)
        : ExprAST(Expr_Print, loc), Arg(std::move(Arg)) {}

    ExprAST *getArg() { return Arg.get(); }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_Print; }
  };

  /// Expression class for builtin print calls.
  class PrintElapsedTimeExprAST : public ExprAST
  {
    std::unique_ptr<ExprAST> start;
    std::unique_ptr<ExprAST> end;

  public:
    PrintElapsedTimeExprAST(Location loc, std::unique_ptr<ExprAST> s, std::unique_ptr<ExprAST> e)
        : ExprAST(Expr_PrintElapsed, loc), start(std::move(s)), end(std::move(e)) {}

    ExprAST *getStart() { return start.get(); }
    ExprAST *getEnd() { return end.get(); }

    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_PrintElapsed; }
  };

  class GetTimeExprAST : public ExprAST
  {

  public:
    GetTimeExprAST(Location loc)
        : ExprAST(Expr_GetTime, loc) {}

    /// LLVM style RTTI
    static bool classof(const ExprAST *C) { return C->getKind() == Expr_GetTime; }
  };

  /// This class represents the "prototype" for a function, which captures its
  /// name, and its argument names (thus implicitly the number of arguments the
  /// function takes).
  class PrototypeAST
  {
    Location location;
    std::string name;
    std::vector<std::unique_ptr<FuncArgAST>> args;

  public:
    PrototypeAST(Location location, const std::string &name,
                 std::vector<std::unique_ptr<FuncArgAST>> args)
        : location(location), name(name), args(std::move(args)) {}

    const Location &loc() { return location; }
    const std::string &getName() const { return name; }
    // TODO(gkestor): check FuncArgAST
    const std::vector<std::unique_ptr<FuncArgAST>> &getArgs()
    {
      return args;
    }
  };

  /// This class represents a function definition itself.
  class FunctionAST
  {
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprASTList> Body;

  public:
    FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                std::unique_ptr<ExprASTList> Body)
        : Proto(std::move(Proto)), Body(std::move(Body)) {}
    PrototypeAST *getProto() { return Proto.get(); }
    ExprASTList *getBody() { return Body.get(); }
  };

  /// This class represents a list of functions to be processed together
  class ModuleAST
  {
    std::vector<FunctionAST> functions;

  public:
    ModuleAST(std::vector<FunctionAST> functions)
        : functions(std::move(functions)) {}

    auto begin() -> decltype(functions.begin()) { return functions.begin(); }
    auto end() -> decltype(functions.end()) { return functions.end(); }
  };

  void dump(ModuleAST &);

} /// namespace tensorAlgebra

#endif /// COMET_DSL_AST_H_
