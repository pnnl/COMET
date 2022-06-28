//===------------ Parser.h - COMET Domain Specific Language Parser =--------------===//
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
// This file implements the parser for the COMET Domain Specific Language. It processes
// the Token provided by the Lexer and returns an AST.
//
//===----------------------------------------------------------------------===//

#ifndef COMET_DSL_PARSER_H
#define COMET_DSL_PARSER_H

#include "AST.h"
#include "Lexer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <utility>
#include <vector>

// *********** For debug purpose *********//
// #ifndef DEBUG_MODE_PARSER
// #define DEBUG_MODE_PARSER
// #endif

#ifdef DEBUG_MODE_PARSER
#define comet_errs() llvm::errs() << __FILE__ << " " << __LINE__ << " "
#define comet_pdump(n) n->dump()
#define comet_vdump(n) n.dump()
#else
#define comet_errs() llvm::nulls()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
// *********** For debug purpose *********//

namespace tensorAlgebra
{

  /// This is a simple recursive parser for the Tensor Algebra language. It produces a well
  /// formed AST from a stream of Token supplied by the Lexer. No semantic checks
  /// or symbol resolution is performed. For example, variables are referenced by
  /// string and the code could reference an undeclared variable and the parsing
  /// succeeds.
  class Parser
  {
  public:
    /// Create a Parser for the supplied lexer.
    Parser(Lexer &lexer) : lexer(lexer) {}

    /// Parse a full Module. A module is a list of function definitions.
    std::unique_ptr<ModuleAST> parseModule()
    {
      lexer.getNextToken(); // prime the lexer

      // Parse functions one at a time and accumulate in this vector.
      std::vector<FunctionAST> functions;
      while (auto f = parseDefinition())
      {
        functions.push_back(std::move(*f));
        if (lexer.getCurToken() == tok_eof)
          break;
      }
      // If we didn't reach EOF, there was an error during parsing
      if (lexer.getCurToken() != tok_eof)
        return parseError<ModuleAST>("nothing", "at end of module");

      return std::make_unique<ModuleAST>(std::move(functions));
    }

  private:
    Lexer &lexer;

    /// Parse a return statement.
    /// return :== return ; | return expr ;
    std::unique_ptr<ReturnExprAST> parseReturn()
    {
      auto loc = lexer.getLastLocation();
      lexer.consume(tok_return);

      // return takes an optional argument
      llvm::Optional<std::unique_ptr<ExprAST>> expr;
      if (lexer.getCurToken() != ';')
      {
        expr = parseExpression();
        if (!expr)
          return nullptr;
      }
      return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
    }

    /// Parse a literal number.
    /// numberexpr ::= number
    std::unique_ptr<ExprAST> parseNumberExpr()
    {
      auto loc = lexer.getLastLocation();
      auto result =
          std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
      lexer.consume(tok_number);
      return std::move(result);
    }

    /// Parse an ElewsExpr
    /// elewsexpr ::= labeledTensor .* ...
    std::unique_ptr<ExprAST> parseElewsExpr()
    {
      auto loc = lexer.getLastLocation();
      comet_errs() << " going to consume tok_elews\n";
      lexer.consume(tok_elews);

      // return takes an optional argument
      llvm::Optional<std::unique_ptr<ExprAST>> expr;
      if (lexer.getCurToken() != ';')
      {
        expr = parseExpression();
        if (!expr)
          return nullptr;
      }
      return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
    }

    /// Parse a literal array expression.
    /// tensorLiteral ::= [ literalList ] | number
    /// literalList ::= tensorLiteral | tensorLiteral, literalList
    std::unique_ptr<ExprAST> parseTensorLiteralExpr()
    {
      comet_errs() << " in parseTensorLiteralExpr\n";
      auto loc = lexer.getLastLocation();
      lexer.consume(Token('['));

      // Hold the list of values at this nesting level.
      std::vector<std::unique_ptr<ExprAST>> values;
      // Hold the dimensions for all the nesting inside this level.
      std::vector<int64_t> dims;
      do
      {
        // We can have either another nested array or a number literal.
        if (lexer.getCurToken() == '[')
        {
          values.push_back(parseTensorLiteralExpr());
          if (!values.back())
            return nullptr; // parse error in the nested array.
        }
        else
        {
          if (lexer.getCurToken() != tok_number)
            return parseError<ExprAST>("<num> or [", "in literal expression");
          values.push_back(parseNumberExpr());
        }

        // End of this list on ']'
        if (lexer.getCurToken() == ']')
          break;

        // Elements are separated by a comma.
        if (lexer.getCurToken() != ',')
          return parseError<ExprAST>("] or ,", "in literal expression");

        lexer.getNextToken(); // eat ,
      } while (true);
      if (values.empty())
        return parseError<ExprAST>("<something>", "to fill literal expression");
      lexer.getNextToken(); // eat ]

      /// Fill in the dimensions now. First the current nesting level:
      dims.push_back(values.size());

      /// If there is any nested array, process all of them and ensure that
      /// dimensions are uniform.
      if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr)
                       { return llvm::isa<LiteralExprAST>(expr.get()); }))
      {
        auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
        if (!firstLiteral)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");

        // Append the nested dimensions to the current level
        auto firstDims = firstLiteral->getDims();
        dims.insert(dims.end(), firstDims.begin(), firstDims.end());

        // Sanity check that shape is uniform across all elements of the list.
        for (auto &expr : values)
        {
          auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
          if (!exprLiteral)
            return parseError<ExprAST>("uniform well-nested dimensions",
                                       "inside literal expression");
          if (exprLiteral->getDims() != firstDims)
            return parseError<ExprAST>("uniform well-nested dimensions",
                                       "inside literal expression");
        }
      }
      return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                              std::move(dims));
    }

    /// parenexpr ::= '(' expression ')'
    std::unique_ptr<ExprAST> parseParenExpr()
    {
      lexer.getNextToken(); // eat (.
      auto v = parseExpression();
      if (!v)
        return nullptr;

      if (lexer.getCurToken() != ')')
        return parseError<ExprAST>(")", "to close expression with parentheses");
      lexer.consume(Token(')'));
      return v;
    }

    /// identifierexpr
    ///   ::= identifier
    ///   ::= identifier '(' expression ')'
    ///   ::= identifier '[' identifier-list ']'
    std::unique_ptr<ExprAST> parseIdentifierExpr()
    {
      std::string name(lexer.getId());
      comet_errs() << name << " \n";

      auto loc = lexer.getLastLocation();
      lexer.getNextToken(); // eat identifier.
      comet_errs() << lexer.getCurToken() << " \n";

      if (lexer.getCurToken() != '(' &&
          lexer.getCurToken() != '[')
      { // Simple variable ref.
        comet_errs() << " \n";
        return std::make_unique<VariableExprAST>(std::move(loc), name);
      }

      if (lexer.getCurToken() == '[')
      {
        comet_errs() << " \n";
        return ParseLabeledTensor(name, std::move(loc));
      }
      comet_errs() << " \n";

      // This is a function call.
      lexer.consume(Token('('));
      comet_errs() << "\n";
      std::vector<std::string> string_args;
      std::vector<std::unique_ptr<ExprAST>> args;
      if (lexer.getCurToken() != ')')
      {
        comet_errs() << "\n";
        while (true)
        {
          comet_errs() << "\n";
          if (auto arg = parseExpression())
          {
            comet_errs() << "\n";
            args.push_back(std::move(arg));
            comet_errs() << args.size() << "\n";
          }
          else
          {
            comet_errs() << "\n";
            return nullptr;
          }
          comet_errs() << args.size() << "\n";

          if (lexer.getCurToken() == ')')
          {
            comet_errs() << "\n";
            break;
          }

          if (lexer.getCurToken() != ',')
          {
            comet_errs() << "\n";
            return parseError<ExprAST>(", or )", "in argument list");
          }

          comet_errs() << "\n";
          // That means if ',', go to parse next token
          lexer.getNextToken();
        }
      }
      lexer.consume(Token(')'));
      comet_errs() << args.size() << "\n";

      // It can be a builtin call to print
      if (name == "getTime")
      {
        if (args.size() != 0)
          return parseError<ExprAST>("<no arg>", "as argument to getTime()");

        return std::make_unique<GetTimeExprAST>(std::move(loc));
      }

      // It can be a builtin call to print elapsed time
      if (name == "printElapsedTime")
      {
        if (args.size() != 2)
          return parseError<ExprAST>("<two args>", "as argument to printElapsedTime()");

        return std::make_unique<PrintElapsedTimeExprAST>(std::move(loc), std::move(args[0]), std::move(args[1]));
      }
      // It can be a builtin call to print
      if (name == "print")
      {
        if (args.size() != 1)
          return parseError<ExprAST>("<single arg>", "as argument to print()");

        return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
      }

      if (name == "read_from_file")
      { // It can be a builtin call to read_from_file
        comet_errs() << " read_from_file\n";
        if (args.size() == 0)
        {
          args.push_back(nullptr);
        }
      }

      if (name == "random")
      {
        comet_errs() << " random\n";
        if (args.size() == 0)
        {
          args.push_back(nullptr);
        }
        // CallExprAST is generated for random()
      }
      comet_errs() << " generate CallExprAST node\n ";
      return std::make_unique<CallExprAST>(std::move(loc), name, std::move(args[0]));
    }

    /// primary
    ///   ::= identifierexpr
    ///   ::= numberexpr
    ///   ::= parenexpr
    ///   ::= tensorliteral
    std::unique_ptr<ExprAST> parsePrimary()
    {
      comet_errs() << " in parsePrimary, curToken: " << lexer.getCurToken() << " \n";
      switch (lexer.getCurToken())
      {
      default:
      {
        llvm::errs() << __FILE__ << " " << __LINE__ << " unknown token '" << lexer.getCurToken()
                     << "' when expecting an expression\n";
        return nullptr;
      }
      case tok_identifier:
      {
        comet_errs() << " call parseIdentifierExpr start\n";
        auto lhs = parseIdentifierExpr();
        comet_errs() << " call parseIdentifierExpr end\n";
        if (lhs.get()->getKind() == tensorAlgebra::ExprAST::Expr_Call)
        {
          comet_errs() << "\n";
          CallExprAST *call = llvm::cast<CallExprAST>(lhs.get());
          llvm::StringRef callee = call->getCallee();

          if (callee == "read_from_file")
          {
            comet_errs() << "\n";
          }
        }
        else
        {
          comet_errs() << "\n";
        }
        return lhs;
      }

      // for .* (elews in ta)
      case tok_elews:
      {
        comet_errs() << " parse elementwise start\n";
        return parseElewsExpr();
      }

      // for @(xx,xx) semirings in ta
      case tok_semiring:
      {
        comet_errs() << " parse semirings start\n";
        //   return parseSemiringExpr();
        break;
      }

      // for @(xx) monoid in ta
      case tok_monoid:
      {
        comet_errs() << " parse monoid start\n";
        break;
      }

      // return parseIdentifierExpr();
      case tok_number:
      {
        comet_errs() << " is tok_number\n ";
        return parseNumberExpr();
      }
      case '(':
        return parseParenExpr();
      case '[':
        return parseTensorLiteralExpr();
      case ';':
        return nullptr;
      case '}':
        return nullptr;
      case tok_transpose:
        return ParseTranspose();
      }
      return nullptr;
    }

    /// Recursively parse the right hand side of a binary expression, the ExprPrec
    /// argument indicates the precedence of the current binary operator.
    ///
    /// binoprhs ::= ('+' primary)*
    std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                           std::unique_ptr<ExprAST> lhs)
    {
      comet_errs() << " in parseBinOpRHS\n";
      // If this is a binop, find its precedence.
      while (true)
      {
        int tokPrec = getTokPrecedence();
        comet_errs() << " tokPrec: " << tokPrec << ", exprPrec: " << exprPrec << "\n";
        // If this is a binop that binds at least as tightly as the current binop,
        // consume it, otherwise we are done.
        if (tokPrec < exprPrec)
        {
          comet_errs() << " done\n";
          return lhs;
        }

        // Okay, we know this is a binop.
        int binOp = lexer.getCurToken();
        comet_errs() << " going to consume current tok: " << lexer.getCurToken() << "\n";
        auto loc_old = lexer.getLastLocation(); // ruiqin
        comet_errs() << " loc_old: " << loc_old.line << " " << loc_old.col << " \n";
        lexer.consume(Token(binOp));

        // if binOp is semiring, eat the close parenthese
        if (binOp == tok_semiring)
        {
          lexer.getNextToken();
        }

        if (binOp == tok_monoid)
        {
          lexer.getNextToken();
        }

        auto loc = lexer.getLastLocation();
        comet_errs() << " loc: " << loc.line << " " << loc.col << " \n";

        comet_errs() << " parsePrimary for rhs start"
                     << "\n";
        // Parse the primary expression after the binary operator.
        auto rhs = parsePrimary();
        if (!rhs)
          return parseError<ExprAST>("expression", "to complete binary operator");

        comet_errs() << " parsePrimary for rhs finished"
                     << "\n";
        // If BinOp binds less tightly with rhs than the operator after rhs, let
        // the pending operator take rhs as its lhs.
        int nextPrec = getTokPrecedence();
        if (tokPrec < nextPrec)
        {
          rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
          if (!rhs)
            return nullptr;
        }

        // Merge lhs/RHS.
        lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp, lexer.getSemiring1st(), lexer.getSemiring2nd(),
                                              std::move(lhs), std::move(rhs));
      }
    }

    /// expression::= primary binop rhs
    std::unique_ptr<ExprAST> parseExpression()
    {
      comet_errs() << " in parseExpression"
                   << "\n";
      auto lhs = parsePrimary();
      if (!lhs)
        return nullptr;

      comet_errs() << "finished lhs parse\n";
      comet_errs() << " call parseBinOpRHS\n";
      return parseBinOpRHS(0, std::move(lhs));
    }

    /// type ::= < shape_list >
    /// shape_list ::= num | num , shape_list
    std::unique_ptr<VarType> parseType()
    {
      if (lexer.getCurToken() != '<')
        return parseError<VarType>("<", "to begin type");
      lexer.getNextToken(); // eat <

      auto type = std::make_unique<VarType>();

      while (lexer.getCurToken() == tok_number)
      {
        type->shape.push_back(lexer.getValue());
        lexer.getNextToken();
        if (lexer.getCurToken() == ',')
          lexer.getNextToken();
      }

      if (lexer.getCurToken() != '>')
        return parseError<VarType>(">", "to end type");
      lexer.getNextToken(); // eat >
      return type;
    }

    /// Parse a variable declaration, it starts with a `var` keyword followed by
    /// and identifier and an optional type (shape specification) before the
    /// initializer.
    /// decl ::= var identifier [ type ] = expr
    std::unique_ptr<VarDeclExprAST> parseDeclaration()
    {
      if (lexer.getCurToken() != tok_var)
        return parseError<VarDeclExprAST>("var", "to begin declaration");
      auto loc = lexer.getLastLocation();
      lexer.getNextToken(); // eat var

      if (lexer.getCurToken() != tok_identifier)
        return parseError<VarDeclExprAST>("identified",
                                          "after 'var' declaration");
      std::string id(lexer.getId());
      lexer.getNextToken(); // eat id

      std::unique_ptr<VarType> type; // Type is optional, it can be inferred
      if (lexer.getCurToken() == '<')
      {
        type = parseType();
        if (!type)
          return nullptr;
      }

      if (!type)
        type = std::make_unique<VarType>();
      lexer.consume(Token('='));
      auto expr = parseExpression();
      return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                              std::move(*type), std::move(expr));
    }

    std::vector<std::unique_ptr<IndexLabelDeclExprAST>>
    ParseIndexLabelDeclaration()
    {
      comet_errs() << " parse index label dynamic decl\n";
      std::vector<std::unique_ptr<IndexLabelDeclExprAST>> ret;

      comet_errs() << " " << lexer.getCurToken() << "\n";

      if (lexer.getCurToken() != tok_index_label)
      {
        ret.push_back(parseError<IndexLabelDeclExprAST>(
            "IndexLabel", "to begin index label declaration"));
        return ret;
      }
      auto loc = lexer.getLastLocation();
      lexer.getNextToken(); // eat IndexLabel

      std::vector<std::string> id_list;
      if (lexer.getCurToken() == '[')
      {
        lexer.consume(Token('[')); // eat [
        while (lexer.getCurToken() != ']')
        {
          if (lexer.getCurToken() != tok_identifier)
          {
            ret.push_back(parseError<IndexLabelDeclExprAST>(
                "identifier", "after 'IndexLabel' declaration"));
            return ret;
          }
          id_list.push_back(lexer.getId().str());
          // comet_errs() << " index label: " << lexer.getId().str() << "\n";
          lexer.getNextToken(); // eat id
          if (lexer.getCurToken() == ',')
            lexer.consume(Token(',')); // eat ,
        }
        lexer.consume(Token(']')); // eat ]
      }
      else
      {
        if (lexer.getCurToken() != tok_identifier)
        {
          ret.push_back(parseError<IndexLabelDeclExprAST>(
              "identifier", "after 'IndexLabel' declaration"));
          return ret;
        }

        id_list.push_back(lexer.getId().str());
        lexer.getNextToken(); // eat id
      }

      if (lexer.getCurToken() != '=')
      {
        ret.push_back(parseError<IndexLabelDeclExprAST>(
            "=", "after 'IndexLabel' declaration(s)"));
        return ret;
      }

      lexer.consume(Token('='));

      if (lexer.getCurToken() != '[')
      {
        ret.push_back(parseError<IndexLabelDeclExprAST>(
            "[", "to begin range definition"));
        return ret;
      }

      lexer.consume(Token('[')); // eat [

      // ruiqin: [0:?:1] [0:?] [?]
      //  if(lexer.getCurToken() == '?' || lexer.getNextToken() == '?' ){
      //  comet_errs() << " size is ?\n";
      int64_t start = 0;
      int64_t end = start;
      int64_t incr = 1;

      // For [?]
      if (lexer.getCurToken() == '?')
      {
        end = mlir::ShapedType::kDynamicSize;
        lexer.getNextToken(); // eat start

        if (lexer.getCurToken() == ']')
        {
          // start = 0;
          lexer.consume(Token(']')); // eat ]

          for (auto &id : id_list)
          {
            ret.push_back(std::make_unique<IndexLabelDeclExprAST>(
                loc, std::move(id), start, end, incr));
            // comet_errs() << " Generate dynamic index label\n";
          }

          return ret;
        }
        else
        {
          ret.push_back(parseError<IndexLabelDeclExprAST>(
              "]", "to end range definition"));
          return ret;
        }
      }

      // for [32] or [0:32] or [0:32:1] or [0:?] or [0:?:1]
      if (lexer.getCurToken() == tok_number)
      {
        start = lexer.getValue();
        lexer.getNextToken(); // eat start

        if (lexer.getCurToken() == ':')
        {
          lexer.consume(Token(':')); // eat :
        }
        else if (lexer.getCurToken() == ']')
        {
          end = start;
          start = 0;
          lexer.consume(Token(']')); // eat ]
          for (auto &id : id_list)
          {
            ret.push_back(std::make_unique<IndexLabelDeclExprAST>(
                loc, std::move(id), start, end, incr));
            // comet_errs() << " Generate dynamic index label\n";
          }
          return ret;
        }
        else
        {
          ret.push_back(parseError<IndexLabelDeclExprAST>(
              "]", "to end range definition"));
          return ret;
        }
      }

      if (lexer.getCurToken() == '?')
      {
        lexer.getNextToken(); // eat ?
        // end = -1;
        end = mlir::ShapedType::kDynamicSize;
      }
      else if (lexer.getCurToken() == tok_number)
      {
        end = lexer.getValue();
        lexer.getNextToken(); // eat num
      }

      // int64_t incr = 1;
      if (lexer.getCurToken() == ':')
      {
        lexer.consume(Token(':')); // eat :
        incr = lexer.getValue();
        lexer.getNextToken(); // eat num
      }

      if (lexer.getCurToken() != ']')
      {
        ret.push_back(parseError<IndexLabelDeclExprAST>(
            "]", "to end range definition"));
        return ret;
      }

      lexer.consume(Token(']'));

      for (auto &id : id_list)
      {
        ret.push_back(std::make_unique<IndexLabelDeclExprAST>(
            loc, std::move(id), start, end, incr));
        comet_errs() << " Generate dynamic index label\n";
      }

      return ret;
    }

    std::vector<std::unique_ptr<IndexLabelDeclDynamicExprAST>>
    ParseIndexLabelDynamicDeclaration()
    {
      comet_errs() << " parse index label dynamic decl\n";
      std::vector<std::unique_ptr<IndexLabelDeclDynamicExprAST>> ret;

      comet_errs() << " " << lexer.getCurToken() << "\n";

      if (lexer.getCurToken() != tok_index_label_dynamic)
      {
        ret.push_back(parseError<IndexLabelDeclDynamicExprAST>(
            "IndexLabelDynamic", "to begin index label declaration"));
        return ret;
      }
      auto loc = lexer.getLastLocation();
      lexer.getNextToken(); // eat IndexLabelDynamic

      std::vector<std::string> id_list;
      if (lexer.getCurToken() == '[')
      {
        lexer.consume(Token('[')); // eat [
        while (lexer.getCurToken() != ']')
        {
          if (lexer.getCurToken() != tok_identifier)
          {
            ret.push_back(parseError<IndexLabelDeclDynamicExprAST>(
                "identifier", "after 'IndexLabel' declaration"));
            return ret;
          }
          id_list.push_back(lexer.getId().str());
          comet_errs() << " index label: " << lexer.getId().str() << "\n";
          lexer.getNextToken(); // eat id
          if (lexer.getCurToken() == ',')
            lexer.consume(Token(',')); // eat ,
        }
        lexer.consume(Token(']')); // eat ]
      }
      else
      {
        if (lexer.getCurToken() != tok_identifier)
        {
          ret.push_back(parseError<IndexLabelDeclDynamicExprAST>(
              "identifier", "after 'IndexLabel' declaration"));
          return ret;
        }

        id_list.push_back(lexer.getId().str());
        lexer.getNextToken(); // eat id
      }

      if (lexer.getCurToken() != '=')
      {
        ret.push_back(parseError<IndexLabelDeclDynamicExprAST>(
            "=", "after 'IndexLabel' declaration(s)"));
        return ret;
      }

      lexer.consume(Token('='));

      if (lexer.getCurToken() != '[')
      {
        ret.push_back(parseError<IndexLabelDeclDynamicExprAST>(
            "[", "to begin range definition"));
        return ret;
      }

      lexer.consume(Token('[')); // eat [

      int64_t start = 0;
      if (lexer.getCurToken() == tok_number)
      {
        start = lexer.getValue();
        lexer.getNextToken(); // eat start

        if (lexer.getCurToken() == ':')
        {
          lexer.consume(Token(':')); // eat :
        }
        else
        {
          ret.push_back(parseError<IndexLabelDeclDynamicExprAST>(
              "]", "to end range definition"));
          return ret;
        }
      }

      if (lexer.getCurToken() == '?')
      {
        lexer.getNextToken(); // eat ?
      }

      int64_t incr = 1;
      if (lexer.getCurToken() == ':')
      {
        lexer.consume(Token(':')); // eat :
        incr = lexer.getValue();
        lexer.getNextToken(); // eat num
      }

      if (lexer.getCurToken() != ']')
      {
        ret.push_back(parseError<IndexLabelDeclDynamicExprAST>(
            "]", "to end range definition"));
        return ret;
      }

      lexer.consume(Token(']'));

      for (auto &id : id_list)
      {
        ret.push_back(std::make_unique<IndexLabelDeclDynamicExprAST>(
            loc, std::move(id), start, incr));
        comet_errs() << " Generate dynamic index label\n";
      }

      return ret;
      // }
    }

    /// Parse a tensor declaration with format attribute, (e.g. CSR, COO, ...)
    /// it starts with a `Tensor` keyword followed by a type (double, float, int),
    /// decl ::= Tensor<el_type> identifier({Dimension list, format)
    /// Tensor<double> A([a, b], COO);
    /// Tensor<double> A([a, b], [CN, S]);
    std::unique_ptr<TensorDeclExprAST> ParseTensorDeclaration()
    {
      if (lexer.getCurToken() != tok_tensor)
        return parseError<TensorDeclExprAST>("Tensor",
                                             "to begin tensor declaration");
      auto loc = lexer.getLastLocation();
      lexer.getNextToken(); // eat Tensor

      std::unique_ptr<VarType> type; // Type is optional, it can be inferred
      if (lexer.getCurToken() == '<')
      {
        lexer.consume(Token('<')); // eat <
        type = std::make_unique<VarType>();
        if (lexer.getCurToken() == tok_double)
        {
          type->elt_ty = VarType::TY_DOUBLE;
        }
        else if (lexer.getCurToken() == tok_float)
        {
          type->elt_ty = VarType::TY_FLOAT;
        }
        else if (lexer.getCurToken() == tok_int)
        {
          type->elt_ty = VarType::TY_INT;
        }
        lexer.getNextToken(); // eat el_type
        if (lexer.getCurToken() != '>')
          return parseError<TensorDeclExprAST>(">", "to end type");
        lexer.getNextToken(); // eat >
      }

      if (lexer.getCurToken() != tok_identifier)
        return parseError<TensorDeclExprAST>("identifier",
                                             "after 'Tensor' declaration");
      std::string id(lexer.getId());
      lexer.getNextToken(); // eat id

      std::vector<std::string> dims;
      std::string format;
      std::vector<std::string> formatattr;
      if (lexer.getCurToken() == '(')
      {
        lexer.getNextToken(); // eat (

        if (lexer.getCurToken() == '[')
        {
          lexer.getNextToken(); // eat [

          while (lexer.getCurToken() == tok_identifier)
          {
            dims.push_back(lexer.getId().str());
            lexer.getNextToken(); // eat id
            if (lexer.getCurToken() == ',')
            {
              lexer.getNextToken(); // eat ,
            }
          }
          if (lexer.getCurToken() != ']')
          {
            return parseError<TensorDeclExprAST>(
                "}", "after 'Tensor' dimension declaration");
          }
          lexer.getNextToken(); // eat ]
        }
        if (lexer.getCurToken() != ',')
        {
          return parseError<TensorDeclExprAST>(
              ",", "after 'Tensor' dimension declaration");
        }
        lexer.getNextToken(); // eat ,

        if (lexer.getCurToken() == tok_identifier)
        {
          format = lexer.getId().str();
          comet_errs() << " format: " << format << "\n";
          lexer.getNextToken(); // eat format
        }
        else
        { // {CN, S}
          if (lexer.getCurToken() == '{')
          {
            lexer.getNextToken(); // eat {

            while (lexer.getCurToken() == tok_identifier)
            {
              formatattr.push_back(lexer.getId().str());
              // comet_errs() << lexer.getId().str() << "\n";
              lexer.getNextToken(); // eat id
              if (lexer.getCurToken() == ',')
              {
                lexer.getNextToken(); // eat ,
              }
            }
            if (lexer.getCurToken() != '}')
            {
              return parseError<TensorDeclExprAST>(
                  "}", "after 'Tensor' dimension declaration");
            }
            lexer.getNextToken(); // eat }
          }
          // std::cout <<
        }

        if (formatattr.size() > 0)
        {
          for (unsigned int i = 0; i < formatattr.size() - 1; i++)
          {
            format = format + formatattr[i] + ", ";
          }
          format = format + formatattr[formatattr.size() - 1];
          comet_errs() << "format: " << format << "\n";
        }

        if (lexer.getCurToken() != ')')
          return parseError<TensorDeclExprAST>(
              ")", "after 'Tensor' format declaration");
        lexer.getNextToken(); // eat )
      }

      return std::make_unique<TensorDeclExprAST>(std::move(loc), std::move(id),
                                                 std::move(*type), dims, format);
    }

    /// Parse a tensor declaration with format attribute, (e.g. CSR, COO, ...)
    /// it starts with a `Tensor` keyword followed by a type (double, float, int), ///// an identifier and a list of dimension labels
    /// decl ::= Tensor<el_type> identifier({Dimension list, format)
    /// Tensor<double> A([a, b], COO);
    /// Tensor<double> A([a, b], [CN, S]);
    std::unique_ptr<OutputTensorDeclExprAST> ParseOutputTensorDeclaration()
    {
      comet_errs() << __FILE__ << __LINE__ << " is in ParseOutputTensorDeclaration\n";
      if (lexer.getCurToken() != tok_outputtensor)
        return parseError<OutputTensorDeclExprAST>("OTensor",
                                                   "to begin output tensor declaration");
      auto loc = lexer.getLastLocation();
      lexer.getNextToken(); // eat Tensor

      std::unique_ptr<VarType> type; // Type is optional, it can be inferred
      if (lexer.getCurToken() == '<')
      {
        lexer.consume(Token('<')); // eat <
        type = std::make_unique<VarType>();
        if (lexer.getCurToken() == tok_double)
        {
          type->elt_ty = VarType::TY_DOUBLE;
        }
        else if (lexer.getCurToken() == tok_float)
        {
          type->elt_ty = VarType::TY_FLOAT;
        }
        else if (lexer.getCurToken() == tok_int)
        {
          type->elt_ty = VarType::TY_INT;
        }
        lexer.getNextToken(); // eat el_type
        if (lexer.getCurToken() != '>')
          return parseError<OutputTensorDeclExprAST>(">", "to end type");
        lexer.getNextToken(); // eat >
      }

      if (lexer.getCurToken() != tok_identifier)
        return parseError<OutputTensorDeclExprAST>("identifier",
                                                   "after 'OTensor' declaration");
      std::string id(lexer.getId());
      lexer.getNextToken(); // eat id

      std::vector<std::string> dims;
      std::string format;
      std::vector<std::string> formatattr;
      if (lexer.getCurToken() == '(')
      {
        lexer.getNextToken(); // eat (

        if (lexer.getCurToken() == '[')
        {
          lexer.getNextToken(); // eat [

          while (lexer.getCurToken() == tok_identifier)
          {
            dims.push_back(lexer.getId().str());
            lexer.getNextToken(); // eat id
            if (lexer.getCurToken() == ',')
            {
              lexer.getNextToken(); // eat ,
            }
          }
          if (lexer.getCurToken() != ']')
          {
            return parseError<OutputTensorDeclExprAST>(
                "}", "after 'OTensor' dimension declaration");
          }
          lexer.getNextToken(); // eat ]
        }
        if (lexer.getCurToken() != ',')
        {
          return parseError<OutputTensorDeclExprAST>(
              ",", "after 'OTensor' dimension declaration");
        }
        lexer.getNextToken(); // eat ,

        if (lexer.getCurToken() == tok_identifier)
        {
          format = lexer.getId().str();
          // LLVM_DEBUG(comet_errs() << " format: " << format << "\n");
          lexer.getNextToken(); // eat format
        }
        else
        { // {CN, S}

          if (lexer.getCurToken() == '{')
          {
            lexer.getNextToken(); // eat {

            while (lexer.getCurToken() == tok_identifier)
            {
              formatattr.push_back(lexer.getId().str());
              // comet_errs() << lexer.getId().str() << "\n";
              lexer.getNextToken(); // eat id
              if (lexer.getCurToken() == ',')
              {
                lexer.getNextToken(); // eat ,
              }
            }
            if (lexer.getCurToken() != '}')
            {
              return parseError<OutputTensorDeclExprAST>(
                  "}", "after 'OTensor' dimension declaration");
            }
            lexer.getNextToken(); // eat }
          }
        }

        if (formatattr.size() > 0)
        {
          for (unsigned int i = 0; i < formatattr.size() - 1; i++)
          {
            format = format + formatattr[i] + ", ";
          }
          format = format + formatattr[formatattr.size() - 1];
          comet_errs() << "format: " << format << "\n";
        }

        if (lexer.getCurToken() != ')')
          return parseError<OutputTensorDeclExprAST>(
              ")", "after 'OTensor' format declaration");
        lexer.getNextToken(); // eat )
      }

      return std::make_unique<OutputTensorDeclExprAST>(std::move(loc), std::move(id),
                                                       std::move(*type), dims, format);
    }

    /// Parse "transpose(A[i, j], {j, i})" operation in DSL
    /// it starts with a `transpose` keyword a tensor and a list of dimension labels
    /// decl ::= transpose(LabeledTensor, {Dimension list})
    /// transpose(A[i, j], {j, i})
    std::unique_ptr<TransposeExprAST> ParseTranspose()
    {
      comet_errs() << __FILE__ << __LINE__ << " ParseTranspose\n";
      if (lexer.getCurToken() != tok_transpose)
        return parseError<TransposeExprAST>("transpose",
                                            "to begin transpose op");
      auto loc = lexer.getLastLocation();
      lexer.getNextToken(); // eat Tensor

      std::unique_ptr<VarType> type; // Type is optional, it can be inferred
      if (lexer.getCurToken() == '(')
      {
        lexer.consume(Token('(')); // eat (
      }

      if (lexer.getCurToken() != tok_identifier)
        return parseError<TransposeExprAST>("identifier",
                                            "in 'transpose' op");
      std::string id(lexer.getId());
      // name = id;  ? this id is the tensor name ?
      lexer.getNextToken(); // eat id

      std::vector<std::string> src_dims;
      std::vector<std::string> dst_dims;
      if (lexer.getCurToken() == '[')
      {
        lexer.getNextToken(); // eat [
        while (lexer.getCurToken() == tok_identifier)
        {
          src_dims.push_back(lexer.getId().str());
          lexer.getNextToken(); // eat id
          if (lexer.getCurToken() == ',')
          {
            lexer.getNextToken(); // eat ,
          }
        }
        if (lexer.getCurToken() != ']')
        {
          return parseError<TransposeExprAST>(
              "]", "after source tensor dimension declaration");
        }
        lexer.getNextToken(); // eat ]
        if (lexer.getCurToken() != ',')
        {
          return parseError<TransposeExprAST>(
              ",", "after source tensor declaration");
        }
        lexer.getNextToken(); // eat ,

        if (lexer.getCurToken() == '{')
        {
          lexer.getNextToken(); // eat {

          while (lexer.getCurToken() == tok_identifier)
          {
            dst_dims.push_back(lexer.getId().str());
            lexer.getNextToken(); // eat id
            if (lexer.getCurToken() == ',')
            {
              lexer.getNextToken(); // eat ,
            }
          }
          if (lexer.getCurToken() != '}')
          {
            return parseError<TransposeExprAST>(
                "}", "after dest tensor dimension declaration");
          }
          lexer.getNextToken(); // eat }
        }

        if (lexer.getCurToken() != ')')
          return parseError<TransposeExprAST>(
              ")", "after 'OTensor' format declaration");
        lexer.getNextToken(); // eat )
      }

      return std::make_unique<TransposeExprAST>(std::move(loc), std::move(id),
                                                src_dims, dst_dims);
    }

    std::unique_ptr<LabeledTensorExprAST>
    ParseLabeledTensor(const std::string &tensor_name, Location loc)
    {
      if (lexer.getCurToken() != '[')
        return parseError<LabeledTensorExprAST>('[', "tensor identifier");
      // This is a labeled tensor .
      lexer.consume(Token('[')); // eat [
      std::vector<std::string> labels;
      if (lexer.getCurToken() != ']')
      {
        while (true)
        {
          if (lexer.getCurToken() != tok_identifier)
            return parseError<LabeledTensorExprAST>("identifier",
                                                    "in label list");

          labels.push_back(lexer.getId().str());
          lexer.getNextToken(); // eat identifier.

          if (lexer.getCurToken() == ']')
            break;

          if (lexer.getCurToken() != ',')
            return parseError<LabeledTensorExprAST>(", or ]", "in label list");
          lexer.getNextToken();
        }
      }
      lexer.consume(Token(']'));
      return std::make_unique<LabeledTensorExprAST>(std::move(loc), tensor_name,
                                                    labels);
    }

    // %t = ta.tc(a, b)
    // ta.set_op(%t, c) // src, dest  ta.tc(rhs1, rhs2, lhs)
    // ta.transpose(src, dest)

    /// tensor_expression
    ///         ::= identifier '=' expression
    ///         ::= identifier '+=' expression
    std::unique_ptr<ExprAST> ParseTensorExpression()
    {
      comet_errs() << __FILE__ << __LINE__ << " in ParseTensorExpression\n";

      if (lexer.getCurToken() != tok_identifier)
        return parseError<ExprAST>("identifier", "in tensor expression");

      std::string name(lexer.getId());
      comet_errs() << __FILE__ << __LINE__ << " name: " << name << " \n";
      auto loc = lexer.getLastLocation();
      lexer.getNextToken(); // eat identifier.

      if (lexer.getCurToken() != '[')
        return nullptr;

      auto LHS = ParseLabeledTensor(name, loc);
      if (!LHS)
        return nullptr;
      comet_errs() << __FILE__ << __LINE__ << " finished parse lhs\n ";

      // TODO(gkestor): add ast support for `-=` op
      TensorOpKind op;
      int beta = 0;
      if (lexer.getCurToken() == '=')
      {
        op = TensorOpKind::Tensor_Set;
        beta = 0;
      }
      else if (lexer.getCurToken() == '+' && lexer.lookAhead() == '=')
      {
        lexer.getNextToken(); // eat +
        op = TensorOpKind::Tensor_Red_Add;
        beta = 1;
      }
      else if (lexer.getCurToken() == '-' && lexer.lookAhead() == '=')
      {
        lexer.getNextToken(); // eat -
        op = TensorOpKind::Tensor_Red_Sub;
        beta = -1;
      }
      else
        return parseError<ExprAST>("= or +=", "in tensor expression");

      lexer.getNextToken(); // consume op

      comet_errs() << " parse rhs, call parseExpression()\n";
      auto RHS = parseExpression();
      if (!RHS)
        return nullptr;

      if (RHS.get()->getKind() == tensorAlgebra::ExprAST::Expr_Call)
      {
        comet_errs() << __FILE__ << __LINE__ << " TensorOpExprAST rhs is Expr_Transpose\n";

        //CallExprAST *call = llvm::cast<CallExprAST>(RHS.get());
        //llvm::StringRef callee = call->getCallee();
        //comet_errs() << __FILE__ << __LINE__ << " callee: " << callee << "\n";
      }
      else if (RHS.get()->getKind() == tensorAlgebra::ExprAST::Expr_Transpose)
      {
        comet_errs() << __FILE__ << __LINE__ << " TensorOpExprAST rhs is Expr_Transpose\n";
        //TransposeExprAST *call = llvm::cast<TransposeExprAST>(RHS.get());
      }
      return std::make_unique<TensorOpExprAST>(std::move(loc), op, std::move(LHS),
                                               std::move(RHS), beta);
    }

    /// var_expression
    ///         ::= identifier '=' expression
    ///         ::= identifier '+=' expression
    std::unique_ptr<ExprAST> ParseVarExpression()
    {
      auto loc = lexer.getLastLocation();

      std::string id(lexer.getId());
      lexer.getNextToken(); // eat id

      std::unique_ptr<VarType> type; // Type is optional, it can be inferred
      if (lexer.getCurToken() == '<')
      {
        type = parseType();
        if (!type)
          return nullptr;
      }

      if (!type)
        type = std::make_unique<VarType>();
      lexer.consume(Token('='));
      auto expr = parseExpression();
      return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                              std::move(*type), std::move(expr));
    }

    /// Parse a block: a list of expression separated by semicolons and wrapped in
    /// curly braces.
    ///
    /// block ::= { expression_list }
    /// expression_list ::= block_expr ; expression_list
    /// block_expr ::= decl | "return" | expr
    std::unique_ptr<ExprASTList> parseBlock()
    {
      if (lexer.getCurToken() != '{')
        return parseError<ExprASTList>("{", "to begin block");
      lexer.consume(Token('{'));

      auto exprList = std::make_unique<ExprASTList>();

      // Ignore empty expressions: swallow sequences of semicolons.
      while (lexer.getCurToken() == ';')
        lexer.consume(Token(';'));

      while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof)
      {
        if (lexer.getCurToken() == tok_index_label)
        {
          // IndexLabel declaration
          // comet_errs() << "ParseIndexLabelDeclaration\n";
          auto indexLabelDecls = ParseIndexLabelDeclaration();
          if (indexLabelDecls.empty())
            return nullptr;
          for (auto &decl : indexLabelDecls)
          {
            exprList->push_back(std::move(decl));
          }
        }
        else if (lexer.getCurToken() == tok_index_label_dynamic)
        {
          // IndexLabel declaration
          // comet_errs() << " ParseIndexLabelDynamicDeclaration\n";
          auto indexLabelDecls = ParseIndexLabelDynamicDeclaration();
          if (indexLabelDecls.empty())
            return nullptr;
          for (auto &decl : indexLabelDecls)
          {
            exprList->push_back(std::move(decl));
          }
        }
        else if (lexer.getCurToken() == tok_tensor)
        {
          // Tensor declaration
          // comet_errs() << "ParseTensorDeclaration\n";
          auto tensorDecl = ParseTensorDeclaration();
          if (!tensorDecl)
            return nullptr;
          exprList->push_back(std::move(tensorDecl));
        }
        else if (lexer.getCurToken() == tok_outputtensor)
        {
          // Tensor declaration
          // comet_errs() << "ParseOutputTensorDeclaration\n";
          auto outputtensorDecl = ParseOutputTensorDeclaration();
          if (!outputtensorDecl)
            return nullptr;
          exprList->push_back(std::move(outputtensorDecl));
        }
        else if (lexer.getCurToken() == tok_var)
        {
          // Variable declaration
          auto varDecl = parseDeclaration();
          if (!varDecl)
            return nullptr;
          exprList->push_back(std::move(varDecl));
        }
        else if (lexer.getCurToken() == tok_transpose)
        {
          auto transpose = ParseTranspose();
          if (!transpose)
            return nullptr;
          exprList->push_back(std::move(transpose));
        }
        else if (lexer.getCurToken() == tok_return)
        {
          // Return statement
          auto ret = parseReturn();
          if (!ret)
            return nullptr;
          exprList->push_back(std::move(ret));
        }
        else if (lexer.getCurToken() == tok_identifier &&
                 lexer.lookAhead() == '[')
        {
          auto tensOp = ParseTensorExpression();
          if (!tensOp)
            return nullptr;
          exprList->push_back(std::move(tensOp));
        }
        else if (lexer.getCurToken() == tok_identifier &&
                 lexer.lookAhead() == ' ')
        {
          auto varOp = ParseVarExpression();
          if (!varOp)
            return nullptr;
          exprList->push_back(std::move(varOp));
        }
        else
        {
          auto loc = lexer.getLastLocation();
          auto expr = parseExpression();
          if (!expr)
            return nullptr;
          exprList->push_back(std::move(expr));
        }
        // Ensure that elements are separated by a semicolon.
        if (lexer.getCurToken() != ';')
          return parseError<ExprASTList>(";", "after expression");

        // Ignore empty expressions: swallow sequences of semicolons.
        while (lexer.getCurToken() == ';')
          lexer.consume(Token(';'));
      }

      if (lexer.getCurToken() != '}')
        return parseError<ExprASTList>("}", "to close block");

      lexer.consume(Token('}'));
      return exprList;
    }

    /// prototype ::= def id '(' decl_list ')'
    /// decl_list ::= identifier | identifier, decl_list
    std::unique_ptr<PrototypeAST> parsePrototype()
    {
      auto loc = lexer.getLastLocation();
      lexer.consume(tok_def);
      if (lexer.getCurToken() != tok_identifier)
        return parseError<PrototypeAST>("function name", "in prototype");

      std::string fnName(lexer.getId());
      lexer.consume(tok_identifier);

      if (lexer.getCurToken() != '(')
        return parseError<PrototypeAST>("(", "in prototype");
      lexer.consume(Token('('));

      std::vector<std::unique_ptr<VariableExprAST>> args;
      if (lexer.getCurToken() != ')')
      {
        do
        {
          std::string name(lexer.getId());
          auto loc = lexer.getLastLocation();
          lexer.consume(tok_identifier);
          auto decl = std::make_unique<VariableExprAST>(std::move(loc), name);
          args.push_back(std::move(decl));
          if (lexer.getCurToken() != ',')
            break;
          lexer.consume(Token(','));
          if (lexer.getCurToken() != tok_identifier)
            return parseError<PrototypeAST>(
                "identifier", "after ',' in function parameter list");
        } while (true);
      }
      if (lexer.getCurToken() != ')')
        return parseError<PrototypeAST>("}", "to end function prototype");

      // success.
      lexer.consume(Token(')'));
      return std::make_unique<PrototypeAST>(std::move(loc), fnName,
                                            std::move(args));
    }

    /// Parse a function definition, we expect a prototype initiated with the
    /// `def` keyword, followed by a block containing a list of expressions.
    ///
    /// definition ::= prototype block
    std::unique_ptr<FunctionAST> parseDefinition()
    {
      auto proto = parsePrototype();
      if (!proto)
        return nullptr;

      if (auto block = parseBlock())
        return std::make_unique<FunctionAST>(std::move(proto), std::move(block));
      return nullptr;
    }

    /// Get the precedence of the pending binary operator token.
    int getTokPrecedence()
    {
      comet_errs() << " In getTokPrecedence " << lexer.getCurToken() << "\n";
      if (lexer.getCurToken() == tok_elews)
      {
        comet_errs() << " set .* elementwise op priority\n";
        return 40;
      }

      if (lexer.getCurToken() == tok_semiring)
      {
        comet_errs() << " set @(xx,xx) semiring op priority\n";
        return 40;
      }

      if (lexer.getCurToken() == tok_monoid)
      {
        comet_errs() << " set .(xx) monoid op priority\n";
        return 40;
      }

      if (!isascii(lexer.getCurToken()))
      {
        comet_errs() << " not ascii: " << lexer.getCurToken() << "\n";
        return -1;
      }

      // 1 is lowest precedence.
      switch (static_cast<char>(lexer.getCurToken()))
      {
      case '-':
        return 20;
      case '+':
        return 20;
      case '*':
        return 40;
      case '/':
        return 40;
      default:
        return -1;
      }
    }

    /// Helper function to signal errors while parsing, it takes an argument
    /// indicating the expected token and another argument giving more context.
    /// Location is retrieved from the lexer to enrich the error message.
    template <typename R, typename T, typename U = const char *>
    std::unique_ptr<R> parseError(T &&expected, U &&context = "")
    {
      auto curToken = lexer.getCurToken();
      llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                   << lexer.getLastLocation().col << "): expected '" << expected
                   << "' " << context << " but has Token " << curToken;
      if (isprint(curToken))
        llvm::errs() << " '" << (char)curToken << "'";
      llvm::errs() << "\n";
      return nullptr;
    }
  };

} // namespace tensorAlgebra

#endif // COMET_DSL_PARSER_PARSER_H