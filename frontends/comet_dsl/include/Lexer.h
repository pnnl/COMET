//===- Lexer.h - Lexer for the COMET Domain Specific Language (DSL)
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
// This file implements a simple Lexer for the COMET Domain Specific Langauge
//
//===----------------------------------------------------------------------===//

#ifndef COMET_DSL_LEXER_H_
#define COMET_DSL_LEXER_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

namespace tensorAlgebra
{
  /// Structure definition a location in a file.
  struct Location
  {
    std::shared_ptr<std::string> file; ///< filename.
    int line;                          ///< line number.
    int col;                           ///< column number.
  };

  // List of Token returned by the lexer.
  enum Token : int
  {
    tok_semicolon = ';',
    tok_colon = ':',
    tok_parenthese_open = '(',
    tok_parenthese_close = ')',
    tok_bracket_open = '{',
    tok_bracket_close = '}',
    tok_sbracket_open = '[',
    tok_sbracket_close = ']',
    tok_mask_open = '<',
    tok_mask_close = '>',
    tok_quotation = '"',
    tok_eof = -1,

    // commands
    tok_return = -2,
    tok_var = -3,
    tok_def = -4,

    // primary
    tok_identifier = -5,
    tok_number = -6,

    /// tensor algebra
    tok_index_space = -7,
    tok_index_label = -8,
    tok_tensor = -9,
    tok_double = -10,
    tok_float = -11,
    tok_int = -12,
    tok_dynamic_index_label = -13,
    tok_transpose = -14,
    tok_elews = -15,
    tok_semiring = -16,
    tok_monoid = -17,
    tok_for = -18,      /// for loop
    tok_end = -19,      /// for loop-body end
    tok_maskPush = -20, /// push keyword for masking support
    tok_maskPull = -21, /// pull keyword for masking support
    tok_maskAuto = -22  /// auto keyword for masking support
  };

  enum SemiringOp : int
  {
    semiring_lor = 100,
    semiring_land = 101,
    semiring_min = 102,
    semiring_first = 103,
    semiring_plus = 104,
    semiring_times = 105,
    semiring_any = 106,
    semiring_pair = 107,
    semiring_second = 108,
    semiring_minus = 109,
    semiring_noop = 110 /// for monoids
  };

  /// The Lexer is an abstract base class providing all the facilities that the
  /// Parser expects. It goes through the stream one token at a time and keeps
  /// track of the location in the file for debugging purpose.
  /// It relies on a subclass to provide a `readNextLine()` method. The subclass
  /// can proceed by reading the next line from the standard input or from a
  /// memory mapped file.
  class Lexer
  {
  public:
    /// Create a lexer for the given filename. The filename is kept only for
    /// debugging purpose (attaching a location to a Token).
    Lexer(std::string filename)
        : lastLocation(
              {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}
    virtual ~Lexer() = default;

    /// Look at the current token in the stream.
    Token getCurToken() { return curTok; }

    /// Move to the next token in the stream and return it.
    Token getNextToken() { return curTok = getTok(); }

    /// Move to the next token in the stream, asserting on the current token
    /// matching the expectation.
    void consume(Token tok)
    {
      assert(tok == curTok && "consume Token mismatch expectation");
      comet_debug() << "consume one token: " << tok << " \n";
      Token nextTok = getNextToken();
      comet_debug() << "next token: " << nextTok << " \n";
    }

    /// Return the current identifier (prereq: getCurToken() == tok_identifier)
    llvm::StringRef getId()
    {
      assert(curTok == tok_identifier);
      return IdentifierStr;
    }

    /// Return the current number (prereq: getCurToken() == tok_number)
    double getValue()
    {
      assert(curTok == tok_number);
      return NumVal;
    }

    /// Return the location for the beginning of the current token.
    Location getLastLocation() { return lastLocation; }

    /// Return the current line in the file.
    int getLine() { return curLineNum; }

    /// Return the current column in the file.
    int getCol() { return curCol; }

    Token lookAhead() { return LastChar; }

    /// Return 1st operator of semiring
    SemiringOp getSemiring1st() { return Op1st; }

    /// Return 2nd operator of semiring
    SemiringOp getSemiring2nd() { return Op2nd; }

  private:
    /// Delegate to a derived class fetching the next line. Returns an empty
    /// string to signal end of file (EOF). Lines are expected to always finish
    /// with "\n"
    virtual llvm::StringRef readNextLine() = 0;

    /// Return the next character from the stream. This manages the buffer for the
    /// current line and request the next line buffer to the derived class as
    /// needed.
    int getNextChar()
    {
      /// The current line buffer should not be empty unless it is the end of file.
      if (curLineBuffer.empty())
        return EOF;
      ++curCol;
      auto nextchar = curLineBuffer.front();
      curLineBuffer = curLineBuffer.drop_front();
      if (curLineBuffer.empty())
        curLineBuffer = readNextLine();
      if (nextchar == '\n')
      {
        ++curLineNum;
        curCol = 0;
      }
      comet_debug() << "next char: " << nextchar << "\n";
      return nextchar;
    }

    /// Check next char
    int checkNextChar()
    {
      /// The current line buffer should not be empty unless it is the end of file.
      if (curLineBuffer.empty())
        return EOF;
      /// ++curCol;
      auto nextchar = curLineBuffer.front();
      comet_debug() << "next char: " << nextchar << "\n";
      return nextchar;
    }

    SemiringOp getSemiringOp(std::string semiringStr)
    {
      if (semiringStr == "|")
      {
        return semiring_lor;
      }
      else if (semiringStr == "&")
      {
        return semiring_land;
      }
      else if (semiringStr == "min")
      {
        return semiring_min;
      }
      else if (semiringStr == "first")
      {
        return semiring_first;
      }
      else if (semiringStr == "+")
      {
        return semiring_plus;
      }
      else if (semiringStr == "-")
      {
        return semiring_minus;
      }
      else if (semiringStr == "*")
      {
        return semiring_times;
      }
      else if (semiringStr == "any")
      {
        return semiring_any;
      }
      else if (semiringStr == "pair")
      {
        return semiring_pair;
      }
      else if (semiringStr == "second")
      {
        return semiring_second;
      }
      else
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << "[ERR] Undefined semiring operator. Please check syntax!\n";
        exit(1);
      }
    }

    /// get semiringStr
    std::string getSemiringStr()
    {
      std::string semiringStr;
      while (true)
      {
        LastChar = Token(getNextChar());
        if (LastChar == ' ')
          continue;
        if (LastChar == ',' || LastChar == ')')
          break;
        semiringStr += (char)LastChar;
      }
      return semiringStr;
    }

    /// get monoid string
    std::string getMonoidStr()
    {
      std::string monoidStr;
      while (true)
      {
        LastChar = Token(getNextChar());
        if (LastChar == ' ')
          continue; // ignore spaces
        if (LastChar == ')')
          break;
        monoidStr += (char)LastChar;
      }
      return monoidStr;
    }

    ///  Return the next token from standard input.
    Token getTok()
    {
      comet_debug() << "go to next token\n";
      /// Skip any whitespace.
      while (isspace(LastChar))
        LastChar = Token(getNextChar());
      comet_debug() << "LastChar: " << LastChar << "\n";

      /// Save the current location before reading the token characters.
      lastLocation.line = curLineNum;
      lastLocation.col = curCol;
      comet_debug() << "lastLocation: " << lastLocation.line << " " << lastLocation.col << "\n";

      if (isalpha(LastChar))
      { /// identifier: [a-zA-Z][a-zA-Z0-9_]*
        comet_debug() << "LastChar (" << LastChar << ") isalpha\n";
        IdentifierStr = (char)LastChar;
        while (isalnum((LastChar = Token(getNextChar()))) || LastChar == '_')
          IdentifierStr += (char)LastChar;

        if (IdentifierStr == "return")
          return tok_return;
        if (IdentifierStr == "def")
          return tok_def;
        /// var is used to define as a return value for utility functions
        /// such as "var t_start = getTime();""
        if (IdentifierStr == "var")
          return tok_var;
        if (IdentifierStr == "IndexSpace")
          return tok_index_space;
        if (IdentifierStr == "IndexLabel")
          return tok_index_label;
        if (IdentifierStr == "Tensor")
          return tok_tensor;
        if (IdentifierStr == "double")
          return tok_double;
        if (IdentifierStr == "float")
          return tok_float;
        if (IdentifierStr == "int")
          return tok_int;
        if (IdentifierStr == "transpose")
          return tok_transpose;
        if (IdentifierStr == "for")
          return tok_for;
        if (IdentifierStr == "end")
          return tok_end;
        if (IdentifierStr == "push")
          return tok_maskPush;
        if (IdentifierStr == "pull")
          return tok_maskPull;
        if (IdentifierStr == "auto")
          return tok_maskAuto;

        comet_debug() << "Identifier:" << IdentifierStr << "\n";
        return tok_identifier;
      }

      if (LastChar == '*')
      {
        Op1st = semiring_plus;
        Op2nd = semiring_times;
      }

      if (LastChar == '+')
      {
        Op1st = semiring_noop;
        Op2nd = semiring_plus;
      }

      if (LastChar == '-')
      {
        Op1st = semiring_noop;
        Op2nd = semiring_minus;
      }

      if (LastChar == '/')
      {
        comet_debug() << " ";
        if (curLineBuffer.front() == ' ')
        {
          Token curChar = Token(LastChar);
          LastChar = Token(getNextChar());
          return curChar;
        }
        else
        {
          IdentifierStr = (char)LastChar;

          while (isalnum((LastChar = Token(getNextChar()))) || LastChar == '_' || LastChar == '/' || LastChar == '.')
          {
            if (LastChar == '/')
              comet_debug() << '/' << "\n";
            comet_debug() << IdentifierStr << "\n";
            IdentifierStr += (char)LastChar;
          }

          comet_debug() << IdentifierStr << "\n";
          return tok_identifier;
        }
      }

      /// for "/home/a.mtx"
      if (LastChar == tok_quotation)
      {
        comet_debug() << " the token is \" \n";

        IdentifierStr = "";
        LastChar = Token(getNextChar());
        while (LastChar != tok_quotation)
        {
          comet_debug() << IdentifierStr << "\n";
          if (LastChar == tok_semicolon || LastChar == tok_colon || LastChar == tok_parenthese_open || LastChar == tok_parenthese_close || LastChar == tok_bracket_open || LastChar == tok_bracket_close || LastChar == tok_sbracket_open || LastChar == tok_sbracket_close || LastChar == tok_mask_open || LastChar == tok_mask_close)
          {
            comet_debug() << "not allow special tokens in strings\n";
            llvm::errs() << "Special tokens in strings are not allowed, source location (" << curLineNum << ", " << curCol << ")\n";
            exit(0);
          }
          IdentifierStr += (char)LastChar;
          LastChar = Token(getNextChar());
        }

        comet_debug() << " " << IdentifierStr << "\n";
        LastChar = Token(getNextChar()); /// consume the other " symbol
        return tok_identifier;
      }

      if (LastChar == '.' && checkNextChar() == '*')
      {
        comet_debug() << " \n";
        std::string elewsStr;
        elewsStr += LastChar;
        LastChar = Token(getNextChar());
        elewsStr += LastChar;
        comet_debug() << " the elewsStr: " << elewsStr << "\n";
        comet_debug() << " .* for elews\n";
        LastChar = Token(getNextChar());
        Op1st = semiring_noop;  /// default for element-wise mult op
        Op2nd = semiring_times; /// default for element-wise mult op
        return tok_elews;
      }

      /// for semiring and monoid operations
      /// semiring: @(xx,xx)
      /// monoid: @(xx)
      /// TODO(rizwan): the above syntax needs to be followed strictly at this time.
      ///               the call to getSemiringStr() will however catch non-supported
      ///               semiring or monoid ops.
      if (LastChar == '@' && checkNextChar() == '(')
      {
        comet_debug() << "semiring operations \n";
        std::string semiringStr[2] = {"", ""};
        LastChar = Token(getNextChar());
        semiringStr[0] = getSemiringStr();

        if (LastChar == ')')
        { /// it must be a monoid, don't call getSemiringStr()
          /// since it makes a call to getNextChar().
          Op1st = semiring_noop;
          Op2nd = getSemiringOp(semiringStr[0]);
          return tok_monoid;
        }
        else
        { /// it must be a semiring, it's ok to call getSemiringStr()
          semiringStr[1] = getSemiringStr();
          Op1st = getSemiringOp(semiringStr[0]);
          Op2nd = getSemiringOp(semiringStr[1]);
          return tok_semiring;
        }
      }

      if (isdigit(LastChar) || LastChar == '.')
      { /// Number: [0-9.]+
        std::string NumStr;
        do
        {
          NumStr += LastChar;
          LastChar = Token(getNextChar());
        } while (isdigit(LastChar) || LastChar == '.');

        NumVal = strtod(NumStr.c_str(), nullptr);
        return tok_number;
      }

      if (LastChar == '#')
      {
        /// Comment until end of line.
        do
          LastChar = Token(getNextChar());
        while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

        if (LastChar != EOF)
          return getTok();
      }

      /// Check for end of file.  Don't eat the EOF.
      if (LastChar == EOF)
        return tok_eof;

      /// Otherwise, just return the character as its ascii value.
      Token ThisChar = Token(LastChar);
      LastChar = Token(getNextChar());
      return ThisChar;
    }

    /// The last token read from the input.
    Token curTok = tok_eof;

    /// Location for `curTok`.
    Location lastLocation;

    /// If the current Token is an identifier, this string contains the value.
    std::string IdentifierStr;

    /// If the current Token is a number, this contains the value.
    double NumVal = 0;

    /// The last value returned by getNextChar(). We need to keep it around as we
    /// always need to read ahead one character to decide when to end a token and
    /// we can't put it back in the stream after reading from it.
    Token LastChar = Token(' ');

    /// Keep track of the current line number in the input stream
    int curLineNum = 0;

    /// Keep track of the current column number in the input stream
    int curCol = 0;

    /// Buffer supplied by the derived class on calls to `readNextLine()`
    llvm::StringRef curLineBuffer = "\n";

    /// declaration of 1st Operator for semiring
    SemiringOp Op1st = SemiringOp(semiring_noop); // default (to be set properly per operand)
    /// declaration of 2nd Operator for semiring
    SemiringOp Op2nd = SemiringOp(semiring_noop); // default (to be set properly per operand)
  };

  /// A lexer implementation operating on a buffer in memory.
  class LexerBuffer final : public Lexer
  {
  public:
    LexerBuffer(const char *begin, const char *end, std::string filename)
        : Lexer(std::move(filename)), current(begin), end(end) {}

  private:
    /// Provide one line at a time to the Lexer, return an empty string when
    /// reaching the end of the buffer.
    llvm::StringRef readNextLine() override
    {
      auto *begin = current;
      while (current <= end && *current && *current != '\n')
        ++current;
      if (current <= end && *current)
        ++current;
      llvm::StringRef result{begin, static_cast<size_t>(current - begin)};
      return result;
    }
    const char *current, *end;
  };
} /// namespace tensorAlgebra

#endif /// COMET_DSL_LEXER_H_