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

#ifndef ITERATIONDOMAIN_H
#define ITERATIONDOMAIN_H

#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <cassert>

using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;

class Tensor;

typedef std::pair<Tensor *, int> BDDElem;

/*
 * node1:
 *   op: '*'
 *   left: node2
 *   right: node3
 *   tensor: null
 * node2:
 *   op: ''
 *   left: null
 *   right: null
 *   tensor: A
 *   dim: 1
 * node3:
 *   op: ''
 *   left: null
 *   right: null
 *   tensor: B
 *   dim: 0
 */
class IterDomain
{
  Tensor *tensor = nullptr;
  int dim;
  char op;
  IterDomain *left = nullptr;
  IterDomain *right = nullptr;

public:
  static IterDomain *makeDomain(Tensor *tensor, int dim);

  IterDomain(Tensor *tensor, int dim) : tensor(tensor), dim(dim){};

  IterDomain(char op, IterDomain *left, IterDomain *right) : op(op), left(left), right(right){};

  Tensor *getTensor() const;
  void setTensor(Tensor *Tensor);
  int getDim() const;
  void setDim(int Dim);
  char getOp() const;
  void setOp(char Op);
  IterDomain *getLeft() const;
  void setLeft(IterDomain *Left);
  IterDomain *getRight() const;
  void setRight(IterDomain *Right);

  bool isLeafNode()
  {
    return left == nullptr && right == nullptr;
  }

  bool equals(IterDomain *that);

  IterDomain *getSimplified();

  std::string getFormat();

  bool isDense();

  std::string str();

  static IterDomain *conjunct(IterDomain *a, IterDomain *b);

  static IterDomain *conjunct(std::vector<IterDomain *> &domains);
};

class BoolExpr
{
  typedef BDDElem T;
  T value;
  shared_ptr<BoolExpr> low;
  shared_ptr<BoolExpr> high;
  bool isConstantTrue = false;
  bool isConstantFalse = false;

public:
  static BoolExpr *conjunct(std::vector<BoolExpr *> &conds)
  {
    assert(conds.size() > 0);
    for (auto c : conds)
    {
      if (!c->isTrue())
      {
        return c;
      }
    }

    auto cond0 = conds[0];
    return cond0;
  }

  BoolExpr(){};

  BoolExpr(T value) : value(value) {}

  const shared_ptr<BoolExpr> &getLow() const
  {
    return low;
  }
  void setLow(const shared_ptr<BoolExpr> &Low)
  {
    low = Low;
  }
  const shared_ptr<BoolExpr> &getHigh() const
  {
    return high;
  }
  void setHigh(const shared_ptr<BoolExpr> &High)
  {
    high = High;
  }

  void setTrue();
  void setFalse();

  bool isTrue()
  {
    return isConstantTrue;
  }

  bool isFalse()
  {
    return isConstantFalse;
  }

  bool equals(BoolExpr *that)
  {
    llvm::errs() << "ERROR: Unsupported boolean operation\n";
    return false;
  }

  std::string str();
};

class BoolExprManager
{

public:
  static std::vector<unique_ptr<BoolExpr>> exprs;
  static unique_ptr<BoolExpr> trueNode;
  static BoolExpr *getTrue();

  static BoolExpr *makeBoolExpr(BDDElem elem);
};

#endif // ITERATIONDOMAIN_H
