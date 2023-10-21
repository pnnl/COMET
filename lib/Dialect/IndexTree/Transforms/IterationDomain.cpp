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

#include "comet/Dialect/IndexTree/Transforms/IterationDomain.h"
#include "comet/Dialect/IndexTree/Transforms/Tensor.h"

using namespace std;
// Since multiple threads can lower different functions,
// we need one for each thread lowering.
thread_local std::vector<unique_ptr<IterDomain>> domains;

IterDomain *IterDomain::makeDomain(Tensor *tensor, int dim)
{
  auto d = make_unique<IterDomain>(tensor, dim);
  auto p = d.get();
  domains.push_back(std::move(d));
  return p;
}

IterDomain *IterDomain::conjunct(IterDomain *a, IterDomain *b)
{
  auto d = make_unique<IterDomain>('*', a, b);
  auto p = d.get();
  domains.push_back(std::move(d));
  return p;
}

IterDomain *IterDomain::conjunct(std::vector<IterDomain *> &domains)
{
  assert(!domains.empty());
  auto d = domains[0];
  for (unsigned long i = 1; i < domains.size(); i++)
  {
    d = conjunct(d, domains[i]);
  }
  return d;
}

bool IterDomain::equals(IterDomain *that)
{
  auto thisSimplified = this->getSimplified();
  auto thatSimplified = that->getSimplified();
  return thisSimplified == thatSimplified;
}

IterDomain *IterDomain::getSimplified()
{
  if (getOp() == '*')
  {
    if (getLeft()->isDense())
    {
      return getRight();
    }
    else if (getRight()->isDense())
    {
      return getLeft();
    }
  }

  return this;
}

std::string IterDomain::str()
{
  if (isLeafNode())
  {
    string s = "(" + getTensor()->str() + "," + to_string(getDim()) + ")";
    return s;
  }
  else
  {
    assert(getLeft() != nullptr && getRight() != nullptr);
    string s = getLeft()->str() + string(1, getOp()) + getRight()->str();
    return s;
  }
}

std::string IterDomain::getFormat()
{
  return getTensor()->getFormat(getDim());
}

bool IterDomain::isDense()
{
  return getFormat() == "D";
}

std::string BoolExpr::str()
{
  if (isTrue())
  {
    return "true";
  }

  auto t = value.first;
  assert(t != nullptr);
  auto dim = value.second;
  return "(" + t->str() + ", " + to_string(dim) + ")";
}

void BoolExpr::setTrue()
{
  isConstantTrue = true;
}

void BoolExpr::setFalse()
{
  isConstantFalse = true;
}

unique_ptr<BoolExpr> BoolExprManager::trueNode;

std::vector<unique_ptr<BoolExpr>> BoolExprManager::exprs;

BoolExpr *BoolExprManager::getTrue()
{
  if (trueNode == nullptr)
  {
    trueNode = make_unique<BoolExpr>();
    trueNode->setTrue();
  }

  return trueNode.get();
}

BoolExpr *BoolExprManager::makeBoolExpr(BDDElem elem)
{
  auto expr = make_unique<BoolExpr>(elem);
  BoolExpr *ret = expr.get();
  exprs.push_back(std::move(expr));
  return ret;
}
Tensor *IterDomain::getTensor() const
{
  return tensor;
}
void IterDomain::setTensor(Tensor *Tensor)
{
  tensor = Tensor;
}
int IterDomain::getDim() const
{
  return dim;
}
void IterDomain::setDim(int Dim)
{
  dim = Dim;
}
char IterDomain::getOp() const
{
  return op;
}
void IterDomain::setOp(char Op)
{
  op = Op;
}
IterDomain *IterDomain::getLeft() const
{
  return left;
}
void IterDomain::setLeft(IterDomain *Left)
{
  left = Left;
}
IterDomain *IterDomain::getRight() const
{
  return right;
}
void IterDomain::setRight(IterDomain *Right)
{
  right = Right;
}
