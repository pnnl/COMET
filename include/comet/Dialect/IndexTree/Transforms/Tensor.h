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

#ifndef INDEXTREE_TENSOR_H
#define INDEXTREE_TENSOR_H

#include "comet/Dialect/IndexTree/Transforms/IterationDomain.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <algorithm>
#include <map>
#include "mlir/IR/Value.h"

typedef std::vector<unsigned int> IndicesType;
typedef std::vector<std::string> FormatsType;
typedef std::vector<IterDomain *> DomainsType;

using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

class UnitExpression;

class Tensor
{
private:
  mlir::Value value;
  string name;
  IndicesType indices;
  FormatsType format;
  IndicesType hidden;
  DomainsType domains;
  UnitExpression *definingExpr = nullptr;
  int id = 0;

public:
  static int count;

  Tensor(mlir::Value &value, IndicesType &indices, vector<string> &format)
  {
    assert(format.size() == indices.size());
    this->value = value;
    this->indices = indices;
    this->format = format;
    id = count++;
    domains = std::vector<IterDomain *>(indices.size());
  }

  Tensor(mlir::Value &value)
  {
    this->value = value;
  }

  bool isConstantScalar();

  const string getName() const
  {
    assert(id >= 0);
    return "T" + std::to_string(id);
  }

  void setName(const string &name)
  {
    Tensor::name = name;
  }

  int getDimNum()
  {
    return indices.size();
  }

  const IndicesType &getIndices() const
  {
    return indices;
  }

  void setIndices(IndicesType &Updatedindices)
  {
    indices = Updatedindices;
  }

  int getIndex(int i)
  {
    return indices.at(i);
  }

  IterDomain *getDomainAtDim(int i);

  bool hasComputedDomain(int i)
  {
    return domains.at(i) != nullptr;
  }

  void setDomain(int dim, IterDomain *e)
  {
    domains[dim] = e;
  }

  void setDomain(std::map<int, IterDomain *> &indexDomains);

  UnitExpression *getDefiningExpr()
  {
    return definingExpr;
  }

  void setDefiningExpr(UnitExpression *expr)
  {
    definingExpr = expr;
  }

  string getStrIndex(int i);

  void setIndices(const IndicesType &indices)
  {
    Tensor::indices = indices;
  }

  const string &getFormat(int i) const
  {
    return format.at(i);
  }

  FormatsType &getFormats()
  {
    return format;
  }

  void setFormat(const vector<string> &format)
  {
    Tensor::format = format;
  }

  void setHiddenIndices(IndicesType &hiddenIndices)
  {
    hidden = hiddenIndices;
  }

  string str(int withFormats = false);
  mlir::Value &getValue();
};

#endif // INDEXTREE_TENSOR_H
