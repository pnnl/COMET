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

#ifndef INDEXTREE_UNITEXPRESSION_H
#define INDEXTREE_UNITEXPRESSION_H

#include <mlir/IR/Operation.h>

#include "comet/Dialect/IndexTree/Transforms/Tensor.h"

class UnitExpression
{
  Tensor *output = nullptr;
  Tensor *mask = nullptr;
  vector<Tensor *> operands;
  vector<UnitExpression *> users;
  mlir::Operation *operation;
  string opType;
  llvm::StringRef semiring;
  llvm::StringRef maskType;
  int numOps = 2;

  bool traceDomainCompute = false;

public:
  UnitExpression(Tensor *output,
                 Tensor *operand1,
                 Tensor *operand2, string op)
      : output(output), opType(op)
  {
    operands.push_back(operand1);
    operands.push_back(operand2);
  }

  UnitExpression(Tensor *output,
                 Tensor *operand1,
                 string op)
      : output(output), opType(op)
  {
    operands.push_back(operand1);
  }

  UnitExpression(Tensor *output, string op)
      : output(output), opType(op)
  {
  }

  UnitExpression(Tensor *output,
                 Tensor *operand1,
                 Tensor *operand2,
                 Tensor *mask,
                 string op)
      : output(output), mask(mask), opType(op)
  {
    operands.push_back(operand1);
    operands.push_back(operand2);
  } /// constructor with mask operand

  Tensor *getLHS()
  {
    return output;
  }

  Tensor *getMask()
  {
    return mask;
  }

  int getNumOfOperands()
  {
    return numOps;
  }

  Tensor *getOperand(int i)
  {
    return operands.at(i);
  }

  const vector<Tensor *> &getOperands()
  {
    return operands;
  }

  const vector<Tensor *> getResults()
  {
    vector<Tensor *> ret;
    ret.push_back(getLHS());
    return ret;
  }

  const vector<Tensor *> getResultsAndOperands()
  {
    vector<Tensor *> ret;
    ret.push_back(getLHS());
    for (auto &o : getOperands())
    {
      if (o != nullptr)
        ret.push_back(o);
    }
    return ret;
  }

  const vector<Tensor *> getOperandsAndResults()
  {
    vector<Tensor *> ret;
    for (auto &o : getOperands())
    {
      if (o != nullptr)
        ret.push_back(o);
    }
    ret.push_back(getLHS());
    return ret;
  }

  mlir::Operation *getOperation() const;
  void setOperation(mlir::Operation *Operation);
  void addUser(UnitExpression *user);

  const vector<UnitExpression *> &getUsers()
  {
    return users;
  }

  string str();

  void reduceOutputDimension(IndicesType &indices)
  {
    output->setHiddenIndices(indices);
  }

  std::map<int, IterDomain *> computeInputIterDomains();

  std::map<int, IterDomain *> computeOutputIterDomains();

  static unique_ptr<UnitExpression> createMemsetExpression(Tensor *t)
  {
    auto e = make_unique<UnitExpression>(t, "memset");
    return e;
  }

  const string &getOpType() const;
  void setOpType(const string &OpType);
  const llvm::StringRef &getSemiring() const;
  void setSemiring(const llvm::StringRef &Semiring);
  const llvm::StringRef &getMaskType() const;
  void setMaskType(const llvm::StringRef &MaskType);
};

#endif /// INDEXTREE_UNITEXPRESSION_H
