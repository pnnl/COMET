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
#include "comet/Dialect/IndexTree/Transforms/UnitExpression.h"

#include <algorithm>

using std::map;
using std::vector;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

/**
 * An example that shows how to compute iteration domains.
 * T0[i0, i2, i1] = T1[i0, i2, i3] * T2[i3, i1]
 *
 * T0:
 *   index_map: 0, 2, 1
 *   formats: D, D, D
 * T1:
 *   index_map: 0, 2, 3
 *   formats: CU, CU, CU
 * T2:
 *   index_map: 3, 1
 *   formats: D, D
 *
 * To compute the iteration domains, we first need to record the tensors
 * that access each index variable, and the associated format.
 *
 * i0:
 *   T0[0], D
 *   T1[0], CU
 *   domain: T1[0]
 * i1:
 *   T0[2], D
 *   T2[1], D
 *   domain: D
 * i2:
 *   T0[1], D
 *   T1[1], CU
 *   domain: T1[1]
 * i3:
 *   T1[2], CU
 *   T2[0], D
 *   domain: T1[2]
 * @param op
 */
map<int, IterDomain *> UnitExpression::computeInputIterDomains()
{
  auto lhs = getLHS();

  if (traceDomainCompute)
  {
    comet_debug() << lhs->str() << "\n";
  }

  // Build the map from index to its domains
  map<int, vector<IterDomain *>> indexToDomains;
  vector<Tensor *> allTensors;
  allTensors.push_back(lhs);
  for (int i = 0; i < getNumOfOperands(); i++)
  {
    allTensors.push_back(getOperand(i));
  }

  allTensors = getOperands();

  for (unsigned int i = 0; i < allTensors.size(); i++)
  {
    auto t = allTensors[i];
    auto indices = t->getIndices();
    for (unsigned int dim = 0; dim < indices.size(); dim++)
    {
      int index = indices[dim];
      if (indexToDomains.count(index) == 0)
      {
        indexToDomains[index] = vector<IterDomain *>();
      }

      indexToDomains[index].push_back(t->getDomainAtDim(dim));
    }
  }

  map<int, IterDomain *> indexDomains;

  if (traceDomainCompute)
  {
    comet_debug() << "index and their domains\n";
  }

  for (auto &it : indexToDomains)
  {
    auto index = it.first;
    auto domains = it.second;

    if (traceDomainCompute)
    {
      comet_debug() << "index: " << index << "\n";
      for (auto d : domains)
      {
        comet_debug() << "  " << d->str() << "\n";
      }
    }

    if (opType == "*")
    {
      indexDomains[index] = IterDomain::conjunct(domains);
      if (traceDomainCompute)
      {
        comet_debug() << "  after conjunction: " << indexDomains[index]->str() << "\n";
      }
    }
    else if (opType == "+")
    {
      // TODO(gkestor): need support
      llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Unsupported operand\n";
    }
  }

  return indexDomains;
}

std::map<int, IterDomain *> UnitExpression::computeOutputIterDomains()
{
  map<int, IterDomain *> indexToDomains;
  auto lhs = getLHS();
  auto lhsIndices = lhs->getIndices();
  for (unsigned int d = 0; d < lhsIndices.size(); d++)
  {
    indexToDomains[lhs->getIndex(d)] = lhs->getDomainAtDim(d);
  }
  return indexToDomains;
}

string UnitExpression::str()
{
  assert(output != nullptr);

  string s;
  if (opType == "memset")
  {
    s = "memset(" + output->str() + ")";
  }
  else
  {
    auto operand1 = getOperand(0);
    auto operand2 = getOperand(1);
    assert(operand1 != nullptr);
    s = output->str() + " = " + operand1->str();
    if (operand2 != nullptr)
    {
      s += " " + opType + " " + operand2->str();
    }
  }
  return s;
}

void UnitExpression::addUser(UnitExpression *user)
{
  if (std::find(users.begin(), users.end(), user) == users.end())
  {
    users.push_back(user);
  }
}
mlir::Operation *UnitExpression::getOperation() const
{
  return operation;
}
void UnitExpression::setOperation(mlir::Operation *Operation)
{
  operation = Operation;
}
const string &UnitExpression::getOpType() const
{
  return opType;
}
void UnitExpression::setOpType(const string &OpType)
{
  opType = OpType;
}

const llvm::StringRef &UnitExpression::getSemiring() const
{
  return semiring;
}

void UnitExpression::setSemiring(const llvm::StringRef &Semiring)
{
  semiring = Semiring;
}

const llvm::StringRef &UnitExpression::getMaskType() const
{
  return maskType;
}

void UnitExpression::setMaskType(const llvm::StringRef &MaskType)
{
  maskType = MaskType;
}