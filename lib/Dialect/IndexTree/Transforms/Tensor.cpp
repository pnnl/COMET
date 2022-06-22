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

#include "comet/Dialect/IndexTree/Transforms/Tensor.h"
#include "comet/Dialect/IndexTree/Transforms/IterationDomain.h"

using namespace std;

int Tensor::count = 0;

IterDomain *Tensor::getDomainAtDim(int dim)
{
  if (domains.at(dim) == nullptr)
  {
    setDomain(dim, IterDomain::makeDomain(this, dim));
  }

  return domains.at(dim);
}

void Tensor::setDomain(std::map<int, IterDomain *> &indexDomains)
{
  for (int d = 0; d < getDimNum(); d++)
  {
    int index = getIndex(d);
    setDomain(d, indexDomains.at(index));
  }
}

string Tensor::str(int withFormats)
{
  string s = getName();
  s += '[';
  bool first = true;
  for (int i = 0; i < indices.size(); i++)
  {
    int index = getIndex(i);

    if (find(hidden.begin(), hidden.end(), index) != hidden.end())
    {
      continue;
    }

    string format = "";
    if (withFormats)
    {
      format = "(" + getFormat(i) + ")";
    }

    if (first)
    {
      s += "i" + to_string(index) + format;
      first = false;
    }
    else
    {
      s += ", i" + to_string(index) + format;
    }
  }
  s += ']';
  return s;
}

mlir::Value &Tensor::getValue()
{
  return value;
}
