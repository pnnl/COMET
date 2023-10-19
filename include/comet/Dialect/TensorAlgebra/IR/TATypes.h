//===- TATypes.h - Types definitions for the TensorAlgebra IR ----------------------===//
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
// This file implements the IR Dialect Types for Tensor Algebra Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TENSORALGEBRA_TYPES_H_
#define TENSORALGEBRA_TYPES_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir
{
  class MLIRContext;

  namespace tensorAlgebra
  {

    class RangeType : public Type::TypeBase<RangeType, Type, TypeStorage>
    {
    public:
      /// Used for generic hooks in TypeBase.
      using Base::Base;

      static RangeType get(MLIRContext *context)
      {
        /// Custom, uniq'ed construction in the MLIRContext.
        return Base::get(context);
      }
    };

  } /// namespace tensorAlgebra
} /// namespace mlir

#endif /// TENSORALGEBRA_TYPES_H_