//===- TADialect.h - Tensor Algebra Dialect definition----------------------===//
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
// This file implements the Tensor algebra Dialect
//
//===----------------------------------------------------------------------===//

#ifndef TENSORALGEBRA_DIALECT_H_
#define TENSORALGEBRA_DIALECT_H_

#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/IR/PatternMatch.h"

/// Include the auto-generated header file containing the declaration of the Tensor Algebra
/// dialect.
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// tensorAlgbra operations and also the operations of the Shape Inference Op Interface.
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "comet/Dialect/TensorAlgebra/IR/TAOps.h.inc"

namespace mlir
{
  namespace tensorAlgebra
  {
    std::vector<Value> getFormatsValue(std::string formats_str, int rank_size, PatternRewriter &rewriter, Location loc, IndexType indexType);

    namespace detail
    {
      struct SparseTensorTypeStorage;
    } /// end namespace detail

    void populateMultiOpFactorizationPatterns(
        RewritePatternSet &patterns, MLIRContext *context);

    void populateLowerTAMulChainPatterns(
        RewritePatternSet &patterns, MLIRContext *context);

    void populateSCFLowerToSCFParallelPatterns(
        RewritePatternSet &patterns, MLIRContext *context);

    void populateSTCRemoveDeadOpsPatterns(
        RewritePatternSet &patterns, MLIRContext *context);

    //===----------------------------------------------------------------------===//
    /// Tensor Algebra Types
    //===----------------------------------------------------------------------===//

    /// This class defines the TA sparse tensor type. It represents a collection of
    /// element types for data and indices of COO format.
    /// All derived types in MLIR must inherit from the CRTP class
    /// 'Type::TypeBase'. It takes as template parameters the concrete type
    /// (SparseTensorType), the base class to use (Type), and the storage class
    /// (SparseTensorTypeStorage).
    class SparseTensorType : public mlir::Type::TypeBase<SparseTensorType, mlir::Type,
                                                         detail::SparseTensorTypeStorage>
    {
    public:
      /// Inherit some necessary constructors from 'TypeBase'.
      using Base::Base;

      /// Create an instance of a `SparseTensorType` with the given element types. There
      /// *must* be atleast one element type.
      static SparseTensorType get(llvm::ArrayRef<mlir::Type> elementTypes);

      /// Returns the element types of this sparse tensor type.
      llvm::ArrayRef<mlir::Type> getElementTypes();

      /// Returns the number of element type held by this sparse tensor.
      size_t getNumElementTypes() { return getElementTypes().size(); }
    };

  } /// end namespace tensorAlgebra
} /// end namespace mlir

#endif /// TENSORALGEBRA_DIALECT_H_
