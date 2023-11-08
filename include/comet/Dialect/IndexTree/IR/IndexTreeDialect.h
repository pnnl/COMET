//===- IndexTreeDialect.h - Dialect definition for the IndexTree IR ----------------------===//
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
// This file implements the Index Tree dialect
//
//===----------------------------------------------------------------------===//

#ifndef INDEXTREE_DIALECT_H_
#define INDEXTREE_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

/// Include the auto-generated header file containing the declaration of the index tree
/// dialect.
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h.inc"

/// Include the auto-generated header file containing the declaration of the index tree
/// types.
#define GET_TYPEDEF_CLASSES
#include "comet/Dialect/IndexTree/IR/IndexTreeTypes.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// Index Tree operations and also the operations of the Shape Inference Op Interface.
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "comet/Dialect/IndexTree/IR/IndexTreeOps.h.inc"

//===----------------------------------------------------------------------===//


// namespace mlir
// {
//   namespace indexTree
//   {
//     /// This is the definition of the Index Tree (IT) dialect. A dialect inherits from
//     /// mlir::Dialect and registers custom attributes, operations, and types (in its
//     /// constructor). It can also override some general behavior exposed via virtual
//     /// methods.
//     class IndexTreeDialect : public mlir::Dialect
//     {
//     public:
//       explicit IndexTreeDialect(mlir::MLIRContext *ctx);

//       /// Provide a utility accessor to the dialect namespace. This is used by
//       /// several utilities for casting between dialects.
//       static llvm::StringRef getDialectNamespace() { return "it"; }

//       /// Parse a type registered to this dialect.
//       Type parseType(DialectAsmParser &parser) const override;
//     };
// //===----------------------------------------------------------------------===//
// #define GET_OP_CLASSES
// #include "comet/Dialect/IndexTree/IR/IndexTreeOps.h.inc"

//   } // end namespace indexTree
// } // end namespace mlir

#endif // INDEXTREE_DIALECT_H_
