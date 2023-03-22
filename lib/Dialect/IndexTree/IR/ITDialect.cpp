//===- Dialect.cpp - Index Tree  Dialect registration in MLIR ------------------===//
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
// This file implements the dialect for the IT IR
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include "comet/Dialect/IndexTree/IR/ITDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/InliningUtils.h"
//#include "mlir/Support/TypeID.h"

using namespace mlir;
using namespace mlir::indexTree;

//===----------------------------------------------------------------------===//
// ITDialect
//===----------------------------------------------------------------------===//

#include "comet/Dialect/IndexTree/IR/ITDialect.cpp.inc"

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
// ITDialect::ITDialect(mlir::MLIRContext *ctx) : mlir::Dialect("it", ctx, mlir::TypeID::get<ITDialect>())
// {
//   addOperations<
// #define GET_OP_LIST
// #include "comet/Dialect/IndexTree/IR/ITOps.cpp.inc"
//       >();
// }

Type mlir::indexTree::ITDialect::parseType(DialectAsmParser &parser) const
{
  // Parse the main keyword for the type.
  StringRef keyword;
  // for "range" and "sptensor" type
  if (parser.parseKeyword(&keyword))
    return Type();

  parser.emitError(parser.getNameLoc(),
                   "unknown IndexTree type: " + keyword);
  return Type();
}

// //TODO(gkestor): do we need this placeholder function?
// void printType(Type type, DialectAsmPrinter &os) const{
//     return;
// }

#define GET_OP_CLASSES
#include "comet/Dialect/IndexTree/IR/ITOps.cpp.inc"

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ITDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "comet/Dialect/IndexTree/IR/ITOps.cpp.inc"
      >();
}