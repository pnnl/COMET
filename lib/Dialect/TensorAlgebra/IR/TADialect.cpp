//===- Dialect.cpp - TA IR Dialect registration in MLIR ------------------===//
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
// This file implements the dialect for the TA IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TATypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/Parser.h"

using namespace mlir;
using namespace mlir::tensorAlgebra;

//===----------------------------------------------------------------------===//
// TADialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
// TADialect::TADialect(mlir::MLIRContext *ctx) : mlir::Dialect("ta", ctx) { //Ruiqin update into the following
TADialect::TADialect(mlir::MLIRContext *ctx) : mlir::Dialect("ta", ctx, mlir::TypeID::get<TADialect>())
{
  addTypes<RangeType, SparseTensorType>();
  addOperations<
#define GET_OP_LIST
#include "comet/Dialect/TensorAlgebra/IR/TAOps.cpp.inc"
      >();
}

Type mlir::tensorAlgebra::TADialect::parseType(DialectAsmParser &parser) const
{
  // Parse the main keyword for the type.
  StringRef keyword;
  // for "range" and "sptensor" type
  if (parser.parseKeyword(&keyword))
    return Type();

  MLIRContext *context = getContext();

  // Handle 'range' types.
  if (keyword == "range")
  {
    return RangeType::get(context);
  }

  // Parse the element types of the sptensor.
  if (keyword == "sptensor")
  {
    if (parser.parseLess())
    {
      return Type();
    }

    SmallVector<mlir::Type, 1> elementTypes;
    do
    {
      // Parse the current element type.
      llvm::SMLoc typeLoc = parser.getCurrentLocation();
      mlir::Type elementType;

      if (parser.parseType(elementType))
        return nullptr;

      // Check that the type is either a TensorType or another StructType.
      if (!elementType.isa<mlir::TensorType>() && !elementType.isa<SparseTensorType>())
      {
        parser.emitError(typeLoc, "element type for a struct must either "
                                  "be a TensorType or a StructType, got: ")
            << elementType;
        return Type();
      }
      elementTypes.push_back(elementType);

      // Parse the optional: `,`
    } while (succeeded(parser.parseOptionalComma()));

    // Parse: `>`
    if (parser.parseGreater())
      return Type();

    return SparseTensorType::get(elementTypes);
  }

  parser.emitError(parser.getNameLoc(),
                   "unknown TensorAlgebra type: " + keyword);
  return Type();
}

/// RangeType prints as just "range".
static void print(RangeType type, DialectAsmPrinter &printer)
{
  printer << "range";
}

void mlir::tensorAlgebra::TADialect::printType(
    Type type, DialectAsmPrinter &printer) const
{

  if (type.isa<RangeType>())
  {
    print(type.cast<RangeType>(), printer);
  }
  else if (type.isa<SparseTensorType>())
  {
    // Currently the only toy type is a struct type.
    SparseTensorType sparseTensorType = type.cast<SparseTensorType>();

    // Print the struct type according to the parser format.
    printer << "sptensor<";
    llvm::interleaveComma(sparseTensorType.getElementTypes(), printer);
    printer << '>';
  }
  else
  {
    llvm_unreachable("Unhandled TensorAlgebra type");
  }
}

//===----------------------------------------------------------------------===//
// ConstantOp

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void DenseConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                            double value)
{
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  DenseConstantOp::build(builder, state, dataType, dataAttribute);
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
static mlir::LogicalResult verify(DenseConstantOp op)
{
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank())
  {
    return op.emitOpError(
               "return type must match the one of the attached value "
               "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim)
  {
    if (attrType.getShape()[dim] != resultType.getShape()[dim])
    {
      return op.emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

static mlir::LogicalResult verify(SparseTensorConstantOp op)
{
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank())
  {
    return op.emitOpError(
               "return type must match the one of the attached value "
               "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim)
  {
    if (attrType.getShape()[dim] != resultType.getShape()[dim])
    {
      return op.emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

static mlir::LogicalResult verify(SparseTensorVarOp op)
{
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank())
  {
    return op.emitOpError(
               "return type must match the one of the attached value "
               "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim)
  {
    if (attrType.getShape()[dim] != resultType.getShape()[dim])
    {
      return op.emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// SubstractOp
void SubstractOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, mlir::Value arguments)
{
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee()
{
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() {
  return nullptr;
}

//===----------------------------------------------------------------------===//
// MulOp
void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::TensorType resultType,
                  mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(resultType);
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// DivOp
void DivOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

void DivOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::TensorType resultType,
                  mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(resultType);
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// ReturnOp
static mlir::LogicalResult verify(TAReturnOp op)
{
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  // auto function = cast<FuncOp>(op.getParentOp()); //Ruiqin change into following
  auto function = cast<FuncOp>(op.getOperation()->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError()
           << "does not return the same number of values ("
           << op.getNumOperands() << ") as the enclosing function ("
           << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!op.hasOperand())
    return mlir::success();

  auto inputType = *op.operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return op.emitError() << "type of return operand ("
                        << *op.operand_type_begin()
                        << ") doesn't match function result type ("
                        << results.front() << ")";
}

//===----------------------------------------------------------------------===//
// TA Types
//===----------------------------------------------------------------------===//

namespace mlir
{
  namespace tensorAlgebra
  {
    namespace detail
    {
      /// This class represents the internal storage of the Toy `SparseTensorType`.
      struct SparseTensorTypeStorage : public mlir::TypeStorage
      {
        /// The `KeyTy` is a required type that provides an interface for the storage
        /// instance. This type will be used when uniquing an instance of the type
        /// storage. For our struct type, we will unique each instance structurally on
        /// the elements that it contains.
        using KeyTy = llvm::ArrayRef<mlir::Type>;

        /// A constructor for the type storage instance.
        SparseTensorTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
            : elementTypes(elementTypes) {}

        /// Define the comparison function for the key type with the current storage
        /// instance. This is used when constructing a new instance to ensure that we
        /// haven't already uniqued an instance of the given key.
        bool operator==(const KeyTy &key) const { return key == elementTypes; }

        /// Define a hash function for the key type. This is used when uniquing
        /// instances of the storage, see the `StructType::get` method.
        /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
        /// have hash functions available, so we could just omit this entirely.
        static llvm::hash_code hashKey(const KeyTy &key)
        {
          return llvm::hash_value(key);
        }

        /// Define a construction function for the key type from a set of parameters.
        /// These parameters will be provided when constructing the storage instance
        /// itself.
        /// Note: This method isn't necessary because KeyTy can be directly
        /// constructed with the given parameters.
        static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes)
        {
          return KeyTy(elementTypes);
        }

        /// Define a construction method for creating a new instance of this storage.
        /// This method takes an instance of a storage allocator, and an instance of a
        /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
        /// allocations used to create the type storage and its internal.
        static SparseTensorTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                                  const KeyTy &key)
        {
          // Copy the elements from the provided `KeyTy` into the allocator.
          llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

          // Allocate the storage instance and construct it.
          return new (allocator.allocate<SparseTensorTypeStorage>())
              SparseTensorTypeStorage(elementTypes);
        }

        /// The following field contains the element types of the struct.
        llvm::ArrayRef<mlir::Type> elementTypes;
      };

    } // end namespace detail
  }   // end namespace tensoralgebra
} // end namespace mlir

/// Create an instance of a `SparseTensorType` with the given element types. There
/// *must* be at least one element type.
SparseTensorType SparseTensorType::get(llvm::ArrayRef<mlir::Type> elementTypes)
{
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first two parameters are the context to unique in and the
  // kind of the type. The parameters after the type kind are forwarded to the
  // storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  // return Base::get(ctx, TATypes::SparseTensor, elementTypes); //Ruiqin comment
  return Base::get(ctx, elementTypes); // Ruiqin add
}

/// Returns the element types of this sparse tensor type.
llvm::ArrayRef<mlir::Type> SparseTensorType::getElementTypes()
{
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}

#define GET_OP_CLASSES
#include "comet/Dialect/TensorAlgebra/IR/TAOps.cpp.inc"
