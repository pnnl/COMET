//===- TAOps.td - TA dialect operation definitions ----------*- tablegen -*-===//
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
// Defines the operations of the TA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TA_OPS
#define TA_OPS

#ifndef MLIR_CALLINTERFACES
include "mlir/Interfaces/CallInterfaces.td"
#endif // MLIR_CALLINTERFACES

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td" 

// Provide a definition of the 'TA' dialect in the ODS framework so that we
// can define our operations.
def TA_Dialect : Dialect {
  let name = "ta";
  let cppNamespace = ""; 
}


// An implementation of RangeType.
def TA_RangeType :
    DialectType<TA_Dialect,
                CPred<"$_self.isa<RangeType>()">,
                "RangeType">;


// Whether a type is a RangeType.
def TAIsRangeTypePred : CPred<"$_self.isa<RangeType>()">;
def Range : Type<TAIsRangeTypePred, "range">;

// Base class for ta dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class TA_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TA_Dialect, mnemonic, traits>;

// Provide a definition for the TA SparseTensorType for use in ODS. 
// This allows for using SparseTensorType in a similar way to Tensor or MemRef.
def SparseTensor :
    Type<CPred<"$_self.isa<SparseTensorType>()">, "TA sparse tensor type">;

// Provide a definition of the types that are used within the TA dialect.
def TA_AnyTensor : AnyTypeOf<[TensorOf<[AnyType]>, SparseTensor]>;


//===----------------------------------------------------------------------===//
// Tensor Algebra Operations
//===----------------------------------------------------------------------===//
def IndexLabelStaticOp :
    TA_Op<"static_index_label", [NoSideEffect]>,
    Arguments<(ins Index:$min, Index:$max, Index:$step)>,
    Results<(outs Range)> {
  let summary = "Create an index label type value, used to create views";
  let description = [{
    The `ta.static_index_label` op creates a ta.static_index_label from 3 values of type `index`
    that represent the min, max and step values of a range.

    Example:

      %3 = ta.static_index_label %0:%1:%2 : !ta.index_label
  }];

  let builders = [OpBuilder<(ins "mlir::Value":$min, "mlir::Value":$max, "mlir::Value":$step), 
    [{
      auto rangeType = RangeType::get($_builder.getContext());
      build($_builder, $_state, rangeType, min, max, step);
    }]>];
  

  // Fully specified by traits.
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// Tensor Algebra Operations
//===----------------------------------------------------------------------===//
def IndexLabelDynamicOp :
    TA_Op<"dynamic_index_label", [NoSideEffect]>,
    Arguments<(ins Index:$min, Index:$step)>,
    Results<(outs Range)> {
  let summary = "Create an index label dynamic type value, used to create views";
  let description = [{
    The `ta.dynamic_index_label` op creates a ta.dynamic_index_label from 2 values of type `index` that represent the min and step values of a range.

    Example:

      %3 = ta.dynamic_index_label %0:%1 : !ta.dynamic_index_label
  }];

  let builders = [OpBuilder<(ins "mlir::Value":$min, "mlir::Value":$step), 
    [{
      auto rangeType = RangeType::get($_builder.getContext());
      build($_builder, $_state, rangeType, min, step);
    }]>];

  // Fully specified by traits.
  let verifier = ?;
}

def DenseTensorDeclOp : TA_Op<"dense_tensor_decl", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "Create an output tensor declaration";
  let description = [{
    Tensor declaration operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.dense_tensor_decl"()
         { labels = vector[] : tensor<*x*xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins Variadic<Range>:$labels, StrAttr:$format);

  // The constant operation returns a single value of TensorType.
  let results = (outs AnyTensor);

  // Invoke a static verify method to verify this constant operation.
  let verifier = ?;
}

def SparseTensorDeclOp : TA_Op<"sparse_tensor_decl", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "Create a tensor declaration";
  let description = [{
    Tensor declaration operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.sparse_tensor_decl"()
         { labels = vector[] : tensor<*x*xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins Variadic<Range>:$labels, StrAttr:$format, BoolAttr:$temporal_tensor);

  // The constant operation returns a single value of TensorType.
  let results = (outs AnyTensor);

  // Invoke a static verify method to verify this constant operation.
  let verifier = ?;
}

def SparseOutputTensorDeclOp : TA_Op<"sparse_output_tensor_decl", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "Create a sparse output tensor declaration";
  let description = [{
    Tensor declaration operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.sparse_output_tensor_decl"()
         { labels = vector[] : tensor<*x*xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins Variadic<Range>:$labels, StrAttr:$format);

  // The constant operation returns a single value of TensorType.
  let results = (outs AnyTensor);

  // Invoke a static verify method to verify this constant operation.
  let verifier = ?;
}

//TempSparseOutputTensorDeclOp operation is similar to SparseOutputTensorDeclOp.
//TempSparseOutputTensorDeclOp should be lowered before SparseOutputTensorDeclOp in the compilation pipeline to produce  
//necessary information to lower SparseOutputTensorDeclOp properly
def TempSparseOutputTensorDeclOp : TA_Op<"temp_sparse_output_tensor_decl", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "Create a sparse output tensor declaration for temporaries generated by parsing compound operations";
  let description = [{
    Tensor declaration operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.temp_sparse_output_tensor_decl"()
         { labels = vector[] : tensor<*x*xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins Variadic<Range>:$labels, StrAttr:$format);

  // The constant operation returns a single value of TensorType.
  let results = (outs AnyTensor);

  // Invoke a static verify method to verify this constant operation.
  let verifier = ?;
}

//TODO(gkestor): need to evaluate its use cases
def LabeledTensorOp : TA_Op<"labeled_tensor", [NoSideEffect]>{

  let summary = "Produce a sliced `subview` of a base `tensor`.";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$tensor, Variadic<Range>:$labels);
  let results = (outs TA_AnyTensor);

  let verifier = ?;
}

def TensorChainSetOp : TA_Op<"chain_set_op", [NoSideEffect]>{

  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);
  
  //let hasCanonicalizer = 1; 

  let verifier = ?;
}

def TensorSetOp : TA_Op<"set_op", [NoSideEffect]>{

  let summary = "";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$lhs, TA_AnyTensor:$rhs);
  
  //let hasCanonicalizer = 1; 

  let verifier = ?;
}

def TensorMultOp : TA_Op<"tc", [NoSideEffect]>{

  let summary = "";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$rhs1, TA_AnyTensor:$rhs2, Variadic<Range>:$index_labels, AffineMapArrayAttr:$indexing_maps, StrArrayAttr:$formats, StrAttr:$semiring);

  // Return value
  let results = (outs TA_AnyTensor);

  let verifier = ?;
  
}

def TensorElewsMultOp : TA_Op<"elews_mul", [NoSideEffect]>{

  let summary = "";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$rhs1, 
                   TA_AnyTensor:$rhs2, 
                   Variadic<Range>:$index_labels, 
                   AffineMapArrayAttr:$indexing_maps, 
                   StrArrayAttr:$formats, 
                   StrAttr:$semiring); 
  
  let results = (outs TA_AnyTensor);

  let verifier = ?;
  
}

def TensorFillOp : TA_Op<"fill", [NoSideEffect]>{
  
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$lhs, AnyAttr:$value);
  let verifier = ?;
  
}


// We define a ta operation by inherting from our base 'TA_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'NoSideEffect' as it is a pure operation
// and may be removed if dead.
def DenseConstantOp : TA_Op<"constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), 
    [{
      build($_builder, $_state, value.getType(), value);
    }]>, 
    OpBuilder<(ins "double":$value)>];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

def SparseTensorConstructOp : TA_Op<"sptensor_construct", [NoSideEffect]>{

  let summary = "";
  let description = [{
  }];

  //The size of arguments depends on the rank of sparse tensors
  //For example, for 2D sparse matrix, sptensor_construct consists of 12 elements:
  //A1pos, A1crd (first dimension) - each dimension consists of pos and crd arrays 
  //A2pos, A2crd (second dimension)
  //Aval (value array, nonzero elements)
  //A1pos_size, A1crd_size, (size of each pos and crd arrays)
  //A2pos_size, A2crd_size, 
  //Aval_size (size of value array) 
  //dim1_size, dim2_size (size of each dimension in sparse tensor) 
  //TODO(gkestor): might be better to have a struct with all the data elements
  let arguments = (ins Variadic<AnyType>:$indices);
  let results = (outs TA_AnyTensor:$output);

  let assemblyFormat = [{
    `(` $indices `)` attr-dict `:` `(` type($indices) `)` `->` `(` type($output) `)`
  }];

  let verifier = ?;
  
}

def TransposeOp : TA_Op<"transpose",[NoSideEffect]> {
  let summary = "transpose operation";

  let arguments = (ins TA_AnyTensor:$rhs, Variadic<Range>:$index_labels, AffineMapArrayAttr:$indexing_maps, StrArrayAttr:$formats);
  
  let results = (outs TA_AnyTensor); 

  let verifier = ?;
}

def SUMOp : TA_Op<"sum",[NoSideEffect]> {
  let summary = "reduction operation";

  let arguments = (ins TA_AnyTensor:$rhs); 

  let results = (outs F64:$lhs);

  let builders = [OpBuilder<
    (ins "mlir::Value":$input)>];

  let verifier = ?;
}

// The interface CallOpInterface 
def GenericCallOp : TA_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = "ta.generic_call"(%1, %3) {callee = @my_func}
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins SymbolRefAttr:$callee, TA_AnyTensor:$rhs);
  

  // The generic call operation returns a single value of TensorType.
  let results = (outs TA_AnyTensor:$lhs);

  let builders = [OpBuilder<(ins "StringRef":$callee,
    "::mlir::Value":$arguments)>
  ];
}

def PrintOp : TA_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We can extend the list of supported datatype for print with F64Tensor, I8MemRef, I64MemRef, F32MemRef, etc.
  let arguments = (ins AnyTypeOf<[F64,
                                  F64MemRef, 
                                  TA_AnyTensor]>:$input);
}

def GetTimeOp : TA_Op<"getTime"> {
  let summary = "getTime operation";
  let description = [{
    The "getTime" builtin operation for getting the clock time
  }];

  let results = (outs F64);
}

def PrintElapsedTimeOp : TA_Op<"print_elapsed_time"> {
  let summary = "printElapsedTime operation";
  let description = [{
    The "print_elapsed_time" builtin operation prints the time elapsed for the start
    and end times
  }];

  // The print elapsed time gets two clock times as an input to compute and print
  // elapsed time
  let arguments = (ins F64:$start, F64:$end);
}

def TAReturnOp : TA_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        ta.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  let builders = [OpBuilder<
    (ins ), 
    [{
      build($_builder, $_state, llvm::None);
    }]>];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return ::verify(*this); }];
}

///////////

def TensorFillFromFileOp : TA_Op<"fill_from_file", [NoSideEffect]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$lhs, AnyAttr:$filename);
  let verifier = ?;
  
}

def AddOp : TA_Op<"add",
    [NoSideEffect]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def SubstractOp : TA_Op<"substract",
    [NoSideEffect]> {
  let summary = "element-wise substract operation";
  let description = [{
    The "substract" operation performs element-wise substract between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def DivOp : TA_Op<"div",
    [NoSideEffect]> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, Variadic<Range>:$sum_labels);

  let results = (outs AnyTensor);

  let verifier = ?;

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>,
    OpBuilder<(ins "TensorType":$resultType, "mlir::Value":$lhs, "mlir::Value":$rhs)>,
  ];
  
}

def CastOp : TA_Op<"cast",
    [NoSideEffect,
     SameOperandsAndResultShape]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked
    then the rank should be the same and static dimensions should match. The
    operation is invalid if converting to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  // Set the folder bit so that we can fold redundant cast operations.
  let hasFolder = 1;
}

def MulOp : TA_Op<"mul",
    [NoSideEffect]> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, Variadic<Range>:$sum_labels);

  let results = (outs AnyTensor);

  let verifier = ?;
  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>,
    OpBuilder<(ins "TensorType":$resultType, "mlir::Value":$lhs, "mlir::Value":$rhs)>,
  ]; 
}

def TensorCopyOp : TA_Op<"copy", [NoSideEffect]>{
  
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, AffineMapAttr:$inputPerm, AffineMapAttr:$outputPerm);

  let verifier = ?;
}

// The interface CallOpInterface is defined in mlir/Analysis/CallInterfaces.td
def ScalarOp : TA_Op<"scalar",[NoSideEffect]> {
  let summary = "scalar arithmetic operation";

  let arguments = (ins F64:$rhs, F64:$lhs, AnyAttr:$op); 

  // The generic call operation returns a single value of TensorType.
  let results = (outs F64:$res);

  let verifier = ?;
}

def ScalarAssignOp : TA_Op<"scalarAssign",[NoSideEffect]> {
  let summary = "scalarAssign operation";

  let arguments = (ins AnyAttr:$rhs);

  let results = (outs F64:$lhs);

  let verifier = ?;
}


#endif // TA_OPS




