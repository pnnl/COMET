//===- ITOps.td - IT dialect operation definitions ----------*- tablegen -*-===//
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
// Defines the operations of the IT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef IT_OPS
#define IT_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td" 

// Provide a definition of the 'TA' dialect in the ODS framework so that we
// can define our operations.
def IT_Dialect : Dialect {
  let name = "it";
  let cppNamespace = ""; 
}

// Base class for ta dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class IT_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IT_Dialect, mnemonic, traits>;
            
//===----------------------------------------------------------------------===//
// Index Tree Operations
//===----------------------------------------------------------------------===//

def IndexTreeComputeLHSOp : IT_Op<"itComputeLHS", [NoSideEffect]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$tensors, ArrayAttr:$allPerms, ArrayAttr:$allFormats);
  let results = (outs AnyType:$output);
}

def IndexTreeComputeRHSOp : IT_Op<"itComputeRHS", [NoSideEffect]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$tensors, ArrayAttr:$allPerms, ArrayAttr:$allFormats);
  let results = (outs AnyType:$output);
}

def IndexTreeComputeOp : IT_Op<"itCompute", [NoSideEffect]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$rhs, 
                   AnyType:$lhs, StrAttr:$op_type, StrAttr:$opt_type, StrAttr:$semiring); 

  let results = (outs I64:$output);

  let verifier = ?;
  
}

def IndexTreeIndicesOp : IT_Op<"itIndices", [NoSideEffect]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$children, ArrayAttr:$indices);
  let results = (outs I64:$output);

  let verifier = ?;  
}

def IndexTreeOp : IT_Op<"itree", [NoSideEffect]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyType:$children);
  let results = (outs I64:$output);

  let verifier = ?;  
}

#endif // IT_OPS