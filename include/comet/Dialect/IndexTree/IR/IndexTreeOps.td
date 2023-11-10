//===- IndexTreeOps.td - IT dialect operation definitions ----------*- tablegen -*-===//
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
// Defines the operations of the IndexTree dialect.
//
//===----------------------------------------------------------------------===//

#ifndef INDEXTREE_OPS
#define INDEXTREE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

include "comet/Dialect/IndexTree/IR/IndexTreeTypes.td"
include "comet/Dialect/TensorAlgebra/IR/TATypes.td"

// Base class for ta dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class IndexTree_Op<string mnemonic, list<Trait> traits = []> :
    Op<IndexTreeDialect, mnemonic, traits>;
            
//===----------------------------------------------------------------------===//
// Index Tree Operations
//===----------------------------------------------------------------------===//

def IndexTreeOp : IndexTree_Op<"itree", 
    [SingleBlockImplicitTerminator<"indexTree::YieldOp">,
    IsolatedFromAbove,
    RecursiveMemoryEffects]> {
  let summary = "Create a scope for index tree iteration";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
}

def YieldOp : IndexTree_Op<"yield", [Pure, ReturnLike, Terminator,
    HasParent<"IndexTreeOp">]> {
  let summary = "index tree yield and termination operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

def IndexTreeRootOp : IndexTree_Op<"RootOp", [
    HasParent<"IndexTreeOp">, Pure]> {
  let summary = "Create the base of the iteration tree";
  let description = [{}];

  let results = (outs IndexTree_TreeType:$output);
}

def IndexTreeIndicesOp : IndexTree_Op<"IndexOp", [Pure]>{
  let summary = "Create an index variable bound to a specific computation.";
  let description = [{
  }];

  let arguments = (ins IndexTree_NodeType:$parent, Optional<IndexTree_DomainType>:$domain);

  let results = (outs IndexTree_IndexNodeType:$output);
}

def IndexTreeMaskOp : IndexTree_Op<"MaskOp", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins TA_AnyTensor:$tensor, TA_AnyTensor:$mask, StrAttr:$mask_type);
  let results = (outs TA_AnyTensor:$masked_tensor);
}

def IndexTreeOperandOp : IndexTree_Op<"OperandOp", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins TA_AnyTensor:$tensor, Variadic<IndexTree_IndexNodeType>:$indices);
  let results = (outs IndexTree_OperandType:$result);
}

def IndexTreeComputeOp : IndexTree_Op<"ComputeOp", [Pure]>{
  let summary = "";
  let description = [{
  }];

  //TODO(gkestor): rethink the use of comp_worksp_opt, should we decouple that?
  /// MaskType attribute: {push, pull, auto, none}
  let arguments = (ins IndexTree_IndexNodeType:$parent, IndexTree_OperandType:$lhs, Variadic<IndexTree_OperandType>:$rhs, BoolAttr:$comp_worksp_opt, StrAttr:$semiring);

  let results = (outs TA_AnyTensor);

   //TODO(gkestor): add verifier
  //let hasVerifier = 1;
  
}

def IndexTreeTensorDomainOp : IndexTree_Op<"DomainOp", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$tensor, UI32Attr:$dim, OptionalAttr<StrAttr>:$type);

  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeMemRefDomainOp : IndexTree_Op<"MemRefDomainOp", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins 
      StrAttr:$type, 
      AnyMemRef:$pos,
      AnyMemRef:$crd,
      AnyMemRef:$value,
      Optional<AnyTypeOf<[Index, IndexTree_NodeType]>>:$parent);

  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeDenseDomainOp : IndexTree_Op<"DenseDomainOp", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins UI32:$min, UI32:$max, Optional<UI32>:$step);

  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeDomainUnionOp : IndexTree_Op<"DomainUnionOp", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins Variadic<IndexTree_DomainType>:$domains);

  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeDomainIntersectionOp : IndexTree_Op<"DomainIntersectionOp", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins Variadic<IndexTree_DomainType>:$domains);

  let results = (outs IndexTree_DomainType:$domain);
}


#endif // INDEXTREE_OPS