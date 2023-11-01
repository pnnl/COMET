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
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

include "comet/Dialect/IndexTree/IR/IndexTreeTypes.td"

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

def IndexTreeComputeLHSOp : IndexTree_Op<"ComputeLHS", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$tensors, ArrayAttr:$allPerms, ArrayAttr:$allFormats);
  let results = (outs AnyType:$output);
}

def IndexTreeComputeRHSOp : IndexTree_Op<"ComputeRHS", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$tensors, ArrayAttr:$allPerms, ArrayAttr:$allFormats);
  let results = (outs AnyType:$output);
}

def IndexTreeComputeOp : IndexTree_Op<"ComputeOp", [Pure]>{
  let summary = "";
  let description = [{
  }];

  //TODO(gkestor): rethink the use of comp_worksp_opt, should we decouple that?
  /// MaskType attribute: {push, pull, auto, none}
  let arguments = (ins Variadic<AnyType>:$rhs, AnyType:$lhs, BoolAttr:$comp_worksp_opt, StrAttr:$semiring, StrAttr:$MaskType);

  let results = (outs IndexTree_ComputeNodeType:$output);

   //TODO(gkestor): add verifier
  //let hasVerifier = 1;
  
}


def IndexTreeIndicesOp : IndexTree_Op<"IndexOp", [Pure]>{
  let summary = "Create an index variable bound to a specific computation.";
  let description = [{
  }];

  let arguments = (
    ins Variadic<IndexTree_NodeType>:$children,
    ArrayAttr:$indices,
    OptionalAttr<DeviceMappingArrayAttr>:$mapping
  );

  let results = (outs IndexTree_IndexNodeType:$output);
}

def IndexTreeDerivedIndicesOp : IndexTree_Op<"DerivedIndexOp", [Pure]>{
  let summary = "Create an index variable derviced from a provenance graph.";
  let description = [{
  }];

  let arguments = (
    ins Variadic<IndexTree_NodeType>:$children,
    IndexTree_ProvenanceGraphType:$prov_graph,
    DefaultValuedAttr<I32Attr, "0">:$graph_index,
    OptionalAttr<DeviceMappingArrayAttr>:$mapping
  );

  let results = (outs IndexTree_IndexNodeType:$output);
}

//TODO(alok): Figure out return type. Eventually want ops to return a (sparse?) tensor
def IndexTreeOp : IndexTree_Op<"itree", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins IndexTree_IndexNodeType:$children);
  let results = (outs I64:$output);
}

class IndexTree_TransformOp<string mnemonic> 
    : IndexTree_Op<mnemonic, [Pure]> {
  // Base class for Index transformations
  let results = (outs IndexTree_ProvenanceGraphType:$output);
}

def IndexTree_TransformCollapseOp : IndexTree_TransformOp<"collapse"> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins IndexTree_IndexNodeType:$first, IndexTree_IndexNodeType:$second);
}

def IndexTree_TransformSplitOp : IndexTree_TransformOp<"split"> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins IndexTree_IndexNodeType:$src);
}

def IndexTree_TransformPosOp : IndexTree_TransformOp<"pos"> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins IndexTree_IndexNodeType:$src);
}

def IndexTree_TransformCoordOp : IndexTree_TransformOp<"coord"> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins IndexTree_IndexNodeType:$src);
}

#endif // INDEXTREE_OPS