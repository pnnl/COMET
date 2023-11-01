#ifndef INDEXTREE_TYPES
#define INDEXTREE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "comet/Dialect/IndexTree/IR/IndexTreeBase.td"

class IndexTree_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IndexTreeDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IndexTree_ComputeNodeType : IndexTree_Type<"Compute", "compute"> {
  let summary = "Compute operation of index tree dialect";
  let description =[{}];
}

def IndexTree_IndexNodeType : IndexTree_Type<"IndexNode", "index"> {
  let summary = "Interation Tree for Index Variables";
  let description = [{
    Type for storing iteration tree of index variables.
  }];
}

def IndexTree_NodeType = AnyTypeOf<[IndexTree_ComputeNodeType, IndexTree_IndexNodeType]>;

def IndexTree_ProvenanceGraphType : IndexTree_Type<"ProvenanceGraph", "prov_graph"> {
    let summary = "Provenance graph for index tree transformations";
    let description = [{
        Type for storing provenance graph associated with a index.
    }];
}

#endif // INDEXTREE_TYPES
