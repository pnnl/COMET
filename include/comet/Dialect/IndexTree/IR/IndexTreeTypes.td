#ifndef INDEXTREE_TYPES
#define INDEXTREE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "comet/Dialect/IndexTree/IR/IndexTreeBase.td"

class IndexTree_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IndexTreeDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IndexTree_TreeType : IndexTree_Type<"IndexTree", "index_tree"> {
  let summary = "Operand for compute expression";
  let description = [{}];
}

def IndexTree_IndexNodeType : IndexTree_Type<"IndexNode", "index"> {
  let summary = "Iteration Tree for Index Variables";
  let description = [{
    Type for storing iteration tree of index variables.
  }];
}

def IndexTree_NodeType : AnyTypeOf<[IndexTree_TreeType, IndexTree_IndexNodeType]>;

def IndexTree_OperandType : IndexTree_Type<"Operand", "operand"> {
  let summary = "Operand for compute expression";
  let description = [{}];
}

//TODO(alokvk2): Eventually get rid of this and just return a (sparse) tensor
def IndexTree_ComputeType : IndexTree_Type<"Compute", "compute"> {
  let summary = "Compute operation of index tree dialect";
  let description =[{}];
}

def IndexTree_ProvenanceGraphType : IndexTree_Type<"ProvenanceGraph", "prov_graph"> {
    let summary = "Provenance graph for index tree transformations";
    let description = [{
        Type for storing provenance graph associated with a index.
    }];
}

#endif // INDEXTREE_TYPES
