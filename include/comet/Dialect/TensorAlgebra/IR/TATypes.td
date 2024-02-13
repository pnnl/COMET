#ifndef TA_TYPES
#define TA_TYPES

include "comet/Dialect/TensorAlgebra/IR/TABase.td"

class TensorAlgebra_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TA_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

/// An implementation of RangeType.
def TA_RangeType :
    DialectType<TA_Dialect,
                CPred<"$_self.isa<RangeType>()">,
                "RangeType">;


/// Whether a type is a RangeType.
def TAIsRangeTypePred : CPred<"$_self.isa<RangeType>()">;
def Range : Type<TAIsRangeTypePred, "range">;

/// Provide a definition for the TA SparseTensorType for use in ODS. 
/// This allows for using SparseTensorType in a similar way to Tensor or MemRef.
def SparseTensor :
    Type<CPred<"$_self.isa<SparseTensorType>()">, "TA sparse tensor type">;

def WorkspaceTensor : TensorAlgebra_Type<"Workspace", "workspace", [ShapedTypeInterface]> {
    let summary = "Temporary tensor generated from a workspace transfrom"
    description = [{
        Dense, temporary tensor generated from a workspace transformation.
        Needed to represent dense row as well as mark array.
    }];
    let parameters = (ins ArrayRefParameter<"int", "Dimensions of workspace">:$dims);
}

/// Provide a definition of the types that are used within the TA dialect.
def TA_AnyTensor : AnyTypeOf<[TensorOf<[AnyType]>, SparseTensor, WorkspaceTensor]>;

#endif //TA_TYPES