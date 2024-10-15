#ifndef TA_TYPES
#define TA_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "comet/Dialect/TensorAlgebra/IR/TABase.td"

class TensorAlgebra_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TA_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IndexLabel : TensorAlgebra_Type<"IndexLabel","index"> {
    let summary = "Type representing the index of one dimnsion of a tensor";
    let description = [{}];
}

def Range : TensorAlgebra_Type<"Range","range"> {
    let summary = "Type representing the range of one dimnsion of a tensor";
    let description = [{}];
}

def SparseTensor : TensorAlgebra_Type<"SparseTensor", "sparse_tensor", [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
    let summary = "Sparse tensor to use in tensor algebra dialect";
    let description = [{}];

    let parameters = (ins
        "::mlir::Type":$element_type, 
        ArrayRefParameter<"int64_t", "Dimensions of tensor">:$dims,
        ArrayRefParameter<"int32_t", "Format">:$format
    );
    let assemblyFormat = "`<` $element_type `,` $dims `,` $format `>`";

    // TODO: Implement custom builder from "common" format strings into format strings
}

def WorkspaceTensor : TensorAlgebra_Type<"Workspace", "workspace", [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
    let summary = "Temporary tensor generated from a workspace transfrom";
    let description = [{
        Dense, temporary tensor generated from a workspace transformation.
        Needed to represent dense row as well as mark array.
    }];
    let parameters = (
        ins "::mlir::Type":$element_type,
        ArrayRefParameter<"int64_t", "Dimensions of workspace">:$dims
    );
    let assemblyFormat = "`<` $element_type `,` $dims `>`";
}

/// Provide a definition of the types that are used within the TA dialect.
def TA_AnyTensor : AnyTypeOf<[TensorOf<[AnyType]>, SparseTensor, WorkspaceTensor]>;

#endif //TA_TYPES