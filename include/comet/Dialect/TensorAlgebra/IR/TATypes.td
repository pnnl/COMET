#ifndef TA_TYPES
#define TA_TYPES

include "comet/Dialect/TensorAlgebra/IR/TABase.td"

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

/// Provide a definition of the types that are used within the TA dialect.
def TA_AnyTensor : AnyTypeOf<[TensorOf<[AnyType]>, SparseTensor]>;

#endif //TA_TYPES