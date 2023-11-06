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

/// Base class for ta dialect operations. This operation inherits from the base
/// `Op` class in OpBase.td, and provides:
///   * The parent dialect of the operation.
///   * The mnemonic for the operation, or the name without the dialect prefix.
///   * A list of traits for the operation.
class TA_Op<string mnemonic, list<Trait> traits = []> :
    Op<TA_Dialect, mnemonic, traits>;

/// Provide a definition for the TA SparseTensorType for use in ODS. 
/// This allows for using SparseTensorType in a similar way to Tensor or MemRef.
def SparseTensor :
    Type<CPred<"$_self.isa<SparseTensorType>()">, "TA sparse tensor type">;

/// Provide a definition of the types that are used within the TA dialect.
def TA_AnyTensor : AnyTypeOf<[TensorOf<[AnyType]>, SparseTensor]>;

#endif //TA_TYPES