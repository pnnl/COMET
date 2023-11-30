#ifndef TA_ATTRS
#define TA_ATTRS

include "comet/Dialect/TensorAlgebra/IR/TAEnums.td"

def TAFormatArrayAttr : TypedArrayAttrBase<TAFormatEnum, "Format array"> {
  let constBuilderCall = "$_builder.getI32ArrayAttr($0)";
}

#endif //TA_ATTRS