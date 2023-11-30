#ifndef TA_ENUMS
#define TA_ENUMS

include "mlir/IR/EnumAttr.td"

def Unknown: I32EnumAttrCase<"UNK", 0, "unk">; 
def Dense: I32EnumAttrCase<"D", 1, "d">;
def Compressed: I32EnumAttrCase<"CU", 2, "cu">;
def CompressedNonunique: I32EnumAttrCase<"CN", 3, "cn">;
def Singular: I32EnumAttrCase<"S", 4, "s">; 

def TAFormatEnum: I32EnumAttr<"TensorFormatEnum", "Valid format specifiers",
                           [Unknown, Dense, Compressed, CompressedNonunique, Singular]> {
  let cppNamespace = "::mlir::tensorAlgebra";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

#endif //TA_ENUMS