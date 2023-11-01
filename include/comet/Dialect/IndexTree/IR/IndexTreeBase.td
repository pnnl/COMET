#ifndef INDEXTREE_BASE
#define INDEXTREE_BASE

include "mlir/IR/OpBase.td"

// Provide a definition of the 'it' dialect in the ODS framework so that we
// can define our operations.
def IndexTreeDialect : Dialect {
  let name = "it";
  let cppNamespace = "::mlir::indexTree"; 

  // We set this bit to generate the declarations for the dialect's type parsing
  // and printing hooks.
  let useDefaultTypePrinterParser = 1;
  
}

#endif // INDEXTREE_BASE