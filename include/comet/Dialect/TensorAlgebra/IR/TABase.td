#ifndef TA_BASE
#define TA_BASE

/// Provide a definition of the 'TA' dialect in the ODS framework so that we
/// can define our operations.
def TA_Dialect : Dialect {
  let name = "ta";
  let cppNamespace = "::mlir::tensorAlgebra"; 

  /// We set this bit to generate the declarations for the dialect's type parsing
  /// and printing hooks.
  let useDefaultTypePrinterParser = 1;
}

#endif //TA_BASE