#ifndef FPMATH_BASE
#define FPMATH_BASE

include "mlir/IR/OpBase.td"

// Here is a simple definition of a dialect.
def FPMathDialect : Dialect {
  let summary = "Dialect for fixed-point types and math";
  let description = [{
    TBD
  }];

  /// This is the namespace of the dialect. It is used to encapsulate the sub-components
  /// of the dialect, such as operations ("my_dialect.foo").
  let name = "fpmath";

  /// The C++ namespace that the dialect, and its sub-components, get placed in.
  let cppNamespace = "::mlir::fpmath";
}

class FPMath_Op<string mnemonic, list<Trait> traits = []> :
Op<FPMathDialect, mnemonic, traits>;

#endif
