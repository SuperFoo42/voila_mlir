#ifndef FPMATH_TYPES
#define FPMATH_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialects/FPMath/IR/FPMathDialect.td"

class FPMath_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FPMathDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def DecimalType : FPMath_Type<"Decimal", "fixed"> {
  let summary = "Fixed point type for fixed precision decimal calculations ";
  let description = [{
    TODO
  }];

  let parameters = (ins "int32_t":$width, "int32_t":$scale);

  let assemblyFormat = "`<` $width `,` $scale `>`";

  //let genVerifyDecl = 1; //TODO: check unsignedness of with and scale
}

def DecimalLike : TypeOrContainer<DecimalType, "decimal-like">;
#endif
