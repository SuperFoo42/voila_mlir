#ifndef FPMATH_TYPES
#define FPMATH_TYPES

include "mlir/IR/AttrTypeBase.td"
//include "FPMathDialect.td"

class FPMath_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FPMath, name, traits> {
  let mnemonic = typeMnemonic;
}

def Decimal : FPMath_Type<"Decimal", "fixed"> {
  let summary = "Fixed point type for fixed precision decimal calculations ";
  let description = [{
    TODO
  }];

  let parameters = (ins IntegerAttr:$width, IntegerAttr:$scale);

  let assemblyFormat = "`<` $width `, ` $scale `>`";

  let genVerifyDecl = 1;
}
#endif
