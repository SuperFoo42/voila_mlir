include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialects/FPMath/IR/FPMathDialect.td"

// It's common to define a base classes for attributes in the same dialect. This
// removes the need to pass in the dialect for each attribute, and can also be used
// to define a few fields ahead of time.
class FPMathDialect_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<FPMathDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class APFixedParameter<string desc> : AttrOrTypeParameter<"::llvm::APFixedPoint", desc> {
  let comparator = "$_lhs.compare($_rhs) == 0";
  let printer = [{ $_printer << $_self.toString() }]; //TODO: implement

    let parser = [{ [&] -> FailureOr<::llvm::APFixedPoint> {
      double a;
      if ($_parser.parseFloat(a))
        return failure();
      return failure(); //TODO: implement
    }() }];
}

// Here is a simple definition of an "integer" attribute, with a type and value parameter.
def FixedAttr : FPMathDialect_Attr<"Fixed", "fixed"> {
  let summary = "An Attribute containing a integer value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];
  /// Here we've defined two parameters, one is a "self" type parameter, and the
  /// other is the integer value of the attribute. The self type parameter is
  /// specially handled by the assembly format.
  let parameters = (ins APFixedParameter<"">:$value);

  /// Here we've defined a custom builder for the type, that removes the need to pass
  /// in an MLIRContext instance; as it can be infered from the `type`.

  /// Here we define the textual format of the attribute declaratively, which will
  /// automatically generate parser and printer logic. This will allow for
  /// instances of the attribute to be output as, for example:
  ///
  ///    #my.int<50> : !my.int<32> // a 32-bit integer of value 50.
  ///
  /// Note that the self type parameter is not included in the assembly format.
  /// Its value is derived from the optional trailing type on all attributes.
  let assemblyFormat = "`<` $value `>`";

  /// Indicate that our attribute will add additional verification to the parameters.
  //let genVerifyDecl = 1;
}