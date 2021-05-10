include "mlir/Interfaces/SideEffectInterfaces.td"
include "VoilaDialect.td"

class Voila_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Voila_Dialect, mnemonic, traits>;

def IntConstOp : Voila_Op<"integer constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "integer constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute.
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins I64:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs I64Tensor);

  // Specify a parser and printer method.
  let parser = [{ return ::parseConstantOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];


  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}