#ifndef VOILA_OPS
#define VOILA_OPS

include "VoilaDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
  //let parser = [{ return ::parseConstantOp(parser, result); }];
  //let printer = [{ return ::print(p, *this); }];


  // Invoke a static verify method to verify this constant operation.
  //let verifier = [{ return ::verify(*this); }];
}

def FltConstOp : Voila_Op<"float constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "float constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute.
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseConstantOp(parser, result); }];
  //let printer = [{ return ::print(p, *this); }];

  // Invoke a static verify method to verify this constant operation.
  //let verifier = [{ return ::verify(*this); }];
}

def BoolConstOp : Voila_Op<"bool constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "bool constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute.
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins I1:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseConstantOp(parser, result); }];
  //let printer = [{ return ::print(p, *this); }];

  // Invoke a static verify method to verify this constant operation.
  //let verifier = [{ return ::verify(*this); }];
}

def AddOp : Voila_Op<"add",[NoSideEffect, SameOperandsAndResultType]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$lhs, AnyFloat:$rhs);
  let results = (outs AnyTensor);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def SubOp : Voila_Op<"sub",[NoSideEffect, SameOperandsAndResultType]> {
  let summary = "element-wise subtraction operation";
  let description = [{
    The "sub" operation performs element-wise subtraction between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$lhs, AnyFloat:$rhs);
  let results = (outs AnyTensor);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def MulOp : Voila_Op<"mul",[NoSideEffect, SameOperandsAndResultType]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$lhs, AnyFloat:$rhs);
  let results = (outs AnyTensor);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def DivOp : Voila_Op<"div",[NoSideEffect, SameOperandsAndResultType]> {
  let summary = "element-wise division operation";
  let description = [{
    The "div" operation performs element-wise division between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$lhs, AnyFloat:$rhs);
  let results = (outs AnyTensor);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def ModOp : Voila_Op<"mod",[NoSideEffect, SameOperandsAndResultType]> {
  let summary = "element-wise modulo operation";
  let description = [{
    The "mod" operation performs element-wise modulo between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$lhs, AnyFloat:$rhs);
  let results = (outs AnyTensor);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def AndOp : Voila_Op<"and",[NoSideEffect, SameOperandsAndResultType]> {
  let summary = "element-wise and operation";
  let description = [{
    The "and" operation performs element-wise and between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I1Tensor:$lhs, I1Tensor:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def OrOp : Voila_Op<"or",[NoSideEffect, SameOperandsAndResultType]> {
  let summary = "element-wise or operation";
  let description = [{
    The "or" operation performs element-wise or between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I1Tensor:$lhs, I1Tensor:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def EqOp : Voila_Op<"eq",[NoSideEffect]> {
  let summary = "element-wise equality comparison operation";
  let description = [{
    The "eq" operation performs element-wise equality comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def NeqOp : Voila_Op<"neq",[NoSideEffect]> {
  let summary = "element-wise inequality comparison operation";
  let description = [{
    The "neq" operation performs element-wise inequality comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def LeOp : Voila_Op<"le",[NoSideEffect]> {
  let summary = "element-wise less comparison operation";
  let description = [{
    The "le" operation performs element-wise less comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def LeqOp : Voila_Op<"leq",[NoSideEffect]> {
  let summary = "element-wise less-equal comparison operation";
  let description = [{
    The "leq" operation performs element-wise less-equal comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def GeOp : Voila_Op<"ge",[NoSideEffect]> {
  let summary = "element-wise greater comparison operation";
  let description = [{
    The "ge" operation performs element-wise greater comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def GeqOp : Voila_Op<"geq",[NoSideEffect]> {
  let summary = "element-wise greater-equal comparison operation";
  let description = [{
    The "geq" operation performs element-wise greater-equal comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

#endif // STANDALONE_OPS