#ifndef VOILA_OPS
#define VOILA_OPS

include "VoilaDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def IntConstOp : Voila_Op<"integer constant", [NoSideEffect,ConstantLike]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "integer constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute.
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins I64Attr:$value);

  // The constant operation returns a single value of TensorType.
  //TODO: 1DTensorOf<[I64]>
  let results = (outs I64Tensor);

let builders = [
OpBuilder<(ins "mlir::IntegerAttr":$value), [{
build($_builder, $_state, value.getType(), value);
}]>,
OpBuilder<(ins "long":$value), [{
    IntConstOp::build($_builder, $_state, RankedTensorType::get({}, $_builder.getI64Type()), $_builder.getI64IntegerAttr(value));
}]>
];

  // Invoke a static verify method to verify this constant operation.
  //let verifier = [{ return ::verify(*this); }];
}

def FltConstOp : Voila_Op<"float constant", [NoSideEffect,ConstantLike]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "float constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute.
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64Attr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseConstantOp(parser, result); }];
  //let printer = [{ return ::print(p, *this); }];

let builders = [
OpBuilder<(ins "mlir::FloatAttr":$value), [{
build($_builder, $_state, value.getType(), value);
}]>,
OpBuilder<(ins "double":$value), [{
    FltConstOp::build($_builder, $_state, RankedTensorType::get({}, $_builder.getF64Type()), $_builder.getF64FloatAttr(value));
}]>
];

  // Invoke a static verify method to verify this constant operation.
  //let verifier = [{ return ::verify(*this); }];
}

def BoolConstOp : Voila_Op<"bool constant", [NoSideEffect,ConstantLike]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "bool constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute.
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins I1Attr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs I1Tensor);

let builders = [
OpBuilder<(ins "mlir::IntegerAttr":$value), [{
build($_builder, $_state, value.getType(), value);
}]>,
OpBuilder<(ins "bool":$value), [{
BoolConstOp::build($_builder, $_state, RankedTensorType::get({}, $_builder.getI1Type()), $_builder.getIntegerAttr($_builder.getI1Type(), value));
}]>
];
  // Specify a parser and printer method.
  //let parser = [{ return ::parseConstantOp(parser, result); }];
  //let printer = [{ return ::print(p, *this); }];

  // Invoke a static verify method to verify this constant operation.
  //let verifier = [{ return ::verify(*this); }];
}

//TODO: SameOperandsAndResultShape,Elementwise,Vectorizable
def AddOp : Voila_Op<"add",[NoSideEffect]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$rhs);
let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>);
  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def SubOp : Voila_Op<"sub",[NoSideEffect]> {
  let summary = "element-wise subtraction operation";
  let description = [{
    The "sub" operation performs element-wise subtraction between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$rhs);
let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def MulOp : Voila_Op<"mul",[NoSideEffect]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$rhs);
  let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def DivOp : Voila_Op<"div",[NoSideEffect]> {
  let summary = "element-wise division operation";
  let description = [{
    The "div" operation performs element-wise division between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$rhs);
let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def ModOp : Voila_Op<"mod",[NoSideEffect]> {
  let summary = "element-wise modulo operation";
  let description = [{
    The "mod" operation performs element-wise modulo between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$rhs);
let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def AndOp : Voila_Op<"and",[NoSideEffect, SameOperandsAndResultType,Elementwise,SameOperandsAndResultShape,Vectorizable]> {
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

def OrOp : Voila_Op<"or",[NoSideEffect, SameOperandsAndResultType,SameOperandsAndResultShape,Elementwise,Vectorizable]> {
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

def NotOp : Voila_Op<"not",[NoSideEffect, SameOperandsAndResultType,SameOperandsAndResultShape,Elementwise,Vectorizable]> {
let summary = "element-wise not operation";
let description = [{
The "bot" operation performs element-wise not on a tensor.
}];

let arguments = (ins I1Tensor:$value);
let results = (outs I1Tensor);

// Specify a parser and printer method.
//let parser = [{ return ::parseBinaryOp(parser, result); }];
//let printer = [{ return ::printBinaryOp(p, *this); }];
}


def EqOp : Voila_Op<"eq",[NoSideEffect,Elementwise,Vectorizable]> {
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

def NeqOp : Voila_Op<"neq",[NoSideEffect,Elementwise,Vectorizable]> {
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

def LeOp : Voila_Op<"le",[NoSideEffect,Elementwise,Vectorizable]> {
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

def LeqOp : Voila_Op<"leq",[NoSideEffect,Elementwise,Vectorizable]> {
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

def GeOp : Voila_Op<"ge",[NoSideEffect,Elementwise,Vectorizable]> {
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

def GeqOp : Voila_Op<"geq",[NoSideEffect,Elementwise,Vectorizable]> {
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

def ReadOp : Voila_Op<"read",[NoSideEffect]> {
let summary = "read elements from column";
let description = [{
The "read" operation loads data from "column" starting at "index".
The start index is expected to be within "column" bounds.
}];

let arguments = (ins AnyUnrankedMemRef:$column, Index:$index);
let results = (outs AnyTensor);

// Specify a parser and printer method.
//let parser = [{ return ::parseBinaryOp(parser, result); }];
//let printer = [{ return ::printBinaryOp(p, *this); }];
}

def GatherOp : Voila_Op<"gather",[NoSideEffect]> {
let summary = "read elements from column";
let description = [{
The "gather" operation loads data from "column" at "indexes".
The start "indexes" are expected to be within "column" bounds and the shape "indexes" is smaller than "column".
}];

let arguments = (ins AnyUnrankedMemRef:$column, IndexTensor:$indexes);
let results = (outs AnyTensor);

// Specify a parser and printer method.
//let parser = [{ return ::parseBinaryOp(parser, result); }];
//let printer = [{ return ::printBinaryOp(p, *this); }];
}

def WriteOp : Voila_Op<"write",[NoSideEffect]> {
let summary = "write elements to column";
let description = [{
The "write" operation writes data from "src" to "dest" starting from "start_idx".
The "start_idx" is expected to be within "column" bounds.
}];

let arguments = (ins AnyTensor:$src, AnyUnrankedMemRef:$dest ,Index:$start_idx);
let results = (outs AnyTensor);

// Specify a parser and printer method.
//let parser = [{ return ::parseBinaryOp(parser, result); }];
//let printer = [{ return ::printBinaryOp(p, *this); }];
}

def ScatterOp : Voila_Op<"scatter",[NoSideEffect]> {
let summary = "scatter elements to column";
let description = [{
The "scatter" operation writes data at "idxs" from "src" from to "dest".
The "idxs" are expected to be within "dest" bounds and the shape smaller than src.
}];

let arguments = (ins AnyTensor:$src, AnyUnrankedMemRef:$dest ,IndexTensor:$idxs);
let results = (outs AnyTensor);

// Specify a parser and printer method.
//let parser = [{ return ::parseBinaryOp(parser, result); }];
//let printer = [{ return ::printBinaryOp(p, *this); }];
}

def SelectOp : Voila_Op<"select",[NoSideEffect]> {
let summary = "filter elements where pred matches";
let description = [{
The "select" operation returns the "values" where "pred" is true.
The "pred" is expected to have the same shape as "values".
}];

let arguments = (ins AnyTensor:$values ,I1Tensor:$pred);
let results = (outs AnyTensor);

// Specify a parser and printer method.
//let parser = [{ return ::parseBinaryOp(parser, result); }];
//let printer = [{ return ::printBinaryOp(p, *this); }];
}

def EmitOp : Voila_Op<"emit", [NoSideEffect,HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "emit operation";
  let description = [{
    The "emit" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand is passed to the calling function.
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<AnyTensor>:$input);

let extraClassDeclaration = [{
bool hasOperand() { return getNumOperands() != 0; }
}];

let builders = [
OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
];
/*
  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return ::verify(*this); }];
*/
}

def GenericCallOp : Voila_Op<"generic_call"> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee.
}];
  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyTensor>:$inputs);
  let results = (outs Variadic<AnyTensor>);
let builders = [
OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments), [{
    // Generic call always returns nothing initially
    //$_state.addTypes(UnrankedTensorType::get($-builder.getF64Type()));
    $_state.addOperands(arguments);
    $_state.addAttribute("callee", $_builder.getSymbolRefAttr(callee));
}]>
];
}

//TODO
def LoopOp : Voila_Op<"loop"> {
let summary = "loop operation";
let description = [{
}];
// The generic call operation takes a symbol reference attribute as the
// callee, and inputs for the call.
let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyTensor>:$inputs);
let results = (outs Variadic<AnyTensor>);
let builders = [
OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments), [{
// Generic call always returns nothing initially
//$_state.addTypes(UnrankedTensorType::get($-builder.getF64Type()));
$_state.addOperands(arguments);
$_state.addAttribute("callee", $_builder.getSymbolRefAttr(callee));
}]>
];
}

/* TODO: ops aggregations*/
#endif // VOILA_OPS