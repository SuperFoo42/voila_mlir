#ifndef VOILA_OPS
#define VOILA_OPS

include "VoilaDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/ShapeInferenceInterface.td"

def IntConstOp : Voila_Op<"integer constant", [NoSideEffect]> {
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
  //TODO:
  let results = (outs AnyTypeOf<[I32,I64]>);

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

def FltConstOp : Voila_Op<"float constant", [NoSideEffect]> {
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
  let results = (outs F64);

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

def BoolConstOp : Voila_Op<"bool constant", [NoSideEffect]> {
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
  let results = (outs I1);

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
def AddOp : Voila_Op<"add",[NoSideEffect, Commutative,DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$rhs);
let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>);
  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def SubOp : Voila_Op<"sub",[NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise subtraction operation";
  let description = [{
    The "sub" operation performs element-wise subtraction between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$rhs);
let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>:$result);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def MulOp : Voila_Op<"mul",[NoSideEffect, Commutative,DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$rhs);
  let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def DivOp : Voila_Op<"div",[NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise division operation";
  let description = [{
    The "div" operation performs element-wise division between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$rhs);
let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def ModOp : Voila_Op<"mod",[NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise modulo operation";
  let description = [{
    The "mod" operation performs element-wise modulo between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

let arguments = (ins AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$lhs, AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>, AnyInteger, AnyFloat]>:$rhs);
let results = (outs AnyTypeOf<[TensorOf<[AnyInteger]>, TensorOf<[AnyFloat]>]>);

  // Specify a parser and printer method.
 // let parser = [{ return ::parseBinaryOp(parser, result); }];
 // let printer = [{ return ::printBinaryOp(p, *this); }];
}

def AndOp : Voila_Op<"and",[NoSideEffect,Commutative,DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise and operation";
  let description = [{
    The "and" operation performs element-wise and between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTypeOf<[I1Tensor,I1]>:$lhs, AnyTypeOf<[I1Tensor,I1]>:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];

  //TODO: verifyer checking same shape
}

def OrOp : Voila_Op<"or",[NoSideEffect,Commutative, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise or operation";
  let description = [{
    The "or" operation performs element-wise or between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTypeOf<[I1Tensor,I1]>:$lhs, AnyTypeOf<[I1Tensor,I1]>:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def NotOp : Voila_Op<"not",[NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
let summary = "element-wise not operation";
let description = [{
The "bot" operation performs element-wise not on a tensor.
}];

let arguments = (ins AnyTypeOf<[I1Tensor,I1]>:$value);
let results = (outs I1Tensor);

// Specify a parser and printer method.
//let parser = [{ return ::parseBinaryOp(parser, result); }];
//let printer = [{ return ::printBinaryOp(p, *this); }];
}


def EqOp : Voila_Op<"eq",[NoSideEffect,Commutative,DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise equality comparison operation";
  let description = [{
    The "eq" operation performs element-wise equality comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def NeqOp : Voila_Op<"neq",[NoSideEffect,Commutative,DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise inequality comparison operation";
  let description = [{
    The "neq" operation performs element-wise inequality comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def LeOp : Voila_Op<"le",[NoSideEffect,DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise less comparison operation";
  let description = [{
    The "le" operation performs element-wise less comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def LeqOp : Voila_Op<"leq",[NoSideEffect,DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise less-equal comparison operation";
  let description = [{
    The "leq" operation performs element-wise less-equal comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def GeOp : Voila_Op<"ge",[NoSideEffect,DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise greater comparison operation";
  let description = [{
    The "ge" operation performs element-wise greater comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1Tensor);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def GeqOp : Voila_Op<"geq",[NoSideEffect,DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise greater-equal comparison operation";
  let description = [{
    The "geq" operation performs element-wise greater-equal comparison between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
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

let arguments = (ins 1DTensorOf<[AnyType]>:$column, 1DTensorOf<[AnySignlessInteger]>:$index);
let results = (outs 1DTensorOf<[AnyType]>);

// Specify a parser and printer method.
//let parser = [{ return ::parseBinaryOp(parser, result); }];
//let printer = [{ return ::printBinaryOp(p, *this); }];
}

def GatherOp : Voila_Op<"gather",[NoSideEffect]> {
let summary = "read elements from column";
let description = [{
The "gather" operation loads data from "column" at "indices".
The start "indices" are expected to be within "column" bounds and the shape "indices" is smaller than "column".
}];

let arguments = (ins 1DTensorOf<[AnyType]>:$column, 1DTensorOf<[AnySignlessInteger]>:$indices);
let results = (outs 1DTensorOf<[AnyType]>);

// Specify a parser and printer method.
//let parser = [{ return ::parseBinaryOp(parser, result); }];
//let printer = [{ return ::printBinaryOp(p, *this); }];
}

//TODO
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

//TODO
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
  let arguments = (ins Variadic<AnyType>:$input);

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

def GenericCallOp : Voila_Op<"generic_call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
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
  let results = (outs AnyTensor);
let builders = [
OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments), [{
    // Generic call always returns nothing initially
    $_state.addTypes(llvm::None);
    $_state.addOperands(arguments);
    $_state.addAttribute("callee", mlir::SymbolRefAttr::get($_builder.getContext(), callee));
}]>,
OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments, "Type":$res_type), [{
// Generic call always returns nothing initially
$_state.addTypes(res_type);
$_state.addOperands(arguments);
$_state.addAttribute("callee", mlir::SymbolRefAttr::get($_builder.getContext(), callee));
}]>
];
}

//TODO
def LoopOp : Voila_Op<"loop",[NoTerminator]> {
let summary = "loop operation";
let description = [{
    //TODO
}];
let arguments = (ins I1Tensor:$cond);
let results = (outs I1); //only needed because we translate this to affine loop with iter_var
let regions = (region AnyRegion:$body);
}

def CastOp : Voila_Op<"cast", [
DeclareOpInterfaceMethods<CastOpInterface>,
NoSideEffect,
SameOperandsAndResultShape]
> {
let summary = "shape cast operation";
let description = [{
The "cast" operation converts a tensor from one type to an equivalent type
        without changing any data elements. The source and destination types
must both be tensor types with the same element type. If both are ranked,
        then shape is required to match. The operation is invalid if converting
        to a mismatching constant dimension.
}];

let arguments = (ins AnyTensor:$input);
let results = (outs AnyTensor:$output);
}

/* TODO: ops aggregations*/
def MaxOp : Voila_Op<"aggr_max", [NoSideEffect]> {
let summary = "maximum value reduction operation";
let description = [{
//TODO
}];

let arguments = (ins 1DTensorOf<[AnyTypeOf<[AnyInteger,AnyFloat]>]>:$input, Optional<1DTensorOf<[Index, AnyInteger]>>:$indices);
let results = (outs AnyTypeOf<[AnyInteger,AnyFloat, 1DTensorOf<[AnyInteger,AnyFloat]>]>:$output);
}

def MinOp : Voila_Op<"aggr_min", [NoSideEffect]> {
let summary = "minimum value reduction operation";
let description = [{
//TODO
}];

let arguments = (ins 1DTensorOf<[AnyTypeOf<[AnyInteger,AnyFloat]>]>:$input, Optional<1DTensorOf<[Index, AnyInteger]>>:$indices);
let results = (outs AnyTypeOf<[AnyInteger,AnyFloat, 1DTensorOf<[AnyInteger,AnyFloat]>]>:$output);
}

def SumOp : Voila_Op<"aggr_sum", [NoSideEffect]> {
let summary = "sum value reduction operation";
let description = [{
//TODO
}];

let arguments = (ins 1DTensorOf<[AnyTypeOf<[AnyInteger,AnyFloat]>]>:$input, Optional<1DTensorOf<[Index, AnyInteger]>>:$indices);
let results = (outs AnyTypeOf<[I64,F64, 1DTensorOf<[AnyTypeOf<[I64,F64]>]>]>:$output);
}

def CountOp : Voila_Op<"aggr_count", [NoSideEffect]> {
let summary = "sum value reduction operation";
let description = [{
//TODO
}];

let arguments = (ins 1DTensorOf<[AnyType]>:$input, Optional<1DTensorOf<[Index, AnyInteger]>>:$indices);
let results = (outs AnyTypeOf<[I64, 1DTensorOf<[I64]>]>:$output);
}

def AvgOp : Voila_Op<"aggr_avg", [NoSideEffect]> {
let summary = "sum value reduction operation";
let description = [{
//TODO
}];

let arguments = (ins 1DTensorOf<[AnyTypeOf<[AnyInteger,AnyFloat]>]>:$input, Optional<1DTensorOf<[Index, AnyInteger]>>:$indices);
let results = (outs AnyTypeOf<[F64, 1DTensorOf<[F64]>]>:$output);
}

def MoveOp : Voila_Op<"move", [SameTypeOperands, SameOperandsShape]> {
let summary = "move result from one ssa value to other in order to keep track of references";
let description = [{
//TODO
}];

let arguments = (ins AnyType:$input, AnyType:$out);
}


def HashOp : Voila_Op<"hash", [NoSideEffect]> {
let summary = "hash input values for use with hash tables";
let description = [{
//TODO
}];

let arguments = (ins Variadic<1DTensorOf<[AnyType]>>:$input);
let results = (outs I64Tensor:$output);
}

def InsertOp : Voila_Op<"insert"> {
let summary = "insert values in a hash table";
let description = [{
This is a convenience function, allocating a hashtable, hashing and scatter + linear probing for collision resolution
}];

let arguments = (ins I64Tensor:$hashValues, Variadic<1DTensorOf<[AnyType]>>:$values);
let results = (outs Variadic<1DTensorOf<[AnyType]>>:$hashtables);
}

def LookupOp : Voila_Op<"lookup", [NoSideEffect,SameVariadicOperandSize]> {
let summary = "lookup index of hash values";
let description = [{
This is a convenience function, basically consisting of hashing and gather + linear probing for collision resolution
}];

let arguments = (ins Variadic<1DTensorOf<[AnyType]>>:$values, Variadic<1DTensorOf<[AnyType]>>:$hashtables, 1DTensorOf<[I64]>:$hashes);
let results = (outs IndexTensor:$output);
//TODO: verify same shape of values and hashtables
}
#endif // VOILA_OPS