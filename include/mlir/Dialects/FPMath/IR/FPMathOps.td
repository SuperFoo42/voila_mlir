#ifndef FPMATH_OPS
#define FPMATH_OPS

include "FPMathDialect.td"
include "FPMathTypes.td"
include "FPMathAttr.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

def ConstantOp : FPMath_Op<"constant"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "constant operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins FPMath_FixedAttr:$value);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def AddOp : FPMath_Op<"add"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "add operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def SubOp : FPMath_Op<"sub"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "sub operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def MulOp : FPMath_Op<"mul"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "mul operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}


def DivOp : FPMath_Op<"div"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "div operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def FpToDecOp : FPMath_Op<"fptodec"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "float to decimal cast operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins AnyFloat:$arg);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def IntToDecOp : FPMath_Op<"inttodec"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "int to decimal cast operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins AnyInteger:$arg);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def DecToFltOp : FPMath_Op<"dectoflt"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "decimal to float cast operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$arg);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs AnyFloat);
}

def DecToIntOp : FPMath_Op<"dectoint"> {
    //TODO: rounding mode
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "decimal to int rounding operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$arg);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs AnyInteger);
}

def EqOp : FPMath_Op<"eq"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "equals operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def NeqOp : FPMath_Op<"neq"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "not-equals operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def LtOp : FPMath_Op<"lt"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "less-than operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def LeOp : FPMath_Op<"le"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "less-equal operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def GeOp : FPMath_Op<"ge"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "greater-equal operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}

def GtOp : FPMath_Op<"gt"> {
// Provide a summary and description for this operation. This can be used to
// auto-generate documentation of the operations within our dialect.
let summary = "greater-than operation";
let description = [{
TBD
}];

// The constant operation takes an attribute as the only input.
// `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
let arguments = (ins DecimalType:$lhs, DecimalType:$rhs);

// The generic call operation returns a single value of TensorType.
// F64Tensor corresponds to a 64-bit floating-point TensorType.
let results = (outs DecimalType);
}
#endif
