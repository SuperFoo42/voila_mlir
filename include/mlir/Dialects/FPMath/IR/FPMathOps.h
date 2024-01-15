#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "FPMathTypes.h"
#include "FPMathAttr.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "mlir/IR/PatternMatch.h"
#pragma GCC diagnostic pop
#define GET_OP_CLASSES
#include "mlir/Dialects/FPMath/IR/FPMathOps.h.inc"