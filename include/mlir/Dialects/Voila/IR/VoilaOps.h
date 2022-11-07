#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialects/FPMath/IR/FPMathAttr.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialects/Voila/Interfaces/PredicationOpInterface.hpp"
#pragma GCC diagnostic pop
#define GET_OP_CLASSES
#include "mlir/Dialects/Voila//IR/VoilaOps.h.inc"