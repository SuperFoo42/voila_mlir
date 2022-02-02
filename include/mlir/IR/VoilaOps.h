#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinOps.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/PredicationOpInterface.hpp"
#pragma GCC diagnostic pop
#define GET_OP_CLASSES
#include "mlir/IR/VoilaOps.h.inc"