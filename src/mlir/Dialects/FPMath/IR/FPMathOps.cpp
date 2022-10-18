#include "mlir/Dialects/FPMath/IR/FPMathOps.h"

#include "NotImplementedException.hpp"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#define GET_OP_CLASSES
#include "mlir/Dialects/FPMath/IR/FPMathOps.cpp.inc"
using namespace ::mlir;
using namespace ::mlir::arith;
using namespace mlir::fpmath;