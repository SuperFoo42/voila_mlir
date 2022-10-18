#include "mlir/Dialects/FPMath/IR/FPMathDialect.h"
#include "mlir/Dialects/FPMath/IR/FPMathOps.h"

using namespace mlir;
using namespace mlir::fpmath;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//
#include "mlir/Dialects/FPMath/IR/FPMathOpsDialect.cpp.inc"
using namespace ::mlir::fpmath;
void FPMathDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialects/FPMath/IR/FPMathOps.cpp.inc"
    >();
}