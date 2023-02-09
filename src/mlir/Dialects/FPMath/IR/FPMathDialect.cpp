#include "mlir/Dialects/FPMath/IR/FPMathDialect.h"
#include "mlir/Dialects/FPMath/IR/FPMathOps.h"
#include "mlir/Dialects/FPMath/IR/FPMathTypes.h"
#include "mlir/Dialects/FPMath/IR/FPMathAttr.h"
#include "llvm/ADT/StringRef.h"                            // for operator==

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialects/FPMath/IR/FPMathAttr.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialects/FPMath/IR/FPMathOpsTypes.cpp.inc"

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
    addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialects/FPMath/IR/FPMathAttr.cpp.inc"
    >();

    /// Add the defined types to the dialect.
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialects/FPMath/IR/FPMathOpsTypes.cpp.inc"
    >();
}