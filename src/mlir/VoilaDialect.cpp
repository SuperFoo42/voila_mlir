#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

using namespace mlir;
using namespace mlir::voila;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void VoilaDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mlir/VoilaOps.cpp.inc"
    >();
}