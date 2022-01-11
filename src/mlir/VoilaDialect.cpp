#include "mlir/IR/VoilaDialect.h"
#include "mlir/IR/VoilaOps.h"
#include "mlir/Interfaces/VoilaInlinerInterface.hpp"

using namespace mlir;
using namespace mlir::voila;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//
#include "mlir/IR/VoilaOpsDialect.cpp.inc"
using namespace ::voila::mlir;
void VoilaDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mlir/IR/VoilaOps.cpp.inc"
    >();
    addInterfaces<VoilaInlinerInterface>();
}