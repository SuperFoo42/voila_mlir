#include "mlir/Dialects/Voila/IR/VoilaDialect.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir//Dialects/Voila/Interfaces/VoilaInlinerInterface.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialects/FPMath/IR/FPMathDialect.h"

using namespace mlir;
using namespace mlir::voila;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//
#include "mlir/Dialects/Voila/IR/VoilaOpsDialect.cpp.inc"
using namespace ::voila::mlir;
void VoilaDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialects/Voila/IR/VoilaOps.cpp.inc"
    >();
    addInterfaces<VoilaInlinerInterface>();
}