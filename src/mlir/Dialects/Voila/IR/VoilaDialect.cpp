#include "mlir/Dialects/Voila/IR/VoilaDialect.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir//Dialects/Voila/Interfaces/VoilaInlinerInterface.hpp"

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