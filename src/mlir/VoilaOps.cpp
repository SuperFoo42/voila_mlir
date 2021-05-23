#include "mlir/VoilaOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/VoilaDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"

#define GET_OP_CLASSES
#include "mlir/VoilaOps.cpp.inc"
