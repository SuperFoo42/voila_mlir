#pragma once

#include "BinaryOpLowering.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"
#include "BinaryOpLowering.hpp"

namespace voila::mlir::lowering
{
    using AndOpLowering = BinaryOpLowering<::mlir::voila::AndOp, SingleTypeBinOpGenerator<::mlir::arith::AndIOp>>;
    using OrOpLowering = BinaryOpLowering<::mlir::voila::OrOp, SingleTypeBinOpGenerator<::mlir::arith::OrIOp>>;

    // TODO: not op lowering
} // namespace voila::mlir::lowering