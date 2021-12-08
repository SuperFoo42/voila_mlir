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

namespace voila::mlir::lowering
{
    using AddOpLowering =
        BinaryOpLowering<::mlir::voila::AddOp, IntFloatBinOpGenerator<::mlir::arith::AddIOp, ::mlir::arith::AddFOp>>;
    using SubOpLowering =
        BinaryOpLowering<::mlir::voila::SubOp, IntFloatBinOpGenerator<::mlir::arith::SubIOp, ::mlir::arith::SubFOp>>;
    using MulOpLowering =
        BinaryOpLowering<::mlir::voila::MulOp, IntFloatBinOpGenerator<::mlir::arith::MulIOp, ::mlir::arith::MulFOp>>;
    using DivOpLowering =
        BinaryOpLowering<::mlir::voila::DivOp, IntFloatBinOpGenerator<::mlir::arith::DivSIOp, ::mlir::arith::DivFOp>>;
    using ModOpLowering =
        BinaryOpLowering<::mlir::voila::ModOp, IntFloatBinOpGenerator<::mlir::arith::RemSIOp, ::mlir::arith::RemFOp>>;

} // namespace voila::mlir::lowering