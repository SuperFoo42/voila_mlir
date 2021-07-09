#pragma once
#include "BinaryOpLowering.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

namespace voila::mlir::lowering
{
    using AddOpLowering =
        BinaryOpLowering<::mlir::voila::AddOp, IntFloatBinOpGenerator<::mlir::AddIOp, ::mlir::AddFOp>>;
    using SubOpLowering =
        BinaryOpLowering<::mlir::voila::SubOp, IntFloatBinOpGenerator<::mlir::SubIOp, ::mlir::SubFOp>>;
    using MulOpLowering =
        BinaryOpLowering<::mlir::voila::MulOp, IntFloatBinOpGenerator<::mlir::MulIOp, ::mlir::MulFOp>>;
    using DivOpLowering =
        BinaryOpLowering<::mlir::voila::DivOp, IntFloatBinOpGenerator<::mlir::SignedDivIOp, ::mlir::DivFOp>>;
    using ModOpLowering =
        BinaryOpLowering<::mlir::voila::ModOp, IntFloatBinOpGenerator<::mlir::SignedRemIOp, ::mlir::RemFOp>>;

} // namespace voila::mlir::lowering