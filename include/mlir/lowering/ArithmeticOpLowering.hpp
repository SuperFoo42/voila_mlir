#pragma once
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
#include "BinaryOpLowering.hpp"

namespace voila::mlir::lowering
{
    //TODO: does this overlaod work for int and float?
    using AddFOpLowering = BinaryOpLowering<::mlir::voila::AddOp, ::mlir::AddFOp>;
    using AddIOpLowering = BinaryOpLowering<::mlir::voila::AddOp, ::mlir::AddIOp>;
    using SubIOpLowering = BinaryOpLowering<::mlir::voila::SubOp, ::mlir::SubIOp>;
    using SubFOpLowering = BinaryOpLowering<::mlir::voila::SubOp, ::mlir::SubFOp>;
    using MulIOpLowering = BinaryOpLowering<::mlir::voila::MulOp, ::mlir::MulIOp>;
    using MulFOpLowering = BinaryOpLowering<::mlir::voila::MulOp, ::mlir::MulFOp>;
    using DivFOpLowering = BinaryOpLowering<::mlir::voila::DivOp, ::mlir::DivFOp>;
    using ModFOpLowering = BinaryOpLowering<::mlir::voila::ModOp, ::mlir::RemFOp>;

} // namespace voila::mlir::lowering