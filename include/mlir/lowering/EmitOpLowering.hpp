#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

namespace voila::mlir::lowering
{
    struct EmitOpLowering : public ::mlir::OpRewritePattern<::mlir::voila::EmitOp>
    {
        ::mlir::FuncOp &function;

        EmitOpLowering(::mlir::MLIRContext *ctx, ::mlir::FuncOp &function);

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::EmitOp op, ::mlir::PatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering