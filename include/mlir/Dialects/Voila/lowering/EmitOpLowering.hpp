#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::voila
{
    class EmitOp;
}

namespace mlir::func
{
    class FuncOp;
}

namespace voila::mlir::lowering
{
    struct EmitOpLowering : public ::mlir::OpRewritePattern<::mlir::voila::EmitOp>
    {
        ::mlir::func::FuncOp &function;

        EmitOpLowering(::mlir::MLIRContext *ctx, ::mlir::func::FuncOp &function);

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::EmitOp op, ::mlir::PatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering