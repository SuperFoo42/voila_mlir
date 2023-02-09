#pragma once

#include "mlir/IR/PatternMatch.h"       // for OpRewritePattern, PatternRew...
#include "mlir/Support/LogicalResult.h" // for LogicalResult

namespace mlir
{
    class MLIRContext;
    namespace voila
    {
        class EmitOp;
    }
    namespace func
    {
        class FuncOp;
    }
} // namespace mlir

namespace voila::mlir::lowering
{
    struct EmitOpLowering : public ::mlir::OpRewritePattern<::mlir::voila::EmitOp>
    {
        ::mlir::func::FuncOp &function;

        EmitOpLowering(::mlir::MLIRContext *ctx, ::mlir::func::FuncOp &function);

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::EmitOp op, ::mlir::PatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering