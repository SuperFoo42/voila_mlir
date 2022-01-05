#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace voila::mlir::lowering
{
    struct NotOpLowering : public ::mlir::ConversionPattern
    {
        explicit NotOpLowering(::mlir::MLIRContext *ctx);

        using LoopIterationFn = ::mlir::function_ref<
            ::mlir::Value(::mlir::OpBuilder &rewriter, ::mlir::ValueRange memRefOperands, ::mlir::ValueRange loopIvs)>;

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering