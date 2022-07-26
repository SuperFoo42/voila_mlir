#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace voila::mlir::lowering
{
    struct InsertOpLowering : public ::mlir::ConversionPattern
    {
        explicit InsertOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering
