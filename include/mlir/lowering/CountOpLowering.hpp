#pragma once

#include "mlir/Transforms/DialectConversion.h"

#include <MLIRLoweringError.hpp>

namespace voila::mlir::lowering
{
    struct CountOpLowering : public ::mlir::ConversionPattern
    {
        explicit CountOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering