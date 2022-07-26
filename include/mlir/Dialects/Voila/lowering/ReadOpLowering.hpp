#pragma once

#include "mlir/Transforms/DialectConversion.h"

#include "MLIRLoweringError.hpp"

namespace voila::mlir::lowering
{
    struct ReadOpLowering : public ::mlir::ConversionPattern
    {
        explicit ReadOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              ::mlir::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering