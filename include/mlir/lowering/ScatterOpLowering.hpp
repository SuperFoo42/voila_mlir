#pragma once

#include "mlir/Transforms/DialectConversion.h"

#include <MLIRLoweringError.hpp>

namespace voila::mlir::lowering
{
    struct ScatterOpLowering : public ::mlir::ConversionPattern
    {
        explicit ScatterOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              ::mlir::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering
