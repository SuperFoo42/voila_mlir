#pragma once
#include "mlir/Transforms/DialectConversion.h"

namespace voila::mlir::lowering
{
    /**
     * Hash function based on XXH3 with seed = 0
     * @link{https://cyan4973.github.io/xxHash/}
     */
    struct HashOpLowering : public ::mlir::ConversionPattern
    {
        explicit HashOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              ::mlir::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering