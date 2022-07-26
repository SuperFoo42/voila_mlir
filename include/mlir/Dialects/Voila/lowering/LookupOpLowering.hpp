#pragma once
#include "mlir/Transforms/DialectConversion.h"

namespace voila::mlir::lowering
{
    struct LookupOpLowering : public ::mlir::ConversionPattern
    {
        explicit LookupOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering