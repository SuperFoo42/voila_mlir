#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h" // for GatherOpAdaptor
#include "MLIRLoweringError.hpp"

namespace voila::mlir::lowering
{
    struct CountOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::CountOp>
    {
        using OpConversionPattern<::mlir::voila::CountOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::CountOp op, OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering