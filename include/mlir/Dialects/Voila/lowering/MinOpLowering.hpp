#pragma once
#include "mlir/Transforms/DialectConversion.h"

#include "MLIRLoweringError.hpp"
#include "mlir/Dialects/Voila/IR/VoilaOps.h" // for GatherOpAdaptor

namespace voila::mlir::lowering
{
    struct MinOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::MinOp>
    {
        using OpConversionPattern<::mlir::voila::MinOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::MinOp op,
                                              OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering