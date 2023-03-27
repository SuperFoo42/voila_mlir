#pragma once

#include "mlir/Support/LogicalResult.h"        // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef
#include "mlir/Dialects/Voila/IR/VoilaOps.h"   // for GatherOpAdaptor

namespace mlir
{
    class Operation;
    class Value;
} // namespace mlir

namespace voila::mlir::lowering
{
    struct ReadOpLowering : public  ::mlir::OpConversionPattern<::mlir::voila::ReadOp>
    {
        using OpConversionPattern<::mlir::voila::ReadOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::ReadOp op,
                                              OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering