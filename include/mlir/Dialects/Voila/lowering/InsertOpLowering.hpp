#pragma once

#include "mlir/Support/LogicalResult.h"        // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef
#include "mlir/Dialects/Voila/IR/VoilaOps.h" // for GatherOpAdaptor

namespace mlir
{
    class MLIRContext;
    class Operation;
    class Value;
} // namespace mlir

namespace voila::mlir::lowering
{
    struct InsertOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::InsertOp>
    {
        using OpConversionPattern<::mlir::voila::InsertOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::InsertOp op, OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering
