#pragma once
#include "mlir/Support/LogicalResult.h"        // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef
#include "mlir/Dialects/Voila/IR/VoilaOps.h" // for GatherOpAdaptor

namespace mlir
{
    class MLIRContext;
    class OpBuilder;
    class Operation;
    class Value;
    class ValueRange;
} // namespace mlir

namespace voila::mlir::lowering
{
    struct GatherOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::GatherOp>
    {
        using OpConversionPattern<::mlir::voila::GatherOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::GatherOp op, OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering