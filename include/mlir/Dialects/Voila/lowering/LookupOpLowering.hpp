#pragma once
#include "mlir/Dialects/Voila/IR/VoilaOps.h"   // for GatherOpAdaptor
#include "mlir/Support/LogicalResult.h"        // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef

namespace mlir
{
    class MLIRContext;
    class Operation;
    class Value;
} // namespace mlir

namespace voila::mlir::lowering
{
    struct LookupOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::LookupOp>
    {
        using OpConversionPattern<::mlir::voila::LookupOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::LookupOp op,
                                              OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering