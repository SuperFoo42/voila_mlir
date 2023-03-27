#pragma once
#include "mlir/Dialects/Voila/IR/VoilaOps.h"   // for GatherOpAdaptor
#include "mlir/Support/LLVM.h"                 // for function_ref
#include "mlir/Support/LogicalResult.h"        // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef

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
    struct MaxOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::MaxOp>
    {
        using OpConversionPattern<::mlir::voila::MaxOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::MaxOp op,
                                              OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering