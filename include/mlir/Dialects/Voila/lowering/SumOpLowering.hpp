#pragma once
#include "mlir/Support/LLVM.h"          // for function_ref
#include "mlir/Support/LogicalResult.h" // for LogicalResult
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h" // for ArrayRef
#include "mlir/Dialects/Voila/IR/VoilaOps.h"   // for GatherOpAdaptor

namespace mlir
{
    class OpBuilder;
    class Operation;
    class Value;
    class ValueRange;
} // namespace mlir

namespace voila::mlir::lowering
{
    struct SumOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::SumOp>
    {
        using OpConversionPattern<::mlir::voila::SumOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::SumOp op,
                                              OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering