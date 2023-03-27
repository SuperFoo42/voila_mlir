#pragma once

#include "mlir/Support/LLVM.h"                 // for function_ref
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
    struct AvgOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::AvgOp>
    {
        using LoopIterationFn = ::mlir::function_ref<::mlir::Value(::mlir::OpBuilder &rewriter,
                                                                   ::mlir::ValueRange memRefOperands,
                                                                   ::mlir::ValueRange loopIvs,
                                                                   ::mlir::Value iter_var)>;
        using OpConversionPattern<::mlir::voila::AvgOp>::OpConversionPattern;


        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::AvgOp op, OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering