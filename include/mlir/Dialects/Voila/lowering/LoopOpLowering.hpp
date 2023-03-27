#pragma once
#include "mlir/IR/Value.h"              // for Value
#include "mlir/IR/ValueRange.h"         // for ValueRange
#include "mlir/Support/LLVM.h"          // for function_ref
#include "mlir/Support/LogicalResult.h" // for LogicalResult
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h" // for ArrayRef
#include "mlir/Dialects/Voila/IR/VoilaOps.h" // for GatherOpAdaptor

namespace mlir
{
    class MLIRContext;
    class Operation;
    class PatternRewriter;
} // namespace mlir

namespace voila::mlir::lowering
{
    struct LoopOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::LoopOp>
    {
        using LoopIterationFn = ::mlir::function_ref<void(::mlir::PatternRewriter &rewriter,
                                                          ::mlir::ValueRange loopIvs,
                                                          ::mlir::Value iter_var)>;

        using OpConversionPattern<::mlir::voila::LoopOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::LoopOp op, OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering
