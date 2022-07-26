#pragma once
#include "mlir/Transforms/DialectConversion.h"

#include "MLIRLoweringError.hpp"

namespace voila::mlir::lowering
{
    struct LoopOpLowering : public ::mlir::ConversionPattern
    {
        using LoopIterationFn = ::mlir::function_ref<void(::mlir::PatternRewriter &rewriter,
                                                          ::mlir::ValueRange memRefOperands,
                                                          ::mlir::ValueRange loopIvs,
                                                          ::mlir::Value iter_var)>;

        explicit LoopOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              ::mlir::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering
