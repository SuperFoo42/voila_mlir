#pragma once
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
    struct MaxOpLowering : public ::mlir::ConversionPattern
    {
        using LoopIterationFn = ::mlir::function_ref<::mlir::Value(::mlir::OpBuilder &rewriter,
                                                                   ::mlir::ValueRange memRefOperands,
                                                                   ::mlir::ValueRange loopIvs,
                                                                   ::mlir::Value iter_var)>;
        explicit MaxOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering