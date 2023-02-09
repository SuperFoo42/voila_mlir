#pragma once

#include "mlir/Support/LLVM.h"                 // for function_ref
#include "mlir/Support/LogicalResult.h"        // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef

namespace mlir
{
    class ImplicitLocOpBuilder;
    class MLIRContext;
    class Operation;
    class ValueRange;
    class Value;
} // namespace mlir

namespace mlir
{
    class ImplicitLocOpBuilder;
}
namespace voila::mlir::lowering
{
    struct NotOpLowering : public ::mlir::ConversionPattern
    {
        explicit NotOpLowering(::mlir::MLIRContext *ctx);

        using LoopIterationFn = ::mlir::function_ref<::mlir::Value(
            ::mlir::ImplicitLocOpBuilder &rewriter, ::mlir::ValueRange memRefOperands, ::mlir::ValueRange loopIvs)>;

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering