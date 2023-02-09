#pragma once
#include "mlir/IR/Value.h"              // for Value
#include "mlir/IR/ValueRange.h"         // for ValueRange
#include "mlir/Support/LLVM.h"          // for function_ref
#include "mlir/Support/LogicalResult.h" // for LogicalResult
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h" // for ArrayRef

namespace mlir
{
    class MLIRContext;
    class Operation;
    class PatternRewriter;
} // namespace mlir

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
