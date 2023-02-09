#pragma once
#include "mlir/IR/Value.h"              // for Value
#include "mlir/IR/ValueRange.h"         // for ValueRange
#include "mlir/Support/LLVM.h"          // for function_ref
#include "mlir/Support/LogicalResult.h" // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef

namespace mlir
{
    class ImplicitLocOpBuilder;
    class MLIRContext;
    class Operation;
    class PatternRewriter;
}

namespace voila::mlir::lowering
{
    class SelectOpLowering : public ::mlir::ConversionPattern
    {
        using LoopIterationFn = ::mlir::function_ref<::mlir::Value(::mlir::ImplicitLocOpBuilder &builder,
                                                                   ::mlir::ValueRange memRefOperands,
                                                                   ::mlir::ValueRange loopIvs,
                                                                   ::mlir::Value iter_var,
                                                                   ::mlir::Value dest)>;
        /// This defines the function type used to process an iteration of a lowered
        /// loop. It takes as input an OpBuilder, an range of memRefOperands
        /// corresponding to the operands of the input operation, and the range of loop
        /// induction variables for the iteration. It returns a value to store at the
        /// current index of the iteration.
        static void lowerOpToLoops(::mlir::Operation *op,
                                   ::mlir::ValueRange operands,
                                   ::mlir::PatternRewriter &rewriter,
                                   LoopIterationFn processIteration);

      public:
        explicit SelectOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              ::mlir::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering
