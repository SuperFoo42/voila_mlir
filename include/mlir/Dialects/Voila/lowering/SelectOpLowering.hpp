#pragma once
#include "mlir/IR/Value.h"              // for Value
#include "mlir/IR/ValueRange.h"         // for ValueRange
#include "mlir/Support/LLVM.h"          // for function_ref
#include "mlir/Support/LogicalResult.h" // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef
#include "mlir/Dialects/Voila/IR/VoilaOps.h"   // for GatherOpAdaptor

namespace mlir
{
    class ImplicitLocOpBuilder;
    class Operation;
    class PatternRewriter;
}

namespace voila::mlir::lowering
{
    class SelectOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::SelectOp>
    {
        using LoopIterationFn = ::mlir::function_ref<::mlir::Value(::mlir::ImplicitLocOpBuilder &builder,
                                                                   ::mlir::voila::SelectOp,
                                                                   ::mlir::ValueRange loopIvs,
                                                                   ::mlir::Value iter_var,
                                                                   ::mlir::Value dest)>;
        /// This defines the function type used to process an iteration of a lowered
        /// loop. It takes as input an OpBuilder, an range of memRefOperands
        /// corresponding to the operands of the input operation, and the range of loop
        /// induction variables for the iteration. It returns a value to store at the
        /// current index of the iteration.
        static void lowerOpToLoops(::mlir::voila::SelectOp op,
                                   ::mlir::PatternRewriter &rewriter,
                                   LoopIterationFn processIteration);

      public:
        using OpConversionPattern<::mlir::voila::SelectOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::SelectOp op,
                                              OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering
