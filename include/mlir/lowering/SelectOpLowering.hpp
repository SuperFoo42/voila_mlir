#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/Sequence.h"
namespace voila::mlir::lowering
{
    class SelectOpLowering : public ::mlir::ConversionPattern
    {
        using LoopIterationFn = ::mlir::function_ref<::mlir::Value(::mlir::OpBuilder &rewriter, ::mlir::ValueRange memRefOperands, ::mlir::ValueRange loopIvs, ::mlir::Value iter_var, ::mlir::Value dest)>;
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
        explicit SelectOpLowering(::mlir::MLIRContext *ctx) : ConversionPattern(::mlir::voila::SelectOp::getOperationName(), 1, ctx) {}

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                      llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowerin
