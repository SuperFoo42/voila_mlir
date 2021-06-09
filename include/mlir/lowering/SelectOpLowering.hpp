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

#include "llvm/ADT/Sequence.h"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::voila;

    using LoopIterationFn = function_ref<Value(OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

    class SelectOpLowering : public ConversionPattern
    {
        /// Convert the given TensorType into the corresponding MemRefType.
        static MemRefType convertTensorToMemRef(TensorType type);

        /// Insert an allocation and deallocation for the given MemRefType.
        static Value
        insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter &rewriter);

        /// This defines the function type used to process an iteration of a lowered
        /// loop. It takes as input an OpBuilder, an range of memRefOperands
        /// corresponding to the operands of the input operation, and the range of loop
        /// induction variables for the iteration. It returns a value to store at the
        /// current index of the iteration.
        static void lowerOpToLoops(Operation *op,
                                   ValueRange operands,
                                   PatternRewriter &rewriter,
                                   LoopIterationFn processIteration);

      public:
        explicit SelectOpLowering(MLIRContext *ctx) : ConversionPattern(::mlir::voila::SelectOp::getOperationName(), 1, ctx) {}

        LogicalResult matchAndRewrite(Operation *op,
                                      llvm::ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowerin
