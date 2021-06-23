#include "mlir/lowering/SumOpLowering.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::SumOp;
    using ::mlir::voila::SumOpAdaptor;

    SumOpLowering::SumOpLowering(MLIRContext *ctx) : ConversionPattern(SumOp::getOperationName(), 1, ctx) {}

    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static void lowerOpToLoops(Operation *op,
                               ValueRange operands,
                               PatternRewriter &rewriter,
                               SumOpLowering::LoopIterationFn processIteration)
    {
        SumOpAdaptor sumOpAdaptor(operands);
        auto loc = op->getLoc();
        auto resType = op->getResultTypes().front(); // only one result

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        Value lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        Value upperBound;

        // find first tensor operand and use its result type
        upperBound = rewriter.create<memref::DimOp>(loc, sumOpAdaptor.input(), 0);

        // start index for store
        SmallVector<Value> iter_args;

        if (resType.isa<FloatType>())
        {
            iter_args.push_back(
                rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0).getValue(), rewriter.getF64Type()));
        }
        else if (resType.isa<IntegerType>())
        {
            iter_args.push_back(rewriter.create<ConstantIntOp>(loc, 0, rewriter.getI64Type()));
        }
        else
        {
            throw MLIRLoweringError();
        }

        auto sumReductionLoop = rewriter.create<AffineForOp>(
            loc, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
            [&](OpBuilder &nestedBuilder, Location loc, Value iter_var /*index on which to store selected value*/,
                ValueRange ivs) -> void
            {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                Value sumRes = processIteration(nestedBuilder, operands, iter_var, ivs.front());
                nestedBuilder.create<AffineYieldOp>(loc, sumRes);
            });

        rewriter.replaceOp(op, sumReductionLoop->getResult(0));
    }

    LogicalResult
    SumOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs, Value iter_var) -> Value
                       {
                           SumOpAdaptor sumOpAdaptor(memRefOperands);
                           Value values;

                           if (sumOpAdaptor.input().getType().isa<TensorType>())
                           {
                               values = builder.create<memref::BufferCastOp>(
                                   loc, convertTensorToMemRef(sumOpAdaptor.input().getType().dyn_cast<TensorType>()),
                                   sumOpAdaptor.input());
                           }
                           else
                           {
                               values = sumOpAdaptor.input();
                           }

                           if (values.getType().isa<MemRefType>())
                           {
                               // Create the binary operation performed on the loaded values.
                               auto loadedVal = builder.create<AffineLoadOp>(loc, values, loopIvs);
                               if (iter_var.getType().isa<FloatType>())
                               {
                                   return builder.create<AddFOp>(loc, iter_var, loadedVal);
                               }
                               else
                               {
                                   return builder.create<AddIOp>(loc, iter_var, loadedVal);
                               }
                           }
                           else
                           {
                               if (iter_var.getType().isa<FloatType>())
                               {
                                   return builder.create<AddFOp>(loc, iter_var, values);
                               }
                               else
                               {
                                   return builder.create<AddIOp>(loc, iter_var, values);
                               }
                           }
                       });
        return success();
    }
} // namespace voila::mlir::lowering