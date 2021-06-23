#include "mlir/lowering/AvgOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::AvgOp;
    using ::mlir::voila::AvgOpAdaptor;

    AvgOpLowering::AvgOpLowering(MLIRContext *ctx) : ConversionPattern(AvgOp::getOperationName(), 1, ctx) {}

    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static void lowerOpToLoops(Operation *op,
                               ValueRange operands,
                               PatternRewriter &rewriter,
                               AvgOpLowering::LoopIterationFn processIteration)
    {
        AvgOpAdaptor avgOpAdaptor(operands);
        auto loc = op->getLoc();
        auto resType = op->getResultTypes().front(); // only one result

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        Value lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        Value upperBound;

        // find first tensor operand and use its result type
        upperBound = rewriter.create<memref::DimOp>(loc, avgOpAdaptor.input(), 0);

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

        // TODO: type conversion?
        rewriter.replaceOpWithNewOp<DivFOp>(op, sumReductionLoop->getResult(0), upperBound);
    }

    LogicalResult
    AvgOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs, Value iter_var) -> Value
                       {
                           AvgOpAdaptor avgOpAdaptor(memRefOperands);
                           Value values;

                           if (avgOpAdaptor.input().getType().isa<TensorType>())
                           {
                               values = builder.create<memref::BufferCastOp>(
                                   loc, convertTensorToMemRef(avgOpAdaptor.input().getType().dyn_cast<TensorType>()),
                                   avgOpAdaptor.input());
                           }
                           else
                           {
                               values = avgOpAdaptor.input();
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