#include "mlir/lowering/MaxOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::MaxOp;
    using ::mlir::voila::MaxOpAdaptor;

    MaxOpLowering::MaxOpLowering(MLIRContext *ctx) : ConversionPattern(MaxOp::getOperationName(), 1, ctx) {}

    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static void lowerOpToLoops(Operation *op,
                               ValueRange operands,
                               PatternRewriter &rewriter,
                               MaxOpLowering::LoopIterationFn processIteration)
    {
        MaxOpAdaptor maxOpAdaptor(operands);
        auto loc = op->getLoc();

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        Value lowerBound = rewriter.create<ConstantIndexOp>(loc, 1);
        Value upperBound;

        // find first tensor operand and use its result type
        upperBound = rewriter.create<memref::DimOp>(loc, maxOpAdaptor.input(), 0);

        // start index for store
        SmallVector<Value> iter_args;
        Value values;
        if (maxOpAdaptor.input().getType().isa<TensorType>())
        {
            values = rewriter.create<memref::BufferCastOp>(
                loc, convertTensorToMemRef(maxOpAdaptor.input().getType().dyn_cast<TensorType>()),
                maxOpAdaptor.input());
        }
        else
        {
            values = maxOpAdaptor.input();
        }

        if (values.getType().isa<MemRefType>())
        {
            SmallVector<Value> idx;
            idx.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
            iter_args.push_back(rewriter.create<memref::LoadOp>(loc, values, idx));
        }
        else
        {
            iter_args.push_back(values);
        }

        auto maxReductionLoop = rewriter.create<AffineForOp>(
            loc, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
            [&](OpBuilder &nestedBuilder, Location loc, Value iter_var /*index on which to store selected value*/,
                ValueRange ivs) -> void
            {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                Value maxRes = processIteration(nestedBuilder, operands, iter_var, ivs.front());
                nestedBuilder.create<AffineYieldOp>(loc, maxRes);
            });

        rewriter.replaceOp(op, maxReductionLoop->getResult(0));
    }

    LogicalResult
    MaxOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(
            op, operands, rewriter,
            [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs, Value iter_var) -> ::mlir::Value
            {
                MaxOpAdaptor maxOpAdaptor(memRefOperands);
                Value values;

                if (maxOpAdaptor.input().getType().isa<TensorType>())
                {
                    values = builder.create<memref::BufferCastOp>(
                        loc, convertTensorToMemRef(maxOpAdaptor.input().getType().dyn_cast<TensorType>()),
                        maxOpAdaptor.input());
                }
                else
                {
                    values = maxOpAdaptor.input();
                }

                if (values.getType().isa<MemRefType>())
                {
                    // Create the binary operation performed on the loaded values.
                    auto loadedVal = builder.create<AffineLoadOp>(loc, values, loopIvs);
                    Value cmpRes;
                    if (iter_var.getType().isa<FloatType>())
                    {
                        cmpRes = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, iter_var, loadedVal);
                    }
                    else
                    {
                        cmpRes = builder.create<CmpIOp>(loc, CmpIPredicate::ugt, iter_var, loadedVal);
                    }
                    return builder.create<SelectOp>(loc, cmpRes, iter_var, loadedVal);
                }
                else
                {
                    Value cmpRes;
                    if (iter_var.getType().isa<FloatType>())
                    {
                        cmpRes = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, iter_var, values);
                    }
                    else
                    {
                        cmpRes = builder.create<CmpIOp>(loc, CmpIPredicate::ugt, iter_var, values);
                    }
                    return builder.create<SelectOp>(loc, cmpRes, iter_var, values);
                }
            });
        return success();
    }
} // namespace voila::mlir::lowering