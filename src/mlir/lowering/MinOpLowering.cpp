#include "mlir/lowering/MinOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::MinOp;
    using ::mlir::voila::MinOpAdaptor;

    MinOpLowering::MinOpLowering(MLIRContext *ctx) : ConversionPattern(MinOp::getOperationName(), 1, ctx) {}

    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static void lowerOpToLoops(Operation *op,
                               ValueRange operands,
                               PatternRewriter &rewriter,
                               MinOpLowering::LoopIterationFn processIteration)
    {
        MinOpAdaptor minOpAdaptor(operands);
        auto loc = op->getLoc();

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        Value lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        Value upperBound;

        // find first tensor operand and use its result type
        upperBound = rewriter.create<memref::DimOp>(loc, minOpAdaptor.input(), 0);

        // start index for store
        SmallVector<Value> iter_args;
        ::mlir::Value values;
        if (minOpAdaptor.input().getType().isa<::mlir::TensorType>())
        {
            values = rewriter.create<::mlir::memref::BufferCastOp>(
                loc, convertTensorToMemRef(minOpAdaptor.input().getType().dyn_cast<::mlir::TensorType>()),
                minOpAdaptor.input());
        }
        else
        {
            values = minOpAdaptor.input();
        }

        if (values.getType().isa<::mlir::MemRefType>())
        {
            ::mlir::SmallVector<::mlir::Value> idx;
            idx.push_back(rewriter.create<::mlir::ConstantIndexOp>(loc, 0));
            iter_args.push_back(rewriter.create<::mlir::memref::LoadOp>(loc, values, idx));
        }
        else
        {
            iter_args.push_back(values);
        }

        auto minReductionLoop = rewriter.create<AffineForOp>(
            loc, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
            [&](OpBuilder &nestedBuilder, Location loc, Value iter_var /*index on which to store selected value*/,
                ValueRange ivs) -> void
            {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                Value minRes = processIteration(nestedBuilder, operands, iter_var, ivs.front());
                nestedBuilder.create<AffineYieldOp>(loc, minRes);
            });

        rewriter.replaceOp(op, minReductionLoop->getResult(0));
    }

    LogicalResult
    MinOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs, Value iter_var) -> Value
                       {
                           MinOpAdaptor minOpAdaptor(memRefOperands);
                           Value values;

                           if (minOpAdaptor.input().getType().isa<TensorType>())
                           {
                               values = builder.create<memref::BufferCastOp>(
                                   loc, convertTensorToMemRef(minOpAdaptor.input().getType().dyn_cast<TensorType>()),
                                   minOpAdaptor.input());
                           }
                           else
                           {
                               values = minOpAdaptor.input();
                           }

                           if (values.getType().isa<MemRefType>())
                           {
                               // Create the binary operation performed on the loaded values.
                               auto loadedVal = builder.create<AffineLoadOp>(loc, values, loopIvs);
                               Value cmpRes;
                               if (iter_var.getType().isa<FloatType>())
                               {
                                   cmpRes = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, iter_var, loadedVal);
                               }
                               else
                               {
                                   cmpRes = builder.create<CmpIOp>(loc, CmpIPredicate::ult, iter_var, loadedVal);
                               }
                               return builder.create<SelectOp>(loc, cmpRes, iter_var, loadedVal);
                           }
                           else
                           {
                               Value cmpRes;
                               if (iter_var.getType().isa<FloatType>())
                               {
                                   cmpRes = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, iter_var, values);
                               }
                               else
                               {
                                   cmpRes = builder.create<CmpIOp>(loc, CmpIPredicate::ult, iter_var, values);
                               }
                               return builder.create<SelectOp>(loc, cmpRes, iter_var, values);
                           }
                       });
        return success();
    }
} // namespace voila::mlir::lowering