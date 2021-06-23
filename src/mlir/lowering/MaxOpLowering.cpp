#include "mlir/lowering/MaxOpLowering.hpp"
voila::mlir::lowering::MaxOpLowering::MaxOpLowering(::mlir::MLIRContext *ctx) :
    ConversionPattern(::mlir::voila::MaxOp::getOperationName(), 1, ctx)
{
}

static ::mlir::MemRefType convertTensorToMemRef(::mlir::TensorType type)
{
    assert(type.hasRank() && "expected only ranked shapes");
    return ::mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static void lowerOpToLoops(::mlir::Operation *op,
                           ::mlir::ValueRange operands,
                           ::mlir::PatternRewriter &rewriter,
                           voila::mlir::lowering::MaxOpLowering::LoopIterationFn processIteration)
{
    ::mlir::voila::MaxOpAdaptor maxOpAdaptor(operands);
    auto loc = op->getLoc();

    // Create a nest of affine loops, with one loop per dimension of the shape.
    // The buildAffineLoopNest function takes a callback that is used to construct
    // the body of the innermost loop given a builder, a location and a range of
    // loop induction variables.

    ::mlir::Value lowerBound = rewriter.create<::mlir::ConstantIndexOp>(loc, 1);
    ::mlir::Value upperBound;

    // find first tensor operand and use its result type
    upperBound = rewriter.create<::mlir::memref::DimOp>(loc, maxOpAdaptor.input(), 0);

    // start index for store
    ::mlir::SmallVector<::mlir::Value> iter_args;
    ::mlir::Value values;
    if (maxOpAdaptor.input().getType().isa<::mlir::TensorType>())
    {
        values = rewriter.create<::mlir::memref::BufferCastOp>(
            loc, convertTensorToMemRef(maxOpAdaptor.input().getType().dyn_cast<::mlir::TensorType>()),
            maxOpAdaptor.input());
    }
    else
    {
        values = maxOpAdaptor.input();
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

    auto maxReductionLoop = rewriter.create<::mlir::AffineForOp>(
        loc, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
        [&](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc,
            ::mlir::Value iter_var /*index on which to store selected value*/, ::mlir::ValueRange ivs) -> void
        {
            // Call the processing function with the rewriter, the memref operands,
            // and the loop induction variables. This function will return the value
            // to store at the current index.
            ::mlir::Value maxRes = processIteration(nestedBuilder, operands, iter_var, ivs.front());
            nestedBuilder.create<::mlir::AffineYieldOp>(loc, maxRes);
        });

    rewriter.replaceOp(op, maxReductionLoop->getResult(0));
}

::mlir::LogicalResult
voila::mlir::lowering::MaxOpLowering::matchAndRewrite(::mlir::Operation *op,
                                                      llvm::ArrayRef<::mlir::Value> operands,
                                                      ::mlir::ConversionPatternRewriter &rewriter) const
{
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](::mlir::OpBuilder &builder, ::mlir::ValueRange memRefOperands, ::mlir::ValueRange loopIvs,
              ::mlir::Value iter_var) -> ::mlir::Value
        {
            ::mlir::voila::MaxOpAdaptor maxOpAdaptor(memRefOperands);
            ::mlir::Value values;

            if (maxOpAdaptor.input().getType().isa<::mlir::TensorType>())
            {
                values = builder.create<::mlir::memref::BufferCastOp>(
                    loc, convertTensorToMemRef(maxOpAdaptor.input().getType().dyn_cast<::mlir::TensorType>()),
                    maxOpAdaptor.input());
            }
            else
            {
                values = maxOpAdaptor.input();
            }

            if (values.getType().isa<::mlir::MemRefType>())
            {
                // Create the binary operation performed on the loaded values.
                auto loadedVal = builder.create<::mlir::AffineLoadOp>(loc, values, loopIvs);
                ::mlir::Value cmpRes;
                if (iter_var.getType().isa<::mlir::FloatType>())
                {
                    cmpRes = builder.create<::mlir::CmpFOp>(loc, ::mlir::CmpFPredicate::OGT, iter_var, loadedVal);
                }
                else
                {
                    cmpRes = builder.create<::mlir::CmpIOp>(loc, ::mlir::CmpIPredicate::ugt, iter_var, loadedVal);
                }
                return builder.create<::mlir::SelectOp>(loc, cmpRes, iter_var, loadedVal);
            }
            else
            {
                ::mlir::Value cmpRes;
                if (iter_var.getType().isa<::mlir::FloatType>())
                {
                    cmpRes = builder.create<::mlir::CmpFOp>(loc, ::mlir::CmpFPredicate::OGT, iter_var, values);
                }
                else
                {
                    cmpRes = builder.create<::mlir::CmpIOp>(loc, ::mlir::CmpIPredicate::ugt, iter_var, values);
                }
                return builder.create<::mlir::SelectOp>(loc, cmpRes, iter_var, values);
            }
        });
    return ::mlir::success();
}
