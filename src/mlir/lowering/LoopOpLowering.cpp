#include "mlir/lowering/LoopOpLowering.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::LoopOp;
    using ::mlir::voila::LoopOpAdaptor;

    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    LoopOpLowering::LoopOpLowering(MLIRContext *ctx) : ConversionPattern(LoopOp::getOperationName(), 1, ctx) {}
//TODO: refactor
    static void lowerOpToLoops(Operation *op,
                               ValueRange operands,
                               PatternRewriter &rewriter,
                               LoopOpLowering::LoopIterationFn processIteration)
    {
        LoopOpAdaptor loopOpAdaptor(operands);
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        Value lowerBound = builder.create<ConstantIndexOp>( 1);
        Value upperBound;

        // find first tensor operand and use its result type

        // start index for store
        SmallVector<Value> iter_args;
        Value cond;
        if (loopOpAdaptor.cond().getType().isa<TensorType>())
        {
            cond = builder.create<ToMemrefOp>(
                 convertTensorToMemRef(loopOpAdaptor.cond().getType().dyn_cast<TensorType>()),
                loopOpAdaptor.cond());
            upperBound = builder.create<AddIOp>( builder.create<memref::DimOp>( loopOpAdaptor.cond(), 0),
                                                builder.create<ConstantIndexOp>( 1));
        }
        else
        {
            cond = loopOpAdaptor.cond();
            upperBound = builder.create<ConstantIndexOp>( 2);
        }

        if (cond.getType().isa<MemRefType>())
        {
            SmallVector<Value> idx;
            idx.push_back(builder.create<ConstantIndexOp>( 0));
            iter_args.push_back(builder.create<memref::LoadOp>( cond, idx));
        }
        else
        {
            iter_args.push_back(cond);
        }

        rewriter.replaceOpWithNewOp<AffineForOp>(
            op, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
            [&](OpBuilder &nestedBuilder, Location loc, Value iter_var /*index on which to store selected value*/,
                ValueRange ivs) -> void
            {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                processIteration(rewriter, operands, iter_var, ivs.front());
                // load next cond bit and yield
                if (cond.getType().isa<MemRefType>())
                {
                    SmallVector<Value> loadedCond;
                    loadedCond.push_back(rewriter.create<AffineLoadOp>(
                        loc, cond, ivs)); // FIXME: oob load in last iteration - add affine if
                    rewriter.create<AffineYieldOp>(loc, loadedCond);
                }
                else
                {
                    rewriter.create<AffineYieldOp>(loc, cond);
                }
            });
    }

    LogicalResult
    LoopOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(
            op, operands, rewriter,
            [op, loc](PatternRewriter &builder, ValueRange memRefOperands, ValueRange loopIvs, Value iter_var)
            {
                LoopOpAdaptor loopOpAdaptor(memRefOperands);
                auto ifOp = builder.create<scf::IfOp>(loc, iter_var, false);

                builder.inlineRegionBefore(op->getRegion(0), &ifOp.getThenRegion().back());
                builder.eraseBlock(&ifOp.getThenRegion().back());
                OpBuilder thenBuilder(&ifOp.getThenRegion().back().back());
                thenBuilder.setInsertionPointAfter(&ifOp.getThenRegion().back().back());
                thenBuilder.create<scf::YieldOp>(loc);
            });
        return success();
    }
} // namespace voila::mlir::lowering