#include "mlir/lowering/CountOpLowering.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using ::mlir::voila::CountOp;
    using ::mlir::voila::CountOpAdaptor;

    CountOpLowering::CountOpLowering(MLIRContext *ctx) : ConversionPattern(CountOp::getOperationName(), 1, ctx) {}

    static auto convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static Value getHTSize(ConversionPatternRewriter &rewriter, Location loc, Value values)
    {
        auto insertSize = rewriter.create<tensor::DimOp>(loc, values, 0);
        /** algorithm to find the next power of 2 taken from
         *  https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
         *
         * v |= v >> 1;
         * v |= v >> 2;
         * v |= v >> 4;
         * v |= v >> 8;
         * v |= v >> 16;
         * v |= v >> 32;
         * v++;
         */
        auto firstOr = rewriter.create<OrIOp>(
            loc, insertSize,
            rewriter.create<ShRUIOp>(loc, insertSize, rewriter.create<ConstantIndexOp>(loc, 1)));
        auto secondOr = rewriter.create<OrIOp>(
            loc, firstOr,
            rewriter.create<ShRUIOp>(loc, firstOr, rewriter.create<ConstantIndexOp>(loc, 2)));
        auto thirdOr = rewriter.create<OrIOp>(
            loc, secondOr,
            rewriter.create<ShRUIOp>(loc, secondOr, rewriter.create<ConstantIndexOp>(loc, 4)));
        auto fourthOr = rewriter.create<OrIOp>(
            loc, thirdOr,
            rewriter.create<ShRUIOp>(loc, thirdOr, rewriter.create<ConstantIndexOp>(loc, 8)));
        auto fithOr = rewriter.create<OrIOp>(
            loc, fourthOr,
            rewriter.create<ShRUIOp>(loc, fourthOr, rewriter.create<ConstantIndexOp>(loc, 16)));
        auto sixthOr = rewriter.create<OrIOp>(
            loc, fithOr, rewriter.create<ShRUIOp>(loc, fithOr, rewriter.create<ConstantIndexOp>(loc, 32)));

        return rewriter.create<AddIOp>(loc, sixthOr, rewriter.create<ConstantIndexOp>(loc, 1));
    }

    static Value scalarCountLowering(Location loc, CountOpAdaptor &countOpAdaptor, ConversionPatternRewriter &rewriter)
    {
        auto cnt = rewriter.create<tensor::DimOp>(loc, countOpAdaptor.input(), 0);
        return rewriter.create<IndexCastOp>(loc, cnt, rewriter.getI64Type());
    }

    static Value groupedCountLowering(Operation *op,
                                      Location loc,
                                      CountOpAdaptor &countOpAdaptor,
                                      ConversionPatternRewriter &rewriter)
    {
        Value res;
        auto allocSize =
            getHTSize(rewriter, loc,
                      countOpAdaptor.input()); // FIXME: not the best solution, indices can be out of range.

        res = rewriter.create<memref::AllocOp>(loc, MemRefType::get(-1, rewriter.getI64Type()),
                                               ::llvm::makeArrayRef(allocSize));
        buildAffineLoopNest(rewriter, loc, ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)),
                            allocSize, {1},
                            [&res](OpBuilder &builder, Location loc, ValueRange vals) {
                                builder.create<AffineStoreOp>(
                                    loc, builder.create<ConstantIntOp>(loc, 0, builder.getI64Type()), res, vals);
                            });

        auto indexMemref = rewriter.create<memref::BufferCastOp>(
            loc, convertTensorToMemRef(countOpAdaptor.indices().getType().dyn_cast<TensorType>()),
            countOpAdaptor.indices());

        auto fn = [&res, &indexMemref](OpBuilder &builder, Location loc, ValueRange vals)
        {
            auto idx = vals.front();
            Value groupIdx = builder.create<IndexCastOp>(loc, builder.create<AffineLoadOp>(loc, indexMemref, idx),
                                                         builder.getIndexType());
            auto oldVal = builder.create<memref::LoadOp>(loc, res, ::llvm::makeArrayRef(groupIdx));
            Value newVal =
                builder.create<AddIOp>(loc, oldVal, builder.create<ConstantIntOp>(loc, 1, builder.getI64Type()));

            builder.create<memref::StoreOp>(loc, newVal, res, groupIdx);
        };

        buildAffineLoopNest(rewriter, loc, ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)),
                            rewriter.create<tensor::DimOp>(loc, countOpAdaptor.input(), 0).result(), {1}, fn);

        return rewriter.create<memref::TensorLoadOp>(loc, res);
    }

    LogicalResult
    CountOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        CountOpAdaptor cntOpAdaptor(operands);
        Value res;

        if (cntOpAdaptor.indices() && op->getResultTypes().front().isa<TensorType>())
        {
            res = groupedCountLowering(op, loc, cntOpAdaptor, rewriter);
        }
        else
        {
            res = scalarCountLowering(loc, cntOpAdaptor, rewriter);
        }

        rewriter.replaceOp(op, res);

        return success();
    }
} // namespace voila::mlir::lowering