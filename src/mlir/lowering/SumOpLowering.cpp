#include "mlir/lowering/SumOpLowering.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::SumOp;
    using ::mlir::voila::SumOpAdaptor;

    SumOpLowering::SumOpLowering(MLIRContext *ctx) : ConversionPattern(SumOp::getOperationName(), 1, ctx) {}

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
        auto firstOr = rewriter.create<OrOp>(
            loc, insertSize,
            rewriter.create<UnsignedShiftRightOp>(loc, insertSize, rewriter.create<ConstantIndexOp>(loc, 1)));
        auto secondOr = rewriter.create<OrOp>(
            loc, firstOr,
            rewriter.create<UnsignedShiftRightOp>(loc, firstOr, rewriter.create<ConstantIndexOp>(loc, 2)));
        auto thirdOr = rewriter.create<OrOp>(
            loc, secondOr,
            rewriter.create<UnsignedShiftRightOp>(loc, secondOr, rewriter.create<ConstantIndexOp>(loc, 4)));
        auto fourthOr = rewriter.create<OrOp>(
            loc, thirdOr,
            rewriter.create<UnsignedShiftRightOp>(loc, thirdOr, rewriter.create<ConstantIndexOp>(loc, 8)));
        auto fithOr = rewriter.create<OrOp>(
            loc, fourthOr,
            rewriter.create<UnsignedShiftRightOp>(loc, fourthOr, rewriter.create<ConstantIndexOp>(loc, 16)));
        auto sixthOr = rewriter.create<OrOp>(
            loc, fithOr, rewriter.create<UnsignedShiftRightOp>(loc, fithOr, rewriter.create<ConstantIndexOp>(loc, 32)));

        return rewriter.create<AddIOp>(loc, sixthOr, rewriter.create<ConstantIndexOp>(loc, 1));
    }

    static Value
    scalarSumLowering(Operation *op, Location loc, SumOpAdaptor &sumOpAdaptor, ConversionPatternRewriter &rewriter)
    {
        SmallVector<int64_t, 1> shape;
        SmallVector<Value, 1> res;

        if (op->getResultTypes().front().isa<IntegerType>())
        {
            auto tmp = rewriter.create<linalg::InitTensorOp>(loc, shape, rewriter.getI64Type());
            res.push_back(
                rewriter.create<linalg::FillOp>(loc, rewriter.create<ConstantIntOp>(loc, 0, rewriter.getI64Type()), tmp)
                    .result());
        }
        else if (op->getResultTypes().front().isa<FloatType>())
        {
            auto tmp = rewriter.create<linalg::InitTensorOp>(loc, shape, rewriter.getF64Type());
            res.push_back(
                rewriter
                    .create<linalg::FillOp>(loc,
                                            rewriter.create<ConstantFloatOp>(
                                                loc, rewriter.getF64FloatAttr(0).getValue(), rewriter.getF64Type()),
                                            tmp)
                    .result());
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }

        SmallVector<Type, 1> res_type;
        res_type.push_back(res.front().getType());

        SmallVector<StringRef, 1> iter_type;
        iter_type.push_back(getReductionIteratorTypeName());

        auto fn = [](OpBuilder &builder, Location loc, ValueRange vals)
        {
            ::mlir::Value res;
            if (vals.front().getType().isa<IntegerType>())
                res = builder.create<AddIOp>(loc, vals);
            else
                res = builder.create<AddFOp>(loc, vals);

            builder.create<linalg::YieldOp>(loc, res);
        };

        SmallVector<AffineExpr, 2> srcExprs;
        srcExprs.push_back(getAffineDimExpr(0, rewriter.getContext()));
        SmallVector<AffineExpr, 2> dstExprs;
        auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs});

        auto linalgOp = rewriter.create<linalg::GenericOp>(loc, /*results*/ res_type,
                                                           /*inputs*/ sumOpAdaptor.input(), /*outputs*/ res,
                                                           /*indexing maps*/ maps,
                                                           /*iterator types*/ iter_type, fn);

        return rewriter.create<tensor::ExtractOp>(loc, linalgOp->getResult(0));
    }

    static auto convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static Value
    groupedSumLowering(Operation *op, Location loc, SumOpAdaptor &sumOpAdaptor, ConversionPatternRewriter &rewriter)
    {
        Value res;
        auto allocSize = getHTSize(rewriter, loc,
                                   sumOpAdaptor.input()); // FIXME: not the best solution, indices can be out of range.
        if (getElementTypeOrSelf(op->getResultTypes().front()).isa<IntegerType>())
        {
            res = rewriter.create<memref::AllocOp>(loc, MemRefType::get(-1, rewriter.getI64Type()),
                                                   ::llvm::makeArrayRef(allocSize));
            buildAffineLoopNest(rewriter, loc, ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)),
                                allocSize, {1},
                                [&res](OpBuilder &builder, Location loc, ValueRange vals) {
                                    builder.create<AffineStoreOp>(
                                        loc, builder.create<ConstantIntOp>(loc, 0, builder.getI64Type()), res, vals);
                                });
        }
        else if (getElementTypeOrSelf(op->getResultTypes().front()).isa<FloatType>())
        {
            res = rewriter.create<memref::AllocOp>(loc, MemRefType::get(-1, rewriter.getF64Type()),
                                                   ::llvm::makeArrayRef(allocSize));
            buildAffineLoopNest(
                rewriter, loc, ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)), allocSize, {1},
                [&res](OpBuilder &builder, Location loc, ValueRange vals)
                {
                    builder.create<AffineStoreOp>(
                        loc, builder.create<ConstantFloatOp>(loc, ::llvm::APFloat(0.0), builder.getF64Type()), res,
                        vals);
                });
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }

        auto inputMemref = rewriter.create<memref::BufferCastOp>(
            loc, convertTensorToMemRef(sumOpAdaptor.input().getType().dyn_cast<TensorType>()), sumOpAdaptor.input());
        auto indexMemref = rewriter.create<memref::BufferCastOp>(
            loc, convertTensorToMemRef(sumOpAdaptor.indices().getType().dyn_cast<TensorType>()),
            sumOpAdaptor.indices());

        auto fn = [&res, &inputMemref, &indexMemref](OpBuilder &builder, Location loc, ValueRange vals)
        {
            auto idx = vals.front();
            auto toSum = builder.create<AffineLoadOp>(loc, inputMemref, idx);
            Value groupIdx = builder.create<IndexCastOp>(loc, builder.create<AffineLoadOp>(loc, indexMemref, idx),
                                                         builder.getIndexType());
            auto oldVal = builder.create<memref::LoadOp>(loc, res, ::llvm::makeArrayRef(groupIdx));
            Value newVal;

            if (toSum.getType().isa<IntegerType>())
            {
                newVal = builder.create<AddIOp>(loc, oldVal, toSum);
            }
            else
            {
                newVal = builder.create<AddFOp>(loc, oldVal, toSum);
            }

            builder.create<memref::StoreOp>(loc, newVal, res, groupIdx);
        };

        buildAffineLoopNest(rewriter, loc, ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)),
                            rewriter.create<tensor::DimOp>(loc, sumOpAdaptor.input(), 0).result(), {1}, fn);

        return rewriter.create<memref::TensorLoadOp>(loc, res);
    }

    LogicalResult
    SumOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        assert(!op->getResultTypes().empty() && op->getResultTypes().size() == 1);
        auto loc = op->getLoc();
        SumOpAdaptor sumOpAdaptor(operands);
        Value res;
        if (sumOpAdaptor.indices() && op->getResultTypes().front().isa<TensorType>())
        {
            res = groupedSumLowering(op, loc, sumOpAdaptor, rewriter);
        }
        else
        {
            res = scalarSumLowering(op, loc, sumOpAdaptor, rewriter);
        }
        rewriter.replaceOp(op, res);

        return success();
    }
} // namespace voila::mlir::lowering