#include "mlir/lowering/MaxOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using ::mlir::voila::MaxOp;
    using ::mlir::voila::MaxOpAdaptor;

    MaxOpLowering::MaxOpLowering(MLIRContext *ctx) : ConversionPattern(MaxOp::getOperationName(), 1, ctx) {}

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
            loc, insertSize, rewriter.create<ShRUIOp>(loc, insertSize, rewriter.create<ConstantIndexOp>(loc, 1)));
        auto secondOr = rewriter.create<OrIOp>(
            loc, firstOr, rewriter.create<ShRUIOp>(loc, firstOr, rewriter.create<ConstantIndexOp>(loc, 2)));
        auto thirdOr = rewriter.create<OrIOp>(
            loc, secondOr, rewriter.create<ShRUIOp>(loc, secondOr, rewriter.create<ConstantIndexOp>(loc, 4)));
        auto fourthOr = rewriter.create<OrIOp>(
            loc, thirdOr, rewriter.create<ShRUIOp>(loc, thirdOr, rewriter.create<ConstantIndexOp>(loc, 8)));
        auto fithOr = rewriter.create<OrIOp>(
            loc, fourthOr, rewriter.create<ShRUIOp>(loc, fourthOr, rewriter.create<ConstantIndexOp>(loc, 16)));
        auto sixthOr = rewriter.create<OrIOp>(
            loc, fithOr, rewriter.create<ShRUIOp>(loc, fithOr, rewriter.create<ConstantIndexOp>(loc, 32)));

        return rewriter.create<AddIOp>(loc, sixthOr, rewriter.create<ConstantIndexOp>(loc, 1));
    }

    static Value
    scalarMinLowering(Operation *op, Location loc, MaxOpAdaptor &maxOpAdaptor, ConversionPatternRewriter &rewriter)
    {
        SmallVector<int64_t, 1> shape;
        SmallVector<Value, 1> res;

        if (op->getResultTypes().front().isa<IntegerType>())
        {
            res.push_back(rewriter.create<arith::ConstantOp>(
                loc, DenseIntElementsAttr::get(RankedTensorType::get(shape, rewriter.getI64Type()),
                                               rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min()).getValue())));
        }
        else if (op->getResultTypes().front().isa<FloatType>())
        {
            res.push_back(rewriter.create<arith::ConstantOp>(
                loc,
                DenseFPElementsAttr::get(RankedTensorType::get(shape, rewriter.getF64Type()),
                                         rewriter.getF64FloatAttr(std::numeric_limits<double>::min()).getValue())));
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
            ::mlir::Value maxVal;
            if (vals.front().getType().isa<IntegerType>())
                maxVal = builder.create<MaxSIOp>(loc, vals[0], vals[1]);
            else
                maxVal = builder.create<MaxFOp>(loc, vals[0], vals[1]);

            builder.create<linalg::YieldOp>(loc, maxVal);
        };

        SmallVector<AffineExpr, 2> srcExprs;
        srcExprs.push_back(getAffineDimExpr(0, rewriter.getContext()));
        SmallVector<AffineExpr, 2> dstExprs;
        auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs});

        auto linalgOp = rewriter.create<linalg::GenericOp>(loc, /*results*/ res_type,
                                                           /*inputs*/ maxOpAdaptor.input(), /*outputs*/ res,
                                                           /*indexing maps*/ maps,
                                                           /*iterator types*/ iter_type, fn);
        return rewriter.create<tensor::ExtractOp>(loc, linalgOp->getResult(0));
    }

    static Value
    groupedMaxLowering(Operation *op, Location loc, MaxOpAdaptor &maxOpAdaptor, ConversionPatternRewriter &rewriter)
    {
        Value res;
        auto allocSize = getHTSize(rewriter, loc,
                                   maxOpAdaptor.input()); // FIXME: not the best solution, indices can be out of range.
        if (getElementTypeOrSelf(op->getResultTypes().front()).isa<IntegerType>())
        {
            res = rewriter.create<memref::AllocOp>(loc, MemRefType::get(-1, rewriter.getI64Type()),
                                                   ::llvm::makeArrayRef(allocSize));
            buildAffineLoopNest(rewriter, loc, ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)),
                                allocSize, {1},
                                [&res, &maxOpAdaptor](OpBuilder &builder, Location loc, ValueRange vals)
                                {
                                    builder.create<AffineStoreOp>(
                                        loc,
                                        builder.create<ConstantIntOp>(loc, std::numeric_limits<int64_t>::min(),
                                                                      getElementTypeOrSelf(maxOpAdaptor.input())),
                                        res, vals);
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
                        loc,
                        builder.create<ConstantFloatOp>(loc, ::llvm::APFloat(std::numeric_limits<double>::min()),
                                                        builder.getF64Type()),
                        res, // TODO: any float type
                        vals);
                });
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }

        auto inputMemref = rewriter.create<memref::BufferCastOp>(
            loc, convertTensorToMemRef(maxOpAdaptor.input().getType().dyn_cast<TensorType>()), maxOpAdaptor.input());
        auto indexMemref = rewriter.create<memref::BufferCastOp>(
            loc, convertTensorToMemRef(maxOpAdaptor.indices().getType().dyn_cast<TensorType>()),
            maxOpAdaptor.indices());

        auto fn = [&res, &inputMemref, &indexMemref](OpBuilder &builder, Location loc, ValueRange vals)
        {
            auto idx = vals.front();
            auto toCmp = builder.create<AffineLoadOp>(loc, inputMemref, idx);
            Value groupIdx = builder.create<IndexCastOp>(loc, builder.create<AffineLoadOp>(loc, indexMemref, idx),
                                                         builder.getIndexType());
            auto oldVal = builder.create<memref::LoadOp>(loc, res, ::llvm::makeArrayRef(groupIdx));

            ::mlir::Value maxVal;
            if (toCmp.getType().isa<IntegerType>())
            {
                maxVal = builder.create<MaxSIOp>(loc, toCmp, oldVal);
            }
            else
            {
                maxVal = builder.create<MaxFOp>(loc, toCmp, oldVal);
            }

            builder.create<memref::StoreOp>(loc, maxVal, res, groupIdx);
        };

        buildAffineLoopNest(rewriter, loc, ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)),
                            rewriter.create<tensor::DimOp>(loc, maxOpAdaptor.input(), 0).result(), {1}, fn);

        return rewriter.create<memref::TensorLoadOp>(loc, res);
    }

    LogicalResult
    MaxOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        assert(!op->getResultTypes().empty() && op->getResultTypes().size() == 1);
        auto loc = op->getLoc();
        MaxOpAdaptor minOpAdaptor(operands);
        Value res;

        if (minOpAdaptor.indices() && op->getResultTypes().front().isa<TensorType>())
        {
            res = groupedMaxLowering(op, loc, minOpAdaptor, rewriter);
        }
        else
        {
            res = scalarMinLowering(op, loc, minOpAdaptor, rewriter);
        }

        rewriter.replaceOp(op, res);

        return success();
    }

} // namespace voila::mlir::lowering