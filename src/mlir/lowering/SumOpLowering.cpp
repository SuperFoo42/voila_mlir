#include "mlir/lowering/SumOpLowering.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::bufferization;
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
    scalarSumLowering(Operation *op, Location loc, SumOpAdaptor &sumOpAdaptor, ConversionPatternRewriter &rewriter)
    {
        SmallVector<int64_t, 1> shape;
        SmallVector<Value, 1> res;

        if (op->getResultTypes().front().isa<IntegerType>())
        {
            res.push_back(rewriter.create<arith::ConstantOp>(
                loc, DenseIntElementsAttr::get(RankedTensorType::get(shape, rewriter.getI64Type()),
                                               rewriter.getI64IntegerAttr(0).getValue())));
        }
        else if (op->getResultTypes().front().isa<FloatType>())
        {
            res.push_back(rewriter.create<arith::ConstantOp>(
                loc, DenseFPElementsAttr::get(RankedTensorType::get(shape, rewriter.getF64Type()),
                                              rewriter.getF64FloatAttr(0).getValue())));
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

        auto fn = [&res, &sumOpAdaptor](OpBuilder &builder, Location loc, ValueRange vals)
        {
            auto idx = vals.front();
            auto toSum = builder.create<tensor::ExtractOp>(loc, sumOpAdaptor.input(), idx);
            Value groupIdx =  builder.create<tensor::ExtractOp>(loc, sumOpAdaptor.indices(), idx);
            auto oldVal = builder.create<memref::LoadOp>(loc, res, ::llvm::makeArrayRef(groupIdx));
            Value newVal;

            if (toSum.getType().isa<IntegerType>())
            {
                Value tmp = toSum;
                if (toSum.getType() != builder.getI64Type())
                {
                    tmp = builder.create<ExtSIOp>(loc, toSum, builder.getI64Type());
                }
                newVal = builder.create<AddIOp>(loc, oldVal, tmp);
            }
            else
            {
                newVal = builder.create<AddFOp>(loc, oldVal, toSum);
            }

            builder.create<memref::StoreOp>(loc, newVal, res, groupIdx);
        };

        buildAffineLoopNest(rewriter, loc, ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)),
                            rewriter.create<tensor::DimOp>(loc, sumOpAdaptor.input(), 0).result(), {1}, fn);

        return rewriter.create<ToTensorOp>(loc, res);
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
            res = groupedSumLowering(op, loc, sumOpAdaptor, rewriter); // grouped aggregation is a pipeline breaker
        }
        else
        {
            res = scalarSumLowering(op, loc, sumOpAdaptor, rewriter);
        }
        rewriter.replaceOp(op, res);

        return success();
    }
} // namespace voila::mlir::lowering