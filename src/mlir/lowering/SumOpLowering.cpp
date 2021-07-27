#include "mlir/lowering/SumOpLowering.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::SumOp;
    using ::mlir::voila::SumOpAdaptor;

    SumOpLowering::SumOpLowering(MLIRContext *ctx) : ConversionPattern(SumOp::getOperationName(), 1, ctx) {}

    LogicalResult
    SumOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        SumOpAdaptor sumOpAdaptor(operands);

        SmallVector<int64_t,1> shape;
        SmallVector<Value, 1> res;

        if (op->getResultTypes().front().isa<IntegerType>())
        {
            auto tmp = rewriter.create<linalg::InitTensorOp>(loc, shape, rewriter.getI64Type());
            res.push_back(rewriter.create<linalg::FillOp>(loc, rewriter.create<ConstantIntOp>(loc, 0, rewriter.getI64Type()), tmp).result());
        }
        else if (op->getResultTypes().front().isa<FloatType>())
        {
            auto tmp = rewriter.create<linalg::InitTensorOp>(loc, shape, rewriter.getF64Type());
            res.push_back(rewriter.create<linalg::FillOp>(loc, rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0).getValue(), rewriter.getF64Type()), tmp).result());
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }


        SmallVector<Type,1> res_type;
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

        rewriter.replaceOp(op, linalgOp->getResults());

        return success();
    }
} // namespace voila::mlir::lowering