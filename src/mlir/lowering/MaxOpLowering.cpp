#include "mlir/lowering/MaxOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::MaxOp;
    using ::mlir::voila::MaxOpAdaptor;

    MaxOpLowering::MaxOpLowering(MLIRContext *ctx) : ConversionPattern(MaxOp::getOperationName(), 1, ctx) {}

    LogicalResult
    MaxOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        MaxOpAdaptor maxOpAdaptor(operands);
        SmallVector<int64_t, 1> shape;
        SmallVector<Value, 1> res;

        if (op->getResultTypes().front().isa<IntegerType>())
        {
            auto tmp = rewriter.create<linalg::InitTensorOp>(loc, shape, rewriter.getI64Type());
            res.push_back(
                rewriter
                    .create<linalg::FillOp>(
                        loc,
                        rewriter.create<ConstantIntOp>(loc, std::numeric_limits<int64_t>::min(), rewriter.getI64Type()),
                        tmp)
                    .result());
        }
        else if (op->getResultTypes().front().isa<FloatType>())
        {
            auto tmp = rewriter.create<linalg::InitTensorOp>(loc, shape, rewriter.getF64Type());
            res.push_back(rewriter
                              .create<linalg::FillOp>(
                                  loc,
                                  rewriter.create<ConstantFloatOp>(
                                      loc, rewriter.getF64FloatAttr(std::numeric_limits<double>::min()).getValue(),
                                      rewriter.getF64Type()),
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
            ::mlir::Value newIsLarger;
            if (vals.front().getType().isa<IntegerType>())
                newIsLarger = builder.create<CmpIOp>(loc, CmpIPredicate::sge, vals[0], vals[1]);
            else
                newIsLarger = builder.create<CmpFOp>(loc, CmpFPredicate::OGE, vals[0], vals[1]);

            builder.create<linalg::YieldOp>(loc, builder.create<SelectOp>(loc, newIsLarger, vals[0], vals[1]).result());
        };

        SmallVector<AffineExpr, 2> srcExprs;
        srcExprs.push_back(getAffineDimExpr(0, rewriter.getContext()));
        SmallVector<AffineExpr, 2> dstExprs;
        auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs});

        auto linalgOp = rewriter.create<linalg::GenericOp>(loc, /*results*/ res_type,
                                                           /*inputs*/ maxOpAdaptor.input(), /*outputs*/ res,
                                                           /*indexing maps*/ maps,
                                                           /*iterator types*/ iter_type, fn);

        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, linalgOp->getResult(0));

        return success();
    }
} // namespace voila::mlir::lowering