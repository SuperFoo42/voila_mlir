#pragma once
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace voila::mlir::lowering
{
    template <class ConstOp>::mlir::Value convertConstOp(ConstOp op, ::mlir::PatternRewriter &rewriter);

    template <>
    ::mlir::Value convertConstOp<::mlir::voila::IntConstOp>(::mlir::voila::IntConstOp op,
                                                            ::mlir::PatternRewriter &rewriter)
    {
        return rewriter.template create<::mlir::arith::ConstantIntOp>(op.getLoc(), op.getValue(),
                                                                      getElementTypeOrSelf(op));
    }

    template <>
    ::mlir::Value convertConstOp<::mlir::voila::BoolConstOp>(::mlir::voila::BoolConstOp op,
                                                             ::mlir::PatternRewriter &rewriter)
    {
        return convertConstOp(static_cast<::mlir::voila::IntConstOp>(op), rewriter);
    }

    template <>
    ::mlir::Value convertConstOp<::mlir::voila::FltConstOp>(::mlir::voila::FltConstOp op,
                                                            ::mlir::PatternRewriter &rewriter)
    {
        return rewriter.template create<::mlir::arith::ConstantFloatOp>(
            op.getLoc(), dyn_cast<::mlir::FloatAttr>(op.getValueAttr()).getValue(), rewriter.getF64Type());
    }

    template <class ConstOp> class ConstOpLowering : public ::mlir::OpRewritePattern<ConstOp>
    {
      public:
        using ::mlir::OpRewritePattern<ConstOp>::OpRewritePattern;

        ::mlir::LogicalResult matchAndRewrite(ConstOp op, ::mlir::PatternRewriter &rewriter) const final
        {
            auto t = op.getType();
            auto loc = op.getLoc();
            ::mlir::Value cst = convertConstOp(op, rewriter);
            if (isTensor(t))
            {
                auto retT =
                    rewriter.template create<::mlir::tensor::EmptyOp>(loc, getShape(t), getElementTypeOrSelf(t));
                rewriter.template replaceOpWithNewOp<::mlir::linalg::FillOp>(op, cst, retT.getResult());
            }
            else
            {
                rewriter.replaceOp(op, cst);
            }

            return ::mlir::success();
        }
    };

    using IntConstOpLowering = ConstOpLowering<::mlir::voila::IntConstOp>;
    using FltConstOpLowering = ConstOpLowering<::mlir::voila::FltConstOp>;
    using BoolConstOpLowering = ConstOpLowering<::mlir::voila::BoolConstOp>;
} // namespace voila::mlir::lowering