#pragma once
#include "mlir/IR/VoilaOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace voila::mlir::lowering
{
    template<class ConstOp>
    class ConstOpLowering : public ::mlir::OpRewritePattern<ConstOp>
    {
      public:
        using ::mlir::OpRewritePattern<ConstOp>::OpRewritePattern;

        ::mlir::LogicalResult matchAndRewrite(ConstOp op, ::mlir::PatternRewriter &rewriter) const final
        {
            ::mlir::Attribute valAttr = op.valueAttr();
            auto t = op.getType();
            auto loc = op.getLoc();
            if (t.template isa<::mlir::TensorType>())
            {
                auto tt = t.template dyn_cast<::mlir::RankedTensorType>();

                ::mlir::Value cst;
                if constexpr (std::is_same_v<ConstOp, ::mlir::voila::IntConstOp>)
                    cst = rewriter.template create<::mlir::arith::ConstantIntOp>(loc, op.value(), tt.getElementType());
                else
                    cst = rewriter.template create<::mlir::arith::ConstantOp>(loc, valAttr, tt.getElementType());
                auto retT =
                    rewriter.template create<::mlir::linalg::InitTensorOp>(loc, tt.getShape(), tt.getElementType());
                rewriter.template replaceOpWithNewOp<::mlir::linalg::FillOp>(op, cst, retT.result());
            }
            else if constexpr (std::is_same_v<ConstOp, ::mlir::voila::IntConstOp>)
            {
                rewriter.template replaceOpWithNewOp<::mlir::arith::ConstantIntOp>(op, op.value(), t);
            }
            else
            {
                rewriter.template replaceOpWithNewOp<::mlir::arith::ConstantOp>(op, valAttr);
            }

            return ::mlir::success();
        }
    };

    using IntConstOpLowering = ConstOpLowering<::mlir::voila::IntConstOp>;
    using FltConstOpLowering = ConstOpLowering<::mlir::voila::FltConstOp>;
    using BoolConstOpLowering = ConstOpLowering<::mlir::voila::BoolConstOp>;
} // namespace voila::mlir::lowering