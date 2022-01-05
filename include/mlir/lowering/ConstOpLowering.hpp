#pragma once
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    template<class ConstOp>
    class ConstOpLowering : public ::mlir::OpRewritePattern<ConstOp>
    {
      public:
        using ::mlir::OpRewritePattern<ConstOp>::OpRewritePattern;

        ::mlir::LogicalResult matchAndRewrite(ConstOp op, ::mlir::PatternRewriter &rewriter) const final
        {
            auto constantValue = op.value();

            ::mlir::Attribute valAttr;
            if constexpr (std::is_same_v<ConstOp, ::mlir::voila::IntConstOp>)
            {
                if (std::numeric_limits<uint_least32_t>::max() >= constantValue)
                    valAttr = rewriter.getI32IntegerAttr(constantValue);
                else
                    valAttr = rewriter.getI64IntegerAttr(constantValue);
            }
            else if constexpr (std::is_same_v<ConstOp, ::mlir::voila::FltConstOp>)
            {
                valAttr = rewriter.getF64FloatAttr(constantValue.convertToDouble());
            }
            else if constexpr (std::is_same_v<ConstOp, ::mlir::voila::BoolConstOp>)
            {
                valAttr = ::mlir::IntegerAttr::get(rewriter.getI1Type(), constantValue);
            }
            else
            {
                return ::mlir::failure();
            }

            rewriter.template replaceOpWithNewOp<::mlir::arith::ConstantOp>(op, valAttr);

            return ::mlir::success();
        }
    };

    using IntConstOpLowering = ConstOpLowering<::mlir::voila::IntConstOp>;
    using FltConstOpLowering = ConstOpLowering<::mlir::voila::FltConstOp>;
    using BoolConstOpLowering = ConstOpLowering<::mlir::voila::BoolConstOp>;
} // namespace voila::mlir::lowering