#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace voila::mlir::lowering
{

    struct BinOpGenerator
    {
        virtual ::mlir::Value
        operator()(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value lhs, ::mlir::Value rhs) const = 0;

        virtual ~BinOpGenerator() = default;
    };

    template <class IntOp, class FloatOp> struct IntFloatBinOpGenerator : BinOpGenerator
    {
        ::mlir::Value operator()(::mlir::OpBuilder &builder,
                                 ::mlir::Location loc,
                                 ::mlir::Value lhs,
                                 ::mlir::Value rhs) const override
        {
            auto lhsType = getElementTypeOrSelf(lhs);
            auto rhsType = getElementTypeOrSelf(rhs);

            if (isFloat(lhsType) && isFloat(rhsType))
            {
                return builder.template create<FloatOp>(loc, lhs, rhs);
            }
            else if (isFloat(lhsType))
            {
                auto castedFlt = builder.template create<::mlir::arith::FPToSIOp>(loc, lhsType, rhs);
                return builder.template create<FloatOp>(loc, lhs, castedFlt);
            }
            else if (isFloat(rhsType))
            {
                auto castedFlt = builder.template create<::mlir::arith::FPToSIOp>(loc, rhsType, lhs);
                return builder.template create<FloatOp>(loc, castedFlt, rhs);
            }
            else
            {
                return builder.template create<IntOp>(loc, lhs, rhs);
            }
        }
    };

    template <class Op> struct SingleTypeBinOpGenerator : BinOpGenerator
    {
        ::mlir::Value operator()(::mlir::OpBuilder &builder,
                                 ::mlir::Location loc,
                                 ::mlir::Value lhs,
                                 ::mlir::Value rhs) const override
        {
            return builder.template create<Op>(loc, lhs, rhs);
        }
    };

    template <typename BinaryOp, class GenClass>
    class BinaryOpLowering : public ::mlir::OpConversionPattern<BinaryOp>, GenClass
    {
        using GenClass::operator();

      public:
        using ::mlir::OpConversionPattern<BinaryOp>::OpConversionPattern;
        using OpAdaptor = typename ::mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

        ::mlir::LogicalResult
        matchAndRewrite(BinaryOp op, OpAdaptor, ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            ::mlir::Value newVal;

            if (isTensor(op.getLhs()) && !isTensor(op.getRhs()))
            {
                ::mlir::Value other;
                if (hasStaticShape(op.getLhs()))
                {
                    other = rewriter.template create<::mlir::tensor::EmptyOp>(loc, getShape(op.getLhs()),
                                                                              op.getRhs().getType());
                }
                else
                {
                    ::mlir::SmallVector<::mlir::Value, 1> size;
                    size.push_back(rewriter.create<::mlir::tensor::DimOp>(loc, op.getLhs(), 0));
                    other = rewriter.template create<::mlir::tensor::EmptyOp>(loc, ::mlir::ShapedType::kDynamic,
                                                                              op.getRhs().getType(), size);
                }
                auto filledOther = rewriter.create<::mlir::linalg::FillOp>(loc, op.getRhs(), other);
                newVal = operator()(rewriter, loc, op.getLhs(), filledOther.result());
            }
            else if (isTensor(op.getRhs()) && !isTensor(op.getLhs()))
            {
                ::mlir::Value other;
                if (hasStaticShape(op.getRhs()))
                {
                    other = rewriter.template create<::mlir::tensor::EmptyOp>(loc, getShape(op.getRhs()),
                                                                              op.getLhs().getType());
                }
                else
                {
                    ::mlir::SmallVector<::mlir::Value, 1> size;
                    size.push_back(rewriter.create<::mlir::tensor::DimOp>(loc, op.getRhs(), 0));
                    other = rewriter.template create<::mlir::tensor::EmptyOp>(loc, ::mlir::ShapedType::kDynamic,
                                                                              op.getLhs().getType(), size);
                }
                auto filledOther = rewriter.create<::mlir::linalg::FillOp>(loc, op.getLhs(), other);
                newVal = operator()(rewriter, loc, filledOther.result(), op.getRhs());
            }
            else // no tensors or all tensors as params
            {
                newVal = operator()(rewriter, loc, op.getLhs(), op.getRhs());
            }

            // TODO: replace with TypeConverter
            if (op->getResult(0).getType() != newVal.getType())
            {
                ::mlir::Value castRes =
                    rewriter.create<::mlir::tensor::CastOp>(loc, op->getResult(0).getType(), newVal);
                rewriter.replaceOp(op, castRes);
            }
            else
            {
                rewriter.replaceOp(op, newVal);
            }
            return ::mlir::success();
        }
    };

} // namespace voila::mlir::lowering