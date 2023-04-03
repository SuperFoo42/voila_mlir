#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace voila::mlir::lowering
{

    template <typename iType, typename fType>
    ::mlir::Value createBinOp(::mlir::ConversionPatternRewriter &rewriter,
                          ::mlir::Location loc,
                          ::mlir::Value lhs,
                          ::mlir::Value rhs)
    {
        if (isFloat(lhs))
            return rewriter.create<fType>(loc, lhs, rhs);
        else
            return rewriter.create<iType>(loc, lhs, rhs);
    }

    template <typename SrcOp> struct OpConverter
    {
    };

    template <class Op> class BinaryOpLowering : public ::mlir::OpConversionPattern<Op>
    {
        using BinaryOp = Op;

        constexpr static auto &createOp = OpConverter<BinaryOp>::createOp;

      public:
        using ::mlir::OpConversionPattern<BinaryOp>::OpConversionPattern;
        using OpAdaptor = typename ::mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

        ::mlir::LogicalResult
        matchAndRewrite(BinaryOp op, OpAdaptor, ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();

            auto [lhs, rhs] = canonicalizeValues(rewriter, loc, op.getLhs(), op.getRhs());

            ::mlir::Value newVal = createOp(rewriter,loc, lhs, rhs);

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