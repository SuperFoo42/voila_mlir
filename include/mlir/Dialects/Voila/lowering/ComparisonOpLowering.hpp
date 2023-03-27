#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"

namespace mlir::voila
{
    class EqOp;
    class NeqOp;
    class LeOp;
    class LeqOp;
    class GeqOp;
    class GeOp;
} // namespace mlir::voila

namespace {
    template <class VCmpOp> struct CmpPred {
    };

    template <> struct CmpPred<::mlir::voila::EqOp> {
        static const auto value = ::mlir::arith::CmpFPredicate::OEQ;
    };
    template <> struct CmpPred<::mlir::voila::NeqOp> {
        static const auto value = ::mlir::arith::CmpFPredicate::ONE;
    };
    template <> struct CmpPred<::mlir::voila::LeOp> {
        static const auto value = ::mlir::arith::CmpFPredicate::OLT;
    };
    template <> struct CmpPred<::mlir::voila::LeqOp> {
        static const auto value = ::mlir::arith::CmpFPredicate::OLE;
    };
    template <> struct CmpPred<::mlir::voila::GeqOp> {
        static const auto value = ::mlir::arith::CmpFPredicate::OGE;
    };
    template <> struct CmpPred<::mlir::voila::GeOp> {
        static const auto value = ::mlir::arith::CmpFPredicate::OGT;
    };
    template <class VCmpOp> struct CmpIPred {
    };

    template <> struct CmpIPred<::mlir::voila::EqOp> {
        static const auto value = ::mlir::arith::CmpIPredicate::eq;
    };
    template <> struct CmpIPred<::mlir::voila::NeqOp> {
        static const auto value = ::mlir::arith::CmpIPredicate::ne;
    };
    template <> struct CmpIPred<::mlir::voila::LeOp> {
        static const auto value = ::mlir::arith::CmpIPredicate::slt;
    };
    template <> struct CmpIPred<::mlir::voila::LeqOp> {
        static const auto value = ::mlir::arith::CmpIPredicate::sle;
    };
    template <> struct CmpIPred<::mlir::voila::GeqOp> {
        static const auto value = ::mlir::arith::CmpIPredicate::sge;
    };
    template <> struct CmpIPred<::mlir::voila::GeOp> {
        static const auto value = ::mlir::arith::CmpIPredicate::sgt;
    };

    template <class VCmpOp>
    inline constexpr ::mlir::arith::CmpFPredicate cmp_pred_v = CmpPred<VCmpOp>::value;

    template <class VCmpOp>
    inline constexpr ::mlir::arith::CmpIPredicate cmp_i_pred_v = CmpIPred<VCmpOp>::value;
}

namespace voila::mlir::lowering
{

    template<typename CmpOp>
    class ComparisonOpLowering : public ::mlir::OpConversionPattern<CmpOp>
    {
        static constexpr auto int_pred = cmp_i_pred_v<CmpOp>;
        static constexpr auto flt_pred = cmp_pred_v<CmpOp>;

        static inline ::mlir::Value
        createTypedCmpOp(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs)
        {
            auto lhsType = ::mlir::getElementTypeOrSelf(lhs);
            auto rhsType = ::mlir::getElementTypeOrSelf(rhs);

            if (lhsType.isa<::mlir::FloatType>() && rhsType.isa<::mlir::FloatType>())
            {
                return builder.create<::mlir::arith::CmpFOp>(flt_pred, lhs, rhs);
            }
            else if (lhsType.template isa<::mlir::FloatType>())
            {
                auto castedFlt = builder.template create<::mlir::arith::SIToFPOp>(lhs.getType(), rhs);
                return builder.create<::mlir::arith::CmpFOp>(flt_pred, lhs, castedFlt);
            }
            else if (rhsType.template isa<::mlir::FloatType>())
            {
                auto castedFlt = builder.template create<::mlir::arith::SIToFPOp>(rhs.getType(), lhs);
                return builder.create<::mlir::arith::CmpFOp>(flt_pred, castedFlt, rhs);
            }
            else
            {
                return builder.create<::mlir::arith::CmpIOp>(int_pred, lhs, rhs);
            }
        }

      public:
        using ::mlir::OpConversionPattern<CmpOp>::OpConversionPattern;
        using OpAdaptor =  typename::mlir::OpConversionPattern<CmpOp>::OpAdaptor;

        ::mlir::LogicalResult matchAndRewrite(CmpOp op, OpAdaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            ::mlir::ImplicitLocOpBuilder builder(loc, rewriter);
            auto lhs = op.getLhs();
            auto rhs = op.getRhs();

            if (lhs.getType().template isa<::mlir::TensorType>() && !rhs.getType().template isa<::mlir::TensorType>())
            {
                auto rhsTensor = builder.template create<::mlir::tensor::EmptyOp>(
                    lhs.getType().template dyn_cast<::mlir::RankedTensorType>().getShape(), rhs.getType());
                rhs = builder.template create<::mlir::linalg::FillOp>(rhs, rhsTensor.getResult()).result();
            }
            else if (!lhs.getType().template isa<::mlir::TensorType>() &&
                     rhs.getType().template isa<::mlir::TensorType>())
            {
                auto lhsTensor = builder.template create<::mlir::tensor::EmptyOp>(
                    rhs.getType().template dyn_cast<::mlir::RankedTensorType>().getShape(), lhs.getType());
                lhs = builder.template create<::mlir::linalg::FillOp>(lhs, lhsTensor.getResult()).result();
            }

            if (::mlir::getElementTypeOrSelf(lhs).template isa<::mlir::IndexType>() xor
                ::mlir::getElementTypeOrSelf(rhs).template isa<::mlir::IndexType>())
            {
                if (::mlir::getElementTypeOrSelf(lhs).template isa<::mlir::IndexType>())
                {
                    rhs = builder.template create<::mlir::arith::IndexCastOp>(
                        rhs.getType().template isa<::mlir::TensorType>() ?
                            static_cast<::mlir::Type>(::mlir::RankedTensorType::get(
                                rhs.getType().template dyn_cast<::mlir::TensorType>().getShape(),
                                builder.getIndexType())) :
                            static_cast<::mlir::Type>(builder.getIndexType()),
                        rhs);
                }
                else
                {
                    lhs = builder.template create<::mlir::arith::IndexCastOp>(
                        lhs.getType().template isa<::mlir::TensorType>() ?
                            static_cast<::mlir::Type>(::mlir::RankedTensorType::get(
                                lhs.getType().template dyn_cast<::mlir::TensorType>().getShape(),
                                builder.getIndexType())) :
                            static_cast<::mlir::Type>(builder.getIndexType()),
                        lhs);
                }
            }

            auto res = createTypedCmpOp(builder, lhs, rhs);
            rewriter.replaceOp(op.getOperation(), res);
            return ::mlir::success();
        }
    };

    using EqOpLowering = ComparisonOpLowering<::mlir::voila::EqOp>;
    using NeqOpLowering = ComparisonOpLowering<::mlir::voila::NeqOp>;
    using LeOpLowering = ComparisonOpLowering<::mlir::voila::LeOp>;
    using LeqOpLowering = ComparisonOpLowering<::mlir::voila::LeqOp>;
    using GeOpLowering = ComparisonOpLowering<::mlir::voila::GeOp>;
    using GeqOpLowering = ComparisonOpLowering<::mlir::voila::GeqOp>;
} // namespace voila::mlir::lowering