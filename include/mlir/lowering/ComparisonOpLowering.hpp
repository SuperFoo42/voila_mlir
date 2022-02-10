#pragma once

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::voila
{
    class EqOp;
    class NeqOp;
    class LeOp;
    class LeqOp;
    class GeqOp;
    class GeOp;
} // namespace mlir::voila

namespace voila::mlir::lowering
{
    template<typename CmpOp>
    class ComparisonOpLowering : public ::mlir::ConversionPattern
    {
        static constexpr auto getIntCmpPred()
        {
            if constexpr (std::is_same_v<CmpOp, ::mlir::voila::EqOp>)
                return ::mlir::arith::CmpIPredicate::eq;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::NeqOp>)
                return ::mlir::arith::CmpIPredicate::ne;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeOp>)
                return ::mlir::arith::CmpIPredicate::slt;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeqOp>)
                return ::mlir::arith::CmpIPredicate::sle;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeqOp>)
                return ::mlir::arith::CmpIPredicate::sge;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeOp>)
                return ::mlir::arith::CmpIPredicate::sgt;
            else
                throw std::logic_error("Sth. went wrong");
        }
        static constexpr auto getFltCmpPred()
        {
            if constexpr (std::is_same_v<CmpOp, ::mlir::voila::EqOp>)
                return ::mlir::arith::CmpFPredicate::OEQ;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::NeqOp>)
                return ::mlir::arith::CmpFPredicate::ONE;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeOp>)
                return ::mlir::arith::CmpFPredicate::OLT;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeqOp>)
                return ::mlir::arith::CmpFPredicate::OLE;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeqOp>)
                return ::mlir::arith::CmpFPredicate::OGE;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeOp>)
                return ::mlir::arith::CmpFPredicate::OGT;
            else
                throw std::logic_error("Sth. went wrong");
        }

        static inline ::mlir::Value
        createTypedCmpOp(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs)
        {
            auto lhsType = ::mlir::getElementTypeOrSelf(lhs);
            auto rhsType = ::mlir::getElementTypeOrSelf(rhs);

            if (lhsType.isa<::mlir::FloatType>() && rhsType.isa<::mlir::FloatType>())
            {
                return builder.create<::mlir::arith::CmpFOp>(getFltCmpPred(), lhs, rhs);
            }
            else if (lhsType.template isa<::mlir::FloatType>())
            {
                auto castedFlt = builder.template create<::mlir::arith::SIToFPOp>(lhs.getType(), rhs);
                return builder.create<::mlir::arith::CmpFOp>(getFltCmpPred(), lhs, castedFlt);
            }
            else if (rhsType.template isa<::mlir::FloatType>())
            {
                auto castedFlt = builder.template create<::mlir::arith::SIToFPOp>(rhs.getType(), lhs);
                return builder.create<::mlir::arith::CmpFOp>(getFltCmpPred(), castedFlt, rhs);
            }
            else
            {
                return builder.create<::mlir::arith::CmpIOp>(getIntCmpPred(), lhs, rhs);
            }
        }

      public:
        [[maybe_unused]] explicit ComparisonOpLowering(::mlir::MLIRContext *ctx) :
            ConversionPattern(CmpOp::getOperationName(), 1, ctx)
        {
        }

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            typename CmpOp::Adaptor opAdaptor(operands);
            auto loc = op->getLoc();
            ::mlir::ImplicitLocOpBuilder builder(loc, rewriter);
            auto lhs = opAdaptor.lhs();
            auto rhs = opAdaptor.rhs();

            if (lhs.getType().template isa<::mlir::TensorType>() && !rhs.getType().template isa<::mlir::TensorType>())
            {
                auto rhsTensor = builder.template create<::mlir::linalg::InitTensorOp>(
                    lhs.getType().template dyn_cast<::mlir::RankedTensorType>().getShape(), rhs.getType());
                rhs = builder.template create<::mlir::linalg::FillOp>(rhs, rhsTensor).result();
            }
            else if (!lhs.getType().template isa<::mlir::TensorType>() &&
                     rhs.getType().template isa<::mlir::TensorType>())
            {
                auto lhsTensor = builder.template create<::mlir::linalg::InitTensorOp>(
                    rhs.getType().template dyn_cast<::mlir::RankedTensorType>().getShape(), lhs.getType());
                lhs = builder.template create<::mlir::linalg::FillOp>(lhs, lhsTensor).result();
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
            rewriter.replaceOp(op, res);
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