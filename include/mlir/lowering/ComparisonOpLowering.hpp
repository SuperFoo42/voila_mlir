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
        createTypedCmpOp(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value lhs, ::mlir::Value rhs)
        {
            auto lhsType = ::mlir::getElementTypeOrSelf(lhs);
            auto rhsType = ::mlir::getElementTypeOrSelf(rhs);

            if (isFloat(lhsType) && isFloat(rhsType))
            {
                return builder.create<::mlir::arith::CmpFOp>(loc, getFltCmpPred(), lhs, rhs);
            }
            else if (lhsType.template isa<::mlir::FloatType>())
            {
                auto castedFlt = builder.template create<::mlir::arith::SIToFPOp>(loc, rhs, lhs.getType());
                return builder.create<::mlir::arith::CmpFOp>(loc, getFltCmpPred(), lhs, castedFlt);
            }
            else if (rhsType.template isa<::mlir::FloatType>())
            {
                auto castedFlt = builder.template create<::mlir::arith::SIToFPOp>(loc, lhs, rhs.getType());
                return builder.create<::mlir::arith::CmpFOp>(loc, getFltCmpPred(), castedFlt, rhs);
            }
            else
            {
                return builder.create<::mlir::arith::CmpIOp>(loc, getIntCmpPred(), lhs, rhs);
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

            if (opAdaptor.lhs().getType().template isa<::mlir::TensorType>() &&
                opAdaptor.rhs().getType().template isa<::mlir::TensorType>())
            {
                auto res = createTypedCmpOp(rewriter, loc, opAdaptor.lhs(), opAdaptor.rhs());
                rewriter.replaceOp(op, res);
            }
            else if (opAdaptor.lhs().getType().template isa<::mlir::TensorType>())
            {
                auto rhsTensor = rewriter.template create<::mlir::linalg::InitTensorOp>(
                    loc, opAdaptor.lhs().getType().template dyn_cast<::mlir::RankedTensorType>().getShape(),
                    opAdaptor.rhs().getType());
                auto filledTensor = rewriter.template create<::mlir::linalg::FillOp>(loc, opAdaptor.rhs(), rhsTensor);

                auto res = createTypedCmpOp(rewriter, loc, opAdaptor.lhs(), filledTensor.result());
                rewriter.replaceOp(op, res);
            }
            else if (opAdaptor.rhs().getType().template isa<::mlir::TensorType>())
            {
                auto lhsTensor = rewriter.template create<::mlir::linalg::InitTensorOp>(
                    loc, opAdaptor.rhs().getType().template dyn_cast<::mlir::RankedTensorType>().getShape(),
                    opAdaptor.lhs().getType());
                auto filledTensor = rewriter.template create<::mlir::linalg::FillOp>(loc, opAdaptor.lhs(), lhsTensor);
                auto res = createTypedCmpOp(rewriter, loc, filledTensor.result(), opAdaptor.rhs());
                rewriter.replaceOp(op, res);
            }
            else // no tensors as params
            {
                auto res = createTypedCmpOp(rewriter, loc, opAdaptor.lhs(), opAdaptor.rhs());
                rewriter.replaceOp(op, res);
            }
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