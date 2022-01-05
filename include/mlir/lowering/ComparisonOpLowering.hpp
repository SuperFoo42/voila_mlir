#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

namespace mlir::voila
{
    class EqOp;
    class NeqOp;
    class LeOp;
    class LeqOp;
    class GeqOp;
    class GeOp;
}

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
        static inline auto isFloat(const ::mlir::Type &t)
        {
            return t.isF64() || t.isF32() || t.isF128() || t.isF80();
        }

        static inline ::mlir::Type getFloatType(const ::mlir::OpBuilder &builder, const ::mlir::Type &t)
        {
            if (t.isF64())
                return ::mlir::Float64Type::get(builder.getContext());
            if (t.isF32())
                return ::mlir::Float32Type::get(builder.getContext());
            if (t.isF128())
                return ::mlir::Float128Type::get(builder.getContext());
            if (t.isF80())
                return ::mlir::Float80Type::get(builder.getContext());
            throw std::logic_error("No float type");
        }

        static inline ::mlir::Value
        createTypedCmpOp(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value lhs, ::mlir::Value rhs)
        {
            ::mlir::Type lhsType, rhsType;
            if (lhs.getType().template isa<::mlir::TensorType>())
            {
                lhsType = lhs.getType().template dyn_cast<::mlir::TensorType>().getElementType();
            }
            else
            {
                lhsType = lhs.getType();
            }

            if (rhs.getType().template isa<::mlir::TensorType>())
            {
                rhsType = rhs.getType().template dyn_cast<::mlir::TensorType>().getElementType();
            }
            else
                rhsType = rhs.getType();

            if (isFloat(lhsType) && isFloat(rhsType))
            {
                return builder.create<::mlir::arith::CmpFOp>(loc, getFltCmpPred(), lhs, rhs);
            }
            else if (isFloat(lhsType))
            {
                auto castedFlt = builder.template create<::mlir::arith::SIToFPOp>(loc, rhs, getFloatType(builder, lhsType));
                return builder.create<::mlir::arith::CmpFOp>(loc, getFltCmpPred(), lhs, castedFlt);
            }
            else if (isFloat(rhsType))
            {
                auto castedFlt = builder.template create<::mlir::arith::SIToFPOp>(loc, lhs, getFloatType(builder, rhsType));
                return builder.create<::mlir::arith::CmpFOp>(loc, getFltCmpPred(), castedFlt, rhs);
            }
            else
            {
                return builder.create<::mlir::arith::CmpIOp>(loc, getIntCmpPred(), lhs, rhs);
            }
        }

      public:
        [[maybe_unused]] explicit ComparisonOpLowering(::mlir::MLIRContext *ctx) : ConversionPattern(CmpOp::getOperationName(), 1, ctx)
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