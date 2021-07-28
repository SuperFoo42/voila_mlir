#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"
namespace voila::mlir::lowering
{
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

    struct BinOpGenerator
    {
        virtual ::mlir::Value
        operator()(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value lhs, ::mlir::Value rhs) const = 0;
        virtual ~BinOpGenerator() = default;
    };

    template<class IntOp, class FloatOp>
    struct IntFloatBinOpGenerator : BinOpGenerator
    {
        ::mlir::Value operator()(::mlir::OpBuilder &builder,
                                 ::mlir::Location loc,
                                 ::mlir::Value lhs,
                                 ::mlir::Value rhs) const override
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
            {
                rhsType = rhs.getType();
            }

            if (isFloat(lhsType) && isFloat(rhsType))
            {
                return builder.template create<FloatOp>(loc, lhs, rhs);
            }
            else if (isFloat(lhsType))
            {
                auto castedFlt = builder.template create<::mlir::SIToFPOp>(loc, rhs, lhsType);
                return builder.template create<FloatOp>(loc, lhs, castedFlt);
            }
            else if (isFloat(rhsType))
            {
                auto castedFlt = builder.template create<::mlir::SIToFPOp>(loc, lhs, rhsType);
                return builder.template create<FloatOp>(loc, castedFlt, rhs);
            }
            else
            {
                return builder.template create<IntOp>(loc, lhs, rhs);
            }
        }
    };

    template<class Op>
    struct SingleTypeBinOpGenerator : BinOpGenerator
    {
        ::mlir::Value operator()(::mlir::OpBuilder &builder,
                                 ::mlir::Location loc,
                                 ::mlir::Value lhs,
                                 ::mlir::Value rhs) const override
        {
            return builder.template create<Op>(loc, lhs, rhs);
        }
    };

    template<typename BinaryOp, class GenClass>
    class BinaryOpLowering : public ::mlir::ConversionPattern, GenClass
    {
        using GenClass::operator();

      public:
        explicit BinaryOpLowering(::mlir::MLIRContext *ctx) : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            typename BinaryOp::Adaptor opAdaptor(operands);
            auto loc = op->getLoc();

            if (opAdaptor.lhs().getType().template isa<::mlir::TensorType>() &&
                opAdaptor.rhs().getType().template isa<::mlir::TensorType>())
            {
                rewriter.replaceOp(op, operator()(rewriter, loc, opAdaptor.lhs(), opAdaptor.rhs()));
            }
            else if (opAdaptor.lhs().getType().template isa<::mlir::TensorType>())
            {
                ::mlir::Value other;
                if (opAdaptor.lhs().getType().template dyn_cast<::mlir::RankedTensorType>().hasStaticShape())
                {
                    other = rewriter.template create<::mlir::linalg::InitTensorOp>(
                        loc, opAdaptor.lhs().getType().template dyn_cast<::mlir::RankedTensorType>().getShape(),
                        opAdaptor.rhs().getType());
                }
                else
                {
                    ::mlir::SmallVector<::mlir::Value, 1> size;
                    size.push_back(rewriter.create<::mlir::tensor::DimOp>(loc, opAdaptor.lhs(), 0));
                    other =
                        rewriter.template create<::mlir::linalg::InitTensorOp>(loc, size, opAdaptor.rhs().getType());
                }
                auto filledOther = rewriter.create<::mlir::linalg::FillOp>(loc, opAdaptor.rhs(), other);
                rewriter.replaceOp(op, operator()(rewriter, loc, opAdaptor.lhs(), filledOther.result()));
            }
            else if (opAdaptor.rhs().getType().template isa<::mlir::TensorType>())
            {
                ::mlir::Value other;
                if (opAdaptor.rhs().getType().template dyn_cast<::mlir::RankedTensorType>().hasStaticShape())
                {
                    other = rewriter.template create<::mlir::linalg::InitTensorOp>(
                        loc, opAdaptor.rhs().getType().template dyn_cast<::mlir::RankedTensorType>().getShape(),
                        opAdaptor.lhs().getType());
                }
                else
                {
                    ::mlir::SmallVector<::mlir::Value, 1> size;
                    size.push_back(rewriter.create<::mlir::tensor::DimOp>(loc, opAdaptor.rhs(), 0));
                    other =
                        rewriter.template create<::mlir::linalg::InitTensorOp>(loc, size, opAdaptor.lhs().getType());
                }
                auto filledOther = rewriter.create<::mlir::linalg::FillOp>(loc, opAdaptor.lhs(), other);
                rewriter.replaceOp(op, operator()(rewriter, loc, filledOther.result(), opAdaptor.rhs()));
            }
            else // no tensors as params
            {
                rewriter.replaceOp(op, operator()(rewriter, loc, opAdaptor.lhs(), opAdaptor.rhs()));
            }
            return ::mlir::success();
        }
    };

} // namespace voila::mlir::lowering