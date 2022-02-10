#pragma once
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace voila::mlir::lowering
{

    struct BinOpGenerator
    {
        virtual ::mlir::Value
        operator()(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value lhs, ::mlir::Value rhs) const = 0;
        virtual ::mlir::Value operator()(::mlir::OpBuilder &builder,
                                         ::mlir::Location loc,
                                         ::mlir::Value lhs,
                                         ::mlir::Value rhs,
                                         ::mlir::Value pred) const = 0;
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
            auto lhsType = getElementTypeOrSelf(lhs);
            auto rhsType = getElementTypeOrSelf(rhs);

            if (lhsType.isa<::mlir::FloatType>() && rhsType.isa<::mlir::FloatType>())
            {
                return builder.template create<FloatOp>(loc, lhs, rhs);
            }
            else if (lhsType.isa<::mlir::FloatType>())
            {
                auto castedFlt = builder.template create<::mlir::arith::FPToSIOp>(loc, lhsType, rhs);
                return builder.template create<FloatOp>(loc, lhs, castedFlt);
            }
            else if (rhsType.isa<::mlir::FloatType>())
            {
                auto castedFlt = builder.template create<::mlir::arith::FPToSIOp>(loc, rhsType, lhs);
                return builder.template create<FloatOp>(loc, castedFlt, rhs);
            }
            else
            {
                return builder.template create<IntOp>(loc, lhs, rhs);
            }
        }

        ::mlir::Value operator()(::mlir::OpBuilder &builder,
                                 ::mlir::Location loc,
                                 ::mlir::Value lhs,
                                 ::mlir::Value rhs,
                                 ::mlir::Value pred) const override
        {
            ::mlir::ImplicitLocOpBuilder b(loc, builder);
            // sum predicate values to get size of selection
            ::mlir::SmallVector<::mlir::AffineExpr, 2> srcExprs;
            srcExprs.push_back(getAffineDimExpr(0, builder.getContext()));
            ::mlir::SmallVector<::mlir::AffineExpr, 2> dstExprs;
            auto maps = ::mlir::AffineMap::inferFromExprList({srcExprs, dstExprs});
            ::mlir::Value selectionSize =
                b.template create<::mlir::linalg::GenericOp>(
                     llvm::makeArrayRef<::mlir::Type>(b.getIndexType()), llvm::makeArrayRef<::mlir::Value>(pred),
                     llvm::makeArrayRef<::mlir::Value>(b.template create<::mlir::arith::ConstantIndexOp>(0)), maps,
                     llvm::makeArrayRef(::mlir::getReductionIteratorTypeName()),
                     [](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc, ::mlir::ValueRange vals)
                     {
                         auto toAdd = nestedBuilder.create<::mlir::arith::IndexCastOp>(
                             loc, nestedBuilder.getIndexType(), vals[0]);
                         nestedBuilder.create<::mlir::linalg::YieldOp>(
                             loc, llvm::makeArrayRef<::mlir::Value>(
                                      nestedBuilder.create<::mlir::arith::AddIOp>(loc, toAdd, vals[1])));
                     })
                    .getResult(0);
            auto unifiedType = getElementTypeOrSelf(lhs).isa<::mlir::FloatType>() ? getElementTypeOrSelf(lhs) :
                                                                                    getElementTypeOrSelf(rhs);
            ::mlir::Value resTensor = b.template create<::mlir::linalg::InitTensorOp>(selectionSize, unifiedType);

            ::mlir::SmallVector<::mlir::AffineMap, 5> m(4, b.getDimIdentityMap());
            m.push_back(maps.front());
            return b
                .create<::mlir::linalg::GenericOp>(
                    ::mlir::TypeRange{resTensor.getType(), selectionSize.getType()}, ::mlir::ValueRange{lhs, rhs, pred},
                    ::mlir::ValueRange{resTensor, selectionSize}, m,
                    llvm::makeArrayRef(::mlir::getReductionIteratorTypeName()),
                    [&](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc, ::mlir::ValueRange vals)
                    {
                        ::mlir::ImplicitLocOpBuilder b(loc, nestedBuilder);
                        ::mlir::Value lhs = vals[0];
                        ::mlir::Value rhs = vals[1];
                        ::mlir::Value pred = vals[2];
                        ::mlir::Value outBuffer = vals[4];
                        ::mlir::Value idx = vals[3];
                        auto res = b.create<::mlir::scf::IfOp>(
                            pred,
                            [&](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc)
                            {
                                ::mlir::ImplicitLocOpBuilder b(loc, nestedBuilder);
                                auto res = IntFloatBinOpGenerator::operator()(b, loc, lhs, rhs);
                                auto updatedOutBuffer =
                                    b.template create<::mlir::tensor::InsertOp>(res, outBuffer, idx);
                                auto newIdx = b.template create<::mlir::arith::AddIOp>(
                                    idx, b.template create<::mlir::arith::ConstantIndexOp>(1));
                                b.template create<::mlir::scf::YieldOp>(loc,
                                                                        ::mlir::ValueRange({updatedOutBuffer, newIdx}));
                            },
                            [&outBuffer, &idx](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc) {
                                nestedBuilder.template create<::mlir::scf::YieldOp>(
                                    loc, ::mlir::ValueRange({outBuffer, idx}));
                            });
                        b.template create<::mlir::linalg::YieldOp>(res.getResults());
                    })
                .getResult(0);
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

        ::mlir::Value operator()(::mlir::OpBuilder &builder,
                                 ::mlir::Location loc,
                                 ::mlir::Value lhs,
                                 ::mlir::Value rhs,
                                 ::mlir::Value pred) const override
        {
            ::mlir::ImplicitLocOpBuilder b(loc, builder);
            // sum predicate values to get size of selection
            ::mlir::SmallVector<::mlir::AffineExpr, 2> srcExprs;
            srcExprs.push_back(getAffineDimExpr(0, builder.getContext()));
            ::mlir::SmallVector<::mlir::AffineExpr, 2> dstExprs;
            auto maps = ::mlir::AffineMap::inferFromExprList({srcExprs, dstExprs});
            ::mlir::Value selectionSize =
                b.template create<::mlir::linalg::GenericOp>(
                     llvm::makeArrayRef<::mlir::Type>(b.getIndexType()), llvm::makeArrayRef<::mlir::Value>(pred),
                     llvm::makeArrayRef<::mlir::Value>(b.template create<::mlir::arith::ConstantIndexOp>(0)), maps,
                     llvm::makeArrayRef(::mlir::getReductionIteratorTypeName()),
                     [](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc, ::mlir::ValueRange vals)
                     {
                         auto toAdd = nestedBuilder.create<::mlir::arith::IndexCastOp>(
                             loc, nestedBuilder.getIndexType(), vals[0]);
                         nestedBuilder.create<::mlir::linalg::YieldOp>(
                             loc, llvm::makeArrayRef<::mlir::Value>(
                                      nestedBuilder.create<::mlir::arith::AddIOp>(loc, toAdd, vals[1])));
                     })
                    .getResult(0);

            ::mlir::Value resTensor =
                b.template create<::mlir::linalg::InitTensorOp>(selectionSize, getElementTypeOrSelf(lhs));

            ::mlir::SmallVector<::mlir::AffineMap, 5> m(4, b.getDimIdentityMap());
            m.push_back(maps.front());
            return b
                .create<::mlir::linalg::GenericOp>(
                    ::mlir::TypeRange{resTensor.getType(), selectionSize.getType()}, ::mlir::ValueRange{lhs, rhs, pred},
                    ::mlir::ValueRange{resTensor, selectionSize}, m,
                    llvm::makeArrayRef(::mlir::getReductionIteratorTypeName()),
                    [&](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc, ::mlir::ValueRange vals)
                    {
                        ::mlir::ImplicitLocOpBuilder b(loc, nestedBuilder);
                        ::mlir::Value lhs = vals[0];
                        ::mlir::Value rhs = vals[1];
                        ::mlir::Value pred = vals[2];
                        ::mlir::Value outBuffer = vals[4];
                        ::mlir::Value idx = vals[3];
                        auto res = b.create<::mlir::scf::IfOp>(
                            pred,
                            [&](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc)
                            {
                                ::mlir::ImplicitLocOpBuilder b(loc, nestedBuilder);
                                auto res = SingleTypeBinOpGenerator::operator()(b, loc, lhs, rhs);
                                auto updatedOutBuffer =
                                    b.template create<::mlir::tensor::InsertOp>(res, outBuffer, idx);
                                auto newIdx = b.template create<::mlir::arith::AddIOp>(
                                    idx, b.template create<::mlir::arith::ConstantIndexOp>(1));
                                b.template create<::mlir::scf::YieldOp>(loc,
                                                                        ::mlir::ValueRange({updatedOutBuffer, newIdx}));
                            },
                            [&outBuffer, &idx](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc) {
                                nestedBuilder.template create<::mlir::scf::YieldOp>(
                                    loc, ::mlir::ValueRange({outBuffer, idx}));
                            });
                        b.template create<::mlir::linalg::YieldOp>(res.getResults());
                    })
                .getResult(0);
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
            ::mlir::Value newVal;

            if (opAdaptor.lhs().getType().template isa<::mlir::TensorType>() &&
                !opAdaptor.rhs().getType().template isa<::mlir::TensorType>())
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
                newVal = operator()(rewriter, loc, opAdaptor.lhs(), filledOther.result());
            }
            else if (opAdaptor.rhs().getType().template isa<::mlir::TensorType>() &&
                     !opAdaptor.lhs().getType().template isa<::mlir::TensorType>())
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
                newVal = operator()(rewriter, loc, filledOther.result(), opAdaptor.rhs());
            }
            else // no tensors or all tensors as params
            {
                newVal = operator()(rewriter, loc, opAdaptor.lhs(), opAdaptor.rhs());
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