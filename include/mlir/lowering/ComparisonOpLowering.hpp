#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

namespace voila::mlir::lowering
{
    template<typename CmpOp>
    class ComparisonOpLowering : public ::mlir::ConversionPattern
    {
        static constexpr auto getIntCmpPred()
        {
            if constexpr (std::is_same_v<CmpOp, ::mlir::voila::EqOp>)
                return ::mlir::CmpIPredicate::eq;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::NeqOp>)
                return ::mlir::CmpIPredicate::ne;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeOp>)
                return ::mlir::CmpIPredicate::slt;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeqOp>)
                return ::mlir::CmpIPredicate::sle;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeqOp>)
                return ::mlir::CmpIPredicate::sge;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeOp>)
                return ::mlir::CmpIPredicate::sgt;
            else
                throw std::logic_error("Sth. went wrong");
        }
        static constexpr auto getFltCmpPred()
        {
            if constexpr (std::is_same_v<CmpOp, ::mlir::voila::EqOp>)
                return ::mlir::CmpFPredicate::OEQ;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::NeqOp>)
                return ::mlir::CmpFPredicate::ONE;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeOp>)
                return ::mlir::CmpFPredicate::OLT;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeqOp>)
                return ::mlir::CmpFPredicate::OLE;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeqOp>)
                return ::mlir::CmpFPredicate::OGE;
            else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeOp>)
                return ::mlir::CmpFPredicate::OGT;
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
            if (isFloat(lhs.getType()) && isFloat(rhs.getType()))
            {
                return builder.create<::mlir::CmpFOp>(loc, builder.getI1Type(), getFltCmpPred(), lhs, rhs);
            }
            else if (isFloat(lhs.getType()))
            {
                auto castedFlt =
                    builder.template create<::mlir::SIToFPOp>(loc, rhs, getFloatType(builder, lhs.getType()));
                return builder.create<::mlir::CmpFOp>(loc, builder.getI1Type(), getFltCmpPred(), lhs, castedFlt);
            }
            else if (isFloat(rhs.getType()))
            {
                auto castedFlt =
                    builder.template create<::mlir::SIToFPOp>(loc, lhs, getFloatType(builder, rhs.getType()));
                return builder.create<::mlir::CmpFOp>(loc, builder.getI1Type(), getFltCmpPred(), castedFlt, rhs);
            }
            else
            {
                return builder.create<::mlir::CmpIOp>(loc, builder.getI1Type(), getIntCmpPred(), lhs, rhs);
            }
        }

      public:
        explicit ComparisonOpLowering(::mlir::MLIRContext *ctx) : ConversionPattern(CmpOp::getOperationName(), 1, ctx)
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
                ::mlir::SmallVector<::mlir::Value, 1> outTensorSize;
                outTensorSize.push_back(rewriter.create<::mlir::tensor::DimOp>(loc, opAdaptor.lhs(), 0));
                auto outTensor =
                    rewriter.create<::mlir::linalg::InitTensorOp>(loc, outTensorSize, rewriter.getI1Type());
                ::mlir::SmallVector<::mlir::Value, 1> res;
                res.push_back(outTensor);

                ::mlir::SmallVector<::mlir::StringRef, 2> iter_type;
                iter_type.push_back("parallel");

                auto fn = [&](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange vals)
                {
                    ::mlir::SmallVector<::mlir::Value, 1> res;
                    res.push_back(createTypedCmpOp(builder, loc, vals[0], vals[1]));

                    builder.create<::mlir::linalg::YieldOp>(loc, res);
                };

                ::mlir::SmallVector<::mlir::Type, 1> ret_type;
                ret_type.push_back(outTensor.getType());
                ::mlir::SmallVector<::mlir::AffineMap, 3> indexing_maps;
                indexing_maps.push_back(rewriter.getDimIdentityMap());
                indexing_maps.push_back(rewriter.getDimIdentityMap());
                indexing_maps.push_back(rewriter.getDimIdentityMap());

                auto linalgOp = rewriter.create<::mlir::linalg::GenericOp>(loc, /*results*/ ret_type,
                                                                           /*inputs*/ operands, /*outputs*/ res,
                                                                           /*indexing maps*/ indexing_maps,
                                                                           /*iterator types*/ iter_type, fn);

                rewriter.replaceOp(op, linalgOp->getResults());
            }
            else if (opAdaptor.lhs().getType().template isa<::mlir::TensorType>())
            {
                ::mlir::SmallVector<::mlir::Value, 1> outTensorSize;
                outTensorSize.push_back(rewriter.create<::mlir::tensor::DimOp>(loc, opAdaptor.lhs(), 0));
                auto outTensor =
                    rewriter.create<::mlir::linalg::InitTensorOp>(loc, outTensorSize, rewriter.getI1Type());
                ::mlir::SmallVector<::mlir::Value, 1> res;
                res.push_back(outTensor);

                ::mlir::SmallVector<::mlir::StringRef, 1> iter_type;
                iter_type.push_back("parallel");

                auto fn = [&](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange vals)
                {
                    ::mlir::SmallVector<::mlir::Value, 1> res;
                    res.push_back(createTypedCmpOp(builder, loc, vals.front(), opAdaptor.rhs()));

                    builder.create<::mlir::linalg::YieldOp>(loc, res);
                };

                ::mlir::SmallVector<::mlir::Type, 1> ret_type;
                ret_type.push_back(outTensor.getType());
                ::mlir::SmallVector<::mlir::AffineMap, 2> indexing_maps;
                indexing_maps.push_back(rewriter.getDimIdentityMap());
                indexing_maps.push_back(rewriter.getDimIdentityMap());
                ::mlir::SmallVector<::mlir::Value, 1> ops;
                ops.push_back(opAdaptor.lhs());

                auto linalgOp = rewriter.create<::mlir::linalg::GenericOp>(loc, /*results*/ ret_type,
                                                                           /*inputs*/ ops, /*outputs*/ res,
                                                                           /*indexing maps*/ indexing_maps,
                                                                           /*iterator types*/ iter_type, fn);

                rewriter.replaceOp(op, linalgOp->getResults());
            }
            else if (opAdaptor.rhs().getType().template isa<::mlir::TensorType>())
            {
                ::mlir::SmallVector<::mlir::Value, 1> outTensorSize;
                outTensorSize.push_back(rewriter.create<::mlir::tensor::DimOp>(loc, opAdaptor.rhs(), 0));
                auto outTensor =
                    rewriter.create<::mlir::linalg::InitTensorOp>(loc, outTensorSize, rewriter.getI1Type());
                ::mlir::SmallVector<::mlir::Value, 1> res;
                res.push_back(outTensor);

                ::mlir::SmallVector<::mlir::StringRef, 1> iter_type;
                iter_type.push_back("parallel");

                auto fn = [&](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange vals)
                {
                    ::mlir::SmallVector<::mlir::Value, 1> res;
                    res.push_back(createTypedCmpOp(builder, loc, opAdaptor.lhs(), vals.front()));

                    builder.create<::mlir::linalg::YieldOp>(loc, res);
                };

                ::mlir::SmallVector<::mlir::Type, 1> ret_type;
                ret_type.push_back(outTensor.getType());
                ::mlir::SmallVector<::mlir::AffineMap, 2> indexing_maps;
                indexing_maps.push_back(rewriter.getDimIdentityMap());
                indexing_maps.push_back(rewriter.getDimIdentityMap());
                ::mlir::SmallVector<::mlir::Value, 1> ops;
                ops.push_back(opAdaptor.rhs());

                auto linalgOp = rewriter.create<::mlir::linalg::GenericOp>(loc, /*results*/ ret_type,
                                                                           /*inputs*/ ops, /*outputs*/ res,
                                                                           /*indexing maps*/ indexing_maps,
                                                                           /*iterator types*/ iter_type, fn);

                rewriter.replaceOp(op, linalgOp->getResults());
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