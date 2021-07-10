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
            if (isFloat(lhs.getType()) && isFloat(rhs.getType()))
            {
                return builder.template create<FloatOp>(loc, lhs, rhs);
            }
            else if (isFloat(lhs.getType()))
            {
                auto castedFlt =
                    builder.template create<::mlir::SIToFPOp>(loc, rhs, getFloatType(builder, lhs.getType()));
                return builder.template create<FloatOp>(loc, lhs, castedFlt);
            }
            else if (isFloat(rhs.getType()))
            {
                auto castedFlt =
                    builder.template create<::mlir::SIToFPOp>(loc, lhs, getFloatType(builder, rhs.getType()));
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
                ::mlir::SmallVector<::mlir::Value, 1> outTensorSize;
                outTensorSize.push_back(rewriter.create<::mlir::tensor::DimOp>(loc, opAdaptor.lhs(), 0));
                ::mlir::Type outTensorType;
                if (isFloat(opAdaptor.lhs().getType().template dyn_cast<::mlir::TensorType>().getElementType()))
                    outTensorType = opAdaptor.lhs().getType().template dyn_cast<::mlir::TensorType>().getElementType();
                else
                    outTensorType = opAdaptor.rhs().getType().template dyn_cast<::mlir::TensorType>().getElementType();
                auto outTensor = rewriter.create<::mlir::linalg::InitTensorOp>(loc, outTensorSize, outTensorType);
                ::mlir::SmallVector<::mlir::Value, 1> res;
                res.push_back(outTensor);

                ::mlir::SmallVector<::mlir::StringRef, 2> iter_type;
                iter_type.push_back("parallel");

                auto fn = [&](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange vals)
                {
                    ::mlir::SmallVector<::mlir::Value, 1> res;
                    res.push_back(operator()(builder, loc, vals[0], vals[1]));

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
                ::mlir::Type outTensorType;
                if (isFloat(opAdaptor.lhs().getType().template dyn_cast<::mlir::TensorType>().getElementType()))
                    outTensorType = opAdaptor.lhs().getType().template dyn_cast<::mlir::TensorType>().getElementType();
                else
                    outTensorType = opAdaptor.rhs().getType();
                auto outTensor = rewriter.create<::mlir::linalg::InitTensorOp>(loc, outTensorSize, outTensorType);
                ::mlir::SmallVector<::mlir::Value, 1> res;
                res.push_back(outTensor);

                ::mlir::SmallVector<::mlir::StringRef, 1> iter_type;
                iter_type.push_back("parallel");

                auto fn = [&](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange vals)
                {
                    ::mlir::SmallVector<::mlir::Value, 1> res;
                    res.push_back(operator()(builder, loc, vals.front(), opAdaptor.rhs()));

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
                ::mlir::Type outTensorType;
                if (isFloat(opAdaptor.lhs().getType()))
                    outTensorType = opAdaptor.lhs().getType();
                else
                    outTensorType = opAdaptor.rhs().getType().template dyn_cast<::mlir::TensorType>().getElementType();
                auto outTensor = rewriter.create<::mlir::linalg::InitTensorOp>(loc, outTensorSize, outTensorType);
                ::mlir::SmallVector<::mlir::Value, 1> res;
                res.push_back(outTensor);

                ::mlir::SmallVector<::mlir::StringRef, 1> iter_type;
                iter_type.push_back("parallel");

                auto fn = [&](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange vals)
                {
                    ::mlir::SmallVector<::mlir::Value, 1> res;
                    res.push_back(operator()(builder, loc, opAdaptor.lhs(), vals.front()));

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
                auto res = operator()(rewriter, loc, opAdaptor.lhs(), opAdaptor.rhs());
                rewriter.replaceOp(op, res);
            }
            return ::mlir::success();
        }
    };

} // namespace voila::mlir::lowering