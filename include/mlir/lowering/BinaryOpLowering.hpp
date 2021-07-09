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
    template<typename BinaryOp, typename LoweredBinaryOp>
    class BinaryOpLowering : public ::mlir::ConversionPattern
    {
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
                auto outTensor = rewriter.create<::mlir::linalg::InitTensorOp>(
                    loc, outTensorSize, rewriter.getI64Type()); // TODO: float types
                ::mlir::SmallVector<::mlir::Value, 1> res;
                res.push_back(outTensor);

                ::mlir::SmallVector<::mlir::StringRef, 2> iter_type;
                iter_type.push_back("parallel");

                auto fn = [](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange vals)
                {
                    ::mlir::SmallVector<::mlir::Value, 1> res;
                    res.push_back(builder.create<LoweredBinaryOp>(loc, vals[0], vals[1]));

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
                auto outTensor = rewriter.create<::mlir::linalg::InitTensorOp>(
                    loc, outTensorSize, rewriter.getI64Type()); // TODO: float types
                ::mlir::SmallVector<::mlir::Value, 1> res;
                res.push_back(outTensor);

                ::mlir::SmallVector<::mlir::StringRef, 1> iter_type;
                iter_type.push_back("parallel");

                auto fn = [&opAdaptor](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange vals)
                {
                    ::mlir::SmallVector<::mlir::Value, 1> res;
                    res.push_back(builder.create<LoweredBinaryOp>(loc, vals.front(), opAdaptor.rhs()));

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
                auto outTensor = rewriter.create<::mlir::linalg::InitTensorOp>(
                    loc, outTensorSize, rewriter.getI64Type()); // TODO: float types
                ::mlir::SmallVector<::mlir::Value, 1> res;
                res.push_back(outTensor);

                ::mlir::SmallVector<::mlir::StringRef, 1> iter_type;
                iter_type.push_back("parallel");

                auto fn = [&opAdaptor](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange vals)
                {
                    ::mlir::SmallVector<::mlir::Value, 1> res;
                    res.push_back(builder.create<LoweredBinaryOp>(loc, opAdaptor.lhs(), vals.front()));

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
                rewriter.replaceOpWithNewOp<LoweredBinaryOp>(op, opAdaptor.lhs(), opAdaptor.rhs());
            }
            return ::mlir::success();
        }
    };
} // namespace voila::mlir::lowering