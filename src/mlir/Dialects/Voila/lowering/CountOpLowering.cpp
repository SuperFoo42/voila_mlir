#include "mlir/Dialects/Voila/lowering/CountOpLowering.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Dialects/Voila/lowering/utility/HashingUtils.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::tensor;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::CountOp;
    using ::voila::mlir::lowering::utils::getHTSize;

    static Value scalarCountLowering(CountOp &op, ImplicitLocOpBuilder &builder)
    {
        if (op.getPred())
        {
            Value start = builder.create<arith::ConstantOp>(DenseIntElementsAttr::get(
                RankedTensorType::get({}, builder.getIndexType()), builder.getIndexAttr(0).getValue()));
            SmallVector<AffineMap, 3> maps(1, builder.getDimIdentityMap());
            SmallVector<AffineExpr, 1> srcExprs;
            srcExprs.push_back(getAffineDimExpr(0, builder.getContext()));
            SmallVector<AffineExpr, 1> dstExprs;
            auto inferred = AffineMap::inferFromExprList({srcExprs, dstExprs});
            return builder
                .create<linalg::GenericOp>(
                    start.getType(), op.getPred(), start, inferred, ::mlir::utils::IteratorType::reduction,
                    [](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
                    {
                        auto indexPred =
                            nestedBuilder.create<arith::IndexCastOp>(loc, nestedBuilder.getIndexType(), vals.front());
                        Value res = nestedBuilder.create<AddIOp>(loc, indexPred, vals.back());
                        nestedBuilder.create<linalg::YieldOp>(loc, res);
                    })
                ->getResult(0);
        }
        else
        {
            auto cnt = builder.create<DimOp>(op.getInput(), 0);
            return cnt;
        }
    }

    static Value groupedCountLowering(CountOp &op, ImplicitLocOpBuilder &rewriter)
    {
        Value res;
        auto allocSize =
            getHTSize(rewriter, op.getInput()); // FIXME: not the best solution, indices can be out of range.

        res = rewriter.create<memref::AllocOp>(MemRefType::get(-1, rewriter.getI64Type()), ArrayRef(allocSize));
        buildAffineLoopNest(rewriter, rewriter.getLoc(), rewriter.create<ConstantIndexOp>(0).getResult(), allocSize,
                            {1},
                            [&res](OpBuilder &builder, Location loc, ValueRange vals) {
                                builder.create<AffineStoreOp>(
                                    loc, builder.create<ConstantIntOp>(loc, 0, builder.getI64Type()), res, vals);
                            });

        auto fn = [&res, &op](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            auto idx = vals.front();
            if (op.getPred())
            {
                auto pred = builder.create<tensor::ExtractOp>(op.getPred(), idx);
                builder.create<scf::IfOp>(pred,
                                          [&](OpBuilder &b, Location loc)
                                          {
                                              ImplicitLocOpBuilder nb(loc, b);
                                              Value groupIdx = nb.create<tensor::ExtractOp>(op.getIndices(), idx);
                                              auto oldVal = nb.create<memref::LoadOp>(res, groupIdx);
                                              Value newVal = nb.create<AddIOp>(
                                                  oldVal, nb.create<ConstantIntOp>(1, builder.getI64Type()));

                                              nb.create<memref::StoreOp>(newVal, res, groupIdx);
                                              nb.create<scf::YieldOp>();
                                          });
            }
            else
            {
                Value groupIdx = builder.create<tensor::ExtractOp>(op.getIndices(), idx);
                auto oldVal = builder.create<memref::LoadOp>(res, groupIdx);
                Value newVal = builder.create<AddIOp>(oldVal, builder.create<ConstantIntOp>(1, builder.getI64Type()));

                builder.create<memref::StoreOp>(newVal, res, groupIdx);
            }
        };

        buildAffineLoopNest(rewriter, rewriter.getLoc(), ValueRange(rewriter.create<ConstantIndexOp>(0).getResult()),
                            rewriter.create<DimOp>(op.getInput(), 0).getResult(), {1}, fn);

        return rewriter.create<ToTensorOp>(res);
    }

    LogicalResult
    CountOpLowering::matchAndRewrite(CountOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        Value res;
        ImplicitLocOpBuilder builder(loc, rewriter);

        if (op.getIndices() && op->getResult(0).getType().isa<TensorType>())
        {
            res = groupedCountLowering(op, builder);
        }
        else
        {
            res = scalarCountLowering(op, builder);
        }

        rewriter.replaceOp(op.getOperation(), res);

        return success();
    }
} // namespace voila::mlir::lowering