#include "mlir/Dialects/Voila/lowering/MinOpLowering.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialects/Voila/lowering/utility/HashingUtils.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using ::mlir::voila::MinOp;
    using ::mlir::voila::MinOpAdaptor;
    using ::voila::mlir::lowering::utils::getHTSize;

    static Value scalarMinLowering(MinOp op, ImplicitLocOpBuilder &builder)
    {
        SmallVector<int64_t, 1> shape;
        SmallVector<Value, 1> res;

        if (op->getResultTypes().front().isa<IntegerType>())
        {
            res.push_back(builder.create<arith::ConstantOp>(

                DenseIntElementsAttr::get(RankedTensorType::get(shape, builder.getI64Type()),
                                          builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max()).getValue())));
        }
        else if (op->getResultTypes().front().isa<FloatType>())
        {
            res.push_back(builder.create<arith::ConstantOp>(

                DenseFPElementsAttr::get(RankedTensorType::get(shape, builder.getF64Type()),
                                         builder.getF64FloatAttr(std::numeric_limits<double>::max()).getValue())));
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }

        SmallVector<Type, 1> res_type;
        res_type.push_back(res.front().getType());

        auto fn = [](OpBuilder &builder, Location loc, ValueRange vals)
        {
            Value input;
            Value output = vals.back();
            if (vals.size() == 3) // predication
            {
                if (input.getType().isa<FloatType>())
                    input = builder.create<ConstantFloatOp>(
                        loc, builder.getF64FloatAttr(std::numeric_limits<double>::max()).getValue(),
                        builder.getF64Type());
                else if (input.getType().isa<IntegerType>())
                    input =
                        builder.create<ConstantIntOp>(loc, std::numeric_limits<int64_t>::max(), builder.getI64Type());
                else
                    std::logic_error("Type not supported");
            }
            else
                input = vals.front();
            ::mlir::Value minVal;
            if (vals.front().getType().isa<IntegerType>())
                minVal = builder.create<MinSIOp>(loc, input, output);
            else
                minVal = builder.create<MinFOp>(loc, input, output);

            builder.create<linalg::YieldOp>(loc, minVal);
        };

        SmallVector<AffineExpr, 2> srcExprs;
        srcExprs.push_back(getAffineDimExpr(0, builder.getContext()));
        SmallVector<AffineExpr, 2> dstExprs;
        auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs});

        auto linalgOp = builder.create<linalg::GenericOp>(/*results*/ res_type,
                                                          /*inputs*/ op.getInput(), /*outputs*/ res,
                                                          /*indexing maps*/ maps,
                                                          /*iterator types*/ ::mlir::utils::IteratorType::reduction, fn);

        return builder.create<tensor::ExtractOp>(linalgOp->getResult(0));
    }

    static Value groupedMinLowering(MinOp op, ImplicitLocOpBuilder &builder)
    {
        Value res;
        auto allocSize =
            getHTSize(builder, op.getInput()); // FIXME: not the best solution, indices can be out of range.
        if (getElementTypeOrSelf(op->getResultTypes().front()).isa<IntegerType>())
        {
            res = builder.create<memref::AllocOp>(MemRefType::get(ShapedType::kDynamic, builder.getI64Type()),
                                                  ArrayRef(allocSize));
            buildAffineLoopNest(builder, builder.getLoc(),
                                builder.create<ConstantIndexOp>(0).getResult(), allocSize, {1},
                                [&res, &op](OpBuilder &builder, Location loc, ValueRange vals)
                                {
                                    builder.create<AffineStoreOp>(
                                        loc,
                                        builder.create<ConstantIntOp>(loc, std::numeric_limits<int64_t>::max(),
                                                                      getElementTypeOrSelf(op.getInput())),
                                        res, vals);
                                });
        }
        else if (getElementTypeOrSelf(op->getResultTypes().front()).isa<FloatType>())
        {
            res = builder.create<memref::AllocOp>(MemRefType::get(ShapedType::kDynamic, builder.getF64Type()),
                                                  ArrayRef(allocSize));
            buildAffineLoopNest(
                builder, builder.getLoc(), builder.create<ConstantIndexOp>(0)->getResults(), allocSize,
                {1},
                [&res](OpBuilder &builder, Location loc, ValueRange vals)
                {
                    ImplicitLocOpBuilder b(loc, builder);
                    b.create<AffineStoreOp>(
                        b.create<ConstantFloatOp>(::llvm::APFloat(std::numeric_limits<double>::max()), b.getF64Type()),
                        res, // TODO: any float type
                        vals);
                });
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }

        auto fn = [&res, &op](OpBuilder &builder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder b(loc, builder);
            auto idx = vals.front();
            if (op.getPred())
            {
                auto pred = b.create<tensor::ExtractOp>(op.getPred(), idx);
                b.create<scf::IfOp>(pred,
                                    [&](OpBuilder &b, Location loc)
                                    {
                                        ImplicitLocOpBuilder nb(loc, b);
                                        auto toCmp = nb.create<tensor::ExtractOp>(op.getInput(), idx);
                                        Value groupIdx = nb.create<tensor::ExtractOp>(op.getIndices(), idx);
                                        auto oldVal = nb.create<memref::LoadOp>(res, groupIdx);

                                        ::mlir::Value minVal;
                                        if (toCmp.getType().isa<IntegerType>())
                                            minVal = nb.create<MinSIOp>(toCmp, oldVal);
                                        else
                                            minVal = nb.create<MinFOp>(toCmp, oldVal);

                                        nb.create<memref::StoreOp>(minVal, res, groupIdx);
                                        nb.create<scf::YieldOp>();
                                    });
            }
            else
            {
                auto toCmp = b.create<tensor::ExtractOp>(op.getInput(), idx);
                Value groupIdx = b.create<tensor::ExtractOp>(op.getIndices(), idx);
                auto oldVal = b.create<memref::LoadOp>(res, groupIdx);

                ::mlir::Value minVal;
                if (toCmp.getType().isa<IntegerType>())
                    minVal = b.create<MinSIOp>(toCmp, oldVal);
                else
                    minVal = b.create<MinFOp>(toCmp, oldVal);

                b.create<memref::StoreOp>(minVal, res, groupIdx);
            }
        };

        buildAffineLoopNest(builder, builder.getLoc(), builder.create<ConstantIndexOp>(0)->getResults(),
                            builder.create<tensor::DimOp>(op.getInput(), 0).getResult(), {1}, fn);

        return builder.create<ToTensorOp>(res);
    }

    LogicalResult
    MinOpLowering::matchAndRewrite(::mlir::voila::MinOp op,
                                   OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const
    {
        assert(!op->getResultTypes().empty() && op->getResultTypes().size() == 1);
        auto loc = op->getLoc();
        Value res;
        ImplicitLocOpBuilder builder(loc, rewriter);

        if (op.getIndices() && op->getResultTypes().front().isa<TensorType>())
        {
            res = groupedMinLowering(op, builder);
        }
        else
        {
            res = scalarMinLowering(op, builder);
        }

        rewriter.replaceOp(op, res);

        return success();
    }
} // namespace voila::mlir::lowering