#include "mlir/Dialects/Voila/lowering/MaxOpLowering.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith//IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/lowering/utility/HashingUtils.hpp"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using ::mlir::voila::MaxOp;
    using ::mlir::voila::MaxOpAdaptor;
    using ::voila::mlir::lowering::utils::getHTSize;

    static Value scalarMaxLowering(MaxOp op, ImplicitLocOpBuilder &builder)
    {
        SmallVector<int64_t, 1> shape;
        SmallVector<Value, 1> res;

        if (isInteger(op->getResult(0)))
        {
            res.push_back(builder.create<arith::ConstantOp>(
                DenseIntElementsAttr::get(RankedTensorType::get(shape, builder.getI64Type()),
                                          builder.getI64IntegerAttr(std::numeric_limits<int64_t>::min()).getValue())));
        }
        else if (isFloat(op->getResult(0)))
        {
            res.push_back(builder.create<arith::ConstantOp>(
                DenseFPElementsAttr::get(RankedTensorType::get(shape, builder.getF64Type()),
                                         builder.getF64FloatAttr(std::numeric_limits<double>::min()).getValue())));
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }

        SmallVector<Type, 1> res_type;
        res_type.push_back(res.front().getType());

        auto fn = [](OpBuilder &builder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder b(loc, builder);
            Value input;
            Value output = vals.back();
            if (vals.size() == 3) // predication
            {
                if (isFloat(input))
                    input = builder.create<ConstantFloatOp>(
                        loc, builder.getF64FloatAttr(std::numeric_limits<double>::min()).getValue(),
                        builder.getF64Type());
                else if (isInteger(input))
                    input =
                        builder.create<ConstantIntOp>(loc, std::numeric_limits<int64_t>::min(), builder.getI64Type());
                else
                    std::logic_error("Type not supported");
            }
            else
                input = vals.front();
            ::mlir::Value maxVal;
            if (isInteger(vals.front()))
                maxVal = b.create<MaxSIOp>(input, output);
            else
                maxVal = b.create<MaximumFOp>(input, output);

            b.create<linalg::YieldOp>(maxVal);
        };

        SmallVector<AffineExpr, 2> srcExprs;
        srcExprs.push_back(getAffineDimExpr(0, builder.getContext()));
        SmallVector<AffineExpr, 2> dstExprs;
        auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs});

        auto linalgOp =
            builder.create<linalg::GenericOp>(builder.getLoc(), /*results*/ res_type,
                                              /*inputs*/ op.getInput(), /*outputs*/ res,
                                              /*indexing maps*/ maps,
                                              /*iterator types*/ ::mlir::utils::IteratorType::reduction, fn);
        return builder.create<tensor::ExtractOp>(linalgOp->getResult(0));
    }

    static Value groupedMaxLowering(MaxOp op, ImplicitLocOpBuilder &rewriter)
    {
        Value res;
        auto allocSize = getHTSize(rewriter,
                                   op.getInput()); // FIXME: not the best solution, indices can be out of range.
        if (isInteger(getElementTypeOrSelf(op->getResult(0))))
        {
            res = rewriter.create<memref::AllocOp>(MemRefType::get(ShapedType::kDynamic, rewriter.getI64Type()),
                                                   ArrayRef(allocSize));
            affine::buildAffineLoopNest(
                rewriter, rewriter.getLoc(), rewriter.create<ConstantIndexOp>(0).getResult(), allocSize, {1},
                [&res, &op](OpBuilder &builder, Location loc, ValueRange vals)
                {
                    ImplicitLocOpBuilder b(loc, builder);
                    b.create<affine::AffineStoreOp>(b.create<ConstantIntOp>(std::numeric_limits<int64_t>::min(),
                                                                    getElementTypeOrSelf(op.getInput())),
                                            res, vals);
                });
        }
        else if (isFloat(getElementTypeOrSelf(op->getResult(0))))
        {
            res = rewriter.create<memref::AllocOp>(MemRefType::get(ShapedType::kDynamic, rewriter.getF64Type()),
                                                   ArrayRef(allocSize));
            affine::buildAffineLoopNest(rewriter, rewriter.getLoc(), rewriter.create<ConstantIndexOp>(0).getResult(), allocSize,
                                {1},
                                [&res](OpBuilder &builder, Location loc, ValueRange vals)
                                {
                                    ImplicitLocOpBuilder b(loc, builder);
                                    b.create<affine::AffineStoreOp>(
                                        b.create<ConstantFloatOp>(::llvm::APFloat(std::numeric_limits<double>::min()),
                                                                  builder.getF64Type()),
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

                                        ::mlir::Value maxVal;
                                        if (isInteger(toCmp))
                                            maxVal = nb.create<MaxSIOp>(toCmp, oldVal);
                                        else
                                            maxVal = nb.create<MaximumFOp>(toCmp, oldVal);

                                        nb.create<memref::StoreOp>(maxVal, res, groupIdx);
                                        nb.create<scf::YieldOp>();
                                    });
            }
            else
            {
                auto toCmp = b.create<tensor::ExtractOp>(op.getInput(), idx);
                Value groupIdx = b.create<tensor::ExtractOp>(op.getIndices(), idx);
                auto oldVal = b.create<memref::LoadOp>(res, groupIdx);

                ::mlir::Value maxVal;
                if (isInteger(toCmp))
                {
                    maxVal = b.create<MaxSIOp>(toCmp, oldVal);
                }
                else
                {
                    maxVal = b.create<MaximumFOp>(toCmp, oldVal);
                }

                b.create<memref::StoreOp>(maxVal, res, groupIdx);
            }
        };

        affine::buildAffineLoopNest(rewriter, rewriter.getLoc(), rewriter.create<ConstantIndexOp>(0).getResult(),
                            rewriter.create<tensor::DimOp>(op.getInput(), 0).getResult(), {1}, fn);

        return rewriter.create<ToTensorOp>(res);
    }

    LogicalResult MaxOpLowering::matchAndRewrite(::mlir::voila::MaxOp op,
                                                 OpAdaptor adaptor,
                                                 ConversionPatternRewriter &rewriter) const
    {
        assert(!op->getResultTypes().empty() && op->getResultTypes().size() == 1);
        auto loc = op->getLoc();
        Value res;

        ImplicitLocOpBuilder builder(loc, rewriter);
        if (op.getIndices() && isTensor(op->getResult(0)))
        {
            res = groupedMaxLowering(op, builder);
        }
        else
        {
            res = scalarMaxLowering(op, builder);
        }

        rewriter.replaceOp(op, res);

        return success();
    }

} // namespace voila::mlir::lowering