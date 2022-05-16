#include "mlir/lowering/SumOpLowering.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using ::mlir::voila::SumOp;
    using ::mlir::voila::SumOpAdaptor;

    SumOpLowering::SumOpLowering(MLIRContext *ctx) : ConversionPattern(SumOp::getOperationName(), 1, ctx) {}

    static Value getHTSize(ImplicitLocOpBuilder &builder, Value values)
    {
        auto valType = values.getType().dyn_cast<TensorType>();
        // can calculate ht size from static shape
        if (valType.hasStaticShape())
        {
            auto size = std::bit_ceil<size_t>(valType.getShape().front() + 1);
            assert(size <= std::numeric_limits<int64_t>::max());
            return builder.create<ConstantIndexOp>(size);
        }
        else
        {
            auto insertSize = builder.create<tensor::DimOp>(values, 0);
            /** algorithm to find the next power of 2 taken from
             *  https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
             *
             * v |= v >> 1;
             * v |= v >> 2;
             * v |= v >> 4;
             * v |= v >> 8;
             * v |= v >> 16;
             * v |= v >> 32;
             * v++;
             */
            auto firstOr = builder.create<OrIOp>(
                insertSize, builder.create<ShRUIOp>(insertSize, builder.create<ConstantIndexOp>(1)));
            auto secondOr =
                builder.create<OrIOp>(firstOr, builder.create<ShRUIOp>(firstOr, builder.create<ConstantIndexOp>(2)));
            auto thirdOr =
                builder.create<OrIOp>(secondOr, builder.create<ShRUIOp>(secondOr, builder.create<ConstantIndexOp>(4)));
            auto fourthOr =
                builder.create<OrIOp>(thirdOr, builder.create<ShRUIOp>(thirdOr, builder.create<ConstantIndexOp>(8)));
            auto fithOr =
                builder.create<OrIOp>(fourthOr, builder.create<ShRUIOp>(fourthOr, builder.create<ConstantIndexOp>(16)));
            auto sixthOr =
                builder.create<OrIOp>(fithOr, builder.create<ShRUIOp>(fithOr, builder.create<ConstantIndexOp>(32)));

            return builder.create<AddIOp>(sixthOr, builder.create<ConstantIndexOp>(1));
        }
    }

    static Value scalarSumLowering(Operation *op, SumOpAdaptor &sumOpAdaptor, ImplicitLocOpBuilder &builder)
    {
        SmallVector<int64_t, 1> shape;
        SmallVector<Value, 1> res;

        if (op->getResultTypes().front().isa<IntegerType>())
        {
            res.push_back(builder.create<arith::ConstantOp>(DenseIntElementsAttr::get(
                RankedTensorType::get(shape, builder.getI64Type()), builder.getI64IntegerAttr(0).getValue())));
        }
        else if (op->getResultTypes().front().isa<FloatType>())
        {
            res.push_back(builder.create<arith::ConstantOp>(DenseFPElementsAttr::get(
                RankedTensorType::get(shape, builder.getF64Type()), builder.getF64FloatAttr(0).getValue())));
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }

        SmallVector<Type, 1> res_type;
        res_type.push_back(res.front().getType());

        SmallVector<StringRef, 1> iter_type;
        iter_type.push_back(getReductionIteratorTypeName());

        auto fn = [](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            auto input = vals.front();
            auto output = vals.back();
            Value neutralElem;
            if (input.getType().isa<FloatType>())
                neutralElem =
                    builder.create<ConstantFloatOp>(builder.getF64FloatAttr(0).getValue(), builder.getF64Type());
            else if (input.getType().isa<IntegerType>())
                neutralElem = builder.create<ConstantIntOp>(0, input.getType());
            else
                throw std::logic_error("Not usable with type");
            if (vals.size() == 3) // predicated
            {
                input = builder.create<arith::SelectOp>(vals[1], input, neutralElem);
            }
            ::mlir::Value res;
            if (input.getType().isa<IntegerType>())
                res = builder.create<AddIOp>(input, output);
            else
                res = builder.create<AddFOp>(input, output);

            builder.create<linalg::YieldOp>(res);
        };

        SmallVector<AffineMap, 3> maps;
        if (sumOpAdaptor.pred())
                maps.push_back(builder.getDimIdentityMap());
        SmallVector<AffineExpr, 1> srcExprs;
        srcExprs.push_back(getAffineDimExpr(0, builder.getContext()));
        SmallVector<AffineExpr, 1> dstExprs;
        auto inferred = AffineMap::inferFromExprList({srcExprs, dstExprs});
        maps.insert(maps.end(), inferred.begin(), inferred.end());
        SmallVector<Value, 2> inputs;
        inputs.push_back(sumOpAdaptor.input());
        if (sumOpAdaptor.pred())
            inputs.push_back(sumOpAdaptor.pred());

        auto linalgOp = builder.create<linalg::GenericOp>(builder.getLoc(), /*results*/ res_type,
                                                          /*inputs*/ inputs, /*outputs*/ res,
                                                          /*indexing maps*/ maps,
                                                          /*iterator types*/ iter_type, fn);

        return builder.create<tensor::ExtractOp>(linalgOp->getResult(0));
    }

    static Value groupedSumLowering(Operation *op, SumOpAdaptor &sumOpAdaptor, ImplicitLocOpBuilder &builder)
    {
        Value res;
        auto allocSize = getHTSize(builder,
                                   sumOpAdaptor.input()); // FIXME: not the best solution, indices can be out of range.
        if (getElementTypeOrSelf(op->getResultTypes().front()).isa<IntegerType>())
        {
            res = builder.create<memref::AllocOp>(MemRefType::get(-1, builder.getI64Type()),
                                                  ::llvm::makeArrayRef(allocSize));
            buildAffineLoopNest(builder, builder.getLoc(),
                                ::llvm::makeArrayRef<Value>(builder.create<ConstantIndexOp>(0)), allocSize, {1},
                                [&res](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
                                {
                                    ImplicitLocOpBuilder builder(loc, nestedBuilder);
                                    builder.create<AffineStoreOp>(
                                        builder.create<ConstantIntOp>(0, builder.getI64Type()), res, vals);
                                });
        }
        else if (getElementTypeOrSelf(op->getResultTypes().front()).isa<FloatType>())
        {
            res = builder.create<memref::AllocOp>(MemRefType::get(-1, builder.getF64Type()),
                                                  ::llvm::makeArrayRef(allocSize));
            buildAffineLoopNest(builder, builder.getLoc(),
                                ::llvm::makeArrayRef<Value>(builder.create<ConstantIndexOp>(0)), allocSize, {1},
                                [&res](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
                                {
                                    ImplicitLocOpBuilder builder(loc, nestedBuilder);
                                    builder.create<AffineStoreOp>(
                                        builder.create<ConstantFloatOp>(::llvm::APFloat(0.0), builder.getF64Type()),
                                        res, vals);
                                });
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }

        auto fn = [&res, &sumOpAdaptor](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            auto idx = vals.front();
            if (sumOpAdaptor.pred())
            {
                auto pred = builder.create<tensor::ExtractOp>(sumOpAdaptor.pred(), idx);
                builder.create<scf::IfOp>(pred,
                                          [&](OpBuilder &b, Location loc)
                                          {
                                              ImplicitLocOpBuilder nb(loc, b);
                                              auto toSum = nb.create<tensor::ExtractOp>(sumOpAdaptor.input(), idx);
                                              Value groupIdx =
                                                  nb.create<tensor::ExtractOp>(sumOpAdaptor.indices(), idx);
                                              auto oldVal =
                                                  nb.create<memref::LoadOp>(res, ::llvm::makeArrayRef(groupIdx));
                                              Value newVal;

                                              if (toSum.getType().isa<IntegerType>())
                                              {
                                                  Value tmp = toSum;
                                                  if (toSum.getType() != nb.getI64Type())
                                                  {
                                                      tmp = nb.create<ExtSIOp>(nb.getI64Type(), toSum);
                                                  }
                                                  newVal = nb.create<AddIOp>(oldVal, tmp);
                                              }
                                              else
                                              {
                                                  newVal = nb.create<AddFOp>(oldVal, toSum);
                                              }

                                              nb.create<memref::StoreOp>(newVal, res, groupIdx);
                                              nb.create<scf::YieldOp>();
                                          });
            }
            else
            {
                auto toSum = builder.create<tensor::ExtractOp>(sumOpAdaptor.input(), idx);
                Value groupIdx = builder.create<tensor::ExtractOp>(sumOpAdaptor.indices(), idx);
                auto oldVal = builder.create<memref::LoadOp>(res, ::llvm::makeArrayRef(groupIdx));
                Value newVal;

                if (toSum.getType().isa<IntegerType>())
                {
                    Value tmp = toSum;
                    if (toSum.getType() != builder.getI64Type())
                    {
                        tmp = builder.create<ExtSIOp>(builder.getI64Type(), toSum);
                    }
                    newVal = builder.create<AddIOp>(oldVal, tmp);
                }
                else
                {
                    newVal = builder.create<AddFOp>(oldVal, toSum);
                }

                builder.create<memref::StoreOp>(newVal, res, groupIdx);
            }
        };

        buildAffineLoopNest(builder, builder.getLoc(), ::llvm::makeArrayRef<Value>(builder.create<ConstantIndexOp>(0)),
                            builder.create<tensor::DimOp>(sumOpAdaptor.input(), 0).result(), {1}, fn);

        return builder.create<ToTensorOp>(res);
    }

    LogicalResult
    SumOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        assert(!op->getResultTypes().empty() && op->getResultTypes().size() == 1);
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);
        auto sumOp = dyn_cast<SumOp>(op);
        SumOpAdaptor sumOpAdaptor(sumOp);
        Value res;
        if (sumOp.indices() && op->getResultTypes().front().isa<TensorType>())
        {
            res = groupedSumLowering(op, sumOpAdaptor, builder); // grouped aggregation is a pipeline breaker
        }
        else
        {
            res = scalarSumLowering(op, sumOpAdaptor, builder);
        }
        rewriter.replaceOp(op, res);

        return success();
    }
} // namespace voila::mlir::lowering