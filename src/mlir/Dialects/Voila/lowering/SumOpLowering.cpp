#include "mlir/Dialects/Voila/lowering/SumOpLowering.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "llvm/Support/Casting.h"
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mlir
{
    class OpBuilder;
} // namespace mlir

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using ::mlir::voila::SumOp;
    using ::mlir::voila::SumOpAdaptor;
    using ::voila::mlir::lowering::utils::getHTSize;

    static Value scalarSumLowering(SumOp op, ImplicitLocOpBuilder &builder)
    {
        SmallVector<int64_t, 1> shape;
        SmallVector<Value, 1> res;

        if (isInteger(op->getResult(0)))
        {
            res.push_back(builder.create<arith::ConstantOp>(DenseIntElementsAttr::get(
                RankedTensorType::get(shape, builder.getI64Type()), builder.getI64IntegerAttr(0).getValue())));
        }
        else if (isFloat(op->getResult(0)))
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

        auto fn = [](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            auto input = vals.front();
            auto output = vals.back();
            Value neutralElem;
            if (isFloat(input))
                neutralElem =
                    builder.create<ConstantFloatOp>(builder.getF64FloatAttr(0).getValue(), builder.getF64Type());
            else if (isInteger(input))
                neutralElem = builder.create<ConstantIntOp>(0, input.getType());
            else
                throw std::logic_error("Not usable with type");
            if (vals.size() == 3) // predicated
            {
                input = builder.create<arith::SelectOp>(vals[1], input, neutralElem);
            }
            ::mlir::Value res;
            if (isInteger(input))
                res = builder.create<AddIOp>(input, output);
            else
                res = builder.create<AddFOp>(input, output);

            builder.create<linalg::YieldOp>(res);
        };

        SmallVector<AffineMap, 3> maps;
        if (op.getPred())
            maps.push_back(builder.getDimIdentityMap());
        SmallVector<AffineExpr, 1> srcExprs;
        srcExprs.push_back(getAffineDimExpr(0, builder.getContext()));
        SmallVector<AffineExpr, 1> dstExprs;
        auto inferred = AffineMap::inferFromExprList({srcExprs, dstExprs});
        maps.insert(maps.end(), inferred.begin(), inferred.end());
        SmallVector<Value, 2> inputs;
        inputs.push_back(op.getInput());
        if (op.getPred())
            inputs.push_back(op.getPred());

        auto linalgOp =
            builder.create<linalg::GenericOp>(builder.getLoc(), /*results*/ res_type,
                                              /*inputs*/ inputs, /*outputs*/ res,
                                              /*indexing maps*/ maps,
                                              /*iterator types*/ ::mlir::utils::IteratorType::reduction, fn);

        return builder.create<tensor::ExtractOp>(linalgOp->getResult(0));
    }

    static Value groupedSumLowering(SumOp op, ImplicitLocOpBuilder &builder)
    {
        Value res;
        auto allocSize = getHTSize(builder,
                                   op.getInput()); // FIXME: not the best solution, indices can be out of range.
        if (isInteger(getElementTypeOrSelf(op->getResult(0))))
        {
            res = builder.create<memref::AllocOp>(MemRefType::get(ShapedType::kDynamic, builder.getI64Type()),
                                                  ArrayRef(allocSize));
            buildAffineLoopNest(
                builder, builder.getLoc(), builder.create<ConstantIndexOp>(0).getResult(), allocSize, {1},
                [&res](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
                {
                    ImplicitLocOpBuilder builder(loc, nestedBuilder);
                    builder.create<AffineStoreOp>(builder.create<ConstantIntOp>(0, builder.getI64Type()), res, vals);
                });
        }
        else if (isFloat(getElementTypeOrSelf(op->getResult(0))))
        {
            res = builder.create<memref::AllocOp>(MemRefType::get(ShapedType::kDynamic, builder.getF64Type()),
                                                  ArrayRef(allocSize));
            buildAffineLoopNest(
                builder, builder.getLoc(), builder.create<ConstantIndexOp>(0).getResult(), allocSize, {1},
                [&res](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
                {
                    ImplicitLocOpBuilder builder(loc, nestedBuilder);
                    builder.create<AffineStoreOp>(
                        builder.create<ConstantFloatOp>(::llvm::APFloat(0.0), builder.getF64Type()), res, vals);
                });
        }
        else
        {
            throw std::logic_error("Invalid type"); // TODO
        }

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
                                              auto toSum = nb.create<tensor::ExtractOp>(op.getInput(), idx);
                                              Value groupIdx = nb.create<tensor::ExtractOp>(op.getIndices(), idx);
                                              auto oldVal = nb.create<memref::LoadOp>(res, groupIdx);
                                              Value newVal;

                                              if (isInteger(toSum))
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
                auto toSum = builder.create<tensor::ExtractOp>(op.getInput(), idx);
                Value groupIdx = builder.create<tensor::ExtractOp>(op.getIndices(), idx);
                auto oldVal = builder.create<memref::LoadOp>(res, groupIdx);
                Value newVal;

                if (isInteger(toSum))
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

        buildAffineLoopNest(builder, builder.getLoc(), builder.create<ConstantIndexOp>(0).getResult(),
                            builder.create<tensor::DimOp>(op.getInput(), 0).getResult(), {1}, fn);

        return builder.create<ToTensorOp>(res);
    }

    LogicalResult SumOpLowering::matchAndRewrite(::mlir::voila::SumOp op,
                                                 OpAdaptor adaptor,
                                                 ConversionPatternRewriter &rewriter) const
    {
        assert(!op->getResultTypes().empty() && op->getResultTypes().size() == 1);
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);
        Value res;
        if (op.getIndices() && isTensor(op->getResult(0)))
        {
            res = groupedSumLowering(op, builder); // grouped aggregation is a pipeline breaker
        }
        else
        {
            res = scalarSumLowering(op, builder);
        }
        rewriter.replaceOp(op, res);

        return success();
    }
} // namespace voila::mlir::lowering