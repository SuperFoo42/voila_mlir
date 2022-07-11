#include "mlir/lowering/MaxOpLowering.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic//IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/VoilaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace voila::mlir::lowering {
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using ::mlir::voila::MaxOp;
    using ::mlir::voila::MaxOpAdaptor;

    MaxOpLowering::MaxOpLowering(MLIRContext *ctx) : ConversionPattern(MaxOp::getOperationName(), 1, ctx) {}

    static Value getHTSize(ImplicitLocOpBuilder &builder, Value values) {
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
        auto firstOr =
                builder.create<OrIOp>(insertSize,
                                      builder.create<ShRUIOp>(insertSize, builder.create<ConstantIndexOp>(1)));
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

    static Value scalarMaxLowering(Operation *op, MaxOpAdaptor &maxOpAdaptor, ImplicitLocOpBuilder &builder) {
        SmallVector<int64_t, 1> shape;
        SmallVector<Value, 1> res;

        if (op->getResultTypes().front().isa<IntegerType>()) {
            res.push_back(builder.create<arith::ConstantOp>(
                    DenseIntElementsAttr::get(RankedTensorType::get(shape, builder.getI64Type()),
                                              builder.getI64IntegerAttr(
                                                      std::numeric_limits<int64_t>::min()).getValue())));
        } else if (op->getResultTypes().front().isa<FloatType>()) {
            res.push_back(builder.create<arith::ConstantOp>(
                    DenseFPElementsAttr::get(RankedTensorType::get(shape, builder.getF64Type()),
                                             builder.getF64FloatAttr(std::numeric_limits<double>::min()).getValue())));
        } else {
            throw std::logic_error("Invalid type"); // TODO
        }

        SmallVector<Type, 1> res_type;
        res_type.push_back(res.front().getType());

        SmallVector<StringRef, 1> iter_type;
        iter_type.push_back(getReductionIteratorTypeName());

        auto fn = [](OpBuilder &builder, Location loc, ValueRange vals) {
            ImplicitLocOpBuilder b(loc, builder);
            Value input;
            Value output = vals.back();
            if (vals.size() == 3) // predication
            {
                if (input.getType().isa<FloatType>())
                    input = builder.create<ConstantFloatOp>(
                            loc, builder.getF64FloatAttr(std::numeric_limits<double>::min()).getValue(),
                            builder.getF64Type());
                else if (input.getType().isa<IntegerType>())
                    input =
                            builder.create<ConstantIntOp>(loc, std::numeric_limits<int64_t>::min(),
                                                          builder.getI64Type());
                else
                    std::logic_error("Type not supported");
            } else
                input = vals.front();
            ::mlir::Value maxVal;
            if (vals.front().getType().isa<IntegerType>())
                maxVal = b.create<MaxSIOp>(input, output);
            else
                maxVal = b.create<MaxFOp>(input, output);

            b.create<linalg::YieldOp>(maxVal);
        };

        SmallVector<AffineExpr, 2> srcExprs;
        srcExprs.push_back(getAffineDimExpr(0, builder.getContext()));
        SmallVector<AffineExpr, 2> dstExprs;
        auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs});

        auto linalgOp = builder.create<linalg::GenericOp>(builder.getLoc(), /*results*/ res_type,
                /*inputs*/ maxOpAdaptor.input(), /*outputs*/ res,
                /*indexing maps*/ maps,
                /*iterator types*/ iter_type, fn);
        return builder.create<tensor::ExtractOp>(linalgOp->getResult(0));
    }

    static Value groupedMaxLowering(Operation *op, MaxOpAdaptor &maxOpAdaptor, ImplicitLocOpBuilder &rewriter) {
        Value res;
        auto allocSize = getHTSize(rewriter,
                                   maxOpAdaptor.input()); // FIXME: not the best solution, indices can be out of range.
        if (getElementTypeOrSelf(op->getResultTypes().front()).isa<IntegerType>()) {
            res = rewriter.create<memref::AllocOp>(MemRefType::get(-1, rewriter.getI64Type()),
                                                   ::llvm::makeArrayRef(allocSize));
            buildAffineLoopNest(rewriter, rewriter.getLoc(),
                                ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(0)), allocSize, {1},
                                [&res, &maxOpAdaptor](OpBuilder &builder, Location loc, ValueRange vals) {
                                    ImplicitLocOpBuilder b(loc, builder);
                                    b.create<AffineStoreOp>(
                                            b.create<ConstantIntOp>(std::numeric_limits<int64_t>::min(),
                                                                    getElementTypeOrSelf(maxOpAdaptor.input())),
                                            res, vals);
                                });
        } else if (getElementTypeOrSelf(op->getResultTypes().front()).isa<FloatType>()) {
            res = rewriter.create<memref::AllocOp>(MemRefType::get(-1, rewriter.getF64Type()),
                                                   ::llvm::makeArrayRef(allocSize));
            buildAffineLoopNest(rewriter, rewriter.getLoc(),
                                ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(0)), allocSize, {1},
                                [&res](OpBuilder &builder, Location loc, ValueRange vals) {
                                    ImplicitLocOpBuilder b(loc, builder);
                                    b.create<AffineStoreOp>(
                                            b.create<ConstantFloatOp>(
                                                    ::llvm::APFloat(std::numeric_limits<double>::min()),
                                                    builder.getF64Type()),
                                            res, // TODO: any float type
                                            vals);
                                });
        } else {
            throw std::logic_error("Invalid type"); // TODO
        }
        auto fn = [&res, &maxOpAdaptor](OpBuilder &builder, Location loc, ValueRange vals) {
            ImplicitLocOpBuilder b(loc, builder);
            auto idx = vals.front();
            if (maxOpAdaptor.pred()) {
                auto pred = b.create<tensor::ExtractOp>(maxOpAdaptor.pred(), idx);
                b.create<scf::IfOp>(pred,
                                    [&](OpBuilder &b, Location loc) {
                                        ImplicitLocOpBuilder nb(loc, b);
                                        auto toCmp = nb.create<tensor::ExtractOp>(maxOpAdaptor.input(), idx);
                                        Value groupIdx = nb.create<tensor::ExtractOp>(maxOpAdaptor.indices(), idx);
                                        auto oldVal = nb.create<memref::LoadOp>(res, ::llvm::makeArrayRef(groupIdx));

                                        ::mlir::Value maxVal;
                                        if (toCmp.getType().isa<IntegerType>())
                                            maxVal = nb.create<MaxSIOp>(toCmp, oldVal);
                                        else
                                            maxVal = nb.create<MaxFOp>(toCmp, oldVal);

                                        nb.create<memref::StoreOp>(maxVal, res, groupIdx);
                                        nb.create<scf::YieldOp>();
                                    });
            } else {
                auto toCmp = b.create<tensor::ExtractOp>(maxOpAdaptor.input(), idx);
                Value groupIdx = b.create<tensor::ExtractOp>(maxOpAdaptor.indices(), idx);
                auto oldVal = b.create<memref::LoadOp>(res, ::llvm::makeArrayRef(groupIdx));

                ::mlir::Value maxVal;
                if (toCmp.getType().isa<IntegerType>()) {
                    maxVal = b.create<MaxSIOp>(toCmp, oldVal);
                } else {
                    maxVal = b.create<MaxFOp>(toCmp, oldVal);
                }

                b.create<memref::StoreOp>(maxVal, res, groupIdx);
            }
        };

        buildAffineLoopNest(rewriter, rewriter.getLoc(),
                            ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(0)),
                            rewriter.create<tensor::DimOp>(maxOpAdaptor.input(), 0).result(), {1}, fn);

        return rewriter.create<ToTensorOp>(res);
    }

    LogicalResult
    MaxOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const {
        assert(!op->getResultTypes().empty() && op->getResultTypes().size() == 1);
        auto loc = op->getLoc();
        MaxOpAdaptor minOpAdaptor(operands);
        Value res;

        ImplicitLocOpBuilder builder(loc, rewriter);
        if (minOpAdaptor.indices() && op->getResultTypes().front().isa<TensorType>()) {
            res = groupedMaxLowering(op, minOpAdaptor, builder);
        } else {
            res = scalarMaxLowering(op, minOpAdaptor, builder);
        }

        rewriter.replaceOp(op, res);

        return success();
    }

} // namespace voila::mlir::lowering