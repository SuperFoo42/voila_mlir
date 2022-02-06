#include "mlir/lowering/CountOpLowering.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::tensor;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::CountOp;
    using ::mlir::voila::CountOpAdaptor;

    CountOpLowering::CountOpLowering(MLIRContext *ctx) : ConversionPattern(CountOp::getOperationName(), 1, ctx) {}

    static Value getHTSize(ImplicitLocOpBuilder &builder, Value values)
    {
        auto insertSize = builder.create<DimOp>(values, 0);
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
            builder.create<OrIOp>(insertSize, builder.create<ShRUIOp>(insertSize, builder.create<ConstantIndexOp>(1)));
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

    static Value scalarCountLowering(CountOpAdaptor &countOpAdaptor, ImplicitLocOpBuilder &builder)
    {
        auto cnt = builder.create<DimOp>(countOpAdaptor.input(), 0);
        return builder.create<IndexCastOp>(cnt, builder.getI64Type());
    }

    static Value groupedCountLowering(Operation *op, CountOpAdaptor &countOpAdaptor, ImplicitLocOpBuilder &rewriter)
    {
        Value res;
        auto allocSize =
            getHTSize(rewriter, countOpAdaptor.input()); // FIXME: not the best solution, indices can be out of range.

        res = rewriter.create<memref::AllocOp>(MemRefType::get(-1, rewriter.getI64Type()),
                                               ::llvm::makeArrayRef(allocSize));
        buildAffineLoopNest(rewriter, rewriter.getLoc(),
                            ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(0)), allocSize, {1},
                            [&res](OpBuilder &builder, Location loc, ValueRange vals) {
                                builder.create<AffineStoreOp>(
                                    loc, builder.create<ConstantIntOp>(loc, 0, builder.getI64Type()), res, vals);
                            });

        auto fn = [&res, &countOpAdaptor](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            auto idx = vals.front();
            Value groupIdx = builder.create<tensor::ExtractOp>(countOpAdaptor.indices(), idx);
            auto oldVal = builder.create<memref::LoadOp>(res, ::llvm::makeArrayRef(groupIdx));
            Value newVal = builder.create<AddIOp>(oldVal, builder.create<ConstantIntOp>(1, builder.getI64Type()));

            builder.create<memref::StoreOp>(newVal, res, groupIdx);
        };

        buildAffineLoopNest(rewriter, rewriter.getLoc(),
                            ::llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(0)),
                            rewriter.create<DimOp>(countOpAdaptor.input(), 0).result(), {1}, fn);

        return rewriter.create<ToTensorOp>(res);
    }

    LogicalResult
    CountOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        auto cOp = llvm::dyn_cast<CountOp>(op);
        CountOpAdaptor cntOpAdaptor(cOp);
        Value res;
        ImplicitLocOpBuilder builder(loc, rewriter);

        if (cntOpAdaptor.indices() && op->getResultTypes().front().isa<TensorType>())
        {
            res = groupedCountLowering(op, cntOpAdaptor, builder);
        }
        else
        {
            res = scalarCountLowering(cntOpAdaptor, builder);
        }

        rewriter.replaceOp(op, res);

        return success();
    }
} // namespace voila::mlir::lowering