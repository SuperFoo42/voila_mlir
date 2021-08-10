#include "mlir/lowering/InsertOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::InsertOp;
    using ::mlir::voila::InsertOpAdaptor;
    InsertOpLowering::InsertOpLowering(MLIRContext *ctx) : ConversionPattern(InsertOp::getOperationName(), 1, ctx) {}

    static auto convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static auto allocHashTable(ConversionPatternRewriter &rewriter, Value keys, Location loc)
    {
        auto insertSize = rewriter.create<tensor::DimOp>(loc, keys, 0);
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
        auto firstOr = rewriter.create<OrOp>(
            loc, insertSize,
            rewriter.create<UnsignedShiftRightOp>(loc, insertSize, rewriter.create<ConstantIndexOp>(loc, 1)));
        auto secondOr = rewriter.create<OrOp>(
            loc, firstOr,
            rewriter.create<UnsignedShiftRightOp>(loc, firstOr, rewriter.create<ConstantIndexOp>(loc, 2)));
        auto thirdOr = rewriter.create<OrOp>(
            loc, secondOr,
            rewriter.create<UnsignedShiftRightOp>(loc, secondOr, rewriter.create<ConstantIndexOp>(loc, 4)));
        auto fourthOr = rewriter.create<OrOp>(
            loc, thirdOr,
            rewriter.create<UnsignedShiftRightOp>(loc, thirdOr, rewriter.create<ConstantIndexOp>(loc, 8)));
        auto fithOr = rewriter.create<OrOp>(
            loc, fourthOr,
            rewriter.create<UnsignedShiftRightOp>(loc, fourthOr, rewriter.create<ConstantIndexOp>(loc, 16)));
        auto sixthOr = rewriter.create<OrOp>(
            loc, fithOr, rewriter.create<UnsignedShiftRightOp>(loc, fithOr, rewriter.create<ConstantIndexOp>(loc, 32)));
        SmallVector<Value, 1> htSize;
        htSize.push_back(rewriter.create<AddIOp>(loc, sixthOr, rewriter.create<ConstantIndexOp>(loc, 1)));
        auto ht = rewriter.create<memref::AllocOp>(
            loc, MemRefType::get(-1, keys.getType().dyn_cast<TensorType>().getElementType()), htSize);
        return std::make_pair(ht, htSize);
    }

    ::mlir::LogicalResult InsertOpLowering::matchAndRewrite(Operation *op,
                                                            ArrayRef<Value> operands,
                                                            ConversionPatternRewriter &rewriter) const
    {
        InsertOpAdaptor insertOpAdaptor(operands);
        auto loc = op->getLoc();

        // allocate and prefill hashtable
        auto tmp = allocHashTable(rewriter, insertOpAdaptor.input(), loc);
        Value ht = tmp.first;
        auto htSize = tmp.second;

        SmallVector<Value, 1> lb;
        // TODO: dirty hack, add separate bucket for 0 keys to not run into problems when inserting 0
        auto hashInvalidConst = rewriter.create<ConstantIntOp>(
            loc, 0, insertOpAdaptor.input().getType().dyn_cast<TensorType>().getElementType());

        lb.push_back(rewriter.create<ConstantIndexOp>(loc, 0));

        buildAffineLoopNest(rewriter, loc, lb, htSize, {1},
                            [&ht, &hashInvalidConst](OpBuilder &builder, Location loc, ValueRange vals)
                            { builder.create<AffineStoreOp>(loc, hashInvalidConst, ht, vals); });

        // hash values
        auto hashVals = rewriter.create<::mlir::voila::HashOp>(loc, RankedTensorType::get(insertOpAdaptor.input().getType().dyn_cast<TensorType>().getShape(), rewriter.getIndexType()),
                                                               insertOpAdaptor.input());

        // map to ht size
        // mapping by mod of power 2 which is just x & (htSize-1)
        auto modSize = rewriter.create<SubIOp>(loc, htSize.front(), rewriter.create<ConstantIndexOp>(loc, 1));
        auto intMod = rewriter.create<IndexCastOp>(loc, modSize, rewriter.getI64Type());
        auto mappedHashVals = rewriter.create<::mlir::voila::AndOp>(loc, hashVals.getType(), hashVals, intMod);
        auto indexMappedHashVals = rewriter.create<IndexCastOp>(loc, mappedHashVals, RankedTensorType::get(hashVals.getType().getShape(), rewriter.getIndexType()));
        auto mappedHashValsMemref = rewriter.create<memref::BufferCastOp>(
            loc, convertTensorToMemRef(indexMappedHashVals.getType().dyn_cast<TensorType>()), indexMappedHashVals);
        auto keysMemref = rewriter.create<memref::BufferCastOp>(
            loc, convertTensorToMemRef(insertOpAdaptor.input().getType().dyn_cast<TensorType>()),
            insertOpAdaptor.input());
        // insert values in ht
        buildAffineLoopNest(rewriter, loc, lb, {rewriter.create<tensor::DimOp>(loc, hashVals, 0)}, {1},
                            [&modSize, &ht, &hashInvalidConst, &mappedHashValsMemref,
                             &keysMemref](OpBuilder &builder, Location loc, ValueRange vals)
                            {
                                // load values
                                SmallVector<Value, 1> hashIdx;
                                hashIdx.push_back(builder.create<AffineLoadOp>(loc, mappedHashValsMemref, vals));
                                auto toStore = builder.create<AffineLoadOp>(loc, keysMemref, vals);
                                SmallVector<Type, 1> resTypes;
                                resTypes.push_back(hashIdx[0].getType());
                                // probing
                                auto loop = builder.create<::mlir::scf::WhileOp>(loc, resTypes, hashIdx);

                                // condition block
                                auto beforeBlock = builder.createBlock(&loop.before());
                                beforeBlock->addArgument(loop->getOperands().front().getType());
                                auto beforeBuilder = OpBuilder::atBlockEnd(beforeBlock);
                                auto bucketVal =
                                    beforeBuilder.create<memref::LoadOp>(loc, ht, loop.before().getArguments());
                                auto notEmpty = beforeBuilder.create<CmpIOp>(
                                    loc, beforeBuilder.getI1Type(), CmpIPredicate::ne, bucketVal, hashInvalidConst);
                                beforeBuilder.create<scf::ConditionOp>(loc, notEmpty, loop->getOperands());
                                // body block
                                auto afterBlock = builder.createBlock(&loop.after());
                                afterBlock->addArgument(loop->getOperands().front().getType());
                                auto afterBuilder = OpBuilder::atBlockEnd(afterBlock);

                                auto nextIdx = afterBuilder.create<AddIOp>(
                                    loc, afterBlock->getArgument(0), afterBuilder.create<ConstantIndexOp>(loc, 1));
                                auto nextIndexWrapOver = afterBuilder.create<AndOp>(loc, nextIdx, modSize);
                                SmallVector<Value, 1> res;
                                res.push_back(nextIndexWrapOver);
                                afterBuilder.create<scf::YieldOp>(loc, res);

                                // insert
                                builder.setInsertionPointAfter(loop);
                                builder.create<memref::StoreOp>(loc, toStore, ht, loop->getResults());
                            });

        rewriter.replaceOp(op, ht);

        return success();
    }

} // namespace voila::mlir::lowering