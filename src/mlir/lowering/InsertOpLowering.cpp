#include "mlir/lowering/InsertOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using ::mlir::voila::InsertOp;
    using ::mlir::voila::InsertOpAdaptor;
    InsertOpLowering::InsertOpLowering(MLIRContext *ctx) : ConversionPattern(InsertOp::getOperationName(), 1, ctx) {}

    static auto convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static auto allocHashTables(ConversionPatternRewriter &rewriter, ValueRange values, Location loc)
    {
        // TODO: verifier for this
        // assume all values are of same size
        auto insertSize = rewriter.create<tensor::DimOp>(loc, values.front(), 0);
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
        auto firstOr = rewriter.create<OrIOp>(
            loc, insertSize, rewriter.create<ShRUIOp>(loc, insertSize, rewriter.create<ConstantIndexOp>(loc, 1)));
        auto secondOr = rewriter.create<OrIOp>(
            loc, firstOr, rewriter.create<ShRUIOp>(loc, firstOr, rewriter.create<ConstantIndexOp>(loc, 2)));
        auto thirdOr = rewriter.create<OrIOp>(
            loc, secondOr, rewriter.create<ShRUIOp>(loc, secondOr, rewriter.create<ConstantIndexOp>(loc, 4)));
        auto fourthOr = rewriter.create<OrIOp>(
            loc, thirdOr, rewriter.create<ShRUIOp>(loc, thirdOr, rewriter.create<ConstantIndexOp>(loc, 8)));
        auto fithOr = rewriter.create<OrIOp>(
            loc, fourthOr, rewriter.create<ShRUIOp>(loc, fourthOr, rewriter.create<ConstantIndexOp>(loc, 16)));
        auto sixthOr = rewriter.create<OrIOp>(
            loc, fithOr, rewriter.create<ShRUIOp>(loc, fithOr, rewriter.create<ConstantIndexOp>(loc, 32)));
        SmallVector<Value, 1> htSize;
        htSize.push_back(rewriter.create<AddIOp>(loc, sixthOr, rewriter.create<ConstantIndexOp>(loc, 1)));

        SmallVector<Value> hts;
        for (auto val : values)
        {
            hts.push_back(rewriter.create<memref::AllocOp>(
                loc, MemRefType::get(-1, val.getType().dyn_cast<TensorType>().getElementType()), htSize));
        }
        return std::make_pair(hts, htSize);
    }

    static auto createKeyComparisons(OpBuilder &builder,
                                     Location loc,
                                     ValueRange hts,
                                     ValueRange hashInvalidConsts,
                                     ValueRange toStores,
                                     ValueRange idx)
    {
        SmallVector<Value> bucketVals;
        for (auto ht : hts)
        {
            bucketVals.push_back(builder.create<memref::LoadOp>(loc, ht, idx));
        }

        SmallVector<Value> empties;
        for (size_t i = 0; i < bucketVals.size(); ++i)
        {
            empties.push_back(builder.create<CmpIOp>(loc, builder.getI1Type(), CmpIPredicate::ne, bucketVals[i],
                                                     hashInvalidConsts[i]));
        }

        Value anyNotEmpty = empties[0];
        for (size_t i = 1; i < empties.size(); ++i)
        {
            anyNotEmpty = builder.create<AndIOp>(loc, anyNotEmpty, empties[i]);
        }

        SmallVector<Value> founds;
        for (size_t i = 0; i < bucketVals.size(); ++i)
        {
            founds.push_back(builder.create<CmpIOp>(loc, CmpIPredicate::eq, bucketVals[i], toStores[i]));
        }
        Value allFound = founds[0];
        for (size_t i = 1; i < founds.size(); ++i)
        {
            allFound = builder.create<AndIOp>(loc, allFound, founds[i]);
        }

        return builder.create<OrIOp>(loc, builder.getI1Type(), anyNotEmpty, allFound);
    }

    ::mlir::LogicalResult InsertOpLowering::matchAndRewrite(Operation *op,
                                                            ArrayRef<Value> operands,
                                                            ConversionPatternRewriter &rewriter) const
    {
        InsertOpAdaptor insertOpAdaptor(operands);
        auto loc = op->getLoc();

        // allocate and prefill hashtable
        auto tmp = allocHashTables(rewriter, insertOpAdaptor.values(), loc);
        auto hts = tmp.first;
        auto htSize = tmp.second;

        SmallVector<Value, 1> lb;
        // TODO: add separate bucket for 0 keys to not run into problems when inserting 0

        SmallVector<Value> hashInvalidConsts;
        for (auto val : insertOpAdaptor.values())
        { // FIXME: floats
            hashInvalidConsts.push_back(
                rewriter.create<ConstantIntOp>(loc, std::numeric_limits<uint64_t>::max(), getElementTypeOrSelf(val)));
        }

        lb.push_back(rewriter.create<ConstantIndexOp>(loc, 0));

        for (size_t i = 0; i < hts.size(); ++i)
        {
            rewriter.create<linalg::FillOp>(loc, hashInvalidConsts[i], hts[i]);
        }

        // hash values
        auto hashVals = insertOpAdaptor.hashValues();

        // map to ht size
        // mapping by mod of power 2 which is just x & (htSize-1)
        auto modSize = rewriter.create<SubIOp>(loc, htSize.front(), rewriter.create<ConstantIndexOp>(loc, 1));
        auto intMod = rewriter.create<IndexCastOp>(loc, modSize, rewriter.getI64Type());
        auto mappedHashVals = rewriter.create<::mlir::voila::AndOp>(loc, hashVals.getType(), hashVals, intMod);
        auto indexMappedHashVals = rewriter.create<IndexCastOp>(
            loc, mappedHashVals,
            RankedTensorType::get(hashVals.getType().dyn_cast<TensorType>().getShape(), rewriter.getIndexType()));
        auto mappedHashValsMemref = rewriter.create<memref::BufferCastOp>(
            loc, convertTensorToMemRef(indexMappedHashVals.getType().dyn_cast<TensorType>()), indexMappedHashVals);
        SmallVector<Value> keyMemrefs;
        for (auto val : insertOpAdaptor.values())
        {
            keyMemrefs.push_back(rewriter.create<memref::BufferCastOp>(
                loc, convertTensorToMemRef(val.getType().dyn_cast<TensorType>()), val));
        }
        // insert values in ht
        buildAffineLoopNest(
            rewriter, loc, lb, {rewriter.create<tensor::DimOp>(loc, hashVals, 0)}, {1},
            [&modSize, &hts, &hashInvalidConsts, &mappedHashValsMemref, &keyMemrefs](OpBuilder &builder, Location loc,
                                                                                     ValueRange vals)
            {
                // load values
                SmallVector<Value, 1> hashIdx;
                hashIdx.push_back(builder.create<AffineLoadOp>(loc, mappedHashValsMemref, vals));
                SmallVector<Value> toStores;
                for (auto val : keyMemrefs)
                {
                    toStores.push_back(builder.create<AffineLoadOp>(loc, val, vals));
                }
                SmallVector<Type, 1> resTypes;
                resTypes.push_back(hashIdx[0].getType());
                // probing
                auto loop = builder.create<::mlir::scf::WhileOp>(loc, resTypes, hashIdx);

                // condition block
                auto beforeBlock = builder.createBlock(&loop.before());
                beforeBlock->addArgument(loop->getOperands().front().getType());
                auto beforeBuilder = OpBuilder::atBlockEnd(beforeBlock);

                beforeBuilder.create<scf::ConditionOp>(loc,
                                                       createKeyComparisons(beforeBuilder, loc, hts, hashInvalidConsts,
                                                                            toStores, loop.before().getArguments()),
                                                       loop->getOperands());
                // body block
                auto afterBlock = builder.createBlock(&loop.after());
                afterBlock->addArgument(loop->getOperands().front().getType());
                auto afterBuilder = OpBuilder::atBlockEnd(afterBlock);

                auto nextIdx = afterBuilder.create<AddIOp>(loc, afterBlock->getArgument(0),
                                                           afterBuilder.create<ConstantIndexOp>(loc, 1));
                auto nextIndexWrapOver = afterBuilder.create<AndIOp>(loc, nextIdx, modSize);
                SmallVector<Value, 1> res;
                res.push_back(nextIndexWrapOver);
                afterBuilder.create<scf::YieldOp>(loc, res);

                // insert
                builder.setInsertionPointAfter(loop);
                for (size_t i = 0; i < toStores.size(); ++i)
                {
                    builder.create<memref::StoreOp>(loc, toStores[i], hts[i], loop->getResults());
                }
            });

        SmallVector<Value> ret;
        for (size_t i = 0; i < hts.size(); ++i)
        {
            ret.push_back(rewriter.create<memref::TensorLoadOp>(loc, hts[i]));
        }
        rewriter.replaceOp(op, ret);

        return success();
    }

} // namespace voila::mlir::lowering