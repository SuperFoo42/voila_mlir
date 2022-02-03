#include "mlir/lowering/InsertOpLowering.hpp"

#include "NotImplementedException.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/VoilaOps.h"

#include <bit>

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::InsertOp;
    using ::mlir::voila::InsertOpAdaptor;
    InsertOpLowering::InsertOpLowering(MLIRContext *ctx) : ConversionPattern(InsertOp::getOperationName(), 1, ctx) {}

    static auto allocHashTables(ImplicitLocOpBuilder &rewriter, ValueRange values)
    {
        Value htSize;
        SmallVector<Value> hts;
        auto valType = values.front().getType().dyn_cast<TensorType>();
        // can calculate ht size from static shape
        if (valType.hasStaticShape())
        {
            auto size = std::bit_ceil<size_t>(valType.getShape().front() + 1);
            assert(size <= std::numeric_limits<int64_t>::max());
            htSize = rewriter.create<ConstantIndexOp>(size);
        }
        else
        {
            auto insertSize = rewriter.create<tensor::DimOp>(values.front(), 0);
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
                insertSize, rewriter.create<ShRUIOp>(insertSize, rewriter.create<ConstantIndexOp>(1)));
            auto secondOr =
                rewriter.create<OrIOp>(firstOr, rewriter.create<ShRUIOp>(firstOr, rewriter.create<ConstantIndexOp>(2)));
            auto thirdOr = rewriter.create<OrIOp>(
                secondOr, rewriter.create<ShRUIOp>(secondOr, rewriter.create<ConstantIndexOp>(4)));
            auto fourthOr =
                rewriter.create<OrIOp>(thirdOr, rewriter.create<ShRUIOp>(thirdOr, rewriter.create<ConstantIndexOp>(8)));
            auto fithOr = rewriter.create<OrIOp>(
                fourthOr, rewriter.create<ShRUIOp>(fourthOr, rewriter.create<ConstantIndexOp>(16)));
            auto sixthOr =
                rewriter.create<OrIOp>(fithOr, rewriter.create<ShRUIOp>(fithOr, rewriter.create<ConstantIndexOp>(32)));

            htSize = rewriter.create<AddIOp>(sixthOr, rewriter.create<ConstantIndexOp>(1));
        }
        for (auto val : values)
        {
            hts.push_back(rewriter.create<memref::AllocOp>(
                MemRefType::get(-1, val.getType().dyn_cast<TensorType>().getElementType()), htSize));
        }

        return std::make_pair(hts, htSize);
    }

    static auto createKeyComparisons(ImplicitLocOpBuilder &builder,
                                     ValueRange hts,
                                     ValueRange hashInvalidConsts,
                                     ValueRange toStores,
                                     ValueRange idx)
    {
        SmallVector<Value> bucketVals;
        for (auto ht : hts)
        {
            bucketVals.push_back(builder.create<memref::LoadOp>(ht, idx));
        }

        SmallVector<Value> empties;
        for (size_t i = 0; i < bucketVals.size(); ++i)
        {
            if (hashInvalidConsts[i].getType().isa<IntegerType>())
            {
                empties.push_back(builder.create<CmpIOp>(builder.getI1Type(), CmpIPredicate::ne, bucketVals[i],
                                                         hashInvalidConsts[i]));
            }
            else if (hashInvalidConsts[i].getType().isa<FloatType>())
            {
                empties.push_back(builder.create<CmpFOp>(builder.getI1Type(), CmpFPredicate::ONE, bucketVals[i],
                                                         hashInvalidConsts[i]));
            }
            else
            {
                throw NotImplementedException();
            }
        }

        Value anyNotEmpty = empties[0];
        for (size_t i = 1; i < empties.size(); ++i)
        {
            anyNotEmpty = builder.create<OrIOp>(anyNotEmpty, empties[i]);
        }

        SmallVector<Value> founds;
        for (size_t i = 0; i < bucketVals.size(); ++i)
        {
            if (bucketVals[i].getType().isa<IntegerType>())
            {
                builder.create<AssertOp>(
                    builder.create<CmpIOp>(builder.getI1Type(), CmpIPredicate::ne, hashInvalidConsts[i], toStores[i]),
                    "Trying to insert invalid const");
                founds.push_back(
                    builder.create<CmpIOp>(builder.getI1Type(), CmpIPredicate::ne, bucketVals[i], toStores[i]));
            }
            else if (bucketVals[i].getType().isa<FloatType>())
            {
                founds.push_back(
                    builder.create<CmpFOp>(builder.getI1Type(), CmpFPredicate::ONE, bucketVals[i], toStores[i]));
            }
            else
            {
                throw NotImplementedException();
            }
        }
        Value anyNotFound = founds[0];
        for (size_t i = 1; i < founds.size(); ++i)
        {
            anyNotFound = builder.create<OrIOp>(anyNotFound, founds[i]);
        }

        return builder.create<AndIOp>(anyNotEmpty, anyNotFound);
    }

    // TODO: use atomic compare exchange
    ::mlir::LogicalResult InsertOpLowering::matchAndRewrite(Operation *op,
                                                            ArrayRef<Value> operands,
                                                            ConversionPatternRewriter &rewriter) const
    {
        auto iOp = dyn_cast<InsertOp>(op);
        InsertOpAdaptor insertOpAdaptor(iOp);
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);
        // allocate and prefill hashtable
        auto tmp = allocHashTables(builder, insertOpAdaptor.values());
        auto hts = tmp.first;
        auto htSize = tmp.second;

        SmallVector<Value, 1> lb;
        // TODO: add separate bucket for 0 keys to not run into problems when inserting INVALID

        SmallVector<Value> hashInvalidConsts;
        for (auto val : insertOpAdaptor.values())
        { // we uce all ones value of data type size
            const auto &elementType = getElementTypeOrSelf(val);
            if (elementType.isIntOrFloat())
            {
                hashInvalidConsts.push_back(
                    builder.create<BitcastOp>(builder.create<ConstantIntOp>(std::numeric_limits<uint64_t>::max(),
                                                                            elementType.getIntOrFloatBitWidth()),
                                              elementType));
            }
            else
            {
                throw NotImplementedException();
            }
        }

        lb.push_back(builder.create<ConstantIndexOp>(0));

        for (size_t i = 0; i < hts.size(); ++i)
        {
            builder.create<linalg::FillOp>(hashInvalidConsts[i], hts[i]);
        }

        // hash values
        auto modSize = builder.create<SubIOp>(htSize, builder.create<ConstantIndexOp>(1));

        // insert values in ht
        // we can not use affine loads here, as this would lead to comprehensive bufferize complain about RaW conflicts.
        auto loopFunc = [&modSize, &hts, &hashInvalidConsts, &insertOpAdaptor](OpBuilder &nestedBuilder, Location loc,
                                                                               ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            // load values
            auto hashIdx = builder.create<IndexCastOp>(
                builder.create<tensor::ExtractOp>(insertOpAdaptor.hashValues(), vals), builder.getIndexType());
            Value correctedHashIdx = builder.create<AndIOp>(hashIdx, modSize);

            SmallVector<Value> toStores;
            for (auto val : insertOpAdaptor.values())
            {
                toStores.push_back(builder.create<tensor::ExtractOp>(val, vals));
            }
            SmallVector<Type, 1> resTypes;
            resTypes.push_back(hashIdx.getType());
            // probing
            auto loop = builder.create<::mlir::scf::WhileOp>(resTypes, llvm::makeArrayRef(correctedHashIdx));

            // condition block
            auto beforeBlock = builder.createBlock(&loop.getBefore());
            beforeBlock->addArgument(loop->getOperands().front().getType(), loc);
            ImplicitLocOpBuilder beforeBuilder(loc, OpBuilder::atBlockEnd(beforeBlock));

            auto comparisons =
                createKeyComparisons(beforeBuilder, hts, hashInvalidConsts, toStores, beforeBlock->getArgument(0));
            beforeBuilder.create<scf::ConditionOp>(comparisons, beforeBlock->getArgument(0));
            // body block
            auto afterBlock = builder.createBlock(&loop.getAfter());
            afterBlock->addArgument(loop->getOperands().front().getType(), loc);
            ImplicitLocOpBuilder afterBuilder(loc, OpBuilder::atBlockEnd(afterBlock));

            auto nextIdx =
                afterBuilder.create<AddIOp>(afterBlock->getArgument(0), afterBuilder.create<ConstantIndexOp>(loc, 1));
            auto nextIndexWrapOver = afterBuilder.create<AndIOp>(nextIdx, modSize);
            SmallVector<Value, 1> res;
            res.push_back(nextIndexWrapOver);
            afterBuilder.create<scf::YieldOp>(res);

            // insert
            builder.setInsertionPointAfter(loop);
            for (const auto &en : llvm::enumerate(llvm::zip(toStores, hts)))
            {
                Value toStore, ht;
                std::tie(toStore, ht) = en.value();
                builder.create<memref::StoreOp>(toStore, ht, loop->getResults());
            }
        };

        buildAffineLoopNest(builder, builder.getLoc(), lb,
                            {builder.create<tensor::DimOp>(insertOpAdaptor.hashValues(), 0)}, {1}, loopFunc);

        SmallVector<Value> ret;
        for (const auto &ht : hts)
        {
            ret.push_back(builder.create<ToTensorOp>(ht));
        }
        rewriter.replaceOp(op, ret);

        return success();
    }

} // namespace voila::mlir::lowering