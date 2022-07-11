#include "mlir/lowering/LookupOpLowering.hpp"

#include "NotImplementedException.hpp"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using ::mlir::voila::LookupOp;
    using ::mlir::voila::LookupOpAdaptor;

    // TODO: extract insert + lookup common utils
    LookupOpLowering::LookupOpLowering(::mlir::MLIRContext *ctx) :
        ConversionPattern(LookupOp::getOperationName(), 1, ctx)
    {
    }

    static auto anyNotEmpty(ImplicitLocOpBuilder &builder, ValueRange hashInvalidConsts, ValueRange bucketVals)
    {
        llvm::SmallVector<Value> empties;
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

        return anyNotEmpty;
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
            bucketVals.push_back(builder.create<tensor::ExtractOp>(ht, idx));
        }

        Value notEmpty = anyNotEmpty(builder, hashInvalidConsts, bucketVals);

        SmallVector<Value> founds;
        for (size_t i = 0; i < bucketVals.size(); ++i)
        {
            if (bucketVals[i].getType().isa<IntegerType>())
            {
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

        return builder.create<AndIOp>(notEmpty, anyNotFound);
    }

    ::mlir::LogicalResult LookupOpLowering::matchAndRewrite(::mlir::Operation *op,
                                                            llvm::ArrayRef<::mlir::Value> operands,
                                                            ConversionPatternRewriter &rewriter) const
    {
        auto lOop = llvm::dyn_cast<LookupOp>(op);
        LookupOpAdaptor lookupOpAdaptor(lOop);
        auto loc = op->getLoc();

        auto htSizes = rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashtables().front(), 0);
        auto modSize = rewriter.create<SubIOp>(loc, htSizes, rewriter.create<ConstantIndexOp>(loc, 1));

        SmallVector<Value> hashInvalidConsts;
        for (auto val : lookupOpAdaptor.values())
        { // we uce all ones value of data type size
            const auto &elementType = getElementTypeOrSelf(val);
            if (elementType.isIntOrFloat())
            {
                hashInvalidConsts.push_back(
                    rewriter.create<BitcastOp>(loc, elementType,
                                               rewriter.create<ConstantIntOp>(loc, std::numeric_limits<uint64_t>::max(),
                                                                              elementType.getIntOrFloatBitWidth())));
            }
            else
            {
                throw NotImplementedException();
            }
        }
        auto lookupFunc = [&](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            auto hashVal = vals.take_front();
            vals = vals.drop_front();
            vals = vals.drop_back();
            // probing
            Value idx =
                builder.create<AndIOp>(builder.create<IndexCastOp>(builder.getIndexType(), hashVal.front()), modSize);
            // probing with do while
            // Maybe we should keep track of the max probing count during generation and iterate only so many times
            auto loop = builder.create<scf::WhileOp>(builder.getIndexType(), idx);
            // condition

            auto beforeBlock = builder.createBlock(&loop.getBefore());
            beforeBlock->addArgument(loop->getOperands().front().getType(), loc);
            auto condBuilder = ImplicitLocOpBuilder(loc, OpBuilder::atBlockEnd(beforeBlock));
            Value probeIdx = beforeBlock->getArgument(0);

            // lookup entries
            auto comparison =
                createKeyComparisons(condBuilder, lookupOpAdaptor.hashtables(), hashInvalidConsts, vals, probeIdx);
            condBuilder.create<scf::ConditionOp>(comparison, probeIdx);

            // body
            auto afterBlock = builder.createBlock(&loop.getAfter());
            afterBlock->addArgument(loop->getOperands().front().getType(), loc);
            auto bodyBuilder = OpBuilder::atBlockEnd(afterBlock);
            Value inc = bodyBuilder.create<AndIOp>(
                loc,
                bodyBuilder.create<AddIOp>(loc, afterBlock->getArgument(0), builder.create<ConstantIndexOp>(loc, 1)),
                modSize);
            bodyBuilder.create<scf::YieldOp>(loc, llvm::makeArrayRef(inc));
            builder.setInsertionPointAfter(loop);
            // check index empty

            Value resIdx = loop->getResults().front();
            SmallVector<Value> bucketVals;
            for (auto ht : lookupOpAdaptor.hashtables())
            {
                bucketVals.push_back(builder.create<tensor::ExtractOp>(ht, resIdx));
            }
            auto notEmpty = anyNotEmpty(builder, hashInvalidConsts, bucketVals);

            Value res = builder.create<SelectOp>(loc, notEmpty, resIdx,
                                                 builder.create<ConstantIndexOp>(std::numeric_limits<uint64_t>::max()));
            // store result
            builder.create<linalg::YieldOp>(loc, res);
        };

        llvm::SmallVector<Value> inputs(1, lookupOpAdaptor.hashes());
        inputs.append(lookupOpAdaptor.values().begin(), lookupOpAdaptor.values().end());
        if (lookupOpAdaptor.pred())
        {
            Value outMemref = rewriter.create<memref::AllocOp>(
                loc,
                MemRefType::get(lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>().getShape(),
                                rewriter.getIndexType()),
                lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>().hasStaticShape() ?
                    ValueRange() :
                    llvm::makeArrayRef<Value>(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashes(), 0)));
            buildAffineLoopNest(
                rewriter, loc, llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)),
                rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashes(), 0).result(),
                llvm::makeArrayRef<int64_t>(1),
                [&](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
                {
                    ImplicitLocOpBuilder builder(loc, nestedBuilder);
                    auto pred = builder.create<tensor::ExtractOp>(lookupOpAdaptor.pred(), vals);
                    builder.create<scf::IfOp>(
                        pred,
                        [&](OpBuilder &b, Location loc)
                        {
                            ImplicitLocOpBuilder nb(loc, b);
                            auto hashVal = nb.create<tensor::ExtractOp>(lookupOpAdaptor.hashes(), vals);
                            SmallVector<Value, 6> values;
                            for (auto v : lookupOpAdaptor.values())
                            {
                                values.push_back(nb.create<tensor::ExtractOp>(v, vals));
                            }
                            // probing
                            Value idx = nb.create<AndIOp>(nb.create<IndexCastOp>(nb.getIndexType(), hashVal), modSize);
                            // probing with do while
                            // Maybe we should keep track of the max probing count during generation and iterate only so
                            // many times
                            auto loop = nb.create<scf::WhileOp>(nb.getIndexType(), idx);
                            // condition

                            auto beforeBlock = nb.createBlock(&loop.getBefore());
                            beforeBlock->addArgument(loop->getOperands().front().getType(), loc);
                            auto condBuilder = ImplicitLocOpBuilder(loc, OpBuilder::atBlockEnd(beforeBlock));
                            Value probeIdx = beforeBlock->getArgument(0);

                            // lookup entries
                            auto comparison = createKeyComparisons(condBuilder, lookupOpAdaptor.hashtables(),
                                                                   hashInvalidConsts, values, probeIdx);
                            condBuilder.create<scf::ConditionOp>(comparison, probeIdx);

                            // body
                            auto afterBlock = nb.createBlock(&loop.getAfter());
                            afterBlock->addArgument(loop->getOperands().front().getType(), loc);
                            auto bodyBuilder = OpBuilder::atBlockEnd(afterBlock);
                            Value inc = bodyBuilder.create<AndIOp>(
                                loc,
                                bodyBuilder.create<AddIOp>(loc, afterBlock->getArgument(0),
                                                           nb.create<ConstantIndexOp>(loc, 1)),
                                modSize);
                            bodyBuilder.create<scf::YieldOp>(loc, llvm::makeArrayRef(inc));
                            nb.setInsertionPointAfter(loop);
                            // check index empty

                            Value resIdx = loop->getResults().front();
                            SmallVector<Value> bucketVals;
                            for (auto ht : lookupOpAdaptor.hashtables())
                            {
                                bucketVals.push_back(nb.create<tensor::ExtractOp>(ht, resIdx));
                            }
                            auto notEmpty = anyNotEmpty(nb, hashInvalidConsts, bucketVals);

                            Value res =
                                nb.create<SelectOp>(loc, notEmpty, resIdx,
                                                    nb.create<ConstantIndexOp>(std::numeric_limits<uint64_t>::max()));
                            // store result
                            nb.create<AffineStoreOp>(loc, res, outMemref, vals);
                            nb.create<scf::YieldOp>();
                        });
                });
            rewriter.replaceOpWithNewOp<ToTensorOp>(op, outMemref);
        }
        else
        {
            Value outTensor = rewriter.create<linalg::InitTensorOp>(
                loc,
                lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>().hasStaticShape() ?
                    ValueRange() :
                    llvm::makeArrayRef<Value>(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashes(), 0)),
                lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>().getShape(), rewriter.getIndexType());

            llvm::SmallVector<AffineMap> indexing_maps(/*hashes+outTensor*/ 2 + lookupOpAdaptor.values().size(),
                                                       rewriter.getDimIdentityMap());
            auto linalgOp = rewriter.create<linalg::GenericOp>(
                loc, /*results*/ llvm::makeArrayRef(outTensor.getType()),
                /*inputs*/ inputs,
                /*outputs*/ llvm::makeArrayRef(outTensor),
                /*indexing maps*/ indexing_maps,
                /*iterator types*/ llvm::makeArrayRef(getParallelIteratorTypeName()), lookupFunc);
            rewriter.replaceOp(op, linalgOp->getResults());
        }

        return success();
    }

} // namespace voila::mlir::lowering
