#include "mlir/lowering/LookupOpLowering.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::LookupOp;
    using ::mlir::voila::LookupOpAdaptor;

    LookupOpLowering::LookupOpLowering(::mlir::MLIRContext *ctx) :
        ConversionPattern(LookupOp::getOperationName(), 1, ctx)
    {
    }

    ::mlir::LogicalResult LookupOpLowering::matchAndRewrite(::mlir::Operation *op,
                                                            llvm::ArrayRef<::mlir::Value> operands,
                                                            ConversionPatternRewriter &rewriter) const
    {
        LookupOpAdaptor lookupOpAdaptor(operands);
        auto loc = op->getLoc();

        Value outTensor = rewriter.create<linalg::InitTensorOp>(
            loc,
            lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>().hasStaticShape() ?
                ValueRange() :
                llvm::makeArrayRef<Value>(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashes(), 0)),
            lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>().getShape(), rewriter.getIndexType());

        auto htSizes = rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashtables().front(), 0);
        auto modSize = rewriter.create<SubIOp>(loc, htSizes, rewriter.create<ConstantIndexOp>(loc, 1));
        auto intMod = rewriter.create<IndexCastOp>(loc, modSize, rewriter.getI64Type());

        auto lookupFunc = [&](OpBuilder &builder, Location loc, ValueRange vals)
        {
            auto hashVal = vals.take_front();
            vals = vals.drop_front();
            vals = vals.drop_back();
            // probing
            SmallVector<Type, 1> resType;
            resType.push_back(builder.getI64Type());
            Value idx = builder.create<AndIOp>(loc, hashVal.front(), intMod);
            // probing with do while
            // Maybe we should keep track of the max probing count during generation and iterate only so many times
            auto loop = builder.create<scf::WhileOp>(loc, resType, idx);
            // condition

            auto beforeBlock = builder.createBlock(&loop.getBefore());
            beforeBlock->addArgument(loop->getOperands().front().getType());
            auto condBuilder = OpBuilder::atBlockEnd(beforeBlock);
            Value probeIdx =
                condBuilder.create<IndexCastOp>(loc, loop.getBefore().getArgument(0), builder.getIndexType());

            // lookup entries
            SmallVector<Value> entries;
            std::transform(lookupOpAdaptor.hashtables().begin(), lookupOpAdaptor.hashtables().end(),
                           std::back_inserter(entries),
                           [&builder, &loc, &probeIdx](auto elem) -> Value
                           {
                               auto res = builder.create<tensor::ExtractOp>(loc, elem, probeIdx);
                               if (!elem.getType().template isa<IntegerType>())
                                   return builder.create<arith::BitcastOp>(
                                       loc, res, builder.getIntegerType(res.getType().getIntOrFloatBitWidth()));
                               else
                                   return res;
                           });

            SmallVector<Value> vs;
            std::transform(vals.begin(), vals.end(), std::back_inserter(vs),
                           [&builder, &loc](auto elem) -> Value
                           {
                               if (!elem.getType().template isa<IntegerType>())
                                   return builder.create<arith::BitcastOp>(
                                       loc, elem, builder.getIntegerType(elem.getType().getIntOrFloatBitWidth()));
                               else
                                   return elem;
                           });

            Value isEmpty = builder.create<CmpIOp>(
                loc, CmpIPredicate::ne, entries[0],
                builder.create<ConstantIntOp>(loc, std::numeric_limits<uint64_t>::max(), entries[0].getType()));
            Value notFound = condBuilder.create<CmpIOp>(loc, CmpIPredicate::ne, entries[0], vs[0]);
            for (auto &en : llvm::enumerate(llvm::zip(entries, vs)))
            {
                Value entry, value;
                std::tie(entry, value) = en.value();
                auto tmp = builder.create<CmpIOp>(loc, CmpIPredicate::ne, entry,
                                                  builder.create<ConstantIntOp>(loc, 0, entry.getType()));
                isEmpty = builder.create<AndIOp>(loc, isEmpty, tmp);
                auto tmp2 = condBuilder.create<CmpIOp>(loc, CmpIPredicate::ne, entry, value);
                notFound = builder.create<OrIOp>(loc, isEmpty, tmp2);
            }
            condBuilder.create<scf::ConditionOp>(
                loc, condBuilder.create<AndIOp>(loc, builder.getI1Type(), isEmpty, notFound), loop->getOperands());
            // body
            auto afterBlock = builder.createBlock(&loop.getAfter());
            afterBlock->addArgument(loop->getOperands().front().getType());
            auto bodyBuilder = OpBuilder::atBlockEnd(afterBlock);
            SmallVector<Value, 1> inc;
            inc.push_back(bodyBuilder.create<AndIOp>(
                loc,
                bodyBuilder.create<AddIOp>(loc, loop.getAfterArguments().front(),
                                           builder.create<ConstantIntOp>(loc, 1, builder.getI64Type())),
                intMod));
            bodyBuilder.create<scf::YieldOp>(loc, inc);
            builder.setInsertionPointAfter(loop);
            // check index empty

            entries.clear();
            Value resIdx = builder.create<IndexCastOp>(loc, loop->getResults().front(), builder.getIndexType());
            std::transform(lookupOpAdaptor.hashtables().begin(), lookupOpAdaptor.hashtables().end(),
                           std::back_inserter(entries),
                           [&builder, &loc, &resIdx](auto elem) -> Value
                           {
                               auto res = builder.create<tensor::ExtractOp>(loc, elem, resIdx);
                               if (!elem.getType().template isa<IntegerType>())
                                   return builder.create<arith::BitcastOp>(
                                       loc, res, builder.getIntegerType(res.getType().getIntOrFloatBitWidth()));
                               else
                                   return res;
                           });
            Value empty = builder.create<CmpIOp>(
                loc, CmpIPredicate::ne, entries[0],
                builder.create<ConstantIntOp>(loc, std::numeric_limits<uint64_t>::max(), entries[0].getType()));
            for (auto &en : entries)
            {
                auto tmp = builder.create<CmpIOp>(loc, CmpIPredicate::ne, en,
                                                  builder.create<ConstantIntOp>(loc, 0, en.getType()));
                empty = builder.create<AndIOp>(loc, empty, tmp);
            }
            Value res = builder.create<SelectOp>(
                loc, empty, builder.create<ConstantIndexOp>(loc, std::numeric_limits<uint64_t>::max()), resIdx);
            // store result
            builder.create<linalg::YieldOp>(loc, res);
        };

        llvm::SmallVector<AffineMap> indexing_maps(/*hashes+outTensor*/ 2 + lookupOpAdaptor.values().size(),
                                                   rewriter.getDimIdentityMap());
        llvm::SmallVector<Value> inputs(1, lookupOpAdaptor.hashes());
        inputs.append(lookupOpAdaptor.values().begin(), lookupOpAdaptor.values().end());
        auto linalgOp = rewriter.create<linalg::GenericOp>(
            loc, /*results*/ llvm::makeArrayRef(outTensor.getType()),
            /*inputs*/ inputs,
            /*outputs*/ llvm::makeArrayRef(outTensor),
            /*indexing maps*/ indexing_maps,
            /*iterator types*/ llvm::makeArrayRef(getParallelIteratorTypeName()), lookupFunc);

        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
    }

} // namespace voila::mlir::lowering
