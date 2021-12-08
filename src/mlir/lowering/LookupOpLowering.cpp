#include "mlir/lowering/LookupOpLowering.hpp"

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

    static auto convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    ::mlir::LogicalResult LookupOpLowering::matchAndRewrite(::mlir::Operation *op,
                                                            llvm::ArrayRef<::mlir::Value> operands,
                                                            ConversionPatternRewriter &rewriter) const
    {
        LookupOpAdaptor lookupOpAdaptor(operands);
        auto loc = op->getLoc();
        ::mlir::Value outMemref;
        if (lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>().hasStaticShape())
        {
            outMemref = rewriter.create<memref::AllocOp>(
                loc, convertTensorToMemRef(lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>()));
        }
        else
        {
            SmallVector<Value, 1> outTensor;
            outTensor.push_back(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashes(), 0));
            outMemref = rewriter.create<memref::AllocOp>(
                loc, convertTensorToMemRef(lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>()), outTensor);
        }

        auto htSizes = rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashtables().front(), 0);
        auto modSize = rewriter.create<SubIOp>(loc, htSizes, rewriter.create<ConstantIndexOp>(loc, 1));
        auto intMod = rewriter.create<IndexCastOp>(loc, modSize, rewriter.getI64Type());

        auto hashesBuffer = rewriter.create<ToMemrefOp>(
            loc, convertTensorToMemRef(lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>()),
            lookupOpAdaptor.hashes());

        SmallVector<Value> valueBuffers;
        std::transform(
            lookupOpAdaptor.values().begin(), lookupOpAdaptor.values().end(), std::back_inserter(valueBuffers),
            [&rewriter, &loc](auto elem) -> auto {
                return rewriter.create<ToMemrefOp>(
                    loc, convertTensorToMemRef(elem.getType().template dyn_cast<TensorType>()), elem);
            });

        SmallVector<Value> hashtableBuffers;
        std::transform(
            lookupOpAdaptor.hashtables().begin(), lookupOpAdaptor.hashtables().end(),
            std::back_inserter(hashtableBuffers), [&rewriter, &loc](auto elem) -> auto {
                return rewriter.create<ToMemrefOp>(
                    loc, convertTensorToMemRef(elem.getType().template dyn_cast<TensorType>()), elem);
            });

        auto lookupFunc = [&intMod, &outMemref, &hashesBuffer, &valueBuffers,
                           &hashtableBuffers](OpBuilder &builder, Location loc, ValueRange vals)
        {
            auto hashVals = builder.create<AffineLoadOp>(loc, hashesBuffer, vals.front());

            // lookup values
            SmallVector<Value> values;
            std::transform(
                valueBuffers.begin(), valueBuffers.end(), std::back_inserter(values),
                [&builder, &loc, &vals](auto elem) -> auto
                { return builder.create<AffineLoadOp>(loc, elem, vals.front()); });

            // probing
            SmallVector<Type, 1> resType;
            resType.push_back(builder.getI64Type());
            SmallVector<Value, 1> hashVal;
            hashVal.push_back(builder.create<AndIOp>(loc, hashVals, intMod));
            // probing with do while
            // Maybe we should keep track of the max probing count during generation and iterate only so many times
            auto loop = builder.create<scf::WhileOp>(loc, resType, hashVal);
            // condition

            auto beforeBlock = builder.createBlock(&loop.before());
            beforeBlock->addArgument(loop->getOperands().front().getType());
            auto condBuilder = OpBuilder::atBlockEnd(beforeBlock);
            SmallVector<Value, 1> idx;
            idx.push_back(condBuilder.create<IndexCastOp>(loc, loop.before().getArgument(0), builder.getIndexType()));

            // lookup entries
            SmallVector<Value> entries;
            std::transform(
                hashtableBuffers.begin(), hashtableBuffers.end(), std::back_inserter(entries),
                [&builder, &loc, &idx](auto elem) -> auto { return builder.create<memref::LoadOp>(loc, elem, idx); });

            Value isEmpty = builder.create<CmpIOp>(loc, CmpIPredicate::eq, entries[0],
                                                   builder.create<ConstantIntOp>(loc, std::numeric_limits<uint64_t>::max(), entries[0].getType()));
            Value notFound = condBuilder.create<CmpIOp>(loc, CmpIPredicate::ne, entries[0], values[0]);
            for (size_t i = 1; i < entries.size(); ++i)
            {
                auto tmp = builder.create<CmpIOp>(loc, CmpIPredicate::eq, entries[i],
                                                  builder.create<ConstantIntOp>(loc, 0, entries[i].getType()));
                isEmpty = builder.create<AndIOp>(loc, isEmpty, tmp);
                auto tmp2 = condBuilder.create<CmpIOp>(loc, CmpIPredicate::ne, entries[i], values[i]);
                notFound = builder.create<OrIOp>(loc, isEmpty, tmp2);
            }

            condBuilder.create<scf::ConditionOp>(
                loc, condBuilder.create<OrIOp>(loc, builder.getI1Type(), isEmpty, notFound), loop->getOperands());
            // body
            auto afterBlock = builder.createBlock(&loop.after());
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
            // store result
            builder.create<AffineStoreOp>(loc, loop->getResults().front(), outMemref, vals.front());
        };

        SmallVector<Value, 1> lb, ub;
        lb.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        ub.push_back(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashes(), 0));

        buildAffineLoopNest(rewriter, loc, lb, ub, 1, lookupFunc);

        rewriter.replaceOpWithNewOp<ToTensorOp>(op, outMemref);
        return success();
    }

} // namespace voila::mlir::lowering
