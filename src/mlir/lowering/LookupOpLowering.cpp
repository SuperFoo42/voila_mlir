#include "mlir/lowering/LookupOpLowering.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
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

        auto htSize = rewriter.create<memref::DimOp>(loc, lookupOpAdaptor.hashtable(), 0);
        auto hashValues = rewriter.create<::mlir::voila::HashOp>(loc, RankedTensorType::get(lookupOpAdaptor.keys().getType().dyn_cast<TensorType>().getShape(), rewriter.getI64Type()),
                                                                 lookupOpAdaptor.keys());
        auto modSize = rewriter.create<SubIOp>(loc, htSize, rewriter.create<ConstantIndexOp>(loc, 1));
        auto intMod = rewriter.create<IndexCastOp>(loc, modSize, rewriter.getI64Type());
        auto mappedHashVals = rewriter.create<::mlir::voila::AndOp>(loc, hashValues.getType(), hashValues, intMod);
        auto indexMappedHashVals = rewriter.create<IndexCastOp>(loc, mappedHashVals, RankedTensorType::get(hashValues.getType().getShape(), rewriter.getIndexType()));

        auto mappedHashValsMemRef =
            rewriter.create<memref::BufferCastOp>(loc, convertTensorToMemRef(indexMappedHashVals.getType().dyn_cast<TensorType>()), indexMappedHashVals);

        SmallVector<Value, 1> resSize;
        resSize.push_back(rewriter.create<tensor::DimOp>(loc, hashValues, 0));
        SmallVector<Value, 1> results;
        results.push_back(rewriter.create<linalg::InitTensorOp>(loc, resSize, rewriter.getIndexType()));
        SmallVector<Type, 1> ret_type;
        ret_type.push_back(results.back().getType());

        SmallVector<Value, 1> allocSize;
        allocSize.push_back(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.keys(), 0));
        auto res = rewriter.create<memref::AllocOp>(loc, MemRefType::get(-1, rewriter.getIndexType()), allocSize);

        auto loopBody =
            [&modSize, &res, &mappedHashValsMemRef, &lookupOpAdaptor](OpBuilder &builder, Location loc, ValueRange vals)
        {
            // probing
            SmallVector<Type, 1> resTypes;
            resTypes.push_back(builder.getIndexType());
            SmallVector<Value, 1> hashVal;
            hashVal.push_back(builder.create<AffineLoadOp>(loc, mappedHashValsMemRef, vals));
            // probing with do while
            // FIXME: this probing runs over the end of the hashtable, if no empty bucket or matching key is found.
            // Maybe we should keep track of the max probing count during generation and iterate only so many times
            auto loop = builder.create<scf::WhileOp>(loc, resTypes, hashVal);
            // condition

            auto beforeBlock = builder.createBlock(&loop.before());
            beforeBlock->addArgument(loop->getOperands().front().getType());
            auto condBuilder = OpBuilder::atBlockEnd(beforeBlock);
            auto entry =
                condBuilder.create<memref::LoadOp>(loc, lookupOpAdaptor.hashtable(), loop.before().getArguments());
            auto isEmpty = builder.create<CmpIOp>(loc, CmpIPredicate::eq, entry,
                                                  builder.create<ConstantIntOp>(loc, 0, builder.getI64Type()));

            auto key = condBuilder.create<AffineLoadOp>(
                loc,
                builder.create<memref::BufferCastOp>(
                    loc, convertTensorToMemRef(lookupOpAdaptor.keys().getType().dyn_cast<TensorType>()),
                    lookupOpAdaptor.keys()),
                vals);
            auto notFound = condBuilder.create<CmpIOp>(loc, CmpIPredicate::ne, entry, key);
            condBuilder.create<scf::ConditionOp>(
                loc, condBuilder.create<OrOp>(loc, builder.getI1Type(), isEmpty, notFound), loop->getOperands());
            // body
            auto afterBlock = builder.createBlock(&loop.after());
            afterBlock->addArgument(loop->getOperands().front().getType());
            auto bodyBuilder = OpBuilder::atBlockEnd(afterBlock);
            SmallVector<Value, 1> inc;
            inc.push_back(bodyBuilder.create<AndOp>(loc,
                                                    bodyBuilder.create<AddIOp>(loc, loop.getAfterArguments().front(),
                                                                               builder.create<ConstantIndexOp>(loc, 1)),
                                                    modSize));
            bodyBuilder.create<scf::YieldOp>(loc, inc);
            builder.setInsertionPointAfter(loop);
            // store result
            builder.create<AffineStoreOp>(loc, loop->getResult(0), res, vals);
        };
        SmallVector<Value, 1> lb, ub;
        lb.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        ub.push_back(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.keys(), 0));
        buildAffineLoopNest(rewriter, loc, lb, ub, 1, loopBody);

        rewriter.replaceOp(op, {res});
        return success();
    }

} // namespace voila::mlir::lowering
