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
        const auto &shape = lookupOpAdaptor.hashes().getType().dyn_cast<::mlir::TensorType>().getShape();
        ::mlir::Value outTensor;
        if (lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>().hasStaticShape())
        {
            outTensor = rewriter.create<linalg::InitTensorOp>(loc, shape, rewriter.getI64Type());
        }
        else
        {
            SmallVector<Value, 1> outTensorSize;
            outTensorSize.push_back(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashes(), 0));
            outTensor = rewriter.create<linalg::InitTensorOp>(loc, outTensorSize, rewriter.getI64Type());
        }

        SmallVector<Value, 1> res;
        res.push_back(outTensor);

        SmallVector<StringRef, 1> iter_type;
        iter_type.push_back(getParallelIteratorTypeName());

        SmallVector<Type, 1> ret_type;
        ret_type.push_back(outTensor.getType());
        SmallVector<AffineMap, 2> indexing_maps;
        indexing_maps.push_back(rewriter.getDimIdentityMap());
        indexing_maps.push_back(rewriter.getDimIdentityMap());

        SmallVector<Value, 1> hashIndices;
        hashIndices.push_back(lookupOpAdaptor.hashes());

        auto htSize = rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashtable(), 0);
        auto modSize = rewriter.create<SubIOp>(loc, htSize, rewriter.create<ConstantIndexOp>(loc, 1));
        auto intMod = rewriter.create<IndexCastOp>(loc, modSize, rewriter.getI64Type());

        auto lookupFunc = [&modSize, &lookupOpAdaptor](OpBuilder &builder, Location loc, ValueRange vals)
        {
            auto hashVals = vals[0];
            auto values = vals[1];
            // probing
            SmallVector<Type, 1> resTypes;
            resTypes.push_back(builder.getIndexType());
            SmallVector<Value, 1> hashVal;
            hashVal.push_back(builder.create<AndOp>(loc, hashVals, modSize));
            // probing with do while
            // Maybe we should keep track of the max probing count during generation and iterate only so many times
            auto loop = builder.create<scf::WhileOp>(loc, resTypes, hashVal);
            // condition

            auto beforeBlock = builder.createBlock(&loop.before());
            beforeBlock->addArgument(loop->getOperands().front().getType());
            auto condBuilder = OpBuilder::atBlockEnd(beforeBlock);
            auto entry =
                condBuilder.create<tensor::ExtractOp>(loc, lookupOpAdaptor.hashtable(), loop.before().getArguments());
            auto isEmpty = builder.create<CmpIOp>(loc, CmpIPredicate::eq, entry,
                                                  builder.create<ConstantIntOp>(loc, 0, entry.getType()));

            auto notFound = condBuilder.create<CmpIOp>(loc, CmpIPredicate::ne, entry, values);
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
            builder.create<linalg::YieldOp>(loc, loop->getResults());
        };

        auto linalgOp = rewriter.create<linalg::GenericOp>(loc, /*results*/ ret_type,
                                                           /*inputs*/ hashIndices, /*outputs*/ res,
                                                           /*indexing maps*/ indexing_maps,
                                                           /*iterator types*/ iter_type, lookupFunc);

        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
    }

} // namespace voila::mlir::lowering
