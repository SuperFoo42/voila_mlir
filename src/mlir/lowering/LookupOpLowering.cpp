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
        ::mlir::Value outTensor;
        if (lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>().hasStaticShape())
        {
            outTensor = rewriter.create<memref::AllocOp>(
                loc, convertTensorToMemRef(lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>()));
        }
        else
        {
            SmallVector<Value, 1> outTensorSize;
            outTensorSize.push_back(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashes(), 0));
            outTensor = rewriter.create<memref::AllocOp>(
                loc, convertTensorToMemRef(lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>()), outTensorSize);
        }

        auto htSize = rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashtable(), 0);
        auto modSize = rewriter.create<SubIOp>(loc, htSize, rewriter.create<ConstantIndexOp>(loc, 1));
        auto intMod = rewriter.create<IndexCastOp>(loc, modSize, rewriter.getI64Type());

        auto lookupFunc = [&intMod, &lookupOpAdaptor, &outTensor](OpBuilder &builder, Location loc, ValueRange vals)
        {
            auto hashVals = builder.create<AffineLoadOp>(
                loc,
                builder.create<memref::BufferCastOp>(
                    loc, convertTensorToMemRef(lookupOpAdaptor.hashes().getType().dyn_cast<TensorType>()),
                    lookupOpAdaptor.hashes()),
                vals.front());

            // TODO: lookup in every variadic memref
            auto values = builder.create<AffineLoadOp>(
                loc,
                builder.create<memref::BufferCastOp>(
                    loc, convertTensorToMemRef(lookupOpAdaptor.values().getType().dyn_cast<TensorType>()),
                    lookupOpAdaptor.values()),
                vals.front());

            // probing
            SmallVector<Type, 1> resTypes;
            resTypes.push_back(builder.getI64Type());
            SmallVector<Value, 1> hashVal;
            hashVal.push_back(builder.create<AndOp>(loc, hashVals, intMod));
            // probing with do while
            // Maybe we should keep track of the max probing count during generation and iterate only so many times
            auto loop = builder.create<scf::WhileOp>(loc, resTypes, hashVal);
            // condition

            auto beforeBlock = builder.createBlock(&loop.before());
            beforeBlock->addArgument(loop->getOperands().front().getType());
            auto condBuilder = OpBuilder::atBlockEnd(beforeBlock);
            SmallVector<Value, 1> idx;
            idx.push_back(condBuilder.create<IndexCastOp>(loc, loop.before().getArgument(0), builder.getIndexType()));
            auto entry = condBuilder.create<memref::LoadOp>(loc, builder.create<memref::BufferCastOp>(loc, convertTensorToMemRef(lookupOpAdaptor.hashtable().getType().dyn_cast<TensorType>()),lookupOpAdaptor.hashtable()), idx);
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
            inc.push_back(bodyBuilder.create<AndOp>(
                loc,
                bodyBuilder.create<AddIOp>(loc, loop.getAfterArguments().front(),
                                           builder.create<ConstantIntOp>(loc, 1, builder.getI64Type())),
                intMod));
            bodyBuilder.create<scf::YieldOp>(loc, inc);
            builder.setInsertionPointAfter(loop);
            // store result
            builder.create<AffineStoreOp>(loc, loop->getResults().front(), outTensor, vals.front());
        };

        SmallVector<Value, 1> lb, ub;
        lb.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        ub.push_back(rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashes(), 0));

        buildAffineLoopNest(rewriter, loc, lb, ub, 1, lookupFunc);

        rewriter.replaceOpWithNewOp<memref::TensorLoadOp>(op, outTensor);
        return success();
    }

} // namespace voila::mlir::lowering
