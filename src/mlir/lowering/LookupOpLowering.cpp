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
    ::mlir::LogicalResult LookupOpLowering::matchAndRewrite(::mlir::Operation *op,
                                                            llvm::ArrayRef<::mlir::Value> operands,
                                                            ConversionPatternRewriter &rewriter) const
    {
        LookupOpAdaptor lookupOpAdaptor(operands);
        auto loc = op->getLoc();

        auto htSize = rewriter.create<tensor::DimOp>(loc, lookupOpAdaptor.hashtable(), 0);
        auto hashValues = rewriter.create<::mlir::voila::HashOp>(loc, RankedTensorType::get(-1, rewriter.getI64Type()),
                                                                 lookupOpAdaptor.keys());
        auto modSize =
            rewriter.create<SubIOp>(loc, htSize, rewriter.create<ConstantIntOp>(loc, 1, rewriter.getI64Type()));
        SmallVector<Value, 1> mappedHashVals;
        mappedHashVals.push_back(rewriter.create<::mlir::voila::AndOp>(loc, hashValues.getType(), hashValues, modSize));

        SmallVector<Value, 1> resSize;
        resSize.push_back(rewriter.create<tensor::DimOp>(loc, hashValues, 0));
        SmallVector<Value, 1> results;
        results.push_back(rewriter.create<linalg::InitTensorOp>(loc, resSize, rewriter.getIndexType()));
        SmallVector<Type, 1> ret_type;
        ret_type.push_back(results.back().getType());
        SmallVector<AffineMap, 2> indexing_maps;
        indexing_maps.push_back(rewriter.getDimIdentityMap());
        indexing_maps.push_back(rewriter.getDimIdentityMap());
        SmallVector<StringRef, 1> iter_type;
        iter_type.push_back("parallel");
        auto loopBody = [&lookupOpAdaptor](OpBuilder &builder, Location loc, ValueRange vals)
        {
            // probing
            SmallVector<Type, 1> resTypes;
            resTypes.push_back(builder.getIndexType());
            // probing with do while
            // FIXME: this probing runs over the end of the hashtable, if no empty bucket or matching key is found.
            // Maybe we should keep track of the max probing count during generation and iterate only so many times
            auto loop = builder.create<scf::WhileOp>(loc, resTypes, vals);
            // condition
            auto condBuilder = OpBuilder(loop.before());
            auto entry =
                condBuilder.create<memref::LoadOp>(loc, lookupOpAdaptor.hashtable(), loop.before().getArguments());
            auto isEmpty = builder.create<CmpIOp>(loc, CmpIPredicate::eq, entry,
                                                  builder.create<ConstantIntOp>(loc, 0, builder.getI64Type()));
            SmallVector<Value, 1> curIdx;
            curIdx.push_back(builder.create<linalg::IndexOp>(loc, 0));
            auto key = builder.create<tensor::ExtractOp>(loc, lookupOpAdaptor.keys(), curIdx);
            auto notFound = builder.create<CmpIOp>(loc, CmpIPredicate::ne, entry, key);
            builder.create<scf::ConditionOp>(loc, builder.create<OrOp>(loc, builder.getI1Type(), isEmpty, notFound),
                                             loop.before().getArguments());
            // body
            auto bodyBuilder = OpBuilder(loop.after());
            SmallVector<Value, 1> inc;
            inc.push_back(bodyBuilder.create<AddIOp>(loc, loop.getAfterArguments().front(),
                                                     builder.create<ConstantIntOp>(loc, 1, builder.getI64Type())));
            bodyBuilder.create<scf::YieldOp>(loc, inc);
            // result
            builder.create<linalg::YieldOp>(loc, loop->getResults());
        };

        auto linalgOp = rewriter.create<::mlir::linalg::GenericOp>(loc, /*results*/ ret_type, /*inputs*/ mappedHashVals,
                                                                   /*outputs*/ results, /*indexing maps*/ indexing_maps,
                                                                   /*iterator types*/ iter_type, loopBody);

        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
    }

} // namespace voila::mlir::lowering
