#include "mlir/lowering/HashOpLowering.hpp"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::HashOp;
    using ::mlir::voila::HashOpAdaptor;

    HashOpLowering::HashOpLowering(MLIRContext *ctx) : ConversionPattern(HashOp::getOperationName(), 1, ctx) {}

    LogicalResult
    HashOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        HashOpAdaptor hashOpAdaptor(operands);
        auto loc = op->getLoc();
        // TODO: murmur3 for strings
        assert(hashOpAdaptor.input().getType().isa<TensorType>());

        SmallVector<Value,1> outTensorSize;
        outTensorSize.push_back(rewriter.create<tensor::DimOp>(loc, hashOpAdaptor.input(), 0));
        auto outTensor = rewriter.create<linalg::InitTensorOp>(loc, outTensorSize, rewriter.getI64Type());
        SmallVector<Value, 1> res;
        res.push_back(outTensor);

        SmallVector<StringRef, 1> iter_type;
        iter_type.push_back("parallel");

        auto fn = [](OpBuilder &builder, Location loc, ValueRange vals)
        {
            /**
             * Hash function based on splitmix
             * @link{https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h}
             * static inline uint64_t
             * splitmix64_stateless(uint64_t index) { uint64_t z = (index + UINT64_C(0x9E3779B97F4A7C15)); z = (z ^ (z
             * >> 30)) * UINT64_C(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB); return z ^ (z
             * >> 31);
             * }
             */
            auto c1 = builder.create<ConstantIntOp>(loc, 0x9E3779B97F4A7C15, builder.getI64Type());
            auto c2 = builder.create<ConstantIntOp>(loc, 0xBF58476D1CE4E5B9, builder.getI64Type());
            auto c3 = builder.create<ConstantIntOp>(loc, 0x94D049BB133111EB, builder.getI64Type());
            // init
            auto a1 = builder.create<AddIOp>(loc, vals.front(), c1);
            // first mix
            auto s1 = builder.create<UnsignedShiftRightOp>(
                loc, a1, builder.create<ConstantIntOp>(loc, 30, builder.getI64Type()));
            auto x1 = builder.create<XOrOp>(loc, a1, s1);
            auto m1 = builder.create<MulIOp>(loc, x1, c2);
            // second mix
            auto s2 = builder.create<UnsignedShiftRightOp>(
                loc, m1, builder.create<ConstantIntOp>(loc, 27, builder.getI64Type()));
            auto x2 = builder.create<XOrOp>(loc, m1, s2);
            auto m2 = builder.create<MulIOp>(loc, x2, c3);
            // finalize
            auto s3 = builder.create<UnsignedShiftRightOp>(
                loc, m2, builder.create<ConstantIntOp>(loc, 31, builder.getI64Type()));
            SmallVector<Value, 1> res;
            res.push_back(builder.create<XOrOp>(loc, m2, s3));

            builder.create<linalg::YieldOp>(loc, res);
        };

        SmallVector<Type, 1> ret_type;
        ret_type.push_back(outTensor.getType());
        SmallVector<AffineMap, 2> indexing_maps;
        indexing_maps.push_back(rewriter.getDimIdentityMap());
        indexing_maps.push_back(rewriter.getDimIdentityMap());

        auto linalgOp = rewriter.create<linalg::GenericOp>(loc, /*results*/ ret_type,
                                                       /*inputs*/ operands, /*outputs*/ res,
                                                       /*indexing maps*/ indexing_maps,
                                                       /*iterator types*/ iter_type, fn);

        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
    }
} // namespace voila::mlir::lowering