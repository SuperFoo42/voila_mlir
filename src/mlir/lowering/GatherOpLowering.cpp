#include "mlir/lowering/GatherOpLowering.hpp"


namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::GatherOp;
    using ::mlir::voila::GatherOpAdaptor;

    GatherOpLowering::GatherOpLowering(MLIRContext *ctx) : ConversionPattern(GatherOp::getOperationName(), 1, ctx) {}

    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static Value insertAllocAndDealloc(MemRefType type, Value dynamicMemref, Location loc, PatternRewriter &rewriter)
    {
        // This has to be located before the loop and after participating ops
        memref::AllocOp alloc;
        if (type.hasStaticShape())
        {
            alloc = rewriter.create<memref::AllocOp>(loc, type);
        }
        else if (dynamicMemref.getType().dyn_cast<TensorType>().isDynamicDim(0))
        {
            auto allocSize = rewriter.create<memref::DimOp>(loc, dynamicMemref, 0);
            alloc = rewriter.create<memref::AllocOp>(loc, MemRefType::get(-1, type.getElementType()), Value(allocSize));
        }
        else
        {
            auto allocSize =
                rewriter.create<ConstantIndexOp>(loc, dynamicMemref.getType().dyn_cast<TensorType>().getDimSize(0));
            alloc = rewriter.create<memref::AllocOp>(loc, MemRefType::get(-1, type.getElementType()), Value(allocSize));
        }

        // buffer deallocation instructions are added in the buffer deallocation pass
        return alloc;
    }

    static void lowerOpToLoops(Operation *op,
                               ValueRange operands,
                               PatternRewriter &rewriter,
                               GatherOpLowering::LoopIterationFn processIteration)
    {
        GatherOpAdaptor gatherOpAdaptor(operands);
        auto tensorType = (*op->result_type_begin()).dyn_cast<TensorType>();
        auto loc = op->getLoc();
        Value tensorOp;
        assert(op->getOperand(0).getType().isa<TensorType>());
        tensorOp = op->getOperand(0);

        // Insert an allocation and deallocation for the result of this operation.
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, tensorOp, loc, rewriter);

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        Value lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        Value upperBound;

        // find first tensor operand and use its result type
        upperBound = rewriter.create<memref::DimOp>(loc, gatherOpAdaptor.indices(), 0);

        // start index for store
        SmallVector<Value> iter_args;
        Value idxs;
        if (gatherOpAdaptor.indices().getType().isa<TensorType>())
        {
            idxs = rewriter.create<ToMemrefOp>(
                loc, convertTensorToMemRef(gatherOpAdaptor.indices().getType().dyn_cast<TensorType>()),
                gatherOpAdaptor.indices());
        }
        else
        {
            idxs = rewriter.create<IndexCastOp>(loc, gatherOpAdaptor.indices(), rewriter.getIndexType());
        }

        if (idxs.getType().isa<MemRefType>())
        {
            SmallVector<Value> zeroIdx;
            zeroIdx.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
            iter_args.push_back(rewriter.create<IndexCastOp>(loc, rewriter.create<AffineLoadOp>(loc, idxs, zeroIdx),
                                                             rewriter.getIndexType()));
        }
        else
        {
            iter_args.push_back(idxs);
        }

        rewriter.create<AffineForOp>(
            loc, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
            [&](OpBuilder &nestedBuilder, Location loc, Value iter_var /*index on which to store selected value*/,
                ValueRange ivs) -> void
            {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                Value nextIdx = processIteration(nestedBuilder, operands, iter_var, ivs.front(), alloc);
                nestedBuilder.create<AffineYieldOp>(loc, nextIdx);
            });

        rewriter.replaceOp(op, alloc);
    }

    LogicalResult GatherOpLowering::matchAndRewrite(Operation *op,
                                                    ArrayRef<Value> operands,
                                                    ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(
            op, operands, rewriter,
            [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs, Value iter_var,
                  Value dest) -> Value
            {
                GatherOpAdaptor gatherOpAdaptor(memRefOperands);
                Value values;
                Value idxs;
                if (gatherOpAdaptor.column().getType().isa<TensorType>())
                {
                    values = builder.create<ToMemrefOp>(
                        loc, convertTensorToMemRef(gatherOpAdaptor.column().getType().dyn_cast<TensorType>()),
                        gatherOpAdaptor.column());
                }
                else
                {
                    values = gatherOpAdaptor.column();
                }

                if (gatherOpAdaptor.indices().getType().isa<TensorType>())
                {
                    idxs = builder.create<ToMemrefOp>(
                        loc, convertTensorToMemRef(gatherOpAdaptor.indices().getType().dyn_cast<TensorType>()),
                        gatherOpAdaptor.indices());
                }
                else
                {
                    idxs = gatherOpAdaptor.indices();
                }

                if (values.getType().isa<MemRefType>() && idxs.getType().isa<MemRefType>())
                {
                    // Create the binary operation performed on the loaded values.
                    auto loadedVal = builder.create<AffineLoadOp>(loc, values, iter_var);
                    builder.create<AffineStoreOp>(loc, loadedVal, dest, loopIvs);

                    auto idx = builder.create<AffineLoadOp>(loc, idxs, loopIvs);
                    return builder.create<IndexCastOp>(loc, idx, builder.getIndexType());
                }
                else if (values.getType().isa<MemRefType>())
                {
                    // Create the binary operation performed on the loaded values.
                    auto loadedVal = builder.create<AffineLoadOp>(loc, values, iter_var);
                    builder.create<AffineStoreOp>(loc, loadedVal, dest, loopIvs);

                    return builder.create<IndexCastOp>(loc, idxs, builder.getIndexType());
                }
                else if (idxs.getType().isa<MemRefType>())
                {
                    // Create the binary operation performed on the loaded values.

                    builder.create<AffineStoreOp>(loc, values, dest, loopIvs);
                    auto idx = builder.create<AffineLoadOp>(loc, idxs, loopIvs);
                    return builder.create<IndexCastOp>(loc, idx, builder.getIndexType());
                }
                else
                {
                    // Create the binary operation performed on the loaded values.

                    builder.create<AffineStoreOp>(loc, values, dest, loopIvs);
                    return builder.create<IndexCastOp>(loc, idxs, builder.getIndexType());
                }
            });
        return success();
    }
} // namespace voila::mlir::lowering