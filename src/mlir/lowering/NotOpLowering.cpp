#include "mlir/lowering/NotOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::NotOp;
    using ::mlir::voila::NotOpAdaptor;

    NotOpLowering::NotOpLowering(MLIRContext *ctx) : ConversionPattern(NotOp::getOperationName(), 1, ctx) {}

    /// Convert the given TensorType into the corresponding MemRefType.
    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    /// Insert an allocation and deallocation for the given MemRefType.
    static ::mlir::Value
    insertAllocAndDealloc(MemRefType type, Value dynamicMemref, Location loc, PatternRewriter &rewriter)
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
            alloc = rewriter.create<memref::AllocOp>(loc, type, Value(allocSize));
        }
        else
        {
            auto allocSize =
                rewriter.create<ConstantIndexOp>(loc, dynamicMemref.getType().dyn_cast<TensorType>().getDimSize(0));
            alloc = rewriter.create<memref::AllocOp>(loc, type, Value(allocSize));
        }

        // buffer deallocation instructions are added in the buffer deallocation pass
        return alloc;
    }

    /// This defines the function type used to process an iteration of a lowered
    /// loop. It takes as input an OpBuilder, an range of memRefOperands
    /// corresponding to the operands of the input operation, and the range of loop
    /// induction variables for the iteration. It returns a value to store at the
    /// current index of the iteration.
    static void lowerOpToLoops(Operation *op,
                               ValueRange operands,
                               PatternRewriter &rewriter,
                               NotOpLowering::LoopIterationFn processIteration)
    {
        auto tensorType = (*op->result_type_begin()).template dyn_cast<TensorType>();
        auto loc = op->getLoc();

        // Insert an allocation and deallocation for the result of this operation.
        auto memRefType = convertTensorToMemRef(tensorType);

        Value tensorOp;
        for (auto operand : op->getOperands())
        {
            if (operand.getType().template isa<TensorType>() ||
                operand.getType().template isa<MemRefType>())
            {
                tensorOp = operand;
                break;
            }
        }

        auto alloc = insertAllocAndDealloc(memRefType, tensorOp, loc, rewriter);

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        auto lb = rewriter.template create<ConstantIndexOp>(loc, 0);
        SmallVector<Value> lowerBounds(tensorType.getRank(), lb);
        SmallVector<Value> upperBounds;

        // find first tensor operand and use its result type
        // TODO: this does not look like a clean solution

        for (auto dim = 0; dim < tensorOp.getType().template dyn_cast<TensorType>().getRank(); ++dim)
        {
            if (tensorOp.getType().template dyn_cast<TensorType>().isDynamicDim(dim))
            {
                upperBounds.push_back(rewriter.template create<memref::DimOp>(loc, tensorOp, dim));
            }
            else
            {
                upperBounds.push_back(rewriter.template create<ConstantIndexOp>(
                    loc, tensorOp.getType().template dyn_cast<TensorType>().getDimSize(dim)));
            }
        }

        SmallVector<int64_t> steps(tensorType.getRank(), 1);

        buildAffineLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
                            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs)
                            {
                                // Call the processing function with the rewriter, the memref operands,
                                // and the loop induction variables. This function will return the value
                                // to store at the current index.
                                Value valueToStore = processIteration(nestedBuilder, operands, ivs);
                                nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
                            });

        // Replace this operation with the generated alloc.
        rewriter.replaceOp(op, alloc);
    }

    LogicalResult
    NotOpLowering::matchAndRewrite(Operation *op,
                                                          ArrayRef<Value> operands,
                                                          ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(
            op, operands, rewriter,
            [loc](OpBuilder &builder, ValueRange memRefOperands,
                  ValueRange loopIvs) -> Value
            {
                // Generate an adaptor for the remapped operands of the BinaryOp. This
                // allows for using the nice named accessors that are generated by the
                // ODS.
                NotOpAdaptor binaryAdaptor(memRefOperands);

                Value value;
                if (binaryAdaptor.value().getType().isa<TensorType>())
                {
                    value = builder.create<memref::BufferCastOp>(
                        loc,
                        convertTensorToMemRef(binaryAdaptor.value().getType().template dyn_cast<TensorType>()),
                        binaryAdaptor.value());
                }
                else if (binaryAdaptor.value().getType().template isa<MemRefType>())
                {
                    value = binaryAdaptor.value();
                }
                else
                {
                    value = binaryAdaptor.value();
                }

                auto oneConst = builder.create<ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), 1));
                if (value.getType().isa<MemRefType>())
                {
                    auto loadedLhs = builder.create<AffineLoadOp>(loc, value, loopIvs);
                    // Create the binary operation performed on the loaded values.
                    return builder.create<XOrOp>(loc, loadedLhs, oneConst);
                }
                else
                {
                    // Create the binary operation performed on the loaded values.
                    return builder.create<XOrOp>(loc, binaryAdaptor.value().getType(), binaryAdaptor.value(),
                                                         oneConst);
                }
            });
        return success();
    }
} // namespace voila::mlir::lowering