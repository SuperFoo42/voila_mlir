#include "mlir/lowering/SelectOpLowering.hpp"
using namespace mlir;
using namespace ::voila::mlir::lowering;
MemRefType SelectOpLowering::convertTensorToMemRef(TensorType type)
{
    assert(type.hasRank() && "expected only ranked shapes");
    return MemRefType::get(type.getShape(), type.getElementType());
}

Value SelectOpLowering::insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter &rewriter)
{
    // TODO: get dynamic size of memref
    auto allocSize = rewriter.template create<ConstantIndexOp>(loc, 0);
    auto alloc = rewriter.create<memref::AllocOp>(loc, type, Value(allocSize));

    // Make sure to allocate at the beginning of the block.
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());
    allocSize->moveBefore(alloc);
    // Make sure to deallocate this alloc at the end of the block. This should be fine
    // as voila functions have no control flow.
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}

void SelectOpLowering::lowerOpToLoops(Operation *op,
                                      ValueRange operands,
                                      PatternRewriter &rewriter,
                                      LoopIterationFn processIteration)
{
    auto tensorType = (*op->result_type_begin()).template dyn_cast<TensorType>();
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // Create a nest of affine loops, with one loop per dimension of the shape.
    // The buildAffineLoopNest function takes a callback that is used to construct
    // the body of the innermost loop given a builder, a location and a range of
    // loop induction variables.

    llvm::SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
    llvm::SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
    buildAffineLoopNest(rewriter, loc, lowerBounds, tensorType.getShape(), steps,
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

LogicalResult SelectOpLowering::matchAndRewrite(Operation *op,
                                                llvm::ArrayRef<Value> operands,
                                                ConversionPatternRewriter &rewriter) const
{
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs)
                   {
                      /* // Generate an adaptor for the remapped operands of the BinaryOp. This
                       // allows for using the nice named accessors that are generated by the
                       // ODS.
                       ::mlir::voila::SelectOp::Adaptor binaryAdaptor(memRefOperands);

                       if (binaryAdaptor.values().getType().isa<MemRefType>() &&
                           binaryAdaptor.pred().getType().isa<MemRefType>())
                       {
                           // Generate loads for the element of 'lhs' and 'rhs' at the inner
                           // loop.
                           auto loadedLhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.values(), loopIvs);
                           auto loadedRhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.pred(), loopIvs);

                           // Create the binary operation performed on the loaded values.

                           //return builder.create<AffineIfOp>(loc, loadedLhs.getType(), loadedLhs, loadedRhs);
                       }
                       else if (binaryAdaptor.values().getType().isa<MemRefType>())
                       {
                           auto loadedLhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.values(), loopIvs);
                           // Create the binary operation performed on the loaded values.

                           return builder.create<AffineIfOp>(loc, binaryAdaptor.pred().getType(), loadedLhs,
                                                                  binaryAdaptor.pred());
                       }
                       else if (binaryAdaptor.pred().getType().isa<MemRefType>())
                       {
                           auto loadedRhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.pred(), loopIvs);

                           // Create the binary operation performed on the loaded values.

                           return builder.create<AffineIfOp>(loc, binaryAdaptor.values().getType(),
                                                                  binaryAdaptor.values(), loadedRhs);
                       }
                       else
                       {
                           // Create the binary operation performed on the loaded values.

                           return builder.create<AffineIfOp>(loc, binaryAdaptor.values().getType(),
                                                                  binaryAdaptor.values(), binaryAdaptor.pred());
                       }*/
                      return nullptr;
                   });
    return success();
}
