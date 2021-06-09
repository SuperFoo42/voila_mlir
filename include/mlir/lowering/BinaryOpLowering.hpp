#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::voila;

    using LoopIterationFn = function_ref<Value(OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

    template<typename BinaryOp, typename LoweredBinaryOp>
    class BinaryOpLowering : public ConversionPattern
    {
        /// Convert the given TensorType into the corresponding MemRefType.
        static MemRefType convertTensorToMemRef(TensorType type)
        {
            assert(type.hasRank() && "expected only ranked shapes");
            return MemRefType::get(type.getShape(), type.getElementType());
        }

        /// Insert an allocation and deallocation for the given MemRefType.
        static Value insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter &rewriter)
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

        /// This defines the function type used to process an iteration of a lowered
        /// loop. It takes as input an OpBuilder, an range of memRefOperands
        /// corresponding to the operands of the input operation, and the range of loop
        /// induction variables for the iteration. It returns a value to store at the
        /// current index of the iteration.
        static void
        lowerOpToLoops(Operation *op, ValueRange operands, PatternRewriter &rewriter, LoopIterationFn processIteration)
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

            auto lb = rewriter.template create<ConstantIndexOp>(loc, 0);
            llvm::SmallVector<Value> lowerBounds(tensorType.getRank(), lb);
            llvm::SmallVector<Value> upperBounds;

            //find first tensor operand and use its result type
            //TODO: this does not look like a clean solution
            Value tensorOp;
            for (auto operand : op->getOperands())
            {
                if (operand.getType().template isa<TensorType>() || operand.getType().template isa<MemRefType>())
                {
                    tensorOp = operand;
                    break;
                }
            }

            for (auto dim = 0; dim < tensorOp.getType().template dyn_cast<TensorType>().getRank(); ++dim)
            {
                if (tensorOp.getType().template dyn_cast<TensorType>().isDynamicDim(dim))
                {
                    upperBounds.push_back(rewriter.template create<memref::DimOp>(loc, tensorOp, dim));
                }
                else
                {
                    upperBounds.push_back(rewriter.template create<ConstantIndexOp>(loc, tensorOp.getType().template dyn_cast<TensorType>().getDimSize(dim)));
                }
            }

            llvm::SmallVector<int64_t> steps(tensorType.getRank(), 1);

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

      public:
        explicit BinaryOpLowering(MLIRContext *ctx) : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, llvm::ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            lowerOpToLoops(op, operands, rewriter,
                           [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs)
                           {
                               // Generate an adaptor for the remapped operands of the BinaryOp. This
                               // allows for using the nice named accessors that are generated by the
                               // ODS.
                               typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                               if (binaryAdaptor.lhs().getType().template isa<MemRefType>() &&
                                   binaryAdaptor.rhs().getType().template isa<MemRefType>())
                               {
                                   // Generate loads for the element of 'lhs' and 'rhs' at the inner
                                   // loop.
                                   auto loadedLhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
                                   auto loadedRhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

                                   // Create the binary operation performed on the loaded values.

                                   return builder.create<LoweredBinaryOp>(loc, loadedLhs.getType(), loadedLhs,
                                                                          loadedRhs);
                               }
                               else if (binaryAdaptor.lhs().getType().template isa<MemRefType>())
                               {
                                   auto loadedLhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
                                   // Create the binary operation performed on the loaded values.

                                   return builder.create<LoweredBinaryOp>(loc, binaryAdaptor.rhs().getType(), loadedLhs,
                                                                          binaryAdaptor.rhs());
                               }
                               else if (binaryAdaptor.rhs().getType().template isa<MemRefType>())
                               {
                                   auto loadedRhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

                                   // Create the binary operation performed on the loaded values.

                                   return builder.create<LoweredBinaryOp>(loc, binaryAdaptor.lhs().getType(),
                                                                          binaryAdaptor.lhs(), loadedRhs);
                               }
                               else
                               {
                                   // Create the binary operation performed on the loaded values.

                                   return builder.create<LoweredBinaryOp>(loc, binaryAdaptor.lhs().getType(),
                                                                          binaryAdaptor.lhs(), binaryAdaptor.rhs());
                               }
                           });
            return success();
        }
    };
} // namespace voila::mlir::lowering