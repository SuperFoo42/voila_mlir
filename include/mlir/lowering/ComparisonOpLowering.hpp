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

    template<typename CmpOp, typename LoweredBinaryOp>
    class ComparisonOpLowering : public ConversionPattern
    {
        static constexpr auto getCmpPred()
        {
            if constexpr (std::is_same_v<LoweredBinaryOp, CmpIOp>)
            {
                if constexpr (std::is_same_v<CmpOp, EqOp>)
                    return CmpIPredicate::eq;
                else if constexpr (std::is_same_v<CmpOp, NeqOp>)
                    return CmpIPredicate::ne;
                else if constexpr (std::is_same_v<CmpOp, LeOp>)
                    return CmpIPredicate::slt;
                else if constexpr (std::is_same_v<CmpOp, LeqOp>)
                    return CmpIPredicate::sle;
                else if constexpr (std::is_same_v<CmpOp, GeqOp>)
                    return CmpIPredicate::sge;
                else if constexpr (std::is_same_v<CmpOp, GeOp>)
                    return CmpIPredicate::sgt;
                else
                    throw std::logic_error("Sth. went wrong");
            }
            else if constexpr (std::is_same_v<LoweredBinaryOp, CmpFOp>)
            {
                if constexpr (std::is_same_v<CmpOp, EqOp>)
                    return CmpFPredicate::OEQ;
                else if constexpr (std::is_same_v<CmpOp, NeqOp>)
                    return CmpFPredicate::ONE;
                else if constexpr (std::is_same_v<CmpOp, LeOp>)
                    return CmpFPredicate::OLT;
                else if constexpr (std::is_same_v<CmpOp, LeqOp>)
                    return CmpFPredicate::OLE;
                else if constexpr (std::is_same_v<CmpOp, GeqOp>)
                    return CmpFPredicate::OGE;
                else if constexpr (std::is_same_v<CmpOp, GeOp>)
                    return CmpFPredicate::OGT;
                else
                    throw std::logic_error("Sth. went wrong");
            }
            else
                throw std::logic_error("Sth. went wrong");
        }

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

      public:
        explicit ComparisonOpLowering(MLIRContext *ctx) : ConversionPattern(CmpOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, llvm::ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            lowerOpToLoops(
                op, operands, rewriter,
                [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs)
                {
                    // Generate an adaptor for the remapped operands of the BinaryOp. This
                    // allows for using the nice named accessors that are generated by the
                    // ODS.
                    typename CmpOp::Adaptor binaryAdaptor(memRefOperands);

                    if (binaryAdaptor.lhs().getType().template isa<MemRefType>() &&
                        binaryAdaptor.rhs().getType().template isa<MemRefType>())
                    {
                        // Generate loads for the element of 'lhs' and 'rhs' at the inner
                        // loop.
                        auto loadedLhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
                        auto loadedRhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

                        // Create the binary operation performed on the loaded values.

                        return builder.create<LoweredBinaryOp>(loc, getCmpPred(), loadedLhs, loadedRhs);
                    }
                    else if (binaryAdaptor.lhs().getType().template isa<MemRefType>())
                    {
                        auto loadedLhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
                        // Create the binary operation performed on the loaded values.

                        return builder.create<LoweredBinaryOp>(loc, getCmpPred(), loadedLhs, binaryAdaptor.rhs());
                    }
                    else if (binaryAdaptor.rhs().getType().template isa<MemRefType>())
                    {
                        auto loadedRhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

                        // Create the binary operation performed on the loaded values.

                        return builder.create<LoweredBinaryOp>(loc, getCmpPred(), binaryAdaptor.lhs(), loadedRhs);
                    }
                    else
                    {
                        // Create the binary operation performed on the loaded values.

                        return builder.create<LoweredBinaryOp>(loc, getCmpPred(), binaryAdaptor.lhs(),
                                                               binaryAdaptor.rhs());
                    }
                });
            return success();
        }
    };

    using EqIOpLowering = ComparisonOpLowering<EqOp, ::mlir::CmpIOp>;
    using NeqIOpLowering = ComparisonOpLowering<NeqOp, ::mlir::CmpIOp>;
    using LeIOpLowering = ComparisonOpLowering<LeOp, ::mlir::CmpIOp>;
    using LeqIOpLowering = ComparisonOpLowering<LeqOp, ::mlir::CmpIOp>;
    using GeIOpLowering = ComparisonOpLowering<GeOp, ::mlir::CmpIOp>;
    using GeqIOpLowering = ComparisonOpLowering<GeqOp, ::mlir::CmpIOp>;

    using EqFOpLowering = ComparisonOpLowering<EqOp, ::mlir::CmpFOp>;
    using NeqFOpLowering = ComparisonOpLowering<NeqOp, ::mlir::CmpFOp>;
    using LeFOpLowering = ComparisonOpLowering<LeOp, ::mlir::CmpFOp>;
    using LeqFOpLowering = ComparisonOpLowering<LeqOp, ::mlir::CmpFOp>;
    using GeFOpLowering = ComparisonOpLowering<GeOp, ::mlir::CmpFOp>;
    using GeqFOpLowering = ComparisonOpLowering<GeqOp, ::mlir::CmpFOp>;
} // namespace voila::mlir::lowering