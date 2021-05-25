#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::voila;

    template<class ConstOp>
    class ConstOpLowering : public OpRewritePattern<ConstOp>
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
            auto alloc = rewriter.create<memref::AllocOp>(loc, type);

            // Make sure to allocate at the beginning of the block.
            auto *parentBlock = alloc->getBlock();
            alloc->moveBefore(&parentBlock->front());

            // Make sure to deallocate this alloc at the end of the block. This should be fine
            // as voila functions have no control flow.
            auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
            dealloc->moveBefore(&parentBlock->back());
            return alloc;
        }
      public:
        using OpRewritePattern<ConstOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ConstOp op, PatternRewriter &rewriter) const final
        {
            auto constantValue = op.value();
            auto loc = op.getLoc();

            // When lowering the constant operation, we allocate and assign the constant
            // values to a corresponding memref allocation.
            auto tensorType = op.getType().template cast<TensorType>();
            auto memRefType = convertTensorToMemRef(tensorType);
            auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

            // We will be generating constant indices up-to the largest dimension.
            // Create these constants up-front to avoid large amounts of redundant
            // operations.
            auto valueShape = memRefType.getShape();
            assert(!valueShape.empty());
            SmallVector<Value, 1> indices;

            // TODO: in future maybe also AffineVectorStoreOp
            auto cv =  rewriter.create<ConstantOp>(loc, DenseElementsAttr::get(tensorType, constantValue));
            rewriter.create<AffineStoreOp>(loc, cv, alloc,
                                           llvm::makeArrayRef(indices));
            indices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
            // Replace this operation with the generated alloc.
            rewriter.replaceOp(op, alloc);
            return success();
        }
    };

    using IntConstOpLowering = ConstOpLowering<IntConstOp>;
    using FltConstOpLowering = ConstOpLowering<FltConstOp>;
    using BoolConstOpLowering = ConstOpLowering<BoolConstOp>;
} // namespace voila::mlir::lowering