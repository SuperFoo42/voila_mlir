#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

#include <MLIRLoweringError.hpp>

namespace voila::mlir::lowering
{
    struct GatherOpLowering : public ::mlir::ConversionPattern
    {
        using LoopIterationFn = ::mlir::function_ref<::mlir::Value(::mlir::OpBuilder &rewriter,
                                                                   ::mlir::ValueRange memRefOperands,
                                                                   ::mlir::ValueRange loopIvs,
                                                                   ::mlir::Value iter_var,
                                                                   ::mlir::Value dest)>;
        explicit GatherOpLowering(::mlir::MLIRContext *ctx) :
            ConversionPattern(::mlir::voila::GatherOp::getOperationName(), 1, ctx)
        {
        }

        static ::mlir::MemRefType convertTensorToMemRef(::mlir::TensorType type)
        {
            assert(type.hasRank() && "expected only ranked shapes");
            return ::mlir::MemRefType::get(type.getShape(), type.getElementType());
        }

        static ::mlir::Value insertAllocAndDealloc(::mlir::MemRefType type,
                                                   ::mlir::Value dynamicMemref,
                                                   ::mlir::Location loc,
                                                   ::mlir::PatternRewriter &rewriter)
        {
            // This has to be located before the loop and after participating ops
            ::mlir::memref::AllocOp alloc;
            if (type.hasStaticShape())
            {
                alloc = rewriter.create<::mlir::memref::AllocOp>(loc, type);
            }
            else if (dynamicMemref.getType().dyn_cast<::mlir::TensorType>().isDynamicDim(0))
            {
                auto allocSize = rewriter.create<::mlir::memref::DimOp>(loc, dynamicMemref, 0);
                alloc = rewriter.create<::mlir::memref::AllocOp>(
                    loc, ::mlir::MemRefType::get(-1, type.getElementType()), ::mlir::Value(allocSize));
            }
            else
            {
                auto allocSize = rewriter.create<::mlir::ConstantIndexOp>(
                    loc, dynamicMemref.getType().dyn_cast<::mlir::TensorType>().getDimSize(0));
                alloc = rewriter.create<::mlir::memref::AllocOp>(
                    loc, ::mlir::MemRefType::get(-1, type.getElementType()), ::mlir::Value(allocSize));
            }

            // buffer deallocation instructions are added in the buffer deallocation pass
            return alloc;
        }

        static void lowerOpToLoops(::mlir::Operation *op,
                                   ::mlir::ValueRange operands,
                                   ::mlir::PatternRewriter &rewriter,
                                   LoopIterationFn processIteration)
        {
            ::mlir::voila::GatherOpAdaptor gatherOpAdaptor(operands);
            auto tensorType = (*op->result_type_begin()).dyn_cast<::mlir::TensorType>();
            auto loc = op->getLoc();
            ::mlir::Value tensorOp;
            assert(op->getOperand(0).getType().isa<::mlir::TensorType>());
            tensorOp = op->getOperand(0);

            // Insert an allocation and deallocation for the result of this operation.
            auto memRefType = convertTensorToMemRef(tensorType);
            auto alloc = insertAllocAndDealloc(memRefType, tensorOp, loc, rewriter);

            // Create a nest of affine loops, with one loop per dimension of the shape.
            // The buildAffineLoopNest function takes a callback that is used to construct
            // the body of the innermost loop given a builder, a location and a range of
            // loop induction variables.

            ::mlir::Value lowerBound = rewriter.create<::mlir::ConstantIndexOp>(loc, 0);
            ::mlir::Value upperBound;

            // find first tensor operand and use its result type
            upperBound = rewriter.create<::mlir::memref::DimOp>(loc, gatherOpAdaptor.indices(), 0);

            // start index for store
            ::mlir::SmallVector<::mlir::Value> iter_args;
            ::mlir::Value idxs;
            if (gatherOpAdaptor.indices().getType().isa<::mlir::TensorType>())
            {
                idxs = rewriter.create<::mlir::memref::BufferCastOp>(
                    loc, convertTensorToMemRef(gatherOpAdaptor.indices().getType().dyn_cast<::mlir::TensorType>()),
                    gatherOpAdaptor.indices());
            }
            else
            {
                idxs = rewriter.create<::mlir::IndexCastOp>(loc, gatherOpAdaptor.indices(), rewriter.getIndexType());
            }

            if (idxs.getType().isa<::mlir::MemRefType>())
            {
                ::mlir::SmallVector<::mlir::Value> zeroIdx;
                zeroIdx.push_back(rewriter.create<::mlir::ConstantIndexOp>(loc, 0));
                iter_args.push_back(rewriter.create<::mlir::IndexCastOp>(loc, rewriter.create<::mlir::AffineLoadOp>(loc, idxs, zeroIdx), rewriter.getIndexType()));
            }
            else
            {
                iter_args.push_back(idxs);
            }

            rewriter.create<::mlir::AffineForOp>(
                loc, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
                [&](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc,
                    ::mlir::Value iter_var /*index on which to store selected value*/, ::mlir::ValueRange ivs) -> void
                {
                    // Call the processing function with the rewriter, the memref operands,
                    // and the loop induction variables. This function will return the value
                    // to store at the current index.
                    ::mlir::Value nextIdx = processIteration(nestedBuilder, operands, iter_var, ivs.front(), alloc);
                    nestedBuilder.create<::mlir::AffineYieldOp>(loc, nextIdx);
                });

            rewriter.replaceOp(op, alloc);
        }

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            lowerOpToLoops(
                op, operands, rewriter,
                [loc](::mlir::OpBuilder &builder, ::mlir::ValueRange memRefOperands, ::mlir::ValueRange loopIvs,
                      ::mlir::Value iter_var, ::mlir::Value dest) -> ::mlir::Value
                {
                    ::mlir::voila::GatherOpAdaptor gatherOpAdaptor(memRefOperands);
                    ::mlir::Value values;
                    ::mlir::Value idxs;
                    if (gatherOpAdaptor.column().getType().isa<::mlir::TensorType>())
                    {
                        values = builder.create<::mlir::memref::BufferCastOp>(
                            loc,
                            convertTensorToMemRef(gatherOpAdaptor.column().getType().dyn_cast<::mlir::TensorType>()),
                            gatherOpAdaptor.column());
                    }
                    else
                    {
                        values = gatherOpAdaptor.column();
                    }

                    if (gatherOpAdaptor.indices().getType().isa<::mlir::TensorType>())
                    {
                        idxs = builder.create<::mlir::memref::BufferCastOp>(
                            loc,
                            convertTensorToMemRef(gatherOpAdaptor.indices().getType().dyn_cast<::mlir::TensorType>()),
                            gatherOpAdaptor.indices());
                    }
                    else
                    {
                        idxs = gatherOpAdaptor.indices();
                    }

                    if (values.getType().isa<::mlir::MemRefType>() && idxs.getType().isa<::mlir::MemRefType>())
                    {
                        // Create the binary operation performed on the loaded values.
                        auto loadedVal = builder.create<::mlir::AffineLoadOp>(loc, values, iter_var);
                        builder.create<::mlir::AffineStoreOp>(loc, loadedVal, dest, loopIvs);

                        auto idx = builder.create<::mlir::AffineLoadOp>(loc, idxs, loopIvs);
                        return builder.create<::mlir::IndexCastOp>(loc, idx, builder.getIndexType());
                    }
                    else if (values.getType().isa<::mlir::MemRefType>())
                    {
                        // Create the binary operation performed on the loaded values.
                        auto loadedVal = builder.create<::mlir::AffineLoadOp>(loc, values, iter_var);
                        builder.create<::mlir::AffineStoreOp>(loc, loadedVal, dest, loopIvs);

                        return builder.create<::mlir::IndexCastOp>(loc, idxs, builder.getIndexType());
                    }
                    else if (idxs.getType().isa<::mlir::MemRefType>())
                    {
                        // Create the binary operation performed on the loaded values.

                        builder.create<::mlir::AffineStoreOp>(loc, values, dest, loopIvs);
                        auto idx = builder.create<::mlir::AffineLoadOp>(loc, idxs, loopIvs);
                        return builder.create<::mlir::IndexCastOp>(loc, idx, builder.getIndexType());
                    }
                    else
                    {
                        // Create the binary operation performed on the loaded values.

                        builder.create<::mlir::AffineStoreOp>(loc, values, dest, loopIvs);
                        return builder.create<::mlir::IndexCastOp>(loc, idxs, builder.getIndexType());
                    }
                });
            return ::mlir::success();
        }
    };
} // namespace voila::mlir::lowering