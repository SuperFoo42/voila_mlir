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


    template<typename CmpOp, typename LoweredBinaryOp>
    class ComparisonOpLowering : public ::mlir::ConversionPattern
    {
        using LoopIterationFn = ::mlir::function_ref<
            ::mlir::Value(::mlir::OpBuilder &rewriter, ::mlir::ValueRange memRefOperands, ::mlir::ValueRange loopIvs)>;
        static constexpr auto getCmpPred()
        {
            if constexpr (std::is_same_v<LoweredBinaryOp, ::mlir::CmpIOp>)
            {
                if constexpr (std::is_same_v<CmpOp, ::mlir::voila::EqOp>)
                    return ::mlir::CmpIPredicate::eq;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::NeqOp>)
                    return ::mlir::CmpIPredicate::ne;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeOp>)
                    return ::mlir::CmpIPredicate::slt;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeqOp>)
                    return ::mlir::CmpIPredicate::sle;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeqOp>)
                    return ::mlir::CmpIPredicate::sge;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeOp>)
                    return ::mlir::CmpIPredicate::sgt;
                else
                    throw std::logic_error("Sth. went wrong");
            }
            else if constexpr (std::is_same_v<LoweredBinaryOp, ::mlir::CmpFOp>)
            {
                if constexpr (std::is_same_v<CmpOp, ::mlir::voila::EqOp>)
                    return ::mlir::CmpFPredicate::OEQ;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::NeqOp>)
                    return ::mlir::CmpFPredicate::ONE;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeOp>)
                    return ::mlir::CmpFPredicate::OLT;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::LeqOp>)
                    return ::mlir::CmpFPredicate::OLE;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeqOp>)
                    return ::mlir::CmpFPredicate::OGE;
                else if constexpr (std::is_same_v<CmpOp, ::mlir::voila::GeOp>)
                    return ::mlir::CmpFPredicate::OGT;
                else
                    throw std::logic_error("Sth. went wrong");
            }
            else
                throw std::logic_error("Sth. went wrong");
        }
/// Convert the given TensorType into the corresponding MemRefType.
        static ::mlir::MemRefType convertTensorToMemRef(::mlir::TensorType type)
        {
            assert(type.hasRank() && "expected only ranked shapes");
            return ::mlir::MemRefType::get(type.getShape(), type.getElementType());
        }

        /// Insert an allocation and deallocation for the given MemRefType.
        static ::mlir::Value insertAllocAndDealloc(::mlir::MemRefType type, ::mlir::Value dynamicMemref, ::mlir::Location loc, ::mlir::PatternRewriter &rewriter)
        {
            //This has to be located before the loop and after participating ops
            ::mlir::memref::AllocOp alloc;
            if (dynamicMemref.getType().dyn_cast<::mlir::TensorType>().isDynamicDim(0))
            {
                auto allocSize = rewriter.create<::mlir::memref::DimOp>(loc, dynamicMemref, 0);
                alloc = rewriter.create<::mlir::memref::AllocOp>(loc, type, ::mlir::Value(allocSize));

            }
            else
            {
                auto allocSize = rewriter.create<::mlir::ConstantIndexOp>(
                    loc, dynamicMemref.getType().dyn_cast<::mlir::TensorType>().getDimSize(0));
                alloc = rewriter.create<::mlir::memref::AllocOp>(loc, type, ::mlir::Value(allocSize));
            }

            // buffer deallocation instructions are added in the buffer deallocation pass
            return alloc;
        }

        /// This defines the function type used to process an iteration of a lowered
        /// loop. It takes as input an OpBuilder, an range of memRefOperands
        /// corresponding to the operands of the input operation, and the range of loop
        /// induction variables for the iteration. It returns a value to store at the
        /// current index of the iteration.
        static void
        lowerOpToLoops(::mlir::Operation *op, ::mlir::ValueRange operands, ::mlir::PatternRewriter &rewriter, LoopIterationFn processIteration)
        {
            auto tensorType = (*op->result_type_begin()).template dyn_cast<::mlir::TensorType>();
            auto loc = op->getLoc();

            // Insert an allocation and deallocation for the result of this operation.
            auto memRefType = convertTensorToMemRef(tensorType);

            ::mlir::Value tensorOp;
            for (auto operand : op->getOperands())
            {
                if (operand.getType().template isa<::mlir::TensorType>() || operand.getType().template isa<::mlir::MemRefType>())
                {
                    tensorOp = operand;
                    break;
                }
            }

            auto alloc = insertAllocAndDealloc(memRefType,tensorOp, loc, rewriter);

            // Create a nest of affine loops, with one loop per dimension of the shape.
            // The buildAffineLoopNest function takes a callback that is used to construct
            // the body of the innermost loop given a builder, a location and a range of
            // loop induction variables.

            auto lb = rewriter.template create<::mlir::ConstantIndexOp>(loc, 0);
            llvm::SmallVector<::mlir::Value> lowerBounds(tensorType.getRank(), lb);
            llvm::SmallVector<::mlir::Value> upperBounds;

            //find first tensor operand and use its result type
            //TODO: this does not look like a clean solution

            for (auto dim = 0; dim <tensorOp.getType().template dyn_cast<::mlir::TensorType>().getRank(); ++dim)
            {
                if (tensorOp.getType().template dyn_cast<::mlir::TensorType>().isDynamicDim(dim))
                {
                    upperBounds.push_back(rewriter.template create<::mlir::memref::DimOp>(loc, tensorOp, dim));
                }
                else
                {
                    upperBounds.push_back(rewriter.template create<::mlir::ConstantIndexOp>(loc, tensorOp.getType().template dyn_cast<::mlir::TensorType>().getDimSize(dim)));
                }
            }

            llvm::SmallVector<int64_t> steps(tensorType.getRank(), 1);

            buildAffineLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
                                [&](::mlir::OpBuilder &nestedBuilder, ::mlir::Location loc, ::mlir::ValueRange ivs)
                                {
                                  // Call the processing function with the rewriter, the memref operands,
                                  // and the loop induction variables. This function will return the value
                                  // to store at the current index.
                                  ::mlir::Value valueToStore = processIteration(nestedBuilder, operands, ivs);
                                  nestedBuilder.create<::mlir::AffineStoreOp>(loc, valueToStore, alloc, ivs);
                                });

            // Replace this operation with the generated alloc.
            rewriter.replaceOp(op, alloc);
        }
      public:
        explicit ComparisonOpLowering(::mlir::MLIRContext *ctx) : ConversionPattern(CmpOp::getOperationName(), 1, ctx) {}

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            lowerOpToLoops(
                op, operands, rewriter,
                [loc](::mlir::OpBuilder &builder, ::mlir::ValueRange memRefOperands, ::mlir::ValueRange loopIvs)
                {
                    // Generate an adaptor for the remapped operands of the BinaryOp. This
                    // allows for using the nice named accessors that are generated by the
                    // ODS.
                    typename CmpOp::Adaptor binaryAdaptor(memRefOperands);

                    if (binaryAdaptor.lhs().getType().template isa<::mlir::MemRefType>() &&
                        binaryAdaptor.rhs().getType().template isa<::mlir::MemRefType>())
                    {
                        // Generate loads for the element of 'lhs' and 'rhs' at the inner
                        // loop.
                        auto loadedLhs = builder.create<::mlir::AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
                        auto loadedRhs = builder.create<::mlir::AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

                        // Create the binary operation performed on the loaded values.

                        return builder.create<LoweredBinaryOp>(loc, loadedLhs.getType(), getCmpPred(), loadedLhs,
                                                               loadedRhs);
                    }
                    else if (binaryAdaptor.lhs().getType().template isa<::mlir::MemRefType>())
                    {
                        auto loadedLhs = builder.create<::mlir::AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
                        // Create the binary operation performed on the loaded values.

                        return builder.create<LoweredBinaryOp>(loc, binaryAdaptor.rhs().getType(), getCmpPred(),
                                                               loadedLhs, binaryAdaptor.rhs());
                    }
                    else if (binaryAdaptor.rhs().getType().template isa<::mlir::MemRefType>())
                    {
                        auto loadedRhs = builder.create<::mlir::AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

                        // Create the binary operation performed on the loaded values.

                        return builder.create<LoweredBinaryOp>(loc, binaryAdaptor.lhs().getType(), getCmpPred(),
                                                               binaryAdaptor.lhs(), loadedRhs);
                    }
                    else
                    {
                        // Create the binary operation performed on the loaded values.

                        return builder.create<LoweredBinaryOp>(loc, binaryAdaptor.lhs().getType(), getCmpPred(),
                                                               binaryAdaptor.lhs(), binaryAdaptor.rhs());
                    }
                });
            return ::mlir::success();
        }
    };

    using EqIOpLowering = ComparisonOpLowering<::mlir::voila::EqOp, ::mlir::CmpIOp>;
    using NeqIOpLowering = ComparisonOpLowering<::mlir::voila::NeqOp, ::mlir::CmpIOp>;
    using LeIOpLowering = ComparisonOpLowering<::mlir::voila::LeOp, ::mlir::CmpIOp>;
    using LeqIOpLowering = ComparisonOpLowering<::mlir::voila::LeqOp, ::mlir::CmpIOp>;
    using GeIOpLowering = ComparisonOpLowering<::mlir::voila::GeOp, ::mlir::CmpIOp>;
    using GeqIOpLowering = ComparisonOpLowering<::mlir::voila::GeqOp, ::mlir::CmpIOp>;

    using EqFOpLowering = ComparisonOpLowering<::mlir::voila::EqOp, ::mlir::CmpFOp>;
    using NeqFOpLowering = ComparisonOpLowering<::mlir::voila::NeqOp, ::mlir::CmpFOp>;
    using LeFOpLowering = ComparisonOpLowering<::mlir::voila::LeOp, ::mlir::CmpFOp>;
    using LeqFOpLowering = ComparisonOpLowering<::mlir::voila::LeqOp, ::mlir::CmpFOp>;
    using GeFOpLowering = ComparisonOpLowering<::mlir::voila::GeOp, ::mlir::CmpFOp>;
    using GeqFOpLowering = ComparisonOpLowering<::mlir::voila::GeqOp, ::mlir::CmpFOp>;
} // namespace voila::mlir::lowering