#include "mlir/Dialects/Voila/lowering/NotOpLowering.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"      // for Value
#include "mlir/IR/ValueRange.h" // for ValueRange
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include <cassert>
#include <cstdint>

namespace mlir
{
    class OpBuilder;
} // namespace mlir

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using ::mlir::voila::NotOp;
    using ::mlir::voila::NotOpAdaptor;

    /// Insert an allocation and deallocation for the given MemRefType.
    static Value insertAllocAndDealloc(MemRefType type, Value dynamicMemref, Location, ImplicitLocOpBuilder &builder)
    {
        // This has to be located before the loop and after participating ops
        if (type.hasStaticShape())
        {
            return builder.create<memref::AllocOp>(type);
        }
        else if (asShapedType(dynamicMemref).isDynamicDim(0))
        {
            auto allocSize = builder.create<memref::DimOp>(dynamicMemref, 0);
            return builder.create<memref::AllocOp>(type, Value(allocSize));
        }
        else
        {
            auto allocSize = builder.create<ConstantIndexOp>(asShapedType(dynamicMemref).getDimSize(0));
            return builder.create<memref::AllocOp>(type, Value(allocSize));
        }
    }

    /// This defines the function type used to process an iteration of a lowered
    /// loop. It takes as input an OpBuilder, an range of memRefOperands
    /// corresponding to the operands of the input operation, and the range of loop
    /// induction variables for the iteration. It returns a value to store at the
    /// current index of the iteration.
    static void lowerOpToLoops(NotOp op, PatternRewriter &rewriter, NotOpLowering::LoopIterationFn processIteration)
    {
        auto tensorType = asTensorType(*op->result_type_begin());
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);
        // Insert an allocation and deallocation for the result of this operation.
        auto memRefType = convertTensorToMemRef(tensorType);

        Value tensorOp;
        for (auto operand : op->getOperands())
        {
            if (isTensor(operand) || isMemRef(operand))
            {
                tensorOp = operand;
                break;
            }
        }

        auto alloc = insertAllocAndDealloc(memRefType, tensorOp, loc, builder);

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        auto lb = builder.template create<ConstantIndexOp>(0);
        SmallVector<Value> lowerBounds(tensorType.getRank(), lb);
        SmallVector<Value> upperBounds;

        // find first tensor operand and use its result type
        // TODO: this does not look like a clean solution

        for (auto dim = 0; dim < tensorOp.getType().template dyn_cast<TensorType>().getRank(); ++dim)
        {
            if (asShapedType(tensorOp).isDynamicDim(dim))
            {
                upperBounds.push_back(builder.template create<memref::DimOp>(tensorOp, dim));
            }
            else
            {
                upperBounds.push_back(builder.template create<ConstantIndexOp>(
                    tensorOp.getType().template dyn_cast<TensorType>().getDimSize(dim)));
            }
        }

        SmallVector<int64_t> steps(tensorType.getRank(), 1);

        buildAffineLoopNest(builder, loc, lowerBounds, upperBounds, steps,
                            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs)
                            {
                                // Call the processing function with the rewriter, the memref operands,
                                // and the loop induction variables. This function will return the value
                                // to store at the current index.
                                ImplicitLocOpBuilder builder(loc, nestedBuilder);
                                Value valueToStore = processIteration(builder, op, ivs);
                                builder.create<AffineStoreOp>(valueToStore, alloc, ivs);
                            });

        // Replace this operation with the generated alloc.
        rewriter.replaceOp(op, alloc);
    }

    LogicalResult NotOpLowering::matchAndRewrite(::mlir::voila::NotOp op,
                                                 OpAdaptor adaptor,
                                                 ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(op, rewriter,
                       [loc](ImplicitLocOpBuilder &builder, NotOp op, ValueRange loopIvs) -> Value
                       {
                           // Generate an adaptor for the remapped operands of the BinaryOp. This
                           // allows for using the nice named accessors that are generated by the
                           // ODS.

                           Value value;
                           if (isTensor(op.getValue()))
                           {
                               value = builder.create<ToMemrefOp>(
                                   loc, convertTensorToMemRef(asTensorType(op.getValue())), op.getValue());
                           }
                           else
                           {
                               value = op.getValue();
                           }

                           auto oneConst =
                               builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), 1));
                           if (isMemRef(value))
                           {
                               auto loadedLhs = builder.create<AffineLoadOp>(loc, value, loopIvs);
                               // Create the binary operation performed on the loaded values.
                               return builder.create<XOrIOp>(loc, loadedLhs, oneConst);
                           }
                           else
                           {
                               // Create the binary operation performed on the loaded values.
                               return builder.create<XOrIOp>(loc, op.getValue().getType(), op.getValue(), oneConst);
                           }
                       });
        return success();
    }
} // namespace voila::mlir::lowering