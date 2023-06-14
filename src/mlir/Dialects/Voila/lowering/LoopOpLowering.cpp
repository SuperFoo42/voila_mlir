#include "mlir/Dialects/Voila/lowering/LoopOpLowering.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using ::mlir::voila::LoopOp;
    using ::mlir::voila::LoopOpAdaptor;

    static void lowerOpToLoops(::mlir::voila::LoopOp op,
                               PatternRewriter &rewriter,
                               LoopOpLowering::LoopIterationFn processIteration)
    {
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        Value lowerBound = builder.create<ConstantIndexOp>(1);
        Value upperBound;

        // find first tensor operand and use its result type

        // start index for store
        SmallVector<Value> iter_args;
        Value cond;
        if (isTensor(op.getCond()))
        {
            cond = builder.create<ToMemrefOp>(convertTensorToMemRef(op.getCond().getType().dyn_cast<TensorType>()),
                                              op.getCond());
            upperBound = builder.create<AddIOp>(builder.create<memref::DimOp>(op.getCond(), 0),
                                                builder.create<ConstantIndexOp>(1));
        }
        else
        {
            cond = op.getCond();
            upperBound = builder.create<ConstantIndexOp>(2);
        }

        if (isMemRef(cond))
        {
            SmallVector<Value> idx;
            idx.push_back(builder.create<ConstantIndexOp>(0));
            iter_args.push_back(builder.create<memref::LoadOp>(cond, idx));
        }
        else
        {
            iter_args.push_back(cond);
        }

        rewriter.replaceOpWithNewOp<affine::AffineForOp>(
            op, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
            [&](OpBuilder &nestedBuilder, Location loc, Value iter_var /*index on which to store selected value*/,
                ValueRange ivs) -> void
            {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                processIteration(rewriter, iter_var, ivs.front());
                // load next cond bit and yield
                if (isMemRef(cond))
                {
                    SmallVector<Value> loadedCond;
                    loadedCond.push_back(rewriter.create<affine::AffineLoadOp>(
                        loc, cond, ivs)); // FIXME: oob load in last iteration - add affine if
                    rewriter.create<affine::AffineYieldOp>(loc, loadedCond);
                }
                else
                {
                    rewriter.create<affine::AffineYieldOp>(loc, cond);
                }
            });
    }

    LogicalResult LoopOpLowering::matchAndRewrite(::mlir::voila::LoopOp op,
                                                  OpAdaptor adaptor,
                                                  ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(op, rewriter,
                       [op, loc](PatternRewriter &builder, ValueRange loopIvs, Value iter_var)
                       {
                           auto ifOp = builder.create<scf::IfOp>(loc, iter_var, false);

                           builder.inlineRegionBefore(op->getRegion(0), &ifOp.getThenRegion().back());
                           builder.eraseBlock(&ifOp.getThenRegion().back());
                           OpBuilder thenBuilder(&ifOp.getThenRegion().back().back());
                           thenBuilder.setInsertionPointAfter(&ifOp.getThenRegion().back().back());
                           thenBuilder.create<scf::YieldOp>(loc);
                       });
        return success();
    }
} // namespace voila::mlir::lowering