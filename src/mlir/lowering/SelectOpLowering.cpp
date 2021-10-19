#include "mlir/lowering/SelectOpLowering.hpp"

#include "mlir/IR/IntegerSet.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using ::mlir::voila::SelectOpAdaptor;

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

    void SelectOpLowering::lowerOpToLoops(Operation *op,
                                          ValueRange operands,
                                          PatternRewriter &rewriter,
                                          LoopIterationFn processIteration)
    {
        auto tensorType = (*op->result_type_begin()).dyn_cast<TensorType>();
        auto loc = op->getLoc();
        ::mlir::Value tensorOp;
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
        // TODO: this does not look like a clean solution

        if (tensorOp.getType().dyn_cast<TensorType>().isDynamicDim(0))
        {
            upperBound = rewriter.create<memref::DimOp>(loc, tensorOp, 0);
        }
        else
        {
            upperBound = rewriter.create<ConstantIndexOp>(loc, tensorOp.getType().dyn_cast<TensorType>().getDimSize(0));
        }

        // start index for store
        SmallVector<Value> iter_args;
        iter_args.push_back(rewriter.create<ConstantIndexOp>(loc, 0));

        auto resultSize = rewriter.create<AffineForOp>(
            loc, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
            [&](OpBuilder &nestedBuilder, ::mlir::Location loc,
                Value iter_var /*index on which to store selected value*/, ValueRange ivs) -> void
            {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                Value nextIdx = processIteration(nestedBuilder, operands, iter_var, ivs.front(), alloc);
                nestedBuilder.create<AffineYieldOp>(loc, nextIdx);
            });

        // Replace this operation with the generated reshaped alloc.
        auto resSizeMemRef = rewriter.create<memref::AllocaOp>(loc, MemRefType::get(1, rewriter.getIndexType()));
        auto zeroConst = rewriter.create<ConstantIndexOp>(loc, 0);
        SmallVector<Value> indices;
        indices.push_back(zeroConst);
        SmallVector<Value> sizes;
        sizes.push_back(resultSize.getResult(0));
        rewriter.create<memref::StoreOp>(loc, resultSize->getResult(0), resSizeMemRef, indices);
        auto res = rewriter.create<memref::ReinterpretCastOp>(
            loc, MemRefType::get(-1, alloc.getType().dyn_cast<MemRefType>().getElementType()), alloc, zeroConst, sizes,
            indices);
        rewriter.replaceOpWithNewOp<memref::TensorLoadOp>(
            op, res);
    }

    LogicalResult SelectOpLowering::matchAndRewrite(Operation *op,
                                                    ArrayRef<Value> operands,
                                                    ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs, Value iter_var,
                             Value dest) -> Value
                       {
                           SelectOpAdaptor binaryAdaptor(memRefOperands);
                           Value values;
                           Value pred;
                           if (binaryAdaptor.values().getType().isa<TensorType>())
                           {
                               values = builder.create<memref::BufferCastOp>(
                                   loc, convertTensorToMemRef(binaryAdaptor.values().getType().dyn_cast<TensorType>()),
                                   binaryAdaptor.values());
                           }
                           else
                           {
                               values = binaryAdaptor.values();
                           }

                           if (binaryAdaptor.pred().getType().isa<TensorType>())
                           {
                               pred = builder.create<memref::BufferCastOp>(
                                   loc, convertTensorToMemRef(binaryAdaptor.pred().getType().dyn_cast<TensorType>()),
                                   binaryAdaptor.pred());
                           }
                           else
                           {
                               pred = binaryAdaptor.pred();
                           }

                           if (values.getType().isa<MemRefType>() && pred.getType().isa<MemRefType>())
                           {
                               auto loadedRhs = builder.create<AffineLoadOp>(loc, pred, loopIvs);
                               // Create the binary operation performed on the loaded values.

                               auto ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), loadedRhs, true);
                               auto thenBuilder = ifOp.getElseBodyBuilder();
                               auto thenBranch = thenBuilder.create<scf::YieldOp>(loc, iter_var);
                               thenBuilder.setInsertionPoint(thenBranch);
                               auto elseBuilder = ifOp.getThenBodyBuilder();
                               // value is constant
                               auto valToStore = elseBuilder.create<AffineLoadOp>(loc, values, loopIvs);
                               elseBuilder.create<AffineStoreOp>(loc, valToStore, dest, iter_var);
                               auto oneConst = elseBuilder.create<ConstantIndexOp>(loc, 1);
                               SmallVector<Value> res;
                               auto addOp = elseBuilder.create<AddIOp>(loc, iter_var, oneConst);
                               res.push_back(addOp);
                               elseBuilder.create<scf::YieldOp>(loc, res);
                               elseBuilder.setInsertionPoint(oneConst);
                               return ifOp.getResult(0); // only new index to return
                           }
                           else if (values.getType().isa<MemRefType>())
                           {
                               // Create the binary operation performed on the loaded values.
                               auto ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), pred, true);
                               auto thenBuilder = ifOp.getElseBodyBuilder();
                               auto thenBranch = thenBuilder.create<scf::YieldOp>(loc, iter_var);
                               thenBuilder.setInsertionPoint(thenBranch);
                               auto elseBuilder = ifOp.getThenBodyBuilder();
                               // value is constant
                               auto valToStore = elseBuilder.create<AffineLoadOp>(loc, values, loopIvs);
                               elseBuilder.create<AffineStoreOp>(loc, valToStore, dest, iter_var);
                               auto oneConst = elseBuilder.create<ConstantIndexOp>(loc, 1);
                               SmallVector<Value> res;
                               auto addOp = elseBuilder.create<AddIOp>(loc, iter_var, oneConst);
                               res.push_back(addOp);
                               elseBuilder.create<scf::YieldOp>(loc, res);
                               elseBuilder.setInsertionPoint(oneConst);
                               return ifOp.getResult(0); // only new index to return
                           }
                           else if (pred.getType().isa<MemRefType>())
                           {
                               auto loadedRhs = builder.create<AffineLoadOp>(loc, pred, loopIvs);
                               // Create the binary operation performed on the loaded values.
                               auto ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), loadedRhs, true);
                               auto ifBuilder = ifOp.getThenBodyBuilder();
                               // value is constant
                               auto valToStore = values;
                               ifBuilder.create<AffineStoreOp>(loc, valToStore, dest, iter_var);
                               auto oneConst = ifBuilder.create<ConstantIndexOp>(loc, 1);
                               SmallVector<Value> res;
                               auto addOp = ifBuilder.create<AddIOp>(loc, iter_var, oneConst);
                               res.push_back(addOp);
                               ifBuilder.create<scf::YieldOp>(loc, res);
                               ifBuilder.setInsertionPoint(oneConst);
                               auto elseBuilder = ifOp.getElseBodyBuilder();
                               auto thenBranch = elseBuilder.create<scf::YieldOp>(loc, iter_var);
                               elseBuilder.setInsertionPoint(thenBranch);
                               return ifOp.getResult(0); // only new index to return
                           }
                           else
                           {
                               // Create the binary operation performed on the loaded values.
                               auto ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), pred, true);
                               auto ifBuilder = ifOp.getThenBodyBuilder();
                               // value is constant
                               auto valToStore = values;
                               ifBuilder.create<AffineStoreOp>(loc, valToStore, dest, iter_var);
                               auto oneConst = ifBuilder.create<ConstantIndexOp>(loc, 1);
                               SmallVector<Value> res;
                               auto addOp = ifBuilder.create<AddIOp>(loc, iter_var, oneConst);
                               res.push_back(addOp);
                               ifBuilder.create<scf::YieldOp>(loc, res);
                               ifBuilder.setInsertionPoint(oneConst);
                               auto elseBuilder = ifOp.getElseBodyBuilder();
                               auto thenBranch = elseBuilder.create<scf::YieldOp>(loc, iter_var);
                               elseBuilder.setInsertionPoint(thenBranch);
                               return ifOp.getResult(0); // only new index to return
                           }
                       });
        return success();
    }
} // namespace voila::mlir::lowering
