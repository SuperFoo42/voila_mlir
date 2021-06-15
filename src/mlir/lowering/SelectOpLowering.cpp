#include "mlir/lowering/SelectOpLowering.hpp"

#include "mlir/IR/IntegerSet.h"
using namespace mlir;
using namespace ::voila::mlir::lowering;
static MemRefType convertTensorToMemRef(TensorType type)
{
    assert(type.hasRank() && "expected only ranked shapes");
    return MemRefType::get(type.getShape(), type.getElementType());
}

static ::mlir::Value insertAllocAndDealloc(::mlir::MemRefType type,
                                           ::mlir::Value dynamicMemref,
                                           ::mlir::Location loc,
                                           ::mlir::PatternRewriter &rewriter)
{
    // This has to be located before the loop and after participating ops
    ::mlir::memref::AllocOp alloc;
    if (dynamicMemref.getType().dyn_cast<::mlir::TensorType>().isDynamicDim(0))
    {
        auto allocSize = rewriter.create<::mlir::memref::DimOp>(loc, dynamicMemref, 0);
        alloc = rewriter.create<::mlir::memref::AllocOp>(loc, type, ::mlir::Value(allocSize));
    }
    else
    {
        alloc = rewriter.create<::mlir::memref::AllocOp>(loc, type);
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

    Value lowerBound = rewriter.create<::mlir::ConstantIndexOp>(loc, 0);
    ::mlir::Value upperBound;

    // find first tensor operand and use its result type
    // TODO: this does not look like a clean solution

    if (tensorOp.getType().dyn_cast<::mlir::TensorType>().isDynamicDim(0))
    {
        upperBound = rewriter.create<::mlir::memref::DimOp>(loc, tensorOp, 0);
    }
    else
    {
        upperBound = rewriter.create<::mlir::ConstantIndexOp>(
            loc, tensorOp.getType().dyn_cast<::mlir::TensorType>().getDimSize(0));
    }

    // start index for store
    SmallVector<Value> iter_args;
    iter_args.push_back(rewriter.create<ConstantIndexOp>(loc, 0));

    auto resultSize = rewriter.create<AffineForOp>(
        loc, lowerBound, rewriter.getDimIdentityMap(), upperBound, rewriter.getDimIdentityMap(), 1, iter_args,
        [&](OpBuilder &nestedBuilder, ::mlir::Location loc, Value iter_var /*index on which to store selected value*/,
            ValueRange ivs) -> void
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
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(op, MemRefType::get(-1, alloc.getType().dyn_cast<MemRefType>().getElementType()),alloc, zeroConst, sizes, indices);
}

LogicalResult SelectOpLowering::matchAndRewrite(Operation *op,
                                                llvm::ArrayRef<Value> operands,
                                                ConversionPatternRewriter &rewriter) const
{
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs, Value iter_var, Value dest) -> Value
        {
            ::mlir::voila::SelectOp::Adaptor binaryAdaptor(memRefOperands);

            if (binaryAdaptor.values().getType().isa<::mlir::MemRefType>() &&
                binaryAdaptor.pred().getType().isa<::mlir::MemRefType>())
            {
                auto loadedRhs = builder.create<::mlir::AffineLoadOp>(loc, binaryAdaptor.pred(), loopIvs);
                // Create the binary operation performed on the loaded values.
                SmallVector<Value> vals;
                vals.push_back(builder.create<IndexCastOp>(loc, loadedRhs, builder.getIndexType()));
                auto ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), loadedRhs, true);
                auto thenBuilder = ifOp.getThenBodyBuilder();
                auto thenBranch = thenBuilder.create<scf::YieldOp>(loc, iter_var);
                thenBuilder.setInsertionPoint(thenBranch);
                auto elseBuilder = ifOp.getElseBodyBuilder();
                // value is constant
                auto valToStore = elseBuilder.create<AffineLoadOp>(loc, binaryAdaptor.values(), loopIvs);
                elseBuilder.create<AffineStoreOp>(loc, valToStore, dest, iter_var);
                auto oneConst = elseBuilder.create<ConstantIndexOp>(loc, 1);
                SmallVector<Value> res;
                auto addOp = elseBuilder.create<::mlir::AddIOp>(loc, iter_var, oneConst);
                res.push_back(addOp);
                elseBuilder.create<scf::YieldOp>(loc, res);
                elseBuilder.setInsertionPoint(oneConst);
                return ifOp.getResult(0); // only new index to return
            }
            else if (binaryAdaptor.values().getType().isa<::mlir::MemRefType>())
            {
                // Create the binary operation performed on the loaded values.
                auto ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), binaryAdaptor.pred(), true);
                auto thenBuilder = ifOp.getElseBodyBuilder();
                auto thenBranch = thenBuilder.create<scf::YieldOp>(loc, iter_var);
                thenBuilder.setInsertionPoint(thenBranch);
                auto elseBuilder = ifOp.getThenBodyBuilder();
                // value is constant
                auto valToStore = elseBuilder.create<AffineLoadOp>(loc, binaryAdaptor.values(), loopIvs);
                elseBuilder.create<AffineStoreOp>(loc, valToStore, dest, iter_var);
                auto oneConst = elseBuilder.create<ConstantIndexOp>(loc, 1);
                SmallVector<Value> res;
                auto addOp = elseBuilder.create<::mlir::AddIOp>(loc, iter_var, oneConst);
                res.push_back(addOp);
                elseBuilder.create<scf::YieldOp>(loc, res);
                elseBuilder.setInsertionPoint(oneConst);
                return ifOp.getResult(0); // only new index to return
            }
            else if (binaryAdaptor.pred().getType().isa<::mlir::MemRefType>())
            {
                auto loadedRhs = builder.create<::mlir::AffineLoadOp>(loc, binaryAdaptor.pred(), loopIvs);
                // Create the binary operation performed on the loaded values.
                SmallVector<Value> vals;
                vals.push_back(builder.create<IndexCastOp>(loc, loadedRhs, builder.getIndexType()));
                auto ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), loadedRhs, true);
                auto ifBuilder = ifOp.getThenBodyBuilder();
                // value is constant
                auto valToStore = binaryAdaptor.values();
                ifBuilder.create<AffineStoreOp>(loc, valToStore, dest, iter_var);
                auto oneConst = ifBuilder.create<ConstantIndexOp>(loc, 1);
                SmallVector<Value> res;
                auto addOp = ifBuilder.create<::mlir::AddIOp>(loc, iter_var, oneConst);
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
                SmallVector<Value> vals;
                vals.push_back(builder.create<IndexCastOp>(loc, binaryAdaptor.pred(), builder.getIndexType()));
                auto ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), binaryAdaptor.pred(), true);
                auto ifBuilder = ifOp.getThenBodyBuilder();
                // value is constant
                auto valToStore = binaryAdaptor.values();
                ifBuilder.create<AffineStoreOp>(loc, valToStore, dest, iter_var);
                auto oneConst = ifBuilder.create<ConstantIndexOp>(loc, 1);
                SmallVector<Value> res;
                auto addOp = ifBuilder.create<::mlir::AddIOp>(loc, iter_var, oneConst);
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
