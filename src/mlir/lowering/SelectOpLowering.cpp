#include "mlir/lowering/SelectOpLowering.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::SelectOpAdaptor;

    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    static Value
    insertAllocAndDealloc(MemRefType type, Value dynamicMemref, Location loc, ImplicitLocOpBuilder &builder)
    {
        // This has to be located before the loop and after participating ops
        memref::AllocOp alloc;
        if (type.hasStaticShape())
        {
            alloc = builder.create<memref::AllocOp>(type);
        }
        else if (dynamicMemref.getType().dyn_cast<TensorType>().isDynamicDim(0))
        {
            auto allocSize = builder.create<memref::DimOp>(dynamicMemref, 0);
            alloc = builder.create<memref::AllocOp>(MemRefType::get(-1, type.getElementType()), Value(allocSize));
        }
        else
        {
            auto allocSize =
                builder.create<ConstantIndexOp>(dynamicMemref.getType().dyn_cast<TensorType>().getDimSize(0));
            alloc = builder.create<memref::AllocOp>(MemRefType::get(-1, type.getElementType()), Value(allocSize));
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
        ImplicitLocOpBuilder builder(loc, rewriter);
        ::mlir::Value tensorOp;
        assert(op->getOperand(0).getType().isa<TensorType>());
        tensorOp = op->getOperand(0);

        // Insert an allocation and deallocation for the result of this operation.
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, tensorOp, loc, builder);

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.

        Value lowerBound = builder.create<ConstantIndexOp>(0);
        Value upperBound;

        // find first tensor operand and use its result type
        // TODO: this does not look like a clean solution

        if (tensorOp.getType().dyn_cast<TensorType>().isDynamicDim(0))
        {
            upperBound = builder.create<memref::DimOp>(tensorOp, 0);
        }
        else
        {
            upperBound = builder.create<ConstantIndexOp>(tensorOp.getType().dyn_cast<TensorType>().getDimSize(0));
        }

        // start index for store
        SmallVector<Value> iter_args;
        iter_args.push_back(builder.create<ConstantIndexOp>(0));

        auto resultSize = builder.create<AffineForOp>(
            lowerBound, builder.getDimIdentityMap(), upperBound, builder.getDimIdentityMap(), 1, iter_args,
            [&](OpBuilder &nestedBuilder, ::mlir::Location loc,
                Value iter_var /*index on which to store selected value*/, ValueRange ivs) -> void
            {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                ImplicitLocOpBuilder b(loc, nestedBuilder);
                Value nextIdx = processIteration(b, operands, iter_var, ivs.front(), alloc);
                nestedBuilder.create<AffineYieldOp>(loc, nextIdx);
            });

        // Replace this operation with the generated reshaped alloc.
        auto resSizeMemRef = builder.create<memref::AllocaOp>(MemRefType::get(1, builder.getIndexType()));
        auto zeroConst = builder.create<ConstantIndexOp>(0);
        SmallVector<Value> indices;
        indices.push_back(zeroConst);
        SmallVector<Value> sizes;
        sizes.push_back(resultSize.getResult(0));
        builder.create<memref::StoreOp>(resultSize->getResult(0), resSizeMemRef, indices);
        auto res = builder.create<memref::ReinterpretCastOp>(
            MemRefType::get(-1, alloc.getType().dyn_cast<MemRefType>().getElementType()), alloc, zeroConst, sizes,
            indices);
        rewriter.replaceOpWithNewOp<ToTensorOp>(op, res);
    }

    LogicalResult SelectOpLowering::matchAndRewrite(Operation *op,
                                                    ArrayRef<Value> operands,
                                                    ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](ImplicitLocOpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs,
                             Value iter_var, Value dest) -> Value
                       {
                           SelectOpAdaptor binaryAdaptor(memRefOperands);
                           auto values = binaryAdaptor.values();
                           auto pred = binaryAdaptor.pred();

                           if (values.getType().isa<TensorType>() && pred.getType().isa<TensorType>())
                           {
                               auto cond = builder.create<tensor::ExtractOp>(pred, loopIvs);
                               // Create the binary operation performed on the loaded values.

                               auto ifOp = builder.create<scf::IfOp>(builder.getIndexType(), cond, true);
                               auto thenBuilder = ImplicitLocOpBuilder(loc, ifOp.getElseBodyBuilder());
                               auto thenBranch = thenBuilder.create<scf::YieldOp>(iter_var);
                               thenBuilder.setInsertionPoint(thenBranch);
                               auto elseBuilder = ImplicitLocOpBuilder(loc, ifOp.getThenBodyBuilder());
                               // value is constant
                               auto valToStore = elseBuilder.create<tensor::ExtractOp>(values, loopIvs);
                               elseBuilder.create<AffineStoreOp>(valToStore, dest, iter_var);
                               auto oneConst = elseBuilder.create<ConstantIndexOp>(1);
                               SmallVector<Value> res;
                               auto addOp = elseBuilder.create<AddIOp>(iter_var, oneConst);
                               res.push_back(addOp);
                               elseBuilder.create<scf::YieldOp>(res);
                               elseBuilder.setInsertionPoint(oneConst);
                               return ifOp.getResult(0); // only new index to return
                           }
                           else if (values.getType().isa<TensorType>())
                           {
                               // Create the binary operation performed on the loaded values.
                               auto ifOp = builder.create<scf::IfOp>(builder.getIndexType(), pred, true);
                               auto thenBuilder = ImplicitLocOpBuilder(loc, ifOp.getElseBodyBuilder());
                               auto thenBranch = thenBuilder.create<scf::YieldOp>(iter_var);
                               thenBuilder.setInsertionPoint(thenBranch);
                               auto elseBuilder = ImplicitLocOpBuilder(loc, ifOp.getThenBodyBuilder());
                               // value is constant
                               auto valToStore = elseBuilder.create<tensor::ExtractOp>(values, loopIvs);
                               elseBuilder.create<AffineStoreOp>(valToStore, dest, iter_var);
                               auto oneConst = elseBuilder.create<ConstantIndexOp>(1);
                               SmallVector<Value> res;
                               auto addOp = elseBuilder.create<AddIOp>(iter_var, oneConst);
                               res.push_back(addOp);
                               elseBuilder.create<scf::YieldOp>(res);
                               elseBuilder.setInsertionPoint(oneConst);
                               return ifOp.getResult(0); // only new index to return
                           }
                           else if (pred.getType().isa<TensorType>())
                           {
                               auto cond = builder.create<tensor::ExtractOp>(pred, loopIvs);
                               // Create the binary operation performed on the loaded values.
                               auto ifOp = builder.create<scf::IfOp>(builder.getIndexType(), cond, true);
                               auto ifBuilder = ImplicitLocOpBuilder(loc, ifOp.getThenBodyBuilder());
                               // value is constant
                               auto valToStore = values;
                               ifBuilder.create<AffineStoreOp>(valToStore, dest, iter_var);
                               auto oneConst = ifBuilder.create<ConstantIndexOp>(1);
                               SmallVector<Value> res;
                               auto addOp = ifBuilder.create<AddIOp>(iter_var, oneConst);
                               res.push_back(addOp);
                               ifBuilder.create<scf::YieldOp>(res);
                               ifBuilder.setInsertionPoint(oneConst);
                               auto elseBuilder = ImplicitLocOpBuilder(loc, ifOp.getElseBodyBuilder());
                               auto thenBranch = elseBuilder.create<scf::YieldOp>(iter_var);
                               elseBuilder.setInsertionPoint(thenBranch);
                               return ifOp.getResult(0); // only new index to return
                           }
                           else
                           {
                               // Create the binary operation performed on the loaded values.
                               auto ifOp = builder.create<scf::IfOp>(builder.getIndexType(), pred, true);
                               auto ifBuilder = ImplicitLocOpBuilder(loc, ifOp.getThenBodyBuilder());
                               // value is constant
                               auto valToStore = values;
                               ifBuilder.create<AffineStoreOp>(valToStore, dest, iter_var);
                               auto oneConst = ifBuilder.create<ConstantIndexOp>(1);
                               SmallVector<Value> res;
                               auto addOp = ifBuilder.create<AddIOp>(iter_var, oneConst);
                               res.push_back(addOp);
                               ifBuilder.create<scf::YieldOp>(res);
                               ifBuilder.setInsertionPoint(oneConst);
                               auto elseBuilder = ImplicitLocOpBuilder(loc, ifOp.getElseBodyBuilder());
                               auto thenBranch = elseBuilder.create<scf::YieldOp>(iter_var);
                               elseBuilder.setInsertionPoint(thenBranch);
                               return ifOp.getResult(0); // only new index to return
                           }
                       });
        return success();
    }
    SelectOpLowering::SelectOpLowering(::mlir::MLIRContext *ctx) :
        ConversionPattern(::mlir::voila::SelectOp::getOperationName(), 1, ctx)
    {
    }
} // namespace voila::mlir::lowering