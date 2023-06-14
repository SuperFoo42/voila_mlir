#include "mlir/Dialects/Voila/lowering/SelectOpLowering.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
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
    using namespace ::mlir::arith;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::SelectOpAdaptor;

    static Value
    insertAllocAndDealloc(MemRefType type, Value dynamicMemref, Location loc, ImplicitLocOpBuilder &builder)
    {
        // This has to be located before the loop and after participating ops
        memref::AllocOp alloc;
        if (type.hasStaticShape())
        {
            alloc = builder.create<memref::AllocOp>(type);
        }
        else if (asShapedType(dynamicMemref).isDynamicDim(0))
        {
            auto allocSize = builder.create<memref::DimOp>(dynamicMemref, 0);
            alloc = builder.create<memref::AllocOp>(MemRefType::get(ShapedType::kDynamic, type.getElementType()),
                                                    Value(allocSize));
        }
        else
        {
            auto allocSize = builder.create<ConstantIndexOp>(asShapedType(dynamicMemref).getDimSize(0));
            alloc = builder.create<memref::AllocOp>(MemRefType::get(ShapedType::kDynamic, type.getElementType()),
                                                    Value(allocSize));
        }

        // buffer deallocation instructions are added in the buffer deallocation pass
        return alloc;
    }

    void SelectOpLowering::lowerOpToLoops(::mlir::voila::SelectOp op,
                                          PatternRewriter &rewriter,
                                          LoopIterationFn processIteration)
    {
        auto tensorType = asTensorType(*op->result_type_begin());
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);
        ::mlir::Value tensorOp;
        assert(isTensor(op->getOperand(0)));
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
            upperBound = builder.create<ConstantIndexOp>(asShapedType(tensorOp).getDimSize(0));
        }

        // start index for store
        SmallVector<Value> iter_args;
        iter_args.push_back(builder.create<ConstantIndexOp>(0));

        auto resultSize = builder.create<affine::AffineForOp>(
            lowerBound, builder.getDimIdentityMap(), upperBound, builder.getDimIdentityMap(), 1, iter_args,
            [&](OpBuilder &nestedBuilder, ::mlir::Location loc,
                Value iter_var /*index on which to store selected value*/, ValueRange ivs) -> void
            {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                ImplicitLocOpBuilder b(loc, nestedBuilder);
                Value nextIdx = processIteration(b, op, iter_var, ivs.front(), alloc);
                nestedBuilder.create<affine::AffineYieldOp>(loc, nextIdx);
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
            MemRefType::get(-1, getElementTypeOrSelf(alloc)), alloc, zeroConst, sizes,
            indices);
        rewriter.replaceOpWithNewOp<ToTensorOp>(op, res);
    }

    LogicalResult SelectOpLowering::matchAndRewrite(::mlir::voila::SelectOp op,
                                                    OpAdaptor adaptor,
                                                    ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        lowerOpToLoops(op, rewriter,
                       [loc](ImplicitLocOpBuilder &builder, ::mlir::voila::SelectOp op, ValueRange loopIvs,
                             Value iter_var, Value dest) -> Value
                       {
                           auto values = op.getValues();
                           auto pred = op.getPred();

                           if (isTensor(values) && isTensor(pred))
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
                               elseBuilder.create<affine::AffineStoreOp>(valToStore, dest, iter_var);
                               auto oneConst = elseBuilder.create<ConstantIndexOp>(1);
                               SmallVector<Value> res;
                               auto addOp = elseBuilder.create<AddIOp>(iter_var, oneConst);
                               res.push_back(addOp);
                               elseBuilder.create<scf::YieldOp>(res);
                               elseBuilder.setInsertionPoint(oneConst);
                               return ifOp.getResult(0); // only new index to return
                           }
                           else if (isTensor(values))
                           {
                               // Create the binary operation performed on the loaded values.
                               auto ifOp = builder.create<scf::IfOp>(builder.getIndexType(), pred, true);
                               auto thenBuilder = ImplicitLocOpBuilder(loc, ifOp.getElseBodyBuilder());
                               auto thenBranch = thenBuilder.create<scf::YieldOp>(iter_var);
                               thenBuilder.setInsertionPoint(thenBranch);
                               auto elseBuilder = ImplicitLocOpBuilder(loc, ifOp.getThenBodyBuilder());
                               // value is constant
                               auto valToStore = elseBuilder.create<tensor::ExtractOp>(values, loopIvs);
                               elseBuilder.create<affine::AffineStoreOp>(valToStore, dest, iter_var);
                               auto oneConst = elseBuilder.create<ConstantIndexOp>(1);
                               SmallVector<Value> res;
                               auto addOp = elseBuilder.create<AddIOp>(iter_var, oneConst);
                               res.push_back(addOp);
                               elseBuilder.create<scf::YieldOp>(res);
                               elseBuilder.setInsertionPoint(oneConst);
                               return ifOp.getResult(0); // only new index to return
                           }
                           else if (isTensor(pred))
                           {
                               auto cond = builder.create<tensor::ExtractOp>(pred, loopIvs);
                               // Create the binary operation performed on the loaded values.
                               auto ifOp = builder.create<scf::IfOp>(builder.getIndexType(), cond, true);
                               auto ifBuilder = ImplicitLocOpBuilder(loc, ifOp.getThenBodyBuilder());
                               // value is constant
                               auto valToStore = values;
                               ifBuilder.create<affine::AffineStoreOp>(valToStore, dest, iter_var);
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
                               ifBuilder.create<affine::AffineStoreOp>(valToStore, dest, iter_var);
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
} // namespace voila::mlir::lowering