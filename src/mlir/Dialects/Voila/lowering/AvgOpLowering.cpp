#include "mlir/Dialects/Voila/lowering/AvgOpLowering.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/Interfaces/PredicationOpInterface.hpp"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include <cassert>

namespace mlir
{
    class MLIRContext;
    class OpBuilder;
} // namespace mlir

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace ::mlir::voila;

    AvgOpLowering::AvgOpLowering(MLIRContext *ctx) : ConversionPattern(AvgOp::getOperationName(), 1, ctx) {}

    static Value castITensorToFTensor(ImplicitLocOpBuilder &builder, Value tensor)
    {
        Value res;
        if (tensor.getType().dyn_cast<TensorType>().hasStaticShape())
        {
            res = builder.create<tensor::EmptyOp>(tensor.getType().dyn_cast<TensorType>().getShape(),
                                                  builder.getF64Type());
        }
        else
        {
            Value dimSize = builder.create<::mlir::tensor::DimOp>(tensor, 0);
            res = builder.create<tensor::EmptyOp>(ShapedType::kDynamic, builder.getF64Type(), dimSize);
        }

        SmallVector<Type, 1> res_type(1, res.getType());

        auto fn = [](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            Value fVal = builder.create<SIToFPOp>(builder.getF64Type(), vals[0]);
            builder.create<linalg::YieldOp>(fVal);
        };

        SmallVector<AffineMap, 2> maps(2, builder.getDimIdentityMap());

        auto linalgOp = builder.create<linalg::GenericOp>(/*results*/ res_type,
                                                          /*inputs*/ tensor, /*outputs*/ res,
                                                          /*indexing maps*/ maps,
                                                          /*iterator types*/ utils::IteratorType::parallel, fn);
        return linalgOp.getResult(0);
    }

    LogicalResult
    AvgOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        assert(!op->getResultTypes().empty() && op->getResultTypes().size() == 1);
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);
        auto avgOp = llvm::dyn_cast<AvgOp>(op);
        AvgOpAdaptor adaptor(avgOp);

        // this should work as long as ieee 754 is supported and division by 0 is inf
        if (adaptor.getIndices() && op->getResultTypes().front().isa<TensorType>())
        {
            auto sum = builder.create<SumOp>(RankedTensorType::get(ShapedType::kDynamic, builder.getF64Type()), adaptor.getInput(),
                                             adaptor.getIndices(), adaptor.getPred());
            auto count = builder.create<CountOp>(RankedTensorType::get(ShapedType::kDynamic, builder.getI64Type()), adaptor.getInput(),
                                                 adaptor.getIndices(), adaptor.getPred());

            Value fltCnt, fltSum;
            if (!getElementTypeOrSelf(count).isa<FloatType>())
            {
                fltCnt = castITensorToFTensor(builder, count);
            }
            else
            {
                fltCnt = count;
            }

            if (!getElementTypeOrSelf(sum).isa<FloatType>())
            {
                fltSum = castITensorToFTensor(builder, sum);
            }
            else
            {
                fltSum = sum;
            }

            rewriter.replaceOpWithNewOp<DivOp>(op, RankedTensorType::get(ShapedType::kDynamic, builder.getF64Type()), fltSum, fltCnt);
        }
        else
        {
            auto sum = builder.create<SumOp>(getElementTypeOrSelf(adaptor.getInput()), adaptor.getInput(),
                                             adaptor.getIndices(), adaptor.getPred());
            auto count = builder.create<CountOp>(builder.getI64Type(), adaptor.getInput(), adaptor.getIndices(),
                                                 adaptor.getPred());

            Value fltCnt, fltSum;
            if (!getElementTypeOrSelf(count).isa<FloatType>())
            {
                fltCnt = builder.create<SIToFPOp>(builder.getF64Type(), count);
            }
            else
            {
                fltCnt = count;
            }

            if (!getElementTypeOrSelf(sum).isa<FloatType>())
            {
                fltSum = builder.create<SIToFPOp>(builder.getF64Type(), sum);
            }
            else
            {
                fltSum = sum;
            }

            rewriter.replaceOpWithNewOp<DivOp>(op, rewriter.getF64Type(), fltSum, fltCnt);
        }
        return success();
    }
} // namespace voila::mlir::lowering