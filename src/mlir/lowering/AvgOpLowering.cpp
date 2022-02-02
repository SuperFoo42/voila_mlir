#include "mlir/lowering/AvgOpLowering.hpp"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::voila;

    AvgOpLowering::AvgOpLowering(MLIRContext *ctx) : ConversionPattern(AvgOp::getOperationName(), 1, ctx) {}

    static Value castITensorToFTensor(ImplicitLocOpBuilder &builder, Value tensor)
    {
        Value res;
        if (tensor.getType().dyn_cast<TensorType>().hasStaticShape())
        {
            res = builder.create<linalg::InitTensorOp>(tensor.getType().dyn_cast<TensorType>().getShape(),
                                                       builder.getF64Type());
        }
        else
        {
            Value dimSize = builder.create<::mlir::tensor::DimOp>(tensor, 0);
            res = builder.create<linalg::InitTensorOp>(::llvm::makeArrayRef(dimSize), builder.getF64Type());
        }

        SmallVector<Type, 1> res_type(1, res.getType());

        SmallVector<StringRef, 1> iter_type(1, getParallelIteratorTypeName());

        auto fn = [](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            Value fVal = builder.create<SIToFPOp>(vals[0], builder.getF64Type());
            builder.create<linalg::YieldOp>(fVal);
        };

        SmallVector<AffineMap, 2> maps(2, builder.getDimIdentityMap());

        auto linalgOp = builder.create<linalg::GenericOp>(/*results*/ res_type,
                                                           /*inputs*/ tensor, /*outputs*/ res,
                                                           /*indexing maps*/ maps,
                                                           /*iterator types*/ iter_type, fn);
        return linalgOp.getResult(0);
    }

    LogicalResult
    AvgOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        assert(!op->getResultTypes().empty() && op->getResultTypes().size() == 1);
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);
        AvgOpAdaptor adaptor(operands);

        // this should work as long as ieee 754 is supported and division by 0 is inf
        if (adaptor.indices() && op->getResultTypes().front().isa<TensorType>())
        {
            auto sum = builder.create<SumOp>(
                RankedTensorType::get(-1, getElementTypeOrSelf(adaptor.input()).isa<FloatType>() ?
                                              static_cast<Type>(builder.getF64Type()) :
                                              static_cast<Type>(builder.getI64Type())),
                adaptor.input(), adaptor.indices(), adaptor.pred());
            auto count = builder.create<CountOp>(RankedTensorType::get(-1, builder.getI64Type()),
                                                                 adaptor.input(), adaptor.indices(), adaptor.pred());

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
            //TODO: replace with predicates
            rewriter.replaceOpWithNewOp<DivOp>(op, RankedTensorType::get(-1, builder.getF64Type()),
                                                              fltSum, fltCnt, Value());
        }
        else
        {
            auto sum = builder.create<SumOp>(getElementTypeOrSelf(adaptor.input()),
                                                             adaptor.input(), adaptor.indices(), adaptor.pred());
            auto count =
                builder.create<CountOp>(builder.getI64Type(), adaptor.input(), adaptor.indices(), adaptor.pred());

            Value fltCnt, fltSum;
            if (!getElementTypeOrSelf(count).isa<FloatType>())
            {
                fltCnt = builder.create<SIToFPOp>(count, builder.getF64Type());
            }
            else
            {
                fltCnt = count;
            }

            if (!getElementTypeOrSelf(sum).isa<FloatType>())
            {
                fltSum = builder.create<SIToFPOp>(sum, builder.getF64Type());
            }
            else
            {
                fltSum = sum;
            }

            //TODO: replace with predicates
            rewriter.replaceOpWithNewOp<DivOp>(op, rewriter.getF64Type(), fltSum, fltCnt, Value());
        }
        return success();
    }
} // namespace voila::mlir::lowering