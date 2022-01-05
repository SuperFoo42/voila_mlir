#include "mlir/lowering/AvgOpLowering.hpp"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using ::mlir::voila::AvgOp;
    using ::mlir::voila::AvgOpAdaptor;

    AvgOpLowering::AvgOpLowering(MLIRContext *ctx) : ConversionPattern(AvgOp::getOperationName(), 1, ctx) {}

    static Value castITensorToFTensor(Location loc, ConversionPatternRewriter &rewriter, Value tensor)
    {
        Value res;
        if (tensor.getType().dyn_cast<TensorType>().hasStaticShape())
        {
            res = rewriter.create<linalg::InitTensorOp>(loc, tensor.getType().dyn_cast<TensorType>().getShape(),
                                                        rewriter.getF64Type());
        }
        else
        {
            Value dimSize = rewriter.create<::mlir::tensor::DimOp>(loc, tensor, 0);
            res = rewriter.create<linalg::InitTensorOp>(loc, ::llvm::makeArrayRef(dimSize), rewriter.getF64Type());
        }

        SmallVector<Type, 1> res_type(1, res.getType());

        SmallVector<StringRef, 1> iter_type(1, getParallelIteratorTypeName());

        auto fn = [](OpBuilder &builder, Location loc, ValueRange vals)
        {
            Value fVal = builder.create<SIToFPOp>(loc, vals[0], builder.getF64Type());
            builder.create<linalg::YieldOp>(loc, fVal);
        };

        SmallVector<AffineMap, 2> maps(2, rewriter.getDimIdentityMap());

        auto linalgOp = rewriter.create<linalg::GenericOp>(loc, /*results*/ res_type,
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
        AvgOpAdaptor adaptor(operands);

        // this should work as long as ieee 754 is supported and division by 0 is inf
        if (adaptor.indices() && op->getResultTypes().front().isa<TensorType>())
        {
            auto sum = rewriter.create<::mlir::voila::SumOp>(
                loc,
                RankedTensorType::get(-1, getElementTypeOrSelf(adaptor.input()).isa<FloatType>() ?
                                              static_cast<Type>(rewriter.getF64Type()) :
                                              static_cast<Type>(rewriter.getI64Type())),
                adaptor.input(), adaptor.indices());
            auto count = rewriter.create<::mlir::voila::CountOp>(loc, RankedTensorType::get(-1, rewriter.getI64Type()),
                                                                 adaptor.input(), adaptor.indices());

            Value fltCnt, fltSum;
            if (!getElementTypeOrSelf(count).isa<FloatType>())
            {
                fltCnt = castITensorToFTensor(loc, rewriter, count);
            }
            else
            {
                fltCnt = count;
            }

            if (!getElementTypeOrSelf(sum).isa<FloatType>())
            {
                fltSum = castITensorToFTensor(loc, rewriter, sum);
            }
            else
            {
                fltSum = sum;
            }

            rewriter.replaceOpWithNewOp<::mlir::voila::DivOp>(op, RankedTensorType::get(-1, rewriter.getF64Type()),
                                                              fltSum, fltCnt);
        }
        else
        {
            auto sum = rewriter.create<::mlir::voila::SumOp>(loc, getElementTypeOrSelf(adaptor.input()),
                                                             adaptor.input(), adaptor.indices());
            auto count =
                rewriter.create<::mlir::voila::CountOp>(loc, rewriter.getI64Type(), adaptor.input(), adaptor.indices());

            Value fltCnt, fltSum;
            if (!getElementTypeOrSelf(count).isa<FloatType>())
            {
                fltCnt = rewriter.create<SIToFPOp>(loc, count, rewriter.getF64Type());
            }
            else
            {
                fltCnt = count;
            }

            if (!getElementTypeOrSelf(sum).isa<FloatType>())
            {
                fltSum = rewriter.create<SIToFPOp>(loc, sum, rewriter.getF64Type());
            }
            else
            {
                fltSum = sum;
            }

            rewriter.replaceOpWithNewOp<::mlir::voila::DivOp>(op, rewriter.getF64Type(), fltSum, fltCnt);
        }
        return success();
    }
} // namespace voila::mlir::lowering