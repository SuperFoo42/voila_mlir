#include "mlir/lowering/AvgOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::AvgOp;
    using ::mlir::voila::AvgOpAdaptor;

    AvgOpLowering::AvgOpLowering(MLIRContext *ctx) : ConversionPattern(AvgOp::getOperationName(), 1, ctx) {}

    LogicalResult
    AvgOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        AvgOpAdaptor adaptor(operands);

        auto sum =
            rewriter.create<::mlir::voila::SumOp>(loc, getElementTypeOrSelf(adaptor.input()), adaptor.input(), nullptr);
        auto count = rewriter.create<::mlir::voila::CountOp>(loc, rewriter.getI64Type(), adaptor.input());

        rewriter.replaceOpWithNewOp<::mlir::voila::DivOp>(op, rewriter.getF64Type(),
                                                          rewriter.create<SIToFPOp>(loc, sum, rewriter.getF64Type()),
                                                          rewriter.create<SIToFPOp>(loc, count, rewriter.getF64Type()));

        return success();
    }
} // namespace voila::mlir::lowering